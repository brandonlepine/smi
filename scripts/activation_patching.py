


"""
activation_patching.py

Stage 1: Layer-level activation patching (original ↔ adversarial)

For each item pair, patches ALL token activations at each layer from the 
adversarial forward pass into the original forward pass, measuring how much 
the model's output distribution shifts. This identifies which layers are 
most causally implicated in processing domain-specific knowledge.

Follows the methodology of Marks & Tegmark (2024), "The Geometry of Truth."

Usage:
    python activation_patching.py --domain medical
    python activation_patching.py --domain legal
    python activation_patching.py --domain swe
    python activation_patching.py --domain all
    python activation_patching.py --aggregate   # combine results and plot
"""

import os
import json
import argparse
import time
import difflib
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_ID = "/Users/brandonlepine/Repositories/Research_Repositories/smi/models/llama2-7b"
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_DIR = PROJECT_ROOT / "benchmark_datasets"
CONDITIONS_DIR = DATA_DIR / "conditions"
RESULTS_DIR = PROJECT_ROOT / "results" / "patching"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load LLaMA 2 7B and tokenizer."""
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer, n_layers


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_item_pairs(domain: str):
    """Load original items and their adversarial conditions, return paired list."""

    # Load originals
    if domain == "medical":
        originals = pd.read_parquet(DATA_DIR / "medical" / "medqa_sample.parquet")
    elif domain == "legal":
        originals = pd.read_parquet(DATA_DIR / "legal" / "legalbench_sample.parquet")
    elif domain == "swe":
        originals = pd.read_parquet(DATA_DIR / "swe" / "swe_sample.parquet")
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Load adversarial conditions
    conditions_path = CONDITIONS_DIR / f"{domain}_conditions.jsonl"
    adversarial_map = {}
    with open(conditions_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["condition"] == "adversarial" and rec["success"]:
                adversarial_map[rec["item_idx"]] = rec["result"]

    # Build pairs
    pairs = []
    for i, (_, row) in enumerate(originals.iterrows()):
        if i not in adversarial_map:
            continue

        adv = adversarial_map[i]

        # Extract text based on domain
        orig_text = row.get("item_text", row.get("question", ""))

        if domain == "medical":
            adv_text = adv.get("adversarial_text", "")
        elif domain == "legal":
            adv_text = adv.get("adversarial_text", "")
        elif domain == "swe":
            adv_text = adv.get("adversarial_text", "")

        # Guard against NaN / non-string values (tokenizer requires str)
        if not isinstance(orig_text, str) or not orig_text.strip():
            continue
        if not isinstance(adv_text, str) or not adv_text.strip():
            continue

        pairs.append({
            "item_idx": i,
            "domain": domain,
            "stratum": row.get("stratum", ""),
            "original_text": orig_text.strip(),
            "adversarial_text": adv_text.strip(),
        })

    print(f"Loaded {len(pairs)} item pairs for {domain}")
    return pairs


# ---------------------------------------------------------------------------
# Activation caching
# ---------------------------------------------------------------------------

def get_residual_stream_activations(model, input_ids, n_layers):
    """
    Run a forward pass and cache residual stream activations at every layer.
    Returns dict: layer_idx -> tensor of shape (seq_len, hidden_dim)
    """
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is hidden states
            hidden = output[0].detach().clone()
            # Ensure 3D: (batch, seq_len, hidden_dim)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            activations[layer_idx] = hidden
        return hook_fn

    # Register hooks on each transformer layer
    for i in range(n_layers):
        hook = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(hook)

    with torch.no_grad():
        outputs = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations, outputs.logits


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def compute_token_diff(orig_ids, adv_ids):
    """
    Align two token sequences and return a mapping of positions to patch.
    
    Returns:
        patch_map: dict mapping orig_position -> adv_position for changed tokens
        n_changed: number of changed positions
        n_total: total original positions
    """
    orig_tokens = orig_ids[0].cpu().tolist()
    adv_tokens = adv_ids[0].cpu().tolist()

    matcher = difflib.SequenceMatcher(None, orig_tokens, adv_tokens)
    opcodes = matcher.get_opcodes()

    patch_map = {}  # orig_pos -> adv_pos

    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            # Tokens match — don't patch these
            continue
        elif op == "replace":
            # Tokens differ — patch orig positions with corresponding adv positions
            orig_len = i2 - i1
            adv_len = j2 - j1
            # Map as many as we can 1:1, extras get mapped to last available
            for k in range(orig_len):
                adv_k = min(k, adv_len - 1)
                patch_map[i1 + k] = j1 + adv_k
        elif op == "delete":
            # Tokens in original but not in adversarial — these were removed
            # Patch with nearest adversarial context (use j1 which is the 
            # adversarial position right after the deletion point)
            nearest_adv = min(j1, len(adv_tokens) - 1)
            for k in range(i1, i2):
                patch_map[k] = nearest_adv
        elif op == "insert":
            # Tokens in adversarial but not in original — can't patch into 
            # original since there's no corresponding position. Skip.
            continue

    return patch_map, len(patch_map), len(orig_tokens)


# ---------------------------------------------------------------------------
# Layer-level patching (diff-aware)
# ---------------------------------------------------------------------------

def patch_layer(model, orig_input_ids, adv_activations, patch_layer_idx, patch_map):
    """
    Run original input through the model, but at patch_layer_idx replace ONLY
    the changed token positions (identified by patch_map) with adversarial 
    activations. Shared tokens are left untouched.
    """
    def patch_hook(module, input, output):
        # HF internals vary: decoder layer may return Tensor or tuple(Tensor, ...)
        if torch.is_tensor(output):
            hidden = output
            tail = None
            returns_tuple = False
        else:
            hidden = output[0]
            tail = output[1:]
            returns_tuple = True
        was_2d = hidden.dim() == 2
        if was_2d:
            hidden = hidden.unsqueeze(0)

        adv_hidden = adv_activations[patch_layer_idx]  # (1, adv_seq_len, hidden_dim)
        new_hidden = hidden.clone()

        for orig_pos, adv_pos in patch_map.items():
            if orig_pos < new_hidden.shape[1] and adv_pos < adv_hidden.shape[1]:
                new_hidden[:, orig_pos, :] = adv_hidden[:, adv_pos, :]

        if was_2d:
            new_hidden = new_hidden.squeeze(0)

        if returns_tuple:
            return (new_hidden,) + tail
        return new_hidden

    hook = model.model.layers[patch_layer_idx].register_forward_hook(patch_hook)

    with torch.no_grad():
        patched_outputs = model(orig_input_ids)

    hook.remove()

    return patched_outputs.logits


def compute_kl_divergence(orig_logits, patched_logits):
    """
    Compute KL divergence between original and patched output distributions
    at the last token position. Returns scalar.
    """
    # Use float32 for numerical stability
    orig_probs = torch.nn.functional.softmax(orig_logits[0, -1, :].float(), dim=-1)
    patched_probs = torch.nn.functional.softmax(patched_logits[0, -1, :].float(), dim=-1)

    # KL(orig || patched): how much does patching change the output?
    kl = torch.nn.functional.kl_div(
        patched_probs.log(),
        orig_probs,
        reduction="sum",
        log_target=False,
    )
    return kl.item()


def compute_logprob_diff(orig_logits, patched_logits, target_token_id):
    """
    Compute the difference in log-probability of a specific target token.
    Positive = patching increased probability, Negative = decreased.
    """
    orig_logprobs = torch.nn.functional.log_softmax(orig_logits[0, -1, :].float(), dim=-1)
    patched_logprobs = torch.nn.functional.log_softmax(patched_logits[0, -1, :].float(), dim=-1)

    diff = patched_logprobs[target_token_id] - orig_logprobs[target_token_id]
    return diff.item()


# ---------------------------------------------------------------------------
# Main patching loop
# ---------------------------------------------------------------------------

def run_patching(domain: str, model, tokenizer, n_layers):
    """Run layer-level patching for a domain."""

    pairs = load_item_pairs(domain)
    if not pairs:
        print(f"No pairs found for {domain}, skipping.")
        return

    out_path = RESULTS_DIR / f"{domain}_layer_patching.jsonl"

    # Check for existing progress
    done_items = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_items.add(rec["item_idx"])
        print(f"Resuming: {len(done_items)} items already done")

    remaining = [p for p in pairs if p["item_idx"] not in done_items]
    print(f"Items to process: {len(remaining)}")

    for pair_i, pair in enumerate(remaining):
        t0 = time.time()
        item_idx = pair["item_idx"]

        # Tokenize
        orig_ids = tokenizer(
            pair["original_text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(DEVICE)

        adv_ids = tokenizer(
            pair["adversarial_text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(DEVICE)

        # Cache activations for both
        orig_acts, orig_logits = get_residual_stream_activations(model, orig_ids, n_layers)
        adv_acts, adv_logits = get_residual_stream_activations(model, adv_ids, n_layers)

        # Compute token-level diff
        patch_map, n_changed, n_total = compute_token_diff(orig_ids, adv_ids)

        if n_changed == 0:
            # Adversarial is token-identical to original — skip
            print(f"  [{pair_i+1}/{len(remaining)}] item {item_idx:3d} | SKIPPED (0 tokens changed)")
            del orig_acts, adv_acts, orig_logits, adv_logits
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
            continue

        # Get the original model's top prediction for reference
        orig_top_token = orig_logits[0, -1, :].argmax().item()

        # Patch each layer and measure KL divergence
        kl_by_layer = []
        logprob_diff_by_layer = []

        for layer_idx in range(n_layers):
            patched_logits = patch_layer(
                model, orig_ids, adv_acts, layer_idx, patch_map
            )

            kl = compute_kl_divergence(orig_logits, patched_logits)
            lp_diff = compute_logprob_diff(orig_logits, patched_logits, orig_top_token)

            kl_by_layer.append(kl)
            logprob_diff_by_layer.append(lp_diff)

        # Clean up GPU memory
        del orig_acts, adv_acts, orig_logits, adv_logits
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        elapsed = time.time() - t0

        # Save result
        result = {
            "item_idx": item_idx,
            "domain": domain,
            "stratum": pair["stratum"],
            "orig_len": orig_ids.shape[1],
            "adv_len": adv_ids.shape[1],
            "len_diff": abs(orig_ids.shape[1] - adv_ids.shape[1]),
            "n_tokens_changed": n_changed,
            "n_tokens_total": n_total,
            "pct_changed": round(n_changed / n_total * 100, 1),
            "kl_by_layer": kl_by_layer,
            "logprob_diff_by_layer": logprob_diff_by_layer,
            "elapsed_sec": round(elapsed, 2),
        }

        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Progress
        avg_kl = np.mean(kl_by_layer)
        max_kl_layer = int(np.argmax(kl_by_layer))
        kl_range = max(kl_by_layer) - min(kl_by_layer)
        print(
            f"  [{pair_i+1}/{len(remaining)}] item {item_idx:3d} | "
            f"changed {n_changed}/{n_total} tokens ({result['pct_changed']}%) | "
            f"max KL layer={max_kl_layer} (KL={kl_by_layer[max_kl_layer]:.4f}) | "
            f"KL range={kl_range:.4f} | {elapsed:.1f}s"
        )

    print(f"\nDone. Output: {out_path}")


# ---------------------------------------------------------------------------
# Aggregation and visualization
# ---------------------------------------------------------------------------

def aggregate_results():
    """Aggregate patching results across domains and produce summary + plots."""

    all_records = []
    for domain in ["medical", "legal", "swe"]:
        path = RESULTS_DIR / f"{domain}_layer_patching.jsonl"
        if not path.exists():
            print(f"Skipping {domain}: no results found")
            continue
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))

    if not all_records:
        print("No results to aggregate.")
        return

    print(f"Aggregating {len(all_records)} items across domains")

    # Build a matrix: (items × layers) for KL divergence
    domains = sorted(set(r["domain"] for r in all_records))
    n_layers = len(all_records[0]["kl_by_layer"])

    # Per-domain average KL by layer
    summary = {}
    for domain in domains:
        domain_records = [r for r in all_records if r["domain"] == domain]
        kl_matrix = np.array([r["kl_by_layer"] for r in domain_records])
        lp_matrix = np.array([r["logprob_diff_by_layer"] for r in domain_records])

        summary[domain] = {
            "n_items": len(domain_records),
            "mean_kl_by_layer": kl_matrix.mean(axis=0).tolist(),
            "std_kl_by_layer": kl_matrix.std(axis=0).tolist(),
            "mean_lp_diff_by_layer": lp_matrix.mean(axis=0).tolist(),
            "max_kl_layer": int(kl_matrix.mean(axis=0).argmax()),
            "overall_mean_kl": float(kl_matrix.mean()),
        }

        print(f"\n--- {domain.upper()} ({len(domain_records)} items) ---")
        print(f"  Peak KL at layer {summary[domain]['max_kl_layer']}")
        print(f"  Overall mean KL: {summary[domain]['overall_mean_kl']:.4f}")
        print(f"  Top 5 layers by mean KL:")
        ranked = np.argsort(kl_matrix.mean(axis=0))[::-1][:5]
        for rank, layer in enumerate(ranked):
            print(f"    {rank+1}. Layer {layer}: KL={kl_matrix.mean(axis=0)[layer]:.4f} ± {kl_matrix.std(axis=0)[layer]:.4f}")

    # Per-stratum breakdown
    strata = sorted(set(r["stratum"] for r in all_records))
    print(f"\n--- Stratum breakdown ---")
    for stratum in strata:
        stratum_records = [r for r in all_records if r["stratum"] == stratum]
        kl_matrix = np.array([r["kl_by_layer"] for r in stratum_records])
        max_layer = int(kl_matrix.mean(axis=0).argmax())
        print(f"  {stratum:40s} | n={len(stratum_records):3d} | peak layer={max_layer} | mean KL={kl_matrix.mean():.4f}")

    # Save summary
    summary_path = RESULTS_DIR / "patching_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5), sharey=True)
        if len(domains) == 1:
            axes = [axes]

        for ax, domain in zip(axes, domains):
            mean_kl = np.array(summary[domain]["mean_kl_by_layer"])
            std_kl = np.array(summary[domain]["std_kl_by_layer"])
            layers = np.arange(n_layers)

            ax.plot(layers, mean_kl, color="steelblue", linewidth=2)
            ax.fill_between(layers, mean_kl - std_kl, mean_kl + std_kl,
                            alpha=0.2, color="steelblue")
            ax.axvline(summary[domain]["max_kl_layer"], color="red",
                        linestyle="--", alpha=0.7, label=f"Peak: layer {summary[domain]['max_kl_layer']}")
            ax.set_xlabel("Layer")
            ax.set_title(f"{domain.upper()} (n={summary[domain]['n_items']})")
            ax.legend(fontsize=9)

        axes[0].set_ylabel("KL Divergence (orig → patched)")
        fig.suptitle("Layer-Level Activation Patching: Original ↔ Adversarial", fontsize=14)
        plt.tight_layout()

        plot_path = RESULTS_DIR / "layer_patching_kl.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot generation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-level activation patching")
    parser.add_argument("--domain", choices=["medical", "legal", "swe", "all"], default=None)
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results and plot")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
    else:
        model, tokenizer, n_layers = load_model()

        if args.domain == "all":
            for d in ["medical", "legal", "swe"]:
                run_patching(d, model, tokenizer, n_layers)
        elif args.domain:
            run_patching(args.domain, model, tokenizer, n_layers)
        else:
            parser.print_help()
