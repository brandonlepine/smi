#!/usr/bin/env python3
"""
Stage 4: Causal Tracing  (Revised)

Central question:
    Is causal influence disproportionately concentrated in early layer ranges,
    and does that concentration differ by attribute family?

Primary metrics (priority order):
    1. recovery_to_orig_logit  — does mean-ablation of focal heads move the CF
                                  run back toward the original run?
    2. recovery_to_orig_margin — same, in margin space
    3. pred_matches_orig       — prediction-level recovery
    4. flip_reduction          — reduction in answer-flip rate

Secondary / corroborating metrics:
    5. destructive_causal_fraction_logit  — fraction of |Δ logit| eliminated
                                            under ZERO-ablation. Can be negative
                                            if zero-ablation induces OOD behaviour.
    6. destructive_causal_fraction_margin — same for margin

Ablation modes:
    PRIMARY:   Matched-QID baseline ablation — replaces head activations with a
                               baseline captured from the ORIGINAL prompt for the
                               same question (same QID, same layer). Avoids
                               out-of-distribution activation values and avoids
                               mixing baselines across unrelated questions.
    SECONDARY: Zero ablation — replaces head activations with zero.  Retained for
                               comparison; negative causal fractions here should be
                               interpreted as OOD collateral damage, not as evidence
                               against causal involvement.

Two experimental cases:
  Case 1 — Gender Identity, Race, Sex/Gender
    A1. Individual ablation of early-layer heads: per-head recovery statistics
    A2. Stepwise cumulative ablation: early-heads curve vs full-heads curve
    A3. Layer-group comparison: early / mid / late
    A4. Direction patching: CF→Orig (recovery) and Orig→CF (injection)

  Case 2 — Sexual Orientation  (no heads in layers 0–3)
    B1. Multi-head ablation at top-5 / top-10 / top-20
    B2. Residual stream patching by layer: carrier-layer identification
    B3. Context-sensitivity split: partner-framed vs explicit-identity edits

Output hierarchy:
    stage4_results/
      assignment_log.json
      summary.json
      plots/
      {family}/
        {experiment}/
          metadata.json
          aggregate.json
          per_pair.csv
          per_head.csv   (A1 only)
          curve.csv      (A2 only)
          per_layer.csv  (B2 only)

Usage
-----
  python analyze_stage4_causal_tracing.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --top_heads_path stage3_results/top_heads.json \\
    --output_dir stage4_results \\
    --device auto \\
    --max_pairs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("Warning: matplotlib not available — plots skipped")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Question:
{question}

Answer choices:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

AIDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDXA = {0: "A", 1: "B", 2: "C", 3: "D"}

CASE1_FAMILIES = ["gender_identity", "race", "sex_gender"]
CASE2_FAMILIES = ["sexual_orientation"]
ALL_FOCAL = CASE1_FAMILIES + CASE2_FAMILIES

# Minimum |Δ gold logit| to include a pair in ratio-based summaries.
# Passed explicitly through the call stack; never mutated at runtime.
MIN_EFFECT_THRESHOLD = 0.05

# Probability deltas are on a different scale than logits; treat probability-based
# recovery/injection as exploratory and gate only on a tiny epsilon for stability.
PROB_EFFECT_EPS = 1e-6

# Clearing the CUDA cache on every forward pass is usually counterproductive.
# Gate behind a CLI flag (default: off).
AGGRESSIVE_CUDA_CACHE_CLEAR = False

FAMILY_COLORS = {
    "gender_identity":    "#9B59B6",
    "race":               "#E67E22",
    "sex_gender":         "#4C96D7",
    "sexual_orientation": "#2ECC71",
    "control":            "#95A5A6",
}

_GI_FRAGMENTS = {
    "non-binary", "nonbinary", "non binary",
    "transgender", "trans man", "trans woman", "transman", "transwoman",
    "gender non-conforming", "genderqueer", "agender",
}

_PARTNER_PATTERNS = {"partner", "same sex partner", "partner same sex"}
_EXPLICIT_PATTERNS = {"gay", "straight", "lesbian", "bisexual", "queer",
                      "heterosexual", "homosexual"}

# Ordered priority for CSV column output
_CSV_PRIORITY_COLS = [
    "qid", "family", "itype", "attr_val", "gold",
    "primary_estimand",
    "orig_correct", "is_significant", "answer_flip", "correctness_flip",
    "orig_gold_logit", "cf_gold_logit", "abl_gold_logit",
    "signed_delta_gold_logit", "abs_delta_gold_logit",
    "recovery_to_orig_logit", "recovery_to_orig_margin", "recovery_to_orig_prob",
    "pred_matches_orig", "pred_matches_cf",
    "flip_reduction",
    "overshoot_logit", "no_effect_logit",
    "same_direction_after_intervention",
    "attenuation_without_reversal",
    "reversal_after_intervention",
    "amplification_without_reversal",
    "overshoot_margin", "no_effect_margin",
    "overshoot_prob", "no_effect_prob",
    "destructive_causal_fraction_logit",
    "baseline_delta_sign", "post_intervention_delta_sign", "delta_sign_changed",
]


# ---------------------------------------------------------------------------
# Family assignment
# ---------------------------------------------------------------------------

def assign_family(itype: str, attr_val) -> str | None:
    """
    Map (intervention_type, attribute_value) → family key.
    Priority:
      1. intervention_type is authoritative when unambiguous.
      2. For "sex_gender" and legacy "sex" itypes, attr_val heuristics separate
         binary-sex labels from gender-identity labels.
      3. Returns None for unrecognised itypes.
    """
    itype = (itype or "").strip()
    attr_norm = _normalize_label(attr_val)
    if itype in {"neutral_rework", "irrelevant_surface"}:
        return "control"
    if itype in {"race_ethnicity", "omit_race"}:
        return "race"
    if itype == "sexual_orientation":
        return "sexual_orientation"
    if itype == "gender_identity":
        return "gender_identity"
    if itype in {"sex", "sex_gender"}:
        if any(frag in attr_norm for frag in _GI_FRAGMENTS):
            return "gender_identity"
        return "sex_gender"
    return None


def _normalize_label(label) -> str:
    s = str(label or "").lower()
    s = re.sub(r"['\"\-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def format_prompt(q, options):
    return PROMPT_TEMPLATE.format(
        question=q.strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


def load_pairs(data_path: str, include_controls: bool = True) -> tuple[list, dict]:
    """
    Returns (pairs, assignment_log).
    Controls included when include_controls=True so downstream experiments
    can do matched focal-vs-control ablation comparisons.
    """
    with open(data_path) as f:
        records = json.load(f)

    pairs = []
    assignment_counts: Counter = Counter()
    assignment_examples: dict = defaultdict(list)

    for rec in records:
        qid = rec["question_id"]
        orig = rec["original"]
        options = orig["options"]
        gold = orig.get("answer_idx", "")
        if not gold:
            for k, v in options.items():
                if v == orig["answer"]:
                    gold = k
                    break
        if gold not in AIDX:
            continue

        orig_prompt = format_prompt(orig["question"], options)
        variants = rec.get("counterfactuals", {}).get("variants", [])

        for v in variants:
            if not isinstance(v, dict) or v.get("text") is None:
                continue
            itype = v.get("intervention_type", "")
            attr_val = v.get("attribute_value_counterfactual", "")
            family = assign_family(itype, attr_val)

            if family is None:
                assignment_counts["_dropped"] += 1
                continue
            if family == "control" and not include_controls:
                continue

            assignment_counts[family] += 1
            if len(assignment_examples[family]) < 5:
                assignment_examples[family].append(
                    {"itype": itype, "attr_val": str(attr_val), "qid": qid}
                )

            pairs.append({
                "qid": qid,
                "family": family,
                "itype": itype,
                "attr_val": str(attr_val or itype),
                "attr_norm": _normalize_label(attr_val),
                "gold": gold,
                "options": options,
                "orig_q": orig["question"],
                "cf_q": v["text"],
                "orig_prompt": orig_prompt,
                "cf_prompt": format_prompt(v["text"], options),
            })

    assignment_log = {
        fam: {"count": assignment_counts[fam], "examples": assignment_examples[fam]}
        for fam in sorted(assignment_counts)
    }
    return pairs, assignment_log


def sample_pairs(all_pairs: list, family: str, max_n: int, rng) -> list:
    subset = [p for p in all_pairs if p["family"] == family]
    if len(subset) > max_n:
        idx = rng.choice(len(subset), max_n, replace=False)
        subset = [subset[i] for i in sorted(idx)]
    return subset


def sample_matched_controls(all_pairs: list, focal_pairs: list, max_n: int, rng,
                             global_warnings: list) -> tuple[list, float]:
    """
    Sample control pairs maximising matched-QID coverage.
    Take all matched controls first (up to max_n), fill remainder with unmatched.
    Returns (controls, frac_matched).
    If matched fraction < 50%, adds a warning and outputs should label them
    'reference controls' rather than 'matched controls'.
    """
    focal_qids = {p["qid"] for p in focal_pairs}
    matched = [p for p in all_pairs if p["family"] == "control" and p["qid"] in focal_qids]
    unmatched = [p for p in all_pairs if p["family"] == "control" and p["qid"] not in focal_qids]

    n_matched_avail, n_unmatched_avail = len(matched), len(unmatched)

    if len(matched) >= max_n:
        idx = rng.choice(len(matched), max_n, replace=False)
        controls = [matched[i] for i in sorted(idx)]
    else:
        remainder = max_n - len(matched)
        fill = unmatched if len(unmatched) <= remainder else [
            unmatched[i] for i in sorted(rng.choice(len(unmatched), remainder, replace=False))
        ]
        controls = matched + fill

    n_matched_sampled = sum(1 for c in controls if c["qid"] in focal_qids)
    n_unmatched_sampled = len(controls) - n_matched_sampled
    frac_matched = n_matched_sampled / len(controls) if controls else 0.0
    ctrl_label = "matched controls" if frac_matched >= 0.5 else "reference controls"
    print(f"  Control sampling: {n_matched_sampled} matched + {n_unmatched_sampled} unmatched "
          f"= {len(controls)} total ({frac_matched:.0%} matched) → '{ctrl_label}'  "
          f"[pool: {n_matched_avail} matched, {n_unmatched_avail} unmatched]")
    if controls and frac_matched < 0.5:
        w = (f"Matched control fraction {frac_matched:.0%} < 50% — "
             f"label as 'reference controls' in writeup.")
        print(f"  Warning: {w}")
        global_warnings.append(w)
    return controls, frac_matched


# ---------------------------------------------------------------------------
# Orientation context split
# ---------------------------------------------------------------------------

def orientation_split_label(attr_norm: str) -> str:
    if any(p in attr_norm for p in _PARTNER_PATTERNS):
        return "partner"
    if any(p in attr_norm for p in _EXPLICIT_PATTERNS):
        return "explicit"
    return "other"


# ---------------------------------------------------------------------------
# Model loading + validation
# ---------------------------------------------------------------------------

def choose_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path, device, dtype):
    print(f"Loading tokenizer: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("Loading model...")
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", **kw
    )
    if device in {"cpu", "mps"}:
        model = model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"  layers={n_layers}  heads={n_heads}  head_dim={head_dim}")
    return model, tok, n_layers, n_heads, head_dim


def get_answer_token_ids(tok):
    ids = {}
    for letter in "ABCD":
        for s in [letter, f" {letter}"]:
            toks = tok.encode(s, add_special_tokens=False)
            if toks and letter in tok.decode([toks[-1]]).strip():
                ids[letter] = toks[-1]
                break
        if letter not in ids:
            raise ValueError(f"Cannot resolve token id for '{letter}'")
    return ids


def validate_o_proj_shape(model, tok, n_heads: int, head_dim: int, device: str):
    """Assert o_proj receives (batch, seq, n_heads*head_dim). Fails loudly before any ablation."""
    captured = {}

    def cap(module, args):
        captured["shape"] = tuple(args[0].shape)

    dummy = tok("Validate o_proj shape.", return_tensors="pt")
    dummy = {k: v.to(device) for k, v in dummy.items()}
    handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(cap)
    with torch.no_grad():
        model(**dummy, use_cache=False, output_hidden_states=False)
    handle.remove()
    shape = captured.get("shape")
    assert shape is not None, "o_proj pre-hook did not fire"
    assert shape[-1] == n_heads * head_dim, (
        f"o_proj input last-dim={shape[-1]} != n_heads*head_dim={n_heads*head_dim}. "
        f"Head slices would be mislabelled."
    )
    print(f"  o_proj shape validated: {shape} ✓  (n_heads={n_heads}, head_dim={head_dim})")


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

def ablation_pre_hook(heads_to_ablate: list[int], head_dim: int):
    """
    ZERO-ablation: replaces head outputs with zero.
    Destructive — use only as secondary/corroborating metric.
    Negative causal fractions from this hook indicate OOD collateral damage.
    """
    def hook(module, args):
        x = args[0].clone()
        for h in heads_to_ablate:
            x[:, :, h * head_dim:(h + 1) * head_dim] = 0.0
        return (x,)
    return hook


def qid_baseline_pre_hook(heads_to_ablate: list[int], head_dim: int,
                          baseline_acts_by_qid: dict, layer_idx: int,
                          qid: str,
                          scope: str = "final_token"):
    """
    Matched-QID baseline ablation: replaces head outputs with a pre-computed baseline
    from the ORIGINAL prompt for the same question ID (QID).

    Primary ablation mode — avoids out-of-distribution activations.
    Falls back to zero for any (qid, layer) whose baseline was not captured.
    """
    def hook(module, args):
        x = args[0].clone()
        mean_vec = (baseline_acts_by_qid.get(qid) or {}).get(layer_idx)
        for h in heads_to_ablate:
            start = h * head_dim
            end = (h + 1) * head_dim
            if mean_vec is not None:
                mv = mean_vec[start:end].to(device=x.device, dtype=x.dtype)
                if scope == "final_token":
                    x[:, -1:, start:end] = mv.unsqueeze(0).unsqueeze(0)
                else:
                    x[:, :, start:end] = mv.unsqueeze(0).unsqueeze(0)
            else:
                if scope == "final_token":
                    x[:, -1:, start:end] = 0.0
                else:
                    x[:, :, start:end] = 0.0  # graceful fallback
        return (x,)
    return hook


# Back-compat alias (kept to minimize churn in older call sites).
mean_ablation_pre_hook = qid_baseline_pre_hook


def save_pre_hook(store: dict, layer_idx: int, scope: str = "final_token"):
    """
    Forward-pre-hook saver for o_proj inputs.
    When scope="final_token", stores only the final sequence position to avoid
    mismatched seq_len between original vs counterfactual prompts.
    """
    def hook(module, args):
        x = args[0].detach()
        store[layer_idx] = (x[:, -1:, :].clone() if scope == "final_token" else x.clone())
    return hook


def patch_pre_hook(source_store: dict, layer_idx: int,
                   heads_to_patch: list[int], head_dim: int,
                   scope: str = "final_token"):
    def hook(module, args):
        src = source_store.get(layer_idx)
        if src is None:
            return None
        x = args[0].clone()
        src_t = src.to(device=x.device, dtype=x.dtype)
        if scope == "final_token":
            # Patch only the final token to remain well-defined under seq_len mismatch.
            if src_t.ndim == 3 and src_t.shape[1] != 1:
                src_tok = src_t[:, -1:, :]
            else:
                src_tok = src_t
            for h in heads_to_patch:
                start = h * head_dim
                end = (h + 1) * head_dim
                x[:, -1:, start:end] = src_tok[:, -1:, start:end]
            return (x,)

        # all_positions: patch the overlapping prefix only (never change seq_len)
        min_len = min(x.shape[1], src_t.shape[1])
        for h in heads_to_patch:
            start = h * head_dim
            end = (h + 1) * head_dim
            x[:, :min_len, start:end] = src_t[:, :min_len, start:end]
        return (x,)
    return hook


def save_residual_hook(store: dict, layer_idx: int):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        store[layer_idx] = hs.detach().clone()
    return hook


def residual_patch_hook(source_store: dict, layer_idx: int):
    """Replace final-token residual stream from source run."""
    def hook(module, args, output):
        src = source_store.get(layer_idx)
        if src is None:
            return output
        hs = output[0] if isinstance(output, (tuple, list)) else output
        new_hs = hs.clone()
        src_t = src.to(device=new_hs.device, dtype=new_hs.dtype)
        new_hs[:, -1:, :] = src_t[:, -1:, :]
        if isinstance(output, (tuple, list)):
            return (new_hs,) + tuple(output[1:])
        return new_hs
    return hook


# ---------------------------------------------------------------------------
# MLP and attention layer-level hooks  (for C1–C7 component experiments)
# ---------------------------------------------------------------------------

def save_attn_out_hook(store: dict, layer_idx: int):
    """
    Post-hook on model.model.layers[l].self_attn.
    Saves output[0] — the attention output AFTER o_proj, before the residual add.
    Shape: (batch, seq_len, hidden_size).
    """
    def hook(module, args, output):
        store[layer_idx] = output[0].detach().clone()
    return hook


def patch_attn_out_hook(source_store: dict, layer_idx: int,
                         intervention_scope: str = "final_token"):
    """
    Post-hook on self_attn.
    Replaces output[0] with saved source attention output.
    When intervention_scope="final_token", only the final sequence position is patched.
    """
    def hook(module, args, output):
        src = source_store.get(layer_idx)
        if src is None:
            return output
        new_hs = output[0].clone()
        src_t = src.to(device=new_hs.device, dtype=new_hs.dtype)
        if intervention_scope == "final_token":
            new_hs[:, -1:, :] = src_t[:, -1:, :]
        else:
            # Patch overlapping prefix only; keep target seq_len intact.
            min_len = min(new_hs.shape[1], src_t.shape[1])
            new_hs[:, :min_len, :] = src_t[:, :min_len, :]
        return (new_hs,) + output[1:]
    return hook


def save_mlp_out_hook(store: dict, layer_idx: int):
    """
    Post-hook on model.model.layers[l].mlp.
    Saves the MLP output tensor BEFORE the residual add.
    Shape: (batch, seq_len, hidden_size).
    """
    def hook(module, args, output):
        store[layer_idx] = output.detach().clone()
    return hook


def patch_mlp_out_hook(source_store: dict, layer_idx: int,
                        intervention_scope: str = "final_token"):
    """
    Post-hook on mlp.
    Replaces mlp output with saved source MLP output.
    When intervention_scope="final_token", only the final sequence position is patched.
    """
    def hook(module, args, output):
        src = source_store.get(layer_idx)
        if src is None:
            return output
        src_t = src.to(device=output.device, dtype=output.dtype)
        if intervention_scope == "final_token":
            out = output.clone()
            out[:, -1:, :] = src_t[:, -1:, :]
            return out
        # Patch overlapping prefix only; keep target seq_len intact.
        out = output.clone()
        min_len = min(out.shape[1], src_t.shape[1])
        out[:, :min_len, :] = src_t[:, :min_len, :]
        return out
    return hook


# ---------------------------------------------------------------------------
# Mean activation computation
# ---------------------------------------------------------------------------

def compute_mean_head_activations(model, tok, pairs: list, layers_of_interest: list,
                                   head_dim: int, device: str,
                                   max_pairs: int = 50,
                                   scope: str = "final_token") -> dict:
    """
    Run original prompts and compute per-layer mean o_proj pre-activation vectors.
    Returns {layer_idx: tensor of shape (n_heads * head_dim,)}.
    When scope="final_token", mean is taken over pairs at the final sequence position.
    When scope="all_positions", mean is taken over (pairs × seq_positions).
    Used by mean_ablation_pre_hook as the primary ablation value.
    """
    if not layers_of_interest:
        return {}

    captures: dict[int, list] = defaultdict(list)
    layers = sorted(set(layers_of_interest))

    def make_hook(l):
        def hook(module, args):
            # args[0]: (1, seq_len, n_heads * head_dim)
            if scope == "final_token":
                captures[l].append(args[0].detach().float()[0, -1, :].cpu())
            else:
                captures[l].append(args[0].detach().float().mean(dim=(0, 1)).cpu())
        return hook

    handles = [
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
        for l in layers
    ]
    sample = pairs[:max_pairs]
    with torch.no_grad():
        for pair in sample:
            try:
                inp = {k: v.to(device) for k, v in
                       tok(pair["orig_prompt"], return_tensors="pt",
                           truncation=True, max_length=2048).items()}
                model(**inp, output_hidden_states=False, use_cache=False)
            except Exception:
                pass
    for h in handles:
        h.remove()

    result = {}
    for l in layers:
        if captures[l]:
            result[l] = torch.stack(captures[l]).mean(dim=0)  # (n_heads * head_dim,)
    return result


def compute_qid_matched_head_activations(
    model, tok, pairs: list, layers_of_interest: list,
    device: str,
    scope: str = "final_token",
) -> dict:
    """
    Compute matched-QID baselines for mean ablation.

    Returns:
        {qid: {layer_idx: tensor of shape (n_heads * head_dim,)}}

    Implementation:
      - One forward pass per unique QID using that QID's ORIGINAL prompt.
      - Captures o_proj pre-activations (args[0]) at either the final token or
        mean across positions depending on `scope`.
    """
    if not layers_of_interest:
        return {}

    # Map qid -> representative original prompt (keep first; prompts should match within QID)
    qid_to_prompt: dict = {}
    prompt_mismatch = 0
    for p in pairs:
        qid = p.get("qid")
        if qid is None:
            continue
        op = p.get("orig_prompt")
        if qid not in qid_to_prompt:
            qid_to_prompt[qid] = op
        else:
            if op is not None and qid_to_prompt[qid] is not None and op != qid_to_prompt[qid]:
                prompt_mismatch += 1

    layers = sorted(set(layers_of_interest))
    captures: dict = {}
    current_qid = None
    n_fail = 0

    def make_hook(l):
        def hook(module, args):
            if current_qid is None:
                return None
            x = args[0].detach().float()
            if scope == "final_token":
                vec = x[0, -1, :].cpu()
            else:
                vec = x.mean(dim=(0, 1)).cpu()
            captures.setdefault(current_qid, {})[l] = vec
            return None
        return hook

    handles = [
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
        for l in layers
    ]
    with torch.no_grad():
        for i, (qid, prompt) in enumerate(qid_to_prompt.items()):
            current_qid = qid
            try:
                inp = {k: v.to(device) for k, v in
                       tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).items()}
                model(**inp, output_hidden_states=False, use_cache=False)
            except Exception:
                n_fail += 1
            if (i + 1) % 25 == 0:
                print(f"    QID means: {i+1}/{len(qid_to_prompt)}", end="\r", flush=True)
    current_qid = None
    for h in handles:
        h.remove()
    if qid_to_prompt:
        print()
    if prompt_mismatch:
        print(f"    Warning: {prompt_mismatch} QID prompt mismatches detected; using first prompt per QID.")
    if n_fail:
        print(f"    Warning: QID baseline failures: {n_fail}/{len(qid_to_prompt)}")
    return captures


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def tokenize(tok, prompt: str, device: str) -> dict:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def forward_pass(model, inputs: dict, answer_ids: dict,
                 extra_hooks: list | None = None) -> dict[str, float]:
    handles = []
    if extra_hooks:
        for (target, hook_fn, is_pre) in extra_hooks:
            if is_pre:
                handles.append(target.register_forward_pre_hook(hook_fn))
            else:
                handles.append(target.register_forward_hook(hook_fn))
    try:
        out = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        if AGGRESSIVE_CUDA_CACHE_CLEAR and model.device.type == "cuda":
            torch.cuda.empty_cache()
    logits = out.logits[0, -1, :].float()
    return {k: float(logits[tid].cpu()) for k, tid in answer_ids.items()}


def behavioral_metrics(logit_dict: dict, gold: str) -> dict:
    l = np.array([logit_dict[k] for k in "ABCD"])
    gi = AIDX[gold]
    pred = IDXA[int(np.argmax(l))]
    gl = float(l[gi])
    others = np.delete(l, gi)
    margin = float(gl - others.max())
    shifted = l - l.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    return {
        "gold_logit": gl,
        "margin": margin,
        "gold_prob": float(probs[gi]),
        "pred": pred,
        "correct": pred == gold,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def causal_fraction(abs_original: float, abs_remaining: float,
                    min_effect: float) -> float | None:
    """
    Fraction of |Δ gold logit| eliminated by ablation.
    Returns None when |baseline| < min_effect (ratio unstable).
    CAN BE NEGATIVE: negative values mean the ablation amplified the effect
    (typical of zero-ablation OOD damage). Report with that caveat.
    """
    if abs_original < min_effect:
        return None
    return (abs_original - abs_remaining) / (abs_original + 1e-8)


def recovery_score(delta: float, delta_abl: float,
                   min_effect: float) -> float | None:
    """
    Signed recovery toward original run.
      1.0 = perfect recovery (ablated run == original run)
      0.0 = no effect (ablated run == counterfactual run)
      >1   = overshot original
      <0   = ablation pushed further from original

    Returns None when |delta| < min_effect (low-signal pair).
    """
    if abs(delta) < min_effect:
        return None
    return float(np.clip(1.0 - delta_abl / (delta + 1e-8), -3.0, 3.0))


# ---------------------------------------------------------------------------
# Head candidate selection
# ---------------------------------------------------------------------------

def filter_heads_by_layer(top_heads_data: dict, family: str,
                           layer_range, n: int) -> list[tuple[int, int]]:
    return [
        (h["layer"], h["head"])
        for h in top_heads_data[family]["top_heads"]
        if h["layer"] in layer_range
    ][:n]


def top_n_heads(top_heads_data: dict, family: str, n: int) -> list[tuple[int, int]]:
    return [(h["layer"], h["head"]) for h in top_heads_data[family]["top_heads"][:n]]


def heads_to_layer_dict(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    for l, h in heads:
        d[l].append(h)
    return dict(d)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def agg_stats(vals: list) -> dict:
    """Full statistics for a list of values, excluding Nones."""
    valid = [v for v in vals if v is not None]
    n, nv = len(vals), len(valid)
    if nv == 0:
        return {"n": n, "n_valid": 0, "mean": None, "std": None, "median": None,
                "p25": None, "p75": None, "frac_positive": None, "frac_negative": None}
    arr = np.array(valid, dtype=float)
    return {
        "n": n, "n_valid": nv,
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
        "frac_positive": float(np.mean(arr > 0)),
        "frac_negative": float(np.mean(arr < 0)),
    }


def mean_bool_rate(vals: list) -> float | None:
    """Compute rate of True values in a list, ignoring None. Returns None if no valid values."""
    valid = [bool(v) for v in vals if v is not None]
    return float(sum(valid) / len(valid)) if valid else None


def top_k_layers_by_recovery(layer_agg: dict, k: int) -> list[tuple[int, float]]:
    """
    Return top-k layers by mean recovery from a layer_agg dict.
    Skips layers with None mean or None/0 n_valid.
    Returns list of (layer_idx, mean_recovery) sorted descending.
    """
    candidates = []
    for key, v in layer_agg.items():
        if v.get("mean") is None or not v.get("n_valid"):
            continue
        try:
            candidates.append((int(key), float(v["mean"])))
        except (ValueError, TypeError):
            continue
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]


_PRIMARY_METRICS = [
    "recovery_to_orig_logit",
    "recovery_to_orig_margin",
    "recovery_to_orig_prob",
    "signed_movement_toward_orig_logit",
    "signed_movement_toward_orig_margin",
]
_SECONDARY_METRICS = [
    "destructive_causal_fraction_logit",
    "destructive_causal_fraction_margin",
]


def aggregate_v2(per_pair: list[dict], extra_metrics: list[str] | None = None) -> dict:
    """
    Stratified aggregation for ablation per-pair records.
    Strata: all / orig_correct / significant / correct_sig /
            answer_flip_pairs / correctness_flip_pairs
    Primary metrics: recovery_to_orig_* (reported first and labeled PRIMARY)
    Secondary metrics: destructive_causal_fraction_* (labeled SECONDARY)
    """
    if not per_pair:
        return {}

    all_metrics = _PRIMARY_METRICS + _SECONDARY_METRICS + (extra_metrics or [])
    strata = {
        "all":               per_pair,
        "orig_correct":      [r for r in per_pair if r.get("orig_correct")],
        "significant":       [r for r in per_pair if r.get("is_significant")],
        "correct_sig":       [r for r in per_pair if r.get("orig_correct")
                               and r.get("is_significant")],
        "answer_flip_pairs": [r for r in per_pair if r.get("answer_flip")],
        "correctness_flip":  [r for r in per_pair if r.get("correctness_flip")],
    }

    out = {}
    for sname, rows in strata.items():
        if not rows:
            out[sname] = {"n": 0}
            continue
        s = {"n": len(rows), "_primary_metrics": _PRIMARY_METRICS,
             "_secondary_metrics": _SECONDARY_METRICS}
        for m in all_metrics:
            s[m] = agg_stats([r.get(m) for r in rows])
        for bm in ["pred_matches_orig", "pred_matches_cf",
                   "answer_flip", "correctness_flip", "abl_flip",
                   "overshoot_logit", "overshoot_margin", "overshoot_prob",
                   "no_effect_logit", "no_effect_margin", "no_effect_prob",
                   "same_direction_after_intervention",
                   "attenuation_without_reversal",
                   "reversal_after_intervention",
                   "amplification_without_reversal"]:
            s[bm + "_rate"] = mean_bool_rate([r.get(bm) for r in rows])
        fr = [r.get("flip_reduction") for r in rows if r.get("flip_reduction") is not None]
        s["flip_reduction"] = agg_stats(fr)
        out[sname] = s
    return out


def aggregate_patching_v2(per_pair: list[dict], direction: str) -> dict:
    """Aggregation for patching per-pair records. direction: 'cf2orig' or 'orig2cf'."""
    if not per_pair:
        return {}
    if direction == "cf2orig":
        metrics = ["recovery_to_orig_logit", "recovery_to_orig_margin",
                   "recovery_to_orig_prob", "shift_toward_orig_logit",
                   "shift_toward_orig_margin"]
    else:
        metrics = ["injection_to_cf_logit", "injection_to_cf_margin",
                   "injection_to_cf_prob", "shift_toward_cf_logit",
                   "shift_toward_cf_margin"]

    strata = {
        "all":               per_pair,
        "orig_correct":      [r for r in per_pair if r.get("orig_correct")],
        "significant":       [r for r in per_pair if r.get("is_significant")],
        "correct_sig":       [r for r in per_pair if r.get("orig_correct")
                               and r.get("is_significant")],
        "answer_flip_pairs": [r for r in per_pair
                               if r.get("pred_orig") != r.get("pred_cf")],
    }
    out = {}
    for sname, rows in strata.items():
        if not rows:
            out[sname] = {"n": 0}
            continue
        s = {"n": len(rows)}
        for m in metrics:
            s[m] = agg_stats([r.get(m) for r in rows])
        for bm in ["pred_patched_matches_orig", "pred_patched_matches_cf",
                   "overshoot_logit", "no_effect_logit",
                   "same_direction_after_intervention",
                   "attenuation_without_reversal",
                   "reversal_after_intervention",
                   "amplification_without_reversal"]:
            s[bm + "_rate"] = mean_bool_rate([r.get(bm) for r in rows])
        out[sname] = s
    return out


def aggregate_layer_results(per_pair_records: list[dict], layer_range,
                             primary_metric: str = "recovery_to_orig_logit") -> dict:
    """
    Per-layer aggregation across pairs for component patching experiments.
    Collects `primary_metric` values per layer from significant pairs.
    Returns {str(layer): {n_valid, mean, std, frac_positive, median, p25, p75}}.
    """
    layer_vals: dict[int, list] = {l: [] for l in layer_range}
    for rec in per_pair_records:
        if not rec.get("is_significant"):
            continue
        for l in layer_range:
            v = rec.get("layers", {}).get(l, {}).get(primary_metric)
            if v is not None:
                layer_vals[l].append(v)
    result = {}
    for l in layer_range:
        stats = agg_stats(layer_vals[l])
        stats["layer"] = l
        stats["primary_metric"] = primary_metric
        result[str(l)] = stats
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _js(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _js(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_js(v) for v in obj]
    return obj


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(_js(obj), f, indent=2)
    print(f"  Saved: {path.name}")


def save_csv(records: list[dict], path: Path):
    """Save per-pair records as CSV. Priority columns appear first."""
    if not records:
        return
    all_keys = list(records[0].keys())
    ordered = [k for k in _CSV_PRIORITY_COLS if k in all_keys] + \
              [k for k in all_keys if k not in _CSV_PRIORITY_COLS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            flat = {k: (str(v) if isinstance(v, (dict, list)) else v)
                    for k, v in rec.items()}
            writer.writerow(flat)
    print(f"  Saved: {path.name}  ({len(records)} rows)")


def make_exp_dir(output_dir: Path, family: str, experiment: str) -> Path:
    d = output_dir / family / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_experiment(exp_dir: Path, metadata: dict, aggregate: dict,
                    per_pair: list | None = None,
                    per_head: list | None = None,
                    per_layer: list | None = None,
                    curve: list | None = None):
    # Inject standard estimand fields
    metadata.setdefault("primary_estimand", "behavioral_recovery")
    metadata.setdefault("recovery_is_not_causal_share", True)
    metadata.setdefault("recovery_interpretation",
                        "Behavioral restoration toward original behavior under intervention; "
                        "not a linear decomposition of causal contribution.")
    save_json(metadata, exp_dir / "metadata.json")
    save_json(aggregate, exp_dir / "aggregate.json")
    if per_pair is not None:
        save_csv(per_pair, exp_dir / "per_pair.csv")
    if per_head is not None:
        save_csv(per_head, exp_dir / "per_head.csv")
    if per_layer is not None:
        save_csv(per_layer, exp_dir / "per_layer.csv")
    if curve is not None:
        save_csv(curve, exp_dir / "curve.csv")


# ---------------------------------------------------------------------------
# Core ablation on a single pair
# ---------------------------------------------------------------------------

def run_ablation_pair(model, tok, pair: dict, answer_ids: dict,
                       heads_by_layer: dict[int, list[int]],
                       head_dim: int, device: str,
                       min_effect: float = MIN_EFFECT_THRESHOLD,
                       mean_activations: dict | None = None,
                       scope: str = "final_token") -> dict:
    """
    Run original, counterfactual, mean-ablated CF (primary), and zero-ablated CF
    (secondary) forward passes.  Returns comprehensive per-pair metrics.

    Primary ablation = mean activation (avoids OOD).
    Secondary ablation = zero activation (destructive; retained for comparison).

    Recovery metrics are computed from the PRIMARY (mean) ablation.
    Destructive causal fractions are computed from the SECONDARY (zero) ablation.
    """
    orig_in = tokenize(tok, pair["orig_prompt"], device)
    cf_in   = tokenize(tok, pair["cf_prompt"],   device)

    bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
    bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])

    # ---- Baseline deltas ----
    signed_delta_logit  = bm_cf["gold_logit"] - bm_orig["gold_logit"]
    signed_delta_prob   = bm_cf["gold_prob"]  - bm_orig["gold_prob"]
    signed_delta_margin = bm_cf["margin"]     - bm_orig["margin"]
    answer_flip      = bm_orig["pred"] != bm_cf["pred"]
    correctness_flip = bm_orig["correct"] != bm_cf["correct"]
    is_significant   = abs(signed_delta_logit) >= min_effect

    # ---- PRIMARY: Mean ablation ----
    if mean_activations:
        mean_hooks = [
            (model.model.layers[l].self_attn.o_proj,
             mean_ablation_pre_hook(hs, head_dim, mean_activations, l,
                                    qid=str(pair["qid"]), scope=scope), True)
            for l, hs in heads_by_layer.items()
        ]
    else:
        # If no mean computed, fall back to zero (with a caveat in records)
        mean_hooks = [
            (model.model.layers[l].self_attn.o_proj,
             ablation_pre_hook(hs, head_dim), True)
            for l, hs in heads_by_layer.items()
        ]
    bm_abl = behavioral_metrics(
        forward_pass(model, cf_in, answer_ids, extra_hooks=mean_hooks), pair["gold"]
    )

    # ---- SECONDARY: Zero ablation (only run separately if mean was available) ----
    if mean_activations:
        zero_hooks = [
            (model.model.layers[l].self_attn.o_proj,
             ablation_pre_hook(hs, head_dim), True)
            for l, hs in heads_by_layer.items()
        ]
        bm_zero = behavioral_metrics(
            forward_pass(model, cf_in, answer_ids, extra_hooks=zero_hooks), pair["gold"]
        )
    else:
        bm_zero = bm_abl  # same run used for both when no mean available

    # ---- Recovery metrics (from primary / mean ablation) ----
    abl_delta_logit  = bm_abl["gold_logit"] - bm_orig["gold_logit"]
    abl_delta_margin = bm_abl["margin"]     - bm_orig["margin"]
    abl_delta_prob   = bm_abl["gold_prob"]  - bm_orig["gold_prob"]

    rec_logit  = recovery_score(signed_delta_logit,  abl_delta_logit,  min_effect)
    rec_margin = recovery_score(signed_delta_margin, abl_delta_margin,
                                min_effect) if abs(signed_delta_margin) >= min_effect else None
    # Exploratory: prob recovery is not gated by the logit-scale min_effect.
    rec_prob   = recovery_score(signed_delta_prob,   abl_delta_prob, PROB_EFFECT_EPS)

    abl_flip = bm_orig["pred"] != bm_abl["pred"]
    flip_reduction = int(answer_flip) - int(abl_flip)

    # ---- Destructive metrics (from secondary / zero ablation) ----
    zero_delta_logit  = bm_zero["gold_logit"] - bm_orig["gold_logit"]
    zero_delta_margin = bm_zero["margin"]     - bm_orig["margin"]
    dest_cf_logit  = causal_fraction(abs(signed_delta_logit),  abs(zero_delta_logit),
                                     min_effect)
    dest_cf_margin = causal_fraction(abs(signed_delta_margin), abs(zero_delta_margin),
                                     min_effect)

    # Sign diagnostics
    baseline_sign  = _delta_sign(signed_delta_logit)
    post_sign      = _delta_sign(abl_delta_logit)
    sign_changed   = baseline_sign != post_sign and baseline_sign != "zero"

    # Pair-level recovery diagnostics
    overshoot_logit  = rec_logit  is not None and rec_logit  > 1.0
    overshoot_margin = rec_margin is not None and rec_margin > 1.0
    overshoot_prob   = rec_prob   is not None and rec_prob   > 1.0
    no_effect_logit  = rec_logit  is not None and abs(rec_logit)  < 0.05
    no_effect_margin = rec_margin is not None and abs(rec_margin) < 0.05
    no_effect_prob   = rec_prob   is not None and abs(rec_prob)   < 0.05
    same_dir = baseline_sign == post_sign
    atten_no_rev = (not sign_changed) and (abs(abl_delta_logit) < abs(signed_delta_logit))
    amp_no_rev   = (not sign_changed) and (abs(abl_delta_logit) > abs(signed_delta_logit))

    return {
        "qid":     pair["qid"],
        "family":  pair["family"],
        "itype":   pair["itype"],
        "attr_val": pair["attr_val"],
        "gold":    pair["gold"],
        # Stratum flags
        "orig_correct":      bool(bm_orig["correct"]),
        "cf_correct":        bool(bm_cf["correct"]),
        "is_significant":    bool(is_significant),
        "answer_flip":       bool(answer_flip),
        "correctness_flip":  bool(correctness_flip),
        # Behavioral baseline
        "orig_gold_logit":   float(bm_orig["gold_logit"]),
        "orig_gold_prob":    float(bm_orig["gold_prob"]),
        "orig_margin":       float(bm_orig["margin"]),
        "orig_pred":         bm_orig["pred"],
        "cf_gold_logit":     float(bm_cf["gold_logit"]),
        "cf_gold_prob":      float(bm_cf["gold_prob"]),
        "cf_margin":         float(bm_cf["margin"]),
        "cf_pred":           bm_cf["pred"],
        "signed_delta_gold_logit":  float(signed_delta_logit),
        "abs_delta_gold_logit":     float(abs(signed_delta_logit)),
        "signed_delta_gold_prob":   float(signed_delta_prob),
        "abs_delta_gold_prob":      float(abs(signed_delta_prob)),
        "signed_delta_margin":      float(signed_delta_margin),
        "abs_delta_margin":         float(abs(signed_delta_margin)),
        # Primary ablation quantities (mean ablation)
        "abl_gold_logit":    float(bm_abl["gold_logit"]),
        "abl_gold_prob":     float(bm_abl["gold_prob"]),
        "abl_margin":        float(bm_abl["margin"]),
        "abl_pred":          bm_abl["pred"],
        "abl_correct":       bool(bm_abl["correct"]),
        "abl_flip":          bool(abl_flip),
        # PRIMARY recovery metrics
        "recovery_to_orig_logit":              rec_logit,
        "recovery_to_orig_margin":             rec_margin,
        "recovery_to_orig_prob":               rec_prob,
        "signed_movement_toward_orig_logit":   float(signed_delta_logit - abl_delta_logit),
        "signed_movement_toward_orig_margin":  float(signed_delta_margin - abl_delta_margin),
        "pred_matches_orig":   bool(bm_abl["pred"] == bm_orig["pred"]),
        "pred_matches_cf":     bool(bm_abl["pred"] == bm_cf["pred"]),
        "flip_reduction":      int(flip_reduction),
        # SECONDARY destructive metrics (zero ablation)
        "zero_abl_gold_logit": float(bm_zero["gold_logit"]) if mean_activations else None,
        "destructive_causal_fraction_logit":  dest_cf_logit,
        "destructive_causal_fraction_margin": dest_cf_margin,
        "mean_ablation_used": bool(mean_activations is not None),
        # Sign diagnostics
        "baseline_delta_sign":          baseline_sign,
        "post_intervention_delta_sign": post_sign,
        "delta_sign_changed":           bool(sign_changed),
        # Pair-level recovery diagnostics
        "overshoot_logit":                  bool(overshoot_logit),
        "overshoot_margin":                 bool(overshoot_margin),
        "overshoot_prob":                   bool(overshoot_prob),
        "no_effect_logit":                  bool(no_effect_logit),
        "no_effect_margin":                 bool(no_effect_margin),
        "no_effect_prob":                   bool(no_effect_prob),
        "same_direction_after_intervention": bool(same_dir),
        "attenuation_without_reversal":     bool(atten_no_rev),
        "reversal_after_intervention":      bool(sign_changed),
        "amplification_without_reversal":   bool(amp_no_rev),
        "primary_estimand":                 "behavioral_recovery",
    }


# ---------------------------------------------------------------------------
# Patching on a single pair
# ---------------------------------------------------------------------------

def run_patching_pair(model, tok, pair: dict, answer_ids: dict,
                       heads_by_layer: dict[int, list[int]],
                       head_dim: int, device: str, direction: str,
                       min_effect: float = MIN_EFFECT_THRESHOLD,
                       scope: str = "final_token") -> dict:
    """
    direction "cf2orig": run CF prompt with original's head activations.
        Primary metric: recovery_to_orig_logit
        Interpretation: does restoring original context remove the demographic shift?

    direction "orig2cf": run orig prompt with CF's head activations.
        Primary metric: injection_to_cf_logit
        Interpretation: does injecting CF context induce the demographic shift?
    """
    orig_in = tokenize(tok, pair["orig_prompt"], device)
    cf_in   = tokenize(tok, pair["cf_prompt"],   device)

    bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
    bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])
    delta_base_logit  = bm_cf["gold_logit"] - bm_orig["gold_logit"]
    delta_base_margin = bm_cf["margin"]     - bm_orig["margin"]
    delta_base_prob   = bm_cf["gold_prob"]  - bm_orig["gold_prob"]
    is_significant    = abs(delta_base_logit) >= min_effect

    # Save activations from source run
    source_store: dict = {}
    save_hooks = [
        (model.model.layers[l].self_attn.o_proj, save_pre_hook(source_store, l, scope=scope), True)
        for l in heads_by_layer
    ]
    if direction == "cf2orig":
        forward_pass(model, orig_in, answer_ids, extra_hooks=save_hooks)
        run_in = cf_in
    else:
        forward_pass(model, cf_in, answer_ids, extra_hooks=save_hooks)
        run_in = orig_in

    patch_hooks = [
        (model.model.layers[l].self_attn.o_proj,
         patch_pre_hook(source_store, l, hs, head_dim, scope=scope), True)
        for l, hs in heads_by_layer.items()
    ]
    bm_pat = behavioral_metrics(
        forward_pass(model, run_in, answer_ids, extra_hooks=patch_hooks), pair["gold"]
    )

    pat_delta_logit  = bm_pat["gold_logit"] - bm_orig["gold_logit"]
    pat_delta_margin = bm_pat["margin"]     - bm_orig["margin"]
    pat_delta_prob   = bm_pat["gold_prob"]  - bm_orig["gold_prob"]

    # Sign diagnostics
    baseline_sign  = _delta_sign(delta_base_logit)
    post_sign      = _delta_sign(pat_delta_logit)
    sign_changed   = baseline_sign != post_sign and baseline_sign != "zero"

    # Recovery (cf2orig) — movement back toward original
    rec_logit  = recovery_score(delta_base_logit,  pat_delta_logit,  min_effect)
    rec_margin = recovery_score(delta_base_margin, pat_delta_margin, min_effect) \
                 if abs(delta_base_margin) >= min_effect else None
    # Exploratory: prob recovery is not gated by the logit-scale min_effect.
    rec_prob   = recovery_score(delta_base_prob,   pat_delta_prob,   PROB_EFFECT_EPS)

    # Pair-level recovery diagnostics
    overshoot_logit = rec_logit  is not None and rec_logit  > 1.0
    no_effect_logit = rec_logit  is not None and abs(rec_logit) < 0.05
    same_dir = baseline_sign == post_sign
    atten_no_rev = (not sign_changed) and (abs(pat_delta_logit) < abs(delta_base_logit))
    amp_no_rev   = (not sign_changed) and (abs(pat_delta_logit) > abs(delta_base_logit))

    # Injection (orig2cf) — movement toward counterfactual
    inj_logit = (pat_delta_logit  / (delta_base_logit  + 1e-8)
                 if is_significant else None)
    inj_margin = (pat_delta_margin / (delta_base_margin + 1e-8)
                  if abs(delta_base_margin) >= min_effect else None)
    inj_prob  = (pat_delta_prob   / (delta_base_prob   + 1e-8)
                 if abs(delta_base_prob) >= PROB_EFFECT_EPS else None)

    return {
        "qid":           pair["qid"],
        "gold":          pair["gold"],
        "direction":     direction,
        "orig_correct":  bool(bm_orig["correct"]),
        "cf_correct":    bool(bm_cf["correct"]),
        "answer_flip":   bool(bm_orig["pred"] != bm_cf["pred"]),
        "correctness_flip": bool(bm_orig["correct"] != bm_cf["correct"]),
        "is_significant": bool(is_significant),
        # Baseline
        "orig_gold_logit": float(bm_orig["gold_logit"]),
        "cf_gold_logit":   float(bm_cf["gold_logit"]),
        "delta_base_logit": float(delta_base_logit),
        # Patched result
        "gold_logit_patched":  float(bm_pat["gold_logit"]),
        "pred_orig":           bm_orig["pred"],
        "pred_cf":             bm_cf["pred"],
        "pred_patched":        bm_pat["pred"],
        "pred_patched_matches_orig": bool(bm_pat["pred"] == bm_orig["pred"]),
        "pred_patched_matches_cf":   bool(bm_pat["pred"] == bm_cf["pred"]),
        # cf2orig: PRIMARY recovery metrics
        "recovery_to_orig_logit":  rec_logit,
        "recovery_to_orig_margin": rec_margin,
        "recovery_to_orig_prob":   rec_prob,
        "shift_toward_orig_logit":  float(delta_base_logit  - pat_delta_logit),
        "shift_toward_orig_margin": float(delta_base_margin - pat_delta_margin),
        # orig2cf: PRIMARY injection metrics
        "injection_to_cf_logit":  float(np.clip(inj_logit,  -3.0, 3.0)) if inj_logit  is not None else None,
        "injection_to_cf_margin": float(np.clip(inj_margin, -3.0, 3.0)) if inj_margin is not None else None,
        "injection_to_cf_prob":   float(np.clip(inj_prob,   -3.0, 3.0)) if inj_prob   is not None else None,
        "shift_toward_cf_logit":  float(pat_delta_logit),
        "shift_toward_cf_margin": float(pat_delta_margin),
        # Sign diagnostics
        "baseline_delta_sign":          baseline_sign,
        "post_intervention_delta_sign": post_sign,
        "delta_sign_changed":           bool(sign_changed),
        # Pair-level recovery diagnostics
        "overshoot_logit":                  bool(overshoot_logit),
        "no_effect_logit":                  bool(no_effect_logit),
        "same_direction_after_intervention": bool(same_dir),
        "attenuation_without_reversal":     bool(atten_no_rev),
        "reversal_after_intervention":      bool(sign_changed),
        "amplification_without_reversal":   bool(amp_no_rev),
        "primary_estimand":                 "behavioral_recovery",
    }


# ---------------------------------------------------------------------------
# Residual patching on a single pair
# ---------------------------------------------------------------------------

def run_residual_patch_pair(model, tok, pair: dict, answer_ids: dict,
                              layer_range, device: str,
                              min_effect: float = MIN_EFFECT_THRESHOLD) -> dict:
    orig_in = tokenize(tok, pair["orig_prompt"], device)
    cf_in   = tokenize(tok, pair["cf_prompt"],   device)

    bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
    bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])
    delta_base_logit  = bm_cf["gold_logit"] - bm_orig["gold_logit"]
    delta_base_margin = bm_cf["margin"]     - bm_orig["margin"]
    is_significant    = abs(delta_base_logit) >= min_effect

    orig_residuals: dict = {}
    save_hooks = [
        (model.model.layers[l], save_residual_hook(orig_residuals, l), False)
        for l in layer_range
    ]
    forward_pass(model, orig_in, answer_ids, extra_hooks=save_hooks)

    layer_results = {}
    for l in layer_range:
        patch_hooks = [(model.model.layers[l],
                        residual_patch_hook(orig_residuals, l), False)]
        bm_pat = behavioral_metrics(
            forward_pass(model, cf_in, answer_ids, extra_hooks=patch_hooks), pair["gold"]
        )
        pat_delta_logit  = bm_pat["gold_logit"] - bm_orig["gold_logit"]
        pat_delta_margin = bm_pat["margin"]     - bm_orig["margin"]

        layer_results[l] = {
            # PRIMARY: recovery toward original
            "recovery_to_orig_logit":    recovery_score(delta_base_logit,  pat_delta_logit,  min_effect),
            "recovery_to_orig_margin":   recovery_score(delta_base_margin, pat_delta_margin, min_effect)
                                          if abs(delta_base_margin) >= min_effect else None,
            "remaining_signed_delta_logit":  float(pat_delta_logit),
            "remaining_abs_delta_logit":     float(abs(pat_delta_logit)),
            # SECONDARY: destructive causal fraction (note: this is residual patching, not zero-ablation)
            "residual_causal_fraction":  causal_fraction(abs(delta_base_logit),
                                                          abs(pat_delta_logit), min_effect),
            "is_significant": bool(is_significant),
            # Sign diagnostics
            "baseline_delta_sign":          _delta_sign(delta_base_logit),
            "post_intervention_delta_sign": _delta_sign(pat_delta_logit),
            "delta_sign_changed":           _delta_sign(delta_base_logit) != _delta_sign(pat_delta_logit) and _delta_sign(delta_base_logit) != "zero",
        }

    return {
        "delta_base_logit":  float(delta_base_logit),
        "is_significant":    bool(is_significant),
        "layers": layer_results,
    }


# ---------------------------------------------------------------------------
# Experiment A1: Individual head ablation
# ---------------------------------------------------------------------------

def exp_individual_ablation(model, tok, pairs, answer_ids, candidate_heads,
                              head_dim, device, family, mean_activations,
                              min_effect: float = MIN_EFFECT_THRESHOLD,
                              scope: str = "final_token") -> tuple[list, list]:
    """
    Returns (per_pair_all_heads, per_head_summary).
    per_pair_all_heads: flat list with head_id added to each row.
    per_head_summary: one row per head with aggregated recovery and destructive stats.
    """
    per_pair_all: list = []
    per_head_rows: list = []

    for l, h in candidate_heads:
        hid = f"L{l:02d}H{h:02d}"
        hbl = {l: [h]}
        raw = []
        for i, pair in enumerate(pairs):
            try:
                r = run_ablation_pair(model, tok, pair, answer_ids, hbl, head_dim, device,
                                      min_effect=min_effect,
                                      mean_activations=mean_activations,
                                      scope=scope)
                r["head_id"] = hid
                raw.append(r)
            except Exception as e:
                print(f"      [{hid}] pair {i} error: {e}")
            if (i + 1) % 25 == 0:
                print(f"      [{hid}] {i+1}/{len(pairs)}", end="\r", flush=True)
        print()
        per_pair_all.extend(raw)

        agg = aggregate_v2(raw)
        sig = agg.get("significant", {})
        n_sig = sig.get("n", 0)
        rec  = (sig.get("recovery_to_orig_logit") or {}).get("mean")
        dest = (sig.get("destructive_causal_fraction_logit") or {}).get("mean")
        fr   = (sig.get("flip_reduction") or {}).get("mean")
        pmatch = sig.get("pred_matches_orig_rate")
        print(f"      [{hid}] PRIMARY recovery(sig)={_fmt(rec)}  |  "
              f"SECONDARY destructive_cf(sig)={_fmt(dest)}  "
              f"flip_reduction={_fmt(fr)}  n_sig={n_sig}")

        per_head_rows.append({
            "head_id": hid, "layer": l, "head": h, "family": family,
            "n_pairs": len(raw), "n_significant": n_sig,
            "recovery_to_orig_logit_mean_sig": rec,
            "recovery_to_orig_logit_std_sig":
                (sig.get("recovery_to_orig_logit") or {}).get("std"),
            "recovery_to_orig_logit_frac_positive_sig":
                (sig.get("recovery_to_orig_logit") or {}).get("frac_positive"),
            "destructive_causal_fraction_logit_mean_sig": dest,
            "destructive_causal_fraction_logit_frac_negative_sig":
                (sig.get("destructive_causal_fraction_logit") or {}).get("frac_negative"),
            "flip_reduction_mean": (agg.get("all", {}).get("flip_reduction") or {}).get("mean"),
            "pred_matches_orig_rate_sig": pmatch,
        })

    return per_pair_all, per_head_rows


def _fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"


def _delta_sign(v: float) -> str:
    """Categorise direction of a logit delta for diagnostic output."""
    if v > 1e-6:
        return "positive"
    if v < -1e-6:
        return "negative"
    return "zero"


# ---------------------------------------------------------------------------
# Experiment A2/B1: Stepwise cumulative ablation
# ---------------------------------------------------------------------------

def exp_stepwise_ablation(model, tok, pairs, answer_ids, candidate_heads,
                           head_dim, device, mean_activations,
                           min_effect: float = MIN_EFFECT_THRESHOLD,
                           print_every: int = 5,
                           scope: str = "final_token") -> tuple[list, list]:
    """
    Returns (curve_rows, final_step_pairs).
    curve_rows: one row per k with aggregate stats (for curve.csv).
    final_step_pairs: per-pair records at k=max (for per_pair.csv).
    """
    curve_rows: list = []
    final_step_pairs: list = []
    n_total = len(candidate_heads)

    for k in range(1, n_total + 1):
        active = candidate_heads[:k]
        hbl    = heads_to_layer_dict(active)
        raw    = []
        for pair in pairs:
            try:
                raw.append(run_ablation_pair(model, tok, pair, answer_ids, hbl,
                                             head_dim, device,
                                             min_effect=min_effect,
                                             mean_activations=mean_activations,
                                             scope=scope))
            except Exception:
                pass
        agg = aggregate_v2(raw)
        sig = agg.get("significant", {})
        n_sig   = sig.get("n", 0)
        rec     = (sig.get("recovery_to_orig_logit") or {}).get("mean")
        dest    = (sig.get("destructive_causal_fraction_logit") or {}).get("mean")
        fr      = (agg.get("all", {}).get("flip_reduction") or {}).get("mean")

        curve_rows.append({
            "k": k,
            "heads": ",".join(f"L{l:02d}H{h:02d}" for l, h in active),
            "n_pairs": len(raw),
            "n_significant": n_sig,
            # PRIMARY
            "recovery_to_orig_logit_mean_sig": rec,
            "recovery_to_orig_logit_std_sig":
                (sig.get("recovery_to_orig_logit") or {}).get("std"),
            "recovery_to_orig_logit_frac_positive_sig":
                (sig.get("recovery_to_orig_logit") or {}).get("frac_positive"),
            "recovery_to_orig_margin_mean_sig":
                (sig.get("recovery_to_orig_margin") or {}).get("mean"),
            "pred_matches_orig_rate_sig": sig.get("pred_matches_orig_rate"),
            "flip_reduction_mean": fr,
            # SECONDARY
            "destructive_causal_fraction_logit_mean_sig": dest,
            "destructive_causal_fraction_logit_frac_negative_sig":
                (sig.get("destructive_causal_fraction_logit") or {}).get("frac_negative"),
        })

        if k == n_total:
            final_step_pairs = raw

        if k % print_every == 0 or k == n_total:
            print(f"      k={k}: PRIMARY recovery(sig)={_fmt(rec)}  |  "
                  f"SECONDARY destructive_cf(sig)={_fmt(dest)}  "
                  f"flip_reduction={_fmt(fr)}  n_sig={n_sig}", flush=True)

    return curve_rows, final_step_pairs


# ---------------------------------------------------------------------------
# Experiment A3: Layer-group comparison
# ---------------------------------------------------------------------------

def exp_layer_group_ablation(model, tok, pairs, answer_ids,
                              top_heads_data, family, layer_groups,
                              head_dim, device, n_per_group, mean_activations,
                              min_effect: float = MIN_EFFECT_THRESHOLD,
                              scope: str = "final_token") -> tuple[dict, list]:
    """
    Returns (group_agg_dict, per_pair_all).
    group_agg_dict: {group_name: aggregate_v2 result}
    per_pair_all: flat list with group label added.
    """
    group_agg: dict = {}
    per_pair_all: list = []

    for gname, grange in layer_groups.items():
        gh = filter_heads_by_layer(top_heads_data, family, grange, n_per_group)
        if not gh:
            print(f"    {gname}: no heads in range")
            group_agg[gname] = {"n": 0, "msg": "no heads"}
            continue
        hbl = heads_to_layer_dict(gh)
        raw = []
        for pair in pairs:
            try:
                r = run_ablation_pair(model, tok, pair, answer_ids, hbl,
                                      head_dim, device,
                                      min_effect=min_effect,
                                      mean_activations=mean_activations,
                                      scope=scope)
                r["layer_group"] = gname
                raw.append(r)
            except Exception:
                pass
        agg  = aggregate_v2(raw)
        sig  = agg.get("significant", {})
        rec  = (sig.get("recovery_to_orig_logit") or {}).get("mean")
        dest = (sig.get("destructive_causal_fraction_logit") or {}).get("mean")
        print(f"    {gname}: n_heads={len(gh)}  "
              f"PRIMARY recovery(sig)={_fmt(rec)}  |  "
              f"SECONDARY destructive_cf(sig)={_fmt(dest)}")
        group_agg[gname] = agg
        per_pair_all.extend(raw)

    return group_agg, per_pair_all


# ---------------------------------------------------------------------------
# Experiment A4 / patching: Direction patching
# ---------------------------------------------------------------------------

def exp_direction_patching(model, tok, pairs, answer_ids, heads_by_layer,
                            head_dim, device,
                            min_effect: float = MIN_EFFECT_THRESHOLD,
                            scope: str = "final_token") -> dict:
    cf2orig, orig2cf = [], []
    for i, pair in enumerate(pairs):
        try:
            cf2orig.append(run_patching_pair(model, tok, pair, answer_ids,
                                             heads_by_layer, head_dim, device,
                                             "cf2orig", min_effect=min_effect,
                                             scope=scope))
        except Exception as e:
            print(f"      cf2orig pair {i}: {e}")
        try:
            orig2cf.append(run_patching_pair(model, tok, pair, answer_ids,
                                             heads_by_layer, head_dim, device,
                                             "orig2cf", min_effect=min_effect,
                                             scope=scope))
        except Exception as e:
            print(f"      orig2cf pair {i}: {e}")
        if (i + 1) % 25 == 0:
            print(f"      {i+1}/{len(pairs)}", end="\r", flush=True)
    print()

    agg_c2o = aggregate_patching_v2(cf2orig, "cf2orig")
    agg_o2c = aggregate_patching_v2(orig2cf, "orig2cf")

    # Baseline diagnostics — print before interpreting recovery/injection numbers
    n_total  = len(cf2orig)
    n_sig    = sum(1 for r in cf2orig if r.get("is_significant"))
    n_flip   = sum(1 for r in cf2orig if r.get("pred_orig") != r.get("pred_cf"))
    n_corr_flip = sum(1 for r in cf2orig if r.get("correctness_flip"))
    frac_already_match = (sum(1 for r in cf2orig if r.get("pred_orig") == r.get("pred_cf"))
                          / n_total if n_total else 0.0)
    print(f"    BASELINE n={n_total}  n_sig={n_sig}  "
          f"answer_flip_rate={n_flip/n_total:.3f}  "
          f"correctness_flip_rate={n_corr_flip/n_total:.3f}  "
          f"frac_pred_already_match={frac_already_match:.3f}")

    sig_c2o = agg_c2o.get("significant", {})
    sig_o2c = agg_o2c.get("significant", {})
    afp_c2o = agg_c2o.get("answer_flip_pairs", {})
    afp_o2c = agg_o2c.get("answer_flip_pairs", {})
    print(f"    cf2orig  recovery(sig)={_fmt((sig_c2o.get('recovery_to_orig_logit') or {}).get('mean'))}  "
          f"recovery(flip_pairs)={_fmt((afp_c2o.get('recovery_to_orig_logit') or {}).get('mean'))}  "
          f"pred_matches_orig(sig)={_fmt(sig_c2o.get('pred_patched_matches_orig_rate'))}")
    print(f"    orig2cf  injection(sig)={_fmt((sig_o2c.get('injection_to_cf_logit') or {}).get('mean'))}  "
          f"injection(flip_pairs)={_fmt((afp_o2c.get('injection_to_cf_logit') or {}).get('mean'))}  "
          f"pred_matches_cf(sig)={_fmt(sig_o2c.get('pred_patched_matches_cf_rate'))}")

    return {
        "cf2orig": {"agg": agg_c2o, "pairs": cf2orig},
        "orig2cf": {"agg": agg_o2c, "pairs": orig2cf},
    }


# ---------------------------------------------------------------------------
# Experiment B2: Residual stream patching
# ---------------------------------------------------------------------------

def exp_residual_patching(model, tok, pairs, answer_ids, layer_range, device,
                           min_effect: float = MIN_EFFECT_THRESHOLD) -> dict:
    """
    Returns {"layer_agg": per-layer aggregated stats, "per_pair": flat list,
             "per_layer_rows": rows for per_layer.csv}.
    Primary metric: recovery_to_orig_logit.
    Interprets carrier layers (where signal is transported), not encoding layers.
    """
    layer_recoveries:  dict[int, list] = {l: [] for l in layer_range}
    layer_dest_fracs:  dict[int, list] = {l: [] for l in layer_range}
    per_pair = []

    for i, pair in enumerate(pairs):
        try:
            r = run_residual_patch_pair(model, tok, pair, answer_ids,
                                         layer_range, device, min_effect=min_effect)
            row = {
                "qid": pair["qid"],
                "delta_base_logit": r["delta_base_logit"],
                "is_significant":   r["is_significant"],
            }
            for l in layer_range:
                lr = r["layers"][l]
                row[f"L{l:02d}_recovery_to_orig"] = lr["recovery_to_orig_logit"]
                row[f"L{l:02d}_residual_cf"]       = lr["residual_causal_fraction"]
                if lr["recovery_to_orig_logit"] is not None:
                    layer_recoveries[l].append(lr["recovery_to_orig_logit"])
                if lr["residual_causal_fraction"] is not None:
                    layer_dest_fracs[l].append(lr["residual_causal_fraction"])
            row["primary_estimand"] = "behavioral_recovery"
            per_pair.append(row)
        except Exception as e:
            print(f"      Pair {i}: {e}")
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(pairs)}", end="\r", flush=True)
    print()

    per_layer_rows = []
    layer_agg = {}
    for l in layer_range:
        rec_vals  = layer_recoveries[l]
        dest_vals = layer_dest_fracs[l]
        agg = {
            "layer": l,
            "n_valid": len(rec_vals),
            # PRIMARY
            "recovery_to_orig_logit_mean": float(np.mean(rec_vals))   if rec_vals  else None,
            "recovery_to_orig_logit_std":  float(np.std(rec_vals))    if rec_vals  else None,
            "recovery_frac_positive":      float(np.mean(np.array(rec_vals) > 0)) if rec_vals else None,
            # SECONDARY
            "residual_causal_fraction_mean": float(np.mean(dest_vals)) if dest_vals else None,
            "residual_causal_fraction_std":  float(np.std(dest_vals))  if dest_vals else None,
        }
        layer_agg[str(l)] = agg
        per_layer_rows.append(agg)

    return {"layer_agg": layer_agg, "per_pair": per_pair, "per_layer_rows": per_layer_rows}


# ---------------------------------------------------------------------------
# Experiment B3: Context split
# ---------------------------------------------------------------------------

def exp_context_split(model, tok, all_pairs, answer_ids, heads_by_layer,
                       head_dim, device, max_n, rng, global_warnings,
                       min_effect: float = MIN_EFFECT_THRESHOLD,
                       scope: str = "final_token",
                       mean_activations: dict | None = None) -> dict:
    so_pairs = [p for p in all_pairs if p["family"] == "sexual_orientation"]

    split_counts: Counter = Counter()
    splits: dict[str, list] = defaultdict(list)
    for p in so_pairs:
        s = orientation_split_label(p["attr_norm"])
        split_counts[s] += 1
        splits[s].append(p)

    total_so = len(so_pairs)
    other_frac = split_counts.get("other", 0) / total_so if total_so else 0.0
    print(f"    Orientation split (raw counts): {dict(split_counts)}")
    if other_frac > 0.10:
        w = (f">{other_frac:.0%} of sexual-orientation pairs assigned to 'other' "
             f"— check PARTNER_PATTERNS / EXPLICIT_PATTERNS.")
        print(f"    Warning: {w}")
        global_warnings.append(w)

    results = {}
    for split_name in ["partner", "explicit"]:
        subset = splits[split_name]
        print(f"    {split_name}: {len(subset)} raw → ", end="")
        if not subset:
            print("0 (no pairs)")
            results[split_name] = {"n": 0, "msg": "no pairs",
                                   "control_label": "reference controls"}
            continue
        if len(subset) > max_n:
            idx = rng.choice(len(subset), max_n, replace=False)
            subset = [subset[i] for i in sorted(idx)]
        print(f"{len(subset)} sampled")

        raw = []
        for pair in subset:
            try:
                raw.append(run_ablation_pair(model, tok, pair, answer_ids,
                                             heads_by_layer, head_dim, device,
                                             min_effect=min_effect,
                                             mean_activations=mean_activations,
                                             scope=scope))
            except Exception:
                pass
        agg  = aggregate_v2(raw)
        sig  = agg.get("significant", {})
        rec  = (sig.get("recovery_to_orig_logit") or {}).get("mean")
        dest = (sig.get("destructive_causal_fraction_logit") or {}).get("mean")
        fr   = (agg.get("all", {}).get("flip_reduction") or {}).get("mean")
        print(f"      {split_name} (n={len(raw)}): PRIMARY recovery(sig)={_fmt(rec)}  |  "
              f"SECONDARY destructive_cf(sig)={_fmt(dest)}  flip_reduction={_fmt(fr)}")
        results[split_name] = {"n": len(raw), "agg": agg, "pairs": raw}

    return results


# ---------------------------------------------------------------------------
# Localization summary
# ---------------------------------------------------------------------------

def build_localization_summary(family: str, sw_early, sw_full,
                                early_heads, full_heads,
                                early_layer_max: int) -> dict:
    """
    Builds the per-family localization summary dict for killer_results.
    Uses recovery_to_orig_logit as primary metric, destructive fraction as secondary.
    """
    def _curve_final(sw):
        if not sw:
            return {}
        row = sw[-1] if isinstance(sw[-1], dict) else {}
        return row

    ef = _curve_final(sw_early)
    ff = _curve_final(sw_full)

    rec_early  = ef.get("recovery_to_orig_logit_mean_sig")
    rec_full   = ff.get("recovery_to_orig_logit_mean_sig")
    dest_early = ef.get("destructive_causal_fraction_logit_mean_sig")
    dest_full  = ff.get("destructive_causal_fraction_logit_mean_sig")

    early_share_recovery = (
        rec_early / (rec_full + 1e-8) * 100
        if rec_early is not None and rec_full is not None and rec_full > 0
        else None
    )
    early_share_destructive = (
        dest_early / (dest_full + 1e-8) * 100
        if dest_early is not None and dest_full is not None and dest_full > 0
        else None
    )

    return {
        "family": family,
        # PRIMARY: recovery
        "recovery_early":          rec_early,
        "recovery_full":           rec_full,
        "early_share_recovery":    early_share_recovery,
        # SECONDARY: destructive
        "destructive_cf_early":    dest_early,
        "destructive_cf_full":     dest_full,
        "early_share_destructive": early_share_destructive,
        # Metadata
        "n_early_heads": len(early_heads),
        "n_full_heads":  len(full_heads),
        "early_layer_max": early_layer_max,
        "n_sig_early": ef.get("n_significant"),
        "n_sig_full":  ff.get("n_significant"),
    }


# ---------------------------------------------------------------------------
# C1–C3: Unified layer-patching (residual / MLP / attention)
# ---------------------------------------------------------------------------

# Maps component name → (save_hook_factory, patch_hook_factory, is_pre_hook, module_attr)
# module_attr is an attribute path on model.model.layers[l], or None for the layer itself.
_COMPONENT_HOOKS = {
    "residual":  (save_residual_hook,  residual_patch_hook,  False, None),
    "mlp":       (save_mlp_out_hook,   patch_mlp_out_hook,   False, "mlp"),
    "attention": (save_attn_out_hook,  patch_attn_out_hook,  False, "self_attn"),
}


def _get_layer_module(model, layer_idx: int, attr: str | None):
    layer = model.model.layers[layer_idx]
    return getattr(layer, attr) if attr else layer


def run_layer_patch_pair(model, tok, pair: dict, answer_ids: dict,
                          layer_range, device: str,
                          component: str,   # "residual", "mlp", "attention"
                          direction: str,   # "cf2orig" or "orig2cf"
                          min_effect: float = MIN_EFFECT_THRESHOLD,
                          intervention_scope: str = "final_token") -> dict:
    """
    For each layer in layer_range, patch the specified component from the
    source run into the target run and measure recovery (cf2orig) or
    injection (orig2cf).

    Efficiency: one source-save pass captures all layers simultaneously,
    then one patching pass per layer.

    Returns:
        {"delta_base_logit": float, "is_significant": bool,
         "layers": {layer_idx: per_layer_dict}}
    """
    save_factory, patch_factory, is_pre, mod_attr = _COMPONENT_HOOKS[component]
    orig_in = tokenize(tok, pair["orig_prompt"], device)
    cf_in   = tokenize(tok, pair["cf_prompt"],   device)

    bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
    bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])
    delta_base_logit  = bm_cf["gold_logit"] - bm_orig["gold_logit"]
    delta_base_margin = bm_cf["margin"]     - bm_orig["margin"]
    is_significant    = abs(delta_base_logit) >= min_effect

    # Source-save pass: capture component output at all layers simultaneously
    source_store: dict = {}
    layers = list(layer_range)
    if component == "residual":
        # save_residual_hook is a post-hook on the full layer
        save_hooks = [
            (model.model.layers[l], save_factory(source_store, l), False)
            for l in layers
        ]
    else:
        save_hooks = [
            (getattr(model.model.layers[l], mod_attr), save_factory(source_store, l), False)
            for l in layers
        ]
    source_in = orig_in if direction == "cf2orig" else cf_in
    forward_pass(model, source_in, answer_ids, extra_hooks=save_hooks)

    target_in = cf_in if direction == "cf2orig" else orig_in

    layer_results = {}
    for l in layers:
        mod = _get_layer_module(model, l, mod_attr)
        if component == "residual":
            patch_hooks = [(mod, patch_factory(source_store, l), is_pre)]
        else:
            patch_hooks = [(mod, patch_factory(source_store, l, intervention_scope), is_pre)]
        bm_pat = behavioral_metrics(
            forward_pass(model, target_in, answer_ids, extra_hooks=patch_hooks),
            pair["gold"]
        )
        pat_delta_logit  = bm_pat["gold_logit"] - bm_orig["gold_logit"]
        pat_delta_margin = bm_pat["margin"]     - bm_orig["margin"]

        if direction == "cf2orig":
            primary = recovery_score(delta_base_logit,  pat_delta_logit,  min_effect)
            secondary = recovery_score(delta_base_margin, pat_delta_margin, min_effect) \
                        if abs(delta_base_margin) >= min_effect else None
        else:
            # orig2cf injection: how much of CF delta was induced?
            primary = (float(np.clip(pat_delta_logit / (delta_base_logit + 1e-8), -3.0, 3.0))
                       if is_significant else None)
            secondary = (float(np.clip(pat_delta_margin / (delta_base_margin + 1e-8), -3.0, 3.0))
                         if abs(delta_base_margin) >= min_effect else None)

        layer_results[l] = {
            "recovery_to_orig_logit" if direction == "cf2orig" else "injection_to_cf_logit":
                primary,
            "recovery_to_orig_margin" if direction == "cf2orig" else "injection_to_cf_margin":
                secondary,
            "remaining_signed_delta_logit":  float(pat_delta_logit),
            "pred_matches_orig":  bool(bm_pat["pred"] == bm_orig["pred"]),
            "pred_matches_cf":    bool(bm_pat["pred"] == bm_cf["pred"]),
            "is_significant":     bool(is_significant),
            # Sign diagnostics
            "baseline_delta_sign":          _delta_sign(delta_base_logit),
            "post_intervention_delta_sign": _delta_sign(pat_delta_logit),
            "delta_sign_changed":           _delta_sign(delta_base_logit) != _delta_sign(pat_delta_logit) and _delta_sign(delta_base_logit) != "zero",
            "overshoot_logit": bool(primary is not None and primary > 1.0) if direction == "cf2orig" else None,
            "no_effect_logit": bool(primary is not None and abs(primary) < 0.05) if primary is not None else None,
            "same_direction_after_intervention": bool(_delta_sign(delta_base_logit) == _delta_sign(pat_delta_logit)),
            "attenuation_without_reversal": bool(
                _delta_sign(delta_base_logit) == _delta_sign(pat_delta_logit)
                and abs(pat_delta_logit) < abs(delta_base_logit)
            ),
            "reversal_after_intervention": bool(
                _delta_sign(delta_base_logit) != _delta_sign(pat_delta_logit)
                and _delta_sign(delta_base_logit) != "zero"
            ),
            "amplification_without_reversal": bool(
                _delta_sign(delta_base_logit) == _delta_sign(pat_delta_logit)
                and abs(pat_delta_logit) > abs(delta_base_logit)
            ),
            "primary_estimand": "behavioral_recovery",
        }

    return {"delta_base_logit": float(delta_base_logit),
            "is_significant": bool(is_significant),
            "layers": layer_results}


def exp_component_patching(model, tok, pairs: list, answer_ids: dict,
                            layer_range, device: str,
                            component: str, direction: str,
                            min_effect: float = MIN_EFFECT_THRESHOLD,
                            intervention_scope: str = "final_token") -> dict:
    """
    Run run_layer_patch_pair for each pair and aggregate per layer.
    Returns {"layer_agg": dict, "per_pair": list, "per_layer_rows": list}.

    per_pair: flat list — each record is one pair with its layer-level results
              flattened as L{nn}_{metric} columns (for CSV).
    per_layer_rows: one row per layer with aggregate stats (for per_layer.csv).
    """
    layers = list(layer_range)
    metric_key = "recovery_to_orig_logit" if direction == "cf2orig" else "injection_to_cf_logit"

    layer_vals: dict[int, list] = {l: [] for l in layers}
    per_pair: list = []

    for i, pair in enumerate(pairs):
        try:
            r = run_layer_patch_pair(model, tok, pair, answer_ids, layers, device,
                                     component, direction, min_effect=min_effect,
                                     intervention_scope=intervention_scope)
            flat = {
                "qid":             pair["qid"],
                "family":          pair["family"],
                "component":       component,
                "direction":       direction,
                "intervention_scope": intervention_scope,
                "is_significant":  r["is_significant"],
                "delta_base_logit": r["delta_base_logit"],
                "primary_estimand": "behavioral_recovery",
            }
            for l in layers:
                lr = r["layers"][l]
                flat[f"L{l:02d}_{metric_key}"]   = lr.get(metric_key)
                flat[f"L{l:02d}_remaining_delta"] = lr.get("remaining_signed_delta_logit")
            per_pair.append(flat)
            if r["is_significant"]:
                for l in layers:
                    v = r["layers"][l].get(metric_key)
                    if v is not None:
                        layer_vals[l].append(v)
        except Exception as e:
            print(f"      Pair {i}: {e}")
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(pairs)}", end="\r", flush=True)
    print()

    per_layer_rows = []
    layer_agg = {}
    for l in layers:
        vals = layer_vals[l]
        s = agg_stats(vals)
        s["layer"]       = l
        s["component"]   = component
        s["direction"]   = direction
        layer_agg[str(l)] = s
        per_layer_rows.append(s)

    if layer_vals:
        best_l = max(layer_vals, key=lambda l: np.mean(layer_vals[l]) if layer_vals[l] else -999)
        best_v = float(np.mean(layer_vals[best_l])) if layer_vals[best_l] else None
        print(f"      Peak {metric_key}: layer {best_l} = {_fmt(best_v)}  "
              f"n_sig={sum(1 for p in per_pair if p.get('is_significant'))}")

    return {"layer_agg": layer_agg, "per_pair": per_pair,
            "per_layer_rows": per_layer_rows, "primary_metric": metric_key,
            "intervention_scope": intervention_scope}


# ---------------------------------------------------------------------------
# C4: Within-layer decomposition
# ---------------------------------------------------------------------------

def run_within_layer_decomp_pair(model, tok, pair: dict, answer_ids: dict,
                                  top_layers: list, device: str,
                                  min_effect: float = MIN_EFFECT_THRESHOLD,
                                  intervention_scope: str = "final_token") -> dict:
    """
    For each layer in top_layers, run CF prompt with:
      1. attention output patched from orig  → recovery_attention
      2. MLP output patched from orig        → recovery_mlp
      3. full residual patched from orig     → recovery_residual

    All in cf2orig direction. One source-save pass for all components.
    Returns {"delta_base_logit": float, "is_significant": bool,
             "layers": {l: {"recovery_attention": ..., "recovery_mlp": ...,
                             "recovery_residual": ...}}}
    """
    orig_in = tokenize(tok, pair["orig_prompt"], device)
    cf_in   = tokenize(tok, pair["cf_prompt"],   device)
    bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
    bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])
    delta_base_logit = bm_cf["gold_logit"] - bm_orig["gold_logit"]
    is_significant   = abs(delta_base_logit) >= min_effect

    # Save all three component outputs from orig run simultaneously
    orig_residuals: dict = {}
    orig_attn_outs: dict = {}
    orig_mlp_outs:  dict = {}
    save_hooks = (
        [(model.model.layers[l], save_residual_hook(orig_residuals, l), False)
         for l in top_layers] +
        [(model.model.layers[l].self_attn, save_attn_out_hook(orig_attn_outs, l), False)
         for l in top_layers] +
        [(model.model.layers[l].mlp, save_mlp_out_hook(orig_mlp_outs, l), False)
         for l in top_layers]
    )
    forward_pass(model, orig_in, answer_ids, extra_hooks=save_hooks)

    layer_results = {}
    for l in top_layers:
        rec = {}
        for comp, store, factory in [
            ("attention", orig_attn_outs, patch_attn_out_hook),
            ("mlp",       orig_mlp_outs,  patch_mlp_out_hook),
        ]:
            mod = getattr(model.model.layers[l], "self_attn" if comp == "attention" else "mlp")
            bm_pat = behavioral_metrics(
                forward_pass(model, cf_in, answer_ids,
                             extra_hooks=[(mod, factory(store, l, intervention_scope), False)]),
                pair["gold"]
            )
            pat_delta = bm_pat["gold_logit"] - bm_orig["gold_logit"]
            rec[f"recovery_{comp}"] = recovery_score(delta_base_logit, pat_delta, min_effect)

        # Residual (full layer patch, final token only)
        bm_res = behavioral_metrics(
            forward_pass(model, cf_in, answer_ids,
                         extra_hooks=[(model.model.layers[l],
                                       residual_patch_hook(orig_residuals, l), False)]),
            pair["gold"]
        )
        pat_delta_res = bm_res["gold_logit"] - bm_orig["gold_logit"]
        rec["recovery_residual"] = recovery_score(delta_base_logit, pat_delta_res, min_effect)
        layer_results[l] = rec

    return {"delta_base_logit": float(delta_base_logit),
            "is_significant": bool(is_significant),
            "layers": layer_results}


def exp_within_layer_decomp(model, tok, pairs: list, answer_ids: dict,
                             top_layers: list, device: str,
                             min_effect: float = MIN_EFFECT_THRESHOLD,
                             intervention_scope: str = "final_token") -> dict:
    """
    Runs C4 decomposition for each pair.
    Returns {layer: {attention_mean, mlp_mean, residual_mean, n_valid}} and per_pair records.
    """
    layer_vals: dict = {l: {"attention": [], "mlp": [], "residual": []}
                        for l in top_layers}
    per_pair: list = []

    for i, pair in enumerate(pairs):
        try:
            r = run_within_layer_decomp_pair(model, tok, pair, answer_ids,
                                             top_layers, device, min_effect=min_effect,
                                             intervention_scope=intervention_scope)
            flat = {"qid": pair["qid"], "family": pair["family"],
                    "is_significant": r["is_significant"],
                    "delta_base_logit": r["delta_base_logit"]}
            if r["is_significant"]:
                for l in top_layers:
                    lr = r["layers"][l]
                    flat[f"L{l:02d}_recovery_attention"] = lr.get("recovery_attention")
                    flat[f"L{l:02d}_recovery_mlp"]       = lr.get("recovery_mlp")
                    flat[f"L{l:02d}_recovery_residual"]  = lr.get("recovery_residual")
                    for comp in ["attention", "mlp", "residual"]:
                        v = lr.get(f"recovery_{comp}")
                        if v is not None:
                            layer_vals[l][comp].append(v)
            per_pair.append(flat)
        except Exception as e:
            print(f"      Pair {i}: {e}")
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(pairs)}", end="\r", flush=True)
    print()

    layer_summary = {}
    per_layer_rows = []
    for l in top_layers:
        row = {"layer": l}
        for comp in ["attention", "mlp", "residual"]:
            vals = layer_vals[l][comp]
            row[f"{comp}_n_valid"] = len(vals)
            row[f"{comp}_mean"]    = float(np.mean(vals)) if vals else None
            row[f"{comp}_std"]     = float(np.std(vals))  if vals else None
        n_att = layer_vals[l]["attention"]
        n_mlp = layer_vals[l]["mlp"]
        n_res = layer_vals[l]["residual"]
        print(f"      Layer {l:2d}: "
              f"attention={_fmt(row['attention_mean'])}  "
              f"mlp={_fmt(row['mlp_mean'])}  "
              f"residual={_fmt(row['residual_mean'])}  "
              f"n_valid={row['attention_n_valid']}")
        layer_summary[str(l)] = row
        per_layer_rows.append(row)

    return {"layer_summary": layer_summary, "per_pair": per_pair,
            "per_layer_rows": per_layer_rows, "top_layers": top_layers}


# ---------------------------------------------------------------------------
# C5: Residual direction analysis
# ---------------------------------------------------------------------------

def compute_residual_deltas(model, tok, pairs: list, answer_ids: dict,
                             layer_range, device: str,
                             max_pairs: int = 30) -> list:
    """
    For each pair (up to max_pairs), capture final-token residual at each layer
    for both orig and CF runs. Compute delta = CF - orig per layer.
    Returns list of {qid, is_significant, delta_base_logit,
                     layers: {l: delta_vector (cpu float tensor)}}.
    """
    sample = pairs[:max_pairs]
    layers = list(layer_range)
    results = []

    for pair in sample:
        try:
            orig_in = tokenize(tok, pair["orig_prompt"], device)
            cf_in   = tokenize(tok, pair["cf_prompt"],   device)
            bm_orig = behavioral_metrics(forward_pass(model, orig_in, answer_ids), pair["gold"])
            bm_cf   = behavioral_metrics(forward_pass(model, cf_in,   answer_ids), pair["gold"])
            delta_base = bm_cf["gold_logit"] - bm_orig["gold_logit"]

            orig_res: dict = {}
            cf_res:   dict = {}
            orig_hooks = [(model.model.layers[l], save_residual_hook(orig_res, l), False)
                          for l in layers]
            cf_hooks   = [(model.model.layers[l], save_residual_hook(cf_res,   l), False)
                          for l in layers]
            forward_pass(model, orig_in, answer_ids, extra_hooks=orig_hooks)
            forward_pass(model, cf_in,   answer_ids, extra_hooks=cf_hooks)

            layer_deltas = {}
            for l in layers:
                if l in orig_res and l in cf_res:
                    # final token only; shape: (hidden_size,)
                    delta_vec = (cf_res[l][:, -1, :] - orig_res[l][:, -1, :]).float().cpu().squeeze(0)
                    layer_deltas[l] = delta_vec

            results.append({
                "qid":             pair["qid"],
                "family":          pair["family"],
                "is_significant":  abs(delta_base) >= 0.05,
                "delta_base_logit": float(delta_base),
                "layers":          layer_deltas,
            })
        except Exception as e:
            print(f"      direction pair error: {e}")

    return results


def compute_family_direction(residual_deltas: list, layer_range) -> dict:
    """
    Compute mean delta vector per layer across significant pairs.
    Returns {layer_idx: mean_delta_tensor (hidden_size,)}.
    """
    layers = list(layer_range)
    layer_sums: dict[int, list] = {l: [] for l in layers}
    for rec in residual_deltas:
        if not rec.get("is_significant"):
            continue
        for l in layers:
            v = rec["layers"].get(l)
            if v is not None:
                layer_sums[l].append(v)
    directions = {}
    for l in layers:
        if layer_sums[l]:
            directions[l] = torch.stack(layer_sums[l]).mean(dim=0)
    return directions


def exp_direction_analysis(model, tok, pairs: list, answer_ids: dict,
                            layer_range, device: str,
                            max_pairs: int = 30) -> dict:
    """
    C5: Residual direction analysis.
    Computes per-layer cosine similarity and projection of each pair's delta
    onto the family mean direction.
    Returns {"per_layer_stats": list, "n_sig_pairs": int}.
    Note: family_directions contain raw tensors (not serialised to JSON).
    """
    layers = list(layer_range)
    print(f"      Computing residual deltas ({min(max_pairs, len(pairs))} pairs)...")
    residual_deltas = compute_residual_deltas(model, tok, pairs, answer_ids,
                                              layers, device, max_pairs=max_pairs)
    family_directions = compute_family_direction(residual_deltas, layers)
    n_sig = sum(1 for r in residual_deltas if r.get("is_significant"))
    print(f"      {n_sig} significant pairs for direction analysis.")

    per_layer_stats = []
    for l in layers:
        fam_dir = family_directions.get(l)
        if fam_dir is None:
            per_layer_stats.append({"layer": l, "n_valid": 0})
            continue
        fam_norm = fam_dir / (fam_dir.norm() + 1e-8)
        cos_sims, projs, norms = [], [], []
        for rec in residual_deltas:
            if not rec.get("is_significant"):
                continue
            dv = rec["layers"].get(l)
            if dv is None:
                continue
            dn = dv.norm().item()
            if dn < 1e-8:
                continue
            cos_sims.append(float(torch.dot(dv, fam_norm) / (dv.norm() + 1e-8)))
            projs.append(float(torch.dot(dv, fam_norm)))
            norms.append(dn)

        row = {
            "layer": l,
            "n_valid": len(cos_sims),
            "family_direction_norm": float(fam_dir.norm()),
            "cosine_sim_mean":  float(np.mean(cos_sims))  if cos_sims else None,
            "cosine_sim_std":   float(np.std(cos_sims))   if cos_sims else None,
            "cosine_sim_median": float(np.median(cos_sims)) if cos_sims else None,
            "projection_mean":  float(np.mean(projs))     if projs    else None,
            "projection_std":   float(np.std(projs))      if projs    else None,
            "delta_norm_mean":  float(np.mean(norms))     if norms    else None,
        }
        per_layer_stats.append(row)

    return {"per_layer_stats": per_layer_stats, "n_sig_pairs": n_sig}


# ---------------------------------------------------------------------------
# Warnings helper
# ---------------------------------------------------------------------------

def _warn_if(condition: bool, msg: str, global_warnings: list):
    if condition:
        print(f"  *** Warning: {msg} ***")
        global_warnings.append(msg)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def family_from_key(key: str) -> str:
    for fam in sorted(ALL_FOCAL + ["control"], key=len, reverse=True):
        if key.startswith(fam):
            return fam
    return key.split("_")[0]


def plot_localization_summary(killer: dict, output_path: Path):
    """
    Headline plot: early vs full recovery (PRIMARY) and early share.
    Destructive fraction shown as secondary subplot.
    """
    families = [f for f in killer if killer[f].get("recovery_full") is not None]
    if not families:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(families))
    w = 0.35
    colors = [FAMILY_COLORS.get(f, "grey") for f in families]

    # Panel 1: Recovery (primary)
    ax = axes[0]
    rec_full  = [killer[f].get("recovery_full",  0) or 0 for f in families]
    rec_early = [killer[f].get("recovery_early", 0) or 0 for f in families]
    ax.bar(x - w/2, rec_full,  w, color=colors, alpha=0.9, label="Full heads")
    ax.bar(x + w/2, rec_early, w, color=colors, alpha=0.45, hatch="//",
           label="Early heads only")
    ax.set_xticks(x); ax.set_xticklabels([f.replace("_", "\n") for f in families])
    ax.set_ylabel("Recovery to orig (mean, significant pairs)")
    ax.set_title("PRIMARY: Recovery\n(mean ablation)")
    ax.axhline(0, color="k", linewidth=0.5); ax.legend(fontsize=8)

    # Panel 2: Early share of recovery
    ax2 = axes[1]
    shares = [killer[f].get("early_share_recovery") or 0 for f in families]
    ax2.bar(x, shares, color=colors, alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f.replace("_", "\n") for f in families])
    ax2.set_ylabel("Early / full recovery (%)")
    ax2.set_title("Early-layer share of recovery")
    ax2.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Panel 3: Destructive causal fraction (secondary)
    ax3 = axes[2]
    dest_full  = [killer[f].get("destructive_cf_full",  0) or 0 for f in families]
    dest_early = [killer[f].get("destructive_cf_early", 0) or 0 for f in families]
    ax3.bar(x - w/2, dest_full,  w, color=colors, alpha=0.9, label="Full heads")
    ax3.bar(x + w/2, dest_early, w, color=colors, alpha=0.45, hatch="//",
            label="Early heads only")
    ax3.set_xticks(x); ax3.set_xticklabels([f.replace("_", "\n") for f in families])
    ax3.set_ylabel("Destructive causal fraction")
    ax3.set_title("SECONDARY: Destructive fraction\n(zero-ablation; can be negative)")
    ax3.axhline(0, color="k", linewidth=0.5); ax3.legend(fontsize=8)

    fig.suptitle(
        "Localization of Demographic Sensitivity across Layer Depth\n"
        "Primary: recovery toward original behaviour | Secondary: destructive ablation",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_stepwise_curves(curve_dict: dict, output_path: Path):
    """
    Left: recovery_to_orig_logit (PRIMARY y-axis).
    Right: destructive_causal_fraction_logit (SECONDARY).
    Third: flip_reduction.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    for key, curve in curve_dict.items():
        if not curve:
            continue
        fam = family_from_key(key)
        color = FAMILY_COLORS.get(fam, "grey")
        ls = "--" if key.endswith("early") else ("-." if "control" in key else "-")
        ks   = [r["k"]  for r in curve]
        recs = [r.get("recovery_to_orig_logit_mean_sig") or 0  for r in curve]
        dcs  = [r.get("destructive_causal_fraction_logit_mean_sig") or 0 for r in curve]
        frs  = [r.get("flip_reduction_mean") or 0 for r in curve]
        ax1.plot(ks, recs, color=color, linestyle=ls, linewidth=2,
                 marker="o", markersize=3, label=key)
        ax2.plot(ks, dcs,  color=color, linestyle=ls, linewidth=2,
                 marker="o", markersize=3, label=key)
        ax3.plot(ks, frs,  color=color, linestyle=ls, linewidth=2,
                 marker="o", markersize=3, label=key)

    for ax, yl, title in [
        (ax1, "Recovery to orig (mean, sig)", "PRIMARY: Stepwise recovery"),
        (ax2, "Destructive causal fraction (sig)", "SECONDARY: Stepwise destructive ablation"),
        (ax3, "Flip reduction (mean)", "Flip reduction"),
    ]:
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("k (heads ablated)")
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Cumulative Ablation Curves  (dashed=early, dash-dot=controls, solid=full)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_layer_group(layer_group_results: dict, output_path: Path):
    families = [f for f in layer_group_results if layer_group_results[f]]
    groups = ["early", "mid", "late"]
    if not families:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(groups)); w = 0.75 / len(families)
    for fi, fam in enumerate(families):
        color = FAMILY_COLORS.get(fam, "grey")
        off = (fi - len(families) / 2 + 0.5) * w
        rec_vals = [
            (layer_group_results[fam].get(g, {}).get("significant", {})
             .get("recovery_to_orig_logit") or {}).get("mean") or 0
            for g in groups
        ]
        fl_vals = [
            (layer_group_results[fam].get(g, {}).get("all", {})
             .get("flip_reduction") or {}).get("mean") or 0
            for g in groups
        ]
        atten_vals = [
            (layer_group_results[fam].get(g, {}).get("significant", {})
             .get("attenuation_without_reversal_rate") or 0)
            for g in groups
        ]
        ax1.bar(x + off, rec_vals, w, color=color, alpha=0.8, label=fam)
        ax2.bar(x + off, fl_vals,  w, color=color, alpha=0.8, label=fam)
        ax3.bar(x + off, atten_vals, w, color=color, alpha=0.8, label=fam)
    for ax, yl, t in [
        (ax1, "Recovery to orig (sig)", "PRIMARY: Layer-group recovery"),
        (ax2, "Flip reduction (mean)",  "Flip reduction by layer group"),
        (ax3, "Attenuation without reversal (sig)", "Attenuation rate by layer group"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(["Early\n(0–3)", "Mid\n(10–20)", "Late\n(28+)"])
        ax.set_ylabel(yl); ax.set_title(t); ax.legend(fontsize=9)
        ax.axhline(0, color="k", linewidth=0.5)
    fig.suptitle("Layer-Group Comparison: Recovery (primary) and Flip Reduction", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_direction_patching(patching_results: dict, output_path: Path):
    def _iter_patch_sets(fam_result: dict) -> list[tuple[str, dict]]:
        if not fam_result:
            return []
        # Back-compat: old shape {cf2orig, orig2cf}
        if fam_result.get("cf2orig") and fam_result.get("orig2cf"):
            return [("patch", fam_result)]
        out = []
        for k, v in fam_result.items():
            if isinstance(v, dict) and v.get("cf2orig") and v.get("orig2cf"):
                out.append((k, v))
        return out

    entries: list[tuple[str, str, dict]] = []
    for fam, fam_result in patching_results.items():
        for set_name, v in _iter_patch_sets(fam_result):
            entries.append((fam, set_name, v))

    if not entries:
        return

    fig, axes = plt.subplots(3, len(entries),
                             figsize=(5 * len(entries), 12),
                             squeeze=False)

    for col, (fam, set_name, r) in enumerate(entries):
        color = FAMILY_COLORS.get(fam, "grey")
        # Row 0: recovery (cf2orig) and injection (orig2cf)
        ax = axes[0][col]
        rec_sig = ((r.get("cf2orig", {}).get("agg", {}).get("significant", {})
                    .get("recovery_to_orig_logit") or {}).get("mean") or 0)
        inj_sig = ((r.get("orig2cf", {}).get("agg", {}).get("significant", {})
                    .get("injection_to_cf_logit") or {}).get("mean") or 0)
        rec_flip = ((r.get("cf2orig", {}).get("agg", {}).get("answer_flip_pairs", {})
                     .get("recovery_to_orig_logit") or {}).get("mean") or 0)
        inj_flip = ((r.get("orig2cf", {}).get("agg", {}).get("answer_flip_pairs", {})
                     .get("injection_to_cf_logit") or {}).get("mean") or 0)

        x = np.arange(2)
        w = 0.35
        b1 = ax.bar(x - w/2, [rec_sig, inj_sig], w, color=color, alpha=0.55, label="sig")
        b2 = ax.bar(x + w/2, [rec_flip, inj_flip], w, color=color, alpha=0.90, label="answer_flip_pairs")
        ax.set_xticks(x)
        ax.set_xticklabels(["CF→Orig\n(Recovery)", "Orig→CF\n(Injection)"])
        ax.legend(fontsize=7)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axhline(1, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("Score (↑ = stronger effect)", fontsize=8)
        title = fam.replace("_", "\n")
        if set_name and set_name != "patch":
            title = f"{title}\n[{set_name}]"
        ax.set_title(title, color=color, fontsize=9, fontweight="bold")
        # Row 1: pred match rates
        ax2 = axes[1][col]
        pm_orig = (r.get("cf2orig", {}).get("agg", {}).get("significant", {})
                   .get("pred_patched_matches_orig_rate") or 0)
        pm_cf   = (r.get("orig2cf", {}).get("agg", {}).get("significant", {})
                   .get("pred_patched_matches_cf_rate") or 0)
        bars2 = ax2.bar(["Orig match\n(CF→Orig)", "CF match\n(Orig→CF)"],
                        [pm_orig, pm_cf], color=[color, color], alpha=1.0)
        if len(bars2) >= 2:
            bars2[0].set_alpha(0.85)
            bars2[1].set_alpha(0.50)
        ax2.set_ylim(0, 1); ax2.set_ylabel("Pred match rate", fontsize=8)
        # Row 2: overshoot rates
        ax3 = axes[2][col]
        overshoot_c2o = (r.get("cf2orig", {}).get("agg", {}).get("significant", {})
                         .get("overshoot_logit_rate") or 0)
        overshoot_o2c = (r.get("orig2cf", {}).get("agg", {}).get("significant", {})
                         .get("overshoot_logit_rate") or 0)
        bars3 = ax3.bar(["CF→Orig\novershoot", "Orig→CF\novershoot"],
                        [overshoot_c2o, overshoot_o2c], color=[color, color], alpha=1.0)
        if len(bars3) >= 2:
            bars3[0].set_alpha(0.85)
            bars3[1].set_alpha(0.50)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Overshoot rate (sig)", fontsize=8)
        ax3.set_title("Overshoot logit rate", fontsize=8)
    fig.suptitle(
        "Direction Patching  (significant pairs)\n"
        "CF→Orig recovery: does restoring orig activations remove the shift?\n"
        "Orig→CF injection: does injecting CF activations induce the shift?",
        fontsize=9, y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_residual(per_layer_rows: list, n_layers: int, output_path: Path):
    if not per_layer_rows:
        return
    rows_sorted = sorted(per_layer_rows, key=lambda r: r["layer"])
    layers = [r["layer"] for r in rows_sorted]
    rec  = [r.get("recovery_to_orig_logit_mean") or 0 for r in rows_sorted]
    rstd = [r.get("recovery_to_orig_logit_std")  or 0 for r in rows_sorted]
    dest = [r.get("residual_causal_fraction_mean") or 0 for r in rows_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_layers * 0.35), 5))
    color = FAMILY_COLORS["sexual_orientation"]

    ax1.bar(layers, rec, color=color, alpha=0.75, yerr=rstd, capsize=2,
            error_kw={"elinewidth": 0.6})
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_xlabel("Layer (patched individually)")
    ax1.set_ylabel("Recovery to orig (PRIMARY)")
    ax1.set_title("B2: Residual Patching — Recovery\n"
                  "(carrier layers: where patching CF→orig moves signal back)")

    ax2.bar(layers, dest, color=color, alpha=0.5)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Residual causal fraction (SECONDARY)")
    ax2.set_title("B2: Residual Patching — Causal Fraction (secondary)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_context_split(split_results: dict, output_path: Path):
    splits = ["partner", "explicit"]
    labels = ["Partner-based\n('partner')", "Explicit identity\n('gay','straight')"]
    color  = FAMILY_COLORS["sexual_orientation"]
    rec  = [(split_results.get(s, {}).get("agg", {}).get("significant", {})
             .get("recovery_to_orig_logit") or {}).get("mean") or 0 for s in splits]
    dest = [(split_results.get(s, {}).get("agg", {}).get("significant", {})
             .get("destructive_causal_fraction_logit") or {}).get("mean") or 0 for s in splits]
    fb   = [split_results.get(s, {}).get("agg", {}).get("all", {})
            .get("answer_flip_rate") or 0 for s in splits]
    fa   = [split_results.get(s, {}).get("agg", {}).get("all", {})
            .get("abl_flip_rate") or 0 for s in splits]
    ns   = [split_results.get(s, {}).get("n", 0) for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(2)

    axes[0].bar(x, rec, color=color, alpha=0.75, width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    axes[0].set_ylabel("Recovery to orig (PRIMARY, sig)")
    axes[0].set_title("Recovery by context type"); axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].bar(x, dest, color=color, alpha=0.45, width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    axes[1].set_ylabel("Destructive causal fraction (SECONDARY, sig)")
    axes[1].set_title("Destructive fraction by context type"); axes[1].axhline(0, color="k", linewidth=0.5)

    atten = [(split_results.get(s, {}).get("agg", {}).get("significant", {})
              .get("attenuation_without_reversal_rate") or 0) for s in splits]
    rev   = [(split_results.get(s, {}).get("agg", {}).get("significant", {})
              .get("reversal_after_intervention_rate") or 0) for s in splits]
    w = 0.25
    axes[2].bar(x - w/2, atten, w, color="steelblue", alpha=0.8, label="Attenuation (no reversal)")
    axes[2].bar(x + w/2, rev,   w, color=color,       alpha=0.8, label="Reversal")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    axes[2].set_ylabel("Rate (significant pairs)")
    axes[2].set_title("Intervention outcome rates")
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0, 1)

    fig.suptitle("B3: Context Sensitivity Split — Sexual Orientation", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_component_overlay(comp_results_by_family: dict, output_path: Path):
    """
    06: Per-family overlay of residual / MLP / attention recovery by layer.
    comp_results_by_family: {family: {"residual": layer_agg, "mlp": ..., "attention": ...}}
    where each layer_agg is {str(layer): {mean, std, ...}} from aggregate_layer_results.
    """
    families = [f for f in comp_results_by_family if comp_results_by_family[f]]
    if not families:
        return
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    comp_colors = {"residual": "steelblue", "mlp": "darkorange", "attention": "forestgreen"}

    for col, fam in enumerate(families):
        ax = axes[0][col]
        fam_data = comp_results_by_family[fam]
        for comp in ["residual", "mlp", "attention"]:
            comp_agg = fam_data.get(comp)
            if not comp_agg:
                continue
            rows = sorted(comp_agg.items(), key=lambda kv: int(kv[0]))
            layers = [int(k) for k, _ in rows]
            means  = [v.get("mean") or 0 for _, v in rows]
            stds   = [v.get("std")  or 0 for _, v in rows]
            ax.plot(layers, means, color=comp_colors[comp], linewidth=1.8,
                    marker="o", markersize=3, label=comp)
            ax.fill_between(layers,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=comp_colors[comp], alpha=0.12)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axhline(1, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Recovery to orig (cf→orig)")
        ax.set_title(fam.replace("_", "\n"),
                     color=FAMILY_COLORS.get(fam, "grey"), fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("C1–C3: Component Patching Recovery by Layer\n"
                 "Blue=residual  Orange=MLP  Green=attention (significant pairs, cf→orig)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_within_layer_decomp(decomp_by_family: dict, output_path: Path):
    """
    07: Grouped bar chart per selected layer showing attention/MLP/residual recovery.
    decomp_by_family: {family: exp_within_layer_decomp result dict}
    """
    families = [f for f in decomp_by_family if decomp_by_family[f].get("per_layer_rows")]
    if not families:
        return
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    comp_colors = {"attention": "forestgreen", "mlp": "darkorange", "residual": "steelblue"}

    for col, fam in enumerate(families):
        ax = axes[0][col]
        rows = decomp_by_family[fam]["per_layer_rows"]
        rows_sorted = sorted(rows, key=lambda r: r["layer"])
        layers = [r["layer"] for r in rows_sorted]
        x = np.arange(len(layers))
        w = 0.25

        for ci, comp in enumerate(["residual", "mlp", "attention"]):
            means = [r.get(f"{comp}_mean") or 0 for r in rows_sorted]
            stds  = [r.get(f"{comp}_std")  or 0 for r in rows_sorted]
            offset = (ci - 1) * w
            ax.bar(x + offset, means, w,
                   color=comp_colors[comp], alpha=0.8, label=comp,
                   yerr=stds, capsize=2, error_kw={"elinewidth": 0.7})

        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Recovery to orig")
        ax.set_title(fam.replace("_", "\n"),
                     color=FAMILY_COLORS.get(fam, "grey"), fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("C4: Within-Layer Decomposition\n"
                 "Recovery per component at top causal layers (significant pairs)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_so_context_component(partner_results: dict, explicit_results: dict,
                               output_path: Path):
    """
    08: 3×2 grid: residual/MLP/attention rows × partner/explicit columns.
    partner_results / explicit_results: {"residual": layer_agg, "mlp": ..., "attention": ...}
    """
    components = ["residual", "mlp", "attention"]
    comp_colors = {"residual": "steelblue", "mlp": "darkorange", "attention": "forestgreen"}
    context_labels = {"partner": "Partner-based", "explicit": "Explicit identity"}
    datasets = {"partner": partner_results, "explicit": explicit_results}

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)

    for row, comp in enumerate(components):
        for col, ctx in enumerate(["partner", "explicit"]):
            ax = axes[row][col]
            comp_agg = datasets[ctx].get(comp)
            if not comp_agg:
                ax.set_visible(False)
                continue
            rows_sorted = sorted(comp_agg.items(), key=lambda kv: int(kv[0]))
            layers = [int(k) for k, _ in rows_sorted]
            means  = [v.get("mean") or 0 for _, v in rows_sorted]
            stds   = [v.get("std")  or 0 for _, v in rows_sorted]
            ax.plot(layers, means, color=comp_colors[comp], linewidth=2,
                    marker="o", markersize=3)
            ax.fill_between(layers,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=comp_colors[comp], alpha=0.15)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axhline(1, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
            if row == 0:
                ax.set_title(context_labels[ctx], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{comp}\nRecovery", fontsize=9)
            ax.set_xlabel("Layer", fontsize=8)

    fig.suptitle("C7: Sexual Orientation — Component Patching by Context Type\n"
                 "(cf→orig recovery, significant pairs)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_family_localization(c1_results_by_family: dict, output_path: Path):
    """
    09: Peak recovery layer and magnitude from C1a residual patching per family.
    c1_results_by_family: {family: layer_agg dict from aggregate_layer_results}
    """
    families = [f for f in c1_results_by_family if c1_results_by_family[f]]
    if not families:
        return

    peak_layers = []
    peak_values = []
    for fam in families:
        agg = c1_results_by_family[fam]
        top1 = top_k_layers_by_recovery(agg, 1)
        peak_layers.append(top1[0][0] if top1 else 0)
        peak_values.append(top1[0][1] if top1 else 0)

    top3_vals_by_fam = []
    for fam in families:
        agg = c1_results_by_family[fam]
        top3 = top_k_layers_by_recovery(agg, 3)
        top3_rec = [r for _, r in top3]
        top3_vals_by_fam.append(float(np.mean(top3_rec)) if top3_rec else 0)

    x = np.arange(len(families))
    colors = [FAMILY_COLORS.get(f, "grey") for f in families]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.bar(x, peak_layers, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=9)
    ax1.set_ylabel("Peak causal layer (max recovery)")
    ax1.set_title("Where is the signal? — Peak recovery layer\n(C1a: residual patching cf→orig)")

    ax2.bar(x, peak_values, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=9)
    ax2.set_ylabel("Peak recovery_to_orig_logit (mean, sig)")
    ax2.set_title("How strong? — Peak recovery magnitude\n(C1a: residual patching cf→orig)")
    ax2.axhline(0, color="k", linewidth=0.5)

    ax3.bar(x, top3_vals_by_fam, color=colors, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=9)
    ax3.set_ylabel("Mean of top-3 layer recovery (sig)")
    ax3.set_title("Top-3 layers mean recovery\n(C1a: residual patching cf→orig)")
    ax3.axhline(0, color="k", linewidth=0.5)

    fig.suptitle("C6: Family Localization Comparison — Residual Patching Summary", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     required=True)
    parser.add_argument("--data_path",      required=True)
    parser.add_argument("--top_heads_path", required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--dtype",   default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_pairs",           type=int, default=200)
    parser.add_argument("--top_k_early",         type=int, default=10)
    parser.add_argument("--top_k_full",          type=int, default=20)
    parser.add_argument("--early_layer_max",     type=int, default=3)
    parser.add_argument("--mid_layer_min",       type=int, default=10)
    parser.add_argument("--mid_layer_max",       type=int, default=20)
    parser.add_argument("--late_layer_min",      type=int, default=28)
    parser.add_argument("--n_per_group",         type=int, default=5)
    parser.add_argument("--residual_max_pairs",  type=int, default=50)
    parser.add_argument("--mean_act_max_pairs",  type=int, default=50,
                        help="(Deprecated) Previously used for family-wide means. "
                             "Mean ablation now uses matched-QID baselines.")
    parser.add_argument("--min_effect", type=float, default=MIN_EFFECT_THRESHOLD)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--case1_families", nargs="*", default=None,
                        help="Limit CASE 1 to these families (subset of: "
                             "gender_identity race sex_gender). Default: all.")
    parser.add_argument("--case1_start", default="A1",
                        choices=["A1", "A2", "A3", "A4"],
                        help="Resume CASE 1 at a specific experiment (skips earlier ones).")
    parser.add_argument("--skip_case2", action="store_true",
                        help="Skip CASE 2 (sexual orientation) entirely.")
    parser.add_argument("--run_component_experiments", action="store_true",
                        help="Run C1-C7 component experiments (MLP/attention/residual patching)")
    parser.add_argument("--component_max_pairs", type=int, default=50,
                        help="Max pairs for C1-C3 component patching experiments")
    parser.add_argument("--direction_max_pairs", type=int, default=30,
                        help="Max pairs for C5 direction analysis (memory-heavy)")
    parser.add_argument("--within_layer_top_k",  type=int, default=5,
                        help="Top-K layers from C1a to use in C4 within-layer decomposition")
    parser.add_argument("--skip_direction_analysis", action="store_true",
                        help="Skip C5 direction analysis (saves memory)")
    parser.add_argument("--intervention_scope", default="final_token",
                        choices=["final_token", "all_positions"],
                        help="Scope of causal interventions. "
                             "final_token patches/ablates only the final sequence position. "
                             "all_positions patches all positions (less comparable across components).")
    parser.add_argument("--within_layer_min_recovery", type=float, default=0.05,
                        help="Minimum mean recovery_to_orig_logit for a layer to be "
                             "eligible for C4 within-layer decomposition")
    parser.add_argument("--within_layer_min_n_valid", type=int, default=5,
                        help="Minimum n_valid significant pairs at a layer for C4 eligibility")
    parser.add_argument("--aggressive_cuda_cache_clear", action="store_true",
                        help="If set, calls torch.cuda.empty_cache() after each forward pass. "
                             "Usually slows runs; only use for OOM troubleshooting.")
    args = parser.parse_args()

    scope = args.intervention_scope
    min_effect = args.min_effect
    global_warnings: list[str] = []
    global AGGRESSIVE_CUDA_CACHE_CLEAR
    AGGRESSIVE_CUDA_CACHE_CLEAR = bool(args.aggressive_cuda_cache_clear)

    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    for d in [output_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- Load top heads ----
    print("Loading top_heads.json...")
    with open(args.top_heads_path) as f:
        thd = json.load(f)

    # ---- Layer ranges ----
    early_range = range(0, args.early_layer_max + 1)
    mid_range   = range(args.mid_layer_min, args.mid_layer_max + 1)

    # ---- Model ----
    device = choose_device(args.device)
    dtype  = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}[args.dtype]
    model, tok, n_layers, n_heads, head_dim = load_model(args.model_path, device, dtype)
    late_range = range(args.late_layer_min, n_layers)

    answer_ids = get_answer_token_ids(tok)
    print(f"Answer token IDs: {answer_ids}")

    print("Validating o_proj input shape...")
    validate_o_proj_shape(model, tok, n_heads, head_dim, device)

    # ---- Load pairs ----
    print(f"\nLoading pairs from {args.data_path}...")
    all_pairs, assignment_log = load_pairs(args.data_path, include_controls=True)
    print("Family assignment:")
    for fam, info in assignment_log.items():
        ex = info["examples"][:2]
        print(f"  {fam}: {info['count']}  e.g. {[(e['itype'], e['attr_val']) for e in ex]}")
    save_json(assignment_log, output_dir / "assignment_log.json")

    # ---- Pre-flight: orientation split ----
    so_pairs_all = [p for p in all_pairs if p["family"] == "sexual_orientation"]
    split_counts: Counter = Counter(orientation_split_label(p["attr_norm"])
                                    for p in so_pairs_all)
    print(f"\nOrientation split pre-flight: {dict(split_counts)}")
    if so_pairs_all and split_counts.get("other", 0) / len(so_pairs_all) > 0.10:
        _warn_if(True, ">10% of sexual-orientation pairs assigned to 'other'. "
                 "Inspect PARTNER/EXPLICIT_PATTERNS.", global_warnings)

    # =======================================================================
    # CASE 1
    # =======================================================================
    killer_results   = {}
    stepwise_curves  = {}   # for the stepwise plot
    layer_group_results = {}
    patching_results = {}
    control_matchedness = {}

    case1_start_order = {"A1": 1, "A2": 2, "A3": 3, "A4": 4}
    start_k = case1_start_order.get(args.case1_start, 1)

    def _do_case1(exp: str) -> bool:
        return case1_start_order[exp] >= start_k

    case1_fams = args.case1_families if args.case1_families else CASE1_FAMILIES

    for fam in case1_fams:
        if fam not in thd:
            print(f"\nSkipping {fam} (not in top_heads.json)")
            _warn_if(True, f"No top_heads entry for {fam}.", global_warnings)
            continue

        focal_pairs = sample_pairs(all_pairs, fam, args.max_pairs, rng)
        ctrl_pairs, frac_matched = sample_matched_controls(
            all_pairs, focal_pairs, args.max_pairs, rng, global_warnings
        )
        ctrl_label = "matched_controls" if frac_matched >= 0.5 else "reference_controls"
        control_matchedness[fam] = {"frac_matched": frac_matched, "label": ctrl_label,
                                     "n_controls": len(ctrl_pairs)}
        if not focal_pairs:
            print(f"\nSkipping {fam}: no pairs")
            continue

        print(f"\n{'='*60}")
        print(f"CASE 1: {fam.upper()}  ({len(focal_pairs)} focal, {len(ctrl_pairs)} controls)")
        print('='*60)

        early_heads = filter_heads_by_layer(thd, fam, early_range, args.top_k_early)
        full_heads  = top_n_heads(thd, fam, args.top_k_full)
        print(f"  Early heads (layers 0–{args.early_layer_max}): {len(early_heads)}")
        print(f"  Full top-{args.top_k_full}: {len(full_heads)}")

        if not early_heads:
            _warn_if(True, f"No early heads (layers 0–{args.early_layer_max}) found "
                     f"for {fam}.", global_warnings)

        mean_acts = {}
        if _do_case1("A1") or _do_case1("A2") or _do_case1("A3"):
            # ---- Pre-compute matched-QID mean activations ----
            all_head_layers = list({l for l, _ in full_heads})
            qids_for_means = list({p.get("qid") for p in (focal_pairs + ctrl_pairs) if p.get("qid") is not None})
            print(f"  Computing QID-matched mean activations ({len(qids_for_means)} QIDs, "
                  f"{len(all_head_layers)} layers)...")
            mean_acts = compute_qid_matched_head_activations(
                model, tok, focal_pairs + ctrl_pairs, all_head_layers, device,
                scope=scope
            )
            print(f"  QID-matched mean activations computed for {len(mean_acts)} QIDs.")

        # ----------------------------------------------------------------
        # A1: Individual early-head ablation
        # ----------------------------------------------------------------
        if _do_case1("A1"):
            print(f"\n  A1: Individual ablation ({len(early_heads)} early heads)")
            per_pair_a1, per_head_a1 = exp_individual_ablation(
                model, tok, focal_pairs, answer_ids, early_heads,
                head_dim, device, fam, mean_acts, min_effect=min_effect, scope=scope
            )
            exp_dir = make_exp_dir(output_dir, fam, "a1_individual")
            n_a1_sig = sum(1 for r in per_pair_a1 if r.get("is_significant"))
            save_experiment(
                exp_dir,
                metadata={"family": fam, "experiment": "a1_individual",
                          "n_heads": len(early_heads), "n_pairs_per_head": len(focal_pairs),
                          "n_significant_pairs": n_a1_sig,
                          "early_layer_max": args.early_layer_max,
                          "ablation_mode": "mean_primary_zero_secondary",
                          "intervention_scope": scope,
                          "mean_activation_scope": scope,
                              "mean_activation_mode": "qid_matched_orig",
                              "mean_activation_n_qids": len(mean_acts),
                              "mean_activation_n_pairs": None},
                aggregate=aggregate_v2(per_pair_a1),
                per_pair=per_pair_a1,
                per_head=per_head_a1,
            )
            # Rank by recovery for the summary
            per_head_a1_sorted = sorted(
                [r for r in per_head_a1 if r.get("recovery_to_orig_logit_mean_sig") is not None],
                key=lambda r: r["recovery_to_orig_logit_mean_sig"], reverse=True
            )
            print(f"  Top recovery heads (early): "
                  f"{[(r['head_id'], r['recovery_to_orig_logit_mean_sig']) for r in per_head_a1_sorted[:3]]}")

        # ----------------------------------------------------------------
        # A2: Stepwise ablation — early
        # ----------------------------------------------------------------
        curve_early, curve_full = None, None
        if _do_case1("A2"):
            print(f"\n  A2 stepwise (early, {len(early_heads)} heads)")
            curve_early, final_pairs_early = exp_stepwise_ablation(
                model, tok, focal_pairs, answer_ids, early_heads,
                head_dim, device, mean_acts, min_effect=min_effect, scope=scope
            )
            stepwise_curves[f"{fam}_early"] = curve_early
            exp_dir = make_exp_dir(output_dir, fam, "a2_stepwise_early")
            save_experiment(
                exp_dir,
                metadata={"family": fam, "experiment": "a2_stepwise_early",
                          "n_heads": len(early_heads), "n_pairs": len(focal_pairs),
                          "ablation_mode": "mean_primary_zero_secondary",
                          "intervention_scope": scope,
                          "mean_activation_scope": scope,
                          "mean_activation_mode": "qid_matched_orig",
                          "mean_activation_n_qids": len(mean_acts),
                          "mean_activation_n_pairs": None},
                aggregate=aggregate_v2(final_pairs_early) if final_pairs_early else {},
                per_pair=final_pairs_early,
                curve=curve_early,
            )

            # A2: Stepwise ablation — full
            print(f"\n  A2 stepwise (full, {len(full_heads)} heads)")
            curve_full, final_pairs_full = exp_stepwise_ablation(
                model, tok, focal_pairs, answer_ids, full_heads,
                head_dim, device, mean_acts, min_effect=min_effect, scope=scope
            )
            stepwise_curves[f"{fam}_full"] = curve_full
            exp_dir = make_exp_dir(output_dir, fam, "a2_stepwise_full")
            save_experiment(
                exp_dir,
                metadata={"family": fam, "experiment": "a2_stepwise_full",
                          "n_heads": len(full_heads), "n_pairs": len(focal_pairs),
                          "ablation_mode": "mean_primary_zero_secondary",
                          "intervention_scope": scope,
                          "mean_activation_scope": scope,
                          "mean_activation_mode": "qid_matched_orig",
                          "mean_activation_n_qids": len(mean_acts),
                          "mean_activation_n_pairs": None},
                aggregate=aggregate_v2(final_pairs_full) if final_pairs_full else {},
                per_pair=final_pairs_full,
                curve=curve_full,
            )

            # A2: Stepwise — reference controls (full head set, no mean acts for controls)
            if ctrl_pairs:
                print(f"\n  A2 stepwise ({ctrl_label}, full heads)")
                # Use focal mean acts; controls don't have their own (interpretation note)
                curve_ctrl, final_pairs_ctrl = exp_stepwise_ablation(
                    model, tok, ctrl_pairs, answer_ids, full_heads,
                    head_dim, device, mean_acts, min_effect=min_effect, scope=scope
                )
                stepwise_curves[f"{fam}_control"] = curve_ctrl
                exp_dir = make_exp_dir(output_dir, fam, f"a2_{ctrl_label}")
                save_experiment(
                    exp_dir,
                    metadata={"family": fam, "experiment": f"a2_{ctrl_label}",
                              "n_heads": len(full_heads), "n_pairs": len(ctrl_pairs),
                              "frac_matched": frac_matched, "ctrl_label": ctrl_label,
                              "note": ("Focal heads applied to control pairs. "
                                       "Stage 3 selected focal heads via focal-vs-control "
                                       "contrasts; Stage 4 tests whether those heads "
                                       "causally affect focal families more than reference "
                                       "controls. Controls serve as a perturbation baseline, "
                                       "not a fully symmetric target class.")},
                    aggregate=aggregate_v2(final_pairs_ctrl) if final_pairs_ctrl else {},
                    per_pair=final_pairs_ctrl,
                    curve=curve_ctrl,
                )

            # ---- Localization summary from stepwise ----
            killer_results[fam] = build_localization_summary(
                fam, curve_early, curve_full, early_heads, full_heads, args.early_layer_max
            )
            k = killer_results[fam]
            print(f"\n  *** Localization: "
                  f"recovery_early={_fmt(k['recovery_early'])}  "
                  f"recovery_full={_fmt(k['recovery_full'])}  "
                  f"early_share={_fmt(k['early_share_recovery'])}%  ***")
            _warn_if(
                k["recovery_full"] is not None
                and k["destructive_cf_full"] is not None
                and k["destructive_cf_full"] < -0.1
                and (k["recovery_full"] or 0) < 0.05,
                f"{fam}: destructive ablation is strongly negative while recovery is near zero "
                f"— OOD collateral damage likely dominates the destructive metric.",
                global_warnings
            )

        # ----------------------------------------------------------------
        # A3: Layer-group comparison
        # ----------------------------------------------------------------
        if _do_case1("A3"):
            print(f"\n  A3: Layer-group comparison")
            layer_groups = {"early": early_range, "mid": mid_range, "late": late_range}
            group_agg, per_pair_a3 = exp_layer_group_ablation(
                model, tok, focal_pairs, answer_ids, thd, fam, layer_groups,
                head_dim, device, args.n_per_group, mean_acts, min_effect=min_effect,
                scope=scope
            )
            layer_group_results[fam] = group_agg
            exp_dir = make_exp_dir(output_dir, fam, "a3_layer_groups")
            save_experiment(
                exp_dir,
                metadata={"family": fam, "experiment": "a3_layer_groups",
                          "n_per_group": args.n_per_group,
                          "ablation_mode": "mean_primary_zero_secondary",
                          "intervention_scope": scope,
                          "mean_activation_scope": scope,
                          "mean_activation_mode": "qid_matched_orig",
                          "mean_activation_n_qids": len(mean_acts),
                          "mean_activation_n_pairs": None},
                aggregate={g: group_agg[g] for g in group_agg},
                per_pair=per_pair_a3,
            )

        # ----------------------------------------------------------------
        # A4: Direction patching
        # ----------------------------------------------------------------
        if _do_case1("A4"):
            print(f"\n  A4: Direction patching")
            # Requested: run A4 for both early-top20 and full-top20 head sets
            early_heads_upto20 = filter_heads_by_layer(thd, fam, early_range, 20)
            full_top20  = top_n_heads(thd, fam, 20)

            patch_sets = [
                ("early_heads_upto20", early_heads_upto20),
                ("full_top20",  full_top20),
            ]

            fam_patch_results: dict = {}
            for set_name, heads in patch_sets:
                if not heads:
                    continue
                hbl_for_patching = heads_to_layer_dict(heads)
                pat = exp_direction_patching(
                    model, tok, focal_pairs, answer_ids,
                    hbl_for_patching, head_dim, device,
                    min_effect=min_effect, scope=scope
                )
                fam_patch_results[set_name] = pat

                exp_label = f"a4_direction_patching_{set_name}"
                exp_dir = make_exp_dir(output_dir, fam, exp_label)
                save_experiment(
                    exp_dir,
                    metadata={"family": fam, "experiment": exp_label,
                              "patch_set": set_name,
                              "heads_used": [f"L{l:02d}H{h:02d}" for l, h in heads],
                              "intervention_scope": scope,
                              "note": "cf2orig = recovery (PRIMARY). orig2cf = injection (PRIMARY)."},
                    aggregate={"cf2orig": pat["cf2orig"]["agg"],
                               "orig2cf": pat["orig2cf"]["agg"]},
                    per_pair=pat["cf2orig"]["pairs"] + pat["orig2cf"]["pairs"],
                )
                save_csv(pat["cf2orig"]["pairs"], exp_dir / "cf2orig_pairs.csv")
                save_csv(pat["orig2cf"]["pairs"], exp_dir / "orig2cf_pairs.csv")

            patching_results[fam] = fam_patch_results

    # =======================================================================
    # CASE 2: Sexual Orientation
    # =======================================================================
    if args.skip_case2:
        print("\nSkipping CASE 2 (requested).")
        so_focal = []
        b1_results, b2_results, b3_results = {}, {}, {}
    else:
        print(f"\n{'='*60}")
        print("CASE 2: SEXUAL ORIENTATION")
        print('='*60)

        so_focal = sample_pairs(all_pairs, "sexual_orientation", args.max_pairs, rng)
        b1_results: dict = {}
        b2_results: dict = {}
        b3_results: dict = {}

        if so_focal and "sexual_orientation" in thd:
            # ---- Mean activations for SO ----
            so_full_heads = top_n_heads(thd, "sexual_orientation", args.top_k_full)
            so_head_layers = list({l for l, _ in so_full_heads})
            so_qids = list({p.get("qid") for p in so_focal if p.get("qid") is not None})
            print(f"  Computing SO QID-matched mean activations ({len(so_qids)} QIDs, "
                  f"{len(so_head_layers)} layers)...")
            so_mean_acts = compute_qid_matched_head_activations(
                model, tok, so_focal, so_head_layers, device,
                scope=scope
            )
            print(f"  SO QID-matched mean activations computed for {len(so_mean_acts)} QIDs.")

            # ---- B1: Multi-head ablation at 3 scales ----
            for k_so in [5, 10, args.top_k_full]:
                heads_so = top_n_heads(thd, "sexual_orientation", k_so)
                print(f"\n  B1: Stepwise ablation (top-{k_so})")
                curve_so, final_pairs_so = exp_stepwise_ablation(
                    model, tok, so_focal, answer_ids, heads_so,
                    head_dim, device, so_mean_acts, min_effect=min_effect, scope=scope
                )
                b1_results[f"top_{k_so}"] = curve_so
                stepwise_curves[f"sexual_orientation_top{k_so}"] = curve_so
                scale_label = f"b1_top{k_so}"
                exp_dir = make_exp_dir(output_dir, "sexual_orientation", scale_label)
                save_experiment(
                    exp_dir,
                    metadata={"family": "sexual_orientation", "experiment": scale_label,
                              "n_heads": len(heads_so), "n_pairs": len(so_focal),
                              "ablation_mode": "mean_primary_zero_secondary"},
                    aggregate=aggregate_v2(final_pairs_so) if final_pairs_so else {},
                    per_pair=final_pairs_so,
                    curve=curve_so,
                )

            _warn_if(
                not b1_results,
                "No B1 results for sexual_orientation.", global_warnings
            )

            # ---- B2: Residual stream patching ----
            print(f"\n  B2: Residual stream patching ({n_layers} layers, "
                  f"{args.residual_max_pairs} pairs)")
            b2_results = exp_residual_patching(
                model, tok, so_focal[:args.residual_max_pairs],
                answer_ids, range(n_layers), device, min_effect=min_effect
            )
            exp_dir = make_exp_dir(output_dir, "sexual_orientation", "b2_residual_patching")
            n_b2_sig = sum(1 for r in b2_results["per_pair"] if r.get("is_significant"))
            _warn_if(
                n_b2_sig < 5,
                f"Only {n_b2_sig} significant pairs in B2 residual patching — "
                f"results may be unstable.", global_warnings
            )
            save_experiment(
                exp_dir,
                metadata={"family": "sexual_orientation", "experiment": "b2_residual_patching",
                          "n_layers": n_layers, "n_pairs": len(so_focal[:args.residual_max_pairs]),
                          "n_significant": n_b2_sig,
                          "note": ("Identifies carrier layers where patching CF→orig residual "
                                   "stream moves the run back toward original behaviour. "
                                   "These are transport layers, not necessarily encoding layers.")},
                aggregate=b2_results["layer_agg"],
                per_pair=b2_results["per_pair"],
                per_layer=b2_results["per_layer_rows"],
            )

            # ---- B3: Context sensitivity split ----
            print(f"\n  B3: Context sensitivity split")
            hbl_so = heads_to_layer_dict(so_full_heads)
            b3_results = exp_context_split(
                model, tok, all_pairs, answer_ids, hbl_so, head_dim, device,
                args.max_pairs, rng, global_warnings, min_effect=min_effect, scope=scope,
                mean_activations=so_mean_acts,
            )
            for split_name in ["partner", "explicit"]:
                if b3_results.get(split_name, {}).get("pairs"):
                    exp_dir = make_exp_dir(output_dir, "sexual_orientation",
                                           f"b3_context_split_{split_name}")
                    save_experiment(
                        exp_dir,
                        metadata={"family": "sexual_orientation",
                                  "experiment": f"b3_context_split_{split_name}",
                                  "split": split_name,
                                  "n_pairs": b3_results[split_name]["n"]},
                        aggregate=b3_results[split_name]["agg"],
                        per_pair=b3_results[split_name]["pairs"],
                    )
        else:
            print("  Skipping Case 2 (no SO pairs or no SO top_heads).")
            _warn_if(not so_focal, "No sexual-orientation pairs found.", global_warnings)
            _warn_if("sexual_orientation" not in thd,
                     "sexual_orientation not in top_heads.json.", global_warnings)

    # =======================================================================
    # C1–C7: Component experiments (residual / MLP / attention patching)
    # =======================================================================
    comp_layer_agg: dict = {}   # {family: {"residual": layer_agg, "mlp": ..., "attention": ...}}
    comp_c1a_agg:   dict = {}   # {family: layer_agg} for residual cf2orig — used in C4/C6/C9
    c4_results:     dict = {}   # {family: exp_within_layer_decomp result}
    so_ctx_comp:    dict = {"partner": {}, "explicit": {}}  # for C7 / plot 08

    full_layer_range = range(n_layers)

    if args.run_component_experiments:
        print(f"\n{'='*60}")
        print("C1–C7: COMPONENT EXPERIMENTS  (MLP / Attention / Residual)")
        print('='*60)

        # -------------------------------------------------------------------
        # Per Case 1 family: C1a/b, C2a/b, C3a/b, C4, C5
        # -------------------------------------------------------------------
        for fam in CASE1_FAMILIES:
            focal_pairs = sample_pairs(all_pairs, fam, args.component_max_pairs, rng)
            if not focal_pairs:
                print(f"\n  Skipping {fam}: no pairs for component experiments")
                continue

            print(f"\n  --- {fam.upper()} ({len(focal_pairs)} pairs) ---")
            comp_layer_agg[fam] = {}

            for component, direction, exp_label in [
                ("residual",  "cf2orig", "c1a_residual_patching_c2o"),
                ("residual",  "orig2cf", "c1b_residual_patching_o2c"),
                ("mlp",       "cf2orig", "c2a_mlp_patching_c2o"),
                ("mlp",       "orig2cf", "c2b_mlp_patching_o2c"),
                ("attention", "cf2orig", "c3a_attention_patching_c2o"),
                ("attention", "orig2cf", "c3b_attention_patching_o2c"),
            ]:
                direction_str = "cf→orig" if direction == "cf2orig" else "orig→cf"
                print(f"\n    {exp_label}: {component} {direction_str}")
                res = exp_component_patching(
                    model, tok, focal_pairs, answer_ids, full_layer_range, device,
                    component=component, direction=direction, min_effect=min_effect,
                    intervention_scope=scope
                )
                exp_dir = make_exp_dir(output_dir, fam, exp_label)
                n_sig = sum(1 for r in res["per_pair"] if r.get("is_significant"))
                save_experiment(
                    exp_dir,
                    metadata={"family": fam, "experiment": exp_label,
                               "component": component, "direction": direction,
                               "n_pairs": len(focal_pairs), "n_significant": n_sig,
                               "n_layers": n_layers,
                               "intervention_scope": scope},
                    aggregate=res["layer_agg"],
                    per_pair=res["per_pair"],
                    per_layer=res["per_layer_rows"],
                )
                _warn_if(
                    n_sig < 5,
                    f"{exp_label}: only {n_sig} significant pairs — results may be unstable.",
                    global_warnings
                )
                # Store cf2orig layer_agg per component for overlay plots
                if direction == "cf2orig":
                    comp_layer_agg[fam][component] = res["layer_agg"]
                    if component == "residual":
                        comp_c1a_agg[fam] = res["layer_agg"]

            # ----------------------------------------------------------------
            # C4: Within-layer decomposition (top-K layers from C1a)
            # ----------------------------------------------------------------
            c1a_agg = comp_c1a_agg.get(fam, {})
            if c1a_agg:
                eligible = [
                    (k, v) for k, v in c1a_agg.items()
                    if (v.get("mean") or 0) >= args.within_layer_min_recovery
                    and (v.get("n_valid") or 0) >= args.within_layer_min_n_valid
                ]
                eligible_sorted = sorted(
                    eligible,
                    key=lambda kv: (kv[1].get("mean") or -999),
                    reverse=True
                )
                top_layers = [int(k) for k, _ in eligible_sorted[:args.within_layer_top_k]]
                if not top_layers:
                    print(f"\n    C4: Skipping {fam} — no layers meet "
                          f"min_recovery={args.within_layer_min_recovery} "
                          f"min_n_valid={args.within_layer_min_n_valid}")
                    _warn_if(True,
                             f"C4 skipped for {fam}: no layers with recovery >= "
                             f"{args.within_layer_min_recovery} and n_valid >= "
                             f"{args.within_layer_min_n_valid}. "
                             "Consider lowering --within_layer_min_recovery.",
                             global_warnings)
                    continue  # skip C4 for this family
                print(f"\n    C4: Within-layer decomp (top-{args.within_layer_top_k} "
                      f"layers: {top_layers})")
                # Save eligible-layer table
                if eligible_sorted:
                    elig_rows = []
                    for ek, ev in eligible_sorted:
                        elig_rows.append({
                            "layer":         int(ek),
                            "mean":          ev.get("mean"),
                            "std":           ev.get("std"),
                            "n_valid":       ev.get("n_valid"),
                            "frac_positive": ev.get("frac_positive"),
                            "selected":      int(ek) in top_layers,
                        })
                    elig_dir = output_dir / fam / "c4_within_layer_decomposition"
                    elig_dir.mkdir(parents=True, exist_ok=True)
                    save_csv(elig_rows, elig_dir / "eligible_layers.csv")
                c4_res = exp_within_layer_decomp(
                    model, tok, focal_pairs, answer_ids, top_layers, device,
                    min_effect=min_effect, intervention_scope=scope
                )
                c4_results[fam] = c4_res
                exp_dir = make_exp_dir(output_dir, fam, "c4_within_layer_decomposition")
                save_experiment(
                    exp_dir,
                    metadata={"family": fam, "experiment": "c4_within_layer_decomposition",
                               "top_layers": top_layers, "n_pairs": len(focal_pairs),
                               "within_layer_min_recovery": args.within_layer_min_recovery,
                               "within_layer_min_n_valid": args.within_layer_min_n_valid,
                               "n_eligible_layers": len(eligible),
                               "eligible_layers": [int(k) for k, _ in eligible_sorted],
                               "selection_rule": f"mean >= {args.within_layer_min_recovery} AND n_valid >= {args.within_layer_min_n_valid}, top-{args.within_layer_top_k} by mean",
                               "note": "Compares attention/MLP/residual recovery at same layer"},
                    aggregate=c4_res["layer_summary"],
                    per_pair=c4_res["per_pair"],
                    per_layer=c4_res["per_layer_rows"],
                )
                n_c4_sig = sum(1 for r in c4_res["per_pair"] if r.get("is_significant"))
                _warn_if(
                    n_c4_sig < 5,
                    f"C4 within-layer decomp for {fam}: only {n_c4_sig} significant pairs.",
                    global_warnings
                )
            else:
                print(f"\n    C4: Skipping (no C1a results for {fam})")

            # ----------------------------------------------------------------
            # C5: Direction analysis
            # ----------------------------------------------------------------
            if not args.skip_direction_analysis:
                print(f"\n    C5: Direction analysis ({args.direction_max_pairs} pairs)")
                dir_res = exp_direction_analysis(
                    model, tok, focal_pairs, answer_ids, full_layer_range, device,
                    max_pairs=args.direction_max_pairs
                )
                exp_dir = make_exp_dir(output_dir, fam, "c5_direction_analysis")
                save_experiment(
                    exp_dir,
                    metadata={"family": fam, "experiment": "c5_direction_analysis",
                               "n_sig_pairs": dir_res["n_sig_pairs"],
                               "max_pairs": args.direction_max_pairs,
                               "note": ("Cosine similarity and projection of per-pair "
                                        "residual deltas onto family mean direction")},
                    aggregate={"per_layer_stats": dir_res["per_layer_stats"]},
                    per_layer=dir_res["per_layer_stats"],
                )
            else:
                print(f"\n    C5: Skipped (--skip_direction_analysis)")

            # ---- Component summary (C1a/C2a/C3a) ----
            if comp_layer_agg.get(fam):
                comp_summary_rows = []
                n_layers_model = n_layers
                early_r = range(0, args.early_layer_max + 1)
                mid_r   = range(args.mid_layer_min, args.mid_layer_max + 1)
                late_r  = range(args.late_layer_min, n_layers_model)

                for comp in ["residual", "mlp", "attention"]:
                    agg = comp_layer_agg[fam].get(comp)
                    if not agg:
                        continue
                    # peak layer(s)
                    top3 = top_k_layers_by_recovery(agg, 3)
                    best_k = str(top3[0][0]) if top3 else None
                    best_v = {"mean": top3[0][1]} if top3 else {}
                    second_layer = top3[1][0] if len(top3) > 1 else None
                    second_rec   = top3[1][1] if len(top3) > 1 else None
                    third_layer  = top3[2][0] if len(top3) > 2 else None
                    third_rec    = top3[2][1] if len(top3) > 2 else None
                    top3_vals = [r for _, r in top3]
                    top3_mean = float(np.mean(top3_vals)) if top3_vals else None
                    top3_std  = float(np.std(top3_vals))  if top3_vals else None
                    # Count by region
                    top3_layers = [l for l, _ in top3]
                    early_r_list = list(range(0, args.early_layer_max + 1))
                    mid_r_list   = list(range(args.mid_layer_min, args.mid_layer_max + 1))
                    late_r_list  = list(range(args.late_layer_min, n_layers))
                    early_peak_count = sum(1 for l in top3_layers if l in early_r_list)
                    mid_peak_count   = sum(1 for l in top3_layers if l in mid_r_list)
                    late_peak_count  = sum(1 for l in top3_layers if l in late_r_list)
                    # region means
                    def _region_mean(r):
                        vals = [agg[str(l)].get("mean") for l in r
                                if str(l) in agg and agg[str(l)].get("mean") is not None]
                        return float(np.mean(vals)) if vals else None
                    def _region_n(r):
                        return sum(agg[str(l)].get("n_valid") or 0 for l in r if str(l) in agg)

                    comp_summary_rows.append({
                        "family":       fam,
                        "component":    comp,
                        "direction":    "cf2orig",
                        "peak_layer":   int(best_k) if best_k is not None else None,
                        "peak_recovery": best_v.get("mean"),
                        "peak_layer_mean_recovery": best_v.get("mean"),
                        "second_peak_layer":    second_layer,
                        "second_peak_mean_recovery": second_rec,
                        "third_peak_layer":     third_layer,
                        "third_peak_mean_recovery": third_rec,
                        "top3_mean_recovery":   top3_mean,
                        "top3_std_recovery":    top3_std,
                        "early_peak_count":     early_peak_count,
                        "mid_peak_count":       mid_peak_count,
                        "late_peak_count":      late_peak_count,
                        "early_mean_recovery": _region_mean(early_r),
                        "mid_mean_recovery":   _region_mean(mid_r),
                        "late_mean_recovery":  _region_mean(late_r),
                        "early_n_valid": _region_n(early_r),
                        "mid_n_valid":   _region_n(mid_r),
                        "late_n_valid":  _region_n(late_r),
                    })

                fam_dir = output_dir / fam
                fam_dir.mkdir(parents=True, exist_ok=True)
                save_csv(comp_summary_rows, fam_dir / "component_summary.csv")

        # -------------------------------------------------------------------
        # C7: Sexual orientation deep dive (per context split)
        # -------------------------------------------------------------------
        if so_focal:
            print(f"\n  --- C7: SEXUAL ORIENTATION CONTEXT COMPONENT SPLIT ---")
            for ctx_name in ["partner", "explicit"]:
                ctx_pairs = [p for p in so_focal
                             if orientation_split_label(p["attr_norm"]) == ctx_name]
                ctx_pairs = ctx_pairs[:args.component_max_pairs]
                if not ctx_pairs:
                    print(f"    C7 {ctx_name}: no pairs")
                    continue
                print(f"\n    C7 {ctx_name}: {len(ctx_pairs)} pairs")
                so_ctx_comp[ctx_name] = {}

                for component, direction, exp_label in [
                    ("residual",  "cf2orig", f"c7_{ctx_name}_residual_c2o"),
                    ("mlp",       "cf2orig", f"c7_{ctx_name}_mlp_c2o"),
                    ("attention", "cf2orig", f"c7_{ctx_name}_attention_c2o"),
                ]:
                    print(f"      {exp_label}")
                    res = exp_component_patching(
                        model, tok, ctx_pairs, answer_ids, full_layer_range, device,
                        component=component, direction=direction, min_effect=min_effect,
                        intervention_scope=scope
                    )
                    exp_dir = make_exp_dir(
                        output_dir, "sexual_orientation",
                        f"c7_{ctx_name}/{component}_patching_c2o"
                    )
                    n_sig = sum(1 for r in res["per_pair"] if r.get("is_significant"))
                    save_experiment(
                        exp_dir,
                        metadata={"family": "sexual_orientation",
                                   "experiment": exp_label,
                                   "context_split": ctx_name,
                                   "component": component, "direction": direction,
                                   "n_pairs": len(ctx_pairs), "n_significant": n_sig},
                        aggregate=res["layer_agg"],
                        per_pair=res["per_pair"],
                        per_layer=res["per_layer_rows"],
                    )
                    so_ctx_comp[ctx_name][component] = res["layer_agg"]

                # ---- Component summary for this SO context ----
                if so_ctx_comp.get(ctx_name):
                    ctx_summary_rows = []
                    ctx_early_r = range(0, args.early_layer_max + 1)
                    ctx_mid_r   = range(args.mid_layer_min, args.mid_layer_max + 1)
                    ctx_late_r  = range(args.late_layer_min, n_layers)

                    for comp in ["residual", "mlp", "attention"]:
                        agg = so_ctx_comp[ctx_name].get(comp)
                        if not agg:
                            continue
                        top3 = top_k_layers_by_recovery(agg, 3)
                        best_k = str(top3[0][0]) if top3 else None
                        best_v = {"mean": top3[0][1]} if top3 else {}
                        second_layer = top3[1][0] if len(top3) > 1 else None
                        second_rec   = top3[1][1] if len(top3) > 1 else None
                        third_layer  = top3[2][0] if len(top3) > 2 else None
                        third_rec    = top3[2][1] if len(top3) > 2 else None
                        top3_vals = [r for _, r in top3]
                        top3_mean = float(np.mean(top3_vals)) if top3_vals else None
                        top3_std  = float(np.std(top3_vals))  if top3_vals else None
                        top3_layers = [l for l, _ in top3]
                        early_r_list = list(range(0, args.early_layer_max + 1))
                        mid_r_list   = list(range(args.mid_layer_min, args.mid_layer_max + 1))
                        late_r_list  = list(range(args.late_layer_min, n_layers))
                        early_peak_count = sum(1 for l in top3_layers if l in early_r_list)
                        mid_peak_count   = sum(1 for l in top3_layers if l in mid_r_list)
                        late_peak_count  = sum(1 for l in top3_layers if l in late_r_list)
                        def _ctx_region_mean(r):
                            vals = [agg[str(l)].get("mean") for l in r
                                    if str(l) in agg and agg[str(l)].get("mean") is not None]
                            return float(np.mean(vals)) if vals else None
                        def _ctx_region_n(r):
                            return sum(agg[str(l)].get("n_valid") or 0 for l in r if str(l) in agg)

                        ctx_summary_rows.append({
                            "family":       "sexual_orientation",
                            "context":      ctx_name,
                            "component":    comp,
                            "direction":    "cf2orig",
                            "peak_layer":   int(best_k) if best_k is not None else None,
                            "peak_recovery": best_v.get("mean"),
                            "peak_layer_mean_recovery": best_v.get("mean"),
                            "second_peak_layer":    second_layer,
                            "second_peak_mean_recovery": second_rec,
                            "third_peak_layer":     third_layer,
                            "third_peak_mean_recovery": third_rec,
                            "top3_mean_recovery":   top3_mean,
                            "top3_std_recovery":    top3_std,
                            "early_peak_count":     early_peak_count,
                            "mid_peak_count":       mid_peak_count,
                            "late_peak_count":      late_peak_count,
                            "early_mean_recovery": _ctx_region_mean(ctx_early_r),
                            "mid_mean_recovery":   _ctx_region_mean(ctx_mid_r),
                            "late_mean_recovery":  _ctx_region_mean(ctx_late_r),
                            "early_n_valid": _ctx_region_n(ctx_early_r),
                            "mid_n_valid":   _ctx_region_n(ctx_mid_r),
                            "late_n_valid":  _ctx_region_n(ctx_late_r),
                        })

                    ctx_out_dir = output_dir / "sexual_orientation"
                    ctx_out_dir.mkdir(parents=True, exist_ok=True)
                    save_csv(ctx_summary_rows, ctx_out_dir / f"component_summary_{ctx_name}.csv")

        else:
            print("\n  C7: Skipping (no SO pairs).")

    else:
        print("\nSkipping C1-C7 (pass --run_component_experiments to enable).")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\nBuilding summary...")

    def _get_sig_stat(agg_dict, metric, stratum="significant"):
        return (agg_dict.get(stratum, {}).get(metric) or {}).get("mean")

    def _patching_set_summary(pat: dict) -> dict:
        return {
            "cf2orig_recovery_sig": _get_sig_stat(
                pat["cf2orig"]["agg"],
                "recovery_to_orig_logit"
            ),
            "cf2orig_recovery_answer_flip_pairs": _get_sig_stat(
                pat["cf2orig"]["agg"],
                "recovery_to_orig_logit",
                stratum="answer_flip_pairs",
            ),
            "orig2cf_injection_sig": _get_sig_stat(
                pat["orig2cf"]["agg"],
                "injection_to_cf_logit"
            ),
            "orig2cf_injection_answer_flip_pairs": _get_sig_stat(
                pat["orig2cf"]["agg"],
                "injection_to_cf_logit",
                stratum="answer_flip_pairs",
            ),
            "n_answer_flip_pairs": int(
                (pat.get("cf2orig", {}).get("agg", {}).get("answer_flip_pairs", {}) or {}).get("n", 0)
            ),
            "cf2orig_pred_matches_orig_rate_sig":
                pat["cf2orig"]["agg"].get("significant", {}).get("pred_patched_matches_orig_rate"),
        }

    patching_summaries: dict = {}
    for fam, fam_pat in patching_results.items():
        # Back-compat: old shape {cf2orig, orig2cf}
        if isinstance(fam_pat, dict) and fam_pat.get("cf2orig") and fam_pat.get("orig2cf"):
            patching_summaries[fam] = {"patch": _patching_set_summary(fam_pat)}
            continue
        fam_sets = {}
        if isinstance(fam_pat, dict):
            for set_name, pat in fam_pat.items():
                if isinstance(pat, dict) and pat.get("cf2orig") and pat.get("orig2cf"):
                    fam_sets[set_name] = _patching_set_summary(pat)
        patching_summaries[fam] = fam_sets

    summary = {
        "estimand_definition": {
            "primary_estimand": "behavioral_recovery",
            "primary_metric": "recovery_to_orig_logit",
            "interpretation": (
                "Recovery measures how much of the behavioral shift (CF vs original) "
                "is restored under the intervention. 1.0 = full restoration; "
                "0.0 = no effect; >1.0 = overshoot (intervention moves past original); "
                "<0.0 = amplification of the shift."
            ),
            "not_a_linear_decomposition": True,
        },
        "central_question": (
            "Is causal influence disproportionately concentrated in early layer ranges, "
            "and does that concentration differ by attribute family? "
            "Primary evidence: recovery_to_orig_logit. "
            "Secondary (corroborating only): destructive_causal_fraction_logit."
        ),
        "family_assignment_log":  {
            fam: info["count"] for fam, info in assignment_log.items()
        },
        "control_matchedness":    control_matchedness,
        "orientation_split_preflight": dict(split_counts),
        "warnings":               global_warnings,
        "primary_localization_summaries": killer_results,
        "family_confidence_flags": {
            fam: {
                "low_signal_flag": bool(
                    (r.get("n_sig_full") or 0) < 5 or (r.get("n_sig_early") or 0) < 5
                ),
                "unstable_destructive_flag": bool(
                    r.get("destructive_cf_full") is not None
                    and (r.get("destructive_cf_full") or 0) < -0.1
                    and (r.get("recovery_full") or 0) < 0.05
                ),
                "weak_localization_flag": bool(
                    r.get("recovery_full") is not None
                    and (r.get("recovery_full") or 0) < 0.05
                ),
            }
            for fam, r in killer_results.items()
        },
        "patching_summaries": patching_summaries,
        "so_residual_patching_top_recovery_layers": (
            sorted(
                [(r["layer"], r.get("recovery_to_orig_logit_mean"))
                 for r in b2_results.get("per_layer_rows", [])
                 if r.get("recovery_to_orig_logit_mean") is not None],
                key=lambda x: (x[1] or 0), reverse=True
            )[:5]
            if b2_results else []
        ),
        "so_context_split_recovery": {
            s: _get_sig_stat(b3_results.get(s, {}).get("agg", {}),
                              "recovery_to_orig_logit")
            for s in ["partner", "explicit"]
        } if b3_results else {},
        "interpretation": {
            "recovery_to_orig_logit": (
                "PRIMARY metric. 1.0 = full recovery of original gold logit after "
                "mean-ablating focal heads on the CF run. 0.0 = no change. "
                "Negative = ablation pushed run further from original."
            ),
            "recovery_to_orig_prob": (
                "EXPLORATORY metric. Computed in probability space; not gated by the "
                "logit-scale --min_effect. Gated only by a tiny epsilon for numerical "
                "stability, so treat as descriptive rather than headline evidence."
            ),
            "destructive_causal_fraction_logit": (
                "SECONDARY metric (zero-ablation). "
                "Fraction of |Δ gold logit| eliminated by zeroing focal heads. "
                "NEGATIVE values indicate zero-ablation induced OOD behaviour and "
                "amplified the effect — interpret as collateral damage, NOT as evidence "
                "against causal involvement. Use recovery_to_orig_logit as primary evidence."
            ),
            "mean_ablation": (
                "Matched-QID mean ablation: for each pair, replaces focal head outputs "
                "with the baseline captured from the ORIGINAL prompt for the same QID. "
                "This avoids OOD zeros and avoids mixing baselines across unrelated questions."
            ),
            "early_share_recovery": (
                "recovery_early / recovery_full × 100. "
                "High value = early-layer heads account for most of the recoverable signal. "
                "Low value = signal is distributed across deeper layers (as in sexual orientation)."
            ),
            "injection_to_cf_logit": (
                "For orig2cf direction patching: how much of the CF delta was induced "
                "by injecting CF head activations into the original run. "
                "1.0 = full induction. <0 = injection moved away from CF."
            ),
            "injection_to_cf_prob": (
                "EXPLORATORY metric. Probability-space analogue of injection_to_cf_logit; "
                "interpret cautiously and do not compare directly to logit-scale thresholds."
            ),
        },
        "component_localization": {
            fam: {
                comp: {
                    "peak_layer": (
                        top_k_layers_by_recovery(agg, 1)[0][0]
                        if top_k_layers_by_recovery(agg, 1) else None
                    ),
                    "peak_recovery": (
                        top_k_layers_by_recovery(agg, 1)[0][1]
                        if top_k_layers_by_recovery(agg, 1) else None
                    ),
                }
                for comp, agg in comp_layer_agg.get(fam, {}).items()
            }
            for fam in comp_layer_agg
        } if comp_layer_agg else {},
        "component_confidence_flags": {
            fam: {
                comp: {
                    "weak_localization_flag": bool(
                        comp_data.get("peak_recovery") is not None
                        and (comp_data.get("peak_recovery") or 0) < 0.05
                    ),
                }
                for comp, comp_data in fam_data.items()
            }
            for fam, fam_data in (
                {
                    fam: {
                        comp: {
                            "peak_layer": (
                                top_k_layers_by_recovery(agg, 1)[0][0]
                                if top_k_layers_by_recovery(agg, 1) else None
                            ),
                            "peak_recovery": (
                                top_k_layers_by_recovery(agg, 1)[0][1]
                                if top_k_layers_by_recovery(agg, 1) else None
                            ),
                        }
                        for comp, agg in comp_layer_agg.get(fam, {}).items()
                    }
                    for fam in comp_layer_agg
                }
            ).items()
        } if comp_layer_agg else {},
        "within_layer_decomp_summary": {
            fam: {
                str(row["layer"]): {
                    "attention_mean": row.get("attention_mean"),
                    "mlp_mean":       row.get("mlp_mean"),
                    "residual_mean":  row.get("residual_mean"),
                }
                for row in c4_results[fam].get("per_layer_rows", [])
            }
            for fam in c4_results
        } if c4_results else {},
        "so_context_component_comparison": {
            ctx: {
                comp: {
                    "peak_layer": (
                        top_k_layers_by_recovery(agg, 1)[0][0]
                        if top_k_layers_by_recovery(agg, 1) else None
                    ),
                    "peak_recovery": (
                        top_k_layers_by_recovery(agg, 1)[0][1]
                        if top_k_layers_by_recovery(agg, 1) else None
                    ),
                }
                for comp, agg in so_ctx_comp.get(ctx, {}).items()
            }
            for ctx in ["partner", "explicit"]
        } if any(so_ctx_comp.get(c) for c in ["partner", "explicit"]) else {},
        "args": vars(args),
    }

    # -----------------------------------------------------------------------
    # Summary validation pass
    # -----------------------------------------------------------------------
    for fam, r in killer_results.items():
        if r.get("recovery_full") is None and r.get("recovery_early") is None:
            _warn_if(True, f"summary validation: {fam} has no recovery_full or recovery_early.",
                     global_warnings)
    if comp_layer_agg:
        for fam, comp_data in comp_layer_agg.items():
            for comp, agg in comp_data.items():
                valid_layers = [k for k, v in agg.items()
                                if v.get("mean") is not None and v.get("n_valid")]
                if not valid_layers:
                    _warn_if(True,
                             f"summary validation: {fam}/{comp} has no valid layer stats "
                             "in component_localization.",
                             global_warnings)
    # update warnings in summary after validation
    summary["warnings"] = global_warnings

    save_json(summary, output_dir / "summary.json")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if HAS_PLT:
        print("\nGenerating plots...")
        if killer_results:
            plot_localization_summary(killer_results, plots_dir / "00_localization_summary.png")
        if stepwise_curves:
            plot_stepwise_curves(stepwise_curves, plots_dir / "01_stepwise_curves.png")
        if layer_group_results:
            plot_layer_group(layer_group_results, plots_dir / "02_layer_groups.png")
        if patching_results:
            plot_direction_patching(patching_results, plots_dir / "03_direction_patching.png")
        if b2_results.get("per_layer_rows"):
            plot_residual(b2_results["per_layer_rows"], n_layers,
                          plots_dir / "04_residual_patching.png")
        if b3_results:
            plot_context_split(b3_results, plots_dir / "05_context_split.png")
        if comp_layer_agg:
            plot_component_overlay(comp_layer_agg, plots_dir / "06_component_overlay.png")
        if c4_results:
            plot_within_layer_decomp(c4_results, plots_dir / "07_within_layer_decomp.png")
        if any(so_ctx_comp.get(c) for c in ["partner", "explicit"]):
            plot_so_context_component(
                so_ctx_comp.get("partner", {}),
                so_ctx_comp.get("explicit", {}),
                plots_dir / "08_so_context_component.png"
            )
        if comp_c1a_agg:
            plot_family_localization(comp_c1a_agg, plots_dir / "09_family_localization.png")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("LOCALIZATION SUMMARY  (primary: recovery | secondary: destructive CF)")
    print('='*60)
    for fam, r in killer_results.items():
        print(f"\n  {fam}:")
        print(f"    Early (0–{args.early_layer_max}): "
              f"recovery={_fmt(r['recovery_early'])}  "
              f"destructive_cf={_fmt(r['destructive_cf_early'])}  "
              f"({r['n_early_heads']} heads)")
        print(f"    Full top-{r['n_full_heads']}: "
              f"recovery={_fmt(r['recovery_full'])}  "
              f"destructive_cf={_fmt(r['destructive_cf_full'])}")
        print(f"    Early share of recovery: {_fmt(r['early_share_recovery'])}%  "
              f"[early share of destructive fraction: "
              f"{_fmt(r['early_share_destructive'])}%]")

    if global_warnings:
        print(f"\n{'='*60}")
        print("WARNINGS EMITTED DURING RUN:")
        for w in global_warnings:
            print(f"  • {w}")

    print(f"\nAll outputs → {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
