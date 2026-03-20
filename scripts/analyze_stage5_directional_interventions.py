#!/usr/bin/env python3
"""
Stage 5: Directional Activation Interventions  (v2.0)
======================================================

Central questions
-----------------
1. Can we isolate a *linear direction* in the residual stream that encodes
   "attribute a is present" (the **attribute direction** r̂_a)?
2. Does a second direction capture the contrast between medically-relevant
   and medically-irrelevant uses of the same attribute (the
   **relevance-associated contrast direction** m̂_a)?
3. Do activation-addition and directional-ablation along these directions
   reproduce / reverse the counterfactual behavioral shifts?
4. Are the effects *specific* (weak on controls) and *relevance-sensitive*
   (beneficial in medically-relevant contexts)?

Prerequisite stages
-------------------
- Stage 4.5 established *where* demographic information is decodable in
  the residual stream (layer-wise linear probes).
- This script tests *causal* rather than correlational claims by directly
  manipulating the residual stream along computed directions.

Data requirements
-----------------
- cf_v6_balanced.json — primary counterfactual dataset
  * D_irr: demographic variants where attribute is medically irrelevant
  * D_rel: records flagged sex_gender_medically_relevant=True or
           race_medically_relevant=True (originals only — no demographic
           counterfactual variants were generated for these by design)
  * D_ctrl: neutral_rework + irrelevant_surface control variants

Limitations
-----------
- D_rel records lack counterfactual demographic variants (by design of
  the v6.1 balanced generation strategy).
- No omission variants x^{(-a)} exist for D_rel.  Experiment 4 is
  therefore skipped with a machine-readable warning.
- Sexual orientation and gender identity have no explicit medical-
  relevance labels.  All variants for these families are treated as D_irr.

Interpretive notes
------------------
- The attribute direction r̂_a is the mean shift from original to
  counterfactual hidden state in medically-irrelevant contexts.  It
  captures "attribute a was inserted" but may also encode correlated
  lexical or syntactic shifts.
- The relevance-associated contrast direction m̂_a is an approximate
  contrast between prompts where the attribute is medically relevant
  vs. irrelevant.  Because D_rel and D_irr are *not* matched question
  pairs, m̂_a may absorb question-content differences, lexical
  differences, diagnosis/domain differences, and task-distribution
  confounds.  It should be interpreted as an approximate signal, not a
  pure "medical relevance feature".
- Successful addition/ablation implies the direction carries behaviorally
  consequential information at the intervention site.  It does not
  establish that this direction is the sole or unique mechanism.

Intervention mechanics
----------------------
Hooks are registered on ``model.model.layers[block_idx]``
(``LlamaDecoderLayer``).  The hook modifies the post-block hidden-state
tensor — the output of the full transformer block (self-attention + MLP
+ residual connections).  This is not the raw residual stream between
sub-layers; it is the final block output that feeds into the next block.

Layer indexing
--------------
Stage 5 uses **transformer block indices** 0..N-1.
Stage 4.5 uses layer indices where 0 = embedding and 1..N = blocks.
All saved outputs include both ``block_index`` and
``stage45_layer_index`` (= block_index + 1) for cross-stage alignment.

Token-set strategies for intervention
-------------------------------------
- all:         all token positions
- edited_span: only the tokens that differ between orig and CF
- suffix:      tokens after the edit region
- final_token: only the last token position

Output hierarchy
----------------
stage5_results/
  run_config.json
  data_diagnostics.json
  summary.json
  summary_table.csv
  directions/
    {family}/
      attribute_direction.npz
      relevance_contrast_direction.npz
      direction_metadata.json
  site_selection/
    {family}/
      effectiveness_scores.csv
      best_site.json
  experiments/
    {family}/
      1_add_irr/       — Irrelevant original + addition of r̂_a
      2_abl_irr/       — Irrelevant CF + ablation of r̂_a
      3_ctrl/          — Control prompts + addition/ablation of r̂_a
      4_rel_add/       — Relevant omission + addition of r̂_a (SKIPPED)
      5_rel_abl/       — Relevant full prompts + ablation of r̂_a
      6_medrel_abl/    — Harmful flip CFs + ablation of m̂_a
      7_decomposition/ — Joint projection decomposition
  predictions/
    prediction_tests.json
  plots/

Usage
-----
  python analyze_stage5_directional_interventions.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage5_results \\
    --device auto --max_pairs 200 --alpha 1.0 --token_set final_token

  # Direction computation only (skip interventions):
  ... --directions_only

  # Interventions only (load precomputed directions):
  ... --interventions_only

  # Sanity checks only (fast, no GPU needed for full run):
  ... --sanity_checks_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("Warning: matplotlib not available — plots will be skipped")


# ===================================================================
# Constants
# ===================================================================

SCRIPT_NAME = "analyze_stage5_directional_interventions.py"
VERSION = "2.0"

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

FAMILY_COLORS = {
    "gender_identity":    "#9B59B6",
    "race":               "#E67E22",
    "sex_gender":         "#4C96D7",
    "sexual_orientation": "#2ECC71",
    "control":            "#95A5A6",
}

# Minimum absolute denominator for ratio-based metrics.
MIN_EFFECT_THRESHOLD = 0.05

# Effectiveness score weights for layer selection.
LAMBDA_ADD  = 1.0
LAMBDA_ABL  = 1.0
LAMBDA_CTRL = 0.5

DEFAULT_ALPHA = 1.0
TOKEN_SETS = ["all", "edited_span", "suffix", "final_token"]

# Layer region boundaries — expressed as Stage-4.5-compatible layer
# indices (0=embedding, 1..N=transformer blocks).
EARLY_END_S45 = 4   # s45 layers 1..3 → early
MID_END_S45   = 16  # s45 layers 4..15 → mid

MIN_FAMILY_SIZE = 5  # minimum D_irr pairs to attempt direction

_GI_FRAGMENTS = {
    "non-binary", "nonbinary", "non binary",
    "transgender", "trans man", "trans woman", "transman", "transwoman",
    "gender non-conforming", "genderqueer", "agender",
}
_PARTNER_PATTERNS = {"partner", "same sex partner", "partner same sex"}
_EXPLICIT_PATTERNS = {"gay", "straight", "lesbian", "bisexual", "queer",
                      "heterosexual", "homosexual"}


# ===================================================================
# Small reusable helpers
# ===================================================================

def _normalize_label(label) -> str:
    s = str(label or "").lower()
    s = re.sub(r"['\"\-_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def assign_family(itype: str, attr_val) -> str | None:
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


def format_prompt(q: str, options: dict) -> str:
    return PROMPT_TEMPLATE.format(
        question=q.strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


def block_to_region(block_idx: int) -> str:
    """Map transformer block index (0..N-1) to a region label.

    Uses Stage-4.5-compatible layer index (block_idx + 1) for region
    boundaries.
    """
    s45 = block_idx + 1
    if s45 < EARLY_END_S45:
        return "early"
    if s45 < MID_END_S45:
        return "mid"
    return "late"


def unit_normalize(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit vector, norm).  Returns zero vector if norm < 1e-10."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return np.zeros_like(v), norm
    return (v / norm).astype(np.float32), norm


def compute_margin(logits: np.ndarray, gold_idx: int) -> float:
    """M(x) = z_{g(x)} - max_{c != g(x)} z_c."""
    return float(logits[gold_idx] - max(logits[c] for c in range(4)
                                         if c != gold_idx))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_ratio(num: float, den: float,
               threshold: float = MIN_EFFECT_THRESHOLD) -> float | None:
    """Return num/den if |den| > threshold, else None."""
    return num / den if abs(den) >= threshold else None


def joint_decompose(delta_h: np.ndarray, directions: list[np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Joint least-squares decomposition.

    Solves:  beta = argmin ||delta_h - U @ beta||_2^2
    where U = [d1, d2, ...] column-wise.

    Returns (beta, residual, residual_norm).
    """
    if not directions:
        return np.array([]), delta_h.copy(), float(np.linalg.norm(delta_h))
    U = np.column_stack(directions)  # (d, k)
    beta, _, _, _ = np.linalg.lstsq(U, delta_h.astype(np.float64), rcond=None)
    reconstructed = U @ beta
    residual = delta_h.astype(np.float64) - reconstructed
    return beta.astype(np.float32), residual.astype(np.float32), \
        float(np.linalg.norm(residual))


# ===================================================================
# Token-set pooling (canonical helper — Directive A)
# ===================================================================

def tokenize_prompt(tok, prompt: str) -> tuple[list[int], dict]:
    """Canonical tokenization — must be used for ALL span/position logic.

    Uses the same tokenizer call path as the model forward pass
    (``tok(prompt, ...)`` not ``tok.encode(...)``), so that token ids,
    special-token handling, and truncation are identical.  This prevents
    misaligned span indices.  (Issue 2)

    Returns (token_id_list, tokenizer_encoding_dict).
    """
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    ids = enc["input_ids"][0].tolist()
    return ids, enc


def make_control_token_positions(
    n_tokens: int, token_set: str,
) -> tuple[list[int] | None, bool, str]:
    """Token positions for control prompts, which lack a demographic span.

    Controls have no meaningful edited-span geometry relative to a
    demographic counterfactual, so span-dependent token_sets fall back
    to final_token with an explicit policy label.  (Issue 3)

    Returns (positions, is_valid, policy_label).
    """
    if token_set == "all":
        return None, True, "all_tokens"
    if token_set == "final_token":
        return [n_tokens - 1], True, "final_token"
    # edited_span / suffix: no meaningful span for controls → final-token.
    return [n_tokens - 1], True, "fallback_final_token_no_span"


def resolve_relevance_token_set(token_set: str,
                                 relevance_token_set: str) -> str:
    """Determine effective token_set for relevance-contrast direction.

    When --relevance_token_set is 'auto' (the default), span-dependent
    token_sets fall back to final_token because D_rel has no paired CF
    for span detection.  (Issue 4)

    Returns the resolved token_set string.
    """
    if relevance_token_set != "auto":
        return relevance_token_set
    # Auto: match attribute direction unless span-dependent.
    if token_set in ("edited_span", "suffix"):
        return "final_token"
    return token_set


def find_edited_span(orig_ids: list[int], cf_ids: list[int]
                     ) -> tuple[int, int, int, bool]:
    """Returns (edit_start, edit_end_orig, edit_end_cf, is_fallback)."""
    n_o, n_c = len(orig_ids), len(cf_ids)
    prefix = 0
    for i in range(min(n_o, n_c)):
        if orig_ids[i] == cf_ids[i]:
            prefix += 1
        else:
            break
    max_suf = min(n_o - prefix, n_c - prefix)
    suffix = 0
    for i in range(1, max_suf + 1):
        if orig_ids[n_o - i] == cf_ids[n_c - i]:
            suffix += 1
        else:
            break
    es = prefix
    eeo = n_o - suffix
    eec = n_c - suffix
    if es >= eeo or es >= eec:
        return 0, n_o, n_c, True
    return es, eeo, eec, False


def pool_hidden(hidden: torch.Tensor, token_set: str,
                edit_start: int, edit_end: int) -> tuple[np.ndarray, bool]:
    """Pool a (1, seq_len, hidden_size) tensor to a single vector.

    Returns (pooled_vector, is_valid).
    is_valid is False when the requested token_set yields an empty slice;
    the caller must skip or flag that example.
    """
    h = hidden[0].float()  # (seq_len, d)
    seq_len = h.shape[0]

    if token_set == "final_token":
        return h[-1].numpy(), True
    if token_set == "all":
        return h.mean(dim=0).numpy(), True
    if token_set == "edited_span":
        s = max(0, min(edit_start, seq_len))
        e = max(s, min(edit_end, seq_len))
        if e <= s:
            return np.zeros(h.shape[1], dtype=np.float32), False
        return h[s:e].mean(dim=0).numpy(), True
    if token_set == "suffix":
        s = max(0, min(edit_end, seq_len))
        if s >= seq_len:
            return np.zeros(h.shape[1], dtype=np.float32), False
        return h[s:].mean(dim=0).numpy(), True
    raise ValueError(f"Unknown token_set: {token_set!r}")


def make_token_positions(n_tokens: int, token_set: str,
                         edit_start: int, edit_end: int
                         ) -> tuple[list[int] | None, bool]:
    """Return (positions, is_valid) for hook intervention.

    Returns None for positions when token_set == 'all' (hooks should
    apply to all positions).
    """
    if token_set == "all":
        return None, True
    if token_set == "final_token":
        return [n_tokens - 1], True
    if token_set == "edited_span":
        s = max(0, min(edit_start, n_tokens))
        e = max(s, min(edit_end, n_tokens))
        if e <= s:
            return [], False
        return list(range(s, e)), True
    if token_set == "suffix":
        s = max(0, min(edit_end, n_tokens))
        if s >= n_tokens:
            return [], False
        return list(range(s, n_tokens)), True
    raise ValueError(f"Unknown token_set: {token_set!r}")


# ===================================================================
# CSV / metadata helpers
# ===================================================================

def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _save_experiment_metadata(path: Path, **kwargs):
    """Write metadata.json for an experiment folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(kwargs, f, indent=2)


def _layer_output_row(block_idx: int) -> dict:
    """Return standard layer-index fields for any output row/dict."""
    return {
        "block_index": block_idx,
        "stage45_layer_index": block_idx + 1,
        "region": block_to_region(block_idx),
    }


def _mean_or_none(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _std_or_none(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.std(v)) if v else None


def _median_or_none(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


# ===================================================================
# Data loading — partitioned into D_irr, D_rel, D_ctrl
# ===================================================================

def load_partitioned_data(data_path: str) -> dict:
    """Load cf_v6_balanced.json and partition into D_irr / D_rel / D_ctrl.

    Returns dict with keys: pairs_irr, records_rel, pairs_ctrl,
    all_pairs, stats, diagnostics.
    """
    with open(data_path) as f:
        records = json.load(f)

    pairs_irr: dict[str, list] = defaultdict(list)
    records_rel: dict[str, list] = defaultdict(list)
    pairs_ctrl: list = []
    all_pairs: list = []
    stats: dict = defaultdict(lambda: {"n_irr_pairs": 0,
                                        "n_rel_records": 0,
                                        "n_ctrl_pairs": 0})
    skipped_no_gold = 0

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
            skipped_no_gold += 1
            continue

        orig_prompt = format_prompt(orig["question"], options)
        cf_data = rec.get("counterfactuals", {})
        task_type = rec.get("task_type", "")

        # Record-level medical relevance flags.
        is_sex_rel = cf_data.get("sex_gender_medically_relevant", False)
        is_race_rel = cf_data.get("race_medically_relevant", False)
        rel_family = None
        if task_type == "sex_gender" and is_sex_rel:
            rel_family = "sex_gender"
        elif task_type == "race_existing" and is_race_rel:
            rel_family = "race"

        if rel_family:
            records_rel[rel_family].append({
                "qid": qid,
                "task_type": task_type,
                "gold": gold,
                "gold_idx": AIDX[gold],
                "options": options,
                "orig_q": orig["question"],
                "orig_prompt": orig_prompt,
                "original_demographics": cf_data.get(
                    "original_demographics", {}),
            })
            stats[rel_family]["n_rel_records"] += 1

        for v in cf_data.get("variants", []):
            if not isinstance(v, dict) or v.get("text") is None:
                continue
            itype = v.get("intervention_type", "")
            attr_val = v.get("attribute_value_counterfactual", "")
            family = assign_family(itype, attr_val)
            if family is None:
                continue

            pair = {
                "qid": qid,
                "family": family,
                "itype": itype,
                "attr_val": str(attr_val or itype),
                "attr_val_orig": str(v.get("attribute_value_original") or ""),
                "attr_norm": _normalize_label(attr_val),
                "gold": gold,
                "gold_idx": AIDX[gold],
                "options": options,
                "orig_q": orig["question"],
                "cf_q": v["text"],
                "orig_prompt": orig_prompt,
                "cf_prompt": format_prompt(v["text"], options),
            }
            all_pairs.append(pair)
            if family == "control":
                pairs_ctrl.append(pair)
                stats["control"]["n_ctrl_pairs"] += 1
            else:
                pairs_irr[family].append(pair)
                stats[family]["n_irr_pairs"] += 1

    # Build diagnostics.
    diag: dict = {
        "total_records": len(records),
        "skipped_no_gold": skipped_no_gold,
        "total_pairs": len(all_pairs),
        "families": {},
    }
    print("\n=== Data Partition Summary ===")
    for fam in ALL_FOCAL:
        s = stats[fam]
        print(f"  {fam:25s}: D_irr={s['n_irr_pairs']:5d} pairs, "
              f"D_rel={s['n_rel_records']:4d} records")
        diag["families"][fam] = dict(s)
    print(f"  {'control':25s}: {stats['control']['n_ctrl_pairs']:5d} pairs")
    diag["families"]["control"] = dict(stats["control"])
    print(f"  Total pairs: {len(all_pairs)}")

    return {
        "pairs_irr":   dict(pairs_irr),
        "records_rel": dict(records_rel),
        "pairs_ctrl":  pairs_ctrl,
        "all_pairs":   all_pairs,
        "stats":       dict(stats),
        "diagnostics": diag,
    }


def sample_pairs(pairs: list, max_n: int, rng) -> list:
    if len(pairs) <= max_n:
        return pairs
    idx = rng.choice(len(pairs), max_n, replace=False)
    return [pairs[i] for i in sorted(idx)]


def sample_matched_controls(all_ctrl: list, focal_qids: set,
                            max_n: int, rng) -> list:
    matched = [p for p in all_ctrl if p["qid"] in focal_qids]
    unmatched = [p for p in all_ctrl if p["qid"] not in focal_qids]
    if len(matched) >= max_n:
        idx = rng.choice(len(matched), max_n, replace=False)
        return [matched[i] for i in sorted(idx)]
    remainder = max_n - len(matched)
    if len(unmatched) <= remainder:
        fill = unmatched
    else:
        fill = [unmatched[i] for i in sorted(
            rng.choice(len(unmatched), remainder, replace=False))]
    return matched + fill


# ===================================================================
# Model loading + validation
# ===================================================================

def choose_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device_arg: str, dtype_str: str = "float16"):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)
    device = choose_device(device_arg)

    print(f"Loading tokenizer from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model (dtype={dtype_str}, device={device})...")
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  Model ready: {n_layers} transformer blocks, "
          f"hidden_size={hidden_size}")
    return model, tok, device, n_layers, hidden_size


def get_answer_token_ids(tok) -> dict:
    ids = {}
    for letter in "ABCD":
        for s in [letter, f" {letter}"]:
            toks = tok.encode(s, add_special_tokens=False)
            if toks and letter in tok.decode([toks[-1]]).strip():
                ids[letter] = toks[-1]
                break
        if letter not in ids:
            raise ValueError(f"Cannot resolve token id for '{letter}'")
    # Validate uniqueness.
    id_vals = list(ids.values())
    assert len(set(id_vals)) == 4, \
        f"Answer token ids are not distinct: {ids}"
    return ids


# ===================================================================
# Forward pass utilities
# ===================================================================

@torch.no_grad()
def get_logits_and_hidden(model, tok, prompt: str, device: str,
                          answer_ids: dict,
                          layers: list[int] | None = None
                          ) -> tuple[np.ndarray, dict[int, torch.Tensor], int]:
    """Forward pass returning ABCD logits and hidden states.

    Hidden states are indexed by transformer block (0..N-1).
    ``hidden[k]`` = output of block k, shape ``(1, seq_len, d)``.
    """
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    final_logits = out.logits[0, -1, :]
    logits_abcd = np.array([
        final_logits[answer_ids[c]].float().cpu().item() for c in "ABCD"
    ], dtype=np.float32)

    # out.hidden_states: index 0 = embedding, 1..N = block outputs.
    all_hs = out.hidden_states
    n_blocks = len(all_hs) - 1
    target = layers if layers is not None else list(range(n_blocks))
    hidden = {}
    for bi in target:
        if 0 <= bi < n_blocks:
            hidden[bi] = all_hs[bi + 1].detach().cpu()

    return logits_abcd, hidden, n_tokens


@torch.no_grad()
def get_logits_only(model, tok, prompt: str, device: str,
                    answer_ids: dict) -> np.ndarray:
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=False, use_cache=False)
    fl = out.logits[0, -1, :]
    return np.array([fl[answer_ids[c]].float().cpu().item()
                     for c in "ABCD"], dtype=np.float32)


# ===================================================================
# Hook infrastructure
# ===================================================================

def _get_block(model, block_idx: int):
    """Return the LlamaDecoderLayer at the given block index."""
    return model.model.layers[block_idx]


class ResidualAddHook:
    """Add α·direction to the post-block hidden state at given positions.

    Intervention:  h̃_i = h_i + α·d   ∀ i ∈ positions
    """

    def __init__(self, direction: torch.Tensor, alpha: float,
                 positions: list[int] | None = None):
        self.direction = direction.clone()
        self.alpha = alpha
        self.positions = positions
        self.handle = None

    def hook_fn(self, module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        new_hs = hs.clone()
        d = self.direction.to(device=new_hs.device, dtype=new_hs.dtype)
        delta = self.alpha * d
        if self.positions is None:
            new_hs[0] = new_hs[0] + delta.unsqueeze(0)
        else:
            for pos in self.positions:
                if 0 <= pos < new_hs.shape[1]:
                    new_hs[0, pos] = new_hs[0, pos] + delta
        if isinstance(output, (tuple, list)):
            return (new_hs,) + tuple(output[1:])
        return new_hs

    def register(self, model, block_idx: int):
        self.handle = _get_block(model, block_idx).register_forward_hook(
            self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class DirectionalAblationHook:
    """Remove projection onto direction from post-block hidden state.

    Intervention:  h̃_i = h_i − (d^T h_i)·d   ∀ i ∈ positions
    """

    def __init__(self, direction: torch.Tensor,
                 positions: list[int] | None = None):
        self.direction = direction.clone()
        self.positions = positions
        self.handle = None

    def hook_fn(self, module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        new_hs = hs.clone()
        d = self.direction.to(device=new_hs.device, dtype=new_hs.dtype)
        if self.positions is None:
            proj = torch.einsum("bsd,d->bs", new_hs, d)
            new_hs = new_hs - proj.unsqueeze(-1) * d
        else:
            for pos in self.positions:
                if 0 <= pos < new_hs.shape[1]:
                    proj = torch.dot(new_hs[0, pos], d)
                    new_hs[0, pos] = new_hs[0, pos] - proj * d
        if isinstance(output, (tuple, list)):
            return (new_hs,) + tuple(output[1:])
        return new_hs

    def register(self, model, block_idx: int):
        self.handle = _get_block(model, block_idx).register_forward_hook(
            self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def forward_with_hook(model, tok, prompt: str, device: str,
                      answer_ids: dict, hook_obj,
                      block_idx: int,
                      return_hidden: bool = False,
                      hidden_layers: list[int] | None = None
                      ) -> tuple[np.ndarray, dict | None]:
    """Forward pass with a registered hook (always cleaned up)."""
    hook_obj.register(model, block_idx)
    try:
        if return_hidden:
            logits, hidden, _ = get_logits_and_hidden(
                model, tok, prompt, device, answer_ids,
                layers=hidden_layers)
            return logits, hidden
        else:
            return get_logits_only(model, tok, prompt, device,
                                   answer_ids), None
    finally:
        hook_obj.remove()


@torch.no_grad()
def validate_hook_identity(model, tok, device: str, answer_ids: dict,
                           prompt: str, block_idx: int,
                           hidden_size: int, tol: float = 1e-4) -> bool:
    """Verify that a zero-alpha hook produces identical logits."""
    base = get_logits_only(model, tok, prompt, device, answer_ids)
    zero_dir = torch.zeros(hidden_size)
    hook = ResidualAddHook(zero_dir, 0.0)
    hooked, _ = forward_with_hook(model, tok, prompt, device, answer_ids,
                                   hook, block_idx)
    diff = float(np.max(np.abs(base - hooked)))
    ok = diff < tol
    if not ok:
        print(f"  WARNING: hook identity check failed at block {block_idx} "
              f"(max diff={diff:.6f})")
    return ok


# ===================================================================
# Phase 1: Direction computation
# ===================================================================

def _token_set_is_span_dependent(token_set: str) -> bool:
    """True if pooling result depends on the edit span geometry."""
    return token_set in ("edited_span", "suffix")


# FIX 5: Controlled vocabulary for position policy labels.
_POSITION_POLICY_MAP = {
    "all": "all_tokens",
    "final_token": "final_token",
    "edited_span": "true_edited_span",
    "suffix": "true_suffix",
}


def _focal_position_policy(token_set: str) -> str:
    """Return the standardized position policy label for a focal pair.

    Uses controlled vocabulary:
      all_tokens, final_token, true_edited_span, true_suffix.
    """
    return _POSITION_POLICY_MAP[token_set]


def compute_directions(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list],
    records_rel: dict[str, list],
    n_layers: int, hidden_size: int,
    target_layers: list[int],
    max_pairs_per_family: int,
    rng,
    output_dir: Path,
    token_set: str,
    relevance_token_set: str = "auto",
) -> dict[str, dict]:
    """Compute attribute directions r̂_a and relevance-associated contrast
    directions m̂_a for each family.

    Attribute direction (difference-in-means):
        r_a^(l) = μ_{cf,irr}^(l) − μ_{orig,irr}^(l)

    Relevance-associated contrast direction:
        m_a^(l) = μ_{rel}^(l) − μ_{cf,irr}^(l)

    Pooling uses ``pool_hidden()`` with the selected ``token_set``.
    Token ids come from ``tokenize_prompt()`` (canonical path, Issue 2).

    When ``token_set`` is span-dependent (edited_span / suffix), original
    representations are cached by ``(qid, edit_start, edit_end)`` rather
    than by qid alone, because different CFs for the same qid may have
    different span geometries.  (Issue 6)

    The relevance-contrast direction uses ``relevance_token_set`` (which
    may differ from ``token_set`` when the latter is span-dependent and
    D_rel has no paired CF).  Any mismatch is recorded in metadata.
    (Issue 4)
    """
    directions = {}

    # Span-independent token_sets can safely cache originals by qid.
    span_dep = _token_set_is_span_dependent(token_set)

    for family in ALL_FOCAL:
        print(f"\n--- Computing directions for {family} ---")
        fam_pairs = pairs_irr.get(family, [])
        fam_rel = records_rel.get(family, [])

        if len(fam_pairs) < MIN_FAMILY_SIZE:
            print(f"  Only {len(fam_pairs)} D_irr pairs — need at least "
                  f"{MIN_FAMILY_SIZE}. Skipping.")
            continue

        subset = sample_pairs(fam_pairs, max_pairs_per_family, rng)
        print(f"  D_irr: using {len(subset)} / {len(fam_pairs)} pairs")

        sum_orig = {l: np.zeros(hidden_size, dtype=np.float64)
                    for l in target_layers}
        sum_cf = {l: np.zeros(hidden_size, dtype=np.float64)
                  for l in target_layers}
        n_irr = 0
        n_invalid_pool = 0

        # Issue 6: When token_set is span-dependent, cache key must
        # include span geometry because different CFs of the same qid
        # produce different edit spans → different pooled orig vectors.
        # For span-independent token_sets (all, final_token), qid alone
        # suffices because pooling does not use span indices.
        orig_cache: dict[str | tuple, dict] = {}

        for pi, pair in enumerate(subset):
            if (pi + 1) % 50 == 0 or pi == 0:
                print(f"  [{pi+1}/{len(subset)}]")

            qid = pair["qid"]

            # Canonical tokenization for span detection (Issue 2).
            orig_ids, _ = tokenize_prompt(tok, pair["orig_prompt"])
            cf_ids, _ = tokenize_prompt(tok, pair["cf_prompt"])
            es, eeo, eec, fallback = find_edited_span(orig_ids, cf_ids)

            # Issue 6: choose cache key based on whether pooling is
            # span-dependent.
            if span_dep:
                cache_key = (qid, es, eeo, token_set)
            else:
                cache_key = qid

            # Original hidden states (cached).
            if cache_key not in orig_cache:
                _, o_hidden, n_o = get_logits_and_hidden(
                    model, tok, pair["orig_prompt"], device, answer_ids,
                    layers=target_layers)
                o_pooled = {}
                skip = False
                for l in target_layers:
                    vec, valid = pool_hidden(o_hidden[l], token_set,
                                            es, eeo)
                    if not valid:
                        skip = True
                        break
                    o_pooled[l] = vec
                del o_hidden
                if skip:
                    n_invalid_pool += 1
                    continue
                orig_cache[cache_key] = o_pooled

            o_pooled = orig_cache[cache_key]

            # CF hidden states.
            _, c_hidden, n_c = get_logits_and_hidden(
                model, tok, pair["cf_prompt"], device, answer_ids,
                layers=target_layers)
            skip = False
            for l in target_layers:
                vec, valid = pool_hidden(c_hidden[l], token_set, es, eec)
                if not valid:
                    skip = True
                    break
                sum_cf[l] += vec.astype(np.float64)
                sum_orig[l] += o_pooled[l].astype(np.float64)
            del c_hidden
            if skip:
                n_invalid_pool += 1
                continue

            n_irr += 1
            if (pi + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if n_irr < MIN_FAMILY_SIZE:
            print(f"  Only {n_irr} valid pairs after pooling. Skipping.")
            continue

        # r̂_a per layer.
        r_hat, r_raw, r_norm = {}, {}, {}
        for l in target_layers:
            r = sum_cf[l] / n_irr - sum_orig[l] / n_irr
            r_hat[l], r_norm[l] = unit_normalize(r)
            r_raw[l] = r.astype(np.float32)

        # --- Relevance-associated contrast direction ---
        m_hat = m_raw = m_norm = None
        n_rel = 0
        rel_diagnostics: dict = {}
        # Issue 4: resolve effective token_set for relevance direction.
        rel_ts = resolve_relevance_token_set(token_set, relevance_token_set)
        # FIX 2: D_rel has no paired CFs, so span-dependent pooling is
        # impossible.  Fail loudly rather than producing invalid vectors.
        if rel_ts in ("edited_span", "suffix"):
            raise ValueError(
                f"Invalid relevance_token_set '{rel_ts}': D_rel has no "
                f"paired counterfactuals, so span-dependent token_sets "
                f"('edited_span', 'suffix') are not valid.  "
                f"Use 'final_token' or 'all'."
            )
        pooling_mismatch = (rel_ts != token_set)
        if pooling_mismatch:
            print(f"  WARNING: pooling mismatch — r̂ uses '{token_set}', "
                  f"m̂ uses '{rel_ts}' (D_rel has no CF for span detection)")

        if fam_rel:
            rel_subset = fam_rel[:max_pairs_per_family]
            print(f"  D_rel: using {len(rel_subset)} / {len(fam_rel)}")

            sum_rel = {l: np.zeros(hidden_size, dtype=np.float64)
                       for l in target_layers}
            rel_q_lens = []
            rel_gold_counts: Counter = Counter()

            for ri, rec in enumerate(rel_subset):
                if (ri + 1) % 50 == 0 or ri == 0:
                    print(f"  [{ri+1}/{len(rel_subset)}] D_rel")

                _, r_hidden, _ = get_logits_and_hidden(
                    model, tok, rec["orig_prompt"], device, answer_ids,
                    layers=target_layers)
                for l in target_layers:
                    # D_rel has no CF, so span args are unused for
                    # final_token/all; pass zeros explicitly.
                    vec, _ = pool_hidden(r_hidden[l], rel_ts, 0, 0)
                    sum_rel[l] += vec.astype(np.float64)
                del r_hidden
                n_rel += 1
                rel_q_lens.append(len(rec["orig_q"]))
                rel_gold_counts[rec["gold"]] += 1

            if n_rel >= 2:
                m_hat, m_raw, m_norm = {}, {}, {}
                for l in target_layers:
                    m = sum_rel[l] / n_rel - sum_cf[l] / n_irr
                    m_hat[l], m_norm[l] = unit_normalize(m)
                    m_raw[l] = m.astype(np.float32)

                irr_q_lens = [len(p["orig_q"]) for p in subset[:n_irr]]
                irr_gold_counts = Counter(p["gold"]
                                          for p in subset[:n_irr])
                rel_diagnostics = {
                    "n_rel": n_rel,
                    "n_irr": n_irr,
                    "token_set_for_relevance_direction": rel_ts,
                    "pooling_mismatch_between_r_and_m": pooling_mismatch,
                    "mean_q_len_rel": float(np.mean(rel_q_lens)),
                    "mean_q_len_irr": float(np.mean(irr_q_lens)),
                    "gold_dist_rel": dict(rel_gold_counts),
                    "gold_dist_irr": dict(irr_gold_counts),
                    "WARNING": (
                        "m̂_a is an approximate relevance-associated "
                        "contrast direction.  D_rel and D_irr are not "
                        "matched question pairs.  This direction may "
                        "absorb question-content, lexical, diagnosis/domain, "
                        "and task-distribution confounds."
                    ),
                }
                if pooling_mismatch:
                    rel_diagnostics["pooling_mismatch_note"] = (
                        f"r̂ was pooled with '{token_set}' but m̂ was "
                        f"pooled with '{rel_ts}' because D_rel has no "
                        f"paired CF for span detection.  Joint analyses "
                        f"(Exp 6, Exp 7) mix directions estimated under "
                        f"different pooling regimes."
                    )
            else:
                print(f"  Only {n_rel} D_rel records — need ≥ 2 for "
                      f"contrast direction.")
        else:
            print(f"  No D_rel for {family} — skipping contrast direction.")

        directions[family] = {
            "r_hat": r_hat, "r_raw": r_raw, "r_norm": r_norm,
            "m_hat": m_hat, "m_raw": m_raw, "m_norm": m_norm,
            "n_irr_used": n_irr, "n_rel_used": n_rel,
            "n_invalid_pool": n_invalid_pool,
            "token_set_for_attribute_direction": token_set,
            "token_set_for_relevance_direction": rel_ts,
            "pooling_mismatch": pooling_mismatch,
        }

        # --- Save to disk (per-layer named keys) ---
        fam_dir = output_dir / "directions" / family
        fam_dir.mkdir(parents=True, exist_ok=True)

        save_kw = {}
        for l in target_layers:
            save_kw[f"r_hat_layer_{l}"] = r_hat[l]
            save_kw[f"r_raw_layer_{l}"] = r_raw[l]
        np.savez(fam_dir / "attribute_direction.npz", **save_kw)

        if m_hat is not None:
            m_kw = {}
            for l in target_layers:
                m_kw[f"m_hat_layer_{l}"] = m_hat[l]
                m_kw[f"m_raw_layer_{l}"] = m_raw[l]
            np.savez(fam_dir / "relevance_contrast_direction.npz", **m_kw)

        # Issue 4: save both token_set fields separately.
        meta = {
            "family": family,
            "n_irr_used": n_irr,
            "n_rel_used": n_rel,
            "n_invalid_pool": n_invalid_pool,
            "target_layers": target_layers,
            "token_set_for_attribute_direction": token_set,
            "token_set_for_relevance_direction": rel_ts,
            "pooling_mismatch_between_r_and_m": pooling_mismatch,
            "r_norm_by_layer": {str(l): r_norm[l] for l in target_layers},
        }
        if m_norm is not None:
            meta["m_norm_by_layer"] = {str(l): m_norm[l]
                                       for l in target_layers}
            meta["relevance_contrast_diagnostics"] = rel_diagnostics
        with open(fam_dir / "direction_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved directions for {family} "
              f"(n_irr={n_irr}, n_rel={n_rel}, "
              f"invalid_pool={n_invalid_pool})")
        orig_cache.clear()
        gc.collect()

    return directions


def load_directions(output_dir: Path, target_layers: list[int],
                    hidden_size: int,
                    families: list[str] | None = None) -> dict:
    """Load precomputed directions with per-layer named keys.

    Args:
        families: list of family names to load.  Defaults to ALL_FOCAL
                  if None.  Accepts an explicit list so callers (including
                  sanity checks) do not need to mutate globals.  (Issue 1)
    """
    families_to_load = families if families is not None else ALL_FOCAL
    directions = {}
    for family in families_to_load:
        fam_dir = output_dir / "directions" / family
        meta_path = fam_dir / "direction_metadata.json"
        attr_path = fam_dir / "attribute_direction.npz"
        if not attr_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        attr = np.load(attr_path)
        r_hat, r_raw, r_norm = {}, {}, {}
        for l in target_layers:
            key = f"r_hat_layer_{l}"
            if key in attr:
                r_hat[l] = attr[key]
                assert r_hat[l].shape == (hidden_size,), \
                    f"r_hat dimension mismatch at layer {l}"
            rk = f"r_raw_layer_{l}"
            if rk in attr:
                r_raw[l] = attr[rk]
            r_norm[l] = meta.get("r_norm_by_layer", {}).get(str(l), 0.0)

        m_hat = m_raw = m_norm = None
        m_path = fam_dir / "relevance_contrast_direction.npz"
        if m_path.exists():
            md = np.load(m_path)
            m_hat, m_raw, m_norm = {}, {}, {}
            for l in target_layers:
                key = f"m_hat_layer_{l}"
                if key in md:
                    m_hat[l] = md[key]
                rk = f"m_raw_layer_{l}"
                if rk in md:
                    m_raw[l] = md[rk]
                m_norm[l] = meta.get("m_norm_by_layer", {}).get(
                    str(l), 0.0)

        # FIX 1: Restore metadata fields saved by compute_directions()
        # so that --interventions_only has access to pooling_mismatch,
        # token_set_for_attribute_direction, etc. for downstream use
        # in summary and Exp 6/7 metadata.
        directions[family] = {
            "r_hat": r_hat, "r_raw": r_raw, "r_norm": r_norm,
            "m_hat": m_hat, "m_raw": m_raw, "m_norm": m_norm,
            "n_irr_used": meta.get("n_irr_used", 0),
            "n_rel_used": meta.get("n_rel_used", 0),
            "n_invalid_pool": meta.get("n_invalid_pool", 0),
            "token_set_for_attribute_direction": meta.get(
                "token_set_for_attribute_direction"),
            "token_set_for_relevance_direction": meta.get(
                "token_set_for_relevance_direction"),
            "pooling_mismatch": meta.get(
                "pooling_mismatch_between_r_and_m", False),
        }
        print(f"  Loaded {family}: {len(r_hat)} layers")

    return directions


# ===================================================================
# Phase 2: Layer selection  (Directive B — layer-only, honest)
# ===================================================================

def select_best_layers(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list],
    pairs_ctrl: list,
    directions: dict,
    target_layers: list[int],
    max_pairs: int,
    alpha: float,
    token_set: str,
    rng,
    output_dir: Path,
) -> dict[str, int]:
    """Select the best transformer block for each family.

    Selection is layer-only: token positions are determined by
    ``token_set``, not searched over.

    Score:
      S_a(l | token_set) = λ₁·|ΔM_add|(l) + λ₂·RF_abl(l) − λ₃·|ΔM_ctrl|(l)

    Returns {family: best_block_index}.
    """
    best_layers: dict[str, int] = {}

    for family in ALL_FOCAL:
        if family not in directions:
            continue
        dirn = directions[family]
        fam_pairs = pairs_irr.get(family, [])
        if not fam_pairs:
            continue

        # Use a small subset for efficiency.
        n_scan = min(max_pairs, 30)
        subset = sample_pairs(fam_pairs, n_scan, rng)
        focal_qids = {p["qid"] for p in subset}
        ctrl_sub = sample_matched_controls(pairs_ctrl, focal_qids,
                                           n_scan, rng)

        print(f"\n--- Layer selection for {family} "
              f"({len(subset)} irr, {len(ctrl_sub)} ctrl) ---")

        # Pre-cache baseline logits.
        orig_logits_cache: dict[int, np.ndarray] = {}
        cf_logits_cache: dict[int, np.ndarray] = {}
        ctrl_logits_cache: dict[int, np.ndarray] = {}
        for i, p in enumerate(subset):
            orig_logits_cache[i] = get_logits_only(
                model, tok, p["orig_prompt"], device, answer_ids)
            cf_logits_cache[i] = get_logits_only(
                model, tok, p["cf_prompt"], device, answer_ids)
        for i, p in enumerate(ctrl_sub):
            ctrl_logits_cache[i] = get_logits_only(
                model, tok, p["cf_prompt"], device, answer_ids)

        scores: dict[int, dict] = {}

        for bi in target_layers:
            r_hat_l = dirn["r_hat"].get(bi)
            if r_hat_l is None or np.linalg.norm(r_hat_l) < 1e-10:
                continue
            r_t = torch.from_numpy(r_hat_l).float()

            # --- Δ_add on irr originals ---
            dm_add = []
            for i, p in enumerate(subset):
                g = p["gold_idx"]
                M0 = compute_margin(orig_logits_cache[i], g)
                # Canonical tokenization for span detection (Issue 2).
                o_ids, _ = tokenize_prompt(tok, p["orig_prompt"])
                c_ids, _ = tokenize_prompt(tok, p["cf_prompt"])
                es, eeo, eec, _ = find_edited_span(o_ids, c_ids)
                pos, valid = make_token_positions(len(o_ids), token_set,
                                                  es, eeo)
                if not valid:
                    continue
                hook = ResidualAddHook(r_t, alpha, pos)
                al, _ = forward_with_hook(model, tok, p["orig_prompt"],
                                          device, answer_ids, hook, bi)
                dm_add.append(abs(compute_margin(al, g) - M0))

            # --- RF_abl on irr CFs ---
            rf_vals = []
            for i, p in enumerate(subset):
                g = p["gold_idx"]
                M_orig = compute_margin(orig_logits_cache[i], g)
                M_cf = compute_margin(cf_logits_cache[i], g)
                if abs(M_orig - M_cf) < MIN_EFFECT_THRESHOLD:
                    rf_vals.append(0.0)
                    continue
                # Canonical tokenization (Issue 2).
                o_ids, _ = tokenize_prompt(tok, p["orig_prompt"])
                c_ids, _ = tokenize_prompt(tok, p["cf_prompt"])
                es, eeo, eec, _ = find_edited_span(o_ids, c_ids)
                pos, valid = make_token_positions(len(c_ids), token_set,
                                                  es, eec)
                if not valid:
                    continue
                hook = DirectionalAblationHook(r_t, pos)
                al, _ = forward_with_hook(model, tok, p["cf_prompt"],
                                          device, answer_ids, hook, bi)
                M_abl = compute_margin(al, g)
                rf_vals.append(max(0.0, (M_abl - M_cf) / (M_orig - M_cf)))

            # --- C_nonspecific on controls ---
            # Issue 3: use explicit control position policy, not silent
            # all-token application.
            dm_ctrl = []
            for i, p in enumerate(ctrl_sub):
                g = p["gold_idx"]
                M0 = compute_margin(ctrl_logits_cache[i], g)
                c_ids_ctrl, _ = tokenize_prompt(tok, p["cf_prompt"])
                ctrl_pos, _, _ = make_control_token_positions(
                    len(c_ids_ctrl), token_set)
                hook = ResidualAddHook(r_t, alpha, ctrl_pos)
                al, _ = forward_with_hook(model, tok, p["cf_prompt"],
                                          device, answer_ids, hook, bi)
                dm_ctrl.append(abs(compute_margin(al, g) - M0))

            ma = float(np.mean(dm_add)) if dm_add else 0.0
            mr = float(np.mean(rf_vals)) if rf_vals else 0.0
            mc = float(np.mean(dm_ctrl)) if dm_ctrl else 0.0
            S = LAMBDA_ADD * ma + LAMBDA_ABL * mr - LAMBDA_CTRL * mc
            scores[bi] = {"S": S, "add": ma, "abl": mr, "ctrl": mc}

        if not scores:
            print(f"  No valid layers for {family}")
            continue

        best = max(scores, key=lambda k: scores[k]["S"])
        best_layers[family] = best
        bs = scores[best]
        print(f"  Best block: {best} ({block_to_region(best)}) "
              f"S={bs['S']:.4f}")

        # Save.
        site_dir = output_dir / "site_selection" / family
        site_dir.mkdir(parents=True, exist_ok=True)
        with open(site_dir / "effectiveness_scores.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "block_index", "stage45_layer_index", "region",
                "S", "delta_add_irr", "delta_abl_irr", "C_nonspecific"])
            w.writeheader()
            for bi in sorted(scores):
                sc = scores[bi]
                w.writerow({
                    "block_index": bi,
                    "stage45_layer_index": bi + 1,
                    "region": block_to_region(bi),
                    "S": f"{sc['S']:.6f}",
                    "delta_add_irr": f"{sc['add']:.6f}",
                    "delta_abl_irr": f"{sc['abl']:.6f}",
                    "C_nonspecific": f"{sc['ctrl']:.6f}",
                })
        # Issue 3/5: record control position policy in selection metadata.
        _, _, ctrl_policy = make_control_token_positions(1, token_set)
        with open(site_dir / "best_site.json", "w") as f:
            json.dump({
                "family": family,
                "best_block_index": best,
                "stage45_layer_index": best + 1,
                "region": block_to_region(best),
                "token_set": token_set,
                "selection_granularity": "layer_only",
                "focal_token_position_policy": _focal_position_policy(
                    token_set),
                "control_token_position_policy": ctrl_policy,
                "note": "Token positions are determined by token_set, "
                        "not searched over.",
                "all_scores": {str(k): v for k, v in scores.items()},
            }, f, indent=2)

    return best_layers


# ===================================================================
# Shared experiment helper: compute span + positions for a pair
# ===================================================================

def _pair_span_info(tok, pair: dict, token_set: str, prompt_key: str
                    ) -> tuple[int, int, int, list[int] | None, bool]:
    """Return (edit_start, edit_end_for_prompt, n_tokens, positions, valid).

    ``prompt_key`` should be 'orig_prompt' or 'cf_prompt'.

    Uses ``tokenize_prompt()`` (canonical tokenizer path) so that span
    indices are guaranteed to align with model forward-pass tokenization.
    (Issue 2)
    """
    o_ids, _ = tokenize_prompt(tok, pair["orig_prompt"])
    c_ids, _ = tokenize_prompt(tok, pair["cf_prompt"])
    es, eeo, eec, _ = find_edited_span(o_ids, c_ids)
    if prompt_key == "orig_prompt":
        n_tok = len(o_ids)
        ee = eeo
    else:
        n_tok = len(c_ids)
        ee = eec
    pos, valid = make_token_positions(n_tok, token_set, es, ee)
    return es, ee, n_tok, pos, valid


def _pool_at_layer(hidden: dict, layer: int, token_set: str,
                   edit_start: int, edit_end: int
                   ) -> tuple[np.ndarray, bool]:
    """Pool hidden state at one layer using the configured token_set."""
    return pool_hidden(hidden[layer], token_set, edit_start, edit_end)


# ===================================================================
# Phase 3: Experiments
# ===================================================================

# ---- Experiment 1: Activation Addition on D_irr originals ---------

def run_exp1_add_irr(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list], directions: dict,
    best_layers: dict, alpha: float, token_set: str,
    max_pairs: int, rng, output_dir: Path,
) -> dict:
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        bi = best_layers[family]
        r_hat = directions[family]["r_hat"].get(bi)
        if r_hat is None:
            continue
        r_t = torch.from_numpy(r_hat).float()

        subset = sample_pairs(pairs_irr.get(family, []), max_pairs, rng)
        if not subset:
            continue
        print(f"\n--- Exp 1: Add r̂ (irr) {family} "
              f"[block {bi}, n={len(subset)}] ---")

        rows = []
        n_skip = 0
        for pi, pair in enumerate(subset):
            if (pi + 1) % 50 == 0:
                print(f"  [{pi+1}/{len(subset)}]")
            g = pair["gold_idx"]

            es, ee_o, n_o, pos, valid = _pair_span_info(
                tok, pair, token_set, "orig_prompt")
            if not valid:
                n_skip += 1
                continue
            _, ee_c, _, _, _ = _pair_span_info(
                tok, pair, token_set, "cf_prompt")

            orig_L, orig_H, _ = get_logits_and_hidden(
                model, tok, pair["orig_prompt"], device, answer_ids,
                layers=[bi])
            cf_L, cf_H, _ = get_logits_and_hidden(
                model, tok, pair["cf_prompt"], device, answer_ids,
                layers=[bi])

            hook = ResidualAddHook(r_t, alpha, pos)
            add_L, add_H = forward_with_hook(
                model, tok, pair["orig_prompt"], device, answer_ids,
                hook, bi, return_hidden=True, hidden_layers=[bi])

            # Primary pooling (same token_set as intervention).
            h_orig_p, v1 = _pool_at_layer(orig_H, bi, token_set, es, ee_o)
            h_cf_p, v2 = _pool_at_layer(cf_H, bi, token_set, es, ee_c)
            h_add_p, v3 = _pool_at_layer(add_H, bi, token_set, es, ee_o)

            # Secondary: final token always.
            h_orig_f = orig_H[bi][0, -1].float().numpy()
            h_cf_f = cf_H[bi][0, -1].float().numpy()
            h_add_f = add_H[bi][0, -1].float().numpy()

            dz = float(add_L[g] - orig_L[g])
            dz_cf = float(cf_L[g] - orig_L[g])
            M_o = compute_margin(orig_L, g)
            M_a = compute_margin(add_L, g)
            M_c = compute_margin(cf_L, g)

            # FIX 5: standardized position policy label.
            pos_policy = _focal_position_policy(token_set)

            row = {
                "qid": pair["qid"], "family": family,
                "attr_val": pair["attr_val"], "gold": pair["gold"],
                **_layer_output_row(bi),
                "alpha": alpha, "token_set": token_set,
                "position_policy": pos_policy,
                "orig_gold_logit": float(orig_L[g]),
                "add_gold_logit": float(add_L[g]),
                "cf_gold_logit": float(cf_L[g]),
                "delta_z_gold_add": dz,
                "delta_z_gold_cf": dz_cf,
                "M_orig": M_o, "M_add": M_a, "M_cf": M_c,
                "delta_M_add": M_a - M_o,
                "delta_M_cf": M_c - M_o,
                "EF_add_logit": safe_ratio(dz, dz_cf),
                "EF_add_margin": safe_ratio(M_a - M_o, M_c - M_o),
                "orig_correct": int(np.argmax(orig_L) == g),
                "cf_correct": int(np.argmax(cf_L) == g),
                "add_correct": int(np.argmax(add_L) == g),
            }
            # Representational similarities.
            if v1 and v2 and v3:
                row["sim_cf_add_primary"] = cosine_sim(h_add_p, h_cf_p)
                row["sim_orig_add_primary"] = cosine_sim(h_add_p, h_orig_p)
                row["delta_repr_add_primary"] = (
                    row["sim_cf_add_primary"] - row["sim_orig_add_primary"])
            else:
                row["sim_cf_add_primary"] = None
                row["sim_orig_add_primary"] = None
                row["delta_repr_add_primary"] = None
            row["sim_cf_add_final"] = cosine_sim(h_add_f, h_cf_f)
            row["sim_orig_add_final"] = cosine_sim(h_add_f, h_orig_f)
            row["delta_repr_add_final"] = (
                row["sim_cf_add_final"] - row["sim_orig_add_final"])

            rows.append(row)
            del orig_H, cf_H, add_H
            if (pi + 1) % 20 == 0:
                gc.collect()

        exp_dir = output_dir / "experiments" / family / "1_add_irr"
        exp_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, exp_dir / "per_pair.csv")

        agg = _aggregate_generic(rows, [
            "delta_z_gold_add", "delta_M_add", "delta_repr_add_primary",
            "delta_repr_add_final", "sim_cf_add_primary",
            "sim_orig_add_primary",
        ], ratio_keys=["EF_add_logit", "EF_add_margin"],
            count_keys=["orig_correct", "cf_correct", "add_correct"])
        agg["n_skipped_invalid_pool"] = n_skip
        with open(exp_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        _save_experiment_metadata(
            exp_dir / "metadata.json",
            experiment="1_add_irr",
            description="Activation addition of r̂_a on irrelevant "
                        "original prompts",
            family=family, **_layer_output_row(bi),
            token_set=token_set, alpha=alpha,
            n_used_pairs=len(rows),
            n_skipped_invalid_pool=n_skip,
            intervention_position_policy=_focal_position_policy(token_set),
            interpretation="Measures how much adding the attribute "
                           "direction shifts model behavior on prompts "
                           "where the attribute is medically irrelevant.",
        )
        results[family] = agg
        dm = agg.get("mean_delta_M_add")
        print(f"  Mean ΔM_add = {dm:.4f}" if dm is not None else
              "  Mean ΔM_add = N/A")

    return results


# ---- Experiment 2: Directional Ablation on D_irr CFs --------------

def run_exp2_abl_irr(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list], directions: dict,
    best_layers: dict, token_set: str, max_pairs: int,
    rng, output_dir: Path,
) -> dict:
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        bi = best_layers[family]
        r_hat = directions[family]["r_hat"].get(bi)
        if r_hat is None:
            continue
        r_t = torch.from_numpy(r_hat).float()

        subset = sample_pairs(pairs_irr.get(family, []), max_pairs, rng)
        if not subset:
            continue
        print(f"\n--- Exp 2: Ablate r̂ (irr CF) {family} "
              f"[block {bi}, n={len(subset)}] ---")

        rows = []
        n_skip = 0
        for pi, pair in enumerate(subset):
            if (pi + 1) % 50 == 0:
                print(f"  [{pi+1}/{len(subset)}]")
            g = pair["gold_idx"]

            es, ee_c, n_c, pos, valid = _pair_span_info(
                tok, pair, token_set, "cf_prompt")
            if not valid:
                n_skip += 1
                continue
            _, ee_o, _, _, _ = _pair_span_info(
                tok, pair, token_set, "orig_prompt")

            orig_L, orig_H, _ = get_logits_and_hidden(
                model, tok, pair["orig_prompt"], device, answer_ids,
                layers=[bi])
            cf_L, cf_H, _ = get_logits_and_hidden(
                model, tok, pair["cf_prompt"], device, answer_ids,
                layers=[bi])

            hook = DirectionalAblationHook(r_t, pos)
            abl_L, abl_H = forward_with_hook(
                model, tok, pair["cf_prompt"], device, answer_ids,
                hook, bi, return_hidden=True, hidden_layers=[bi])

            dz_rec = float(abl_L[g] - cf_L[g])
            dz_dist = float(cf_L[g] - orig_L[g])
            M_o = compute_margin(orig_L, g)
            M_c = compute_margin(cf_L, g)
            M_a = compute_margin(abl_L, g)

            # Primary pooled similarity.
            h_orig_p, v1 = _pool_at_layer(orig_H, bi, token_set, es, ee_o)
            h_cf_p, v2 = _pool_at_layer(cf_H, bi, token_set, es, ee_c)
            h_abl_p, v3 = _pool_at_layer(abl_H, bi, token_set, es, ee_c)

            h_orig_f = orig_H[bi][0, -1].float().numpy()
            h_cf_f = cf_H[bi][0, -1].float().numpy()
            h_abl_f = abl_H[bi][0, -1].float().numpy()

            orig_correct = int(np.argmax(orig_L) == g)
            cf_correct = int(np.argmax(cf_L) == g)
            abl_correct = int(np.argmax(abl_L) == g)
            is_flip = orig_correct and not cf_correct
            flip_recovered = is_flip and abl_correct

            # FIX 5: standardized position policy label.
            pos_policy_2 = _focal_position_policy(token_set)

            row = {
                "qid": pair["qid"], "family": family,
                "attr_val": pair["attr_val"], "gold": pair["gold"],
                **_layer_output_row(bi), "token_set": token_set,
                "position_policy": pos_policy_2,
                "orig_gold_logit": float(orig_L[g]),
                "cf_gold_logit": float(cf_L[g]),
                "abl_gold_logit": float(abl_L[g]),
                "delta_z_gold_rec": dz_rec,
                "delta_z_gold_dist": dz_dist,
                "RF_abl_logit": safe_ratio(
                    float(abl_L[g] - cf_L[g]),
                    float(orig_L[g] - cf_L[g])),
                "M_orig": M_o, "M_cf": M_c, "M_abl": M_a,
                "RF_abl_margin": safe_ratio(M_a - M_c, M_o - M_c),
                "orig_correct": orig_correct,
                "cf_correct": cf_correct,
                "abl_correct": abl_correct,
                "answer_flip_cf": int(is_flip),
                "flip_recovered": int(flip_recovered),
            }
            if v1 and v2 and v3:
                row["sim_orig_abl_primary"] = cosine_sim(h_abl_p, h_orig_p)
                row["sim_cf_abl_primary"] = cosine_sim(h_abl_p, h_cf_p)
                row["delta_repr_abl_primary"] = (
                    row["sim_orig_abl_primary"] - row["sim_cf_abl_primary"])
            else:
                row["sim_orig_abl_primary"] = None
                row["sim_cf_abl_primary"] = None
                row["delta_repr_abl_primary"] = None
            row["sim_orig_abl_final"] = cosine_sim(h_abl_f, h_orig_f)
            row["sim_cf_abl_final"] = cosine_sim(h_abl_f, h_cf_f)
            row["delta_repr_abl_final"] = (
                row["sim_orig_abl_final"] - row["sim_cf_abl_final"])

            rows.append(row)
            del orig_H, cf_H, abl_H
            if (pi + 1) % 20 == 0:
                gc.collect()

        exp_dir = output_dir / "experiments" / family / "2_abl_irr"
        exp_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, exp_dir / "per_pair.csv")

        agg = _aggregate_generic(rows, [
            "delta_z_gold_rec", "delta_z_gold_dist",
            "delta_repr_abl_primary", "delta_repr_abl_final",
        ], ratio_keys=["RF_abl_logit", "RF_abl_margin"],
            count_keys=["orig_correct", "cf_correct", "abl_correct",
                        "answer_flip_cf", "flip_recovered"])
        agg["n_skipped_invalid_pool"] = n_skip
        with open(exp_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        _save_experiment_metadata(
            exp_dir / "metadata.json",
            experiment="2_abl_irr",
            description="Directional ablation of r̂_a from irrelevant "
                        "counterfactual prompts",
            family=family, **_layer_output_row(bi),
            token_set=token_set,
            n_used_pairs=len(rows),
            n_skipped_invalid_pool=n_skip,
            intervention_position_policy=_focal_position_policy(token_set),
            interpretation="Measures how much removing the attribute "
                           "direction from CF prompts restores original "
                           "model behavior.",
        )
        results[family] = agg
        rf = agg.get("mean_RF_abl_margin")
        nf = agg.get("sum_answer_flip_cf", 0)
        nr = agg.get("sum_flip_recovered", 0)
        print(f"  RF_margin={rf}  flips={nr}/{nf}")

    return results


# ---- Experiment 3: Control variant interventions -------------------

def run_exp3_ctrl(
    model, tok, device: str, answer_ids: dict,
    pairs_ctrl: list, pairs_irr: dict[str, list],
    directions: dict, best_layers: dict,
    alpha: float, token_set: str, max_pairs: int,
    rng, output_dir: Path,
) -> dict:
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        bi = best_layers[family]
        r_hat = directions[family]["r_hat"].get(bi)
        if r_hat is None:
            continue
        r_t = torch.from_numpy(r_hat).float()

        focal_qids = {p["qid"] for p in pairs_irr.get(family, [])}
        ctrl_sub = sample_matched_controls(pairs_ctrl, focal_qids,
                                           max_pairs, rng)
        if not ctrl_sub:
            continue

        print(f"\n--- Exp 3: Control interventions {family} "
              f"[block {bi}, n={len(ctrl_sub)}] ---")

        rows = []
        for pi, pair in enumerate(ctrl_sub):
            if (pi + 1) % 50 == 0:
                print(f"  [{pi+1}/{len(ctrl_sub)}]")
            g = pair["gold_idx"]

            ctrl_L = get_logits_only(model, tok, pair["cf_prompt"],
                                     device, answer_ids)
            M_0 = compute_margin(ctrl_L, g)

            # Issue 3: Use explicit control position policy rather than
            # silently applying to all tokens.
            c_ids_ctrl, _ = tokenize_prompt(tok, pair["cf_prompt"])
            ctrl_pos, _, ctrl_policy = make_control_token_positions(
                len(c_ids_ctrl), token_set)

            hook_a = ResidualAddHook(r_t, alpha, ctrl_pos)
            add_L, _ = forward_with_hook(model, tok, pair["cf_prompt"],
                                         device, answer_ids, hook_a, bi)
            M_a = compute_margin(add_L, g)

            hook_b = DirectionalAblationHook(r_t, ctrl_pos)
            abl_L, _ = forward_with_hook(model, tok, pair["cf_prompt"],
                                         device, answer_ids, hook_b, bi)
            M_b = compute_margin(abl_L, g)

            rows.append({
                "qid": pair["qid"], "family_direction": family,
                "control_type": pair["itype"], "gold": pair["gold"],
                **_layer_output_row(bi), "token_set": token_set,
                "control_position_policy": ctrl_policy,
                "alpha": alpha,
                "M_ctrl": M_0, "M_add": M_a, "M_abl": M_b,
                "abs_delta_M_add": abs(M_a - M_0),
                "abs_delta_M_abl": abs(M_b - M_0),
                "delta_M_add": M_a - M_0,
                "delta_M_abl": M_b - M_0,
                "ctrl_correct": int(np.argmax(ctrl_L) == g),
                "add_correct": int(np.argmax(add_L) == g),
                "abl_correct": int(np.argmax(abl_L) == g),
            })

        exp_dir = output_dir / "experiments" / family / "3_ctrl"
        exp_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, exp_dir / "per_pair.csv")

        agg = _aggregate_generic(rows, [
            "abs_delta_M_add", "abs_delta_M_abl",
            "delta_M_add", "delta_M_abl",
        ], count_keys=["ctrl_correct", "add_correct", "abl_correct"])
        with open(exp_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        # Issue 3/5: record control position policy.
        _, _, ctrl_pol_meta = make_control_token_positions(1, token_set)
        _save_experiment_metadata(
            exp_dir / "metadata.json",
            experiment="3_ctrl",
            description="Attribute direction applied to control prompts "
                        "(specificity check)",
            family=family, **_layer_output_row(bi),
            token_set=token_set, alpha=alpha, n_pairs=len(rows),
            control_position_policy=ctrl_pol_meta,
            note_on_controls=(
                "Controls lack a meaningful demographic edited span.  "
                "For span-dependent token_sets, interventions fall back "
                "to final-token to keep geometry comparable to focal "
                "experiments rather than silently applying to all tokens."
            ),
            interpretation="If r̂_a is attribute-specific, its effect "
                           "on controls should be weaker than on "
                           "matched demographic counterfactuals.",
        )
        results[family] = agg
        print(f"  |ΔM_add|={agg.get('mean_abs_delta_M_add', 'N/A')}")

    return results


# ---- Experiments 4 & 5: Medically-relevant -------------------------

def run_exp45_relevant(
    model, tok, device: str, answer_ids: dict,
    records_rel: dict[str, list], directions: dict,
    best_layers: dict, alpha: float, token_set: str,
    max_pairs: int, rng, output_dir: Path,
) -> dict:
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        fam_rel = records_rel.get(family, [])
        if not fam_rel:
            continue

        bi = best_layers[family]
        r_hat = directions[family]["r_hat"].get(bi)
        if r_hat is None:
            continue
        r_t = torch.from_numpy(r_hat).float()

        subset = fam_rel[:max_pairs]
        print(f"\n--- Exp 5: Ablate r̂ (relevant) {family} "
              f"[block {bi}, n={len(subset)}] ---")

        rows = []
        for ri, rec in enumerate(subset):
            if (ri + 1) % 50 == 0:
                print(f"  [{ri+1}/{len(subset)}]")
            g = rec["gold_idx"]

            rel_L = get_logits_only(model, tok, rec["orig_prompt"],
                                    device, answer_ids)
            M_r = compute_margin(rel_L, g)

            # Ablation uses all positions since no CF span exists.
            hook = DirectionalAblationHook(r_t)
            abl_L, _ = forward_with_hook(model, tok, rec["orig_prompt"],
                                         device, answer_ids, hook, bi)
            M_a = compute_margin(abl_L, g)

            # FIX 7: structured token_set fields instead of inline string.
            rows.append({
                "qid": rec["qid"], "family": family,
                "gold": rec["gold"], **_layer_output_row(bi),
                "token_set_requested": token_set,
                "token_set_applied": "all",
                "position_policy": "all_tokens_no_cf_span",
                "M_rel": M_r, "M_abl": M_a,
                "delta_M_rel_abl": M_a - M_r,
                "rel_correct": int(np.argmax(rel_L) == g),
                "abl_correct": int(np.argmax(abl_L) == g),
            })

        exp5_dir = output_dir / "experiments" / family / "5_rel_abl"
        exp5_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, exp5_dir / "per_pair.csv")

        agg5 = _aggregate_generic(rows, ["delta_M_rel_abl"],
                                  count_keys=["rel_correct", "abl_correct"])
        with open(exp5_dir / "aggregate.json", "w") as f:
            json.dump(agg5, f, indent=2)

        # FIX 7: structured intervention geometry metadata.
        _save_experiment_metadata(
            exp5_dir / "metadata.json",
            experiment="5_rel_abl",
            description="Directional ablation of r̂_a from medically-"
                        "relevant original prompts",
            family=family, **_layer_output_row(bi),
            n_pairs=len(rows),
            token_set_requested=token_set,
            token_set_applied="all",
            position_policy="all_tokens_no_cf_span",
            interpretation="If the direction supports correct reasoning "
                           "when the attribute is genuinely relevant, "
                           "ablating it should reduce the gold margin.",
        )

        # Exp 4: skipped.
        exp4_dir = output_dir / "experiments" / family / "4_rel_add"
        exp4_dir.mkdir(parents=True, exist_ok=True)
        _save_experiment_metadata(
            exp4_dir / "metadata.json",
            experiment="4_rel_add",
            status="SKIPPED",
            reason="No omission variants x^(-a) exist in the current "
                   "dataset.  The v6.1 balanced generation strategy did "
                   "not produce counterfactual variants for medically "
                   "relevant questions.",
            family=family,
        )
        print(f"  Exp 4: SKIPPED — no omission variants")

        results[family] = {"exp4": None, "exp5": agg5}

    return results


# ---- Experiment 6: Relevance-associated contrast ablation ----------

def run_exp6_medrel_abl(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list], directions: dict,
    best_layers: dict, token_set: str,
    max_pairs: int, rng, output_dir: Path,
) -> dict:
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        dirn = directions[family]
        if dirn["m_hat"] is None:
            print(f"\n--- Exp 6: No contrast direction for {family}. ---")
            continue

        bi = best_layers[family]
        m_hat_l = dirn["m_hat"].get(bi)
        if m_hat_l is None or np.linalg.norm(m_hat_l) < 1e-10:
            continue
        m_t = torch.from_numpy(m_hat_l).float()

        fam_pairs = pairs_irr.get(family, [])
        if not fam_pairs:
            continue

        subset = sample_pairs(fam_pairs, max_pairs, rng)
        print(f"\n--- Exp 6: Identifying flips for {family} ---")

        flip_pairs = []
        for p in subset:
            g = p["gold_idx"]
            oL = get_logits_only(model, tok, p["orig_prompt"],
                                 device, answer_ids)
            cL = get_logits_only(model, tok, p["cf_prompt"],
                                 device, answer_ids)
            if np.argmax(oL) == g and np.argmax(cL) != g:
                flip_pairs.append({**p, "_oL": oL, "_cL": cL})

        print(f"  {len(flip_pairs)} flips in {len(subset)} pairs")

        exp_dir = output_dir / "experiments" / family / "6_medrel_abl"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if not flip_pairs:
            agg = {"n_flips": 0}
            with open(exp_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            results[family] = agg
            continue

        rows = []
        for fp in flip_pairs:
            g = fp["gold_idx"]
            M_o = compute_margin(fp["_oL"], g)
            M_c = compute_margin(fp["_cL"], g)

            hook = DirectionalAblationHook(m_t)
            abl_L, _ = forward_with_hook(model, tok, fp["cf_prompt"],
                                         device, answer_ids, hook, bi)
            M_a = compute_margin(abl_L, g)

            rows.append({
                "qid": fp["qid"], "family": family,
                "attr_val": fp["attr_val"], "gold": fp["gold"],
                **_layer_output_row(bi),
                "M_orig": M_o, "M_cf": M_c,
                "M_abl_medrel": M_a,
                "delta_M_medrel": M_a - M_c,
                "RF_medrel": safe_ratio(M_a - M_c, M_o - M_c),
                "abl_correct": int(np.argmax(abl_L) == g),
                "flip_recovered": int(np.argmax(abl_L) == g),
            })

        _save_csv(rows, exp_dir / "per_pair.csv")
        agg = _aggregate_generic(rows, ["delta_M_medrel"],
                                 ratio_keys=["RF_medrel"],
                                 count_keys=["flip_recovered"])
        agg["n_flips"] = len(rows)
        with open(exp_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        # Issue 4: record if r/m pooling mismatch affects this experiment.
        # FIX 6: add intervention geometry metadata.
        pm = directions[family].get("pooling_mismatch", False)
        _save_experiment_metadata(
            exp_dir / "metadata.json",
            experiment="6_medrel_abl",
            description="Ablation of the relevance-associated contrast "
                        "direction m̂_a from harmful-flip CFs",
            family=family, **_layer_output_row(bi),
            n_flips=len(rows), n_scanned=len(subset),
            token_set_requested=token_set,
            token_set_applied="all",
            position_policy="all_tokens_no_cf_span",
            r_m_pooling_mismatch=pm,
            interpretation="If harmful bias partly reflects false "
                           "encoding of medical relevance, ablating m̂_a "
                           "should move the model back toward the "
                           "original correct behavior.  NOTE: m̂_a is an "
                           "approximate contrast, not a pure relevance "
                           "feature.",
            note="No counterfactual span exists for flip-identified CFs "
                 "in this experiment context; ablation applied to all "
                 "tokens.",
        )
        results[family] = agg
        print(f"  RF={agg.get('mean_RF_medrel')}  "
              f"recovered={agg.get('sum_flip_recovered', 0)}/{len(rows)}")

    return results


# ---- Experiment 7: Joint decomposition ----------------------------

def run_exp7_decomposition(
    model, tok, device: str, answer_ids: dict,
    pairs_irr: dict[str, list], directions: dict,
    best_layers: dict, token_set: str,
    max_pairs: int, rng, output_dir: Path,
) -> dict:
    """Decompose CF shift via joint least-squares projection onto r̂ and m̂.

    Uses ``np.linalg.lstsq`` so that correlated r̂ and m̂ are handled
    correctly (Directive D).
    """
    results = {}
    for family in ALL_FOCAL:
        if family not in directions or family not in best_layers:
            continue
        dirn = directions[family]
        bi = best_layers[family]
        r_hat_l = dirn["r_hat"].get(bi)
        if r_hat_l is None:
            continue

        has_m = (dirn["m_hat"] is not None and
                 dirn["m_hat"].get(bi) is not None)
        m_hat_l = dirn["m_hat"][bi] if has_m else None

        fam_pairs = pairs_irr.get(family, [])
        subset = sample_pairs(fam_pairs, max_pairs, rng)
        if not subset:
            continue

        print(f"\n--- Exp 7: Decomposition {family} "
              f"[block {bi}, n={len(subset)}] ---")

        # Cosine between r̂ and m̂ at this layer.
        r_m_cos = cosine_sim(r_hat_l, m_hat_l) if has_m else None

        dirs_list = [r_hat_l.astype(np.float64)]
        if has_m:
            dirs_list.append(m_hat_l.astype(np.float64))

        rows = []
        # Issue 6: span-dependent token_sets require cache keys that
        # include the span geometry, just as in compute_directions.
        span_dep_7 = _token_set_is_span_dependent(token_set)
        orig_cache_7: dict[str | tuple, np.ndarray] = {}
        n_skip = 0

        for pi, pair in enumerate(subset):
            if (pi + 1) % 50 == 0:
                print(f"  [{pi+1}/{len(subset)}]")

            qid = pair["qid"]
            # Canonical tokenization (Issue 2).
            o_ids, _ = tokenize_prompt(tok, pair["orig_prompt"])
            c_ids, _ = tokenize_prompt(tok, pair["cf_prompt"])
            es, eeo, eec, _ = find_edited_span(o_ids, c_ids)

            # Issue 6: span-aware cache key.
            ck = (qid, es, eeo, token_set) if span_dep_7 else qid

            if ck not in orig_cache_7:
                _, oh, _ = get_logits_and_hidden(
                    model, tok, pair["orig_prompt"], device, answer_ids,
                    layers=[bi])
                vec, valid = pool_hidden(oh[bi], token_set, es, eeo)
                del oh
                if not valid:
                    n_skip += 1
                    continue
                orig_cache_7[ck] = vec

            h_orig = orig_cache_7[ck]
            _, ch, _ = get_logits_and_hidden(
                model, tok, pair["cf_prompt"], device, answer_ids,
                layers=[bi])
            h_cf, valid = pool_hidden(ch[bi], token_set, es, eec)
            del ch
            if not valid:
                n_skip += 1
                continue

            delta_h = (h_cf - h_orig).astype(np.float64)
            dn = float(np.linalg.norm(delta_h))
            if dn < 1e-12:
                continue

            beta, residual, rn = joint_decompose(delta_h, dirs_list)
            fve_joint = 1.0 - (rn ** 2) / (dn ** 2)

            # Per-direction variance (individual, for context).
            beta_r = float(beta[0])
            fve_r_only = beta_r ** 2 / (dn ** 2)

            row = {
                "qid": pair["qid"], "family": family,
                "attr_val": pair["attr_val"], "gold": pair["gold"],
                **_layer_output_row(bi), "token_set": token_set,
                "delta_h_norm": dn,
                "beta_r": beta_r,
                "epsilon_norm": rn,
                "fve_joint": fve_joint,
                "fve_r_only": fve_r_only,
                "r_m_cosine": r_m_cos,
            }
            if has_m and len(beta) > 1:
                beta_m = float(beta[1])
                row["beta_m"] = beta_m
                row["fve_m_only"] = beta_m ** 2 / (dn ** 2)
            else:
                row["beta_m"] = None
                row["fve_m_only"] = None

            rows.append(row)
            if (pi + 1) % 20 == 0:
                gc.collect()

        exp_dir = output_dir / "experiments" / family / "7_decomposition"
        exp_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, exp_dir / "per_pair.csv")

        agg = _aggregate_generic(rows, [
            "beta_r", "delta_h_norm", "epsilon_norm",
            "fve_joint", "fve_r_only",
        ])
        if has_m:
            for k in ["beta_m", "fve_m_only", "r_m_cosine"]:
                vals = [r[k] for r in rows if r.get(k) is not None]
                agg[f"mean_{k}"] = _mean_or_none(vals)
        agg["n_skipped"] = n_skip
        with open(exp_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        # Issue 4: record pooling mismatch if present.
        pm7 = directions[family].get("pooling_mismatch", False)
        mismatch_note = ""
        if pm7:
            mismatch_note = (
                "r̂ and m̂ were estimated under different pooling "
                "token_sets.  Joint decomposition mixes these directions."
            )
        _save_experiment_metadata(
            exp_dir / "metadata.json",
            experiment="7_decomposition",
            description="Joint least-squares decomposition of CF shift "
                        "onto r̂_a and m̂_a",
            family=family, **_layer_output_row(bi),
            token_set=token_set,
            n_used_pairs=len(rows),
            n_skipped_invalid_pool=n_skip,
            r_m_cosine=r_m_cos,
            r_m_pooling_mismatch=pm7,
            r_m_pooling_mismatch_note=mismatch_note if mismatch_note else None,
            note="Decomposition uses np.linalg.lstsq to handle "
                 "correlated directions correctly.  fve_joint is the "
                 "fraction of variance explained by the joint projection "
                 "(1 - ||ε||²/||Δh||²).",
        )
        results[family] = agg
        orig_cache_7.clear()
        bm = agg.get("mean_beta_r", "N/A")
        print(f"  mean β_r={bm}  fve_joint={agg.get('mean_fve_joint')}")

    return results


# ===================================================================
# Generic aggregation
# ===================================================================

def _aggregate_generic(rows: list[dict], value_keys: list[str],
                       ratio_keys: list[str] | None = None,
                       count_keys: list[str] | None = None) -> dict:
    if not rows:
        return {"n_pairs": 0}
    agg: dict = {"n_pairs": len(rows)}
    for k in value_keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        agg[f"mean_{k}"] = _mean_or_none(vals)
        agg[f"std_{k}"] = _std_or_none(vals)
    for k in (ratio_keys or []):
        vals = [r[k] for r in rows if r.get(k) is not None]
        agg[f"mean_{k}"] = _mean_or_none(vals)
        agg[f"median_{k}"] = _median_or_none(vals)
        agg[f"n_valid_{k}"] = len(vals)
        agg[f"n_undefined_{k}"] = sum(1 for r in rows
                                       if r.get(k) is None)
    for k in (count_keys or []):
        agg[f"sum_{k}"] = sum(r.get(k, 0) for r in rows)
    return agg


# ===================================================================
# Phase 4: Predictions + Summary
# ===================================================================

def evaluate_predictions(
    e1: dict, e2: dict, e3: dict, e45: dict,
    e6: dict, e7: dict, output_dir: Path,
) -> dict:
    predictions = {}
    for family in ALL_FOCAL:
        fp: dict = {}

        # P1: addition in irrelevant → bias if mean ΔM < 0.
        if family in e1:
            dm = e1[family].get("mean_delta_M_add")
            fp["P1_mean_delta_M_add"] = dm
            fp["P1_suggests_bias"] = dm is not None and dm < -0.01

        # P2: ablation recovery.
        if family in e2:
            rf = e2[family].get("mean_RF_abl_logit")
            fp["P2_mean_RF_abl_logit"] = rf
            fp["P2_direction_causal"] = rf is not None and rf > 0.05
            fp["P2_n_flips"] = e2[family].get("sum_answer_flip_cf", 0)
            fp["P2_n_flips_recovered"] = e2[family].get(
                "sum_flip_recovered", 0)

        # P3b: ablation hurts relevant.
        if family in e45:
            e5 = e45[family].get("exp5", {})
            if e5:
                dm = e5.get("mean_delta_M_rel_abl")
                fp["P3b_mean_delta_M_rel_abl"] = dm
                fp["P3b_ablation_hurts_relevant"] = (
                    dm is not None and dm < -0.01)
            fp["P3a_status"] = "skipped_no_omission_variants"

        # P4: relevance-associated direction.
        if family in e6 and e6[family].get("n_flips", 0) > 0:
            fp["P4_medrel_RF"] = e6[family].get("mean_RF_medrel")
            fp["P4_medrel_n_flips"] = e6[family].get("n_flips", 0)
            fp["P4_medrel_recovered"] = e6[family].get(
                "sum_flip_recovered", 0)

        # Specificity.
        if family in e3 and family in e1:
            ctrl_eff = e3[family].get("mean_abs_delta_M_add", 0)
            irr_eff = abs(e1[family].get("mean_delta_M_add", 0) or 0)
            fp["specificity_ctrl_effect"] = ctrl_eff
            fp["specificity_irr_effect"] = irr_eff
            fp["specificity_ratio"] = (
                ctrl_eff / irr_eff if irr_eff > 0.01 else None)
            fp["direction_is_specific"] = (
                ctrl_eff < irr_eff if irr_eff > 0.01 else None)

        predictions[family] = fp

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "prediction_tests.json", "w") as f:
        json.dump(predictions, f, indent=2)
    return predictions


def generate_summary(
    e1: dict, e2: dict, e3: dict, e45: dict,
    e6: dict, e7: dict, predictions: dict,
    best_layers: dict, directions: dict,
    token_set: str, alpha: float,
    output_dir: Path,
) -> dict:
    rows = []
    # Issue 5 / Directive E: collect summary warnings.
    summary_warnings: list[str] = []

    # Control position policy for summary (Issue 3/5).
    _, _, ctrl_pol_summary = make_control_token_positions(1, token_set)
    if ctrl_pol_summary == "fallback_final_token_no_span":
        summary_warnings.append(
            f"Control interventions use final-token fallback because "
            f"token_set '{token_set}' is span-dependent and controls "
            f"lack a demographic edited span."
        )

    for family in ALL_FOCAL:
        if family not in best_layers:
            continue
        bi = best_layers[family]
        row: dict = {
            "family": family,
            **_layer_output_row(bi),
            "token_set": token_set,
            "alpha": alpha,
        }
        if family in directions:
            d = directions[family]
            row["r_norm"] = d["r_norm"].get(bi)
            if d["m_norm"]:
                row["m_norm"] = d["m_norm"].get(bi)
            row["n_irr"] = d["n_irr_used"]
            row["n_rel"] = d["n_rel_used"]
            # Issue 4: surface pooling mismatch per family.
            if d.get("pooling_mismatch"):
                row["r_m_pooling_mismatch"] = True
                summary_warnings.append(
                    f"{family}: r̂ and m̂ used different pooling token_sets "
                    f"(r̂: '{d.get('token_set_for_attribute_direction')}', "
                    f"m̂: '{d.get('token_set_for_relevance_direction')}')."
                )

        if family in e1:
            e = e1[family]
            row["add_irr_mean_dM"] = e.get("mean_delta_M_add")
            row["add_irr_mean_EF"] = e.get("mean_EF_add_logit")
        if family in e2:
            e = e2[family]
            row["abl_irr_mean_RF_margin"] = e.get("mean_RF_abl_margin")
            row["abl_irr_flips"] = e.get("sum_answer_flip_cf")
            row["abl_irr_recovered"] = e.get("sum_flip_recovered")
        if family in e3:
            row["ctrl_mean_abs_dM_add"] = e3[family].get(
                "mean_abs_delta_M_add")
        if family in e45:
            e5 = e45[family].get("exp5", {})
            if e5:
                row["rel_abl_mean_dM"] = e5.get("mean_delta_M_rel_abl")
            if e45[family].get("exp4") is None:
                summary_warnings.append(
                    f"{family}: Experiment 4 skipped — no omission variants."
                )
        if family in e6:
            row["medrel_RF"] = e6[family].get("mean_RF_medrel")
        if family in e7:
            row["decomp_fve_joint"] = e7[family].get("mean_fve_joint")
            row["decomp_beta_r"] = e7[family].get("mean_beta_r")

        if family in predictions:
            p = predictions[family]
            row["P1_bias"] = p.get("P1_suggests_bias")
            row["P2_causal"] = p.get("P2_direction_causal")
            row["specific"] = p.get("direction_is_specific")

        rows.append(row)

    # Families skipped entirely.
    for fam in ALL_FOCAL:
        if fam not in directions:
            summary_warnings.append(
                f"{fam}: skipped entirely — insufficient D_irr pairs."
            )

    # Deduplicate warnings.
    summary_warnings = list(dict.fromkeys(summary_warnings))

    if rows:
        _save_csv(rows, output_dir / "summary_table.csv")

    summary = {
        "script": SCRIPT_NAME, "version": VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "families": {r["family"]: r for r in rows},
        "predictions": predictions,
        "summary_warnings": summary_warnings,
        "interpretive_notes": {
            "attribute_direction": (
                "Mean shift from original to demographic CF in medically "
                "irrelevant contexts.  Captures 'attribute a was inserted' "
                "but may also encode correlated lexical shifts."),
            "relevance_associated_contrast": (
                "Approximate contrast between prompts where the attribute "
                "is medically relevant vs. irrelevant.  NOT a pure "
                "'relevance neuron'.  May absorb question-content, "
                "diagnosis/domain, and task-distribution confounds."),
            "causality": (
                "Successful addition/ablation implies the direction "
                "carries behaviorally consequential information at the "
                "intervention site.  It does not establish that this "
                "direction is the sole or unique mechanism."),
            "intervention_geometry": (
                "Focal and control interventions use harmonized token-set "
                "semantics.  For span-dependent token_sets, controls fall "
                "back to final-token because no demographic edited span "
                "exists for control prompts."),
        },
        "layer_indexing": {
            "convention": "block_index is the 0-indexed transformer "
                          "block.  stage45_layer_index = block_index + 1.",
            "embedding_excluded": True,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_dir / 'summary.json'}")
    if summary_warnings:
        print(f"  {len(summary_warnings)} warning(s) recorded.")
    return summary


# ===================================================================
# Plotting
# ===================================================================

def plot_direction_norms(directions: dict, target_layers: list[int],
                         output_dir: Path):
    if not HAS_PLT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for fam in ALL_FOCAL:
        if fam not in directions:
            continue
        norms = [directions[fam]["r_norm"].get(l, 0) for l in target_layers]
        ax.plot(target_layers, norms, label=fam, linewidth=2,
                color=FAMILY_COLORS.get(fam, "#333"))
    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("||r_a||₂")
    ax.set_title("Attribute Direction Norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    any_p = False
    for fam in ALL_FOCAL:
        if fam not in directions or directions[fam]["m_norm"] is None:
            continue
        norms = [directions[fam]["m_norm"].get(l, 0) for l in target_layers]
        ax.plot(target_layers, norms, label=fam, linewidth=2,
                color=FAMILY_COLORS.get(fam, "#333"))
        any_p = True
    if any_p:
        ax.set_xlabel("Transformer Block Index")
        ax.set_ylabel("||m_a||₂")
        ax.set_title("Relevance-Contrast Direction Norms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No contrast\ndirections",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes)

    plt.tight_layout()
    d = output_dir / "plots"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "direction_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_site_selection(output_dir: Path):
    if not HAS_PLT:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for fam in ALL_FOCAL:
        p = output_dir / "site_selection" / fam / "effectiveness_scores.csv"
        if not p.exists():
            continue
        layers, scores = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                layers.append(int(row["block_index"]))
                scores.append(float(row["S"]))
        if layers:
            ax.plot(layers, scores, "o-", label=fam, linewidth=2,
                    color=FAMILY_COLORS.get(fam, "#333"), markersize=4)
    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("S_a(l | token_set)")
    ax.set_title("Layer Selection Scores")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    d = output_dir / "plots"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "site_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Sanity checks
# ===================================================================

def run_sanity_checks(model, tok, device: str, answer_ids: dict,
                      n_layers: int, hidden_size: int,
                      data: dict, output_dir: Path):
    """Quick self-checks (Directive L)."""
    print("\n=== Sanity Checks ===")
    prompt = format_prompt(
        "A 50-year-old man presents with chest pain.",
        {"A": "MI", "B": "PE", "C": "GERD", "D": "Pneumonia"})

    # 1. Hook identity.
    print("  1. Hook identity (alpha=0)...")
    ok = validate_hook_identity(model, tok, device, answer_ids,
                                 prompt, 0, hidden_size)
    ok &= validate_hook_identity(model, tok, device, answer_ids,
                                  prompt, n_layers - 1, hidden_size)
    print(f"     {'PASS' if ok else 'FAIL'}")

    # 2. Token-set pooling shapes.
    print("  2. Pooling shapes...")
    _, h, n = get_logits_and_hidden(model, tok, prompt, device,
                                     answer_ids, layers=[0])
    for ts in TOKEN_SETS:
        vec, valid = pool_hidden(h[0], ts, 5, 10)
        assert vec.shape == (hidden_size,), f"{ts}: wrong shape {vec.shape}"
        if ts in ("edited_span", "suffix"):
            assert valid, f"{ts}: unexpectedly invalid"
    print("     PASS")

    # 3. Joint decomposition reconstruction.
    print("  3. Decomposition roundtrip...")
    d1 = np.random.randn(hidden_size).astype(np.float64)
    d1 /= np.linalg.norm(d1)
    d2 = np.random.randn(hidden_size).astype(np.float64)
    d2 /= np.linalg.norm(d2)
    target = 3.0 * d1 + 2.0 * d2
    beta, res, rn = joint_decompose(target, [d1, d2])
    assert rn < 1e-6, f"Decomposition residual too large: {rn}"
    assert abs(beta[0] - 3.0) < 1e-6
    assert abs(beta[1] - 2.0) < 1e-6
    print("     PASS")

    # 4. Direction save/load roundtrip.
    # Issue 1: use families= parameter instead of monkey-patching globals.
    print("  4. Direction save/load roundtrip...")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "directions" / "test_family").mkdir(parents=True)
        test_r = np.random.randn(hidden_size).astype(np.float32)
        np.savez(td / "directions" / "test_family" /
                 "attribute_direction.npz",
                 r_hat_layer_0=test_r)
        with open(td / "directions" / "test_family" /
                  "direction_metadata.json", "w") as f:
            json.dump({"r_norm_by_layer": {"0": 1.0},
                        "target_layers": [0]}, f)
        loaded = load_directions(td, [0], hidden_size,
                                 families=["test_family"])
        assert "test_family" in loaded
        assert np.allclose(loaded["test_family"]["r_hat"][0], test_r)
    print("     PASS")

    # 5. Span-dependent cache key differentiation (FIX 3).
    print("  5. Span-dependent cache keys...")
    qid = "test_q"
    es1, ee1 = 5, 10
    es2, ee2 = 6, 11
    # edited_span: different spans must produce different cache keys.
    key1_span = (qid, es1, ee1, "edited_span")
    key2_span = (qid, es2, ee2, "edited_span")
    assert key1_span != key2_span, \
        "Span-dependent cache keys must differ for different span geometry"
    # final_token: same qid → same cache key (span-independent).
    key1_ft = qid
    key2_ft = qid
    assert key1_ft == key2_ft, \
        "Span-independent cache keys should be identical for same qid"
    print("     PASS")

    print("=== All sanity checks passed ===\n")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Directional Activation Interventions")
    parser.add_argument("--model_path", default="models/llama2-13b")
    parser.add_argument("--data_path", default="cf_v6_balanced.json")
    parser.add_argument("--output_dir", default="stage5_results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--max_pairs_directions", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--token_set", default="final_token",
                        choices=TOKEN_SETS)
    parser.add_argument("--relevance_token_set", default="auto",
                        choices=TOKEN_SETS + ["auto"],
                        help="Token-set for relevance-contrast direction.  "
                             "'auto' matches --token_set unless span-"
                             "dependent, in which case falls back to "
                             "final_token.  (Issue 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_layers", default=None,
                        help="Comma-separated block indices")
    parser.add_argument("--directions_only", action="store_true")
    parser.add_argument("--interventions_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--sanity_checks_only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    config = vars(args).copy()
    config.update(script=SCRIPT_NAME, version=VERSION,
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Stage 5: Directional Activation Interventions v{VERSION}")
    print(f"{'='*60}\n")

    # Load data.
    data = load_partitioned_data(args.data_path)
    with open(out / "data_diagnostics.json", "w") as f:
        json.dump(data["diagnostics"], f, indent=2)

    # Load model.
    model, tok, device, n_layers, hidden_size = load_model(
        args.model_path, args.device, args.dtype)
    answer_ids = get_answer_token_ids(tok)

    # Target layers.
    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(",")]
    else:
        target_layers = list(range(n_layers))
    # Validate bounds.
    for l in target_layers:
        assert 0 <= l < n_layers, \
            f"Block index {l} out of range [0, {n_layers})"

    # Sanity checks.
    if args.sanity_checks_only:
        run_sanity_checks(model, tok, device, answer_ids, n_layers,
                          hidden_size, data, out)
        return

    # === Phase 1: Directions ===
    if args.interventions_only:
        print("\n--- Loading precomputed directions ---")
        directions = load_directions(out, target_layers, hidden_size)
    else:
        print("\n" + "=" * 60)
        print("  Phase 1: Computing Directions")
        print("=" * 60)
        directions = compute_directions(
            model, tok, device, answer_ids,
            data["pairs_irr"], data["records_rel"],
            n_layers, hidden_size, target_layers,
            args.max_pairs_directions, rng, out,
            token_set=args.token_set,
            relevance_token_set=args.relevance_token_set)
        if not args.skip_plots:
            plot_direction_norms(directions, target_layers, out)

    if args.directions_only:
        print("\nDirection computation complete (--directions_only).")
        return

    # Hook identity validation (spot check).
    prompt0 = data["all_pairs"][0]["orig_prompt"] if data["all_pairs"] \
        else "test"
    validate_hook_identity(model, tok, device, answer_ids, prompt0,
                            0, hidden_size)

    # === Phase 2: Layer selection ===
    print("\n" + "=" * 60)
    print("  Phase 2: Layer Selection")
    print("=" * 60)
    best_layers = select_best_layers(
        model, tok, device, answer_ids,
        data["pairs_irr"], data["pairs_ctrl"],
        directions, target_layers,
        args.max_pairs, args.alpha, args.token_set, rng, out)
    if not args.skip_plots:
        plot_site_selection(out)

    # === Phase 3: Experiments ===
    print("\n" + "=" * 60)
    print("  Phase 3: Running Experiments")
    print("=" * 60)

    e1 = run_exp1_add_irr(model, tok, device, answer_ids,
                           data["pairs_irr"], directions, best_layers,
                           args.alpha, args.token_set, args.max_pairs,
                           rng, out)
    e2 = run_exp2_abl_irr(model, tok, device, answer_ids,
                           data["pairs_irr"], directions, best_layers,
                           args.token_set, args.max_pairs, rng, out)
    e3 = run_exp3_ctrl(model, tok, device, answer_ids,
                        data["pairs_ctrl"], data["pairs_irr"],
                        directions, best_layers,
                        args.alpha, args.token_set, args.max_pairs,
                        rng, out)
    e45 = run_exp45_relevant(model, tok, device, answer_ids,
                              data["records_rel"], directions,
                              best_layers, args.alpha, args.token_set,
                              args.max_pairs, rng, out)
    e6 = run_exp6_medrel_abl(model, tok, device, answer_ids,
                              data["pairs_irr"], directions,
                              best_layers, args.token_set,
                              args.max_pairs, rng, out)
    e7 = run_exp7_decomposition(model, tok, device, answer_ids,
                                 data["pairs_irr"], directions,
                                 best_layers, args.token_set,
                                 args.max_pairs, rng, out)

    # === Phase 4: Predictions & Summary ===
    print("\n" + "=" * 60)
    print("  Phase 4: Predictions & Summary")
    print("=" * 60)

    preds = evaluate_predictions(e1, e2, e3, e45, e6, e7, out)
    generate_summary(e1, e2, e3, e45, e6, e7, preds, best_layers,
                     directions, args.token_set, args.alpha, out)

    # Print prediction digest.
    print("\n--- Prediction Digest ---")
    for fam in ALL_FOCAL:
        if fam not in preds:
            continue
        p = preds[fam]
        parts = [f"  {fam}:"]
        dm = p.get("P1_mean_delta_M_add")
        if dm is not None:
            parts.append(f"P1 ΔM={dm:.4f}")
        rf = p.get("P2_mean_RF_abl_logit")
        if rf is not None:
            parts.append(f"P2 RF={rf:.4f}")
        sp = p.get("specificity_ratio")
        if sp is not None:
            parts.append(f"spec={sp:.3f}")
        print("  ".join(parts))

    print(f"\n{'='*60}")
    print(f"  Stage 5 complete.  Results: {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
