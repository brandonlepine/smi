"""
analyze_stage45_layerwise_decoding.py — Stage 4.5: Layerwise Linear-Probe Decoding

Context
-------
Stage 3 showed strong early-layer edit-mass structure for gender_identity and race,
weaker/more diffuse patterns for sexual_orientation, and a very different profile for
controls.  Stage 4 causal patching showed that attention-head interventions explain
only part of the behavioral effect — especially for sexual_orientation where recovery
at k=5..20 heads was weak.

This script asks: *where in the residual stream is demographic information decodable?*
A linear probe at each layer tells us whether the information is *present* at that
layer, regardless of whether ablating a specific head set removes it.

Key interpretive notes (baked into metadata / summary.json)
------------------------------------------------------------
1. Decodability ≠ causal necessity.
   High probe accuracy means information is present/readable at a layer, not that
   intervening on that layer would change behavior.

2. High decodability + weak causal ablation → distributed or redundant representation.

3. Aligned peaks across Stage 3 (edit-mass), Stage 4 (residual patching recovery),
   and Stage 4.5 (probe accuracy) → convergent evidence for localization.

4. sexual_orientation hypothesis: weaker early decodability, stronger mid/late,
   consistent with Stage 4 B1–B3 showing diffuse recovery and strong context split.

Tasks
-----
A. orig_vs_cf          — binary: original vs CF hidden state (per family)
B. family_identity     — multiclass: which family? (global, CF only)
C. within_family_attribute — multiclass: which attribute value? (per family, CF only)
D. context_split       — binary: partner-framed vs explicit-identity (sexual_orientation)
E. focal_vs_control_cf — binary: focal CF (1) vs control CF (0)

Feature scopes
--------------
final_token, mean_pool, edited_span_mean, edited_window_mean, prefix_mean, suffix_mean

Usage
-----
  python analyze_stage45_layerwise_decoding.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage45_results \\
    --device auto \\
    --max_pairs 200 \\
    --feature_scope final_token
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_NAME = "analyze_stage45_layerwise_decoding.py"
VERSION = "2.1"

PROMPT_TEMPLATE = """\
Question:
{question}

Answer choices:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

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

# Stage 3 findings used as annotation in plots and summary metadata.
STAGE3_EDIT_MASS_PEAK = {
    "gender_identity":    "early (layers 0–3)",
    "race":               "early (layers 0–3)",
    "sex_gender":         "early-to-mid",
    "sexual_orientation": "diffuse / mid-late",
    "control":            "n/a",
}

# Scopes that depend on span detection; set to None when fallback=True
_SPAN_SCOPES = frozenset({"edited_span_mean", "edited_window_mean", "prefix_mean", "suffix_mean"})

FEATURE_SCOPES = [
    "final_token",
    "mean_pool",
    "edited_span_mean",
    "edited_window_mean",
    "prefix_mean",
    "suffix_mean",
]

# ---------------------------------------------------------------------------
# Req 16: Stage 3 alignment constants
# ---------------------------------------------------------------------------

# Layer region boundaries (exclusive upper bounds for the named region).
# Layer 0 = embedding; transformer layers are 1-indexed.
#   early : 1 <= layer < EARLY_END_LAYER  (layers 1–3)
#   mid   : EARLY_END_LAYER <= layer < MID_END_LAYER  (layers 4–15)
#   late  : layer >= MID_END_LAYER  (layers 16+)
EARLY_END_LAYER = 4
MID_END_LAYER   = 16

# Values may be a single region string or a list of acceptable regions.
# sex_gender: Stage 3 showed early-to-mid peaks, so both are accepted.
# sexual_orientation: Stage 3 showed diffuse / mid-late patterns.
STAGE3_EXPECTED_REGION: dict[str, str | list[str] | None] = {
    "gender_identity":    "early",
    "race":               "early",
    "sex_gender":         ["early", "mid"],
    "sexual_orientation": ["mid", "late"],
    "control":            None,
}

_GI_FRAGMENTS = {
    "non-binary", "nonbinary", "non binary",
    "transgender", "trans man", "trans woman", "transman", "transwoman",
    "gender non-conforming", "genderqueer", "agender",
}

_PARTNER_PATTERNS = {"partner", "same sex partner", "partner same sex"}
_EXPLICIT_PATTERNS = {"gay", "straight", "lesbian", "bisexual", "queer",
                      "heterosexual", "homosexual"}

# ---------------------------------------------------------------------------
# Req 16: layer_to_region helper
# ---------------------------------------------------------------------------

def layer_to_region(layer_idx: int) -> str:
    """Map a layer index to a named region.

    layer_idx == 0                       → 'embedding'
    1 <= layer_idx < EARLY_END_LAYER (4) → 'early'
    4 <= layer_idx < MID_END_LAYER  (16) → 'mid'
    layer_idx >= 16                      → 'late'
    """
    if layer_idx == 0:
        return "embedding"
    if layer_idx < EARLY_END_LAYER:
        return "early"
    if layer_idx < MID_END_LAYER:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Req 9: Uniform-chance / normalized gain helpers
# ---------------------------------------------------------------------------

def compute_normalized_gain(balanced_accuracy: float, n_classes: int) -> float:
    """Normalized decoding gain: (bal_acc - uniform) / (1 - uniform).

    Maps [uniform, 1.0] → [0, 1]; below-uniform values map to negative.
    """
    uniform = 1.0 / max(n_classes, 1)
    denom = 1.0 - uniform
    return (balanced_accuracy - uniform) / denom if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Req 3: split validity check
# ---------------------------------------------------------------------------

def validate_split(y_train: np.ndarray, y_test: np.ndarray) -> str | None:
    """Returns an error string or None if the split is valid."""
    if len(y_train) == 0 or len(y_test) == 0:
        return "empty_split"
    if len(np.unique(y_train)) < 2:
        return "single_class_train"
    test_classes = set(np.unique(y_test).tolist())
    train_classes = set(np.unique(y_train).tolist())
    if not test_classes.issubset(train_classes):
        return "train_missing_class"
    return None


# ---------------------------------------------------------------------------
# Req 2: balance_training_set helper
# ---------------------------------------------------------------------------

def balance_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample majority classes to size of smallest class. Operates on train split only."""
    rng_local = np.random.default_rng(seed + 9999)
    counts = Counter(y_train.tolist())
    min_count = min(counts.values())
    indices: list[int] = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        chosen = rng_local.choice(cls_idx, min_count, replace=False)
        indices.extend(chosen.tolist())
    rng_local.shuffle(indices)
    return X_train[indices], y_train[indices]


# ---------------------------------------------------------------------------
# Family assignment
# ---------------------------------------------------------------------------

def _normalize_label(label) -> str:
    s = str(label or "").lower()
    s = re.sub(r"['\"\-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def assign_family(itype: str, attr_val) -> str | None:
    """Map (intervention_type, attribute_value) → family key.  Returns None for unrecognised itypes."""
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


def orientation_split_label(attr_norm: str) -> str:
    if any(p in attr_norm for p in _PARTNER_PATTERNS):
        return "partner"
    if any(p in attr_norm for p in _EXPLICIT_PATTERNS):
        return "explicit"
    return "other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def format_prompt(q: str, options: dict) -> str:
    return PROMPT_TEMPLATE.format(
        question=q.strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


def load_pairs(data_path: str, include_controls: bool = False) -> tuple[list, dict]:
    """Load counterfactual pairs from cf_v6_balanced.json.  Returns (pairs, assignment_log)."""
    with open(data_path) as f:
        records = json.load(f)

    pairs = []
    counts: Counter = Counter()
    examples: dict = defaultdict(list)

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
        if gold not in {"A", "B", "C", "D"}:
            continue

        orig_q = orig["question"]
        orig_prompt = format_prompt(orig_q, options)
        variants = rec.get("counterfactuals", {}).get("variants", [])

        for v in variants:
            if not isinstance(v, dict) or v.get("text") is None:
                continue
            itype = v.get("intervention_type", "")
            attr_val = v.get("attribute_value_counterfactual", "")
            attr_val_orig = v.get("attribute_value_original", "")
            family = assign_family(itype, attr_val)

            if family is None:
                counts["_dropped"] += 1
                continue
            if family == "control" and not include_controls:
                continue

            counts[family] += 1
            if len(examples[family]) < 5:
                examples[family].append({"itype": itype, "attr_val": str(attr_val), "qid": qid})

            pairs.append({
                "qid":          qid,
                "family":       family,
                "itype":        itype,
                "attr_val":     str(attr_val or itype),
                "attr_val_orig": str(attr_val_orig or ""),
                "attr_norm":    _normalize_label(attr_val),
                "gold":         gold,
                "options":      options,
                "orig_q":       orig_q,
                "cf_q":         v["text"],
                "orig_prompt":  orig_prompt,
                "cf_prompt":    format_prompt(v["text"], options),
            })

    log = {fam: {"count": counts[fam], "examples": examples[fam]} for fam in sorted(counts)}
    return pairs, log


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def choose_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tok(model_path: str, device_arg: str, dtype_str: str):
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    device = choose_device(device_arg)

    print(f"Loading tokenizer from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model (dtype={dtype_str}, device={device})...")
    load_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  Model ready: {n_layers} transformer layers, hidden_size={hidden_size}")
    return model, tok, device


# ---------------------------------------------------------------------------
# Edited-span detection at token level
# ---------------------------------------------------------------------------

def find_edited_span(
    orig_ids: list[int],
    cf_ids:   list[int],
) -> tuple[int, int, int, bool]:
    """Locate the token span that differs between orig and cf using LCS prefix/suffix.

    Returns (edit_start, edit_end_orig, edit_end_cf, fallback_used).
      edit_start    — first token that differs (same index in both sequences)
      edit_end_orig — one-past-last changed token in orig_ids
      edit_end_cf   — one-past-last changed token in cf_ids
      fallback_used — True when sequences are identical or alignment failed

    edit_start is guaranteed < edit_end_orig and edit_start < edit_end_cf.
    """
    n_orig, n_cf = len(orig_ids), len(cf_ids)

    # Longest common prefix
    prefix_len = 0
    for i in range(min(n_orig, n_cf)):
        if orig_ids[i] == cf_ids[i]:
            prefix_len += 1
        else:
            break

    # Longest common suffix (cannot overlap with prefix)
    max_suffix = min(n_orig - prefix_len, n_cf - prefix_len)
    suffix_len = 0
    for i in range(1, max_suffix + 1):
        if orig_ids[n_orig - i] == cf_ids[n_cf - i]:
            suffix_len += 1
        else:
            break

    edit_start    = prefix_len
    edit_end_orig = n_orig - suffix_len
    edit_end_cf   = n_cf   - suffix_len

    # Degenerate cases: identical seqs or empty edit region
    if edit_start >= edit_end_orig or edit_start >= edit_end_cf:
        return 0, n_orig, n_cf, True

    return edit_start, edit_end_orig, edit_end_cf, False


# ---------------------------------------------------------------------------
# Feature pooling
# ---------------------------------------------------------------------------

def pool_hidden(
    hidden: torch.Tensor,  # (1, seq_len, hidden_size) — already .float().cpu()
    scope:  str,
    edit_start:  int,
    edit_end:    int,       # use edit_end_orig for orig hidden, edit_end_cf for cf hidden
    window_radius: int = 2,
) -> np.ndarray:
    """Extract a 1-D feature vector from a hidden-state tensor."""
    h = hidden[0]          # (seq_len, H)
    seq_len = h.shape[0]

    def _safe_slice(s: int, e: int) -> np.ndarray:
        s = max(0, min(s, seq_len - 1))
        e = max(s + 1, min(e, seq_len))
        return h[s:e].mean(dim=0).numpy()

    if scope == "final_token":
        return h[-1].numpy()

    if scope == "mean_pool":
        return h.mean(dim=0).numpy()

    if scope == "edited_span_mean":
        return _safe_slice(edit_start, edit_end)

    if scope == "edited_window_mean":
        return _safe_slice(edit_start - window_radius, edit_end + window_radius)

    if scope == "prefix_mean":
        # tokens before the edit — common prefix, so same for orig and cf
        e = max(1, edit_start)
        return _safe_slice(0, e)

    if scope == "suffix_mean":
        # tokens after the edit; edit_end differs per sequence, so caller passes correct one
        return _safe_slice(edit_end, seq_len)

    raise ValueError(f"Unknown feature scope: {scope!r}")


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_forward(model, tok, prompt: str, device: str) -> tuple[list[torch.Tensor], list[int]]:
    """One forward pass with output_hidden_states=True.
    Returns (hidden_states, input_ids).
    hidden_states[0] = embedding layer, hidden_states[1..L] = transformer layers.
    """
    # Explicit truncation policy: 2048 tokens matches the default context length used in
    # other stage scripts and prevents silent OOM errors on long prompts.
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"][0].tolist()
    out = model(**inputs, output_hidden_states=True)
    # Move everything to CPU and cast to float32 immediately to keep GPU memory free
    hidden_states = [h.detach().float().cpu() for h in out.hidden_states]
    return hidden_states, input_ids


def extract_features(
    model,
    tok,
    pairs:         list[dict],
    device:        str,
    scopes:        list[str],
    window_radius: int = 2,
    min_prefix_tokens: int = 2,
    min_suffix_tokens: int = 2,
) -> list[dict]:
    """Run forward passes and pool hidden states for every pair.

    Returns list of records, one per pair:
      qid, family, attr_val, attr_val_orig, attr_norm, gold, split_label,
      edit_start, edit_end_orig, edit_end_cf, edit_fallback,
      n_hidden_states,   <- len(hidden_states) = n_transformer_layers + 1 (embedding)
      orig_feats: {scope: {layer_idx: np.ndarray | None}},
      cf_feats:   {scope: {layer_idx: np.ndarray | None}},

    For span scopes when fallback=True, every layer is stored as None.
    For prefix_mean/suffix_mean, None is stored if the prefix/suffix is too short.
    """
    results = []
    n = len(pairs)
    print(f"  Extracting features: {n} pairs, scopes={scopes}")

    for i, pair in enumerate(pairs):
        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"    {i + 1}/{n}", end="\r", flush=True)

        try:
            orig_hs, orig_ids = _run_forward(model, tok, pair["orig_prompt"], device)
            cf_hs,   cf_ids   = _run_forward(model, tok, pair["cf_prompt"],   device)
        except Exception as exc:
            print(f"\n    Warning: forward pass failed pair {i} qid={pair['qid']}: {exc}")
            continue

        edit_start, edit_end_orig, edit_end_cf, fallback = find_edited_span(orig_ids, cf_ids)
        n_hs = len(orig_hs)  # includes embedding at index 0

        orig_feats: dict = {sc: {} for sc in scopes}
        cf_feats:   dict = {sc: {} for sc in scopes}

        for layer_idx in range(n_hs):
            for sc in scopes:
                # Req 5: span scopes → None when fallback
                if fallback and sc in _SPAN_SCOPES:
                    orig_feats[sc][layer_idx] = None
                    cf_feats[sc][layer_idx]   = None
                    continue

                # Req 6: min prefix/suffix token checks
                if sc == "prefix_mean":
                    if edit_start < min_prefix_tokens:
                        orig_feats[sc][layer_idx] = None
                        cf_feats[sc][layer_idx]   = None
                        continue
                elif sc == "suffix_mean":
                    orig_suffix_len = len(orig_ids) - edit_end_orig
                    cf_suffix_len   = len(cf_ids)   - edit_end_cf
                    if orig_suffix_len < min_suffix_tokens or cf_suffix_len < min_suffix_tokens:
                        orig_feats[sc][layer_idx] = None
                        cf_feats[sc][layer_idx]   = None
                        continue

                orig_feats[sc][layer_idx] = pool_hidden(
                    orig_hs[layer_idx], sc, edit_start, edit_end_orig, window_radius)
                cf_feats[sc][layer_idx] = pool_hidden(
                    cf_hs[layer_idx], sc, edit_start, edit_end_cf, window_radius)

        results.append({
            "qid":           pair["qid"],
            "family":        pair["family"],
            "attr_val":      pair["attr_val"],
            "attr_val_orig": pair["attr_val_orig"],
            "attr_norm":     pair["attr_norm"],
            "gold":          pair["gold"],
            "split_label":   (orientation_split_label(pair["attr_norm"])
                              if pair["family"] == "sexual_orientation" else None),
            "edit_start":    edit_start,
            "edit_end_orig": edit_end_orig,
            "edit_end_cf":   edit_end_cf,
            "edit_fallback": fallback,
            "n_hidden_states": n_hs,
            "orig_feats":    orig_feats,
            "cf_feats":      cf_feats,
        })

    print()  # newline after \r progress
    return results


# ---------------------------------------------------------------------------
# Probe dataset builders
# ---------------------------------------------------------------------------

def _qid_group(qid: str) -> int:
    """Deterministic integer group id from a qid string.

    Uses MD5 so the result is stable across Python processes and platforms,
    regardless of PYTHONHASHSEED. Two runs with identical qids will always
    produce the same group assignments.
    """
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % (2 ** 31)


def build_orig_vs_cf(
    bank: list[dict], family: str, scope: str, layer: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Task A — binary: orig (0) vs CF (1) for one family."""
    X, y, groups = [], [], []
    for r in bank:
        if r["family"] != family:
            continue
        o = r["orig_feats"].get(scope, {}).get(layer)
        c = r["cf_feats"].get(scope, {}).get(layer)
        if o is None or c is None:
            continue
        grp = _qid_group(r["qid"])
        X.extend([o, c])
        y.extend([0, 1])
        groups.extend([grp, grp])
    if not X:
        return None, None, None
    return np.array(X), np.array(y), np.array(groups)


def build_family_identity(
    bank: list[dict], families: list[str], scope: str, layer: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Task B — multiclass: CF only, label by family."""
    label_map = {f: i for i, f in enumerate(families)}
    X, y, groups = [], [], []
    for r in bank:
        if r["family"] not in label_map:
            continue
        c = r["cf_feats"].get(scope, {}).get(layer)
        if c is None:
            continue
        X.append(c)
        y.append(label_map[r["family"]])
        groups.append(_qid_group(r["qid"]))
    if not X:
        return None, None, None, families
    return np.array(X), np.array(y), np.array(groups), families


def build_within_family_attribute(
    bank:           list[dict],
    family:         str,
    scope:          str,
    layer:          int,
    min_examples:   int = 20,
    exclude_labels: list[str] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str], list[str]]:
    """Task C — multiclass: CF only, label by attr_val within one family.
    Returns (X, y, groups, retained_classes, dropped_classes).
    exclude_labels: labels to remove before counting retained/dropped.
    """
    excl = set(exclude_labels or [])
    fam_records = [r for r in bank if r["family"] == family and r["attr_val"] not in excl]
    counts = Counter(r["attr_val"] for r in fam_records)
    all_classes = sorted(counts.items())
    retained = [cls for cls, cnt in all_classes if cnt >= min_examples]
    dropped  = [cls for cls, cnt in all_classes if cnt < min_examples]
    if len(retained) < 2:
        return None, None, None, retained, dropped

    label_map = {cls: i for i, cls in enumerate(retained)}
    X, y, groups = [], [], []
    for r in fam_records:
        if r["attr_val"] not in label_map:
            continue
        c = r["cf_feats"].get(scope, {}).get(layer)
        if c is None:
            continue
        X.append(c)
        y.append(label_map[r["attr_val"]])
        groups.append(_qid_group(r["qid"]))
    if not X:
        return None, None, None, retained, dropped
    return np.array(X), np.array(y), np.array(groups), retained, dropped


def build_context_split(
    bank: list[dict], scope: str, layer: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Task D — binary: partner (0) vs explicit (1) for sexual_orientation.
    Returns (X, y, groups, n_other_dropped).
    """
    so_records = [r for r in bank if r["family"] == "sexual_orientation"]
    n_other = sum(1 for r in so_records if r.get("split_label") == "other")
    label_map = {"partner": 0, "explicit": 1}
    X, y, groups = [], [], []
    for r in so_records:
        lbl = r.get("split_label")
        if lbl not in label_map:
            continue
        c = r["cf_feats"].get(scope, {}).get(layer)
        if c is None:
            continue
        X.append(c)
        y.append(label_map[lbl])
        groups.append(_qid_group(r["qid"]))
    if not X:
        return None, None, None, n_other
    return np.array(X), np.array(y), np.array(groups), n_other


def build_focal_vs_control_cf(
    bank: list[dict], focal_family: str, scope: str, layer: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Task E — Binary: focal CF (1) vs control CF (0). Returns None if no control pairs."""
    X, y, groups = [], [], []
    for r in bank:
        if r["family"] == focal_family:
            c = r["cf_feats"].get(scope, {}).get(layer)
            if c is None:
                continue
            X.append(c)
            y.append(1)
            groups.append(_qid_group(r["qid"]))
        elif r["family"] == "control":
            c = r["cf_feats"].get(scope, {}).get(layer)
            if c is None:
                continue
            X.append(c)
            y.append(0)
            groups.append(_qid_group(r["qid"]))
    if not X or len(set(y)) < 2:
        return None, None, None
    return np.array(X), np.array(y), np.array(groups)


# ---------------------------------------------------------------------------
# Grouped train/test split
# ---------------------------------------------------------------------------

def grouped_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """GroupShuffleSplit by qid; falls back to StratifiedShuffleSplit when groups are too few."""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return train_idx, test_idx, "group_shuffle"

    # Fallback: stratified (may still fail for single-class splits)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        return train_idx, test_idx, "stratified_shuffle_fallback"
    except Exception:
        n = len(X)
        cut = max(1, int(n * (1 - test_size)))
        return np.arange(cut), np.arange(cut, n), "sequential_fallback"


# ---------------------------------------------------------------------------
# Probe fitting and evaluation
# ---------------------------------------------------------------------------

def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    task_type: str,            # "binary" or "multiclass"
    label_names: list[str] | None = None,
    max_iter: int = 1000,
    balance_classes: bool = False,
    probe_C: float = 1.0,
    probe_class_weight=None,   # None or "balanced"
) -> dict:
    """Fit a StandardScaler + LogisticRegression pipeline and return metrics."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"error": "single_class", "n_examples": int(len(y))}

    train_idx, test_idx, splitter = grouped_split(X, y, groups, test_size, seed)

    # Req 3: validate split
    y_train_pre = y[train_idx]
    y_test_pre  = y[test_idx]
    err = validate_split(y_train_pre, y_test_pre)
    if err is not None:
        return {"error": err, "n_examples": int(len(y))}

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_train_pre, y_test_pre

    # Req 2: balance training set if requested
    if balance_classes:
        X_train, y_train = balance_training_set(X_train, y_train, seed)

    balanced_class_counts = Counter(y_train.tolist())

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
            C=probe_C,
            class_weight=probe_class_weight,
        )),
    ])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
    except Exception as exc:
        return {"error": str(exc), "n_examples": int(len(y))}

    y_pred = pipe.predict(X_test)

    # Req 9: compute baselines
    train_counts = Counter(y_train.tolist())
    majority_class = max(train_counts, key=train_counts.get)
    majority_baseline_accuracy = float(train_counts[majority_class] / len(y_train))

    n_classes = int(len(unique_classes))
    uniform_chance_accuracy = float(1.0 / n_classes)

    acc     = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    accuracy_lift_over_majority = float(acc - majority_baseline_accuracy)
    accuracy_lift_over_uniform  = float(acc - uniform_chance_accuracy)
    normalized_gain = compute_normalized_gain(bal_acc, n_classes)

    metrics: dict = {
        "n_examples":                  int(len(y)),
        "n_train":                     int(len(y_train)),
        "n_test":                      int(len(y_test)),
        "n_classes":                   n_classes,
        "splitter_type":               splitter,
        "accuracy":                    acc,
        "balanced_accuracy":           bal_acc,
        "uniform_chance_accuracy":     uniform_chance_accuracy,
        "majority_baseline_accuracy":  majority_baseline_accuracy,
        "accuracy_lift_over_majority": accuracy_lift_over_majority,
        "accuracy_lift_over_uniform":  accuracy_lift_over_uniform,
        "normalized_decoding_gain":    normalized_gain,
        "balanced_class_counts":       dict(balanced_class_counts),
        "f1":                          None,
        "auroc":                       None,
        "macro_f1":                    None,
        "weighted_f1":                 None,
    }

    if task_type == "binary":
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
    else:
        metrics["macro_f1"]    = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion matrix (stored separately, not in the flat CSV)
    all_labels = list(range(len(label_names))) if label_names else list(range(len(unique_classes)))
    metrics["_confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=all_labels).tolist()
    metrics["_label_names"]      = label_names

    return metrics


# ---------------------------------------------------------------------------
# Per-layer probe sweep
# ---------------------------------------------------------------------------

def run_layerwise_probe(
    bank:            list[dict],
    task:            str,
    scope:           str,
    family:          str | None,
    families_global: list[str],
    n_hidden_states: int,
    test_size:       float,
    seed:            int,
    min_examples_per_class: int,
    balance_classes: bool = False,
    probe_C:         float = 1.0,
    probe_max_iter:  int   = 1000,
    probe_class_weight=None,
    within_family_exclude_labels: list[str] | None = None,
) -> list[dict]:
    """Run the probe at every layer and return a list of per-layer result dicts."""
    rows = []
    for layer in range(n_hidden_states):
        # Req 7: layer indexing clarity
        base = {
            "layer":                     layer,
            "layer_index":               layer,
            "layer_type":                "embedding" if layer == 0 else "transformer",
            "transformer_layer_number":  None if layer == 0 else layer,
            "feature_scope":             scope,
            "task":                      task,
            "family":                    family or "global",
        }

        # ----- Build X, y, groups -----
        if task == "orig_vs_cf":
            X, y, groups = build_orig_vs_cf(bank, family, scope, layer)
            task_type = "binary"
            label_names = ["orig", "cf"]
            extra = {}

        elif task == "family_identity":
            X, y, groups, label_names = build_family_identity(
                bank, families_global, scope, layer)
            task_type = "multiclass"
            extra = {}

        elif task == "within_family_attribute":
            X, y, groups, retained, dropped = build_within_family_attribute(
                bank, family, scope, layer, min_examples_per_class,
                exclude_labels=within_family_exclude_labels)
            task_type = "multiclass"
            label_names = retained
            extra = {"retained_classes": retained, "dropped_classes": dropped}
            if X is None:
                rows.append({**base, **extra, "error": "insufficient_classes"})
                continue

        elif task == "context_split":
            X, y, groups, n_other = build_context_split(bank, scope, layer)
            task_type = "binary"
            label_names = ["partner", "explicit"]
            extra = {"n_other_dropped": n_other}

        elif task == "focal_vs_control_cf":
            X, y, groups = build_focal_vs_control_cf(bank, family, scope, layer)
            task_type = "binary"
            label_names = ["control", "focal"]
            extra = {}

        else:
            rows.append({**base, "error": f"unknown_task_{task}"})
            continue

        if X is None:
            rows.append({**base, **extra, "error": "no_data"})
            continue

        metrics = fit_probe(
            X, y, groups, test_size, seed, task_type, label_names,
            max_iter=probe_max_iter,
            balance_classes=balance_classes,
            probe_C=probe_C,
            probe_class_weight=probe_class_weight,
        )

        # Flatten: exclude internal _keys from the row dict
        row = {**base, **extra,
               **{k: v for k, v in metrics.items() if not k.startswith("_")}}
        row["_confusion_matrix"] = metrics.get("_confusion_matrix")
        row["_label_names"]      = metrics.get("_label_names")
        # Req 22: derived convenience field for CSV (backward compat)
        row["f1_or_macro_f1"] = row.get("f1") if row.get("f1") is not None else row.get("macro_f1")
        rows.append(row)

    return rows


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


def save_json(obj: object, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_js(obj), f, indent=2)


# Req 22: updated CSV columns
_CSV_COLS = [
    "layer", "layer_index", "layer_type", "transformer_layer_number",
    "feature_scope", "task", "family",
    "n_examples", "n_train", "n_test", "n_classes",
    "accuracy", "balanced_accuracy",
    "uniform_chance_accuracy", "majority_baseline_accuracy",
    "accuracy_lift_over_majority", "accuracy_lift_over_uniform",
    "normalized_decoding_gain",
    "f1_or_macro_f1", "weighted_f1", "auroc",
    "splitter_type", "error",
]


def save_csv(rows: list[dict], path: Path, priority_cols: list[str] | None = None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    prio = priority_cols or _CSV_COLS
    all_keys = list(dict.fromkeys(
        prio + [k for r in rows for k in r if not k.startswith("_") and k not in prio]
    ))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_confusion_matrix(cm: list[list], labels: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for label, row in zip(labels, cm):
            w.writerow([label] + row)


def save_experiment(exp_dir: Path, metadata: dict, per_layer_rows: list[dict]):
    """Write metadata.json and per_layer.csv for one experiment.
    Also saves confusion_matrix.csv at the peak balanced_accuracy layer.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Req 21: auto-inject standard fields
    metadata.setdefault("script",  SCRIPT_NAME)
    metadata.setdefault("version", VERSION)
    metadata.setdefault("layer_0_is_embedding", True)
    metadata.setdefault(
        "span_scope_fallback_policy",
        "edited_span_mean/edited_window_mean/prefix_mean/suffix_mean set to None when no token-level edit detected",
    )
    metadata.setdefault(
        "interpretive_note",
        "Probe accuracy = linear decodability from hidden states. Does NOT imply causal necessity.",
    )

    save_json(metadata, exp_dir / "metadata.json")
    save_csv(per_layer_rows, exp_dir / "per_layer.csv")

    # Confusion matrix at peak layer
    peak_row = None
    for r in per_layer_rows:
        if r.get("_confusion_matrix") is None:
            continue
        if (peak_row is None
                or (r.get("balanced_accuracy") or 0.0) > (peak_row.get("balanced_accuracy") or 0.0)):
            peak_row = r
    if peak_row is not None and peak_row.get("_confusion_matrix"):
        save_confusion_matrix(
            peak_row["_confusion_matrix"],
            peak_row.get("_label_names") or [],
            exp_dir / "confusion_matrix_peak_layer.csv",
        )


def summarize_layers(rows: list[dict], metric: str = "balanced_accuracy") -> dict:
    """Return peak layer, peak value, and mean value for one experiment."""
    valid = [(r["layer"], r[metric]) for r in rows
             if r.get(metric) is not None and not r.get("error")]
    if not valid:
        return {}
    peak_layer, peak_val = max(valid, key=lambda x: x[1])
    all_vals = [v for _, v in valid]
    return {
        "peak_layer":  int(peak_layer),
        "peak_value":  float(peak_val),
        "mean_value":  float(np.mean(all_vals)),
        "n_layers":    len(valid),
        "metric":      metric,
    }


def _task_sample_summary(rows: list[dict], metric: str = "balanced_accuracy") -> dict:
    """Summarise n_examples/n_train/n_test across all valid probe layers.

    Peak layer is identified by `metric` (default: balanced_accuracy). The returned
    dict records which metric was used so the summary is self-describing.
    Reflects actual layer-by-layer variation rather than silently assuming uniform
    counts (counts can differ when span-scope features are omitted at some layers).
    Returns {} if there are no valid rows.
    """
    valid = [r for r in rows if not r.get("error") and r.get("n_examples") is not None]
    if not valid:
        return {}
    by_ne  = [r["n_examples"] for r in valid]
    by_ntr = [r["n_train"] for r in valid if r.get("n_train") is not None]
    by_nte = [r["n_test"]  for r in valid if r.get("n_test")  is not None]
    peak_r = max(valid, key=lambda r: r.get(metric) or 0.0)
    return {
        "n_valid_layers":                     len(valid),
        "peak_metric":                        metric,
        "peak_layer":                         peak_r["layer"],
        "peak_layer_n_examples":              peak_r.get("n_examples"),
        "peak_layer_n_train":                 peak_r.get("n_train"),
        "peak_layer_n_test":                  peak_r.get("n_test"),
        "min_n_examples_across_valid_layers": min(by_ne),
        "max_n_examples_across_valid_layers": max(by_ne),
        "min_n_train_across_valid_layers":    min(by_ntr) if by_ntr else None,
        "max_n_train_across_valid_layers":    max(by_ntr) if by_ntr else None,
        "min_n_test_across_valid_layers":     min(by_nte) if by_nte else None,
        "max_n_test_across_valid_layers":     max(by_nte) if by_nte else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PLOT_DPI = 150
_LINE_KW  = {"linewidth": 1.8, "alpha": 0.9}


def _add_embed_layer1_separator(ax):
    """Add vertical line at x=0.5 separating embedding from transformer layers."""
    ax.axvline(x=0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.6, label="embed|L1")


def _ax_layer_ticks(ax, n_layers: int):
    step = max(1, n_layers // 10)
    ax.set_xticks(range(0, n_layers, step))
    ax.set_xlabel("Layer  (0 = embedding, 1..N = transformer)")
    ax.tick_params(axis="x", labelsize=8)


def plot_orig_vs_cf_curves(
    results_by_family: dict,      # family → list[dict] of per_layer rows
    scope:             str,
    output_path:       Path,
    metric:            str = "balanced_accuracy",
):
    """Plot A: one line per family showing orig-vs-cf decodability across layers."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=_PLOT_DPI)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.4, label="chance (0.5)")
    _add_embed_layer1_separator(ax)

    max_layer = 0
    for fam, rows in results_by_family.items():
        layers = [r["layer"] for r in rows if r.get(metric) is not None]
        vals   = [r[metric]  for r in rows if r.get(metric) is not None]
        if not layers:
            continue
        ax.plot(layers, vals, label=fam, color=FAMILY_COLORS.get(fam, "#888888"), **_LINE_KW)
        max_layer = max(max_layer, max(layers))

    ax.set_ylabel(metric.replace("_", " ").title())
    # Req 4 & 20: title prefix for orig_vs_cf
    ax.set_title(
        f"Manipulation Detectability (orig vs CF)  |  scope: {scope}\n"
        f"Note: Stage 3 edit-mass peaks were early for gender_identity and race."
    )
    _ax_layer_ticks(ax, max_layer + 1)
    ax.set_ylim(0.35, 1.05)
    ax.legend(framealpha=0.85, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_family_identity_curve(
    rows:        list[dict],
    scope:       str,
    output_path: Path,
    metric:      str = "balanced_accuracy",
):
    """Plot B: family identity decoding curve (global, CF only)."""
    layers = [r["layer"] for r in rows if r.get(metric) is not None]
    vals   = [r[metric]  for r in rows if r.get(metric) is not None]
    if not layers:
        return

    n_classes = next((r.get("n_classes") for r in rows if r.get("n_classes")), 4)
    baseline  = 1.0 / max(n_classes, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=_PLOT_DPI)
    ax.axhline(baseline, color="black", linewidth=1, linestyle="--", alpha=0.4,
               label=f"chance  1/{n_classes} ≈ {baseline:.2f}")
    _add_embed_layer1_separator(ax)
    ax.plot(layers, vals, color="#333333", **_LINE_KW, label="family identity")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"Family identity decodability (CF hidden states)  |  scope: {scope}\n"
        f"How well do {n_classes} demographic families occupy separable representational regions?"
    )
    _ax_layer_ticks(ax, max(layers) + 1)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.85, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_within_family_curves(
    results_by_family: dict,
    scope:             str,
    output_path:       Path,
    metric:            str = "balanced_accuracy",
):
    """Plot C: within-family attribute decoding, one subplot per family."""
    valid = {f: rows for f, rows in results_by_family.items()
             if any(r.get(metric) is not None for r in rows)}
    if not valid:
        return

    n = len(valid)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 4.5 * nrows), dpi=_PLOT_DPI,
                              squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, (fam, rows) in enumerate(valid.items()):
        ax = axes_flat[idx]
        layers = [r["layer"] for r in rows if r.get(metric) is not None]
        vals   = [r[metric]  for r in rows if r.get(metric) is not None]
        n_cls  = next((r.get("n_classes") for r in rows if r.get("n_classes")), 2)
        baseline = 1.0 / max(n_cls, 1)
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.axhline(baseline, color="grey", linewidth=1, linestyle="--", alpha=0.5,
                   label=f"chance  1/{n_cls}")
        _add_embed_layer1_separator(ax)
        ax.plot(layers, vals, color=color, **_LINE_KW)
        ax.set_title(f"{fam}  ({n_cls} attribute classes)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        _ax_layer_ticks(ax, max(layers) + 1 if layers else 33)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(valid):]:
        ax.set_visible(False)

    fig.suptitle(f"Within-family attribute decodability  |  scope: {scope}", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_context_split_curve(
    rows:        list[dict],
    scope:       str,
    output_path: Path,
    metric:      str = "balanced_accuracy",
):
    """Plot D: partner vs explicit decodability for sexual orientation."""
    layers = [r["layer"] for r in rows if r.get(metric) is not None]
    vals   = [r[metric]  for r in rows if r.get(metric) is not None]
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=_PLOT_DPI)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.4, label="chance (0.5)")
    _add_embed_layer1_separator(ax)
    ax.plot(layers, vals, color=FAMILY_COLORS["sexual_orientation"], **_LINE_KW,
            label="partner vs explicit")
    ax.set_ylabel(metric.replace("_", " ").title())
    # Req 12 & 20: updated title
    ax.set_title(
        f"SO Context Split: Partner-Neutralized vs Explicit-Orientation Variants  |  scope: {scope}\n"
        f"partner-neutralized vs explicit-orientation  "
        f"(Stage 4 B3 showed strong behavioral differences between these)"
    )
    _ax_layer_ticks(ax, max(layers) + 1)
    ax.set_ylim(0.35, 1.05)
    ax.legend(framealpha=0.85, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary_heatmap(
    all_results: dict,
    scope: str,
    output_path: Path,
    metric: str = "normalized_decoding_gain",
    cmap: str = "RdBu",
    vcenter: float | None = 0.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
):
    """Heatmap: rows = (family, task), cols = layers, color = metric value.

    Req 10: accept metric, cmap, vcenter, vmin, vmax parameters.
    """
    if not all_results:
        return

    n_layers = max(
        max((r.get("layer", 0) for r in rows), default=0)
        for rows in all_results.values()
    ) + 1

    row_labels, matrix = [], []
    for (fam, task), rows in sorted(all_results.items()):
        vals = [None] * n_layers
        for r in rows:
            l = r.get("layer")
            v = r.get(metric)
            if l is not None and v is not None:
                vals[l] = v
        row_labels.append(f"{fam} / {task}")
        matrix.append(vals)

    if not matrix:
        return

    arr = np.array([[v if v is not None else np.nan for v in row] for row in matrix])

    fig_w = max(10, n_layers * 0.38)
    fig_h = max(3, len(row_labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=_PLOT_DPI)

    if vcenter is not None:
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                       norm=norm, cmap=cmap)
    else:
        im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in range(n_layers)], fontsize=6, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Layer  (0 = embedding, 1..N = transformer)")
    _metric_display = {
        "normalized_decoding_gain": "Normalized Decoding Gain",
        "balanced_accuracy":        "Balanced Accuracy",
        "accuracy":                 "Accuracy",
    }.get(metric, metric.replace("_", " ").title())
    ax.set_title(f"{_metric_display} heatmap  |  scope: {scope}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, label=_metric_display)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 4.5: layerwise linear-probe decoding of demographic signal.")
    p.add_argument("--model_path",  required=True)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--device",   default="auto")
    p.add_argument("--dtype",    default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_pairs",   type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--families",    nargs="+",
                   default=["sex_gender", "gender_identity", "race", "sexual_orientation"],
                   help="Demographic families to include.")
    p.add_argument("--feature_scope", nargs="+", default=["final_token"],
                   choices=FEATURE_SCOPES,
                   help="Feature pooling scope(s). All requested scopes are extracted in one pass.")
    p.add_argument("--window_radius", type=int, default=2,
                   help="Token window radius for edited_window_mean scope.")
    p.add_argument("--probe_tasks",   nargs="+",
                   default=["orig_vs_cf", "family_identity",
                             "within_family_attribute", "context_split"],
                   choices=["orig_vs_cf", "family_identity",
                             "within_family_attribute", "context_split",
                             "focal_vs_control_cf"])
    p.add_argument("--min_examples_per_class", type=int, default=20,
                   help="Minimum examples per class to include in within_family_attribute task.")
    p.add_argument("--test_size",   type=float, default=0.25)
    p.add_argument("--n_bootstrap", type=int,   default=0,
                   help="Bootstrap CI iterations. 0 = disabled (not yet implemented).")
    p.add_argument("--balance_classes", action="store_true",
                   help="Undersample majority classes in training split.")
    p.add_argument("--include_controls", action="store_true",
                   help="Load control pairs separately for focal_vs_control_cf task.")
    # Req 8: probe hyperparameters
    p.add_argument("--probe_C", type=float, default=1.0,
                   help="Logistic regression C (inverse regularization strength).")
    p.add_argument("--probe_max_iter", type=int, default=1000,
                   help="Max iterations for logistic regression solver.")
    p.add_argument("--probe_class_weight", choices=["none", "balanced"], default="none",
                   help="Class weight for logistic regression. 'none' = uniform.")
    # Req 6: min prefix/suffix tokens
    p.add_argument("--min_prefix_tokens", type=int, default=2,
                   help="Minimum prefix tokens needed to compute prefix_mean scope.")
    p.add_argument("--min_suffix_tokens", type=int, default=2,
                   help="Minimum suffix tokens needed to compute suffix_mean scope.")
    # Req 11: within_family exclude labels
    p.add_argument("--within_family_exclude_labels", nargs="*", default=[],
                   help="Labels to exclude before within_family_attribute class counting.")
    p.add_argument("--so_within_family", action="store_true",
                   help="Enable within_family_attribute for sexual_orientation. "
                        "Disabled by default: SO labels mix orientation-identity "
                        "(gay, straight) and relationship-context (partner) variants, "
                        "making attribute-level decoding hard to interpret. "
                        "Only enable if you understand the label heterogeneity.")
    # Req 19: save_pair_level_metadata (new name) + deprecated alias
    p.add_argument("--save_pair_level_metadata", action="store_true",
                   dest="save_pair_level_metadata",
                   help="Save per-pair feature metadata CSV (not raw vectors).")
    p.add_argument("--save_pair_level_features", action="store_true",
                   dest="save_pair_level_metadata",
                   help=argparse.SUPPRESS)  # deprecated alias for --save_pair_level_metadata
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # Req 18: bootstrap check
    if args.n_bootstrap > 0:
        raise NotImplementedError(
            f"--n_bootstrap {args.n_bootstrap} was requested, but bootstrap CI is not yet implemented. "
            "Set --n_bootstrap 0 (default) to skip bootstrap inference.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_warnings: list[str] = []

    # Req 8: resolve probe_class_weight
    probe_class_weight = None if args.probe_class_weight == "none" else args.probe_class_weight

    # -----------------------------------------------------------------------
    # 1. Load and sample data
    # -----------------------------------------------------------------------
    print("Loading pairs...")
    # Req 15: load focal pairs and control pairs separately
    all_pairs_focal, assignment_log_focal = load_pairs(args.data_path, include_controls=False)
    print(f"  Total focal pairs loaded: {len(all_pairs_focal)}")

    # Load control pairs if requested
    control_pairs_all: list = []
    if args.include_controls:
        all_pairs_with_ctrl, _ = load_pairs(args.data_path, include_controls=True)
        control_pairs_all = [p for p in all_pairs_with_ctrl if p["family"] == "control"]
        print(f"  Total control pairs loaded: {len(control_pairs_all)}")

    for fam, info in sorted(assignment_log_focal.items()):
        print(f"    {fam}: {info['count']}")

    requested_families = args.families
    sampled_pairs: list = []
    family_sample_counts: dict = {}
    for fam in requested_families:
        fam_all = [p for p in all_pairs_focal if p["family"] == fam]
        if not fam_all:
            w = f"No pairs found for family '{fam}'."
            print(f"  Warning: {w}")
            global_warnings.append(w)
            continue
        if len(fam_all) > args.max_pairs:
            idx = rng.choice(len(fam_all), args.max_pairs, replace=False)
            fam_all = [fam_all[i] for i in sorted(idx)]
        family_sample_counts[fam] = len(fam_all)
        sampled_pairs.extend(fam_all)

    # Req 15: sample control pairs separately
    sampled_control_pairs: list = []
    if control_pairs_all:
        if len(control_pairs_all) > args.max_pairs:
            ctrl_idx = rng.choice(len(control_pairs_all), args.max_pairs, replace=False)
            sampled_control_pairs = [control_pairs_all[i] for i in sorted(ctrl_idx)]
        else:
            sampled_control_pairs = list(control_pairs_all)
        print(f"  Sampled {len(sampled_control_pairs)} control pairs.")

    print(f"  Sampled: {len(sampled_pairs)} focal pairs from {list(family_sample_counts)}")
    if not sampled_pairs:
        print("ERROR: no pairs after sampling. Check --data_path and --families.")
        return

    # -----------------------------------------------------------------------
    # 2. Load model
    # -----------------------------------------------------------------------
    model, tok, device = load_model_and_tok(args.model_path, args.device, args.dtype)
    # n_hidden_states = n_transformer_layers + 1 (embedding is index 0)
    n_hidden_states      = model.config.num_hidden_layers + 1
    n_transformer_layers = model.config.num_hidden_layers
    hidden_size          = model.config.hidden_size
    print(f"  n_hidden_states (incl. embedding): {n_hidden_states}")

    # -----------------------------------------------------------------------
    # 3. Extract features
    # -----------------------------------------------------------------------
    # Req 15: combine focal + control for extraction
    all_extraction_pairs = sampled_pairs + sampled_control_pairs

    print(f"\nExtracting features (scopes={args.feature_scope})...")
    feature_bank = extract_features(
        model, tok, all_extraction_pairs, device,
        scopes=args.feature_scope,
        window_radius=args.window_radius,
        min_prefix_tokens=args.min_prefix_tokens,
        min_suffix_tokens=args.min_suffix_tokens,
    )
    print(f"  Feature bank: {len(feature_bank)} pairs extracted.")

    n_fallback = sum(1 for r in feature_bank if r["edit_fallback"])
    if n_fallback > 0:
        w = (f"{n_fallback}/{len(feature_bank)} pairs had no detectable token-level edit span "
             f"under prefix/suffix alignment (orig and cf tokenize to an identical sequence, "
             f"or the differing region could not be localized); span-derived scopes "
             f"(edited_span_mean, edited_window_mean, prefix_mean, suffix_mean) "
             f"are unavailable (stored as None) for those pairs.")
        print(f"  Warning: {w}")
        global_warnings.append(w)

    # Count pairs where at least one layer has omitted features per span scope
    n_pairs_with_omitted_features_by_scope: dict[str, int] = {}
    for sc in args.feature_scope:
        if sc in _SPAN_SCOPES:
            omitted = sum(
                1 for r in feature_bank
                if any(r["orig_feats"].get(sc, {}).get(l) is None
                       for l in r["orig_feats"].get(sc, {}))
            )
            n_pairs_with_omitted_features_by_scope[sc] = omitted

    if args.save_pair_level_metadata:
        pair_meta_rows = [
            {k: v for k, v in r.items() if not k.endswith("_feats")}
            for r in feature_bank
        ]
        save_csv(pair_meta_rows, output_dir / "pair_level_metadata.csv")
        print(f"  Saved pair-level metadata to pair_level_metadata.csv")

    # Free model memory — not needed after extraction
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 4. Run probes per scope
    # -----------------------------------------------------------------------
    # Req 1: all_results_by_scope keyed by scope name (outer scope)
    all_results_by_scope: dict[str, dict] = {}

    # Track task sample sizes and invalid layer counts
    task_sample_sizes: dict[str, dict] = {}
    invalid_layer_counts: dict[str, dict] = {}

    for scope in args.feature_scope:
        print(f"\n{'='*60}")
        print(f"PROBES  |  scope: {scope}")
        print("=" * 60)

        # Req 1: fresh scope_results dict for each scope
        scope_results: dict = {}
        task_sample_sizes[scope] = {}
        invalid_layer_counts[scope] = {}

        # ---- Task A: orig_vs_cf per family ----
        if "orig_vs_cf" in args.probe_tasks:
            print("\n  [A] orig_vs_cf  (Manipulation Detectability)")
            orig_vs_cf_by_family = {}
            for fam in requested_families:
                fam_bank = [r for r in feature_bank if r["family"] == fam]
                if not fam_bank:
                    continue
                print(f"    {fam}  n={len(fam_bank)}...", end=" ", flush=True)
                rows = run_layerwise_probe(
                    bank=fam_bank, task="orig_vs_cf", scope=scope,
                    family=fam, families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C,
                    probe_max_iter=args.probe_max_iter,
                    probe_class_weight=probe_class_weight,
                )
                orig_vs_cf_by_family[fam] = rows
                scope_results[(fam, "orig_vs_cf")] = rows

                key = f"{fam}__orig_vs_cf"
                s_samp = _task_sample_summary(rows)
                if s_samp:
                    task_sample_sizes[scope][key] = s_samp
                invalid_layer_counts[scope][key] = sum(1 for r in rows if r.get("error"))

                exp_dir = output_dir / fam / f"orig_vs_cf_{scope}"
                save_experiment(exp_dir, metadata={
                    "task":               "orig_vs_cf",
                    "task_display_name":  "orig_vs_cf_manipulation_detectability",
                    "family":             fam,
                    "feature_scope":      scope,
                    "n_pairs":            len(fam_bank),
                    "n_bank_pairs":       len(fam_bank),
                    "bank_filter":        f"family == '{fam}'",
                    "n_hidden_states":    n_hidden_states,
                    "hidden_size":        hidden_size,
                    "class_labels":       ["orig", "cf"],
                    "grouped_split":      True,
                    "test_size":          args.test_size,
                    "seed":               args.seed,
                    "stage3_expected_peak_description": STAGE3_EDIT_MASS_PEAK.get(fam),
                    "controls_excluded": True,
                    "controls_excluded_reason": (
                        "orig_vs_cf is a per-focal-family task; control pairs are not included."
                    ),
                    "interpretive_note": (
                        "This task detects whether the model's hidden states reflect "
                        "the counterfactual edit (manipulation detectability). It does NOT "
                        "measure latent demographic representation per se."
                    ),
                }, per_layer_rows=rows)

                s = summarize_layers(rows)
                if s:
                    print(f"peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")
                else:
                    print("no valid layers")

            plot_orig_vs_cf_curves(
                orig_vs_cf_by_family, scope=scope,
                output_path=plots_dir / f"01_orig_vs_cf_{scope}.png",
            )

        # ---- Task B: family identity ----
        if "family_identity" in args.probe_tasks:
            print("\n  [B] family_identity")
            focal_bank = [r for r in feature_bank if r["family"] in set(requested_families)]
            rows = run_layerwise_probe(
                bank=focal_bank, task="family_identity", scope=scope,
                family=None, families_global=requested_families,
                n_hidden_states=n_hidden_states,
                test_size=args.test_size, seed=args.seed,
                min_examples_per_class=args.min_examples_per_class,
                balance_classes=args.balance_classes,
                probe_C=args.probe_C,
                probe_max_iter=args.probe_max_iter,
                probe_class_weight=probe_class_weight,
            )
            scope_results[("global", "family_identity")] = rows

            key = "global__family_identity"
            s_samp = _task_sample_summary(rows)
            if s_samp:
                task_sample_sizes[scope][key] = s_samp
            invalid_layer_counts[scope][key] = sum(1 for r in rows if r.get("error"))

            exp_dir = output_dir / "global" / f"family_identity_{scope}"
            save_experiment(exp_dir, metadata={
                "task":             "family_identity",
                "task_display_name": "family_identity_representational_separability",
                "feature_scope":    scope,
                "families":         requested_families,
                "n_classes":        len(requested_families),
                "source":           "CF hidden states only",
                "n_bank_pairs":     len(focal_bank),
                "bank_filter":      "family in requested_families (controls excluded)",
                "grouped_split":    True,
                "controls_excluded": True,
                "controls_excluded_reason": (
                    "family_identity uses only the focal families in --families; "
                    "control pairs are not a valid demographic family class for this task."
                ),
                "contrast_with_orig_vs_cf": (
                    "Unlike orig_vs_cf (manipulation detectability), family_identity asks "
                    "whether different demographic families occupy separable regions in the "
                    "model's CF representation space. High accuracy here reflects "
                    "representational separability, not edit detectability."
                ),
                "note": (
                    "High accuracy here means different demographic families occupy "
                    "separable representational regions. Decodability != causal necessity."
                ),
            }, per_layer_rows=rows)

            s = summarize_layers(rows)
            if s:
                print(f"    peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")

            plot_family_identity_curve(
                rows, scope=scope,
                output_path=plots_dir / f"02_family_identity_{scope}.png",
            )

        # ---- Task C: within_family_attribute ----
        if "within_family_attribute" in args.probe_tasks:
            print("\n  [C] within_family_attribute")
            within_by_family = {}
            for fam in requested_families:
                fam_bank = [r for r in feature_bank if r["family"] == fam]
                if not fam_bank:
                    continue
                excl = set(args.within_family_exclude_labels or [])
                counts = Counter(
                    r["attr_val"] for r in fam_bank if r["attr_val"] not in excl)
                valid_cls = [cls for cls, cnt in counts.items()
                             if cnt >= args.min_examples_per_class]
                if len(valid_cls) < 2:
                    print(f"    {fam}: skip  ({len(valid_cls)} valid class(es) with "
                          f">={args.min_examples_per_class} examples)")
                    continue
                # SO within-family is opt-in due to label heterogeneity (req 11)
                if fam == "sexual_orientation" and not args.so_within_family:
                    w = (
                        "within_family_attribute for sexual_orientation skipped by default "
                        "(labels mix orientation-identity and relationship-context variants). "
                        "Use --so_within_family to enable."
                    )
                    print(f"    {fam}: skipped  (use --so_within_family to enable)")
                    global_warnings.append(w)
                    continue
                print(f"    {fam}  ({len(valid_cls)} classes)...", end=" ", flush=True)
                rows = run_layerwise_probe(
                    bank=fam_bank, task="within_family_attribute", scope=scope,
                    family=fam, families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C,
                    probe_max_iter=args.probe_max_iter,
                    probe_class_weight=probe_class_weight,
                    within_family_exclude_labels=args.within_family_exclude_labels,
                )
                within_by_family[fam] = rows
                scope_results[(fam, "within_family_attribute")] = rows

                key = f"{fam}__within_family_attribute"
                s_samp = _task_sample_summary(rows)
                if s_samp:
                    task_sample_sizes[scope][key] = s_samp
                invalid_layer_counts[scope][key] = sum(1 for r in rows if r.get("error"))

                # Req 11: save all class counts; add caution note for sexual_orientation
                exp_meta = {
                    "task":                   "within_family_attribute",
                    "family":                 fam,
                    "feature_scope":          scope,
                    "all_class_counts":       dict(counts),
                    "valid_classes":          valid_cls,
                    "dropped_classes":        [cls for cls in counts if cls not in valid_cls],
                    "excluded_labels":        list(excl),
                    "min_examples_per_class": args.min_examples_per_class,
                    "n_bank_pairs":           len(fam_bank),
                    "bank_filter":            f"family == '{fam}'",
                    "controls_excluded":      True,
                    "controls_excluded_reason": (
                        "within_family_attribute is a per-focal-family task; "
                        "control pairs are not included."
                    ),
                    "caution_attribute_encoding": (
                        "Performance may partly reflect lexical regularities of the edit "
                        "templates (e.g. shared morphology, word length, or topic within "
                        "an attribute class) rather than abstract demographic encoding. "
                        "Interpret as 'attribute-conditioned edit decodability', not as "
                        "pure representation of demographic attributes per se."
                    ),
                }
                if fam == "sexual_orientation":
                    exp_meta["caution_label_heterogeneity"] = (
                        "Labels for sexual_orientation may mix orientation-identity labels "
                        "(gay, straight) with relationship-context labels (partner). "
                        "Interpret within-family attribute task cautiously for this family."
                    )
                save_experiment(exp_dir=output_dir / fam / f"within_family_attribute_{scope}",
                                metadata=exp_meta,
                                per_layer_rows=rows)

                s = summarize_layers(rows)
                if s:
                    print(f"peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")
                else:
                    print("no valid layers")

            if within_by_family:
                plot_within_family_curves(
                    within_by_family, scope=scope,
                    output_path=plots_dir / f"03_within_family_attribute_{scope}.png",
                )

        # ---- Task D: context_split (SO only) ----
        if "context_split" in args.probe_tasks and "sexual_orientation" in requested_families:
            so_bank = [r for r in feature_bank if r["family"] == "sexual_orientation"]
            n_partner  = sum(1 for r in so_bank if r.get("split_label") == "partner")
            n_explicit = sum(1 for r in so_bank if r.get("split_label") == "explicit")
            n_other    = sum(1 for r in so_bank if r.get("split_label") == "other")
            print(f"\n  [D] context_split  "
                  f"partner={n_partner}  explicit={n_explicit}  other_dropped={n_other}")

            if n_partner >= args.min_examples_per_class and n_explicit >= args.min_examples_per_class:
                rows = run_layerwise_probe(
                    bank=so_bank, task="context_split", scope=scope,
                    family="sexual_orientation", families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C,
                    probe_max_iter=args.probe_max_iter,
                    probe_class_weight=probe_class_weight,
                )
                scope_results[("sexual_orientation", "context_split")] = rows

                key = "sexual_orientation__context_split"
                s_samp = _task_sample_summary(rows)
                if s_samp:
                    task_sample_sizes[scope][key] = s_samp
                invalid_layer_counts[scope][key] = sum(1 for r in rows if r.get("error"))

                exp_dir = output_dir / "sexual_orientation" / f"context_split_{scope}"
                save_experiment(exp_dir, metadata={
                    "task":             "context_split",
                    "family":           "sexual_orientation",
                    "feature_scope":    scope,
                    "class_labels":     ["partner", "explicit"],
                    "n_bank_pairs":     len(so_bank),
                    "bank_filter":      "family == 'sexual_orientation'",
                    "n_partner":        n_partner,
                    "n_explicit":       n_explicit,
                    "n_other_dropped":  n_other,
                    # Req 12: updated task description
                    "task_description": (
                        "Classifies partner-neutralized SO edits (e.g. 'partner') vs "
                        "explicit-orientation edits (e.g. 'gay', 'straight'). High decodability "
                        "means these variant types are representationally separable, not that "
                        "orientation identity itself is decodable."
                    ),
                    "note_stage4_b3": (
                        "Stage 4 B3 showed strong behavioral differences between "
                        "partner-framed and explicit-identity SO pairs. If decodability "
                        "is high here too, the split is encoded in representations, "
                        "not only in behavior."
                    ),
                }, per_layer_rows=rows)

                s = summarize_layers(rows)
                if s:
                    print(f"    peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")

                plot_context_split_curve(
                    rows, scope=scope,
                    output_path=plots_dir / f"04_context_split_{scope}.png",
                )
            else:
                w = (f"context_split skipped: n_partner={n_partner}, n_explicit={n_explicit} "
                     f"(both must be >= {args.min_examples_per_class})")
                print(f"    Warning: {w}")
                global_warnings.append(w)

        # ---- Task E: focal_vs_control_cf ----
        if "focal_vs_control_cf" in args.probe_tasks:
            ctrl_in_bank = [r for r in feature_bank if r["family"] == "control"]
            if not ctrl_in_bank:
                w = "focal_vs_control_cf skipped: no control pairs in feature bank. Use --include_controls."
                print(f"\n  Warning: {w}")
                global_warnings.append(w)
            else:
                print(f"\n  [E] focal_vs_control_cf  n_control={len(ctrl_in_bank)}")
                for fam in requested_families:
                    fam_bank = [r for r in feature_bank if r["family"] == fam]
                    if not fam_bank:
                        continue
                    print(f"    {fam}  n_focal={len(fam_bank)}...", end=" ", flush=True)
                    focal_ctrl_bank = [r for r in feature_bank
                                       if r["family"] in {fam, "control"}]
                    rows = run_layerwise_probe(
                        bank=focal_ctrl_bank, task="focal_vs_control_cf", scope=scope,
                        family=fam, families_global=requested_families,
                        n_hidden_states=n_hidden_states,
                        test_size=args.test_size, seed=args.seed,
                        min_examples_per_class=args.min_examples_per_class,
                        balance_classes=args.balance_classes,
                        probe_C=args.probe_C,
                        probe_max_iter=args.probe_max_iter,
                        probe_class_weight=probe_class_weight,
                    )
                    scope_results[(fam, "focal_vs_control_cf")] = rows

                    key = f"{fam}__focal_vs_control_cf"
                    s_samp = _task_sample_summary(rows)
                    if s_samp:
                        task_sample_sizes[scope][key] = s_samp
                    invalid_layer_counts[scope][key] = sum(1 for r in rows if r.get("error"))

                    exp_dir = output_dir / fam / f"focal_vs_control_cf_{scope}"
                    save_experiment(exp_dir, metadata={
                        "task":          "focal_vs_control_cf",
                        "family":        fam,
                        "feature_scope": scope,
                        "class_labels":  ["control", "focal"],
                        "n_focal":         len(fam_bank),
                        "n_control":       len(ctrl_in_bank),
                        "n_bank_pairs":    len(focal_ctrl_bank),
                        "bank_filter":     f"family in {{'{fam}', 'control'}}",
                        "controls_usage": (
                            "control pairs serve as label=0 (negative class); "
                            "focal CF pairs serve as label=1 (positive class)."
                        ),
                        "controls_role": (
                            "Controls provide a non-demographic CF baseline. "
                            "High accuracy means focal CFs have distinct hidden-state "
                            "representations from neutral/irrelevant edits — "
                            "not merely that an edit was made."
                        ),
                        "caution_confound": (
                            "High accuracy may reflect demographic-specific representation "
                            "AND/OR structural differences between focal and control "
                            "counterfactuals (e.g., edit length, syntactic category, topic). "
                            "These confounds are not separable from this task alone."
                        ),
                    }, per_layer_rows=rows)

                    s = summarize_layers(rows)
                    if s:
                        print(f"peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")
                    else:
                        print("no valid layers")

        # Req 10: generate two heatmaps per scope
        if scope_results:
            plot_summary_heatmap(
                scope_results, scope=scope,
                output_path=plots_dir / f"05_summary_heatmap_normalized_{scope}.png",
                metric="normalized_decoding_gain",
                cmap="RdBu",
                vcenter=0.0,
                vmin=-1.0,
                vmax=1.0,
            )
            plot_summary_heatmap(
                scope_results, scope=scope,
                output_path=plots_dir / f"05b_summary_heatmap_raw_{scope}.png",
                metric="balanced_accuracy",
                cmap="RdYlGn",
                vcenter=None,
                vmin=0.0,
                vmax=1.0,
            )

        # Req 1: store this scope's results into the outer dict
        all_results_by_scope[scope] = scope_results

    # -----------------------------------------------------------------------
    # 5. Summary JSON
    # -----------------------------------------------------------------------
    print("\nBuilding summary.json...")

    # Req 17: rich localization_peaks structure
    localization_peaks: dict[str, dict] = {}
    for scope, scope_results in all_results_by_scope.items():
        localization_peaks[scope] = {}
        for (fam_or_global, task), rows in scope_results.items():
            key = f"{fam_or_global}__{task}"
            s = summarize_layers(rows)
            if s:
                peak_layer = s["peak_layer"]
                observed_region = layer_to_region(peak_layer)
                fam_name = fam_or_global if fam_or_global != "global" else None
                expected_region = STAGE3_EXPECTED_REGION.get(fam_name) if fam_name else None
                if expected_region is None:
                    alignment_flag = None
                elif isinstance(expected_region, list):
                    alignment_flag = observed_region in expected_region
                else:
                    alignment_flag = (observed_region == expected_region)
                localization_peaks[scope][key] = {
                    **s,
                    "observed_region":          observed_region,
                    "stage3_expected_region":   expected_region,
                    "stage3_alignment_flag":    alignment_flag,
                }

    n_extracted_focal   = sum(1 for r in feature_bank if r["family"] != "control")
    n_extracted_control = sum(1 for r in feature_bank if r["family"] == "control")
    summary = {
        "script":  SCRIPT_NAME,
        "version": VERSION,
        "args":    vars(args),
        "data": {
            "n_loaded_focal_pairs":                   len(all_pairs_focal),
            "n_sampled_focal_pairs":                  len(sampled_pairs),
            "n_sampled_control_pairs":                len(sampled_control_pairs),
            "n_extracted_total_pairs":                len(feature_bank),
            "n_extracted_focal_pairs":                n_extracted_focal,
            "n_extracted_control_pairs":              n_extracted_control,
            "controls_loaded":                        bool(args.include_controls),
            "family_sample_counts":                   family_sample_counts,
            "n_edit_fallbacks":                       n_fallback,
            "n_pairs_with_omitted_features_by_scope": n_pairs_with_omitted_features_by_scope,
        },
        "controls": {
            "controls_requested":        bool(args.include_controls),
            "n_sampled_control_pairs":   len(sampled_control_pairs),
            "n_extracted_control_pairs": n_extracted_control,
            "control_participation": {
                "focal_vs_control_cf":     "label=0 (negative class); requires --include_controls",
                "family_identity":         "excluded — task uses focal families only",
                "within_family_attribute": "excluded — task is per focal family",
                "orig_vs_cf":              "excluded — task is per focal family",
                "context_split":           "excluded — SO-only task",
            },
        },
        "task_settings": {
            "so_within_family_attribute_enabled": bool(args.so_within_family),
        },
        "model": {
            "n_hidden_states":           n_hidden_states,
            "n_transformer_layers":      n_transformer_layers,
            "hidden_size":               hidden_size,
            "embedding_at_index_0":      True,
            "layer_region_boundaries": {
                "early_end": EARLY_END_LAYER,
                "mid_end":   MID_END_LAYER,
                "regions":   "embedding=0; early=1..3; mid=4..15; late=16+",
            },
        },
        # Req 8: probe hyperparameters in summary
        "probe_hyperparameters": {
            "C":               args.probe_C,
            "max_iter":        args.probe_max_iter,
            "class_weight":    args.probe_class_weight,
            "balance_classes": args.balance_classes,
        },
        "localization_peaks":   localization_peaks,
        "task_sample_sizes":    task_sample_sizes,
        "invalid_layer_counts": invalid_layer_counts,
        "warnings": global_warnings,
        "interpretive_notes": {
            "decodability_vs_causality": (
                "Probe accuracy measures whether demographic information is PRESENT and "
                "linearly readable at a given layer. It does NOT imply that causally "
                "intervening on that layer (e.g., ablation, patching) would change "
                "model behavior. The two properties can dissociate."
            ),
            "high_decode_weak_causal_means": (
                "If Stage 4 ablation recovery was weak but Stage 4.5 decodability is high, "
                "that suggests distributed or redundant representation: the information is "
                "there, but removing it from one channel does not remove it from the system."
            ),
            "convergent_localization": (
                "If Stage 3 edit-mass peaks, Stage 4 residual patching recovery peaks, "
                "and Stage 4.5 decoding peaks all align at the same layers, that provides "
                "convergent evidence for localization of the demographic signal."
            ),
            "so_hypothesis": (
                "Sexual orientation may show weaker early decodability and stronger "
                "mid/late-layer decodability, consistent with Stage 4 B1 showing diffuse "
                "recovery and Stage 4 B3 showing a strong partner-vs-explicit behavioral split."
            ),
            "stage3_reference": (
                "Stage 3 edit-mass heatmaps showed early-layer peaks for gender_identity and race. "
                "Compare stage45_results localization_peaks to STAGE3_EDIT_MASS_PEAK below."
            ),
            "orig_vs_cf_framing": (
                "The orig_vs_cf task (a.k.a. manipulation_detectability) classifies original vs "
                "counterfactually-edited hidden states. High accuracy means the model encodes "
                "the edit, not necessarily that it encodes the demographic attribute itself."
            ),
            "context_split_framing": (
                "The context_split task classifies partner-neutralized SO edits vs "
                "explicit-orientation edits. High decodability means these variant types are "
                "representationally separable, not that orientation identity itself is decodable."
            ),
            "within_family_attribute_caveat": (
                "within_family_attribute accuracy may partly reflect lexical regularities of "
                "the edit templates (e.g. shared morphology, word length, or topic within an "
                "attribute class) rather than abstract demographic encoding. Interpret as "
                "'attribute-conditioned edit decodability', not pure representation of "
                "demographic attributes. This caveat applies to all families."
            ),
            "focal_vs_control_cf_caveat": (
                "focal_vs_control_cf accuracy may reflect demographic-specific representation "
                "AND/OR structural differences in how focal vs control counterfactuals were "
                "constructed (edit length, syntactic category, topic). These confounds are not "
                "fully separable from this task alone."
            ),
        },
        "stage3_expected_peaks":  STAGE3_EDIT_MASS_PEAK,
        "stage3_expected_regions": STAGE3_EXPECTED_REGION,
        "stage3_expected_regions_note": (
            "Values may be a single string or a list of acceptable region strings. "
            "sex_gender accepts ['early', 'mid'] (Stage 3: early-to-mid peaks). "
            "sexual_orientation accepts ['mid', 'late'] (Stage 3: diffuse / mid-late)."
        ),
    }
    save_json(summary, output_dir / "summary.json")
    print(f"Done.  Results → {output_dir}")


if __name__ == "__main__":
    main()
