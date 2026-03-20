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

Architecture
------------
Phase A (extraction): load model, iterate pairs one at a time, pool hidden states
  layer-by-layer, flush compact numpy chunks to disk every N pairs.
Phase B (probing): unload model, load cached chunks per scope, fit probes.

Usage
-----
  # Full run (extract + probe):
  python analyze_stage45_layerwise_decoding.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage45_results \\
    --device auto --max_pairs 200 \\
    --feature_scope final_token mean_pool

  # Extraction only:
  python analyze_stage45_layerwise_decoding.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage45_results \\
    --extract_only

  # Resume after interruption (same command, skips completed pairs):
  python analyze_stage45_layerwise_decoding.py \\
    --model_path models/llama2-13b \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage45_results

  # Probe only (no model loading):
  python analyze_stage45_layerwise_decoding.py \\
    --data_path cf_v6_balanced.json \\
    --output_dir stage45_results \\
    --probe_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import tempfile
import time
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
VERSION = "3.0"

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

TRUNCATION_MAX_LENGTH = 2048

# ---------------------------------------------------------------------------
# Stage 3 alignment constants
# ---------------------------------------------------------------------------

EARLY_END_LAYER = 4
MID_END_LAYER   = 16

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
# layer_to_region helper
# ---------------------------------------------------------------------------

def layer_to_region(layer_idx: int) -> str:
    if layer_idx == 0:
        return "embedding"
    if layer_idx < EARLY_END_LAYER:
        return "early"
    if layer_idx < MID_END_LAYER:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Uniform-chance / normalized gain helpers
# ---------------------------------------------------------------------------

def compute_normalized_gain(balanced_accuracy: float, n_classes: int) -> float:
    uniform = 1.0 / max(n_classes, 1)
    denom = 1.0 - uniform
    return (balanced_accuracy - uniform) / denom if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Split validity check
# ---------------------------------------------------------------------------

def validate_split(y_train: np.ndarray, y_test: np.ndarray) -> str | None:
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
# balance_training_set helper
# ---------------------------------------------------------------------------

def balance_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    n_orig, n_cf = len(orig_ids), len(cf_ids)

    prefix_len = 0
    for i in range(min(n_orig, n_cf)):
        if orig_ids[i] == cf_ids[i]:
            prefix_len += 1
        else:
            break

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

    if edit_start >= edit_end_orig or edit_start >= edit_end_cf:
        return 0, n_orig, n_cf, True

    return edit_start, edit_end_orig, edit_end_cf, False


# ---------------------------------------------------------------------------
# Feature pooling
# ---------------------------------------------------------------------------

def pool_hidden(
    hidden: torch.Tensor,
    scope:  str,
    edit_start:  int,
    edit_end:    int,
    window_radius: int = 2,
) -> np.ndarray:
    h = hidden[0]
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
        e = max(1, edit_start)
        return _safe_slice(0, e)
    if scope == "suffix_mean":
        return _safe_slice(edit_end, seq_len)
    raise ValueError(f"Unknown feature scope: {scope!r}")


# ---------------------------------------------------------------------------
# Pair key for checkpoint/resume
# ---------------------------------------------------------------------------

def make_pair_key(pair: dict) -> str:
    """Stable unique key for a counterfactual pair."""
    raw = f"{pair['qid']}|{pair['family']}|{pair['attr_val']}|{pair['cf_prompt']}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Single-pair extraction (memory-safe)
# ---------------------------------------------------------------------------

def extract_single_pair(
    model, tok, pair: dict, device: str, scopes: list[str],
    window_radius: int = 2, min_prefix_tokens: int = 2, min_suffix_tokens: int = 2,
) -> tuple[dict, dict, dict, dict]:
    """Extract pooled features for one pair with aggressive memory management.

    Returns:
        orig_pooled:  {scope: np.ndarray [n_layers, hidden_size]}
        cf_pooled:    {scope: np.ndarray [n_layers, hidden_size]}
        valid_arrays: {scope: np.ndarray [n_layers] bool}
        pair_meta:    dict with metadata fields
    """
    # Tokenize both prompts (CPU only)
    enc_orig = tok(pair["orig_prompt"], return_tensors="pt",
                   truncation=True, max_length=TRUNCATION_MAX_LENGTH)
    enc_cf = tok(pair["cf_prompt"], return_tensors="pt",
                 truncation=True, max_length=TRUNCATION_MAX_LENGTH)
    orig_ids = enc_orig["input_ids"][0].tolist()
    cf_ids = enc_cf["input_ids"][0].tolist()

    # Edit span detection
    edit_start, edit_end_orig, edit_end_cf, fallback = find_edited_span(orig_ids, cf_ids)

    # Pre-compute per-scope validity (uniform across all layers for a given scope)
    scope_valid: dict[str, bool] = {}
    for sc in scopes:
        if fallback and sc in _SPAN_SCOPES:
            scope_valid[sc] = False
        elif sc == "prefix_mean" and edit_start < min_prefix_tokens:
            scope_valid[sc] = False
        elif sc == "suffix_mean":
            if (len(orig_ids) - edit_end_orig < min_suffix_tokens or
                    len(cf_ids) - edit_end_cf < min_suffix_tokens):
                scope_valid[sc] = False
            else:
                scope_valid[sc] = True
        else:
            scope_valid[sc] = True

    # ---- Forward ORIG: pool layer-by-layer, free GPU tensors incrementally ----
    enc_orig_gpu = {k: v.to(device) for k, v in enc_orig.items()}
    del enc_orig

    with torch.inference_mode():
        out = model(**enc_orig_gpu, output_hidden_states=True)

    hs_list = list(out.hidden_states)
    n_layers = len(hs_list)
    hidden_size = hs_list[0].shape[-1]
    del out, enc_orig_gpu

    orig_pooled = {sc: np.zeros((n_layers, hidden_size), dtype=np.float32) for sc in scopes}

    for layer_idx in range(n_layers):
        h = hs_list[layer_idx].detach().float().cpu()
        hs_list[layer_idx] = None  # release GPU tensor
        for sc in scopes:
            if scope_valid[sc]:
                orig_pooled[sc][layer_idx] = pool_hidden(
                    h, sc, edit_start, edit_end_orig, window_radius)
        del h
    del hs_list

    # ---- Forward CF: same approach ----
    enc_cf_gpu = {k: v.to(device) for k, v in enc_cf.items()}
    del enc_cf

    with torch.inference_mode():
        out = model(**enc_cf_gpu, output_hidden_states=True)

    hs_list = list(out.hidden_states)
    del out, enc_cf_gpu

    cf_pooled = {sc: np.zeros((n_layers, hidden_size), dtype=np.float32) for sc in scopes}

    for layer_idx in range(n_layers):
        h = hs_list[layer_idx].detach().float().cpu()
        hs_list[layer_idx] = None
        for sc in scopes:
            if scope_valid[sc]:
                cf_pooled[sc][layer_idx] = pool_hidden(
                    h, sc, edit_start, edit_end_cf, window_radius)
        del h
    del hs_list

    # Build validity arrays
    valid_arrays = {sc: np.full(n_layers, scope_valid[sc], dtype=bool) for sc in scopes}

    # Build pair metadata
    pair_meta = {
        "pair_key":       make_pair_key(pair),
        "qid":            pair["qid"],
        "family":         pair["family"],
        "attr_val":       pair["attr_val"],
        "attr_val_orig":  pair["attr_val_orig"],
        "attr_norm":      pair["attr_norm"],
        "gold":           pair["gold"],
        "split_label":    (orientation_split_label(pair["attr_norm"])
                           if pair["family"] == "sexual_orientation" else ""),
        "edit_start":     int(edit_start),
        "edit_end_orig":  int(edit_end_orig),
        "edit_end_cf":    int(edit_end_cf),
        "edit_fallback":  int(fallback),
        "n_hidden_states": n_layers,
        "status":         "extracted",
    }

    return orig_pooled, cf_pooled, valid_arrays, pair_meta


# ---------------------------------------------------------------------------
# Chunk I/O
# ---------------------------------------------------------------------------

def _flush_chunk_to_disk(
    cache_dir: Path, chunk_id: str, scopes: list[str],
    buf_orig: dict[str, list], buf_cf: dict[str, list],
    buf_valid: dict[str, list],
):
    """Atomically write a chunk of pooled features to disk as .npz."""
    arrays = {}
    for sc in scopes:
        arrays[f"orig_{sc}"] = np.array(buf_orig[sc], dtype=np.float32)
        arrays[f"cf_{sc}"] = np.array(buf_cf[sc], dtype=np.float32)
        arrays[f"valid_{sc}"] = np.array(buf_valid[sc], dtype=bool)

    chunks_dir = cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunks_dir / f"{chunk_id}.npz"
    tmp_path = chunks_dir / f"{chunk_id}.npz.tmp"
    try:
        # Write to an open file handle so np.savez cannot append .npz
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(chunk_path))  # atomic on POSIX
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


_PAIR_INDEX_FIELDS = [
    "pair_key", "qid", "family", "attr_val", "attr_val_orig", "attr_norm",
    "gold", "split_label", "edit_start", "edit_end_orig", "edit_end_cf",
    "edit_fallback", "n_hidden_states", "chunk_id", "status",
]
_PAIR_INDEX_INT_FIELDS = {"edit_start", "edit_end_orig", "edit_end_cf",
                          "edit_fallback", "n_hidden_states"}


def _save_pair_index(cache_dir: Path, entries: list[dict]):
    path = cache_dir / "pair_index.csv"
    if not entries:
        return
    tmp_path = path.with_suffix(".csv.tmp")
    try:
        with open(tmp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_PAIR_INDEX_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(entries)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_pair_index(cache_dir: Path) -> list[dict]:
    path = cache_dir / "pair_index.csv"
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    for e in entries:
        for field in _PAIR_INDEX_INT_FIELDS:
            val = e.get(field, "")
            if val != "":
                try:
                    e[field] = int(val)
                except (ValueError, TypeError):
                    pass
    return entries


def load_scope_from_cache(
    cache_dir: Path, scope: str, manifest: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all cached data for one scope across all chunks.

    Returns:
        orig:  [n_pairs, n_layers, hidden_size] float32
        cf:    [n_pairs, n_layers, hidden_size] float32
        valid: [n_pairs, n_layers] bool
    """
    orig_parts, cf_parts, valid_parts = [], [], []
    for chunk_info in manifest["chunks"]:
        chunk_path = cache_dir / chunk_info["file"]
        data = np.load(chunk_path)
        orig_parts.append(data[f"orig_{scope}"])
        cf_parts.append(data[f"cf_{scope}"])
        valid_parts.append(data[f"valid_{scope}"])
        data.close()

    if not orig_parts:
        raise ValueError(f"No chunk data found for scope '{scope}'")

    return (
        np.concatenate(orig_parts, axis=0),
        np.concatenate(cf_parts, axis=0),
        np.concatenate(valid_parts, axis=0),
    )


# ---------------------------------------------------------------------------
# Extraction phase
# ---------------------------------------------------------------------------

def run_extraction_phase(
    model, tok, pairs: list[dict], device: str,
    cache_dir: Path, scopes: list[str], chunk_size: int,
    window_radius: int, min_prefix_tokens: int, min_suffix_tokens: int,
    force_reextract: bool,
):
    """Phase A: extract pooled features for all pairs with chunking and resume."""
    n_layers = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "chunks").mkdir(exist_ok=True)

    # ---- Load existing cache state for resume ----
    completed_keys: set[str] = set()
    existing_pair_entries: list[dict] = []
    existing_chunks: list[dict] = []

    manifest_path = cache_dir / "cache_manifest.json"

    if force_reextract:
        chunks_dir = cache_dir / "chunks"
        if chunks_dir.exists():
            for f in chunks_dir.glob("*.npz"):
                f.unlink()
        for cleanup_file in [manifest_path, cache_dir / "pair_index.csv"]:
            if cleanup_file.exists():
                cleanup_file.unlink()
        print("  Force re-extract: cleared existing cache.")

    elif manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        cached_scopes = set(manifest.get("scopes", []))
        requested_scopes = set(scopes)
        if not requested_scopes.issubset(cached_scopes):
            missing = requested_scopes - cached_scopes
            print(f"  ERROR: requested scopes {sorted(missing)} not in cache "
                  f"(cached: {sorted(cached_scopes)})")
            print("  Use --force_reextract to rebuild cache with new scopes.")
            raise SystemExit(1)
        existing_chunks = manifest.get("chunks", [])

        pair_index_path = cache_dir / "pair_index.csv"
        if pair_index_path.exists():
            existing_pair_entries = _load_pair_index(cache_dir)
            completed_keys = {
                e["pair_key"] for e in existing_pair_entries
                if e.get("status") == "extracted"
            }

    # ---- Determine remaining pairs ----
    all_keyed = [(make_pair_key(p), p) for p in pairs]
    remaining = [(k, p) for k, p in all_keyed if k not in completed_keys]

    n_total = len(all_keyed)
    n_cached = n_total - len(remaining)

    print(f"  Extraction: {n_total} total, {n_cached} cached, "
          f"{len(remaining)} to extract (chunk_size={chunk_size})")

    if not remaining:
        print("  All pairs already extracted. Skipping extraction.")
        return

    # ---- Extraction loop ----
    chunk_id_counter = len(existing_chunks)
    new_pair_entries: list[dict] = []
    new_chunks: list[dict] = []
    n_failed = 0

    buf_metas: list[dict] = []
    buf_orig: dict[str, list] = {sc: [] for sc in scopes}
    buf_cf: dict[str, list] = {sc: [] for sc in scopes}
    buf_valid: dict[str, list] = {sc: [] for sc in scopes}

    start_time = time.time()

    for idx, (pair_key, pair) in enumerate(remaining):
        try:
            orig_pooled, cf_pooled, valid_arrays, pair_meta = extract_single_pair(
                model, tok, pair, device, scopes,
                window_radius=window_radius,
                min_prefix_tokens=min_prefix_tokens,
                min_suffix_tokens=min_suffix_tokens,
            )
        except Exception as exc:
            print(f"\n    WARN: pair failed qid={pair.get('qid')} "
                  f"family={pair.get('family')}: {exc}")
            new_pair_entries.append({
                "pair_key": pair_key, "qid": pair["qid"],
                "family": pair["family"], "attr_val": pair["attr_val"],
                "attr_val_orig": pair.get("attr_val_orig", ""),
                "attr_norm": pair.get("attr_norm", ""),
                "gold": pair.get("gold", ""),
                "split_label": "", "edit_start": "", "edit_end_orig": "",
                "edit_end_cf": "", "edit_fallback": "", "n_hidden_states": "",
                "chunk_id": "", "status": "failed",
            })
            n_failed += 1
            continue

        buf_metas.append(pair_meta)
        for sc in scopes:
            buf_orig[sc].append(orig_pooled[sc])
            buf_cf[sc].append(cf_pooled[sc])
            buf_valid[sc].append(valid_arrays[sc])

        del orig_pooled, cf_pooled, valid_arrays

        # ---- Flush when chunk is full ----
        if len(buf_metas) >= chunk_size:
            chunk_id = f"chunk_{chunk_id_counter:05d}"
            _flush_chunk_to_disk(cache_dir, chunk_id, scopes,
                                 buf_orig, buf_cf, buf_valid)

            for meta in buf_metas:
                meta["chunk_id"] = chunk_id
                new_pair_entries.append(meta)

            new_chunks.append({
                "chunk_id": chunk_id,
                "file": f"chunks/{chunk_id}.npz",
                "n_pairs": len(buf_metas),
            })

            # Reset buffers
            buf_metas = []
            buf_orig = {sc: [] for sc in scopes}
            buf_cf = {sc: [] for sc in scopes}
            buf_valid = {sc: [] for sc in scopes}
            chunk_id_counter += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - start_time
            done = idx + 1
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done) / rate if rate > 0 else 0
            print(f"    Flushed {chunk_id} | {done}/{len(remaining)} pairs | "
                  f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

        elif (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"    {idx + 1}/{len(remaining)} pairs ({elapsed:.0f}s)...",
                  end="\r", flush=True)

    # ---- Flush remaining ----
    if buf_metas:
        chunk_id = f"chunk_{chunk_id_counter:05d}"
        _flush_chunk_to_disk(cache_dir, chunk_id, scopes,
                             buf_orig, buf_cf, buf_valid)
        for meta in buf_metas:
            meta["chunk_id"] = chunk_id
            new_pair_entries.append(meta)
        new_chunks.append({
            "chunk_id": chunk_id,
            "file": f"chunks/{chunk_id}.npz",
            "n_pairs": len(buf_metas),
        })
        print(f"    Flushed final {chunk_id} ({len(buf_metas)} pairs)")

    del buf_metas, buf_orig, buf_cf, buf_valid
    gc.collect()

    # ---- Save updated cache index ----
    all_pair_entries = existing_pair_entries + new_pair_entries
    all_chunks = existing_chunks + new_chunks

    _save_pair_index(cache_dir, all_pair_entries)

    n_extracted = sum(1 for e in all_pair_entries if e.get("status") == "extracted")
    n_total_failed = sum(1 for e in all_pair_entries if e.get("status") == "failed")

    manifest_data = {
        "scopes": list(scopes),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "chunk_size": chunk_size,
        "truncation_max_length": TRUNCATION_MAX_LENGTH,
        "total_pairs_extracted": n_extracted,
        "total_pairs_failed": n_total_failed,
        "chunks": all_chunks,
        "resumed_from_cache": n_cached > 0,
    }
    save_json(manifest_data, cache_dir / "cache_manifest.json")

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete: {n_extracted} extracted, "
          f"{n_total_failed} failed, {elapsed:.1f}s total")


# ---------------------------------------------------------------------------
# Deterministic qid grouping
# ---------------------------------------------------------------------------

def _qid_group(qid: str) -> int:
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % (2 ** 31)


# ---------------------------------------------------------------------------
# Probe dataset builders (cache-backed, per-layer arrays)
# ---------------------------------------------------------------------------

def _build_orig_vs_cf(
    metas: list[dict],
    orig_l: np.ndarray,   # [n_pairs, hidden_size]
    cf_l:   np.ndarray,
    valid_l: np.ndarray,  # [n_pairs] bool
    family: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Task A — binary: orig (0) vs CF (1) for one family."""
    X, y, groups = [], [], []
    for i, m in enumerate(metas):
        if m["family"] != family:
            continue
        if not valid_l[i]:
            continue
        grp = _qid_group(m["qid"])
        X.append(orig_l[i])
        X.append(cf_l[i])
        y.extend([0, 1])
        groups.extend([grp, grp])
    if not X:
        return None, None, None
    return np.stack(X), np.array(y), np.array(groups)


def _build_family_identity(
    metas: list[dict],
    cf_l:    np.ndarray,
    valid_l: np.ndarray,
    families: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Task B — multiclass: CF only, label by family."""
    label_map = {f: i for i, f in enumerate(families)}
    X, y, groups = [], [], []
    for i, m in enumerate(metas):
        if m["family"] not in label_map:
            continue
        if not valid_l[i]:
            continue
        X.append(cf_l[i])
        y.append(label_map[m["family"]])
        groups.append(_qid_group(m["qid"]))
    if not X:
        return None, None, None, families
    return np.stack(X), np.array(y), np.array(groups), families


def _build_within_family_attribute(
    metas:     list[dict],
    cf_l:      np.ndarray,
    valid_l:   np.ndarray,
    family:    str,
    min_examples: int,
    exclude_labels: list[str] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str], list[str]]:
    """Task C — multiclass: CF only, label by attr_val within one family."""
    excl = set(exclude_labels or [])
    fam_indices = [(i, m) for i, m in enumerate(metas)
                   if m["family"] == family and m["attr_val"] not in excl]
    counts = Counter(m["attr_val"] for _, m in fam_indices)
    all_classes = sorted(counts.items())
    retained = [cls for cls, cnt in all_classes if cnt >= min_examples]
    dropped = [cls for cls, cnt in all_classes if cnt < min_examples]
    if len(retained) < 2:
        return None, None, None, retained, dropped

    label_map = {cls: i for i, cls in enumerate(retained)}
    X, y, groups = [], [], []
    for i, m in fam_indices:
        if m["attr_val"] not in label_map:
            continue
        if not valid_l[i]:
            continue
        X.append(cf_l[i])
        y.append(label_map[m["attr_val"]])
        groups.append(_qid_group(m["qid"]))
    if not X:
        return None, None, None, retained, dropped
    return np.stack(X), np.array(y), np.array(groups), retained, dropped


def _build_context_split(
    metas: list[dict],
    cf_l:    np.ndarray,
    valid_l: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Task D — binary: partner (0) vs explicit (1) for sexual_orientation."""
    so_indices = [(i, m) for i, m in enumerate(metas)
                  if m["family"] == "sexual_orientation"]
    n_other = sum(1 for _, m in so_indices if m.get("split_label") == "other")
    label_map = {"partner": 0, "explicit": 1}
    X, y, groups = [], [], []
    for i, m in so_indices:
        lbl = m.get("split_label", "")
        if lbl not in label_map:
            continue
        if not valid_l[i]:
            continue
        X.append(cf_l[i])
        y.append(label_map[lbl])
        groups.append(_qid_group(m["qid"]))
    if not X:
        return None, None, None, n_other
    return np.stack(X), np.array(y), np.array(groups), n_other


def _build_focal_vs_control_cf(
    metas: list[dict],
    cf_l:    np.ndarray,
    valid_l: np.ndarray,
    focal_family: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Task E — binary: focal CF (1) vs control CF (0)."""
    X, y, groups = [], [], []
    for i, m in enumerate(metas):
        if m["family"] == focal_family:
            label = 1
        elif m["family"] == "control":
            label = 0
        else:
            continue
        if not valid_l[i]:
            continue
        X.append(cf_l[i])
        y.append(label)
        groups.append(_qid_group(m["qid"]))
    if not X or len(set(y)) < 2:
        return None, None, None
    return np.stack(X), np.array(y), np.array(groups)


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
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return train_idx, test_idx, "group_shuffle"
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
    task_type: str,
    label_names: list[str] | None = None,
    max_iter: int = 1000,
    balance_classes: bool = False,
    probe_C: float = 1.0,
    probe_class_weight=None,
) -> dict:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"error": "single_class", "n_examples": int(len(y))}

    train_idx, test_idx, splitter = grouped_split(X, y, groups, test_size, seed)

    y_train_pre = y[train_idx]
    y_test_pre  = y[test_idx]
    err = validate_split(y_train_pre, y_test_pre)
    if err is not None:
        return {"error": err, "n_examples": int(len(y))}

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_train_pre, y_test_pre

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

    all_labels = list(range(len(label_names))) if label_names else list(range(len(unique_classes)))
    metrics["_confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=all_labels).tolist()
    metrics["_label_names"]      = label_names

    return metrics


# ---------------------------------------------------------------------------
# Per-layer probe sweep (cache-backed)
# ---------------------------------------------------------------------------

def run_layerwise_probe_cached(
    pair_metas:      list[dict],
    orig_scope:      np.ndarray,
    cf_scope:        np.ndarray,
    valid_scope:     np.ndarray,
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
    """Run the probe at every layer using cached arrays."""
    rows = []
    for layer in range(n_hidden_states):
        orig_l = orig_scope[:, layer, :]
        cf_l = cf_scope[:, layer, :]
        valid_l = valid_scope[:, layer]

        base = {
            "layer":                     layer,
            "layer_index":               layer,
            "layer_type":                "embedding" if layer == 0 else "transformer",
            "transformer_layer_number":  None if layer == 0 else layer,
            "feature_scope":             scope,
            "task":                      task,
            "family":                    family or "global",
        }

        if task == "orig_vs_cf":
            X, y, groups = _build_orig_vs_cf(
                pair_metas, orig_l, cf_l, valid_l, family)
            task_type = "binary"
            label_names = ["orig", "cf"]
            extra = {}

        elif task == "family_identity":
            X, y, groups, label_names = _build_family_identity(
                pair_metas, cf_l, valid_l, families_global)
            task_type = "multiclass"
            extra = {}

        elif task == "within_family_attribute":
            X, y, groups, retained, dropped = _build_within_family_attribute(
                pair_metas, cf_l, valid_l, family, min_examples_per_class,
                exclude_labels=within_family_exclude_labels)
            task_type = "multiclass"
            label_names = retained
            extra = {"retained_classes": retained, "dropped_classes": dropped}
            if X is None:
                rows.append({**base, **extra, "error": "insufficient_classes"})
                continue

        elif task == "context_split":
            X, y, groups, n_other = _build_context_split(
                pair_metas, cf_l, valid_l)
            task_type = "binary"
            label_names = ["partner", "explicit"]
            extra = {"n_other_dropped": n_other}

        elif task == "focal_vs_control_cf":
            X, y, groups = _build_focal_vs_control_cf(
                pair_metas, cf_l, valid_l, family)
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

        row = {**base, **extra,
               **{k: v for k, v in metrics.items() if not k.startswith("_")}}
        row["_confusion_matrix"] = metrics.get("_confusion_matrix")
        row["_label_names"]      = metrics.get("_label_names")
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
    exp_dir.mkdir(parents=True, exist_ok=True)

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
    ax.axvline(x=0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.6, label="embed|L1")


def _ax_layer_ticks(ax, n_layers: int):
    step = max(1, n_layers // 10)
    ax.set_xticks(range(0, n_layers, step))
    ax.set_xlabel("Layer  (0 = embedding, 1..N = transformer)")
    ax.tick_params(axis="x", labelsize=8)


def plot_orig_vs_cf_curves(
    results_by_family: dict,
    scope:             str,
    output_path:       Path,
    metric:            str = "balanced_accuracy",
):
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
    p.add_argument("--model_path",  default=None,
                   help="Path to HF model. Not required with --probe_only.")
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
                   help="Feature pooling scope(s).")
    p.add_argument("--window_radius", type=int, default=2)
    p.add_argument("--probe_tasks",   nargs="+",
                   default=["orig_vs_cf", "family_identity",
                             "within_family_attribute", "context_split"],
                   choices=["orig_vs_cf", "family_identity",
                             "within_family_attribute", "context_split",
                             "focal_vs_control_cf"])
    p.add_argument("--min_examples_per_class", type=int, default=20)
    p.add_argument("--test_size",   type=float, default=0.25)
    p.add_argument("--n_bootstrap", type=int,   default=0)
    p.add_argument("--balance_classes", action="store_true")
    p.add_argument("--include_controls", action="store_true")
    p.add_argument("--probe_C", type=float, default=1.0)
    p.add_argument("--probe_max_iter", type=int, default=1000)
    p.add_argument("--probe_class_weight", choices=["none", "balanced"], default="none")
    p.add_argument("--min_prefix_tokens", type=int, default=2)
    p.add_argument("--min_suffix_tokens", type=int, default=2)
    p.add_argument("--within_family_exclude_labels", nargs="*", default=[])
    p.add_argument("--so_within_family", action="store_true",
                   help="Enable within_family_attribute for sexual_orientation.")
    p.add_argument("--save_pair_level_metadata", action="store_true",
                   dest="save_pair_level_metadata")
    p.add_argument("--save_pair_level_features", action="store_true",
                   dest="save_pair_level_metadata",
                   help=argparse.SUPPRESS)

    # ---- New flags for phased execution ----
    p.add_argument("--extract_only", action="store_true",
                   help="Stop after extraction cache is built. Do not run probes.")
    p.add_argument("--probe_only", action="store_true",
                   help="Skip extraction; load from existing cache. No model loading.")
    p.add_argument("--force_reextract", action="store_true",
                   help="Ignore prior cache and rebuild extraction from scratch.")
    p.add_argument("--extraction_chunk_size", type=int, default=25,
                   help="Number of pairs per extraction chunk (default 25).")
    p.add_argument("--cache_dir", default=None,
                   help="Extraction cache directory. Default: <output_dir>/extraction_cache.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    if args.n_bootstrap > 0:
        raise NotImplementedError(
            f"--n_bootstrap {args.n_bootstrap} requested, but bootstrap CI not yet implemented.")

    if args.extract_only and args.probe_only:
        print("ERROR: --extract_only and --probe_only are mutually exclusive.")
        raise SystemExit(1)

    if not args.probe_only and not args.model_path:
        print("ERROR: --model_path is required unless --probe_only is set.")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "extraction_cache"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_warnings: list[str] = []
    probe_class_weight = None if args.probe_class_weight == "none" else args.probe_class_weight

    # -------------------------------------------------------------------
    # 1. Load and sample data
    # -------------------------------------------------------------------
    print("Loading pairs...")
    all_pairs_focal, assignment_log_focal = load_pairs(args.data_path, include_controls=False)
    print(f"  Total focal pairs loaded: {len(all_pairs_focal)}")

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

    all_extraction_pairs = sampled_pairs + sampled_control_pairs

    # -------------------------------------------------------------------
    # 2. Phase A: Extraction
    # -------------------------------------------------------------------
    if not args.probe_only:
        print(f"\n{'='*60}")
        print(f"PHASE A: EXTRACTION")
        print(f"  scopes={args.feature_scope}, chunk_size={args.extraction_chunk_size}")
        print("=" * 60)

        model, tok, device = load_model_and_tok(args.model_path, args.device, args.dtype)
        n_hidden_states      = model.config.num_hidden_layers + 1
        n_transformer_layers = model.config.num_hidden_layers
        hidden_size          = model.config.hidden_size
        print(f"  n_hidden_states (incl. embedding): {n_hidden_states}")

        run_extraction_phase(
            model, tok, all_extraction_pairs, device,
            cache_dir=cache_dir, scopes=args.feature_scope,
            chunk_size=args.extraction_chunk_size,
            window_radius=args.window_radius,
            min_prefix_tokens=args.min_prefix_tokens,
            min_suffix_tokens=args.min_suffix_tokens,
            force_reextract=args.force_reextract,
        )

        # Free model memory
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  Model unloaded, GPU memory freed.")

    if args.extract_only:
        print(f"\nExtraction complete. Cache directory: {cache_dir}")
        print("Run with --probe_only to fit probes without reloading the model.")
        return

    # -------------------------------------------------------------------
    # 3. Phase B: Probing
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE B: PROBING FROM CACHE")
    print(f"  cache_dir={cache_dir}")
    print("=" * 60)

    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no cache at {manifest_path}. Run extraction first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    pair_metas_all = _load_pair_index(cache_dir)
    pair_metas = [m for m in pair_metas_all if m.get("status") == "extracted"]
    print(f"  Loaded {len(pair_metas)} extracted pairs from cache")

    n_hidden_states      = manifest["n_layers"]
    n_transformer_layers = n_hidden_states - 1
    hidden_size          = manifest["hidden_size"]

    n_fallback = sum(1 for m in pair_metas if m.get("edit_fallback") in (1, "1", True))
    if n_fallback > 0:
        w = (f"{n_fallback}/{len(pair_metas)} pairs had no detectable token-level edit span; "
             f"span-derived scopes are unavailable for those pairs.")
        print(f"  Warning: {w}")
        global_warnings.append(w)

    n_pairs_with_omitted_features_by_scope: dict[str, int] = {}
    for sc in args.feature_scope:
        if sc in _SPAN_SCOPES:
            n_pairs_with_omitted_features_by_scope[sc] = n_fallback

    if args.save_pair_level_metadata:
        save_csv(pair_metas, output_dir / "pair_level_metadata.csv")
        print("  Saved pair_level_metadata.csv")

    # -------------------------------------------------------------------
    # 4. Run probes per scope
    # -------------------------------------------------------------------
    all_results_by_scope: dict[str, dict] = {}
    task_sample_sizes: dict[str, dict] = {}
    invalid_layer_counts: dict[str, dict] = {}

    for scope in args.feature_scope:
        if scope not in manifest["scopes"]:
            w = f"Scope '{scope}' not in cache (cached: {manifest['scopes']}). Skipping."
            print(f"  Warning: {w}")
            global_warnings.append(w)
            continue

        print(f"\n{'='*60}")
        print(f"Loading cached features — scope: {scope}")
        print("=" * 60)

        orig_scope, cf_scope, valid_scope = load_scope_from_cache(
            cache_dir, scope, manifest)

        if orig_scope.shape[0] != len(pair_metas):
            print(f"  ERROR: shape mismatch: {orig_scope.shape[0]} cached rows vs "
                  f"{len(pair_metas)} pair_index entries. Cache may be corrupt.")
            continue

        print(f"  Loaded: {orig_scope.shape} per side, "
              f"~{(orig_scope.nbytes + cf_scope.nbytes) / 1e9:.2f} GB")

        scope_results: dict = {}
        task_sample_sizes[scope] = {}
        invalid_layer_counts[scope] = {}

        # ---- Task A: orig_vs_cf per family ----
        if "orig_vs_cf" in args.probe_tasks:
            print("\n  [A] orig_vs_cf  (Manipulation Detectability)")
            orig_vs_cf_by_family = {}
            for fam in requested_families:
                n_fam = sum(1 for m in pair_metas if m["family"] == fam)
                if n_fam == 0:
                    continue
                print(f"    {fam}  n={n_fam}...", end=" ", flush=True)
                rows = run_layerwise_probe_cached(
                    pair_metas=pair_metas,
                    orig_scope=orig_scope, cf_scope=cf_scope, valid_scope=valid_scope,
                    task="orig_vs_cf", scope=scope,
                    family=fam, families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C, probe_max_iter=args.probe_max_iter,
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
                    "n_pairs":            n_fam,
                    "n_hidden_states":    n_hidden_states,
                    "hidden_size":        hidden_size,
                    "class_labels":       ["orig", "cf"],
                    "grouped_split":      True,
                    "test_size":          args.test_size,
                    "seed":               args.seed,
                    "truncation_max_length": TRUNCATION_MAX_LENGTH,
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

            if orig_vs_cf_by_family:
                plot_orig_vs_cf_curves(
                    orig_vs_cf_by_family, scope=scope,
                    output_path=plots_dir / f"01_orig_vs_cf_{scope}.png",
                )

        # ---- Task B: family identity ----
        if "family_identity" in args.probe_tasks:
            print("\n  [B] family_identity")
            rows = run_layerwise_probe_cached(
                pair_metas=pair_metas,
                orig_scope=orig_scope, cf_scope=cf_scope, valid_scope=valid_scope,
                task="family_identity", scope=scope,
                family=None, families_global=requested_families,
                n_hidden_states=n_hidden_states,
                test_size=args.test_size, seed=args.seed,
                min_examples_per_class=args.min_examples_per_class,
                balance_classes=args.balance_classes,
                probe_C=args.probe_C, probe_max_iter=args.probe_max_iter,
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
                "grouped_split":    True,
                "truncation_max_length": TRUNCATION_MAX_LENGTH,
                "controls_excluded": True,
                "controls_excluded_reason": (
                    "family_identity uses only the focal families in --families; "
                    "control pairs are not a valid demographic family class."
                ),
                "contrast_with_orig_vs_cf": (
                    "Unlike orig_vs_cf (manipulation detectability), family_identity asks "
                    "whether different *types* of demographic edits are separable in the "
                    "model's CF representation space."
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
                n_fam = sum(1 for m in pair_metas if m["family"] == fam)
                if n_fam == 0:
                    continue
                excl = set(args.within_family_exclude_labels or [])
                fam_metas = [m for m in pair_metas
                             if m["family"] == fam and m["attr_val"] not in excl]
                counts = Counter(m["attr_val"] for m in fam_metas)
                valid_cls = [cls for cls, cnt in counts.items()
                             if cnt >= args.min_examples_per_class]
                if len(valid_cls) < 2:
                    print(f"    {fam}: skip  ({len(valid_cls)} valid class(es) with "
                          f">={args.min_examples_per_class} examples)")
                    continue
                if fam == "sexual_orientation" and not args.so_within_family:
                    w = (
                        "within_family_attribute for sexual_orientation skipped by default. "
                        "Use --so_within_family to enable."
                    )
                    print(f"    {fam}: skipped  (use --so_within_family to enable)")
                    global_warnings.append(w)
                    continue
                print(f"    {fam}  ({len(valid_cls)} classes)...", end=" ", flush=True)
                rows = run_layerwise_probe_cached(
                    pair_metas=pair_metas,
                    orig_scope=orig_scope, cf_scope=cf_scope, valid_scope=valid_scope,
                    task="within_family_attribute", scope=scope,
                    family=fam, families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C, probe_max_iter=args.probe_max_iter,
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

                exp_meta = {
                    "task":                   "within_family_attribute",
                    "family":                 fam,
                    "feature_scope":          scope,
                    "all_class_counts":       dict(counts),
                    "valid_classes":          valid_cls,
                    "dropped_classes":        [cls for cls in counts if cls not in valid_cls],
                    "excluded_labels":        list(excl),
                    "min_examples_per_class": args.min_examples_per_class,
                    "truncation_max_length":  TRUNCATION_MAX_LENGTH,
                    "controls_excluded":      True,
                    "controls_excluded_reason": (
                        "within_family_attribute is a per-focal-family task."
                    ),
                    "caution_attribute_encoding": (
                        "Performance may partly reflect lexical regularities of the edit "
                        "templates rather than abstract demographic encoding."
                    ),
                }
                if fam == "sexual_orientation":
                    exp_meta["caution_label_heterogeneity"] = (
                        "Labels for sexual_orientation may mix orientation-identity labels "
                        "(gay, straight) with relationship-context labels (partner)."
                    )
                save_experiment(
                    exp_dir=output_dir / fam / f"within_family_attribute_{scope}",
                    metadata=exp_meta, per_layer_rows=rows)

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
            so_metas = [m for m in pair_metas if m["family"] == "sexual_orientation"]
            n_partner  = sum(1 for m in so_metas if m.get("split_label") == "partner")
            n_explicit = sum(1 for m in so_metas if m.get("split_label") == "explicit")
            n_other    = sum(1 for m in so_metas if m.get("split_label") == "other")
            print(f"\n  [D] context_split  "
                  f"partner={n_partner}  explicit={n_explicit}  other_dropped={n_other}")

            if (n_partner >= args.min_examples_per_class and
                    n_explicit >= args.min_examples_per_class):
                rows = run_layerwise_probe_cached(
                    pair_metas=pair_metas,
                    orig_scope=orig_scope, cf_scope=cf_scope, valid_scope=valid_scope,
                    task="context_split", scope=scope,
                    family="sexual_orientation", families_global=requested_families,
                    n_hidden_states=n_hidden_states,
                    test_size=args.test_size, seed=args.seed,
                    min_examples_per_class=args.min_examples_per_class,
                    balance_classes=args.balance_classes,
                    probe_C=args.probe_C, probe_max_iter=args.probe_max_iter,
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
                    "n_partner":        n_partner,
                    "n_explicit":       n_explicit,
                    "n_other_dropped":  n_other,
                    "truncation_max_length": TRUNCATION_MAX_LENGTH,
                    "task_description": (
                        "Classifies partner-neutralized SO edits vs explicit-orientation edits."
                    ),
                    "note_stage4_b3": (
                        "Stage 4 B3 showed strong behavioral differences between "
                        "partner-framed and explicit-identity SO pairs."
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
            ctrl_count = sum(1 for m in pair_metas if m["family"] == "control")
            if ctrl_count == 0:
                w = ("focal_vs_control_cf skipped: no control pairs in cache. "
                     "Use --include_controls.")
                print(f"\n  Warning: {w}")
                global_warnings.append(w)
            else:
                print(f"\n  [E] focal_vs_control_cf  n_control={ctrl_count}")
                for fam in requested_families:
                    n_fam = sum(1 for m in pair_metas if m["family"] == fam)
                    if n_fam == 0:
                        continue
                    print(f"    {fam}  n_focal={n_fam}...", end=" ", flush=True)
                    rows = run_layerwise_probe_cached(
                        pair_metas=pair_metas,
                        orig_scope=orig_scope, cf_scope=cf_scope, valid_scope=valid_scope,
                        task="focal_vs_control_cf", scope=scope,
                        family=fam, families_global=requested_families,
                        n_hidden_states=n_hidden_states,
                        test_size=args.test_size, seed=args.seed,
                        min_examples_per_class=args.min_examples_per_class,
                        balance_classes=args.balance_classes,
                        probe_C=args.probe_C, probe_max_iter=args.probe_max_iter,
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
                        "n_focal":       n_fam,
                        "n_control":     ctrl_count,
                        "truncation_max_length": TRUNCATION_MAX_LENGTH,
                        "controls_usage": (
                            "control pairs serve as label=0 (negative class); "
                            "focal CF pairs serve as label=1 (positive class)."
                        ),
                        "caution_confound": (
                            "High accuracy may reflect demographic-specific representation "
                            "AND/OR structural differences between focal and control CFs."
                        ),
                    }, per_layer_rows=rows)

                    s = summarize_layers(rows)
                    if s:
                        print(f"peak_layer={s['peak_layer']}  bal_acc={s['peak_value']:.3f}")
                    else:
                        print("no valid layers")

        # Heatmaps per scope
        if scope_results:
            plot_summary_heatmap(
                scope_results, scope=scope,
                output_path=plots_dir / f"05_summary_heatmap_normalized_{scope}.png",
                metric="normalized_decoding_gain",
                cmap="RdBu", vcenter=0.0, vmin=-1.0, vmax=1.0,
            )
            plot_summary_heatmap(
                scope_results, scope=scope,
                output_path=plots_dir / f"05b_summary_heatmap_raw_{scope}.png",
                metric="balanced_accuracy",
                cmap="RdYlGn", vcenter=None, vmin=0.0, vmax=1.0,
            )

        all_results_by_scope[scope] = scope_results

        # Free scope arrays before loading next scope
        del orig_scope, cf_scope, valid_scope
        gc.collect()
        print(f"\n  Freed cached arrays for scope: {scope}")

    # -------------------------------------------------------------------
    # 5. Summary JSON
    # -------------------------------------------------------------------
    print("\nBuilding summary.json...")

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

    n_extracted_focal   = sum(1 for m in pair_metas if m["family"] != "control")
    n_extracted_control = sum(1 for m in pair_metas if m["family"] == "control")
    n_failed_total = sum(1 for m in pair_metas_all if m.get("status") == "failed")

    summary = {
        "script":  SCRIPT_NAME,
        "version": VERSION,
        "args":    vars(args),
        "data": {
            "n_loaded_focal_pairs":                   len(all_pairs_focal),
            "n_sampled_focal_pairs":                  len(sampled_pairs),
            "n_sampled_control_pairs":                len(sampled_control_pairs),
            "n_extracted_total_pairs":                len(pair_metas),
            "n_extracted_focal_pairs":                n_extracted_focal,
            "n_extracted_control_pairs":              n_extracted_control,
            "n_failed_pairs":                         n_failed_total,
            "controls_loaded":                        bool(args.include_controls),
            "family_sample_counts":                   family_sample_counts,
            "n_edit_fallbacks":                       n_fallback,
            "n_pairs_with_omitted_features_by_scope": n_pairs_with_omitted_features_by_scope,
        },
        "extraction_cache": {
            "cache_dir":              str(cache_dir),
            "chunk_size":             args.extraction_chunk_size,
            "truncation_max_length":  TRUNCATION_MAX_LENGTH,
            "resumed_from_cache":     manifest.get("resumed_from_cache", False),
            "n_chunks":               len(manifest.get("chunks", [])),
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
                "intervening on that layer would change model behavior."
            ),
            "high_decode_weak_causal_means": (
                "If Stage 4 ablation recovery was weak but Stage 4.5 decodability is high, "
                "that suggests distributed or redundant representation."
            ),
            "convergent_localization": (
                "If Stage 3 edit-mass peaks, Stage 4 residual patching recovery peaks, "
                "and Stage 4.5 decoding peaks all align, that provides convergent evidence "
                "for localization of the demographic signal."
            ),
            "so_hypothesis": (
                "Sexual orientation may show weaker early decodability and stronger "
                "mid/late-layer decodability, consistent with Stage 4 findings."
            ),
            "stage3_reference": (
                "Stage 3 edit-mass heatmaps showed early-layer peaks for gender_identity "
                "and race."
            ),
            "orig_vs_cf_framing": (
                "The orig_vs_cf task classifies original vs counterfactually-edited hidden "
                "states. High accuracy means the model encodes the edit, not necessarily "
                "the demographic attribute itself."
            ),
            "context_split_framing": (
                "The context_split task classifies partner-neutralized SO edits vs "
                "explicit-orientation edits."
            ),
            "within_family_attribute_caveat": (
                "within_family_attribute accuracy may partly reflect lexical regularities "
                "of edit templates rather than abstract demographic encoding."
            ),
            "focal_vs_control_cf_caveat": (
                "focal_vs_control_cf accuracy may reflect demographic-specific representation "
                "AND/OR structural differences in how focal vs control CFs were constructed."
            ),
        },
        "stage3_expected_peaks":  STAGE3_EDIT_MASS_PEAK,
        "stage3_expected_regions": STAGE3_EXPECTED_REGION,
        "stage3_expected_regions_note": (
            "Values may be a single string or a list of acceptable region strings."
        ),
    }
    save_json(summary, output_dir / "summary.json")
    print(f"Done.  Results -> {output_dir}")


if __name__ == "__main__":
    main()
