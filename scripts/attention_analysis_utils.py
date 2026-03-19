"""
Shared utilities for analyzing compact attention summaries produced by
`extract_representations.py`.

Goal: keep Stage 1/2 analysis scripts consistent and avoid drift.
"""

from __future__ import annotations

import numpy as np


def _mean_over_heads_and_layers(x):
    arr = np.asarray(x)
    return float(np.nanmean(arr))


def _mean_over_late_layers(x, start_layer: int):
    arr = np.asarray(x)
    return float(np.nanmean(arr[start_layer:]))


def normalize_attention_summary(attn: dict | None) -> dict | None:
    """
    Normalize attention-summary dict keys to the schema expected by
    `attention_shift_metrics` / `headwise_attention_table`.

    Supports both:
      - already-normalized keys (mass_to_*, topk_source_positions, ...)
      - extract_representations.py keys (edit_mass, stem_mass, topk_positions, ...)
    """
    if attn is None or not isinstance(attn, dict):
        return None

    # Already in the expected schema
    if any(k in attn for k in ("mass_to_edit_region", "topk_source_positions")):
        return attn

    out: dict = {}
    if "edit_mass" in attn:
        out["mass_to_edit_region"] = attn["edit_mass"]
    if "largest_edit_mass" in attn:
        out["mass_to_largest_region"] = attn["largest_edit_mass"]
    if "stem_mass" in attn:
        out["mass_to_question_span"] = attn["stem_mass"]
    if "entropy" in attn:
        out["entropy"] = attn["entropy"]
    if "topk_positions" in attn:
        out["topk_source_positions"] = attn["topk_positions"]
    if "topk_values" in attn:
        out["topk_weights"] = attn["topk_values"]
    return out


def attention_shift_metrics(attn_orig: dict, attn_cf: dict, late_start: int) -> dict:
    """
    Compare two attention summary dicts, each containing arrays like:
      mass_to_edit_region: (layers, heads)
      mass_to_largest_region: (layers, heads)
      mass_to_question_span: (layers, heads)
      entropy: (layers, heads)
      topk_source_positions: (layers, heads, k)
      topk_weights: (layers, heads, k)
    """
    out: dict[str, float | int] = {}

    for key in [
        "mass_to_edit_region",
        "mass_to_largest_region",
        "mass_to_question_span",
        "entropy",
    ]:
        if key in attn_orig and key in attn_cf:
            o = np.asarray(attn_orig[key], dtype=np.float32)
            c = np.asarray(attn_cf[key], dtype=np.float32)
            d = c - o

            out[f"{key}_mean_orig"] = float(np.nanmean(o))
            out[f"{key}_mean_cf"] = float(np.nanmean(c))
            out[f"{key}_mean_delta"] = float(np.nanmean(d))
            out[f"{key}_abs_mean_delta"] = float(np.nanmean(np.abs(d)))

            out[f"{key}_late_orig"] = float(np.nanmean(o[late_start:]))
            out[f"{key}_late_cf"] = float(np.nanmean(c[late_start:]))
            out[f"{key}_late_delta"] = float(np.nanmean(d[late_start:]))
            out[f"{key}_late_abs_delta"] = float(np.nanmean(np.abs(d[late_start:])))

            # Per-head late-layer average
            late_o = np.nanmean(o[late_start:], axis=0)  # (heads,)
            late_c = np.nanmean(c[late_start:], axis=0)
            late_d = late_c - late_o

            out[f"{key}_max_head_abs_delta"] = float(np.nanmax(np.abs(late_d)))
            out[f"{key}_argmax_head_abs_delta"] = int(np.nanargmax(np.abs(late_d)))

    # Top-k overlap (explicit late computation; less brittle)
    if "topk_source_positions" in attn_orig and "topk_source_positions" in attn_cf:
        o = np.asarray(attn_orig["topk_source_positions"])
        c = np.asarray(attn_cf["topk_source_positions"])

        overlaps = []
        late_overlaps = []
        n_layers = o.shape[0]
        n_heads = o.shape[1]

        for l in range(n_layers):
            for h in range(n_heads):
                so = set(o[l, h].tolist())
                sc = set(c[l, h].tolist())
                denom = max(len(so | sc), 1)
                j = len(so & sc) / denom
                overlaps.append(j)
                if l >= late_start:
                    late_overlaps.append(j)

        out["topk_jaccard_mean"] = float(np.nanmean(overlaps)) if overlaps else float("nan")
        out["topk_jaccard_late"] = float(np.nanmean(late_overlaps)) if late_overlaps else float("nan")

    return out


def headwise_attention_table(attn_orig: dict, attn_cf: dict, late_start: int) -> list[dict]:
    """
    Return one row per head with late-layer averages.
    """
    rows: list[dict] = []
    if attn_orig is None or attn_cf is None:
        return rows

    keys = [
        "mass_to_edit_region",
        "mass_to_largest_region",
        "mass_to_question_span",
        "entropy",
    ]
    if not all(k in attn_orig and k in attn_cf for k in keys):
        return rows

    n_layers, n_heads = np.asarray(attn_orig["entropy"]).shape

    for h in range(n_heads):
        row: dict[str, float | int] = {"head": h}
        for key in keys:
            o = np.asarray(attn_orig[key], dtype=np.float32)
            c = np.asarray(attn_cf[key], dtype=np.float32)

            o_h = float(np.nanmean(o[late_start:, h]))
            c_h = float(np.nanmean(c[late_start:, h]))
            d_h = c_h - o_h

            row[f"{key}_orig"] = o_h
            row[f"{key}_cf"] = c_h
            row[f"{key}_delta"] = float(d_h)
            row[f"{key}_abs_delta"] = float(abs(d_h))

        # Optional per-head top-k overlap (late layers only)
        if "topk_source_positions" in attn_orig and "topk_source_positions" in attn_cf:
            o = np.asarray(attn_orig["topk_source_positions"])
            c = np.asarray(attn_cf["topk_source_positions"])
            late_js = []
            for l in range(late_start, o.shape[0]):
                so = set(o[l, h].tolist())
                sc = set(c[l, h].tolist())
                denom = max(len(so | sc), 1)
                late_js.append(len(so & sc) / denom)
            row["topk_jaccard_late"] = float(np.nanmean(late_js)) if late_js else float("nan")

        rows.append(row)

    return rows

