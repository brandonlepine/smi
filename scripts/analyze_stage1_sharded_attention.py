#!/usr/bin/env python3
"""
Stage 1 analysis for sharded extractions with optional attention summaries.

Expanded with:
  - N-adjusted comparisons (subsampled bootstrap, Cohen's d, permutation tests)
  - Layer×head attention-shift heatmaps with marginal layer/head effect bars
  - Within-attribute label heatmaps (e.g., medicaid vs private vs uninsured)
  - Vig-style per-position attention bar charts for illustrative examples
  - Correctness-aware flip statistics (correct→wrong, wrong→correct)
  - Token-count adjusted metrics (partial correlations)
  - Within-group counterfactual direct comparisons
  - Head clustering, co-activation, top-k overlap
  - Bootstrap CI effect-size comparison panels

Usage:
  python analyze_stage1_sharded_attention.py \
    --extraction_dir ./extractions_memsafe \
    --output_dir ./stage1_results_sharded \
    --margin_threshold 0.5
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch

from load_sharded_extractions import ShardedExtractionStore

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy import stats as sp_stats
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Behavioral metrics
# ---------------------------------------------------------------------------

AIDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDXA = {0: "A", 1: "B", 2: "C", 3: "D"}


def behavioral_metrics(logits: np.ndarray, gold: str) -> dict:
    gi = AIDX[gold]
    pred_i = int(np.argmax(logits))
    gl = float(logits[gi])

    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    gp = float(probs[gi])

    comp = np.concatenate([logits[:gi], logits[gi + 1:]])
    margin = float(gl - np.max(comp))

    return {
        "predicted": IDXA[pred_i],
        "correct": IDXA[pred_i] == gold,
        "gold_logit": gl,
        "gold_prob": gp,
        "margin": margin,
    }


def pairwise_behavioral(orig: dict, cf: dict) -> dict:
    return {
        "orig_predicted": orig["predicted"],
        "cf_predicted": cf["predicted"],
        "answer_flip": orig["predicted"] != cf["predicted"],
        "correctness_flip": orig["correct"] != cf["correct"],
        "orig_correct": orig["correct"],
        "cf_correct": cf["correct"],
        "orig_margin": orig["margin"],
        "delta_gold_logit": cf["gold_logit"] - orig["gold_logit"],
        "abs_delta_gold_logit": abs(cf["gold_logit"] - orig["gold_logit"]),
        "delta_margin": cf["margin"] - orig["margin"],
        "abs_delta_margin": abs(cf["margin"] - orig["margin"]),
        "delta_gold_prob": cf["gold_prob"] - orig["gold_prob"],
        "abs_delta_gold_prob": abs(cf["gold_prob"] - orig["gold_prob"]),
        "orig_gold_prob": orig["gold_prob"],
        "cf_gold_prob": cf["gold_prob"],
    }


# ---------------------------------------------------------------------------
# Repr metrics
# ---------------------------------------------------------------------------

def repr_metrics_per_layer(h_orig: np.ndarray, h_cf: np.ndarray) -> dict:
    delta = h_cf - h_orig
    euclid = np.linalg.norm(delta, axis=1)

    norm_orig = np.linalg.norm(h_orig, axis=1)
    norm_cf = np.linalg.norm(h_cf, axis=1)
    mean_norm = (norm_orig + norm_cf) / 2
    mean_norm = np.maximum(mean_norm, 1e-8)
    norm_euclid = euclid / mean_norm

    cos_sim = np.sum(h_orig * h_cf, axis=1) / (
        np.maximum(norm_orig, 1e-8) * np.maximum(norm_cf, 1e-8)
    )
    cos_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)

    return {
        "euclidean": euclid.tolist(),
        "norm_euclidean": norm_euclid.tolist(),
        "cosine_dist": cos_dist.tolist(),
        "orig_norm": norm_orig.tolist(),
    }


# ---------------------------------------------------------------------------
# Attention summary helpers
# ---------------------------------------------------------------------------

def _normalize_attention_summary(attn):
    if attn is None or not isinstance(attn, dict):
        return None
    if any(k in attn for k in ("mass_to_edit_region", "topk_source_positions")):
        return attn
    out = {}
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
    out = {}
    for key in ["mass_to_edit_region", "mass_to_largest_region", "mass_to_question_span", "entropy"]:
        if key in attn_orig and key in attn_cf:
            o = np.asarray(attn_orig[key], dtype=np.float32)
            c = np.asarray(attn_cf[key], dtype=np.float32)
            d = c - o
            out[f"{key}_mean_orig"] = float(np.mean(o))
            out[f"{key}_mean_cf"] = float(np.mean(c))
            out[f"{key}_mean_delta"] = float(np.mean(d))
            out[f"{key}_abs_mean_delta"] = float(np.mean(np.abs(d)))
            out[f"{key}_late_orig"] = float(np.mean(o[late_start:]))
            out[f"{key}_late_cf"] = float(np.mean(c[late_start:]))
            out[f"{key}_late_delta"] = float(np.mean(d[late_start:]))
            out[f"{key}_late_abs_delta"] = float(np.mean(np.abs(d[late_start:])))
            late_o = np.mean(o[late_start:], axis=0)
            late_c = np.mean(c[late_start:], axis=0)
            late_d = late_c - late_o
            out[f"{key}_max_head_abs_delta"] = float(np.max(np.abs(late_d)))
            out[f"{key}_argmax_head_abs_delta"] = int(np.argmax(np.abs(late_d)))

    if "topk_source_positions" in attn_orig and "topk_source_positions" in attn_cf:
        o = np.asarray(attn_orig["topk_source_positions"])
        c = np.asarray(attn_cf["topk_source_positions"])
        overlaps = []
        for l in range(o.shape[0]):
            for h in range(o.shape[1]):
                so = set(int(x) for x in o[l, h].tolist() if x >= 0)
                sc = set(int(x) for x in c[l, h].tolist() if x >= 0)
                denom = max(len(so | sc), 1)
                overlaps.append(len(so & sc) / denom)
        out["topk_jaccard_mean"] = float(np.mean(overlaps))
        out["topk_jaccard_late"] = float(np.mean(overlaps[(late_start * o.shape[1]):]))
    return out


def headwise_attention_table(attn_orig: dict, attn_cf: dict, late_start: int) -> list[dict]:
    rows = []
    if attn_orig is None or attn_cf is None:
        return rows
    keys = ["mass_to_edit_region", "mass_to_largest_region", "mass_to_question_span", "entropy"]
    if not all(k in attn_orig and k in attn_cf for k in keys):
        return rows
    _, n_heads = np.asarray(attn_orig["entropy"]).shape
    for h in range(n_heads):
        row = {"head": h}
        for key in keys:
            o = np.asarray(attn_orig[key], dtype=np.float32)
            c = np.asarray(attn_cf[key], dtype=np.float32)
            o_h = np.mean(o[late_start:, h])
            c_h = np.mean(c[late_start:, h])
            d_h = c_h - o_h
            row[f"{key}_orig"] = float(o_h)
            row[f"{key}_cf"] = float(c_h)
            row[f"{key}_delta"] = float(d_h)
            row[f"{key}_abs_delta"] = float(abs(d_h))
        rows.append(row)
    return rows


def layerwise_headwise_attention_delta(attn_orig: dict, attn_cf: dict, key: str):
    """Return (n_layers, n_heads) delta array for a given attention metric key."""
    if attn_orig is None or attn_cf is None:
        return None
    if key not in attn_orig or key not in attn_cf:
        return None
    o = np.asarray(attn_orig[key], dtype=np.float32)
    c = np.asarray(attn_cf[key], dtype=np.float32)
    return c - o


# ---------------------------------------------------------------------------
# Metadata normalization
# ---------------------------------------------------------------------------

CORE_BIAS_TYPES = {
    "sex", "gender", "age", "race", "race_ethnicity", "pronoun", "name",
    "pregnancy_status", "reproductive_status"
}
IDENTITY_BIAS_TYPES = {
    "gender_identity", "sexual_orientation", "honorific", "kinship_role",
    "disability_identity"
}
STRUCTURAL_TYPES = {
    "insurance_status", "housing_status", "occupation", "marital_status",
    "socioeconomic_status", "family_structure", "nationality", "religion"
}
CONTROL_TYPES = {"neutral_rework", "irrelevant_surface"}
OLD_TO_NEW_TYPE = {"gender": "sex", "race": "race_ethnicity"}


def safe_str(x):
    if x is None:
        return None
    return str(x)


def normalize_intervention_type(pm: dict) -> str:
    raw = (pm.get("intervention_type") or pm.get("attribute_type")
           or pm.get("attribute") or pm.get("dimension"))
    raw = safe_str(raw)
    if raw is None:
        return "unknown"
    return OLD_TO_NEW_TYPE.get(raw, raw)


def normalize_analysis_group(pm: dict, intervention_type: str) -> str:
    bucket = safe_str(pm.get("analysis_bucket"))
    if bucket in {"core_bias", "identity_bias", "structural_context", "control"}:
        return bucket
    analysis_class = safe_str(pm.get("analysis_class"))
    control_subtype = safe_str(pm.get("control_subtype"))
    if analysis_class == "control" or intervention_type in CONTROL_TYPES:
        return "control"
    if analysis_class in {"class1", "class2", "class3"}:
        if intervention_type in STRUCTURAL_TYPES:
            return "structural_context"
        if intervention_type in IDENTITY_BIAS_TYPES:
            return "identity_bias"
        return "core_bias"
    if intervention_type in CONTROL_TYPES:
        return "control"
    if intervention_type in STRUCTURAL_TYPES:
        return "structural_context"
    if intervention_type in IDENTITY_BIAS_TYPES:
        return "identity_bias"
    if intervention_type in CORE_BIAS_TYPES:
        return "core_bias"
    if control_subtype in {"neutral_rework", "irrelevant_surface"}:
        return "control"
    return "other"


def normalize_control_subtype(pm: dict, intervention_type: str):
    control_subtype = safe_str(pm.get("control_subtype"))
    if control_subtype in {"neutral_rework", "irrelevant_surface"}:
        return control_subtype
    if intervention_type in {"neutral_rework", "irrelevant_surface"}:
        return intervention_type
    return None


def normalize_edit_locality(pm: dict) -> str:
    old = safe_str(pm.get("edit_locality"))
    if old in {"minimal", "sentence_level", "broader", "single"}:
        return old
    edit_strength = safe_str(pm.get("edit_strength"))
    edit_scope = safe_str(pm.get("edit_scope"))
    n_changed = pm.get("n_tokens_changed")
    if edit_strength == "minimal":
        return "minimal"
    if edit_strength == "single":
        if edit_scope == "sentence":
            return "sentence_level"
        if edit_scope == "multi_sentence":
            return "broader"
        if edit_scope in {"token", "phrase"}:
            return "single"
    if edit_scope == "sentence":
        return "sentence_level"
    if edit_scope == "multi_sentence":
        return "broader"
    if edit_scope in {"token", "phrase"}:
        return "minimal"
    if isinstance(n_changed, (int, float)):
        if n_changed <= 3:
            return "minimal"
        if n_changed <= 12:
            return "single"
        return "broader"
    return "unknown"


def normalize_label(pm: dict):
    return (safe_str(pm.get("label"))
            or safe_str(pm.get("attribute_value_counterfactual"))
            or safe_str(pm.get("counterfactual_value"))
            or safe_str(pm.get("value")))


def normalize_metadata(pm: dict) -> dict:
    intervention_type = normalize_intervention_type(pm)
    out = dict(pm)
    out["normalized_attribute"] = intervention_type
    out["normalized_group"] = normalize_analysis_group(pm, intervention_type)
    out["normalized_control_subtype"] = normalize_control_subtype(pm, intervention_type)
    out["normalized_edit_locality"] = normalize_edit_locality(pm)
    out["normalized_label"] = normalize_label(pm)
    return out


def group_alias(g: str) -> str:
    return {"core_bias": "Core bias", "identity_bias": "Identity bias",
            "structural_context": "Structural context", "control": "Control",
            "other": "Other"}.get(g, g)


GROUP_COLORS = {
    "core_bias": "#d62728", "identity_bias": "#ff7f0e",
    "structural_context": "#2ca02c", "control": "#7f7f7f", "other": "#9467bd",
}

# Cycle of distinguishable colors for within-attribute label plots
LABEL_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]


# ---------------------------------------------------------------------------
# Bootstrap / statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=2000, ci=0.95, stat_fn=np.mean):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_stats = np.array([stat_fn(values[rng.randint(0, n, size=n)]) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(stat_fn(values)), float(np.percentile(boot_stats, 100 * alpha)), float(np.percentile(boot_stats, 100 * (1 - alpha)))


def cohens_d(group_a, group_b):
    """Cohen's d effect size between two groups."""
    a, b = np.asarray(group_a, dtype=np.float64), np.asarray(group_b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def permutation_test(group_a, group_b, stat_fn=np.mean, n_perms=5000):
    """Two-sided permutation test for difference in stat_fn."""
    a, b = np.asarray(group_a, dtype=np.float64), np.asarray(group_b, dtype=np.float64)
    observed = abs(stat_fn(a) - stat_fn(b))
    combined = np.concatenate([a, b])
    na = len(a)
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perms):
        rng.shuffle(combined)
        perm_stat = abs(stat_fn(combined[:na]) - stat_fn(combined[na:]))
        if perm_stat >= observed:
            count += 1
    return count / n_perms


def partial_correlation(x, y, z):
    """Partial correlation of x and y controlling for z (Spearman-based)."""
    if not HAS_SCIPY or len(x) < 5:
        return 0.0, 1.0
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    rxy, _ = sp_stats.spearmanr(x, y)
    rxz, _ = sp_stats.spearmanr(x, z)
    ryz, _ = sp_stats.spearmanr(y, z)
    denom = np.sqrt(max(1 - rxz ** 2, 1e-10)) * np.sqrt(max(1 - ryz ** 2, 1e-10))
    r_partial = (rxy - rxz * ryz) / denom
    # Approximate p-value
    n = len(x)
    t_stat = r_partial * np.sqrt(max(n - 3, 1)) / np.sqrt(max(1 - r_partial ** 2, 1e-10))
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(n - 3, 1)))
    return float(r_partial), float(p_val)


# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------

def print_group(name, group):
    if not group:
        print(f"\n  {name}: NO DATA")
        return
    n = len(group)
    print(f"\n  {name} (n={n}):")
    print(f"    Flip rate:               {np.mean([r['answer_flip'] for r in group]):.3f}")
    print(f"    Correctness flip:        {np.mean([r['correctness_flip'] for r in group]):.3f}")
    print(f"    Mean |Δ gold logit|:     {np.mean([r['abs_delta_gold_logit'] for r in group]):.4f}")
    print(f"    Mean |Δ margin|:         {np.mean([r['abs_delta_margin'] for r in group]):.4f}")
    print(f"    Mean token edit ratio:   {np.mean([r.get('token_edit_ratio', 0) for r in group]):.4f}")
    print(f"    Mean norm euclid (mid→): {np.mean([r['mean_norm_euclid_mid_late'] for r in group]):.6f}")
    if any(r.get("has_edit_repr") for r in group):
        edit_grp = [r for r in group if r.get("has_edit_repr")]
        print(f"    Edit-pos norm euclid:    {np.mean([r['edit_mean_norm_euclid_mid_late'] for r in edit_grp]):.6f} (n={len(edit_grp)})")
    if any(r.get("has_attention") for r in group):
        attn_grp = [r for r in group if r.get("has_attention")]
        vals = [r["final_attn_mass_to_edit_region_late_abs_delta"]
                for r in attn_grp if "final_attn_mass_to_edit_region_late_abs_delta" in r]
        if vals:
            print(f"    Final attn |Δ mass-to-edit| late: {np.mean(vals):.6f} (n={len(vals)})")


def rank_candidate_heads(head_df: "pd.DataFrame") -> "pd.DataFrame":
    df = head_df.copy()
    for col in ["mass_to_edit_region_abs_delta", "mass_to_question_span_abs_delta",
                 "entropy_abs_delta", "abs_delta_gold_logit"]:
        if col not in df.columns:
            df[col] = np.nan
    grouped = df.groupby(["normalized_group", "head"], dropna=False).agg(
        n=("head", "size"),
        mean_abs_edit_mass_shift=("mass_to_edit_region_abs_delta", "mean"),
        mean_abs_question_mass_shift=("mass_to_question_span_abs_delta", "mean"),
        mean_abs_entropy_shift=("entropy_abs_delta", "mean"),
        mean_abs_delta_gold_logit=("abs_delta_gold_logit", "mean"),
        flip_rate=("answer_flip", "mean"),
    ).reset_index()

    def zsafe(x):
        s = x.std(ddof=0)
        return np.zeros(len(x)) if (s == 0 or np.isnan(s)) else (x - x.mean()) / s

    grouped["z_edit"] = grouped.groupby("normalized_group")["mean_abs_edit_mass_shift"].transform(zsafe)
    grouped["z_entropy"] = grouped.groupby("normalized_group")["mean_abs_entropy_shift"].transform(zsafe)
    grouped["z_behavior"] = grouped.groupby("normalized_group")["mean_abs_delta_gold_logit"].transform(zsafe)
    grouped["candidate_score"] = grouped["z_edit"] + 0.5 * grouped["z_entropy"] + 0.75 * grouped["z_behavior"]
    return grouped.sort_values(["normalized_group", "candidate_score", "mean_abs_edit_mass_shift"],
                               ascending=[True, False, False])


# ---------------------------------------------------------------------------
# (a) N-adjusted comparisons
# ---------------------------------------------------------------------------

def print_n_adjusted_comparison(rows: list, outdir: Path):
    """Subsample to min group size and report Cohen's d + permutation p-values."""
    groups = ["core_bias", "identity_bias", "structural_context", "control"]
    group_rows = {g: [r for r in rows if r["normalized_group"] == g] for g in groups}
    group_rows = {g: v for g, v in group_rows.items() if len(v) >= 5}

    if len(group_rows) < 2:
        return

    min_n = min(len(v) for v in group_rows.values())
    print(f"\n{'='*70}")
    print(f"N-ADJUSTED COMPARISON (subsampled to n={min_n} per group)")
    print(f"{'='*70}")

    metrics = [("abs_delta_gold_logit", "|Δ gold logit|"), ("abs_delta_margin", "|Δ margin|"),
               ("mean_norm_euclid_mid_late", "norm euclid mid→late")]

    rng = np.random.RandomState(42)
    subsampled = {}
    for g, grp in group_rows.items():
        idx = rng.choice(len(grp), size=min_n, replace=False)
        subsampled[g] = [grp[i] for i in idx]

    ctrl_key = "control" if "control" in subsampled else None

    results = []
    for g in [k for k in subsampled if k != "control"]:
        print(f"\n  {group_alias(g)} (n={min_n}):")
        for metric_key, metric_label in metrics:
            focal_vals = [r[metric_key] for r in subsampled[g] if metric_key in r]
            mean, lo, hi = bootstrap_ci(focal_vals)
            print(f"    {metric_label}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

            if ctrl_key and ctrl_key in subsampled:
                ctrl_vals = [r[metric_key] for r in subsampled[ctrl_key] if metric_key in r]
                d = cohens_d(focal_vals, ctrl_vals)
                p = permutation_test(focal_vals, ctrl_vals) if HAS_SCIPY else float("nan")
                print(f"      vs Control: Cohen's d={d:.3f}, permutation p={p:.4f}")
                results.append({"group": g, "metric": metric_label, "mean": mean,
                                "ci_lo": lo, "ci_hi": hi, "cohens_d": d, "perm_p": p, "n": min_n})

    if HAS_PANDAS and results:
        pd.DataFrame(results).to_csv(outdir / "n_adjusted_comparisons.csv", index=False)
        print(f"  Saved {outdir / 'n_adjusted_comparisons.csv'}")


# ---------------------------------------------------------------------------
# (b) Within-attribute label heatmaps
# ---------------------------------------------------------------------------

def plot_within_attribute_label_heatmaps(rows: list, store, n_layers: int, outdir: Path):
    """For each attribute with multiple labels, show layer×head heatmaps per label + difference."""
    if not HAS_PLT:
        return

    by_attr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r.get("has_attention_final"):
            continue
        attr = r["normalized_attribute"]
        label = r.get("normalized_label")
        if label is None:
            continue
        by_attr[attr][label].append(r)

    plot_dir = outdir / "within_attribute_heatmaps"
    plot_dir.mkdir(exist_ok=True)
    n_plotted = 0

    for attr, label_dict in by_attr.items():
        # Only attributes with at least 2 labels each having >=3 examples
        eligible_labels = {lab: rws for lab, rws in label_dict.items() if len(rws) >= 3}
        if len(eligible_labels) < 2:
            continue

        labels_sorted = sorted(eligible_labels.keys(), key=lambda l: -len(eligible_labels[l]))[:5]

        # Collect mean layer×head edit_mass delta for each label
        label_maps = {}
        for lab in labels_sorted:
            deltas = []
            for r in eligible_labels[lab]:
                od = store.get_original(r["question_id"])
                cd = store.get_cf(r["pair_key"])
                if od is None or cd is None:
                    continue
                a_o = _normalize_attention_summary(od.get("attention_summary", {}).get("final_token"))
                a_c = _normalize_attention_summary(cd.get("attention_summary", {}).get("final_token"))
                lh = layerwise_headwise_attention_delta(a_o, a_c, "mass_to_edit_region")
                if lh is not None:
                    deltas.append(lh)
            if deltas:
                label_maps[lab] = np.mean(np.stack(deltas), axis=0)

        if len(label_maps) < 2:
            continue

        n_panels = len(label_maps)
        # Add pairwise difference panels for top 2 labels
        top2 = list(label_maps.keys())[:2]
        show_diff = len(top2) == 2

        total_panels = n_panels + (1 if show_diff else 0)
        fig, axes = plt.subplots(1, total_panels, figsize=(5.5 * total_panels, 7), squeeze=False)

        vmax = max(np.max(np.abs(m)) for m in label_maps.values())
        if vmax < 1e-10:
            plt.close(fig)
            continue
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for i, lab in enumerate(label_maps):
            ax = axes[0, i]
            im = ax.imshow(label_maps[lab], aspect="auto", cmap="RdBu_r", norm=norm)
            n_lab = len(eligible_labels[lab])
            ax.set_title(f"{lab}\n(n={n_lab})", fontsize=10)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")

        if show_diff:
            ax = axes[0, n_panels]
            diff = label_maps[top2[0]] - label_maps[top2[1]]
            dmax = max(np.max(np.abs(diff)), 1e-10)
            norm_d = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
            im2 = ax.imshow(diff, aspect="auto", cmap="PiYG", norm=norm_d)
            ax.set_title(f"Δ: {top2[0]}\n− {top2[1]}", fontsize=10)
            ax.set_xlabel("Head")
            fig.colorbar(im2, ax=ax, shrink=0.5, label="Difference")

        fig.colorbar(im, ax=axes[0, :n_panels].tolist(), label="Mean Δ Edit Mass", shrink=0.5)
        fig.suptitle(f"Within-Attribute Heatmap: {attr}", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{attr}_label_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1

    if n_plotted:
        print(f"Saved {n_plotted} within-attribute label heatmaps to {plot_dir}")


# ---------------------------------------------------------------------------
# (c) Vig-style per-position attention bar charts
# ---------------------------------------------------------------------------

def _classify_position(pos: int, pm: dict, n_tokens: int) -> str:
    """Classify a token position into a semantic region."""
    edit_positions = set(pm.get("cf_edit_positions", []) or [])
    largest_positions = set(pm.get("largest_cf_positions", []) or [])
    # Question stem is roughly tokens from offset to offset+stem_len
    # Answer region is roughly the last ~20% of tokens before the final "Answer:" token
    if pos in edit_positions:
        return "edit"
    if pos in largest_positions:
        return "largest_edit"
    # Rough heuristic: answer choices are in the last portion
    if n_tokens > 0 and pos >= n_tokens * 0.75:
        return "answer_region"
    return "question"


def plot_vig_style_attention(rows: list, store, n_examples: int, outdir: Path):
    """Vig-style horizontal bar charts showing where candidate heads attend,
    comparing original vs counterfactual for illustrative examples."""
    if not HAS_PLT:
        return

    # Pick examples with flips and attention data
    candidates = sorted(
        [r for r in rows if r.get("has_attention_final") and r.get("answer_flip")],
        key=lambda r: r.get("abs_delta_gold_logit", 0), reverse=True
    )
    if len(candidates) < n_examples:
        candidates += sorted(
            [r for r in rows if r.get("has_attention_final") and not r.get("answer_flip")],
            key=lambda r: r.get("abs_delta_gold_logit", 0), reverse=True
        )[:n_examples - len(candidates)]

    examples = candidates[:n_examples]
    if not examples:
        return

    plot_dir = outdir / "vig_style_examples"
    plot_dir.mkdir(exist_ok=True)

    REGION_COLORS = {"edit": "#d62728", "largest_edit": "#ff7f0e",
                     "question": "#377eb8", "answer_region": "#2ca02c", "other": "#999999"}

    for ex_idx, row in enumerate(examples):
        od = store.get_original(row["question_id"])
        cd = store.get_cf(row["pair_key"])
        if od is None or cd is None:
            continue

        a_o_raw = od.get("attention_summary", {}).get("final_token")
        a_c_raw = cd.get("attention_summary", {}).get("final_token")
        a_o = _normalize_attention_summary(a_o_raw)
        a_c = _normalize_attention_summary(a_c_raw)
        if not a_o or not a_c:
            continue
        if "topk_source_positions" not in a_o or "topk_weights" not in a_o:
            continue

        o_pos = np.asarray(a_o["topk_source_positions"])  # (layers, heads, k)
        o_vals = np.asarray(a_o.get("topk_weights", a_o.get("topk_values")))
        c_pos = np.asarray(a_c["topk_source_positions"])
        c_vals = np.asarray(a_c.get("topk_weights", a_c.get("topk_values")))

        n_layers_attn, n_heads, topk = o_pos.shape

        # Find top-3 heads by edit mass shift in late layers
        mid = n_layers_attn // 2
        if "mass_to_edit_region" in a_o and "mass_to_edit_region" in a_c:
            em_o = np.asarray(a_o["mass_to_edit_region"], dtype=np.float32)
            em_c = np.asarray(a_c["mass_to_edit_region"], dtype=np.float32)
            late_delta = np.mean(np.abs(em_c[mid:] - em_o[mid:]), axis=0)  # (heads,)
            top_heads = np.argsort(late_delta)[::-1][:3]
        else:
            top_heads = [0, 1, 2]

        n_orig_tokens = od.get("n_tokens", row.get("n_orig_tokens", 300))
        n_cf_tokens = cd.get("n_tokens", row.get("n_cf_tokens", 300))

        fig, axes = plt.subplots(len(top_heads), 2, figsize=(14, 3.5 * len(top_heads)))
        if len(top_heads) == 1:
            axes = axes.reshape(1, 2)

        for hi, head_idx in enumerate(top_heads):
            for side, (positions, values, n_tok) in enumerate([
                (o_pos, o_vals, n_orig_tokens),
                (c_pos, c_vals, n_cf_tokens),
            ]):
                ax = axes[hi, side]

                # Aggregate across late layers for this head
                region_mass = defaultdict(float)
                for l in range(mid, n_layers_attn):
                    for ki in range(topk):
                        pos_idx = int(positions[l, head_idx, ki])
                        val = float(values[l, head_idx, ki])
                        if pos_idx < 0:
                            continue
                        region = _classify_position(pos_idx, row, n_tok)
                        region_mass[region] += val

                # Normalize by number of layers
                n_late = n_layers_attn - mid
                for k in region_mass:
                    region_mass[k] /= n_late

                regions = ["edit", "largest_edit", "question", "answer_region"]
                vals = [region_mass.get(r, 0) for r in regions]
                colors = [REGION_COLORS[r] for r in regions]
                region_labels = ["Edit tokens", "Largest edit", "Question stem", "Answer region"]

                ax.barh(range(len(regions)), vals, color=colors)
                ax.set_yticks(range(len(regions)))
                ax.set_yticklabels(region_labels, fontsize=9)
                ax.set_xlabel("Mean Attention Weight (late layers)")
                title = "Original" if side == 0 else "Counterfactual"
                ax.set_title(f"{title} — Head {head_idx}", fontsize=10)
                ax.set_xlim(0, max(max(vals) * 1.2, 0.05))

        attr = row.get("normalized_attribute", "?")
        label = row.get("normalized_label", "?")
        flip_str = "FLIP" if row.get("answer_flip") else "no flip"
        fig.suptitle(
            f"Example {ex_idx}: {attr} → {label} | {flip_str} | "
            f"pred: {row.get('orig_predicted', '?')}→{row.get('cf_predicted', '?')} | "
            f"|Δℓ|={row.get('abs_delta_gold_logit', 0):.3f}",
            fontsize=11, y=1.02
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"vig_example_{ex_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {min(n_examples, len(examples))} Vig-style attention bar charts to {plot_dir}")


# ---------------------------------------------------------------------------
# (d) Correctness-aware flip statistics
# ---------------------------------------------------------------------------

def print_correctness_flip_stats(rows: list, outdir: Path):
    """Print and save correct→wrong, wrong→correct, etc. stats per group."""
    groups = ["core_bias", "identity_bias", "structural_context", "control"]

    print(f"\n{'='*70}")
    print("CORRECTNESS-AWARE FLIP ANALYSIS")
    print(f"{'='*70}")

    all_stats = []
    for g in groups:
        grp = [r for r in rows if r["normalized_group"] == g]
        if not grp:
            continue
        n = len(grp)
        n_flips = sum(r["answer_flip"] for r in grp)
        correct_to_wrong = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and not r["cf_correct"])
        wrong_to_correct = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and r["cf_correct"])
        correct_to_correct_diff = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and r["cf_correct"])
        wrong_to_wrong_diff = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and not r["cf_correct"])
        no_flip = n - n_flips
        no_flip_correct = sum(1 for r in grp if not r["answer_flip"] and r["orig_correct"])
        no_flip_wrong = sum(1 for r in grp if not r["answer_flip"] and not r["orig_correct"])

        stat = {
            "group": g, "n": n, "n_flips": n_flips, "flip_rate": n_flips / max(n, 1),
            "correct_to_wrong": correct_to_wrong, "wrong_to_correct": wrong_to_correct,
            "correct_to_correct_diff": correct_to_correct_diff, "wrong_to_wrong_diff": wrong_to_wrong_diff,
            "no_flip": no_flip, "no_flip_correct": no_flip_correct, "no_flip_wrong": no_flip_wrong,
            "harm_rate": correct_to_wrong / max(n, 1),
            "benefit_rate": wrong_to_correct / max(n, 1),
            "net_harm": (correct_to_wrong - wrong_to_correct) / max(n, 1),
        }
        all_stats.append(stat)

        print(f"\n  {group_alias(g)} (n={n}):")
        print(f"    Total flips: {n_flips} ({n_flips/n:.1%})")
        print(f"      Correct → Wrong (HARMFUL): {correct_to_wrong} ({correct_to_wrong/max(n,1):.1%})")
        print(f"      Wrong → Correct (BENEFICIAL): {wrong_to_correct} ({wrong_to_correct/max(n,1):.1%})")
        print(f"      Correct → Correct (diff answer): {correct_to_correct_diff}")
        print(f"      Wrong → Wrong (diff answer): {wrong_to_wrong_diff}")
        print(f"    No flip: {no_flip} ({no_flip/n:.1%})")
        print(f"      Still correct: {no_flip_correct}  Still wrong: {no_flip_wrong}")
        print(f"    Net harm rate: {stat['net_harm']:.3f}")

    if HAS_PANDAS and all_stats:
        pd.DataFrame(all_stats).to_csv(outdir / "correctness_flip_stats.csv", index=False)
        print(f"\n  Saved {outdir / 'correctness_flip_stats.csv'}")

    # Per-attribute breakdown
    if HAS_PANDAS:
        print(f"\n  Per-attribute harm/benefit breakdown:")
        df = pd.DataFrame(rows)
        for attr in df["normalized_attribute"].unique():
            sub = df[df["normalized_attribute"] == attr]
            n = len(sub)
            if n < 5:
                continue
            c2w = int(((sub["answer_flip"]) & (sub["orig_correct"]) & (~sub["cf_correct"])).sum())
            w2c = int(((sub["answer_flip"]) & (~sub["orig_correct"]) & (sub["cf_correct"])).sum())
            flip_rate = float(sub["answer_flip"].mean())
            print(f"    {attr:30s} n={n:4d}  flip={flip_rate:.3f}  C→W={c2w:3d} ({c2w/n:.1%})  W→C={w2c:3d} ({w2c/n:.1%})")


def plot_correctness_flip_bars(rows: list, outdir: Path):
    """Stacked bar chart of flip types per group."""
    if not HAS_PLT:
        return

    groups = ["core_bias", "identity_bias", "structural_context", "control"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By group
    group_data = []
    for g in groups:
        grp = [r for r in rows if r["normalized_group"] == g]
        if not grp:
            continue
        n = len(grp)
        c2w = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and not r["cf_correct"])
        w2c = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and r["cf_correct"])
        c2c = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and r["cf_correct"])
        w2w = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and not r["cf_correct"])
        nf = n - (c2w + w2c + c2c + w2w)
        group_data.append((g, n, c2w / n, w2c / n, c2c / n, w2w / n, nf / n))

    if group_data:
        ax = axes[0]
        x = np.arange(len(group_data))
        labels = [group_alias(g) + f"\n(n={n})" for g, n, *_ in group_data]
        bottom = np.zeros(len(group_data))
        categories = [("Correct→Wrong", [d[2] for d in group_data], "#d62728"),
                      ("Wrong→Correct", [d[3] for d in group_data], "#2ca02c"),
                      ("Correct→Correct(diff)", [d[4] for d in group_data], "#ff7f0e"),
                      ("Wrong→Wrong(diff)", [d[5] for d in group_data], "#9467bd"),
                      ("No flip", [d[6] for d in group_data], "#cccccc")]
        for cat_name, vals, color in categories:
            ax.bar(x, vals, bottom=bottom, label=cat_name, color=color, width=0.6)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Proportion")
        ax.set_title("Flip Type Breakdown by Group")
        ax.legend(fontsize=8, loc="upper right")

    # By attribute (top 12)
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        attr_counts = df["normalized_attribute"].value_counts()
        top_attrs = attr_counts[attr_counts >= 10].head(12).index.tolist()

        attr_data = []
        for a in top_attrs:
            sub = [r for r in rows if r["normalized_attribute"] == a]
            n = len(sub)
            c2w = sum(1 for r in sub if r["answer_flip"] and r["orig_correct"] and not r["cf_correct"])
            w2c = sum(1 for r in sub if r["answer_flip"] and not r["orig_correct"] and r["cf_correct"])
            attr_data.append((a, n, c2w / n, w2c / n))

        if attr_data:
            ax = axes[1]
            x = np.arange(len(attr_data))
            harm_rates = [d[2] for d in attr_data]
            benefit_rates = [d[3] for d in attr_data]
            ax.barh(x - 0.15, harm_rates, 0.3, label="Correct→Wrong (harm)", color="#d62728")
            ax.barh(x + 0.15, benefit_rates, 0.3, label="Wrong→Correct (benefit)", color="#2ca02c")
            ax.set_yticks(x)
            ax.set_yticklabels([f"{a}\n(n={n})" for a, n, *_ in attr_data], fontsize=8)
            ax.set_xlabel("Rate")
            ax.set_title("Harm vs Benefit Rate by Attribute")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(outdir / "correctness_flip_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'correctness_flip_bars.png'}")


# ---------------------------------------------------------------------------
# (e) Token-count adjustment + layer-effect marginal bars
# ---------------------------------------------------------------------------

def print_token_adjusted_metrics(rows: list):
    """Report partial correlations controlling for token_edit_ratio."""
    if not HAS_SCIPY:
        return

    print(f"\n{'='*70}")
    print("TOKEN-COUNT ADJUSTED METRICS (partial correlations controlling for token_edit_ratio)")
    print(f"{'='*70}")

    focal = [r for r in rows if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}]
    if len(focal) < 10:
        return

    ter = [r.get("token_edit_ratio", 0) for r in focal]

    for metric, label in [
        ("abs_delta_gold_logit", "|Δ gold logit|"),
        ("abs_delta_margin", "|Δ margin|"),
        ("mean_norm_euclid_mid_late", "norm euclid mid→late"),
    ]:
        vals = [r.get(metric, 0) for r in focal]
        raw_r, raw_p = sp_stats.spearmanr(ter, vals)
        partial_r, partial_p = partial_correlation(
            vals,
            [1 if r["normalized_group"] in {"core_bias", "identity_bias"} else 0 for r in focal],
            ter
        )
        print(f"\n  {label}:")
        print(f"    Raw Spearman(token_edit_ratio, metric): r={raw_r:.3f}, p={raw_p:.4f}")
        print(f"    Partial corr(metric, is_bias | token_edit_ratio): r={partial_r:.3f}, p={partial_p:.4f}")


def plot_layerhead_heatmaps_with_marginals(lh_deltas_by_group: dict, metric_name: str, outdir: Path):
    """Layer×head heatmaps with marginal layer-effect and head-effect bars (Figure 13 style)."""
    if not HAS_PLT:
        return

    groups = [g for g in lh_deltas_by_group if len(lh_deltas_by_group[g]) > 0]
    if not groups:
        return

    for g in groups:
        stacked = np.stack(lh_deltas_by_group[g])
        mean_abs = np.mean(np.abs(stacked), axis=0)  # (layers, heads)

        if np.max(mean_abs) < 1e-10:
            continue

        # Layer effect = mean across heads per layer
        layer_effect = np.mean(mean_abs, axis=1)
        # Head effect = mean across layers per head
        head_effect = np.mean(mean_abs, axis=0)

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                               hspace=0.05, wspace=0.05)

        ax_main = fig.add_subplot(gs[1, 0])
        ax_head = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_layer = fig.add_subplot(gs[1, 1], sharey=ax_main)

        im = ax_main.imshow(mean_abs, aspect="auto", cmap="hot_r")
        ax_main.set_xlabel("Head")
        ax_main.set_ylabel("Layer")

        ax_head.bar(range(len(head_effect)), head_effect, color="#4c72b0", alpha=0.8)
        ax_head.set_ylabel("Head Effect")
        ax_head.set_title(f"{group_alias(g)} — {metric_name}\n(n={len(lh_deltas_by_group[g])})")
        plt.setp(ax_head.get_xticklabels(), visible=False)

        ax_layer.barh(range(len(layer_effect)), layer_effect, color="#c44e52", alpha=0.8)
        ax_layer.set_xlabel("Layer\nEffect")
        plt.setp(ax_layer.get_yticklabels(), visible=False)

        fig.colorbar(im, ax=ax_main, shrink=0.5, label=f"Mean |Δ| {metric_name}")
        safe_name = f"{g}_{metric_name}".replace(" ", "_").replace("/", "_")
        fig.savefig(outdir / f"layerhead_marginal_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved layer×head heatmaps with marginals for {metric_name}")


# ---------------------------------------------------------------------------
# (f) Within-group counterfactual direct comparisons
# ---------------------------------------------------------------------------

def plot_within_group_cf_comparisons(rows: list, store, n_layers: int, outdir: Path):
    """Compare different counterfactual labels within the same group directly."""
    if not HAS_PLT or not HAS_PANDAS:
        return

    plot_dir = outdir / "within_group_comparisons"
    plot_dir.mkdir(exist_ok=True)

    groups = ["core_bias", "identity_bias", "structural_context"]

    for g in groups:
        grp = [r for r in rows if r["normalized_group"] == g]
        if len(grp) < 10:
            continue

        # Group by attribute, then compare labels within each attribute
        by_attr = defaultdict(lambda: defaultdict(list))
        for r in grp:
            attr = r["normalized_attribute"]
            label = r.get("normalized_label")
            if label is not None:
                by_attr[attr][label].append(r)

        for attr, label_dict in by_attr.items():
            eligible = {lab: rws for lab, rws in label_dict.items() if len(rws) >= 3}
            if len(eligible) < 2:
                continue

            labels = sorted(eligible.keys(), key=lambda l: -len(eligible[l]))[:6]
            n_labels = len(labels)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # (a) Behavioral comparison
            ax = axes[0, 0]
            metric_data = defaultdict(list)
            for lab in labels:
                for r in eligible[lab]:
                    metric_data[lab].append(r["abs_delta_gold_logit"])

            positions = range(n_labels)
            bp_data = [metric_data[l] for l in labels]
            colors = LABEL_PALETTE[:n_labels]
            bp = ax.boxplot(bp_data, positions=range(n_labels), patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticks(range(n_labels))
            ax.set_xticklabels([f"{l}\n(n={len(eligible[l])})" for l in labels], fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("|Δ Gold Logit|")
            ax.set_title(f"|Δ Gold Logit| by Label")
            ax.grid(True, alpha=0.3, axis="y")

            # (b) Flip rate comparison
            ax = axes[0, 1]
            flip_rates = [np.mean([r["answer_flip"] for r in eligible[l]]) for l in labels]
            harm_rates = [np.mean([r["answer_flip"] and r["orig_correct"] and not r["cf_correct"]
                                   for r in eligible[l]]) for l in labels]
            x = np.arange(n_labels)
            ax.bar(x - 0.15, flip_rates, 0.3, label="Any flip", color="#4c72b0")
            ax.bar(x + 0.15, harm_rates, 0.3, label="Correct→Wrong", color="#d62728")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{l}\n(n={len(eligible[l])})" for l in labels], fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("Rate")
            ax.set_title("Flip & Harm Rate by Label")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            # (c) Repr shift comparison
            ax = axes[1, 0]
            repr_data = [[r.get("mean_norm_euclid_mid_late", 0) for r in eligible[l]] for l in labels]
            bp2 = ax.boxplot(repr_data, positions=range(n_labels), patch_artist=True, widths=0.6)
            for patch, color in zip(bp2["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticks(range(n_labels))
            ax.set_xticklabels([f"{l}" for l in labels], fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("Norm Euclid (mid→late)")
            ax.set_title("Representation Shift by Label")
            ax.grid(True, alpha=0.3, axis="y")

            # (d) Pairwise Cohen's d heatmap between labels
            ax = axes[1, 1]
            if n_labels >= 2:
                d_mat = np.zeros((n_labels, n_labels))
                for i in range(n_labels):
                    for j in range(n_labels):
                        if i != j:
                            d_mat[i, j] = cohens_d(
                                [r["abs_delta_gold_logit"] for r in eligible[labels[i]]],
                                [r["abs_delta_gold_logit"] for r in eligible[labels[j]]]
                            )
                dmax = max(np.max(np.abs(d_mat)), 0.1)
                im = ax.imshow(d_mat, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
                ax.set_xticks(range(n_labels))
                ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
                ax.set_yticks(range(n_labels))
                ax.set_yticklabels(labels, fontsize=8)
                for i in range(n_labels):
                    for j in range(n_labels):
                        if i != j:
                            ax.text(j, i, f"{d_mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
                ax.set_title("Pairwise Cohen's d\n(|Δ Gold Logit|)")
                fig.colorbar(im, ax=ax, shrink=0.6, label="Cohen's d")

            fig.suptitle(f"{group_alias(g)} — {attr}: Within-Label Comparison", fontsize=12, y=1.02)
            fig.tight_layout()
            safe_attr = attr.replace("/", "_").replace(" ", "_")
            fig.savefig(plot_dir / f"{g}_{safe_attr}_cf_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved within-group CF comparisons to {plot_dir}")


# ---------------------------------------------------------------------------
# Existing visualization functions (updated)
# ---------------------------------------------------------------------------

def plot_perlayer_repr_distance(rows: list, n_layers: int, outdir: Path):
    if not HAS_PLT:
        return
    groups_of_interest = ["core_bias", "identity_bias", "structural_context", "control"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for g in groups_of_interest:
        grp = [r for r in rows if r["normalized_group"] == g]
        if not grp:
            continue
        curves = np.array([[r.get(f"norm_euclid_L{l}", np.nan) for l in range(n_layers)] for r in grp])
        mean_curve = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0) / max(np.sqrt(len(grp)), 1)
        color = GROUP_COLORS.get(g, "#333")
        axes[0].plot(range(n_layers), mean_curve, label=f"{group_alias(g)} (n={len(grp)})", color=color, linewidth=2)
        axes[0].fill_between(range(n_layers), mean_curve - sem, mean_curve + sem, alpha=0.15, color=color)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Normalized Euclidean Distance")
    axes[0].set_title("Final-Token Representation Shift by Layer"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    for g in groups_of_interest:
        grp = [r for r in rows if r["normalized_group"] == g and r.get("has_edit_repr")]
        if not grp:
            continue
        curves = np.array([[r.get(f"edit_norm_euclid_L{l}", np.nan) for l in range(n_layers)] for r in grp])
        mean_curve = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0) / max(np.sqrt(len(grp)), 1)
        color = GROUP_COLORS.get(g, "#333")
        axes[1].plot(range(n_layers), mean_curve, label=f"{group_alias(g)} (n={len(grp)})", color=color, linewidth=2)
        axes[1].fill_between(range(n_layers), mean_curve - sem, mean_curve + sem, alpha=0.15, color=color)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Normalized Euclidean Distance")
    axes[1].set_title("Edit-Token Representation Shift by Layer"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "perlayer_repr_distance.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'perlayer_repr_distance.png'}")


def plot_answer_flip_matrices(rows: list, outdir: Path):
    if not HAS_PLT:
        return
    groups_of_interest = ["core_bias", "identity_bias", "structural_context", "control"]
    active_groups = [g for g in groups_of_interest if any(r["normalized_group"] == g for r in rows)]
    if not active_groups:
        return
    n_g = len(active_groups)
    fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 4.5), squeeze=False)
    labels = ["A", "B", "C", "D"]
    for i, g in enumerate(active_groups):
        grp = [r for r in rows if r["normalized_group"] == g]
        mat = np.zeros((4, 4), dtype=int)
        for r in grp:
            oi = AIDX.get(r.get("orig_predicted"))
            ci = AIDX.get(r.get("cf_predicted"))
            if oi is not None and ci is not None:
                mat[oi, ci] += 1
        ax = axes[0, i]
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels)
        ax.set_xlabel("CF predicted"); ax.set_ylabel("Original predicted")
        for row_i in range(4):
            for col_j in range(4):
                val = mat[row_i, col_j]
                if val > 0:
                    color = "white" if val > mat.max() * 0.6 else "black"
                    ax.text(col_j, row_i, str(val), ha="center", va="center", fontsize=10, color=color)
        n_flips = sum(r.get("answer_flip", False) for r in grp)
        ax.set_title(f"{group_alias(g)}\n(n={len(grp)}, flips={n_flips})", fontsize=11)
    fig.suptitle("Answer Prediction: Original vs Counterfactual", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "answer_flip_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'answer_flip_confusion.png'}")


def plot_bootstrap_effect_sizes(rows: list, outdir: Path):
    if not HAS_PLT:
        return
    metrics = [("abs_delta_gold_logit", "|Δ Gold Logit|"), ("abs_delta_margin", "|Δ Margin|"),
               ("abs_delta_gold_prob", "|Δ Gold Prob|"), ("mean_norm_euclid_mid_late", "Norm Euclid (mid→late)")]
    groups = ["core_bias", "identity_bias", "structural_context", "control"]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    for mi, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[0, mi]
        y_positions, y_labels = [], []
        for gi, g in enumerate(groups):
            vals = [r[metric_key] for r in rows if r["normalized_group"] == g and metric_key in r]
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            y_pos = len(groups) - gi
            y_positions.append(y_pos); y_labels.append(f"{group_alias(g)} (n={len(vals)})")
            color = GROUP_COLORS.get(g, "#333")
            ax.errorbar(mean, y_pos, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=5, markersize=8, linewidth=2)
        ax.set_yticks(y_positions); ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel(metric_label, fontsize=10); ax.set_title(metric_label, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Effect Sizes with 95% Bootstrap CIs", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "bootstrap_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'bootstrap_effect_sizes.png'}")


def plot_per_attribute_effects(rows: list, outdir: Path):
    if not HAS_PLT or not HAS_PANDAS:
        return
    df = pd.DataFrame(rows)
    attr_stats = df.groupby("normalized_attribute").agg(
        n=("answer_flip", "size"), flip_rate=("answer_flip", "mean"),
        mean_abs_dlogit=("abs_delta_gold_logit", "mean"),
    ).reset_index()
    attr_stats = attr_stats[attr_stats["n"] >= 5].sort_values("mean_abs_dlogit", ascending=True)
    if len(attr_stats) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(attr_stats) * 0.5)))
    attr_to_group = dict(zip(df["normalized_attribute"], df["normalized_group"]))
    colors = [GROUP_COLORS.get(attr_to_group.get(a, "other"), "#999") for a in attr_stats["normalized_attribute"]]
    axes[0].barh(range(len(attr_stats)), attr_stats["mean_abs_dlogit"].values, color=colors)
    axes[0].set_yticks(range(len(attr_stats))); axes[0].set_yticklabels(attr_stats["normalized_attribute"].values, fontsize=9)
    axes[0].set_xlabel("|Δ Gold Logit|"); axes[0].set_title("Mean |Δ Gold Logit| by Attribute")
    axes[1].barh(range(len(attr_stats)), attr_stats["flip_rate"].values, color=colors)
    axes[1].set_yticks(range(len(attr_stats))); axes[1].set_yticklabels(attr_stats["normalized_attribute"].values, fontsize=9)
    axes[1].set_xlabel("Flip Rate"); axes[1].set_title("Answer Flip Rate by Attribute")
    fig.tight_layout()
    fig.savefig(outdir / "per_attribute_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'per_attribute_effects.png'}")


def plot_head_clustering(head_df: "pd.DataFrame", outdir: Path):
    if not HAS_PLT or not HAS_SCIPY:
        return
    feature_cols = [c for c in head_df.columns if c.endswith("_abs_delta")]
    if not feature_cols:
        return
    pivot_frames = []
    for g in head_df["normalized_group"].dropna().unique():
        sub = head_df[head_df["normalized_group"] == g]
        per_head = sub.groupby("head")[feature_cols].mean()
        per_head.columns = [f"{g}_{c}" for c in per_head.columns]
        pivot_frames.append(per_head)
    if not pivot_frames:
        return
    combined = pd.concat(pivot_frames, axis=1).fillna(0)
    if len(combined) < 3:
        return
    X = combined.values
    dist = pdist(X, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="ward")
    fig, ax = plt.subplots(figsize=(max(12, len(combined) * 0.4), 6))
    dendrogram(Z, labels=[f"H{h}" for h in combined.index], ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Head Clustering by Attention-Shift Profile Across Bias Types")
    ax.set_ylabel("Ward Distance")
    fig.tight_layout()
    fig.savefig(outdir / "head_clustering_dendrogram.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'head_clustering_dendrogram.png'}")
    for n_clusters in [3, 5, 8]:
        if len(combined) >= n_clusters:
            combined[f"cluster_{n_clusters}"] = fcluster(Z, n_clusters, criterion="maxclust")
    combined.to_csv(outdir / "head_cluster_features.csv")


def plot_head_coactivation(head_df: "pd.DataFrame", outdir: Path):
    if not HAS_PLT or not HAS_PANDAS:
        return
    metric = "mass_to_edit_region_abs_delta"
    if metric not in head_df.columns:
        return
    pivot = head_df.pivot_table(index="pair_key", columns="head", values=metric, aggfunc="mean")
    if pivot.shape[1] < 3:
        return
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels([f"H{h}" for h in corr.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels([f"H{h}" for h in corr.index], fontsize=7)
    ax.set_title("Head Co-activation: Correlation of |Δ Edit Mass| Across Pairs")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
    fig.tight_layout()
    fig.savefig(outdir / "head_coactivation.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'head_coactivation.png'}")


def plot_topk_overlap_by_layer(rows: list, store, n_layers: int, outdir: Path):
    if not HAS_PLT:
        return
    groups_of_interest = ["core_bias", "identity_bias", "structural_context", "control"]
    group_overlaps = defaultdict(list)
    for row in rows:
        if not row.get("has_attention_final"):
            continue
        g = row["normalized_group"]
        if g not in groups_of_interest:
            continue
        od = store.get_original(row["question_id"])
        cd = store.get_cf(row["pair_key"])
        if od is None or cd is None:
            continue
        a_o = _normalize_attention_summary(od.get("attention_summary", {}).get("final_token"))
        a_c = _normalize_attention_summary(cd.get("attention_summary", {}).get("final_token"))
        if not a_o or not a_c or "topk_source_positions" not in a_o or "topk_source_positions" not in a_c:
            continue
        o_pos = np.asarray(a_o["topk_source_positions"])
        c_pos = np.asarray(a_c["topk_source_positions"])
        n_l, n_h = o_pos.shape[:2]
        layer_jaccards = np.zeros(n_l)
        for l in range(n_l):
            head_js = []
            for h in range(n_h):
                so = set(int(x) for x in o_pos[l, h] if x >= 0)
                sc = set(int(x) for x in c_pos[l, h] if x >= 0)
                denom = max(len(so | sc), 1)
                head_js.append(len(so & sc) / denom)
            layer_jaccards[l] = np.mean(head_js)
        group_overlaps[g].append(layer_jaccards)
    if not any(group_overlaps.values()):
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for g in groups_of_interest:
        if not group_overlaps[g]:
            continue
        stacked = np.stack(group_overlaps[g])
        mean_curve = np.mean(stacked, axis=0)
        sem = np.std(stacked, axis=0) / max(np.sqrt(stacked.shape[0]), 1)
        color = GROUP_COLORS.get(g, "#333")
        ax.plot(range(len(mean_curve)), mean_curve, label=f"{group_alias(g)} (n={stacked.shape[0]})", color=color, linewidth=2)
        ax.fill_between(range(len(mean_curve)), mean_curve - sem, mean_curve + sem, alpha=0.15, color=color)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Jaccard Overlap")
    ax.set_title("Top-k Attended Position Overlap (Orig vs CF) by Layer")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "topk_overlap_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'topk_overlap_by_layer.png'}")


def plot_flip_vs_repr_shift(rows: list, outdir: Path):
    if not HAS_PLT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for r in rows:
        g = r["normalized_group"]
        color = GROUP_COLORS.get(g, "#999")
        axes[0].scatter(r.get("mean_norm_euclid_mid_late", 0), r.get("abs_delta_gold_logit", 0),
                        c=color, alpha=0.5, s=15, edgecolors="none")
        if r.get("has_edit_repr"):
            axes[1].scatter(r.get("edit_mean_norm_euclid_mid_late", 0), r.get("abs_delta_gold_logit", 0),
                            c=color, alpha=0.5, s=15, edgecolors="none")
    axes[0].set_xlabel("Final-Token Norm Euclid (mid→late)"); axes[0].set_ylabel("|Δ Gold Logit|")
    axes[0].set_title("Final-Token Repr Shift vs Behavioral Effect"); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Edit-Token Norm Euclid (mid→late)"); axes[1].set_ylabel("|Δ Gold Logit|")
    axes[1].set_title("Edit-Token Repr Shift vs Behavioral Effect"); axes[1].grid(True, alpha=0.3)
    legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=group_alias(g))
                       for g in GROUP_COLORS if any(r["normalized_group"] == g for r in rows)]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "flip_vs_repr_shift.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'flip_vs_repr_shift.png'}")


def plot_gold_prob_shift_distributions(rows: list, outdir: Path):
    if not HAS_PLT:
        return
    groups_of_interest = ["core_bias", "identity_bias", "structural_context", "control"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    group_data, group_labels, group_colors_list = [], [], []
    for g in groups_of_interest:
        vals = [r["delta_gold_prob"] for r in rows if r["normalized_group"] == g]
        if vals:
            group_data.append(vals); group_labels.append(f"{group_alias(g)}\n(n={len(vals)})")
            group_colors_list.append(GROUP_COLORS.get(g, "#999"))
    if group_data:
        bp = axes[0].boxplot(group_data, vert=True, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], group_colors_list):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        axes[0].set_xticklabels(group_labels, fontsize=9)
        axes[0].set_ylabel("Δ Gold Probability"); axes[0].set_title("Gold Probability Shift by Group")
        axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8); axes[0].grid(True, alpha=0.3, axis="y")
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        top_attrs = df["normalized_attribute"].value_counts().pipe(lambda s: s[s >= 5]).head(12).index.tolist()
        attr_data, attr_labels = [], []
        for a in top_attrs:
            vals = df[df["normalized_attribute"] == a]["delta_gold_prob"].tolist()
            attr_data.append(vals); attr_labels.append(f"{a}\n(n={len(vals)})")
        if attr_data:
            bp2 = axes[1].boxplot(attr_data, vert=True, patch_artist=True, widths=0.6)
            for patch in bp2["boxes"]:
                patch.set_facecolor("#4c72b0"); patch.set_alpha(0.6)
            axes[1].set_xticklabels(attr_labels, fontsize=7, rotation=45, ha="right")
            axes[1].set_ylabel("Δ Gold Probability"); axes[1].set_title("Gold Probability Shift by Attribute")
            axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8); axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "gold_prob_shift_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'gold_prob_shift_distributions.png'}")


def plot_example_attention_comparison(rows: list, store, n_examples: int, outdir: Path):
    """Layer×head heatmap comparison for top examples."""
    if not HAS_PLT:
        return
    candidates = sorted([r for r in rows if r.get("has_attention_final") and r.get("answer_flip")],
                        key=lambda r: r.get("abs_delta_gold_logit", 0), reverse=True)
    if len(candidates) < n_examples:
        candidates += sorted([r for r in rows if r.get("has_attention_final") and not r.get("answer_flip")],
                             key=lambda r: r.get("abs_delta_gold_logit", 0), reverse=True)
    examples = candidates[:n_examples]
    if not examples:
        return
    for idx, row in enumerate(examples):
        od = store.get_original(row["question_id"])
        cd = store.get_cf(row["pair_key"])
        if od is None or cd is None:
            continue
        a_o = _normalize_attention_summary(od.get("attention_summary", {}).get("final_token"))
        a_c = _normalize_attention_summary(cd.get("attention_summary", {}).get("final_token"))
        if not a_o or not a_c:
            continue
        for metric_key, metric_label in [("mass_to_edit_region", "Edit Mass"), ("entropy", "Entropy")]:
            if metric_key not in a_o or metric_key not in a_c:
                continue
            o_vals = np.asarray(a_o[metric_key], dtype=np.float32)
            c_vals = np.asarray(a_c[metric_key], dtype=np.float32)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            vmax = max(np.max(np.abs(o_vals)), np.max(np.abs(c_vals)))
            if vmax < 1e-10:
                plt.close(fig); continue
            axes[0].imshow(o_vals, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
            axes[0].set_title(f"Original\npred={row.get('orig_predicted', '?')}"); axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer")
            axes[1].imshow(c_vals, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
            axes[1].set_title(f"Counterfactual\npred={row.get('cf_predicted', '?')}"); axes[1].set_xlabel("Head")
            delta = c_vals - o_vals
            dmax = max(np.max(np.abs(delta)), 1e-10)
            norm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
            im = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", norm=norm)
            axes[2].set_title("Δ (CF - Orig)"); axes[2].set_xlabel("Head")
            fig.colorbar(im, ax=axes[2], shrink=0.6)
            attr = row.get("normalized_attribute", "?"); label = row.get("normalized_label", "?")
            flip_str = "FLIP" if row.get("answer_flip") else "no flip"
            fig.suptitle(f"Example {idx}: {attr} → {label} | {flip_str} | |Δℓ|={row.get('abs_delta_gold_logit', 0):.3f}\n{metric_label}",
                         fontsize=12, y=1.02)
            fig.tight_layout()
            fig.savefig(outdir / f"example_{idx}_{metric_key}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {min(n_examples, len(examples))} example attention comparison plots")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(extraction_dir: str, output_dir: str, margin_threshold: float = 0.5,
                 n_example_plots: int = 5):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    store = ShardedExtractionStore(extraction_dir)
    meta = list(store.iter_pair_metadata())
    mcfg = store.model_config
    n_layers = mcfg["n_layers"]

    print(f"Loading from {extraction_dir}...")
    print(f"Model: {n_layers} layers, hidden={mcfg['hidden_size']}")
    print(f"Pairs: {len(meta)}")
    print(f"Attention extracted: {mcfg.get('extract_attention', False)}")

    rows = []
    head_rows = []
    lh_edit_mass_deltas = defaultdict(list)
    lh_entropy_deltas = defaultdict(list)
    lh_stem_mass_deltas = defaultdict(list)

    for raw_pm in meta:
        pm = normalize_metadata(raw_pm)
        qid = pm["question_id"]
        pk = pm["pair_key"]
        gold = pm["gold_answer"]

        od = store.get_original(qid)
        cd = store.get_cf(pk)
        if od is None or cd is None:
            continue

        ob = behavioral_metrics(od["logits_abcd"], gold)
        cb = behavioral_metrics(cd["logits_abcd"], gold)
        pb = pairwise_behavioral(ob, cb)
        row = {**pm, **pb}

        rm_final = repr_metrics_per_layer(od["hidden_final"], cd["hidden_final"])
        for l in range(n_layers):
            row[f"euclid_L{l}"] = rm_final["euclidean"][l]
            row[f"norm_euclid_L{l}"] = rm_final["norm_euclidean"][l]
            row[f"cos_dist_L{l}"] = rm_final["cosine_dist"][l]
            row[f"orig_norm_L{l}"] = rm_final["orig_norm"][l]

        mid = n_layers // 2
        row["mean_euclid_all"] = float(np.mean(rm_final["euclidean"]))
        row["mean_euclid_mid_late"] = float(np.mean(rm_final["euclidean"][mid:]))
        row["mean_norm_euclid_all"] = float(np.mean(rm_final["norm_euclidean"]))
        row["mean_norm_euclid_mid_late"] = float(np.mean(rm_final["norm_euclidean"][mid:]))
        row["mean_cos_dist_mid_late"] = float(np.mean(rm_final["cosine_dist"][mid:]))

        orig_edit_key = pm.get("orig_edit_key")
        od_edit = store.get_original(orig_edit_key) if orig_edit_key else None
        if (od_edit is not None and "hidden_at_edit" in od_edit and "hidden_at_edit" in cd):
            rm_edit = repr_metrics_per_layer(od_edit["hidden_at_edit"], cd["hidden_at_edit"])
            for l in range(n_layers):
                row[f"edit_euclid_L{l}"] = rm_edit["euclidean"][l]
                row[f"edit_norm_euclid_L{l}"] = rm_edit["norm_euclidean"][l]
            row["edit_mean_euclid_all"] = float(np.mean(rm_edit["euclidean"]))
            row["edit_mean_euclid_mid_late"] = float(np.mean(rm_edit["euclidean"][mid:]))
            row["edit_mean_norm_euclid_all"] = float(np.mean(rm_edit["norm_euclidean"]))
            row["edit_mean_norm_euclid_mid_late"] = float(np.mean(rm_edit["norm_euclidean"][mid:]))
            row["has_edit_repr"] = True
        else:
            row["has_edit_repr"] = False

        late_start = mid
        row["has_attention"] = False
        row["has_attention_final"] = False
        row["has_attention_edit"] = False

        od_attn = od.get("attention_summary") if isinstance(od, dict) else None
        cd_attn = cd.get("attention_summary") if isinstance(cd, dict) else None

        if isinstance(od_attn, dict) and isinstance(cd_attn, dict):
            a_o = _normalize_attention_summary(od_attn.get("final_token"))
            a_c = _normalize_attention_summary(cd_attn.get("final_token"))
            if a_o and a_c:
                am = attention_shift_metrics(a_o, a_c, late_start)
                for k, v in am.items():
                    row[f"final_attn_{k}"] = v
                row["has_attention"] = True
                row["has_attention_final"] = True
                g = pm.get("normalized_group", "other")
                for metric_key, delta_store in [
                    ("mass_to_edit_region", lh_edit_mass_deltas),
                    ("entropy", lh_entropy_deltas),
                    ("mass_to_question_span", lh_stem_mass_deltas),
                ]:
                    lh_delta = layerwise_headwise_attention_delta(a_o, a_c, metric_key)
                    if lh_delta is not None:
                        delta_store[g].append(lh_delta)

        if isinstance(od_edit, dict) and isinstance(cd, dict):
            od_edit_attn = od_edit.get("attention_summary")
            cd_attn_inner = cd.get("attention_summary")
            if isinstance(od_edit_attn, dict) and isinstance(cd_attn_inner, dict):
                a_o = _normalize_attention_summary(od_edit_attn.get("edit_region"))
                a_c = _normalize_attention_summary(cd_attn_inner.get("edit_region"))
                if a_o and a_c:
                    am = attention_shift_metrics(a_o, a_c, late_start)
                    for k, v in am.items():
                        row[f"edit_attn_{k}"] = v
                    row["has_attention"] = True
                    row["has_attention_edit"] = True
                    for hr in headwise_attention_table(a_o, a_c, late_start):
                        hr = dict(hr)
                        hr.update({
                            "question_id": qid, "pair_key": pk,
                            "normalized_group": pm.get("normalized_group"),
                            "normalized_attribute": pm.get("normalized_attribute"),
                            "normalized_label": pm.get("normalized_label"),
                            "answer_flip": row.get("answer_flip"),
                            "abs_delta_gold_logit": row.get("abs_delta_gold_logit"),
                        })
                        head_rows.append(hr)

        rows.append(row)

    print(f"Computed metrics for {len(rows)} pairs")

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "stage1_metrics_sharded.csv", index=False)
        print(f"Saved {outdir / 'stage1_metrics_sharded.csv'}")
        if head_rows:
            head_df = pd.DataFrame(head_rows)
            head_df.to_csv(outdir / "attention_head_metrics.csv", index=False)
            ranked = rank_candidate_heads(head_df)
            ranked.to_csv(outdir / "candidate_heads_ranked.csv", index=False)
            print(f"Saved attention_head_metrics.csv and candidate_heads_ranked.csv")
            print("\nTop candidate heads:")
            for grp_name in ranked["normalized_group"].dropna().unique():
                sub = ranked[ranked["normalized_group"] == grp_name].head(10)
                print(f"\n  {group_alias(grp_name)}:")
                print(sub[["head", "n", "candidate_score", "mean_abs_edit_mass_shift", "mean_abs_delta_gold_logit"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # Console reports
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GROUP COUNTS")
    print(f"{'='*70}")
    group_counts = Counter(r["normalized_group"] for r in rows)
    type_counts = Counter(r["normalized_attribute"] for r in rows)
    print("  By normalized group:")
    for k, v in group_counts.items():
        print(f"    {k}: {v}")
    print("  By intervention type:")
    for k, v in type_counts.most_common():
        print(f"    {k}: {v}")

    print(f"\n{'='*70}")
    print("A. SURFACE-MATCHED COMPARISON")
    print(f"{'='*70}")
    focal_surface = [r for r in rows if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                     and r["normalized_edit_locality"] in {"minimal", "single"}]
    ctrl_surface = [r for r in rows if r["normalized_group"] == "control"
                    and r["normalized_edit_locality"] in {"minimal", "single"}]
    ctrl_surface_irrelevant = [r for r in ctrl_surface if r["normalized_control_subtype"] == "irrelevant_surface"]
    ctrl_surface_rework = [r for r in ctrl_surface if r["normalized_control_subtype"] == "neutral_rework"]
    print_group("All focal edits (surface-matched)", focal_surface)
    print_group("Control: irrelevant_surface", ctrl_surface_irrelevant)
    print_group("Control: neutral_rework", ctrl_surface_rework)
    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        print_group(group_alias(grp_name), [r for r in focal_surface if r["normalized_group"] == grp_name])

    # (d) Correctness-aware flip stats
    print_correctness_flip_stats(rows, outdir)

    print(f"\n{'='*70}")
    print("E. CONDITIONAL ON MODEL CONFIDENCE")
    print(f"{'='*70}")
    for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
        grp_all = [r for r in rows if r["normalized_group"] == grp_name]
        grp_correct = [r for r in grp_all if r["orig_correct"]]
        grp_confident = [r for r in grp_correct if r["orig_margin"] > margin_threshold]
        print(f"\n  {group_alias(grp_name)}:")
        print(f"    All: n={len(grp_all)}")
        if grp_all:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_all]):.3f}  |Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_all]):.4f}")
        print(f"    Original correct only: n={len(grp_correct)}")
        if grp_correct:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_correct]):.3f}  |Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_correct]):.4f}")
        print(f"    Correct + margin>{margin_threshold}: n={len(grp_confident)}")
        if grp_confident:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_confident]):.3f}  |Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_confident]):.4f}")

    # (a) N-adjusted comparison
    print_n_adjusted_comparison(rows, outdir)

    # (e) Token-count adjusted metrics
    print_token_adjusted_metrics(rows)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    if HAS_PLT:
        # (e) Layer×head heatmaps with marginal bars
        for metric_key, delta_store, label in [
            ("mass_to_edit_region", lh_edit_mass_deltas, "Edit Mass"),
            ("entropy", lh_entropy_deltas, "Entropy"),
            ("mass_to_question_span", lh_stem_mass_deltas, "Question Span Mass"),
        ]:
            if any(delta_store.values()):
                plot_layerhead_heatmaps_with_marginals(delta_store, label, outdir)

        # Per-layer repr distance
        plot_perlayer_repr_distance(rows, n_layers, outdir)

        # Answer flip confusion matrices
        plot_answer_flip_matrices(rows, outdir)

        # (d) Correctness-aware flip bars
        plot_correctness_flip_bars(rows, outdir)

        # Bootstrap effect sizes
        plot_bootstrap_effect_sizes(rows, outdir)

        # Per-attribute effects
        plot_per_attribute_effects(rows, outdir)

        # Flip vs repr shift
        plot_flip_vs_repr_shift(rows, outdir)

        # Gold prob shift distributions
        plot_gold_prob_shift_distributions(rows, outdir)

        # Example attention comparisons (layer×head)
        plot_example_attention_comparison(rows, store, n_example_plots, outdir)

        # (c) Vig-style attention bar charts
        plot_vig_style_attention(rows, store, n_example_plots, outdir)

        # Top-k overlap by layer
        plot_topk_overlap_by_layer(rows, store, n_layers, outdir)

        # (b) Within-attribute label heatmaps
        plot_within_attribute_label_heatmaps(rows, store, n_layers, outdir)

        # (f) Within-group CF comparisons
        plot_within_group_cf_comparisons(rows, store, n_layers, outdir)

    # Head-specific visualizations
    if HAS_PANDAS and HAS_PLT and head_rows:
        head_df = pd.DataFrame(head_rows)
        if HAS_SCIPY:
            plot_head_clustering(head_df, outdir)
        plot_head_coactivation(head_df, outdir)

    print(f"\n{'='*70}")
    print("STAGE 1 SHARDED ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 analysis for sharded extractions with attention summaries")
    parser.add_argument("--extraction_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="stage1_results_sharded")
    parser.add_argument("--margin_threshold", type=float, default=0.5)
    parser.add_argument("--n_example_plots", type=int, default=5,
                        help="Number of illustrative example attention comparison plots")
    args = parser.parse_args()
    run_analysis(args.extraction_dir, args.output_dir, args.margin_threshold,
                 n_example_plots=args.n_example_plots)


if __name__ == "__main__":
    main()
