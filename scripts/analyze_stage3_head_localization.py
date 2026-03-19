#!/usr/bin/env python3
"""
Stage 3: Attention Head Localization for Causal Tracing Candidate Sites.

For each focal identity family (sex_gender, race, gender_identity,
sexual_orientation), ranks attention heads by their differential response to
identity interventions (delta edit-mass relative to matched originals). Then
computes cross-family overlap to identify shared, unique, and control-nullish
heads — the primary candidate set for subsequent causal tracing.

Output
------
  stage3_results/
    head_scores.npz          — (families+control) x layers x heads score arrays
    top_heads.json           — ranked head lists per family and per attribute
    overlap_stats.json       — intersection / union / Jaccard per family pair
    plots/
      01_score_heatmaps.png  — score heatmap per family (layers × heads)
      02_top_heads_bars.png  — top-K ranked heads with error bars, per family
      03_overlap_matrix.png  — families × top-heads binary membership heatmap
      04_upset.png           — UpSet-style intersection bars
      05_within_family.png   — per-attribute head-score profiles within each family
      06_control_comparison.png — focal vs control head score comparison

Usage
-----
  python analyze_stage3_head_localization.py \\
    --extraction_dir extractions_v6 \\
    --output_dir stage3_results \\
    --top_k 20 \\
    --min_pairs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from load_sharded_extractions import ShardedExtractionStore

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
    print("Warning: matplotlib not found — plots will be skipped")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------

# Intervention types that belong to each focal family.
# sex_gender is split post-hoc by attribute label.
FAMILY_DEFS = {
    "sex_gender": {
        "intervention_types": {"sex", "sex_gender"},
        "label_filter": None,       # refined below via GENDER_IDENTITY_LABELS
        "color": "#4C96D7",
        "display": "Sex / Gender",
    },
    "gender_identity": {
        # Drawn from the same sex_gender intervention type but with identity labels
        "intervention_types": {"sex", "sex_gender", "gender_identity"},
        "label_filter": "gender_identity",
        "color": "#9B59B6",
        "display": "Gender Identity",
    },
    "race": {
        "intervention_types": {"race_ethnicity", "omit_race"},
        "label_filter": None,
        "color": "#E67E22",
        "display": "Race / Ethnicity",
    },
    "sexual_orientation": {
        "intervention_types": {"sexual_orientation"},
        "label_filter": None,
        "color": "#2ECC71",
        "display": "Sexual Orientation",
    },
    "control": {
        "intervention_types": {"neutral_rework", "irrelevant_surface"},
        "label_filter": None,
        "color": "#95A5A6",
        "display": "Control",
    },
}

# Labels that identify gender-identity (non-binary / trans) within sex_gender CFs
GENDER_IDENTITY_LABELS = {
    "non-binary person", "non-binary", "nonbinary",
    "transgender man", "transgender woman",
    "trans man", "trans woman", "transman", "transwoman",
    "gender non-conforming", "genderqueer", "agender",
}

# Labels that identify binary sex/gender
SEX_GENDER_LABELS = {
    "male", "female", "man", "woman",
    "cisgender male", "cisgender female",
    "cis male", "cis female",
}

FOCAL_FAMILIES = ["sex_gender", "gender_identity", "race", "sexual_orientation"]


# ---------------------------------------------------------------------------
# Family assignment
# ---------------------------------------------------------------------------

def assign_family(meta: dict) -> str | None:
    itype = meta.get("intervention_type", "")
    label = str(meta.get("attribute_value_counterfactual") or
                meta.get("label") or "").lower().strip()

    if itype in {"neutral_rework", "irrelevant_surface"}:
        return "control"

    if itype in {"race_ethnicity", "omit_race"}:
        return "race"

    if itype == "sexual_orientation":
        return "sexual_orientation"

    if itype == "gender_identity":
        return "gender_identity"

    if itype in {"sex", "sex_gender"}:
        # Split by label into gender_identity vs sex_gender
        for gi_label in GENDER_IDENTITY_LABELS:
            if gi_label in label:
                return "gender_identity"
        return "sex_gender"

    return None  # insurance, omit_race as standalone type, etc. — skip


# ---------------------------------------------------------------------------
# Attention key normalisation (handles both key conventions)
# ---------------------------------------------------------------------------

def get_edit_mass(attn_block: dict) -> np.ndarray | None:
    """Return (layers, heads) edit-mass array from an attention block dict."""
    if attn_block is None:
        return None
    ft = attn_block.get("final_token")
    if ft is None:
        return None
    # New key convention (from mechanistic_head_tracing.py)
    for key in ("edit_mass", "mass_to_edit_region"):
        val = ft.get(key)
        if val is not None:
            return np.asarray(val, dtype=np.float32)
    return None


# ---------------------------------------------------------------------------
# Score accumulation
# ---------------------------------------------------------------------------

def accumulate_scores(
    store: ShardedExtractionStore,
    top_k: int,
    min_pairs: int,
) -> tuple[dict, dict, dict]:
    """
    Returns:
        scores       — {family: {"sum": (L,H), "sum_sq": (L,H), "n": (L,H), "n_pos": (L,H)}}
        attr_scores  — {family: {attr_label: same dict}}
        n_layers, n_heads
    """
    n_layers = store.model_config["n_layers"]
    n_heads_model = store.model_config.get("n_heads") or store.model_config.get("num_attention_heads")

    # Determine n_heads from first pair that has attention data
    n_heads = None

    scores: dict[str, dict] = {}
    attr_scores: dict[str, dict[str, dict]] = defaultdict(dict)

    def make_acc():
        if n_heads is None:
            return None
        return {
            "sum": np.zeros((n_layers, n_heads), dtype=np.float64),
            "sum_sq": np.zeros((n_layers, n_heads), dtype=np.float64),
            "n": np.zeros((n_layers, n_heads), dtype=np.int32),
            "n_pos": np.zeros((n_layers, n_heads), dtype=np.int32),
        }

    deferred = []   # (family, attr_label, delta) before n_heads is known

    skipped_no_attn = 0
    skipped_align = 0
    processed = 0

    for meta in store.iter_pair_metadata():
        family = assign_family(meta)
        if family is None:
            continue

        if meta.get("alignment_failed", True):
            skipped_align += 1
            continue

        pair_key = meta["pair_key"]
        orig_edit_key = meta.get("orig_edit_key", f"{meta['question_id']}__edit__{pair_key}")

        cf_data = store.get_cf(pair_key)
        orig_data = store.get_original(orig_edit_key)

        if cf_data is None or orig_data is None:
            # Fall back to plain original (no edit positions)
            orig_data = store.get_original(meta["question_id"])
            if cf_data is None or orig_data is None:
                skipped_no_attn += 1
                continue

        cf_mass = get_edit_mass(cf_data.get("attention_summary"))
        orig_mass = get_edit_mass(orig_data.get("attention_summary"))

        if cf_mass is None:
            skipped_no_attn += 1
            continue

        if orig_mass is None or orig_mass.shape != cf_mass.shape:
            # Use zeros as baseline if original edit positions not available
            orig_mass = np.zeros_like(cf_mass)

        delta = cf_mass - orig_mass  # (L, H) — positive means more attention to edited tokens

        if n_heads is None:
            n_heads = delta.shape[1]
            # Flush deferred
            for f, al, d in deferred:
                if f not in scores:
                    scores[f] = {
                        "sum": np.zeros((n_layers, n_heads), dtype=np.float64),
                        "sum_sq": np.zeros((n_layers, n_heads), dtype=np.float64),
                        "n": np.zeros((n_layers, n_heads), dtype=np.int32),
                        "n_pos": np.zeros((n_layers, n_heads), dtype=np.int32),
                    }
                acc = scores[f]
                acc["sum"] += d
                acc["sum_sq"] += d ** 2
                acc["n"] += 1
                acc["n_pos"] += (d > 0).astype(np.int32)
                if al not in attr_scores[f]:
                    attr_scores[f][al] = {
                        "sum": np.zeros((n_layers, n_heads), dtype=np.float64),
                        "sum_sq": np.zeros((n_layers, n_heads), dtype=np.float64),
                        "n": np.zeros((n_layers, n_heads), dtype=np.int32),
                        "n_pos": np.zeros((n_layers, n_heads), dtype=np.int32),
                    }
                a = attr_scores[f][al]
                a["sum"] += d
                a["sum_sq"] += d ** 2
                a["n"] += 1
                a["n_pos"] += (d > 0).astype(np.int32)
            deferred = []

        attr_label = str(
            meta.get("attribute_value_counterfactual") or
            meta.get("label") or
            meta.get("intervention_type") or "unknown"
        )

        if n_heads is None:
            deferred.append((family, attr_label, delta))
            continue

        if family not in scores:
            scores[family] = {
                "sum": np.zeros((n_layers, n_heads), dtype=np.float64),
                "sum_sq": np.zeros((n_layers, n_heads), dtype=np.float64),
                "n": np.zeros((n_layers, n_heads), dtype=np.int32),
                "n_pos": np.zeros((n_layers, n_heads), dtype=np.int32),
            }
        acc = scores[family]
        acc["sum"] += delta
        acc["sum_sq"] += delta ** 2
        acc["n"] += 1
        acc["n_pos"] += (delta > 0).astype(np.int32)

        if attr_label not in attr_scores[family]:
            attr_scores[family][attr_label] = {
                "sum": np.zeros((n_layers, n_heads), dtype=np.float64),
                "sum_sq": np.zeros((n_layers, n_heads), dtype=np.float64),
                "n": np.zeros((n_layers, n_heads), dtype=np.int32),
                "n_pos": np.zeros((n_layers, n_heads), dtype=np.int32),
            }
        a = attr_scores[family][attr_label]
        a["sum"] += delta
        a["sum_sq"] += delta ** 2
        a["n"] += 1
        a["n_pos"] += (delta > 0).astype(np.int32)

        processed += 1

    print(f"\nPairs processed: {processed}")
    print(f"  Skipped (no attention data): {skipped_no_attn}")
    print(f"  Skipped (alignment failed): {skipped_align}")
    if n_heads is None:
        n_heads = n_heads_model or 32
        print(f"  Warning: no attention data found — using n_heads={n_heads} from config")

    print(f"\nFamily pair counts:")
    for fam, acc in scores.items():
        n_total = int(acc["n"].max())
        print(f"  {fam}: {n_total} pairs")

    return scores, dict(attr_scores), n_layers, n_heads


# ---------------------------------------------------------------------------
# Score computation from accumulators
# ---------------------------------------------------------------------------

def compute_mean_scores(scores: dict) -> dict[str, np.ndarray]:
    """Convert accumulators to mean delta_edit_mass per (family, layer, head)."""
    means = {}
    for family, acc in scores.items():
        n = acc["n"].astype(np.float64)
        n = np.maximum(n, 1)
        means[family] = (acc["sum"] / n).astype(np.float32)
    return means


def compute_std_scores(scores: dict) -> dict[str, np.ndarray]:
    stds = {}
    for family, acc in scores.items():
        n = acc["n"].astype(np.float64)
        n = np.maximum(n, 1)
        mean = acc["sum"] / n
        var = acc["sum_sq"] / n - mean ** 2
        stds[family] = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    return stds


def compute_frac_positive(scores: dict) -> dict[str, np.ndarray]:
    fracs = {}
    for family, acc in scores.items():
        n = acc["n"].astype(np.float64)
        n = np.maximum(n, 1)
        fracs[family] = (acc["n_pos"] / n).astype(np.float32)
    return fracs


# ---------------------------------------------------------------------------
# Head ranking
# ---------------------------------------------------------------------------

def rank_heads(mean_scores: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    """Return list of (layer, head, score) sorted descending."""
    flat = mean_scores.flatten()
    indices = np.argsort(flat)[::-1][:top_k]
    n_heads = mean_scores.shape[1]
    return [(int(i // n_heads), int(i % n_heads), float(flat[i])) for i in indices]


def head_key(layer: int, head: int) -> str:
    return f"L{layer:02d}H{head:02d}"


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------

def compute_overlap(top_heads_per_family: dict[str, list]) -> dict:
    """
    Compute pairwise intersection, union, Jaccard, and multi-family intersections.
    top_heads_per_family: {family: [(layer, head, score), ...]}
    """
    sets = {f: {(l, h) for l, h, _ in heads}
            for f, heads in top_heads_per_family.items()}

    results = {"pairwise": {}, "multiway": {}}

    families = list(sets.keys())
    for a, b in combinations(families, 2):
        inter = sets[a] & sets[b]
        union = sets[a] | sets[b]
        j = len(inter) / len(union) if union else 0.0
        key = f"{a}_x_{b}"
        results["pairwise"][key] = {
            "intersection": sorted((l, h) for l, h in inter),
            "n_intersection": len(inter),
            "n_union": len(union),
            "jaccard": round(j, 4),
        }

    # Multi-family intersections across all focal families
    focal_sets = [sets[f] for f in FOCAL_FAMILIES if f in sets]
    if focal_sets:
        shared_all = focal_sets[0].copy()
        for s in focal_sets[1:]:
            shared_all &= s
        results["multiway"]["shared_all_focal"] = sorted(
            (l, h) for l, h in shared_all
        )

    # Unique per family (in focal top-K but not in any other focal top-K)
    for fam in FOCAL_FAMILIES:
        if fam not in sets:
            continue
        others = set.union(*[sets[f] for f in FOCAL_FAMILIES if f != fam and f in sets]) if len(sets) > 1 else set()
        unique = sets[fam] - others
        results["multiway"][f"unique_{fam}"] = sorted((l, h) for l, h in unique)

    # Nullish: high for control, NOT in any focal top-K
    if "control" in sets:
        focal_union = set.union(*[sets[f] for f in FOCAL_FAMILIES if f in sets]) if any(f in sets for f in FOCAL_FAMILIES) else set()
        nullish = sets["control"] - focal_union
        results["multiway"]["nullish_control"] = sorted((l, h) for l, h in nullish)

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _head_score_heatmap(ax, score_mat: np.ndarray, title: str, color: str, vmax=None):
    n_layers, n_heads = score_mat.shape
    if vmax is None:
        vmax = max(float(np.percentile(score_mat, 99)), 1e-6)
    vmin = min(float(np.percentile(score_mat, 1)), 0.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else None
    cmap = "RdBu_r" if vmin < 0 else "Blues"
    im = ax.imshow(score_mat, aspect="auto", cmap=cmap, norm=norm,
                   vmin=vmin if norm is None else None,
                   vmax=vmax if norm is None else None,
                   interpolation="nearest")
    ax.set_title(title, fontsize=10, color=color, fontweight="bold")
    ax.set_xlabel("Head", fontsize=8)
    ax.set_ylabel("Layer", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ edit-mass")
    return im


# ---------------------------------------------------------------------------
# Plot 01: Score heatmaps
# ---------------------------------------------------------------------------

def plot_score_heatmaps(mean_scores: dict, output_path: Path):
    families = FOCAL_FAMILIES + ["control"]
    n_panels = len([f for f in families if f in mean_scores])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6), squeeze=False)
    axes = axes[0]

    # Shared vmax across focal families (not control)
    focal_vals = np.concatenate([mean_scores[f].flatten()
                                 for f in FOCAL_FAMILIES if f in mean_scores])
    vmax_shared = float(np.percentile(focal_vals, 99)) if len(focal_vals) else 1.0

    panel = 0
    for fam in families:
        if fam not in mean_scores:
            continue
        color = FAMILY_DEFS[fam]["color"]
        label = FAMILY_DEFS[fam]["display"]
        vmax = vmax_shared if fam != "control" else None
        _head_score_heatmap(axes[panel], mean_scores[fam], label, color, vmax=vmax)
        panel += 1

    fig.suptitle("Mean Δ Edit-Mass per (Layer, Head)\n"
                 "Positive = head attends more to identity tokens in CF vs Original",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 02: Top-K head bars
# ---------------------------------------------------------------------------

def plot_top_heads_bars(
    top_heads: dict[str, list],
    mean_scores: dict,
    std_scores: dict,
    frac_positive: dict,
    output_path: Path,
    top_k: int,
):
    families = [f for f in FOCAL_FAMILIES if f in top_heads]
    n = len(families)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)

    for row, fam in enumerate(families):
        ax = axes[row][0]
        heads = top_heads[fam][:top_k]
        labels = [f"L{l}H{h}" for l, h, _ in heads]
        vals = [mean_scores[fam][l, h] for l, h, _ in heads]
        errs = [std_scores[fam][l, h] for l, h, _ in heads]
        fracs = [frac_positive[fam][l, h] for l, h, _ in heads]

        color = FAMILY_DEFS[fam]["color"]
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=color, alpha=0.7, width=0.6, yerr=errs,
                      capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})

        # Annotate frac positive
        for xi, (val, frac) in enumerate(zip(vals, fracs)):
            ax.text(xi, val + max(errs) * 0.1 + 0.002, f"{frac:.0%}",
                    ha="center", va="bottom", fontsize=6.5, color="black")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Mean Δ edit-mass", fontsize=9)
        ax.set_title(f"{FAMILY_DEFS[fam]['display']} — Top {top_k} Heads  "
                     f"(% = fraction of pairs with positive delta)",
                     fontsize=10, color=color, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

        # Mark control baseline for same heads
        if "control" in mean_scores:
            ctrl_vals = [mean_scores["control"][l, h] for l, h, _ in heads]
            ax.plot(x, ctrl_vals, "k--", linewidth=1.0, alpha=0.6, label="control baseline")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 03: Overlap membership heatmap
# ---------------------------------------------------------------------------

def plot_overlap_matrix(
    top_heads: dict[str, list],
    mean_scores: dict,
    output_path: Path,
    top_k: int,
):
    families = [f for f in FOCAL_FAMILIES if f in top_heads]
    if not families:
        return

    # Union of all top-K heads across focal families
    all_heads = []
    seen = set()
    for fam in families:
        for l, h, _ in top_heads[fam][:top_k]:
            if (l, h) not in seen:
                all_heads.append((l, h))
                seen.add((l, h))

    # Sort by layer then head
    all_heads.sort()
    head_labels = [f"L{l}H{h}" for l, h in all_heads]
    n_heads_plot = len(all_heads)

    # Binary membership + score matrix
    membership = np.zeros((len(families), n_heads_plot), dtype=np.float32)
    score_mat = np.zeros((len(families), n_heads_plot), dtype=np.float32)

    top_sets = {fam: {(l, h) for l, h, _ in top_heads[fam][:top_k]} for fam in families}

    for fi, fam in enumerate(families):
        for hi, (l, h) in enumerate(all_heads):
            score_mat[fi, hi] = mean_scores[fam][l, h]
            membership[fi, hi] = 1.0 if (l, h) in top_sets[fam] else 0.0

    fig, (ax_mem, ax_score) = plt.subplots(
        2, 1, figsize=(max(14, n_heads_plot * 0.45), 7),
        gridspec_kw={"height_ratios": [1, 1.5]}, squeeze=True
    )

    # Membership
    ax_mem.imshow(membership, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                  interpolation="nearest")
    ax_mem.set_yticks(range(len(families)))
    ax_mem.set_yticklabels([FAMILY_DEFS[f]["display"] for f in families], fontsize=9)
    ax_mem.set_xticks(range(n_heads_plot))
    ax_mem.set_xticklabels(head_labels, rotation=90, fontsize=6.5)
    ax_mem.set_title(f"Top-{top_k} Membership (blue = in top-{top_k})", fontsize=10)

    # Annotate column sums (how many families claim this head)
    col_sums = membership.sum(axis=0)
    for xi, s in enumerate(col_sums):
        if s >= 2:
            ax_mem.text(xi, -0.5, f"{int(s)}", ha="center", va="bottom",
                        fontsize=6, fontweight="bold", color="darkblue")

    # Score heatmap
    vmax = float(np.percentile(score_mat[score_mat > 0], 95)) if (score_mat > 0).any() else 1.0
    norm = TwoSlopeNorm(vmin=score_mat.min(), vcenter=0.0, vmax=vmax) \
        if score_mat.min() < 0 < vmax else None
    im = ax_score.imshow(score_mat, aspect="auto",
                         cmap="RdBu_r" if norm else "Blues",
                         norm=norm, vmax=vmax if norm is None else None,
                         interpolation="nearest")
    ax_score.set_yticks(range(len(families)))
    ax_score.set_yticklabels([FAMILY_DEFS[f]["display"] for f in families], fontsize=9)
    ax_score.set_xticks(range(n_heads_plot))
    ax_score.set_xticklabels(head_labels, rotation=90, fontsize=6.5)
    ax_score.set_title("Mean Δ Edit-Mass Scores across Families", fontsize=10)
    plt.colorbar(im, ax=ax_score, fraction=0.02, pad=0.02, label="Δ edit-mass")

    # Colour y-axis labels by family
    for yi, fam in enumerate(families):
        ax_score.get_yticklabels()[yi].set_color(FAMILY_DEFS[fam]["color"])
        ax_mem.get_yticklabels()[yi].set_color(FAMILY_DEFS[fam]["color"])

    fig.suptitle("Cross-Family Head Membership & Scores\n"
                 "(Union of focal top-K heads)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 04: UpSet-style intersection bars
# ---------------------------------------------------------------------------

def plot_upset(top_heads: dict, overlap_stats: dict, output_path: Path, top_k: int):
    families = [f for f in FOCAL_FAMILIES if f in top_heads]
    if len(families) < 2:
        return

    sets = {f: {(l, h) for l, h, _ in top_heads[f][:top_k]} for f in families}

    # Enumerate all non-empty subset intersections
    intersections = {}
    for r in range(1, len(families) + 1):
        for combo in combinations(families, r):
            inter = sets[combo[0]].copy()
            for f in combo[1:]:
                inter &= sets[f]
            # Only include heads exclusively in this intersection (not in a larger set)
            # i.e., in exactly this combination
            exclusive = inter.copy()
            for other_fam in families:
                if other_fam not in combo:
                    exclusive -= sets[other_fam]
            if exclusive:
                intersections[combo] = exclusive

    if not intersections:
        return

    sorted_intersections = sorted(intersections.items(), key=lambda x: len(x[1]), reverse=True)
    labels = ["+".join(FAMILY_DEFS[f]["display"].split()[0] for f in combo)
              for combo, _ in sorted_intersections]
    counts = [len(heads) for _, heads in sorted_intersections]
    colors = []
    for combo, _ in sorted_intersections:
        if len(combo) == 1:
            colors.append(FAMILY_DEFS[combo[0]]["color"])
        else:
            colors.append("#888888")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"# Heads (exclusive to intersection)", fontsize=10)
    ax.set_title(f"UpSet-style: Exclusive Intersections of Top-{top_k} Head Sets\n"
                 "(single-family = unique heads; multi-family = shared heads)",
                 fontsize=11)
    for xi, c in enumerate(counts):
        ax.text(xi, c + 0.1, str(c), ha="center", va="bottom", fontsize=9)

    legend_patches = [Patch(color=FAMILY_DEFS[f]["color"], label=FAMILY_DEFS[f]["display"])
                      for f in families]
    legend_patches.append(Patch(color="#888888", label="Shared (multi-family)"))
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 05: Within-family attribute comparison
# ---------------------------------------------------------------------------

def plot_within_family(
    attr_scores_raw: dict,
    top_heads: dict,
    output_path: Path,
    top_k: int,
    min_pairs: int,
):
    families = [f for f in FOCAL_FAMILIES if f in top_heads and f in attr_scores_raw]
    if not families:
        return

    n_rows = len(families)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), squeeze=False)

    for row, fam in enumerate(families):
        ax = axes[row][0]
        focal_heads = top_heads[fam][:top_k]
        head_labels = [f"L{l}H{h}" for l, h, _ in focal_heads]
        x = np.arange(len(head_labels))

        attr_data = attr_scores_raw[fam]
        color = FAMILY_DEFS[fam]["color"]

        # Filter to attributes with enough pairs
        valid_attrs = {}
        for attr, acc in attr_data.items():
            n_max = int(acc["n"].max())
            if n_max >= min_pairs:
                mean = (acc["sum"] / np.maximum(acc["n"], 1)).astype(np.float32)
                valid_attrs[attr] = mean

        if not valid_attrs:
            ax.text(0.5, 0.5, f"No attributes with ≥{min_pairs} pairs",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(FAMILY_DEFS[fam]["display"], color=color, fontweight="bold")
            continue

        # Use a colormap to distinguish attributes
        attr_list = sorted(valid_attrs.keys())
        cmap = plt.cm.get_cmap("tab10", len(attr_list))

        for ai, attr in enumerate(attr_list):
            mean_mat = valid_attrs[attr]
            vals = [mean_mat[l, h] for l, h, _ in focal_heads]
            ax.plot(x, vals, marker="o", linewidth=1.5, markersize=5,
                    color=cmap(ai), label=attr, alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Mean Δ edit-mass", fontsize=9)
        ax.set_title(
            f"{FAMILY_DEFS[fam]['display']} — Per-Attribute Profile over Top-{top_k} Heads",
            fontsize=10, color=color, fontweight="bold"
        )
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 06: Control comparison
# ---------------------------------------------------------------------------

def plot_control_comparison(
    top_heads: dict,
    mean_scores: dict,
    overlap_stats: dict,
    output_path: Path,
    top_k: int,
):
    if "control" not in mean_scores:
        return
    families = [f for f in FOCAL_FAMILIES if f in top_heads]
    if not families:
        return

    # For each family's top heads, compare focal score vs control score
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5), squeeze=False)

    for col, fam in enumerate(families):
        ax = axes[0][col]
        focal_heads = top_heads[fam][:top_k]
        focal_vals = [mean_scores[fam][l, h] for l, h, _ in focal_heads]
        ctrl_vals = [mean_scores["control"][l, h] for l, h, _ in focal_heads]
        labels = [f"L{l}H{h}" for l, h, _ in focal_heads]

        color = FAMILY_DEFS[fam]["color"]

        ax.scatter(ctrl_vals, focal_vals, c=color, alpha=0.8, s=50, edgecolors="white",
                   linewidth=0.5, zorder=3)
        for xi, (cv, fv, lab) in enumerate(zip(ctrl_vals, focal_vals, labels)):
            ax.annotate(lab, (cv, fv), fontsize=5.5, alpha=0.8,
                        xytext=(2, 2), textcoords="offset points")

        # Reference line y=x (equal focal and control)
        all_vals = focal_vals + ctrl_vals
        lo, hi = min(all_vals) - 0.002, max(all_vals) + 0.002
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y=x")
        ax.axhline(0, color="grey", linewidth=0.4)
        ax.axvline(0, color="grey", linewidth=0.4)

        ax.set_xlabel("Control Δ edit-mass", fontsize=9)
        ax.set_ylabel(f"{FAMILY_DEFS[fam]['display']} Δ edit-mass", fontsize=9)
        ax.set_title(f"{FAMILY_DEFS[fam]['display']} vs Control\n"
                     f"(Top-{top_k} {FAMILY_DEFS[fam]['display']} heads)",
                     fontsize=9.5, color=color, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Annotate quadrant counts
        above = sum(1 for fv, cv in zip(focal_vals, ctrl_vals) if fv > cv)
        ax.text(0.03, 0.97, f"{above}/{len(focal_vals)} above y=x\n(focal > control)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Focal vs Control Response for Each Family's Top Heads\n"
                 "(Points above y=x = head more specific to identity than surface change)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def make_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serialisable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Attention head localization for causal tracing"
    )
    parser.add_argument("--extraction_dir", required=True,
                        help="Directory with manifest.pt and shard files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top heads to select per family")
    parser.add_argument("--min_pairs", type=int, default=10,
                        help="Min pairs needed to include an attribute in within-family plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sharded store from {args.extraction_dir}...")
    store = ShardedExtractionStore(args.extraction_dir)
    n_layers = store.model_config["n_layers"]
    print(f"  Model: {n_layers} layers")

    # -----------------------------------------------------------------------
    # Step 1: Accumulate scores
    # -----------------------------------------------------------------------
    print("\nAccumulating head scores...")
    scores_raw, attr_scores_raw, n_layers_data, n_heads = accumulate_scores(
        store, args.top_k, args.min_pairs
    )
    print(f"  Resolved: {n_layers_data} layers, {n_heads} heads")

    if not scores_raw:
        print("ERROR: No attention data found. Was --extract_attention used during extraction?")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Derive summary statistics
    # -----------------------------------------------------------------------
    print("\nComputing summary statistics...")
    mean_scores = compute_mean_scores(scores_raw)
    std_scores = compute_std_scores(scores_raw)
    frac_positive = compute_frac_positive(scores_raw)

    # -----------------------------------------------------------------------
    # Step 3: Rank heads per family
    # -----------------------------------------------------------------------
    print(f"\nRanking top-{args.top_k} heads per family...")
    top_heads = {}
    for fam in FOCAL_FAMILIES + ["control"]:
        if fam not in mean_scores:
            print(f"  {fam}: no data")
            continue
        ranked = rank_heads(mean_scores[fam], args.top_k)
        top_heads[fam] = ranked
        print(f"  {fam}: {[head_key(l, h) for l, h, _ in ranked[:5]]} ...")

    # -----------------------------------------------------------------------
    # Step 4: Overlap analysis
    # -----------------------------------------------------------------------
    print("\nComputing overlap statistics...")
    overlap_stats = compute_overlap(top_heads)
    print(f"  Shared across all focal families: "
          f"{len(overlap_stats['multiway'].get('shared_all_focal', []))} heads")
    for fam in FOCAL_FAMILIES:
        key = f"unique_{fam}"
        n_unique = len(overlap_stats["multiway"].get(key, []))
        print(f"  Unique to {fam}: {n_unique}")
    print(f"  Nullish (control, not in any focal): "
          f"{len(overlap_stats['multiway'].get('nullish_control', []))} heads")

    # -----------------------------------------------------------------------
    # Step 5: Save artefacts
    # -----------------------------------------------------------------------
    print("\nSaving artefacts...")

    # Score arrays
    np.savez(
        output_dir / "head_scores.npz",
        **{f"{fam}_mean": mean_scores[fam] for fam in mean_scores},
        **{f"{fam}_std": std_scores[fam] for fam in std_scores},
        **{f"{fam}_frac_pos": frac_positive[fam] for fam in frac_positive},
    )
    print(f"  Saved: head_scores.npz")

    # Top heads JSON
    top_heads_json = {}
    for fam, heads in top_heads.items():
        top_heads_json[fam] = {
            "top_heads": [{"layer": l, "head": h, "score": round(s, 6)}
                          for l, h, s in heads],
            "n_pairs": int(scores_raw[fam]["n"].max()) if fam in scores_raw else 0,
        }
        # Per-attribute top heads
        if fam in attr_scores_raw:
            top_heads_json[fam]["per_attribute"] = {}
            attr_means = {}
            for attr, acc in attr_scores_raw[fam].items():
                n_max = int(acc["n"].max())
                if n_max < args.min_pairs:
                    continue
                m = (acc["sum"] / np.maximum(acc["n"], 1)).astype(np.float32)
                attr_means[attr] = m
                ranked_attr = rank_heads(m, args.top_k)
                top_heads_json[fam]["per_attribute"][attr] = {
                    "n_pairs": n_max,
                    "top_heads": [{"layer": l, "head": h, "score": round(s, 6)}
                                  for l, h, s in ranked_attr],
                }

    with open(output_dir / "top_heads.json", "w") as f:
        json.dump(make_serialisable(top_heads_json), f, indent=2)
    print(f"  Saved: top_heads.json")

    # Overlap JSON
    with open(output_dir / "overlap_stats.json", "w") as f:
        json.dump(make_serialisable(overlap_stats), f, indent=2)
    print(f"  Saved: overlap_stats.json")

    # -----------------------------------------------------------------------
    # Step 6: Plots
    # -----------------------------------------------------------------------
    if not HAS_PLT:
        print("\nSkipping plots (matplotlib not available)")
        print("\nDone.")
        return

    print("\nGenerating plots...")

    plot_score_heatmaps(mean_scores, plots_dir / "01_score_heatmaps.png")

    plot_top_heads_bars(
        top_heads, mean_scores, std_scores, frac_positive,
        plots_dir / "02_top_heads_bars.png", args.top_k
    )

    plot_overlap_matrix(
        top_heads, mean_scores,
        plots_dir / "03_overlap_matrix.png", args.top_k
    )

    plot_upset(
        top_heads, overlap_stats,
        plots_dir / "04_upset.png", args.top_k
    )

    plot_within_family(
        attr_scores_raw, top_heads,
        plots_dir / "05_within_family.png", args.top_k, args.min_pairs
    )

    plot_control_comparison(
        top_heads, mean_scores, overlap_stats,
        plots_dir / "06_control_comparison.png", args.top_k
    )

    print(f"\nAll outputs written to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
