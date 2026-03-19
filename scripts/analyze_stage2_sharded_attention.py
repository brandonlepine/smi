#!/usr/bin/env python3
"""
Stage 2 geometry + attention-aware candidate-head analysis for sharded extractions.

Expanded with:
  - UMAP/t-SNE of representation deltas colored by bias type
  - Representational Similarity Analysis (RSA) across bias types
  - PCA biplots of delta vectors
  - Local-to-global coupling scatter plots with regression lines
  - Layer×head interaction heatmaps for behavioral coupling
  - Answer flip transition analysis (which direction do flips go?)
  - Bootstrap effect-size panels with per-layer breakdown
  - Predicted-token change analysis per bias type
  - Head selectivity index (which heads are bias-type-specific?)
  - Direction alignment across bias types (shared vs type-specific directions)

Usage:
  python analyze_stage2_sharded_attention.py \
    --extraction_dir ./extractions_memsafe \
    --output_dir ./stage2_results_sharded
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

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
    from scipy.spatial.distance import pdist, squareform, cosine as cosine_dist_scipy
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


AIDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDXA = {0: "A", 1: "B", 2: "C", 3: "D"}

GROUP_COLORS = {
    "core_bias": "#d62728",
    "identity_bias": "#ff7f0e",
    "structural_context": "#2ca02c",
    "control": "#7f7f7f",
    "other": "#9467bd",
}

ATTR_CMAP = {
    "sex": "#e41a1c", "age": "#377eb8", "race_ethnicity": "#4daf4a",
    "pronoun": "#984ea3", "name": "#ff7f00",
    "gender_identity": "#a65628", "sexual_orientation": "#f781bf",
    "insurance_status": "#66c2a5", "housing_status": "#fc8d62",
    "occupation": "#8da0cb", "neutral_rework": "#b3b3b3",
    "irrelevant_surface": "#d9d9d9",
}


def behavioral_metrics(logits, gold):
    gi = AIDX[gold]
    pred_i = int(np.argmax(logits))
    gl = float(logits[gi])
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    comp = np.concatenate([logits[:gi], logits[gi + 1:]])
    return {
        "predicted": IDXA[pred_i],
        "correct": IDXA[pred_i] == gold,
        "gold_logit": gl,
        "gold_prob": float(probs[gi]),
        "margin": float(gl - np.max(comp)),
    }


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine_coherence(deltas: np.ndarray) -> dict:
    n = deltas.shape[0]
    if n < 2:
        return {"mean_cos": 0, "median_cos": 0, "std_cos": 0, "n_pairs": 0}

    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = deltas / norms
    cos_matrix = normed @ normed.T
    triu_idx = np.triu_indices(n, k=1)
    cos_vals = cos_matrix[triu_idx]

    return {
        "mean_cos": float(np.mean(cos_vals)),
        "median_cos": float(np.median(cos_vals)),
        "std_cos": float(np.std(cos_vals)),
        "n_pairs": len(cos_vals),
    }


def dim_direction(deltas: np.ndarray) -> np.ndarray:
    mean_delta = np.mean(deltas, axis=0)
    norm = np.linalg.norm(mean_delta)
    if norm < 1e-10:
        return mean_delta
    return mean_delta / norm


def pca_analysis(deltas: np.ndarray, n_components: int = 10) -> dict:
    if deltas.shape[0] < 3:
        return {"top1_var": 0, "top5_var": 0, "explained": [], "pca": None, "transformed": None}
    n_comp = min(n_components, deltas.shape[0] - 1, deltas.shape[1])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(deltas)
    evr = pca.explained_variance_ratio_
    return {
        "top1_var": float(evr[0]) if len(evr) > 0 else 0,
        "top5_var": float(np.sum(evr[:5])) if len(evr) >= 5 else float(np.sum(evr)),
        "explained": evr.tolist(),
        "pca": pca,
        "transformed": transformed,
    }


def loo_probe_accuracy(class0_deltas: np.ndarray, class1_deltas: np.ndarray, pca_dims: int = 50) -> dict:
    n0 = class0_deltas.shape[0]
    n1 = class1_deltas.shape[0]
    if n0 < 3 or n1 < 3:
        return {"accuracy": 0, "mean_prob": 0, "n": 0}

    X = np.vstack([class0_deltas, class1_deltas])
    y = np.array([0] * n0 + [1] * n1)

    n_comp = min(pca_dims, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    loo = LeaveOneOut()
    correct = 0
    probs = []
    total = 0

    for train_idx, test_idx in loo.split(X_scaled):
        clf = LogisticRegression(max_iter=1000, C=0.01, solver="lbfgs")
        try:
            clf.fit(X_scaled[train_idx], y[train_idx])
            pred = clf.predict(X_scaled[test_idx])[0]
            prob = clf.predict_proba(X_scaled[test_idx])[0]
            true_class = y[test_idx[0]]
            correct += int(pred == true_class)
            probs.append(prob[true_class])
            total += 1
        except Exception:
            continue

    return {
        "accuracy": correct / total if total > 0 else 0,
        "mean_prob": float(np.mean(probs)) if probs else 0,
        "n": total,
        "pca_dims_used": n_comp,
    }


def dim_threshold_accuracy(deltas: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return 0.0
    class0 = deltas[labels == 0]
    class1 = deltas[labels == 1]
    dim_dir = np.mean(class1, axis=0) - np.mean(class0, axis=0)
    norm = np.linalg.norm(dim_dir)
    if norm < 1e-10:
        return 0.5
    dim_dir = dim_dir / norm

    projections = deltas @ dim_dir
    threshold = np.mean(projections)
    preds = (projections > threshold).astype(int)
    return float(np.mean(preds == labels))


def bootstrap_ci(values, n_boot=2000, ci=0.95, stat_fn=np.mean):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_stats = np.array([
        stat_fn(values[np.random.randint(0, n, size=n)])
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return float(stat_fn(values)), lo, hi


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
    raw = (
        pm.get("intervention_type")
        or pm.get("attribute_type")
        or pm.get("attribute")
        or pm.get("dimension")
    )
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
    return (
        safe_str(pm.get("label"))
        or safe_str(pm.get("attribute_value_counterfactual"))
        or safe_str(pm.get("counterfactual_value"))
        or safe_str(pm.get("value"))
    )


def normalize_metadata(pm: dict) -> dict:
    intervention_type = normalize_intervention_type(pm)
    norm_group = normalize_analysis_group(pm, intervention_type)
    norm_ctrl = normalize_control_subtype(pm, intervention_type)
    norm_locality = normalize_edit_locality(pm)
    norm_label = normalize_label(pm)

    out = dict(pm)
    out["normalized_attribute"] = intervention_type
    out["normalized_group"] = norm_group
    out["normalized_control_subtype"] = norm_ctrl
    out["normalized_edit_locality"] = norm_locality
    out["normalized_label"] = norm_label
    return out


def group_alias(g: str) -> str:
    return {
        "core_bias": "Core bias",
        "identity_bias": "Identity bias",
        "structural_context": "Structural context",
        "control": "Control",
        "other": "Other",
    }.get(g, g)


def pick_top_label_pairs(rows, min_per_side=3, max_pairs_per_attr=2):
    by_attr = defaultdict(list)
    for r in rows:
        attr = r["normalized_attribute"]
        label = r.get("normalized_label")
        if label is not None:
            by_attr[attr].append(label)

    pairs = []
    for attr, labels in by_attr.items():
        counts = Counter(labels)
        eligible = [(lab, ct) for lab, ct in counts.most_common() if ct >= min_per_side]
        if len(eligible) < 2:
            continue

        pair_count = 0
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                pairs.append((attr, eligible[i][0], eligible[j][0]))
                pair_count += 1
                if pair_count >= max_pairs_per_attr:
                    break
            if pair_count >= max_pairs_per_attr:
                break
    return pairs


# ---------------------------------------------------------------------------
# Visualization: UMAP / t-SNE of delta vectors
# ---------------------------------------------------------------------------

def plot_embedding_scatter(deltas: np.ndarray, labels: list[str], colors: list[str],
                           title: str, outpath: Path, method: str = "pca"):
    """2D scatter of deltas colored by group/attribute."""
    if not HAS_PLT:
        return

    if deltas.shape[0] < 5:
        return

    if method == "umap" and HAS_UMAP:
        n_neighbors = min(15, deltas.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(deltas)
        axis_label = "UMAP"
    elif method == "tsne" and HAS_TSNE:
        perp = min(30, deltas.shape[0] - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(deltas)
        axis_label = "t-SNE"
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(deltas)
        axis_label = "PCA"

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            continue
        c = colors[idx[0]]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=c, label=lab, alpha=0.6, s=20, edgecolors="none")

    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: PCA biplot
# ---------------------------------------------------------------------------

def plot_pca_biplot(deltas: np.ndarray, labels: list[str], colors: list[str],
                    title: str, outpath: Path):
    """PCA biplot showing first 2 PCs with explained variance."""
    if not HAS_PLT or deltas.shape[0] < 5:
        return

    pca = PCA(n_components=min(10, deltas.shape[0] - 1, deltas.shape[1]))
    transformed = pca.fit_transform(deltas)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of PC1 vs PC2
    ax = axes[0]
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        c = colors[idx[0]]
        ax.scatter(transformed[idx, 0], transformed[idx, 1], c=c, label=lab, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)" if len(evr) > 1 else "PC2")
    ax.set_title(f"{title}\nPC1 vs PC2")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.2)

    # Scree plot
    ax2 = axes[1]
    n_show = min(10, len(evr))
    ax2.bar(range(n_show), evr[:n_show], color="#4c72b0", alpha=0.7)
    ax2.plot(range(n_show), np.cumsum(evr[:n_show]), "ro-", markersize=5)
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("Scree Plot (cumulative = red)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: RSA (Representational Similarity Analysis)
# ---------------------------------------------------------------------------

def compute_rdm(deltas: np.ndarray) -> np.ndarray:
    """Compute representational dissimilarity matrix using cosine distance."""
    if deltas.shape[0] < 2:
        return np.zeros((1, 1))
    dists = pdist(deltas, metric="cosine")
    return squareform(dists)


def plot_rsa_across_groups(group_deltas: dict, layer_range_name: str, outpath: Path):
    """Compare RDMs across bias groups using RSA (correlation of flattened RDMs)."""
    if not HAS_PLT or not HAS_SCIPY:
        return

    groups = [g for g in group_deltas if len(group_deltas[g]) >= 5]
    if len(groups) < 2:
        return

    # Build per-group RDMs (subsample to common size for comparison)
    min_n = min(len(group_deltas[g]) for g in groups)
    min_n = min(min_n, 50)  # cap for computational tractability

    rdms = {}
    for g in groups:
        idx = np.random.RandomState(42).choice(len(group_deltas[g]), size=min_n, replace=False)
        rdms[g] = compute_rdm(np.stack([group_deltas[g][i] for i in idx]))

    # RSA: Spearman correlation between flattened upper triangles of RDMs
    triu_idx = np.triu_indices(min_n, k=1)
    n_groups = len(groups)
    rsa_matrix = np.zeros((n_groups, n_groups))
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i == j:
                rsa_matrix[i, j] = 1.0
            elif j > i:
                r, _ = sp_stats.spearmanr(rdms[g1][triu_idx], rdms[g2][triu_idx])
                rsa_matrix[i, j] = r
                rsa_matrix[j, i] = r

    fig, axes = plt.subplots(1, n_groups + 1, figsize=(5 * (n_groups + 1), 4), squeeze=False)

    # Individual RDMs
    for i, g in enumerate(groups):
        ax = axes[0, i]
        im = ax.imshow(rdms[g], cmap="viridis", vmin=0, vmax=2)
        ax.set_title(f"{group_alias(g)}\n(n={min_n})", fontsize=10)
        ax.set_xlabel("Item")
        ax.set_ylabel("Item")

    # RSA matrix
    ax = axes[0, n_groups]
    im2 = ax.imshow(rsa_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([group_alias(g) for g in groups], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([group_alias(g) for g in groups], fontsize=9)
    ax.set_title(f"RSA Matrix\n({layer_range_name} layers)")
    for i in range(n_groups):
        for j in range(n_groups):
            ax.text(j, i, f"{rsa_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(rsa_matrix[i, j]) > 0.5 else "black")
    fig.colorbar(im2, ax=ax, shrink=0.6, label="Spearman r")

    fig.suptitle(f"Representational Similarity Analysis — {layer_range_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: Coupling scatter plots
# ---------------------------------------------------------------------------

def plot_coupling_scatters(coupling_rows: list, late_layers: list, outdir: Path):
    """Scatter plots showing local-to-global coupling with regression lines."""
    if not HAS_PLT or not coupling_rows:
        return

    edit_mags, final_mags, abs_dlogits, flips = [], [], [], []
    groups = []
    for r in coupling_rows:
        edit_d = np.mean([r["_edit_delta"][l] for l in late_layers], axis=0)
        final_d = np.mean([r["_final_delta"][l] for l in late_layers], axis=0)
        edit_mags.append(float(np.linalg.norm(edit_d)))
        final_mags.append(float(np.linalg.norm(final_d)))
        abs_dlogits.append(r["abs_delta_gold_logit"])
        flips.append(int(r["answer_flip"]))
        groups.append(r["normalized_group"])

    edit_mags = np.array(edit_mags)
    final_mags = np.array(final_mags)
    abs_dlogits = np.array(abs_dlogits)
    flips = np.array(flips)
    colors = [GROUP_COLORS.get(g, "#999") for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Edit mag vs |Δ logit|
    ax = axes[0, 0]
    ax.scatter(edit_mags, abs_dlogits, c=colors, alpha=0.5, s=15, edgecolors="none")
    if HAS_SCIPY and len(edit_mags) > 3:
        r, p = sp_stats.spearmanr(edit_mags, abs_dlogits)
        ax.set_title(f"Edit-Token Δ Magnitude vs |Δ Gold Logit|\nSpearman r={r:.3f}, p={p:.4f}")
        # Regression line
        z = np.polyfit(edit_mags, abs_dlogits, 1)
        x_line = np.linspace(edit_mags.min(), edit_mags.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("Edit-Token Δ Magnitude (late layers)")
    ax.set_ylabel("|Δ Gold Logit|")
    ax.grid(True, alpha=0.3)

    # Final mag vs |Δ logit|
    ax = axes[0, 1]
    ax.scatter(final_mags, abs_dlogits, c=colors, alpha=0.5, s=15, edgecolors="none")
    if HAS_SCIPY and len(final_mags) > 3:
        r, p = sp_stats.spearmanr(final_mags, abs_dlogits)
        ax.set_title(f"Final-Token Δ Magnitude vs |Δ Gold Logit|\nSpearman r={r:.3f}, p={p:.4f}")
        z = np.polyfit(final_mags, abs_dlogits, 1)
        x_line = np.linspace(final_mags.min(), final_mags.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("Final-Token Δ Magnitude (late layers)")
    ax.set_ylabel("|Δ Gold Logit|")
    ax.grid(True, alpha=0.3)

    # Edit mag vs Final mag
    ax = axes[1, 0]
    ax.scatter(edit_mags, final_mags, c=colors, alpha=0.5, s=15, edgecolors="none")
    if HAS_SCIPY and len(edit_mags) > 3:
        r, p = sp_stats.spearmanr(edit_mags, final_mags)
        ax.set_title(f"Edit-Token vs Final-Token Δ Magnitude\nSpearman r={r:.3f}, p={p:.4f}")
        z = np.polyfit(edit_mags, final_mags, 1)
        x_line = np.linspace(edit_mags.min(), edit_mags.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("Edit-Token Δ Magnitude")
    ax.set_ylabel("Final-Token Δ Magnitude")
    ax.grid(True, alpha=0.3)

    # Flip probability by edit magnitude (binned)
    ax = axes[1, 1]
    if np.sum(flips) > 0:
        n_bins = min(10, len(edit_mags) // 5)
        if n_bins >= 2:
            bin_edges = np.percentile(edit_mags, np.linspace(0, 100, n_bins + 1))
            bin_centers = []
            flip_rates = []
            for i in range(n_bins):
                mask = (edit_mags >= bin_edges[i]) & (edit_mags < bin_edges[i + 1] + 1e-10)
                if mask.sum() > 0:
                    bin_centers.append(np.mean(edit_mags[mask]))
                    flip_rates.append(np.mean(flips[mask]))
            ax.plot(bin_centers, flip_rates, "o-", color="#d62728", linewidth=2, markersize=6)
            ax.set_xlabel("Edit-Token Δ Magnitude (binned)")
            ax.set_ylabel("Flip Rate")
            ax.set_title("Answer Flip Rate by Edit Magnitude")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

    # Legend
    legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=group_alias(g))
                       for g in GROUP_COLORS if g in set(groups)]
    axes[0, 0].legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(outdir / "coupling_scatters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'coupling_scatters.png'}")


# ---------------------------------------------------------------------------
# Visualization: Direction alignment across bias types
# ---------------------------------------------------------------------------

def plot_direction_alignment(rows: list, n_layers: int, outdir: Path):
    """How aligned are the mean delta directions across different bias types?"""
    if not HAS_PLT:
        return

    focal_groups = ["core_bias", "identity_bias", "structural_context", "control"]
    mid = n_layers // 2
    late_layers = list(range(mid, n_layers))

    # Also compute per-attribute directions
    attr_directions = {}
    group_directions = {}

    for g in focal_groups:
        grp = [r for r in rows if r["normalized_group"] == g and r.get("has_edit")]
        if len(grp) < 3:
            continue
        deltas = np.stack([np.mean([r["_edit_delta"][l] for l in late_layers], axis=0) for r in grp])
        group_directions[g] = dim_direction(deltas)

    for r in rows:
        if not r.get("has_edit"):
            continue
        attr = r["normalized_attribute"]
        if attr not in attr_directions:
            attr_directions[attr] = []
        attr_directions[attr].append(np.mean([r["_edit_delta"][l] for l in late_layers], axis=0))

    attr_dirs = {}
    for attr, deltas_list in attr_directions.items():
        if len(deltas_list) >= 3:
            attr_dirs[attr] = dim_direction(np.stack(deltas_list))

    if len(attr_dirs) < 2:
        return

    # Cosine similarity matrix between attribute directions
    attrs = sorted(attr_dirs.keys())
    n = len(attrs)
    cos_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = cosine_sim(attr_dirs[attrs[i]], attr_dirs[attrs[j]])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Attribute direction alignment
    ax = axes[0]
    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(attrs, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(attrs, fontsize=8)
    ax.set_title("Cosine Similarity of Mean\nEdit-Delta Directions by Attribute")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Cosine Similarity")

    # Group direction alignment (if enough groups)
    if len(group_directions) >= 2:
        ax2 = axes[1]
        g_names = sorted(group_directions.keys())
        ng = len(g_names)
        g_cos = np.zeros((ng, ng))
        for i in range(ng):
            for j in range(ng):
                g_cos[i, j] = cosine_sim(group_directions[g_names[i]], group_directions[g_names[j]])
        im2 = ax2.imshow(g_cos, cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(ng))
        ax2.set_xticklabels([group_alias(g) for g in g_names], rotation=45, ha="right", fontsize=10)
        ax2.set_yticks(range(ng))
        ax2.set_yticklabels([group_alias(g) for g in g_names], fontsize=10)
        ax2.set_title("Cosine Similarity of Mean\nEdit-Delta Directions by Group")
        for i in range(ng):
            for j in range(ng):
                ax2.text(j, i, f"{g_cos[i,j]:.2f}", ha="center", va="center", fontsize=11,
                         color="white" if abs(g_cos[i, j]) > 0.5 else "black")
        fig.colorbar(im2, ax=ax2, shrink=0.6, label="Cosine Similarity")

    fig.tight_layout()
    fig.savefig(outdir / "direction_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'direction_alignment.png'}")


# ---------------------------------------------------------------------------
# Visualization: Per-layer probe accuracy
# ---------------------------------------------------------------------------

def plot_perlayer_probe_accuracy(rows: list, n_layers: int, outdir: Path):
    """LOO probe accuracy at each individual layer for focal vs control."""
    if not HAS_PLT:
        return

    focal_groups = ["core_bias", "identity_bias", "structural_context"]
    ctrl_rows = [r for r in rows if r["normalized_group"] == "control"
                 and r["normalized_control_subtype"] == "irrelevant_surface"
                 and r.get("has_edit")
                 and r["normalized_edit_locality"] in {"minimal", "single"}]

    if len(ctrl_rows) < 3:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for grp_name in focal_groups:
        focal_rows = [r for r in rows if r["normalized_group"] == grp_name
                      and r.get("has_edit")
                      and r["normalized_edit_locality"] in {"minimal", "single"}]
        if len(focal_rows) < 3:
            continue

        accuracies = []
        for l in range(n_layers):
            f_d = np.stack([r["_edit_delta"][l] for r in focal_rows])
            c_d = np.stack([r["_edit_delta"][l] for r in ctrl_rows])
            acc = dim_threshold_accuracy(
                np.vstack([c_d, f_d]),
                np.array([0] * len(c_d) + [1] * len(f_d))
            )
            accuracies.append(acc)

        color = GROUP_COLORS.get(grp_name, "#333")
        ax.plot(range(n_layers), accuracies, label=f"{group_alias(grp_name)} vs Control",
                color=color, linewidth=2)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("DIM Threshold Accuracy")
    ax.set_title("Per-Layer Discriminability: Focal vs Control (Edit-Token Deltas)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)
    fig.tight_layout()
    fig.savefig(outdir / "perlayer_probe_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'perlayer_probe_accuracy.png'}")


# ---------------------------------------------------------------------------
# Visualization: Answer flip transition analysis
# ---------------------------------------------------------------------------

def plot_flip_transitions(rows: list, outdir: Path):
    """What happens when answers flip? Show transition patterns per group."""
    if not HAS_PLT:
        return

    focal_groups = ["core_bias", "identity_bias", "structural_context", "control"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for gi, g in enumerate(focal_groups):
        ax = axes[gi // 2, gi % 2]
        flipped = [r for r in rows if r["normalized_group"] == g and r.get("answer_flip")]

        if not flipped:
            ax.set_title(f"{group_alias(g)}: No flips")
            ax.axis("off")
            continue

        # Count transitions
        transitions = Counter()
        for r in flipped:
            orig_correct = r.get("orig_correct", False)
            cf_correct = r.get("cf_correct", False)
            if orig_correct and not cf_correct:
                transitions["Correct→Wrong"] += 1
            elif not orig_correct and cf_correct:
                transitions["Wrong→Correct"] += 1
            elif orig_correct and cf_correct:
                transitions["Correct→Correct (diff)"] += 1
            else:
                transitions["Wrong→Wrong (diff)"] += 1

        labels = list(transitions.keys())
        values = list(transitions.values())
        colors_bar = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"][:len(labels)]

        ax.barh(labels, values, color=colors_bar)
        ax.set_xlabel("Count")
        ax.set_title(f"{group_alias(g)} Flip Transitions (n={len(flipped)})")

        # Annotate with percentages
        total = sum(values)
        for i, (l, v) in enumerate(zip(labels, values)):
            ax.text(v + 0.5, i, f"{v/total:.0%}", va="center", fontsize=10)

    fig.suptitle("Answer Flip Transition Types", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "flip_transitions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'flip_transitions.png'}")


# ---------------------------------------------------------------------------
# Visualization: Per-layer bootstrap effect sizes
# ---------------------------------------------------------------------------

def plot_perlayer_bootstrap(rows: list, n_layers: int, outdir: Path):
    """Per-layer representation shift with bootstrap CIs by group."""
    if not HAS_PLT:
        return

    focal_groups = ["core_bias", "identity_bias", "structural_context", "control"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for g in focal_groups:
        grp = [r for r in rows if r["normalized_group"] == g]
        if len(grp) < 5:
            continue

        means = []
        los = []
        his = []
        for l in range(n_layers):
            vals = [r.get(f"norm_euclid_L{l}", 0) for r in grp]
            mean, lo, hi = bootstrap_ci(vals, n_boot=500)
            means.append(mean)
            los.append(lo)
            his.append(hi)

        color = GROUP_COLORS.get(g, "#333")
        ax.plot(range(n_layers), means, label=f"{group_alias(g)} (n={len(grp)})",
                color=color, linewidth=2)
        ax.fill_between(range(n_layers), los, his, alpha=0.15, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Euclidean Distance")
    ax.set_title("Per-Layer Representation Shift (Final Token) with 95% Bootstrap CIs")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "perlayer_bootstrap_repr.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'perlayer_bootstrap_repr.png'}")


# ---------------------------------------------------------------------------
# Head selectivity index
# ---------------------------------------------------------------------------

def compute_head_selectivity(rows: list, store, n_layers: int, outdir: Path):
    """Which heads are selective for specific bias types vs being general-purpose?"""
    if not HAS_PANDAS or not HAS_PLT:
        return

    def _normalize_attn(attn_dict):
        if attn_dict is None or not isinstance(attn_dict, dict):
            return None
        if any(k in attn_dict for k in ("mass_to_edit_region", "topk_source_positions")):
            return attn_dict
        out = {}
        if "edit_mass" in attn_dict:
            out["mass_to_edit_region"] = attn_dict["edit_mass"]
        if "entropy" in attn_dict:
            out["entropy"] = attn_dict["entropy"]
        return out if out else None

    # Collect per-group mean |delta| for edit_mass at each (layer, head)
    group_head_deltas = defaultdict(list)

    for row in rows:
        if not row.get("has_attention_final"):
            continue
        g = row["normalized_group"]
        od = store.get_original(row["question_id"])
        cd = store.get_cf(row["pair_key"])
        if od is None or cd is None:
            continue

        a_o = _normalize_attn(od.get("attention_summary", {}).get("final_token"))
        a_c = _normalize_attn(cd.get("attention_summary", {}).get("final_token"))
        if not a_o or not a_c:
            continue
        if "mass_to_edit_region" not in a_o or "mass_to_edit_region" not in a_c:
            continue

        o_em = np.asarray(a_o["mass_to_edit_region"], dtype=np.float32)
        c_em = np.asarray(a_c["mass_to_edit_region"], dtype=np.float32)
        delta = np.abs(c_em - o_em)  # (layers, heads)
        group_head_deltas[g].append(delta)

    if not group_head_deltas:
        return

    groups = sorted(group_head_deltas.keys())
    n_heads = group_head_deltas[groups[0]][0].shape[1]

    # Mean per-group abs delta: (layers, heads)
    group_means = {}
    for g in groups:
        group_means[g] = np.mean(np.stack(group_head_deltas[g]), axis=0)

    # Selectivity: for each (layer, head), how much variance is there across groups?
    # High variance = selective, low variance = general-purpose
    all_means = np.stack([group_means[g] for g in groups])  # (n_groups, layers, heads)
    selectivity = np.std(all_means, axis=0) / (np.mean(all_means, axis=0) + 1e-10)  # CV-like

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Selectivity heatmap
    im = axes[0].imshow(selectivity, aspect="auto", cmap="magma")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Head Selectivity Index\n(high = bias-type-specific)")
    fig.colorbar(im, ax=axes[0], shrink=0.6, label="Selectivity (CV)")

    # Top selective heads table
    flat_idx = np.argsort(selectivity.ravel())[::-1][:20]
    top_layers = flat_idx // n_heads
    top_heads = flat_idx % n_heads
    top_scores = selectivity.ravel()[flat_idx]

    table_data = []
    for l, h, s in zip(top_layers, top_heads, top_scores):
        row_info = {"layer": int(l), "head": int(h), "selectivity": float(s)}
        for g in groups:
            row_info[f"{g}_mean_delta"] = float(group_means[g][l, h])
        table_data.append(row_info)

    # Grouped bar for top 10
    top10 = table_data[:10]
    x = np.arange(len(top10))
    width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        vals = [t[f"{g}_mean_delta"] for t in top10]
        color = GROUP_COLORS.get(g, "#999")
        axes[1].bar(x + gi * width, vals, width, label=group_alias(g), color=color, alpha=0.8)

    axes[1].set_xticks(x + width * len(groups) / 2)
    axes[1].set_xticklabels([f"L{t['layer']}H{t['head']}" for t in top10], rotation=45, fontsize=8)
    axes[1].set_ylabel("Mean |Δ Edit Mass|")
    axes[1].set_title("Top 10 Most Selective Heads")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outdir / "head_selectivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir / 'head_selectivity.png'}")

    if table_data:
        pd.DataFrame(table_data).to_csv(outdir / "head_selectivity_top20.csv", index=False)
        print(f"Saved {outdir / 'head_selectivity_top20.csv'}")


# ---------------------------------------------------------------------------
# N-adjusted comparisons (mirrors Stage 1)
# ---------------------------------------------------------------------------

def cohens_d(group_a, group_b):
    a, b = np.asarray(group_a, dtype=np.float64), np.asarray(group_b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def print_n_adjusted_geometry(rows: list, n_layers: int, outdir: Path):
    """N-adjusted geometry comparison: subsample to min group size, report Cohen's d on coherence."""
    groups = ["core_bias", "identity_bias", "structural_context", "control"]
    group_rows = {}
    for g in groups:
        grp = [r for r in rows if r["normalized_group"] == g and r.get("has_edit")
               and r["normalized_edit_locality"] in {"minimal", "single"}]
        if len(grp) >= 5:
            group_rows[g] = grp

    if len(group_rows) < 2:
        return

    min_n = min(len(v) for v in group_rows.values())
    print(f"\n{'='*70}")
    print(f"N-ADJUSTED GEOMETRY COMPARISON (subsampled to n={min_n})")
    print(f"{'='*70}")

    mid = n_layers // 2
    late_layers = list(range(mid, n_layers))
    rng = np.random.RandomState(42)

    ctrl_key = "control" if "control" in group_rows else None

    results = []
    for g in [k for k in group_rows if k != "control"]:
        idx = rng.choice(len(group_rows[g]), size=min_n, replace=False)
        sub = [group_rows[g][i] for i in idx]
        deltas = np.stack([np.mean([r["_edit_delta"][l] for l in late_layers], axis=0) for r in sub])
        mags = np.linalg.norm(deltas, axis=1)

        print(f"\n  {group_alias(g)} (n={min_n}):")
        mean_mag, lo, hi = bootstrap_ci(mags)
        print(f"    Delta magnitude: {mean_mag:.4f} [{lo:.4f}, {hi:.4f}]")

        if ctrl_key and ctrl_key in group_rows:
            ctrl_idx = rng.choice(len(group_rows[ctrl_key]), size=min_n, replace=False)
            ctrl_sub = [group_rows[ctrl_key][i] for i in ctrl_idx]
            ctrl_deltas = np.stack([np.mean([r["_edit_delta"][l] for l in late_layers], axis=0) for r in ctrl_sub])
            ctrl_mags = np.linalg.norm(ctrl_deltas, axis=1)
            d = cohens_d(mags, ctrl_mags)
            print(f"    vs Control: Cohen's d={d:.3f}")
            results.append({"group": g, "metric": "delta_magnitude_late", "cohens_d": d, "n": min_n})

    if HAS_PANDAS and results:
        pd.DataFrame(results).to_csv(outdir / "n_adjusted_geometry.csv", index=False)


# ---------------------------------------------------------------------------
# Correctness-aware flip stats
# ---------------------------------------------------------------------------

def print_correctness_flip_stats(rows: list, outdir: Path):
    """Correct→Wrong, Wrong→Correct breakdown per group and attribute."""
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
        c2w = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and not r["cf_correct"])
        w2c = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and r["cf_correct"])
        c2c = sum(1 for r in grp if r["answer_flip"] and r["orig_correct"] and r["cf_correct"])
        w2w = sum(1 for r in grp if r["answer_flip"] and not r["orig_correct"] and not r["cf_correct"])

        stat = {"group": g, "n": n, "correct_to_wrong": c2w, "wrong_to_correct": w2c,
                "correct_to_correct_diff": c2c, "wrong_to_wrong_diff": w2w,
                "harm_rate": c2w / max(n, 1), "benefit_rate": w2c / max(n, 1),
                "net_harm": (c2w - w2c) / max(n, 1)}
        all_stats.append(stat)

        print(f"\n  {group_alias(g)} (n={n}):")
        print(f"    Correct → Wrong (HARMFUL): {c2w} ({c2w/max(n,1):.1%})")
        print(f"    Wrong → Correct (BENEFICIAL): {w2c} ({w2c/max(n,1):.1%})")
        print(f"    Net harm rate: {stat['net_harm']:.3f}")

    if HAS_PANDAS and all_stats:
        pd.DataFrame(all_stats).to_csv(outdir / "correctness_flip_stats.csv", index=False)


# ---------------------------------------------------------------------------
# Token-count adjusted coupling
# ---------------------------------------------------------------------------

def print_token_adjusted_coupling(rows: list, n_layers: int):
    """Report coupling correlations controlling for token_edit_ratio."""
    if not HAS_SCIPY:
        return

    coupling = [r for r in rows if r.get("has_edit")
                and r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                and r["normalized_edit_locality"] in {"minimal", "single"}]
    if len(coupling) < 10:
        return

    mid = n_layers // 2
    late_layers = list(range(mid, n_layers))

    edit_mags = np.array([float(np.linalg.norm(np.mean([r["_edit_delta"][l] for l in late_layers], axis=0)))
                          for r in coupling])
    abs_dlogits = np.array([r["abs_delta_gold_logit"] for r in coupling])
    ter = np.array([r.get("token_edit_ratio", 0) for r in coupling])

    print(f"\n{'='*70}")
    print("TOKEN-COUNT ADJUSTED COUPLING")
    print(f"{'='*70}")

    raw_r, raw_p = sp_stats.spearmanr(edit_mags, abs_dlogits)
    print(f"  Raw: edit_mag → |Δ logit|: r={raw_r:.3f}, p={raw_p:.4f}")

    # Partial correlation controlling for token_edit_ratio
    rxy, _ = sp_stats.spearmanr(edit_mags, abs_dlogits)
    rxz, _ = sp_stats.spearmanr(edit_mags, ter)
    ryz, _ = sp_stats.spearmanr(abs_dlogits, ter)
    denom = np.sqrt(max(1 - rxz ** 2, 1e-10)) * np.sqrt(max(1 - ryz ** 2, 1e-10))
    r_partial = (rxy - rxz * ryz) / denom
    print(f"  Partial (controlling token_edit_ratio): r={r_partial:.3f}")
    print(f"  token_edit_ratio → edit_mag: r={rxz:.3f}")
    print(f"  token_edit_ratio → |Δ logit|: r={ryz:.3f}")


# ---------------------------------------------------------------------------
# Within-group CF comparison (geometry-level)
# ---------------------------------------------------------------------------

def plot_within_group_geometry(rows: list, store, n_layers: int, outdir: Path):
    """Compare deltas between different CF labels within the same attribute (PCA space)."""
    if not HAS_PLT:
        return

    mid = n_layers // 2
    late_layers = list(range(mid, n_layers))

    by_attr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r.get("has_edit"):
            continue
        attr = r["normalized_attribute"]
        label = r.get("normalized_label")
        if label is not None:
            by_attr[attr][label].append(r)

    plot_dir = outdir / "within_group_geometry"
    plot_dir.mkdir(exist_ok=True)

    PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb"]

    for attr, label_dict in by_attr.items():
        eligible = {lab: rws for lab, rws in label_dict.items() if len(rws) >= 3}
        if len(eligible) < 2:
            continue

        labels_sorted = sorted(eligible.keys(), key=lambda l: -len(eligible[l]))[:6]

        all_deltas = []
        all_labels = []
        all_colors = []
        for i, lab in enumerate(labels_sorted):
            for r in eligible[lab]:
                all_deltas.append(np.mean([r["_edit_delta"][l] for l in late_layers], axis=0))
                all_labels.append(lab)
                all_colors.append(PALETTE[i % len(PALETTE)])

        if len(all_deltas) < 5:
            continue

        deltas_arr = np.stack(all_deltas)
        pca = PCA(n_components=min(2, deltas_arr.shape[0] - 1, deltas_arr.shape[1]))
        coords = pca.fit_transform(deltas_arr)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA scatter
        ax = axes[0]
        for i, lab in enumerate(labels_sorted):
            mask = [l == lab for l in all_labels]
            idx = [j for j, m in enumerate(mask) if m]
            if idx:
                ax.scatter(coords[idx, 0], coords[idx, 1], c=PALETTE[i % len(PALETTE)],
                           label=f"{lab} (n={len(idx)})", alpha=0.6, s=25, edgecolors="none")
        evr = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})" if len(evr) > 0 else "PC1")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})" if len(evr) > 1 else "PC2")
        ax.set_title(f"{attr}: Edit-Delta PCA by Label")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Pairwise cosine similarity of mean directions
        ax2 = axes[1]
        n_lab = len(labels_sorted)
        cos_mat = np.zeros((n_lab, n_lab))
        label_means = {}
        for lab in labels_sorted:
            lab_deltas = [all_deltas[j] for j, l in enumerate(all_labels) if l == lab]
            label_means[lab] = np.mean(np.stack(lab_deltas), axis=0)
        for i in range(n_lab):
            for j in range(n_lab):
                cos_mat[i, j] = cosine_sim(label_means[labels_sorted[i]], label_means[labels_sorted[j]])
        im = ax2.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(n_lab))
        ax2.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(range(n_lab))
        ax2.set_yticklabels(labels_sorted, fontsize=9)
        for i in range(n_lab):
            for j in range(n_lab):
                ax2.text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
        ax2.set_title("Cosine Similarity of\nMean Edit-Delta Directions")
        fig.colorbar(im, ax=ax2, shrink=0.6)

        fig.suptitle(f"Within-Attribute Geometry: {attr}", fontsize=12, y=1.02)
        fig.tight_layout()
        safe_attr = attr.replace("/", "_").replace(" ", "_")
        fig.savefig(plot_dir / f"{safe_attr}_geometry.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved within-group geometry plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(extraction_dir: str, output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    store = ShardedExtractionStore(extraction_dir)
    meta = list(store.iter_pair_metadata())
    n_layers = store.model_config["n_layers"]

    print(f"Loading from {extraction_dir}...")
    print(f"Model: {n_layers} layers, hidden={store.model_config['hidden_size']}")
    print(f"Pairs: {len(meta)}")

    rows = []
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

        row = {
            **pm,
            "orig_correct": ob["correct"],
            "cf_correct": cb["correct"],
            "orig_predicted": ob["predicted"],
            "cf_predicted": cb["predicted"],
            "orig_margin": ob["margin"],
            "answer_flip": ob["predicted"] != cb["predicted"],
            "correctness_flip": ob["correct"] != cb["correct"],
            "abs_delta_gold_logit": abs(cb["gold_logit"] - ob["gold_logit"]),
            "delta_gold_logit": cb["gold_logit"] - ob["gold_logit"],
            "abs_delta_margin": abs(cb["margin"] - ob["margin"]),
            "delta_gold_prob": cb["gold_prob"] - ob["gold_prob"],
        }

        # Per-layer norm_euclid for bootstrap per-layer plots
        rm_final_delta = cd["hidden_final"] - od["hidden_final"]
        norm_orig = np.linalg.norm(od["hidden_final"], axis=1)
        norm_cf = np.linalg.norm(cd["hidden_final"], axis=1)
        mean_norm = np.maximum((norm_orig + norm_cf) / 2, 1e-8)
        euclid = np.linalg.norm(rm_final_delta, axis=1)
        for l in range(n_layers):
            row[f"norm_euclid_L{l}"] = float(euclid[l] / mean_norm[l])

        row["_final_delta"] = cd["hidden_final"] - od["hidden_final"]

        orig_edit_key = pm.get("orig_edit_key")
        od_edit = store.get_original(orig_edit_key)
        if (od_edit is not None and "hidden_at_edit" in od_edit and "hidden_at_edit" in cd):
            row["_edit_delta"] = cd["hidden_at_edit"] - od_edit["hidden_at_edit"]
            row["has_edit"] = True
        else:
            row["has_edit"] = False

        # Track attention availability
        od_attn = od.get("attention_summary") if isinstance(od, dict) else None
        cd_attn = cd.get("attention_summary") if isinstance(cd, dict) else None
        row["has_attention_final"] = (isinstance(od_attn, dict) and isinstance(cd_attn, dict)
                                       and "final_token" in od_attn and "final_token" in cd_attn)

        rows.append(row)

    print(f"Built {len(rows)} enriched rows")
    print(f"  With edit-token states: {sum(1 for r in rows if r['has_edit'])}")
    print(f"  With attention: {sum(1 for r in rows if r.get('has_attention_final'))}")

    report_layers = [
        ("early", list(range(0, n_layers // 4))),
        ("mid", list(range(n_layers // 4, n_layers // 2))),
        ("mid_late", list(range(n_layers // 2, 3 * n_layers // 4))),
        ("late", list(range(3 * n_layers // 4, n_layers))),
    ]

    geometry_results = []

    # ===================================================================
    # ANALYSIS 1: EDIT-TOKEN DIRECTION GEOMETRY
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 1: EDIT-TOKEN DIRECTION GEOMETRY")
    print(f"{'='*70}")

    def analyze_group(name: str, group_rows: list, state_key: str = "_edit_delta"):
        active = [r for r in group_rows if state_key in r]
        if not active:
            print(f"\n  {name}: NO DATA with {state_key}")
            return

        print(f"\n  {name} (n={len(active)}):")
        for range_name, layer_indices in report_layers:
            deltas = np.stack([
                np.mean([r[state_key][l] for l in layer_indices], axis=0)
                for r in active
            ])
            coh = pairwise_cosine_coherence(deltas)
            pca_res = pca_analysis(deltas)

            geometry_results.append({
                "group": name,
                "state": state_key,
                "layer_range": range_name,
                "n": len(active),
                **{f"coh_{k}": v for k, v in coh.items()},
                **{f"pca_{k}": v for k, v in pca_res.items() if k not in ("pca", "transformed")},
            })

            print(f"    {range_name}:")
            print(f"      Cosine coherence: mean={coh['mean_cos']:.4f}, median={coh['median_cos']:.4f}")
            print(f"      PCA top-1 var:    {pca_res['top1_var']:.4f}")
            print(f"      PCA top-5 var:    {pca_res['top5_var']:.4f}")

    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        grp = [
            r for r in rows
            if r["normalized_group"] == grp_name
            and r.get("has_edit")
            and r["normalized_edit_locality"] in {"minimal", "single"}
        ]
        analyze_group(f"{group_alias(grp_name)} — ALL (EDIT TOKEN)", grp, "_edit_delta")

    ctrl_min = [
        r for r in rows
        if r["normalized_group"] == "control"
        and r["normalized_control_subtype"] == "irrelevant_surface"
        and r.get("has_edit")
        and r["normalized_edit_locality"] in {"minimal", "single"}
    ]
    analyze_group("Control irrelevant_surface (EDIT TOKEN)", ctrl_min, "_edit_delta")

    # ===================================================================
    # PROBE A: Focal vs Control Discriminability
    # ===================================================================
    print(f"\n{'─'*70}")
    print("PROBE A: Focal vs Control Discriminability")
    print(f"{'─'*70}")

    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        focal_edit = [
            r for r in rows
            if r["normalized_group"] == grp_name
            and r.get("has_edit")
            and r["normalized_edit_locality"] in {"minimal", "single"}
        ]
        ctrl_edit = [
            r for r in rows
            if r["normalized_group"] == "control"
            and r["normalized_control_subtype"] == "irrelevant_surface"
            and r.get("has_edit")
            and r["normalized_edit_locality"] in {"minimal", "single"}
        ]

        if len(focal_edit) < 3 or len(ctrl_edit) < 3:
            continue

        print(f"\n  {group_alias(grp_name)} vs control:")
        for range_name, layer_indices in report_layers:
            f_d = np.stack([
                np.mean([r["_edit_delta"][l] for l in layer_indices], axis=0)
                for r in focal_edit
            ])
            c_d = np.stack([
                np.mean([r["_edit_delta"][l] for l in layer_indices], axis=0)
                for r in ctrl_edit
            ])

            probe_res = loo_probe_accuracy(c_d, f_d)
            dim_acc = dim_threshold_accuracy(
                np.vstack([c_d, f_d]),
                np.array([0] * len(c_d) + [1] * len(f_d)),
            )
            print(f"    {range_name}: LOO={probe_res['accuracy']:.3f}, DIM={dim_acc:.3f}, n={probe_res['n']}")

    # ===================================================================
    # PROBE B: Attribute-side encoding
    # ===================================================================
    print(f"\n{'─'*70}")
    print("PROBE B: Attribute-side encoding")
    print(f"{'─'*70}")

    eligible_rows = [
        r for r in rows
        if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
        and r.get("has_edit")
        and not r.get("alignment_failed", True)
        and r.get("normalized_label") is not None
    ]

    auto_pairs = pick_top_label_pairs(eligible_rows, min_per_side=3, max_pairs_per_attr=2)
    if not auto_pairs:
        print("  No eligible automatic label pairs found.")
    else:
        for attr, side0, side1 in auto_pairs:
            grp0 = [r for r in eligible_rows if r["normalized_attribute"] == attr and r["normalized_label"] == side0]
            grp1 = [r for r in eligible_rows if r["normalized_attribute"] == attr and r["normalized_label"] == side1]

            if len(grp0) < 3 or len(grp1) < 3:
                continue

            print(f"\n  {attr}: {side0} vs {side1}")
            for range_name, layer_indices in report_layers:
                states0 = np.stack([
                    np.mean([store.get_cf(r["pair_key"])["hidden_at_edit"][l] for l in layer_indices], axis=0)
                    for r in grp0
                ])
                states1 = np.stack([
                    np.mean([store.get_cf(r["pair_key"])["hidden_at_edit"][l] for l in layer_indices], axis=0)
                    for r in grp1
                ])

                probe_res = loo_probe_accuracy(states0, states1)
                dim_acc = dim_threshold_accuracy(
                    np.vstack([states0, states1]),
                    np.array([0] * len(states0) + [1] * len(states1)),
                )
                print(f"    {range_name}: LOO={probe_res['accuracy']:.3f}, DIM={dim_acc:.3f}, n={len(states0)}+{len(states1)}")

    # ===================================================================
    # ANALYSIS 2: LOCAL-TO-GLOBAL COUPLING
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 2: LOCAL-TO-GLOBAL COUPLING")
    print(f"{'='*70}")

    coupling_rows = [
        r for r in rows
        if r.get("has_edit")
        and r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
        and r["normalized_edit_locality"] in {"minimal", "single"}
    ]

    mid = n_layers // 2
    late_layers = list(range(mid, n_layers))

    if coupling_rows:
        edit_mags, final_mags, abs_dlogits, flips = [], [], [], []

        for r in coupling_rows:
            edit_d = np.mean([r["_edit_delta"][l] for l in late_layers], axis=0)
            final_d = np.mean([r["_final_delta"][l] for l in late_layers], axis=0)
            edit_mags.append(float(np.linalg.norm(edit_d)))
            final_mags.append(float(np.linalg.norm(final_d)))
            abs_dlogits.append(r["abs_delta_gold_logit"])
            flips.append(int(r["answer_flip"]))

        edit_mags = np.array(edit_mags)
        final_mags = np.array(final_mags)
        abs_dlogits = np.array(abs_dlogits)
        flips = np.array(flips)

        print(f"\n  Focal rows with edit states (n={len(coupling_rows)}):")
        if HAS_SCIPY:
            r1, p1 = sp_stats.spearmanr(edit_mags, abs_dlogits)
            r2, p2 = sp_stats.spearmanr(final_mags, abs_dlogits)
            r3, p3 = sp_stats.spearmanr(edit_mags, final_mags)
            print(f"    edit_mag → |Δ gold logit|: r={r1:.3f}, p={p1:.4f}")
            print(f"    final_mag → |Δ gold logit|: r={r2:.3f}, p={p2:.4f}")
            print(f"    edit_mag → final_mag:      r={r3:.3f}, p={p3:.4f}")

        edit_deltas_late = np.stack([
            np.mean([r["_edit_delta"][l] for l in late_layers], axis=0)
            for r in coupling_rows
        ])
        final_deltas_late = np.stack([
            np.mean([r["_final_delta"][l] for l in late_layers], axis=0)
            for r in coupling_rows
        ])

        dim_dir_edit = dim_direction(edit_deltas_late)
        final_projections = final_deltas_late @ dim_dir_edit
        edit_projections = edit_deltas_late @ dim_dir_edit

        print(f"\n    DIM direction projection:")
        print(f"      Mean edit projection:  {np.mean(edit_projections):.4f} (std={np.std(edit_projections):.4f})")
        print(f"      Mean final projection: {np.mean(final_projections):.4f} (std={np.std(final_projections):.4f})")

        if HAS_SCIPY:
            r5, p5 = sp_stats.spearmanr(np.abs(final_projections), abs_dlogits)
            print(f"      |final_proj| → |Δ logit|: r={r5:.3f}, p={p5:.4f}")

        cos_edit_final_dirs = cosine_sim(np.mean(edit_deltas_late, axis=0), np.mean(final_deltas_late, axis=0))
        print(f"      Cosine(mean_edit_delta, mean_final_delta): {cos_edit_final_dirs:.4f}")

    # ===================================================================
    # ANALYSIS 3: DIRECTION PERSISTENCE BY LAYER
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 3: DIRECTION PERSISTENCE BY LAYER")
    print(f"{'='*70}")

    focal_with_edit = [
        r for r in rows
        if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
        and r.get("has_edit")
        and r["normalized_edit_locality"] in {"minimal", "single"}
    ]

    persistence_by_layer = []
    if len(focal_with_edit) >= 5:
        for l in range(n_layers):
            edit_d = np.stack([r["_edit_delta"][l] for r in focal_with_edit])
            final_d = np.stack([r["_final_delta"][l] for r in focal_with_edit])
            persistence_by_layer.append(cosine_sim(np.mean(edit_d, axis=0), np.mean(final_d, axis=0)))

        for start in range(0, n_layers, 10):
            chunk = persistence_by_layer[start:min(start + 10, n_layers)]
            s = "  ".join(f"L{start + i}={v:.4f}" for i, v in enumerate(chunk))
            print(f"    {s}")

    # ===================================================================
    # NEW ANALYSES: N-adjusted, correctness flips, token adjustment
    # ===================================================================

    # Correctness-aware flip analysis
    print_correctness_flip_stats(rows, outdir)

    # N-adjusted geometry comparison
    print_n_adjusted_geometry(rows, n_layers, outdir)

    # Token-count adjusted coupling
    print_token_adjusted_coupling(rows, n_layers)

    # ===================================================================
    # SAVE CSVs
    # ===================================================================
    if HAS_PANDAS and geometry_results:
        pd.DataFrame(geometry_results).to_csv(outdir / "geometry_results.csv", index=False)
        print(f"Saved {outdir / 'geometry_results.csv'}")

    # ===================================================================
    # FIGURES
    # ===================================================================
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    if HAS_PLT:
        # 1. Direction persistence
        if len(focal_with_edit) >= 5 and persistence_by_layer:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(range(n_layers), persistence_by_layer, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine(mean_edit_delta, mean_final_delta)")
            ax.set_title("Direction persistence by layer")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "direction_persistence.png", dpi=150)
            plt.close(fig)
            print(f"Saved {outdir / 'direction_persistence.png'}")

        # 2. Coupling scatter plots
        if coupling_rows:
            plot_coupling_scatters(coupling_rows, late_layers, outdir)

        # 3. UMAP/t-SNE/PCA embeddings of edit deltas colored by group
        edit_rows = [r for r in rows if r.get("has_edit")]
        if len(edit_rows) >= 10:
            deltas_late = np.stack([
                np.mean([r["_edit_delta"][l] for l in late_layers], axis=0)
                for r in edit_rows
            ])
            group_labels = [r["normalized_group"] for r in edit_rows]
            group_cols = [GROUP_COLORS.get(g, "#999") for g in group_labels]
            group_labels_alias = [group_alias(g) for g in group_labels]

            attr_labels = [r["normalized_attribute"] for r in edit_rows]
            attr_cols = [ATTR_CMAP.get(a, "#999") for a in attr_labels]

            # PCA biplot
            plot_pca_biplot(deltas_late, group_labels_alias, group_cols,
                            "Edit-Token Deltas (Late Layers) by Group",
                            outdir / "pca_biplot_group.png")
            print(f"Saved {outdir / 'pca_biplot_group.png'}")

            plot_pca_biplot(deltas_late, attr_labels, attr_cols,
                            "Edit-Token Deltas (Late Layers) by Attribute",
                            outdir / "pca_biplot_attribute.png")
            print(f"Saved {outdir / 'pca_biplot_attribute.png'}")

            # t-SNE
            if HAS_TSNE and len(edit_rows) >= 15:
                plot_embedding_scatter(deltas_late, group_labels_alias, group_cols,
                                       "t-SNE of Edit-Token Deltas (Late) by Group",
                                       outdir / "tsne_group.png", method="tsne")
                print(f"Saved {outdir / 'tsne_group.png'}")

                plot_embedding_scatter(deltas_late, attr_labels, attr_cols,
                                       "t-SNE of Edit-Token Deltas (Late) by Attribute",
                                       outdir / "tsne_attribute.png", method="tsne")
                print(f"Saved {outdir / 'tsne_attribute.png'}")

            # UMAP
            if HAS_UMAP and len(edit_rows) >= 15:
                plot_embedding_scatter(deltas_late, group_labels_alias, group_cols,
                                       "UMAP of Edit-Token Deltas (Late) by Group",
                                       outdir / "umap_group.png", method="umap")
                print(f"Saved {outdir / 'umap_group.png'}")

                plot_embedding_scatter(deltas_late, attr_labels, attr_cols,
                                       "UMAP of Edit-Token Deltas (Late) by Attribute",
                                       outdir / "umap_attribute.png", method="umap")
                print(f"Saved {outdir / 'umap_attribute.png'}")

        # 4. RSA across groups
        if HAS_SCIPY:
            for range_name, layer_indices in report_layers:
                group_deltas = {}
                for g in ["core_bias", "identity_bias", "structural_context", "control"]:
                    grp_rows = [r for r in rows if r["normalized_group"] == g and r.get("has_edit")]
                    if len(grp_rows) >= 5:
                        group_deltas[g] = [
                            np.mean([r["_edit_delta"][l] for l in layer_indices], axis=0)
                            for r in grp_rows
                        ]
                if len(group_deltas) >= 2:
                    plot_rsa_across_groups(group_deltas, range_name, outdir / f"rsa_{range_name}.png")
                    print(f"Saved {outdir / f'rsa_{range_name}.png'}")

        # 5. Direction alignment across bias types
        plot_direction_alignment(rows, n_layers, outdir)

        # 6. Per-layer probe accuracy
        plot_perlayer_probe_accuracy(rows, n_layers, outdir)

        # 7. Answer flip transitions
        plot_flip_transitions(rows, outdir)

        # 8. Per-layer bootstrap representation shift
        plot_perlayer_bootstrap(rows, n_layers, outdir)

        # 9. Head selectivity index
        compute_head_selectivity(rows, store, n_layers, outdir)

        # 10. Within-group CF geometry comparisons
        plot_within_group_geometry(rows, store, n_layers, outdir)

    print(f"\n{'='*70}")
    print("STAGE 2 SHARDED ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 sharded analysis with attention-aware candidate discovery")
    parser.add_argument("--extraction_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="stage2_results_sharded")
    args = parser.parse_args()

    run_analysis(args.extraction_dir, args.output_dir)


if __name__ == "__main__":
    main()
