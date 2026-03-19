"""
Stage 2 Analysis (Adapted for New Counterfactual Schema)
========================================================

Anchored on EDIT-TOKEN representations when available.

Backward-compatible with older extraction metadata, but also supports newer
counterfactual outputs with:
  - intervention_type
  - intervention_family
  - analysis_bucket
  - edit_strength
  - attribute_value_counterfactual
  - medical_relevance
  - social_bias_salience

Key adaptations:
  1. Metadata normalization layer shared in spirit with Stage 1.
  2. Replaces hardcoded gender/age/race assumptions with intervention_type.
  3. Supports newer interventions such as:
       gender_identity, sexual_orientation,
       insurance_status, housing_status.
  4. Generalizes demo-vs-control comparisons to:
       focal group vs control, by normalized group and intervention type.
  5. Replaces hardcoded attribute-side probes with automatic top-label pairs.

Usage:
  python analyze_stage2_adapted.py \
    --extractions stage1_extractions_v2.pt \
    --output_dir stage2_results_adapted/
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from attention_analysis_utils import (
    normalize_attention_summary as _normalize_attention_summary,
    attention_shift_metrics,
    headwise_attention_table,
)

# ---------------------------------------------------------------------------
# Behavioral metrics
# ---------------------------------------------------------------------------

AIDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDXA = {0: "A", 1: "B", 2: "C", 3: "D"}


def behavioral_metrics(logits, gold):
    gi = AIDX[gold]
    pred_i = int(np.argmax(logits))
    gl = float(logits[gi])
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    comp = np.concatenate([logits[:gi], logits[gi+1:]])
    return {
        "predicted": IDXA[pred_i],
        "correct": IDXA[pred_i] == gold,
        "gold_logit": gl,
        "gold_prob": float(probs[gi]),
        "margin": float(gl - np.max(comp)),
    }


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine_coherence(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas)
    if deltas.ndim != 2:
        return {"mean_cos": 0, "median_cos": 0, "std_cos": 0, "n_pairs": 0}

    # Drop rows with any non-finite values (prevents NaN propagation)
    finite_mask = np.isfinite(deltas).all(axis=1)
    deltas = deltas[finite_mask]

    n = deltas.shape[0]
    if n < 2:
        return {"mean_cos": 0, "median_cos": 0, "std_cos": 0, "n_pairs": 0}

    norms = np.linalg.norm(deltas, axis=1)
    good = np.isfinite(norms) & (norms > 1e-10)
    deltas = deltas[good]
    norms = norms[good]
    n = deltas.shape[0]
    if n < 2:
        return {"mean_cos": 0, "median_cos": 0, "std_cos": 0, "n_pairs": 0}

    normed = deltas / norms[:, None]

    # Computing the full n×n cosine matrix is O(n^2) and becomes intractable
    # for large n (thousands of items). For large groups, subsample vectors
    # deterministically for a stable estimate.
    max_vectors = 300  # ~45k pairs; keeps runtime reasonable
    if n > max_vectors:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_vectors, replace=False)
        normed = normed[idx]
        n = normed.shape[0]

    cos_matrix = normed @ normed.T
    triu_idx = np.triu_indices(n, k=1)
    cos_vals = cos_matrix[triu_idx]
    cos_vals = cos_vals[np.isfinite(cos_vals)]
    if cos_vals.size == 0:
        return {"mean_cos": 0, "median_cos": 0, "std_cos": 0, "n_pairs": 0}

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
        return {"top1_var": 0, "top5_var": 0, "explained": []}
    n_comp = min(n_components, deltas.shape[0] - 1, deltas.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(deltas)
    evr = pca.explained_variance_ratio_
    return {
        "top1_var": float(evr[0]) if len(evr) > 0 else 0,
        "top5_var": float(np.sum(evr[:5])) if len(evr) >= 5 else float(np.sum(evr)),
        "explained": evr.tolist(),
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

OLD_TO_NEW_TYPE = {
    "gender": "sex",
    "race": "race_ethnicity",
}


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
    raw = OLD_TO_NEW_TYPE.get(raw, raw)
    return raw


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


def normalize_control_subtype(pm: dict, intervention_type: str) -> str | None:
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


def normalize_label(pm: dict) -> str | None:
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


# ---------------------------------------------------------------------------
# Automatic label-pair selection for attribute-side probes
# ---------------------------------------------------------------------------

def pick_top_label_pairs(rows, min_per_side=3, max_pairs_per_attr=2):
    """
    Automatically choose the most common label pairs within each intervention type.
    """
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
                l0, c0 = eligible[i]
                l1, c1 = eligible[j]
                pairs.append((attr, l0, l1))
                pair_count += 1
                if pair_count >= max_pairs_per_attr:
                    break
            if pair_count >= max_pairs_per_attr:
                break
    return pairs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(extractions_path: str, output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {extractions_path}...")
    data = torch.load(extractions_path, map_location="cpu", weights_only=False)

    orig_r = data["original_results"]
    cf_r = data["cf_results"]
    meta = data["pair_metadata"]
    mcfg = data["model_config"]
    n_layers = mcfg["n_layers"]

    print(f"Model: {n_layers} layers, hidden={mcfg['hidden_size']}")
    print(f"Pairs: {len(meta)}")

    rows = []
    head_rows = []
    for raw_pm in meta:
        pm = normalize_metadata(raw_pm)

        qid = pm["question_id"]
        pk = pm["pair_key"]
        gold = pm["gold_answer"]

        od = orig_r.get(qid)
        cd = cf_r.get(pk)
        if od is None or cd is None:
            continue

        ob = behavioral_metrics(od["logits_abcd"], gold)
        cb = behavioral_metrics(cd["logits_abcd"], gold)

        row = {
            **pm,
            "orig_correct": ob["correct"],
            "cf_correct": cb["correct"],
            "orig_margin": ob["margin"],
            "answer_flip": ob["predicted"] != cb["predicted"],
            "correctness_flip": ob["correct"] != cb["correct"],
            "abs_delta_gold_logit": abs(cb["gold_logit"] - ob["gold_logit"]),
            "delta_gold_logit": cb["gold_logit"] - ob["gold_logit"],
            "abs_delta_margin": abs(cb["margin"] - ob["margin"]),
        }

        row["_final_delta"] = cd["hidden_final"] - od["hidden_final"]

        orig_edit_key = pm.get("orig_edit_key")
        od_edit = orig_r.get(orig_edit_key)
        if (od_edit is not None and "hidden_at_edit" in od_edit and "hidden_at_edit" in cd):
            row["_edit_delta"] = cd["hidden_at_edit"] - od_edit["hidden_at_edit"]
            row["has_edit"] = True
        else:
            row["has_edit"] = False

        cf_largest_key = pm.get("cf_largest_key")
        orig_largest_key = pm.get("orig_largest_key")
        cd_largest = cf_r.get(cf_largest_key) if cf_largest_key else None
        od_largest = orig_r.get(orig_largest_key) if orig_largest_key else None
        if (cd_largest is not None and "hidden_at_edit" in cd_largest
                and od_largest is not None and "hidden_at_edit" in od_largest):
            row["_largest_delta"] = cd_largest["hidden_at_edit"] - od_largest["hidden_at_edit"]
            row["has_largest"] = True
        else:
            row["has_largest"] = False

        # Optional attention metrics (if available in extractions)
        late_start = n_layers // 2
        row["has_attention"] = False
        row["has_attention_final"] = False
        row["has_attention_edit"] = False
        row["has_attention_largest"] = False

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

        if row.get("has_edit") and isinstance(od_edit, dict) and isinstance(cd, dict):
            od_e_attn = od_edit.get("attention_summary")
            cd_attn = cd.get("attention_summary")
            if isinstance(od_e_attn, dict) and isinstance(cd_attn, dict):
                a_o = _normalize_attention_summary(od_e_attn.get("edit_region"))
                a_c = _normalize_attention_summary(cd_attn.get("edit_region"))
                if a_o and a_c:
                    am = attention_shift_metrics(a_o, a_c, late_start)
                    for k, v in am.items():
                        row[f"edit_attn_{k}"] = v
                    row["has_attention"] = True
                    row["has_attention_edit"] = True

                    for hr in headwise_attention_table(a_o, a_c, late_start):
                        hr = dict(hr)
                        hr.update(
                            {
                                "question_id": qid,
                                "pair_key": pk,
                                "normalized_group": pm.get("normalized_group"),
                                "normalized_attribute": pm.get("normalized_attribute"),
                                "normalized_label": pm.get("normalized_label"),
                                "source": "edit",
                                "answer_flip": row.get("answer_flip"),
                                "abs_delta_gold_logit": row.get("abs_delta_gold_logit"),
                            }
                        )
                        head_rows.append(hr)

        # Also collect per-head final-token query attention rows (persistence / coupling)
        od_attn = od.get("attention_summary") if isinstance(od, dict) else None
        cd_attn = cd.get("attention_summary") if isinstance(cd, dict) else None
        if isinstance(od_attn, dict) and isinstance(cd_attn, dict):
            a_o = _normalize_attention_summary(od_attn.get("final_token"))
            a_c = _normalize_attention_summary(cd_attn.get("final_token"))
            if a_o and a_c:
                for hr in headwise_attention_table(a_o, a_c, late_start):
                    hr = dict(hr)
                    hr.update(
                        {
                            "question_id": qid,
                            "pair_key": pk,
                            "normalized_group": pm.get("normalized_group"),
                            "normalized_attribute": pm.get("normalized_attribute"),
                            "normalized_label": pm.get("normalized_label"),
                            "source": "final",
                            "answer_flip": row.get("answer_flip"),
                            "abs_delta_gold_logit": row.get("abs_delta_gold_logit"),
                        }
                    )
                    head_rows.append(hr)

        if row.get("has_largest") and isinstance(od_largest, dict) and isinstance(cd_largest, dict):
            od_l_attn = od_largest.get("attention_summary")
            cd_l_attn = cd_largest.get("attention_summary")
            if isinstance(od_l_attn, dict) and isinstance(cd_l_attn, dict):
                a_o = _normalize_attention_summary(od_l_attn.get("largest_edit") or od_l_attn.get("edit_region"))
                a_c = _normalize_attention_summary(cd_l_attn.get("largest_edit") or cd_l_attn.get("edit_region"))
                if a_o and a_c:
                    am = attention_shift_metrics(a_o, a_c, late_start)
                    for k, v in am.items():
                        row[f"largest_attn_{k}"] = v
                    row["has_attention"] = True
                    row["has_attention_largest"] = True

        rows.append(row)

    print(f"Built {len(rows)} enriched rows")
    print(f"  With edit-token states:    {sum(1 for r in rows if r['has_edit'])}")
    print(f"  With largest-region states:{sum(1 for r in rows if r['has_largest'])}")
    if any(r.get("has_attention") for r in rows):
        print(f"  With attention summaries:  {sum(1 for r in rows if r.get('has_attention'))}")

    if head_rows and HAS_PANDAS:
        pd.DataFrame(head_rows).to_csv(outdir / "attention_head_metrics.csv", index=False)
        print("Saved attention_head_metrics.csv")

        # -----------------------------------------------------------------
        # ANALYSIS 1B: Attention-head candidate discovery + behavior coupling
        # -----------------------------------------------------------------
        head_df = pd.DataFrame(head_rows)
        # Prefer edit-query rows for candidate discovery
        if "source" in head_df.columns:
            edit_df = head_df[head_df["source"] == "edit"].copy()
            final_df = head_df[head_df["source"] == "final"].copy()
        else:
            edit_df = head_df.copy()
            final_df = head_df.iloc[0:0].copy()

        if not edit_df.empty:
            group_summary = (
                edit_df.groupby(["normalized_group", "head"], as_index=False)
                .agg(
                    mean_edit_mass_delta=("mass_to_edit_region_delta", "mean"),
                    mean_abs_edit_mass_delta=("mass_to_edit_region_abs_delta", "mean"),
                    mean_entropy_delta=("entropy_delta", "mean"),
                    mean_abs_entropy_delta=("entropy_abs_delta", "mean"),
                    mean_topk_jaccard_late=("topk_jaccard_late", "mean"),
                    mean_abs_delta_gold_logit=("abs_delta_gold_logit", "mean"),
                    n=("head", "size"),
                )
                .sort_values(["normalized_group", "mean_abs_edit_mass_delta"], ascending=[True, False])
            )
            group_summary.to_csv(outdir / "attention_head_group_summary.csv", index=False)
            print("Saved attention_head_group_summary.csv")

            # Plot: top heads by group (always available when group_summary exists)
            if HAS_PLT:
                try:
                    groups = [
                        g
                        for g in ["core_bias", "identity_bias", "structural_context", "control", "other"]
                        if (group_summary["normalized_group"] == g).any()
                    ]
                    if groups:
                        ncols = 2
                        nrows = int(np.ceil(len(groups) / ncols))
                        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
                        axes = np.array(axes).reshape(-1)

                        for ax_i, g in enumerate(groups):
                            ax = axes[ax_i]
                            sub = group_summary[group_summary["normalized_group"] == g].head(15)
                            ax.barh(
                                [f"H{int(h)}" for h in sub["head"]],
                                sub["mean_abs_edit_mass_delta"].to_numpy(),
                                color="#4a90d9" if g != "control" else "#2ca02c",
                                alpha=0.9,
                            )
                            ax.invert_yaxis()
                            ax.set_title(f"{group_alias(g)}: top heads (|Δ mass→edit|, late)", fontsize=11)
                            ax.set_xlabel("mean_abs_edit_mass_delta", fontsize=10)
                            ax.grid(True, alpha=0.25, axis="x")

                        for j in range(ax_i + 1, len(axes)):
                            axes[j].axis("off")

                        fig.suptitle("Attention head rankings by group (edit-query summaries)", fontsize=13, y=1.02)
                        fig.tight_layout()
                        fig.savefig(outdir / "attention_head_group_summary.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print("Saved attention_head_group_summary.png")
                except Exception as e:
                    print(f"WARNING: attention group-summary plot failed: {e}")

            # Head-feature probe: focal vs control using per-head edit mass deltas
            try:
                pivot = edit_df.pivot_table(
                    index="pair_key",
                    columns="head",
                    values="mass_to_edit_region_delta",
                    aggfunc="mean",
                )
                pk_to_group = {
                    r["pair_key"]: r["normalized_group"]
                    for r in rows
                    if r.get("has_attention_edit")
                }
                y = np.array(
                    [
                        1
                        if pk_to_group.get(pk)
                        in {"core_bias", "identity_bias", "structural_context"}
                        else 0
                        for pk in pivot.index
                    ],
                    dtype=int,
                )
                if len(np.unique(y)) == 2 and pivot.shape[0] >= 50:
                    X = pivot.fillna(0.0).to_numpy(dtype=np.float32)
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    loo = LeaveOneOut()
                    correct = 0
                    total = 0
                    coefs = []
                    for tr, te in loo.split(Xs):
                        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")
                        clf.fit(Xs[tr], y[tr])
                        pred = clf.predict(Xs[te])[0]
                        correct += int(pred == y[te][0])
                        total += 1
                        coefs.append(clf.coef_[0])
                    acc = correct / max(total, 1)
                    mean_coef = np.mean(np.stack(coefs), axis=0)
                    head_ids = pivot.columns.to_numpy()
                    top = np.argsort(np.abs(mean_coef))[::-1][:50]
                    probe_df = pd.DataFrame(
                        {
                            "head": head_ids[top],
                            "mean_abs_coef": np.abs(mean_coef[top]),
                            "mean_coef": mean_coef[top],
                        }
                    )
                    probe_df.to_csv(outdir / "attention_head_probe_top_heads.csv", index=False)
                    print(f"Saved attention_head_probe_top_heads.csv (LOO acc={acc:.3f})")
            except Exception as e:
                print(f"WARNING: attention head-feature probe failed: {e}")

            # Behavior coupling per head (Spearman) across all edit rows
            coupling_rows = []
            for h, sub in edit_df.groupby("head"):
                x = sub["mass_to_edit_region_abs_delta"].to_numpy(dtype=np.float32)
                y = sub["abs_delta_gold_logit"].to_numpy(dtype=np.float32)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or not HAS_SCIPY:
                    continue
                rho, p = sp_stats.spearmanr(x[ok], y[ok])
                coupling_rows.append(
                    {
                        "head": int(h),
                        "spearman_abs_mass_to_edit_vs_abs_delta_gold_logit": float(rho),
                        "spearman_p": float(p),
                        "n": int(ok.sum()),
                        "mean_abs_mass_to_edit": float(np.mean(x[ok])),
                    }
                )
            if coupling_rows:
                coupling_df = pd.DataFrame(coupling_rows).sort_values(
                    "mean_abs_mass_to_edit", ascending=False
                )
                coupling_df.to_csv(outdir / "attention_head_behavior_coupling.csv", index=False)
                print("Saved attention_head_behavior_coupling.csv")

            # Focal vs control effect size per head (mean difference)
            focal = edit_df[edit_df["normalized_group"].isin(["core_bias", "identity_bias", "structural_context"])]
            ctrl = edit_df[edit_df["normalized_group"] == "control"]
            if not focal.empty and not ctrl.empty:
                eff_rows = []
                for h in sorted(set(edit_df["head"].unique().tolist())):
                    fx = focal.loc[focal["head"] == h, "mass_to_edit_region_abs_delta"].to_numpy(dtype=np.float32)
                    cx = ctrl.loc[ctrl["head"] == h, "mass_to_edit_region_abs_delta"].to_numpy(dtype=np.float32)
                    fx = fx[np.isfinite(fx)]
                    cx = cx[np.isfinite(cx)]
                    if fx.size < 10 or cx.size < 10:
                        continue
                    eff_rows.append(
                        {
                            "head": int(h),
                            "mean_abs_mass_to_edit_focal": float(np.mean(fx)),
                            "mean_abs_mass_to_edit_control": float(np.mean(cx)),
                            "mean_diff_focal_minus_control": float(np.mean(fx) - np.mean(cx)),
                            "n_focal": int(fx.size),
                            "n_control": int(cx.size),
                        }
                    )
                if eff_rows:
                    eff_df = pd.DataFrame(eff_rows).sort_values(
                        "mean_diff_focal_minus_control", ascending=False
                    )
                    eff_df.to_csv(outdir / "attention_head_focal_vs_control.csv", index=False)
                    print("Saved attention_head_focal_vs_control.csv")

            # Head persistence: edit-query vs final-query deltas (late-layer mass_to_edit_region)
            if not final_df.empty:
                # pivot by (pair_key, head)
                e = edit_df.pivot_table(index=["pair_key", "head"], values="mass_to_edit_region_delta", aggfunc="mean")
                f = final_df.pivot_table(index=["pair_key", "head"], values="mass_to_edit_region_delta", aggfunc="mean")
                joined = e.join(f, how="inner", lsuffix="_edit", rsuffix="_final").reset_index()
                pers_rows = []
                for h, sub in joined.groupby("head"):
                    x = sub["mass_to_edit_region_delta_edit"].to_numpy(dtype=np.float32)
                    y = sub["mass_to_edit_region_delta_final"].to_numpy(dtype=np.float32)
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 10 or not HAS_SCIPY:
                        continue
                    rho, p = sp_stats.spearmanr(x[ok], y[ok])
                    pers_rows.append(
                        {
                            "head": int(h),
                            "spearman_edit_vs_final_mass_to_edit_delta": float(rho),
                            "spearman_p": float(p),
                            "n": int(ok.sum()),
                        }
                    )
                if pers_rows:
                    pers_df = pd.DataFrame(pers_rows).sort_values(
                        "spearman_edit_vs_final_mass_to_edit_delta", ascending=False
                    )
                    pers_df.to_csv(outdir / "attention_head_persistence.csv", index=False)
                    print("Saved attention_head_persistence.csv")

            # --- Visualizations for attention results ---
            if HAS_PLT:
                try:
                    # Plot A: focal vs control effect size (top 20 heads)
                    if 'eff_df' in locals() and not eff_df.empty:
                        sub = eff_df.head(20).copy()
                        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                        ax.bar(
                            [f"H{int(h)}" for h in sub["head"]],
                            sub["mean_diff_focal_minus_control"].to_numpy(),
                            color="#4a90d9",
                            alpha=0.9,
                        )
                        ax.axhline(0, color="black", linewidth=1)
                        ax.set_title("Attention: focal − control (|Δ mass→edit|, late)", fontsize=12)
                        ax.set_ylabel("mean_diff_focal_minus_control", fontsize=11)
                        ax.set_xlabel("Head", fontsize=11)
                        ax.grid(True, alpha=0.25, axis="y")
                        fig.tight_layout()
                        fig.savefig(outdir / "attention_focal_vs_control_top_heads.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print("Saved attention_focal_vs_control_top_heads.png")

                    # Plot B: behavior coupling (top 20 by mean_abs_mass_to_edit)
                    if 'coupling_df' in locals() and not coupling_df.empty:
                        sub = coupling_df.head(20).copy()
                        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                        ax.scatter(
                            sub["mean_abs_mass_to_edit"].to_numpy(),
                            sub["spearman_abs_mass_to_edit_vs_abs_delta_gold_logit"].to_numpy(),
                            s=60,
                            alpha=0.85,
                            color="#d62728",
                        )
                        for _, r in sub.iterrows():
                            ax.text(
                                float(r["mean_abs_mass_to_edit"]),
                                float(r["spearman_abs_mass_to_edit_vs_abs_delta_gold_logit"]),
                                f"H{int(r['head'])}",
                                fontsize=8,
                                alpha=0.8,
                            )
                        ax.set_title("Attention: head coupling to |Δ gold logit|", fontsize=12)
                        ax.set_xlabel("mean_abs_mass_to_edit (late)", fontsize=11)
                        ax.set_ylabel("Spearman( |Δ mass→edit|, |Δ gold logit| )", fontsize=11)
                        ax.grid(True, alpha=0.25)
                        fig.tight_layout()
                        fig.savefig(outdir / "attention_head_behavior_coupling.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print("Saved attention_head_behavior_coupling.png")

                    # Plot C: persistence (top 25 heads by |rho|)
                    if 'pers_df' in locals() and not pers_df.empty:
                        sub = pers_df.copy()
                        sub["abs_rho"] = np.abs(sub["spearman_edit_vs_final_mass_to_edit_delta"].to_numpy())
                        sub = sub.sort_values("abs_rho", ascending=False).head(25)
                        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                        ax.bar(
                            [f"H{int(h)}" for h in sub["head"]],
                            sub["spearman_edit_vs_final_mass_to_edit_delta"].to_numpy(),
                            color="#9467bd",
                            alpha=0.9,
                        )
                        ax.axhline(0, color="black", linewidth=1)
                        ax.set_title("Attention persistence: edit-query vs final-query (late mass→edit Δ)", fontsize=12)
                        ax.set_ylabel("Spearman rho", fontsize=11)
                        ax.set_xlabel("Head", fontsize=11)
                        ax.grid(True, alpha=0.25, axis="y")
                        fig.tight_layout()
                        fig.savefig(outdir / "attention_head_persistence.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print("Saved attention_head_persistence.png")
                except Exception as e:
                    print(f"WARNING: attention plotting failed: {e}")

            # Lightweight unified artifact for attention outputs
            attn_artifact = {
                "n_head_rows": int(len(head_df)),
                "n_edit_head_rows": int(len(edit_df)),
                "n_final_head_rows": int(len(final_df)) if not final_df.empty else 0,
                "files": [
                    "attention_head_metrics.csv",
                    "attention_head_group_summary.csv",
                    "attention_head_behavior_coupling.csv",
                    "attention_head_focal_vs_control.csv",
                    "attention_head_persistence.csv",
                    "attention_head_probe_top_heads.csv",
                ],
                "note": (
                    "These are descriptive, compressed attention-summary analyses for candidate head discovery. "
                    "They do not constitute causal tracing."
                ),
            }
            try:
                with open(outdir / "attention_results.json", "w") as f:
                    json.dump(attn_artifact, f, indent=2)
                print("Saved attention_results.json")
            except Exception as e:
                print(f"WARNING: could not write attention_results.json: {e}")

    n_failed = sum(1 for r in rows if r.get("alignment_failed", False))
    n_zero_edit = sum(1 for r in rows if r.get("n_tokens_changed", 0) == 0 and not r.get("alignment_failed", False))
    tok_changed = [r.get("n_tokens_changed", 0) for r in rows if not r.get("alignment_failed", False)]
    largest_toks = [r.get("n_largest_region_tokens", 0) for r in rows if not r.get("alignment_failed", False)]
    valid_aligned = [r for r in rows if not r.get("alignment_failed", False)]
    frac_largest_eq_all = (
        sum(1 for r in valid_aligned if r.get("n_largest_region_tokens", 0) == r.get("n_tokens_changed", 0))
        / max(len(valid_aligned), 1)
    )

    print(f"\n  Alignment health:")
    print(f"    Alignment failures:         {n_failed}/{len(rows)}")
    print(f"    Zero edit positions:        {n_zero_edit}")
    if tok_changed:
        print(f"    Median n_tokens_changed:    {np.median(tok_changed):.0f}")
        print(f"    Median n_largest_region:    {np.median(largest_toks):.0f}")
        print(f"    Frac largest==all:          {frac_largest_eq_all:.3f}")

    # ---------------------------------------------------------------------
    # ANALYSIS 1: edit-token direction geometry
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("ANALYSIS 1: EDIT-TOKEN DIRECTION GEOMETRY")
    print(f"{'='*70}")

    report_layers = [
        ("early", list(range(0, n_layers // 4))),
        ("mid", list(range(n_layers // 4, n_layers // 2))),
        ("mid_late", list(range(n_layers // 2, 3 * n_layers // 4))),
        ("late", list(range(3 * n_layers // 4, n_layers))),
    ]

    geometry_results = []

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

            result = {
                "group": name,
                "state": state_key,
                "layer_range": range_name,
                "n": len(active),
                **{f"coh_{k}": v for k, v in coh.items()},
                **{f"pca_{k}": v for k, v in pca_res.items()},
            }
            geometry_results.append(result)

            print(f"    {range_name}:")
            print(f"      Cosine coherence: mean={coh['mean_cos']:.4f}, median={coh['median_cos']:.4f}")
            print(f"      PCA top-1 var:    {pca_res['top1_var']:.4f}")
            print(f"      PCA top-5 var:    {pca_res['top5_var']:.4f}")

    # Focal edits by group
    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        grp = [r for r in rows
               if r["normalized_group"] == grp_name
               and r.get("has_edit")
               and r["normalized_edit_locality"] in {"minimal", "single"}]
        analyze_group(f"{group_alias(grp_name)} — ALL (EDIT TOKEN)", grp, "_edit_delta")

    # By intervention type
    focal_types = sorted(set(
        r["normalized_attribute"] for r in rows
        if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
    ))
    for intervention in focal_types:
        grp = [r for r in rows
               if r["normalized_attribute"] == intervention
               and r.get("has_edit")
               and r["normalized_group"] != "control"]
        if len(grp) >= 3:
            analyze_group(f"Intervention: {intervention} (EDIT TOKEN)", grp, "_edit_delta")

    # Controls
    ctrl_min = [r for r in rows
                if r["normalized_group"] == "control"
                and r["normalized_control_subtype"] == "irrelevant_surface"
                and r.get("has_edit")
                and r["normalized_edit_locality"] in {"minimal", "single"}]
    analyze_group("Control irrelevant_surface (EDIT TOKEN)", ctrl_min, "_edit_delta")

    print(f"\n{'─'*70}")
    print("ROBUSTNESS: Largest contiguous edit region only")
    print(f"{'─'*70}")

    focal_largest = [r for r in rows
                     if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                     and r.get("has_largest")]
    ctrl_largest = [r for r in rows
                    if r["normalized_group"] == "control"
                    and r["normalized_control_subtype"] == "irrelevant_surface"
                    and r.get("has_largest")]

    analyze_group("Focal edits (LARGEST REGION)", focal_largest, "_largest_delta")
    analyze_group("Control (LARGEST REGION)", ctrl_largest, "_largest_delta")

    print(f"\n{'─'*70}")
    print("Same analysis on FINAL TOKEN for comparison")
    print(f"{'─'*70}")

    focal_final = [r for r in rows if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}]
    ctrl_final = [r for r in rows if r["normalized_group"] == "control"
                  and r["normalized_control_subtype"] == "irrelevant_surface"]

    analyze_group("Focal edits — ALL (FINAL TOKEN)", focal_final, "_final_delta")
    analyze_group("Control irrelevant_surface (FINAL TOKEN)", ctrl_final, "_final_delta")

    # ---------------------------------------------------------------------
    # PROBE A: focal vs control discriminability
    # ---------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PROBE A: Focal vs Control Discriminability (PCA-reduced)")
    print(f"{'─'*70}")

    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        demo_edit = [r for r in rows
                     if r["normalized_group"] == grp_name
                     and r.get("has_edit")
                     and r["normalized_edit_locality"] in {"minimal", "single"}]
        ctrl_edit = [r for r in rows
                     if r["normalized_group"] == "control"
                     and r["normalized_control_subtype"] == "irrelevant_surface"
                     and r.get("has_edit")
                     and r["normalized_edit_locality"] in {"minimal", "single"}]

        if len(demo_edit) < 3 or len(ctrl_edit) < 3:
            continue

        print(f"\n  {group_alias(grp_name)} vs control:")
        for range_name, layer_indices in report_layers:
            demo_d = np.stack([
                np.mean([r["_edit_delta"][l] for l in layer_indices], axis=0)
                for r in demo_edit
            ])
            ctrl_d = np.stack([
                np.mean([r["_edit_delta"][l] for l in layer_indices], axis=0)
                for r in ctrl_edit
            ])

            probe_res = loo_probe_accuracy(ctrl_d, demo_d)
            dim_acc = dim_threshold_accuracy(
                np.vstack([ctrl_d, demo_d]),
                np.array([0] * len(ctrl_d) + [1] * len(demo_d)),
            )
            print(f"    {range_name} (edit-token): LOO={probe_res['accuracy']:.3f}, DIM={dim_acc:.3f}, n={probe_res['n']}")

    # ---------------------------------------------------------------------
    # PROBE B: attribute-side encoding
    # ---------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PROBE B: Attribute-Side Encoding (automatic label pairs)")
    print(f"{'─'*70}")

    eligible_rows = [r for r in rows
                     if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                     and r.get("has_edit")
                     and not r.get("alignment_failed", True)
                     and r.get("normalized_label") is not None]

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
                states0, states1 = [], []
                for r in grp0:
                    cd = cf_r.get(r["pair_key"])
                    if cd is not None and "hidden_at_edit" in cd:
                        states0.append(np.mean([cd["hidden_at_edit"][l] for l in layer_indices], axis=0))
                for r in grp1:
                    cd = cf_r.get(r["pair_key"])
                    if cd is not None and "hidden_at_edit" in cd:
                        states1.append(np.mean([cd["hidden_at_edit"][l] for l in layer_indices], axis=0))

                if len(states0) >= 3 and len(states1) >= 3:
                    s0 = np.stack(states0)
                    s1 = np.stack(states1)
                    probe_res = loo_probe_accuracy(s0, s1)
                    dim_acc = dim_threshold_accuracy(
                        np.vstack([s0, s1]),
                        np.array([0] * len(s0) + [1] * len(s1)),
                    )
                    print(f"    {range_name}: LOO={probe_res['accuracy']:.3f}, DIM={dim_acc:.3f}, n={len(s0)}+{len(s1)}")

    # ---------------------------------------------------------------------
    # ANALYSIS 2: local-to-global coupling
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("ANALYSIS 2: LOCAL-TO-GLOBAL COUPLING")
    print(f"{'='*70}")

    coupling_rows = [r for r in rows
                     if r.get("has_edit")
                     and r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                     and r["normalized_edit_locality"] in {"minimal", "single"}]

    if coupling_rows:
        mid = n_layers // 2
        late_layers = list(range(mid, n_layers))

        edit_mags = []
        final_mags = []
        abs_dlogits = []
        flips = []

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
        print(f"    Mid→late layers ({mid}–{n_layers-1})")

        if HAS_SCIPY:
            r1, p1 = sp_stats.spearmanr(edit_mags, abs_dlogits)
            r2, p2 = sp_stats.spearmanr(final_mags, abs_dlogits)
            r3, p3 = sp_stats.spearmanr(edit_mags, final_mags)

            print(f"    edit_mag → |Δ gold logit|: r={r1:.3f}, p={p1:.4f}")
            print(f"    final_mag → |Δ gold logit|: r={r2:.3f}, p={p2:.4f}")
            print(f"    edit_mag → final_mag:      r={r3:.3f}, p={p3:.4f}")

            if np.sum(flips) > 0 and np.sum(flips) < len(flips):
                r4, p4 = sp_stats.pointbiserialr(flips, edit_mags)
                print(f"    edit_mag → answer_flip:    r={r4:.3f}, p={p4:.4f}")

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

        cos_edit_final_dirs = cosine_sim(
            np.mean(edit_deltas_late, axis=0),
            np.mean(final_deltas_late, axis=0),
        )
        print(f"      Cosine(mean_edit_delta, mean_final_delta): {cos_edit_final_dirs:.4f}")

    else:
        print("  No focal rows with edit-token states")

    # ---------------------------------------------------------------------
    # ANALYSIS 3: direction persistence by layer
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("ANALYSIS 3: DIRECTION PERSISTENCE BY LAYER")
    print(f"{'='*70}")

    focal_with_edit = [r for r in rows
                       if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                       and r.get("has_edit")
                       and r["normalized_edit_locality"] in {"minimal", "single"}]

    persistence_by_layer = []
    if len(focal_with_edit) >= 5:
        print(f"\n  Focal rows with edit states (n={len(focal_with_edit)})")

        for l in range(n_layers):
            edit_d = np.stack([r["_edit_delta"][l] for r in focal_with_edit])
            final_d = np.stack([r["_final_delta"][l] for r in focal_with_edit])
            mean_edit = np.mean(edit_d, axis=0)
            mean_final = np.mean(final_d, axis=0)
            cos = cosine_sim(mean_edit, mean_final)
            persistence_by_layer.append(cos)

        for start in range(0, n_layers, 10):
            chunk = persistence_by_layer[start:min(start+10, n_layers)]
            s = "  ".join(f"L{start+i}={v:.4f}" for i, v in enumerate(chunk))
            print(f"    {s}")

        for intervention in sorted(set(r["normalized_attribute"] for r in focal_with_edit)):
            attr_grp = [r for r in focal_with_edit if r["normalized_attribute"] == intervention]
            if len(attr_grp) < 3:
                continue
            persist = []
            for l in range(n_layers):
                ed = np.mean(np.stack([r["_edit_delta"][l] for r in attr_grp]), axis=0)
                fd = np.mean(np.stack([r["_final_delta"][l] for r in attr_grp]), axis=0)
                persist.append(cosine_sim(ed, fd))
            mid = n_layers // 2
            print(f"\n    {intervention} (n={len(attr_grp)}): mid-late mean persistence={np.mean(persist[mid:]):.4f}")
    else:
        print("  Insufficient data for persistence analysis")

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    layerwise_records = []

    if HAS_PLT:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")

        def get_plot_group(state_key, grp_name):
            return [
                r for r in rows
                if r["normalized_group"] == grp_name
                and (state_key != "_edit_delta" or r.get("has_edit"))
                and state_key in r
            ]

        # Plot 1
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_groups = [
            ("core_bias", "#d62728", "Core bias"),
            ("identity_bias", "#1f77b4", "Identity bias"),
            ("structural_context", "#9467bd", "Structural context"),
            ("control", "#2ca02c", "Control"),
        ]

        for ax_i, (state_key, title) in enumerate([
            ("_edit_delta", "Edit-Token Deltas"),
            ("_final_delta", "Final-Token Deltas"),
        ]):
            ax = axes[ax_i]
            for grp_name, color, label in plot_groups:
                grp = get_plot_group(state_key, grp_name)
                if len(grp) < 3:
                    continue

                layer_coherence = []
                for l in range(n_layers):
                    deltas = np.stack([r[state_key][l] for r in grp])
                    coh = pairwise_cosine_coherence(deltas)
                    layer_coherence.append(coh["mean_cos"])

                ax.plot(range(n_layers), layer_coherence, color=color, label=f"{label} (n={len(grp)})", linewidth=2)

            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Mean Pairwise Cosine", fontsize=11)
            ax.set_title(f"Delta Coherence: {title}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Direction Coherence by Layer", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "coherence_by_layer.png", dpi=150, bbox_inches="tight")
        print("  Saved coherence_by_layer.png")
        plt.close(fig)

        # Plot 2
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, (state_key, title) in enumerate([
            ("_edit_delta", "Edit-Token Deltas"),
            ("_final_delta", "Final-Token Deltas"),
        ]):
            ax = axes[ax_i]
            for grp_name, color, label in plot_groups:
                grp = get_plot_group(state_key, grp_name)
                if len(grp) < 3:
                    continue

                pca_vars = []
                for l in range(n_layers):
                    deltas = np.stack([r[state_key][l] for r in grp])
                    pca_res = pca_analysis(deltas, n_components=5)
                    pca_vars.append(pca_res["top1_var"])

                ax.plot(range(n_layers), pca_vars, color=color, label=label, linewidth=2)

            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("PCA Top-1 Variance Ratio", fontsize=11)
            ax.set_title(f"Low-Dim Structure: {title}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle("PCA Top-1 Variance by Layer", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "pca_variance_by_layer.png", dpi=150, bbox_inches="tight")
        print("  Saved pca_variance_by_layer.png")
        plt.close(fig)

        # Plot 3
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        cmap = plt.get_cmap("tab10")
        focal_types = sorted(set(
            r["normalized_attribute"] for r in rows
            if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
        ))

        for idx, intervention in enumerate(focal_types[:10]):
            grp = [r for r in rows
                   if r["normalized_attribute"] == intervention
                   and r.get("has_edit")]
            if len(grp) < 3:
                continue

            layer_coh = []
            for l in range(n_layers):
                deltas = np.stack([r["_edit_delta"][l] for r in grp])
                coh = pairwise_cosine_coherence(deltas)
                layer_coh.append(coh["mean_cos"])

            ax.plot(range(n_layers), layer_coh, color=cmap(idx), label=f"{intervention} (n={len(grp)})", linewidth=2)

        ctrl_grp = [r for r in rows
                    if r["normalized_group"] == "control"
                    and r["normalized_control_subtype"] == "irrelevant_surface"
                    and r.get("has_edit")]
        if len(ctrl_grp) >= 3:
            layer_coh = []
            for l in range(n_layers):
                deltas = np.stack([r["_edit_delta"][l] for r in ctrl_grp])
                coh = pairwise_cosine_coherence(deltas)
                layer_coh.append(coh["mean_cos"])
            ax.plot(range(n_layers), layer_coh, color="gray", linestyle="--", linewidth=2, label=f"Control (n={len(ctrl_grp)})")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Mean Pairwise Cosine of Deltas", fontsize=11)
        ax.set_title("Edit-Token Delta Coherence by Intervention Type", fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "coherence_by_intervention.png", dpi=150)
        print("  Saved coherence_by_intervention.png")
        plt.close(fig)

        # Plot 4
        if coupling_rows:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.scatter(edit_mags, abs_dlogits, alpha=0.5, s=20, color="#d62728")
            ax.set_xlabel("Edit-token delta magnitude", fontsize=11)
            ax.set_ylabel("|Δ gold logit|", fontsize=11)
            ax.set_title("Edit Perturbation → Behavioral Change", fontsize=12)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.scatter(edit_mags, final_mags, alpha=0.5, s=20, color="#1f77b4")
            ax.set_xlabel("Edit-token delta magnitude", fontsize=11)
            ax.set_ylabel("Final-token delta magnitude", fontsize=11)
            ax.set_title("Local → Global Propagation", fontsize=12)
            ax.grid(True, alpha=0.3)

            fig.suptitle("Local-to-Global Coupling (Focal Edits)", fontsize=13, y=1.02)
            fig.tight_layout()
            fig.savefig(outdir / "local_to_global.png", dpi=150, bbox_inches="tight")
            print("  Saved local_to_global.png")
            plt.close(fig)

        # Plot 5
        if len(focal_with_edit) >= 5:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(range(n_layers), persistence_by_layer, color="#d62728", linewidth=2, label="All focal edits")

            for intervention in focal_types[:8]:
                attr_grp = [r for r in focal_with_edit if r["normalized_attribute"] == intervention]
                if len(attr_grp) < 3:
                    continue
                persist = []
                for l in range(n_layers):
                    ed = np.mean(np.stack([r["_edit_delta"][l] for r in attr_grp]), axis=0)
                    fd = np.mean(np.stack([r["_final_delta"][l] for r in attr_grp]), axis=0)
                    persist.append(cosine_sim(ed, fd))
                ax.plot(range(n_layers), persist, linewidth=1.5, linestyle="--", label=f"{intervention} (n={len(attr_grp)})")

            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Cosine(mean_edit_delta, mean_final_delta)", fontsize=11)
            ax.set_title("Direction Persistence: Edit → Final Token", fontsize=12)
            ax.legend(fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
            fig.tight_layout()
            fig.savefig(outdir / "direction_persistence.png", dpi=150)
            print("  Saved direction_persistence.png")
            plt.close(fig)

        # Plot 6
        focal_edit_plot = [r for r in rows
                           if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                           and r.get("has_edit")]
        ctrl_edit_plot = [r for r in rows
                          if r["normalized_group"] == "control"
                          and r["normalized_control_subtype"] == "irrelevant_surface"
                          and r.get("has_edit")]

        if len(focal_edit_plot) >= 5 and len(ctrl_edit_plot) >= 5:
            layer_accs = []
            layer_dim_accs = []

            for l in range(n_layers):
                f_d = np.stack([r["_edit_delta"][l] for r in focal_edit_plot])
                c_d = np.stack([r["_edit_delta"][l] for r in ctrl_edit_plot])

                pr = loo_probe_accuracy(c_d, f_d)
                layer_accs.append(pr["accuracy"])

                da = dim_threshold_accuracy(
                    np.vstack([c_d, f_d]),
                    np.array([0] * len(c_d) + [1] * len(f_d)),
                )
                layer_dim_accs.append(da)

                layerwise_records.append({
                    "layer": l,
                    "loo_accuracy": pr["accuracy"],
                    "dim_accuracy": da,
                })

            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(range(n_layers), layer_accs, color="#d62728", linewidth=2, label="LOO Logistic (PCA-reduced)")
            ax.plot(range(n_layers), layer_dim_accs, color="#1f77b4", linewidth=2, linestyle="--", label="DIM threshold")
            ax.axhline(0.5, color="gray", linewidth=1, linestyle=":")
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_title("Focal vs Control Probe Accuracy by Layer (Edit-Token)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "probe_accuracy_by_layer.png", dpi=150)
            print("  Saved probe_accuracy_by_layer.png")
            plt.close(fig)

    # ---------------------------------------------------------------------
    # Save layerwise metrics
    # ---------------------------------------------------------------------
    if layerwise_records:
        if HAS_PANDAS:
            pd.DataFrame(layerwise_records).to_csv(outdir / "layerwise_metrics.csv", index=False)
            print("Saved layerwise_metrics.csv")
        else:
            with open(outdir / "layerwise_metrics.json", "w") as f:
                json.dump(layerwise_records, f, indent=2)
            print("Saved layerwise_metrics.json")

    if geometry_results:
        with open(outdir / "geometry_results.json", "w") as f:
            json.dump(geometry_results, f, indent=2)
        print("Saved geometry_results.json")

    print(f"\n{'='*70}")
    print("STAGE 2 ADAPTED COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 adapted: direction geometry + local-to-global coupling"
    )
    parser.add_argument("--extractions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="stage2_results_adapted")
    args = parser.parse_args()

    run_analysis(args.extractions, args.output_dir)


if __name__ == "__main__":
    main()