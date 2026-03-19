"""
Stage 1 Analysis (Adapted for New Counterfactual Schema)
========================================================

Backward-compatible with older extraction metadata, but also supports
newer counterfactual outputs that use fields like:

  - intervention_type
  - intervention_family
  - analysis_bucket
  - edit_strength
  - attribute_value_counterfactual
  - medical_relevance
  - social_bias_salience

Key adaptations:
  1. Adds a metadata normalization layer so both old and new schemas work.
  2. Replaces hardcoded gender/age/race logic with intervention_type-aware logic.
  3. Supports newer intervention types like:
       gender_identity, sexual_orientation,
       insurance_status, housing_status, occupation, etc.
  4. Keeps the original representational / behavioral analyses intact.
  5. Produces results by normalized analysis group and by intervention type.

Usage:
  python analyze_stage1_adapted.py \
    --extractions stage1_extractions_v2.pt \
    --output_dir stage1_results_adapted/
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

from attention_analysis_utils import (
    normalize_attention_summary as _normalize_attention_summary,
    attention_shift_metrics,
    headwise_attention_table,
)

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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
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

    comp = np.concatenate([logits[:gi], logits[gi+1:]])
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
    }


# ---------------------------------------------------------------------------
# Representational metrics
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
    """
    Stable normalized groups:
      - core_bias
      - identity_bias
      - structural_context
      - control
      - other
    """
    # Prefer explicit new-style bucket
    bucket = safe_str(pm.get("analysis_bucket"))
    if bucket in {"core_bias", "identity_bias", "structural_context", "control"}:
        return bucket

    # Fallback: legacy analysis_class
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

    # Fallback by intervention type
    if intervention_type in CONTROL_TYPES:
        return "control"
    if intervention_type in STRUCTURAL_TYPES:
        return "structural_context"
    if intervention_type in IDENTITY_BIAS_TYPES:
        return "identity_bias"
    if intervention_type in CORE_BIAS_TYPES:
        return "core_bias"

    # Legacy fallback for control subtype
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
    """
    Stable locality labels:
      - minimal
      - sentence_level
      - broader
      - single
      - unknown

    New schema often uses edit_strength/edit_scope instead of edit_locality.
    """
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

    norm = dict(pm)
    norm["normalized_attribute"] = intervention_type
    norm["normalized_group"] = norm_group
    norm["normalized_control_subtype"] = norm_ctrl
    norm["normalized_edit_locality"] = norm_locality
    norm["normalized_label"] = norm_label
    return norm


def group_alias(g: str) -> str:
    # Convenience aliases for plots/prints
    return {
        "core_bias": "Core bias",
        "identity_bias": "Identity bias",
        "structural_context": "Structural context",
        "control": "Control",
        "other": "Other",
    }.get(g, g)


# ---------------------------------------------------------------------------
# Pretty printing helpers
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


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(extractions_path: str, output_dir: str, margin_threshold: float = 0.5):
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
    print(f"Layer note: {mcfg.get('note', 'n/a')}")

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
        pb = pairwise_behavioral(ob, cb)

        row = {**pm, **pb}

        # final-token repr
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

        # edit-token repr
        orig_edit_key = pm.get("orig_edit_key")
        od_edit = orig_r.get(orig_edit_key) if orig_edit_key else None
        if (od_edit is not None and "hidden_at_edit" in od_edit
                and cd is not None and "hidden_at_edit" in cd):
            rm_edit = repr_metrics_per_layer(
                od_edit["hidden_at_edit"], cd["hidden_at_edit"]
            )
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

        # -----------------------------------------------------------------
        # Optional attention metrics (if extract_representations ran with --extract_attention)
        # -----------------------------------------------------------------
        late_start = mid
        row["has_attention"] = False
        row["has_attention_final"] = False
        row["has_attention_edit"] = False
        row["has_attention_largest"] = False

        od_attn = od.get("attention_summary") if isinstance(od, dict) else None
        cd_attn = cd.get("attention_summary") if isinstance(cd, dict) else None

        # Final-token query attention
        if isinstance(od_attn, dict) and isinstance(cd_attn, dict):
            a_o = _normalize_attention_summary(od_attn.get("final_token"))
            a_c = _normalize_attention_summary(cd_attn.get("final_token"))
            if a_o and a_c:
                am = attention_shift_metrics(a_o, a_c, late_start)
                for k, v in am.items():
                    row[f"final_attn_{k}"] = v
                row["has_attention"] = True
                row["has_attention_final"] = True

        # Edit-region query attention (requires per-edit-position original extraction)
        if isinstance(od_edit, dict) and isinstance(cd, dict):
            od_edit_attn = od_edit.get("attention_summary")
            cd_attn = cd.get("attention_summary")
            if isinstance(od_edit_attn, dict) and isinstance(cd_attn, dict):
                a_o = _normalize_attention_summary(od_edit_attn.get("edit_region"))
                a_c = _normalize_attention_summary(cd_attn.get("edit_region"))
                if a_o and a_c:
                    am = attention_shift_metrics(a_o, a_c, late_start)
                    for k, v in am.items():
                        row[f"edit_attn_{k}"] = v
                    row["has_attention"] = True
                    row["has_attention_edit"] = True

                    # Long-form per-head rows for candidate discovery
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

        # Per-head rows for final-token query attention (persistence / downstream coupling)
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

        # Largest-region query attention (requires cached largest extractions)
        cf_largest_key = pm.get("cf_largest_key")
        orig_largest_key = pm.get("orig_largest_key")
        cd_largest = cf_r.get(cf_largest_key) if cf_largest_key else None
        od_largest = orig_r.get(orig_largest_key) if orig_largest_key else None
        if isinstance(cd_largest, dict) and isinstance(od_largest, dict):
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

    print(f"Computed metrics for {len(rows)} pairs")

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "stage1_metrics_adapted.csv", index=False)
        print(f"Saved {outdir / 'stage1_metrics_adapted.csv'}")

    # ---------------------------------------------------------------------
    # Group summaries
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # ANALYSIS A: surface-matched comparison
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("A. SURFACE-MATCHED COMPARISON")
    print(f"{'='*70}")

    focal_surface = [r for r in rows
                     if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
                     and r["normalized_edit_locality"] in {"minimal", "single"}]

    ctrl_surface = [r for r in rows
                    if r["normalized_group"] == "control"
                    and r["normalized_edit_locality"] in {"minimal", "single"}]

    ctrl_surface_irrelevant = [r for r in ctrl_surface
                               if r["normalized_control_subtype"] == "irrelevant_surface"]
    ctrl_surface_rework = [r for r in ctrl_surface
                           if r["normalized_control_subtype"] == "neutral_rework"]

    print_group("All focal edits (surface-matched)", focal_surface)
    print_group("Control: irrelevant_surface", ctrl_surface_irrelevant)
    print_group("Control: neutral_rework", ctrl_surface_rework)

    for grp_name in ["core_bias", "identity_bias", "structural_context"]:
        grp = [r for r in focal_surface if r["normalized_group"] == grp_name]
        print_group(f"{group_alias(grp_name)}", grp)

    for intervention in sorted(set(r["normalized_attribute"] for r in focal_surface)):
        grp = [r for r in focal_surface if r["normalized_attribute"] == intervention]
        print_group(f"Intervention: {intervention}", grp)

    if focal_surface and ctrl_surface_irrelevant and HAS_SCIPY:
        d_vals = [r["abs_delta_gold_logit"] for r in focal_surface]
        c_vals = [r["abs_delta_gold_logit"] for r in ctrl_surface_irrelevant]
        u, p = sp_stats.mannwhitneyu(d_vals, c_vals, alternative="greater")
        print(f"\n  Mann-Whitney (focal > irrelevant_surface) |Δ gold logit|: U={u:.0f}, p={p:.4f}")

    # ---------------------------------------------------------------------
    # ANALYSIS B: within-question paired comparison by normalized group
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("B. WITHIN-QUESTION PAIRED COMPARISON")
    print(f"{'='*70}")

    by_question = defaultdict(lambda: {
        "core_bias": [],
        "identity_bias": [],
        "structural_context": [],
        "irrelevant_surface": [],
        "neutral_rework": [],
    })

    for r in rows:
        qid = r["question_id"]
        g = r["normalized_group"]
        if g in {"core_bias", "identity_bias", "structural_context"}:
            by_question[qid][g].append(r)
        elif g == "control":
            sub = r["normalized_control_subtype"]
            if sub in {"irrelevant_surface", "neutral_rework"}:
                by_question[qid][sub].append(r)

    paired_diffs = {}
    for focal_group in ["core_bias", "identity_bias", "structural_context"]:
        paired_diffs[focal_group] = {}
        for ctrl_sub in ["irrelevant_surface", "neutral_rework"]:
            diffs_logit = []
            diffs_margin = []
            diffs_ne = []

            for qid, groups in by_question.items():
                if not groups[focal_group] or not groups[ctrl_sub]:
                    continue

                focal_dl = np.mean([r["abs_delta_gold_logit"] for r in groups[focal_group]])
                ctrl_dl = np.mean([r["abs_delta_gold_logit"] for r in groups[ctrl_sub]])
                diffs_logit.append(focal_dl - ctrl_dl)

                focal_dm = np.mean([r["abs_delta_margin"] for r in groups[focal_group]])
                ctrl_dm = np.mean([r["abs_delta_margin"] for r in groups[ctrl_sub]])
                diffs_margin.append(focal_dm - ctrl_dm)

                focal_ne = np.mean([r["mean_norm_euclid_mid_late"] for r in groups[focal_group]])
                ctrl_ne = np.mean([r["mean_norm_euclid_mid_late"] for r in groups[ctrl_sub]])
                diffs_ne.append(focal_ne - ctrl_ne)

            paired_diffs[focal_group][ctrl_sub] = {
                "logit": diffs_logit,
                "margin": diffs_margin,
                "norm_euclid": diffs_ne,
            }

            print(f"\n  {group_alias(focal_group)} vs {ctrl_sub}:")
            print(f"    Questions with both: {len(diffs_logit)}")
            if diffs_logit:
                print(f"    mean(|Δℓ| focal - ctrl): {np.mean(diffs_logit):.4f}")
                print(f"    median(|Δℓ| focal - ctrl): {np.median(diffs_logit):.4f}")
                print(f"    frac focal > ctrl: {np.mean([d > 0 for d in diffs_logit]):.3f}")
                if HAS_SCIPY and len(diffs_logit) >= 5:
                    t, p = sp_stats.wilcoxon(diffs_logit, alternative="greater")
                    print(f"    Wilcoxon: T={t:.0f}, p={p:.4f}")

    # ---------------------------------------------------------------------
    # ANALYSIS C: layerwise normalized distance
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("C. LAYERWISE NORMALIZED EUCLIDEAN DISTANCE")
    print(f"{'='*70}")

    for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
        grp = [r for r in rows if r["normalized_group"] == grp_name]
        if not grp:
            continue
        means = [np.mean([r.get(f"norm_euclid_L{l}", 0) for r in grp])
                 for l in range(n_layers)]
        print(f"\n  {group_alias(grp_name)} (n={len(grp)}):")
        for start in range(0, n_layers, 10):
            chunk = means[start:min(start+10, n_layers)]
            s = "  ".join(f"L{start+i}={v:.5f}" for i, v in enumerate(chunk))
            print(f"    {s}")

    # ---------------------------------------------------------------------
    # ANALYSIS D: edit-token vs final-token
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("D. EDIT-TOKEN vs FINAL-TOKEN REPRESENTATIONAL CHANGE")
    print(f"{'='*70}")

    edit_rows = [r for r in rows if r.get("has_edit_repr")]
    if edit_rows:
        for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
            grp = [r for r in edit_rows if r["normalized_group"] == grp_name]
            if not grp:
                continue
            final_ne = np.mean([r["mean_norm_euclid_mid_late"] for r in grp])
            edit_ne = np.mean([r["edit_mean_norm_euclid_mid_late"] for r in grp])
            print(f"\n  {group_alias(grp_name)} (n={len(grp)}):")
            print(f"    Final-token norm euclid (mid→late): {final_ne:.6f}")
            print(f"    Edit-token norm euclid (mid→late):  {edit_ne:.6f}")
            if final_ne > 0:
                print(f"    Ratio (edit/final):                 {edit_ne/final_ne:.3f}")
    else:
        print("  No edit-position hidden states available")

    # ---------------------------------------------------------------------
    # ANALYSIS E: confidence-conditional behavior
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"E. CONDITIONAL ON MODEL CONFIDENCE (margin > {margin_threshold})")
    print(f"{'='*70}")

    for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
        grp_all = [r for r in rows if r["normalized_group"] == grp_name]
        grp_correct = [r for r in grp_all if r["orig_correct"]]
        grp_confident = [r for r in grp_correct if r["orig_margin"] > margin_threshold]

        print(f"\n  {group_alias(grp_name)}:")
        print(f"    All: n={len(grp_all)}")
        if grp_all:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_all]):.3f}  "
                  f"|Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_all]):.4f}")
        print(f"    Original correct only: n={len(grp_correct)}")
        if grp_correct:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_correct]):.3f}  "
                  f"|Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_correct]):.4f}")
        print(f"    Correct + margin>{margin_threshold}: n={len(grp_confident)}")
        if grp_confident:
            print(f"      flip={np.mean([r['answer_flip'] for r in grp_confident]):.3f}  "
                  f"|Δℓ|={np.mean([r['abs_delta_gold_logit'] for r in grp_confident]):.4f}")

    # ---------------------------------------------------------------------
    # ANALYSIS G: attention summary metrics (descriptive + head ranking)
    # ---------------------------------------------------------------------
    if any(r.get("has_attention_edit") or r.get("has_attention_final") for r in rows):
        print(f"\n{'='*70}")
        print("G. ATTENTION SUMMARY METRICS (DESCRIPTIVE)")
        print(f"{'='*70}")

        def _summarize_metric(metric_key: str, subset_rows: list[dict]):
            vals = [r.get(metric_key) for r in subset_rows if metric_key in r and r.get(metric_key) == r.get(metric_key)]
            if not vals:
                return None
            return float(np.nanmean(vals)), float(np.nanmedian(vals)), len(vals)

        # High-signal metrics to surface (edit query)
        to_report = [
            "edit_attn_mass_to_edit_region_late_abs_delta",
            "edit_attn_mass_to_question_span_late_abs_delta",
            "edit_attn_entropy_late_abs_delta",
            "edit_attn_topk_jaccard_late",
        ]

        for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
            grp = [r for r in rows if r["normalized_group"] == grp_name and r.get("has_attention_edit")]
            if not grp:
                continue
            print(f"\n  {group_alias(grp_name)} (edit-query attention; n={len(grp)}):")
            for k in to_report:
                s = _summarize_metric(k, grp)
                if s is None:
                    continue
                mean_v, med_v, n = s
                print(f"    {k}: mean={mean_v:.6f}, median={med_v:.6f} (n={n})")

        # Export head-level rankings/tables
        if HAS_PANDAS and head_rows:
            head_df = pd.DataFrame(head_rows)
            head_df.to_csv(outdir / "attention_head_metrics.csv", index=False)
            print(f"\nSaved {outdir / 'attention_head_metrics.csv'}")

            # Ranked summary (per normalized_group, per head)
            if "source" in head_df.columns:
                # Focus on edit-query by default (most causal-tracing-relevant)
                use_df = head_df[head_df["source"] == "edit"].copy()
            else:
                use_df = head_df.copy()

            if not use_df.empty:
                summary = (
                    use_df.groupby(["normalized_group", "head"], as_index=False)
                    .agg(
                        mean_abs_mass_to_edit=("mass_to_edit_region_abs_delta", "mean"),
                        mean_abs_entropy=("entropy_abs_delta", "mean"),
                        mean_topk_jaccard_late=("topk_jaccard_late", "mean"),
                        mean_abs_delta_gold_logit=("abs_delta_gold_logit", "mean"),
                        n=("head", "size"),
                    )
                    .sort_values(["normalized_group", "mean_abs_mass_to_edit"], ascending=[True, False])
                )
                summary.to_csv(outdir / "attention_head_rankings.csv", index=False)
                print(f"Saved {outdir / 'attention_head_rankings.csv'}")

                # --- Visualizations for attention results ---
                if HAS_PLT:
                    try:
                        # Plot A: top heads by group (mean_abs_mass_to_edit)
                        groups = [g for g in ["core_bias", "identity_bias", "structural_context", "control"]
                                  if (summary["normalized_group"] == g).any()]
                        if groups:
                            ncols = 2
                            nrows = int(np.ceil(len(groups) / ncols))
                            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
                            axes = np.array(axes).reshape(-1)

                            for ax_i, g in enumerate(groups):
                                ax = axes[ax_i]
                                sub = summary[summary["normalized_group"] == g].head(15)
                                ax.barh(
                                    [f"H{int(h)}" for h in sub["head"]],
                                    sub["mean_abs_mass_to_edit"].to_numpy(),
                                    color="#4a90d9" if g != "control" else "#2ca02c",
                                    alpha=0.9,
                                )
                                ax.invert_yaxis()
                                ax.set_title(f"{group_alias(g)}: top heads (|Δ mass→edit|, late)", fontsize=11)
                                ax.set_xlabel("mean_abs_mass_to_edit", fontsize=10)
                                ax.grid(True, alpha=0.25, axis="x")

                            for j in range(ax_i + 1, len(axes)):
                                axes[j].axis("off")

                            fig.suptitle("Attention head rankings (edit-query summaries)", fontsize=13, y=1.02)
                            fig.tight_layout()
                            fig.savefig(outdir / "attention_head_rankings.png", dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print("  Saved attention_head_rankings.png")
                    except Exception as e:
                        print(f"  WARNING: attention ranking plot failed: {e}")

                # Head-feature probe: focal vs control using per-head edit mass deltas
                if HAS_SKLEARN:
                    try:
                        pivot = use_df.pivot_table(
                            index="pair_key",
                            columns="head",
                            values="mass_to_edit_region_delta",
                            aggfunc="mean",
                        )
                        # Labels from rows (pair_key -> group)
                        pk_to_group = {
                            r["pair_key"]: r["normalized_group"]
                            for r in rows
                            if r.get("has_attention_edit")
                        }
                        y = np.array(
                            [1 if pk_to_group.get(pk) in {"core_bias", "identity_bias", "structural_context"} else 0
                             for pk in pivot.index],
                            dtype=int,
                        )
                        # Need both classes
                        if len(np.unique(y)) == 2 and pivot.shape[0] >= 20:
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
                            top = np.argsort(np.abs(mean_coef))[::-1][:20]
                            probe_df = pd.DataFrame(
                                {
                                    "head": head_ids[top],
                                    "mean_abs_coef": np.abs(mean_coef[top]),
                                    "mean_coef": mean_coef[top],
                                }
                            )
                            probe_df.to_csv(outdir / "attention_head_probe_top_heads.csv", index=False)
                            print(f"Saved {outdir / 'attention_head_probe_top_heads.csv'} (LOO acc={acc:.3f})")
                        else:
                            print("  Skipping head-feature probe (insufficient class balance / n).")
                    except Exception as e:
                        print(f"  WARNING: head-feature probe failed: {e}")

    # ---------------------------------------------------------------------
    # ANALYSIS F: effect size conditional on token edit distance
    # ---------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("F. EFFECT SIZE CONDITIONAL ON TOKEN EDIT DISTANCE")
    print(f"{'='*70}")

    for grp_name in ["core_bias", "identity_bias", "structural_context", "control"]:
        grp = [r for r in rows if r["normalized_group"] == grp_name and r.get("token_edit_ratio", 0) > 0]
        if not grp:
            continue
        edit_sizes = [r["token_edit_ratio"] for r in grp]
        abs_dl = [r["abs_delta_gold_logit"] for r in grp]
        ne = [r["mean_norm_euclid_mid_late"] for r in grp]

        print(f"\n  {group_alias(grp_name)} (n={len(grp)}):")
        print(f"    Token edit ratio: mean={np.mean(edit_sizes):.4f}, median={np.median(edit_sizes):.4f}")

        if HAS_SCIPY and len(grp) > 5:
            r_logit, p_logit = sp_stats.spearmanr(edit_sizes, abs_dl)
            r_euclid, p_euclid = sp_stats.spearmanr(edit_sizes, ne)
            print(f"    Spearman(edit_ratio, |Δℓ|): r={r_logit:.3f}, p={p_logit:.4f}")
            print(f"    Spearman(edit_ratio, norm_euclid): r={r_euclid:.3f}, p={p_euclid:.4f}")

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    if HAS_PLT:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")

        plot_groups = [
            ("core_bias", "#d62728", "Core bias"),
            ("identity_bias", "#1f77b4", "Identity bias"),
            ("structural_context", "#9467bd", "Structural context"),
            ("control", "#2ca02c", "Control"),
        ]

        # Plot 1
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax_i, (metric_prefix, title) in enumerate([
            ("norm_euclid", "Normalized Euclidean Distance"),
            ("cos_dist", "Cosine Distance"),
        ]):
            ax = axes[ax_i]
            for grp_name, color, label in plot_groups:
                grp = [r for r in rows if r["normalized_group"] == grp_name]
                if not grp:
                    continue
                means = [np.mean([r.get(f"{metric_prefix}_L{l}", 0) for r in grp])
                         for l in range(n_layers)]
                sems = [np.std([r.get(f"{metric_prefix}_L{l}", 0) for r in grp]) / np.sqrt(len(grp))
                        for l in range(n_layers)]
                ls = list(range(n_layers))
                ax.plot(ls, means, color=color, label=label, linewidth=2)
                ax.fill_between(
                    ls,
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    color=color,
                    alpha=0.15,
                )

            ax.set_xlabel("Layer (transformer block)", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Normalized Representational Change by Group", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "normalized_distance_by_group.png", dpi=150, bbox_inches="tight")
        print("  Saved normalized_distance_by_group.png")
        plt.close(fig)

        # Plot 2
        edit_any = [r for r in rows if r.get("has_edit_repr")]
        if edit_any:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            for grp_name, color, label in plot_groups:
                grp = [r for r in edit_any if r["normalized_group"] == grp_name]
                if not grp:
                    continue
                final_ne = np.mean([r["mean_norm_euclid_mid_late"] for r in grp])
                edit_ne = np.mean([r["edit_mean_norm_euclid_mid_late"] for r in grp])
                ax.scatter(final_ne, edit_ne, color=color, s=80, label=f"{label} (n={len(grp)})")

            lims = ax.get_xlim(), ax.get_ylim()
            max_lim = max(lims[0][1], lims[1][1], 1e-6)
            ax.plot([0, max_lim], [0, max_lim], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("Final-token norm euclid (mid→late)", fontsize=11)
            ax.set_ylabel("Edit-token norm euclid (mid→late)", fontsize=11)
            ax.set_title("Edit-token vs Final-token Change by Group", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "edit_vs_final_by_group.png", dpi=150)
            print("  Saved edit_vs_final_by_group.png")
            plt.close(fig)

        # Plot 3
        focal_types = sorted(set(
            r["normalized_attribute"]
            for r in rows
            if r["normalized_group"] in {"core_bias", "identity_bias", "structural_context"}
        ))
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        cmap = plt.get_cmap("tab10")
        for idx, intervention in enumerate(focal_types[:10]):
            grp = [r for r in rows if r["normalized_attribute"] == intervention]
            if not grp:
                continue
            means = [np.mean([r.get(f"norm_euclid_L{l}", 0) for r in grp]) for l in range(n_layers)]
            ax.plot(range(n_layers), means, color=cmap(idx), label=f"{intervention} (n={len(grp)})", linewidth=2)

        ctrl_grp = [r for r in rows if r["normalized_group"] == "control"]
        if ctrl_grp:
            means = [np.mean([r.get(f"norm_euclid_L{l}", 0) for r in ctrl_grp]) for l in range(n_layers)]
            ax.plot(range(n_layers), means, color="gray", linestyle="--", linewidth=2, label=f"control (n={len(ctrl_grp)})")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Normalized Euclidean Distance", fontsize=11)
        ax.set_title("Normalized Euclidean by Intervention Type", fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "norm_euclid_by_intervention.png", dpi=150)
        print("  Saved norm_euclid_by_intervention.png")
        plt.close(fig)

        # Plot 4
        for focal_group in ["core_bias", "identity_bias", "structural_context"]:
            for ctrl_sub, color in [("irrelevant_surface", "#4a90d9"), ("neutral_rework", "#e07b54")]:
                diffs = paired_diffs[focal_group][ctrl_sub]
                dl = diffs["logit"]
                dne = diffs["norm_euclid"]
                if not dl:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                ax.hist(dl, bins=30, color=color, edgecolor="white", alpha=0.8)
                ax.axvline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(np.mean(dl), color="red", linestyle="-", linewidth=2, label=f"mean={np.mean(dl):.4f}")
                ax.set_xlabel("|Δℓ|_focal − |Δℓ|_ctrl", fontsize=11)
                ax.set_ylabel("Count (questions)", fontsize=11)
                ax.set_title(f"{group_alias(focal_group)} vs {ctrl_sub}: logit", fontsize=12)
                ax.legend(fontsize=10)

                ax = axes[1]
                ax.hist(dne, bins=30, color=color, edgecolor="white", alpha=0.8)
                ax.axvline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(np.mean(dne), color="red", linestyle="-", linewidth=2, label=f"mean={np.mean(dne):.6f}")
                ax.set_xlabel("norm_euclid_focal − norm_euclid_ctrl", fontsize=11)
                ax.set_ylabel("Count (questions)", fontsize=11)
                ax.set_title(f"{group_alias(focal_group)} vs {ctrl_sub}: repr", fontsize=12)
                ax.legend(fontsize=10)

                fig.suptitle(f"Within-Question Paired Diff: {group_alias(focal_group)} vs {ctrl_sub}", fontsize=13)
                fig.tight_layout()
                fname = f"paired_diffs_{focal_group}_{ctrl_sub}.png"
                fig.savefig(outdir / fname, dpi=150)
                print(f"  Saved {fname}")
                plt.close(fig)

        # Plot 5
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        strata = [
            ("All", lambda r: True),
            ("Orig correct", lambda r: r["orig_correct"]),
            (f"Correct + margin>{margin_threshold}", lambda r: r["orig_correct"] and r["orig_margin"] > margin_threshold),
        ]
        x_pos = np.arange(len(strata))
        plot_order = ["core_bias", "identity_bias", "structural_context", "control"]
        colors = {
            "core_bias": "#d62728",
            "identity_bias": "#1f77b4",
            "structural_context": "#9467bd",
            "control": "#2ca02c",
        }

        # Only plot groups that actually appear (avoid misleading zero-height bars).
        active_groups = [
            g for g in plot_order if any(r["normalized_group"] == g for r in rows)
        ]
        if not active_groups:
            active_groups = plot_order

        width = min(0.22, 0.8 / max(len(active_groups), 1))
        center = (len(active_groups) - 1) / 2.0

        for offset, grp_name in enumerate(active_groups):
            vals = []
            for _, filt in strata:
                grp = [r for r in rows if r["normalized_group"] == grp_name and filt(r)]
                vals.append(np.mean([r["abs_delta_gold_logit"] for r in grp]) if grp else np.nan)
            ax.bar(x_pos + (offset - center) * width, vals, width,
                   label=group_alias(grp_name), color=colors[grp_name], alpha=0.85)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in strata], fontsize=10)
        ax.set_ylabel("Mean |Δ gold logit|", fontsize=11)
        ax.set_title("Behavioral Sensitivity by Confidence Stratum", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "sensitivity_by_confidence.png", dpi=150)
        print("  Saved sensitivity_by_confidence.png")
        plt.close(fig)

    print(f"\n{'='*70}")
    print("STAGE 1 ADAPTED ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {outdir}")
    if HAS_PANDAS:
        print("  Full metrics: stage1_metrics_adapted.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 analysis adapted for newer counterfactual schema"
    )
    parser.add_argument("--extractions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="stage1_results_adapted")
    parser.add_argument("--margin_threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_analysis(args.extractions, args.output_dir, args.margin_threshold)


if __name__ == "__main__":
    main()