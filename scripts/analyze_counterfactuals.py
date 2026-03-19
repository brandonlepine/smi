"""
MedQA Counterfactual Analysis (v4.1 companion)
================================================
Flattens the nested JSON output from generate_counterfactuals_v4.py into
tidy dataframes suitable for analysis in Python (pandas) or R.

Outputs:
  1. variants.csv — one row per variant (main analysis table)
  2. questions.csv — one row per question with original demographics
  3. cue_interactions.csv — one row per question with clinical cue flags
  4. summary_report.txt — human-readable summary statistics

Usage:
  # Basic: flatten JSON to CSVs
  python analyze_counterfactuals.py cf_v4_tier1.json

  # Specify output directory
  python analyze_counterfactuals.py cf_v4_tier1.json --outdir analysis_output/

  # Merge multiple tier files
  python analyze_counterfactuals.py cf_v4_tier1.json cf_v4_tier2.json --outdir merged/

  # Just print summary, no CSV output
  python analyze_counterfactuals.py cf_v4_tier1.json --summary_only

  # Validate only (no CSV, detailed warnings)
  python analyze_counterfactuals.py cf_v4_tier1.json --validate
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from io import StringIO

# Try pandas — fall back to CSV-only mode if unavailable
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import csv


# ---------------------------------------------------------------------------
# Flatten JSON → row dicts
# ---------------------------------------------------------------------------

def flatten_variants(results: list[dict]) -> list[dict]:
    """Flatten the nested JSON into one row per variant.

    Each row contains:
      - question-level fields (question_id, original question text, gold answer)
      - original demographics
      - pass metadata
      - all variant-level fields from the taxonomy
    """
    rows = []
    for r in results:
        qid = r.get("question_id", "")
        orig = r.get("original", {})
        pass_meta = r.get("pass_metadata", {})
        cf = r.get("counterfactuals", {})
        orig_demo = cf.get("original_demographics", {})

        # Question-level fields
        q_base = {
            "question_id": qid,
            "original_question": orig.get("question", ""),
            "gold_answer": orig.get("answer", ""),
            "gold_answer_idx": orig.get("answer_idx", ""),
            # Original demographics
            "orig_gender": orig_demo.get("gender"),
            "orig_age": orig_demo.get("age"),
            "orig_race_ethnicity": orig_demo.get("race_ethnicity"),
            "orig_sexual_orientation": orig_demo.get("sexual_orientation"),
            "orig_relationship_status": orig_demo.get("relationship_status"),
            "orig_has_social_context": bool(orig_demo.get("social_context_cues")),
            "orig_has_names": bool(orig_demo.get("names_present")),
            "orig_pronouns": "|".join(orig_demo.get("pronouns_used", [])),
            # Pass metadata
            "pass_id": pass_meta.get("pass_id", ""),
            "prompt_variant": pass_meta.get("prompt_variant", ""),
            "model": pass_meta.get("model", ""),
            "tiers": "|".join(str(t) for t in pass_meta.get("tiers", [])),
        }

        variants = cf.get("variants", [])
        if not isinstance(variants, list):
            continue

        for v in variants:
            if not isinstance(v, dict):
                continue

            row = {**q_base}

            # All variant-level fields
            for field in [
                "variant_id", "intervention_type", "intervention_family",
                "semantic_class", "analysis_bucket", "ladder_applicable",
                "edit_scope", "edit_strength", "identity_explicitness",
                "attribute_value_original", "attribute_value_counterfactual",
                "medical_relevance", "social_bias_salience",
                "counterfactual_validity", "clinical_coherence",
                "gold_answer_invariance", "prior_shift_expected",
                "annotation_confidence", "rationale",
            ]:
                row[field] = v.get(field)

            # Boolean: has text?
            row["has_text"] = v.get("text") is not None
            row["reason_if_null"] = v.get("reason_if_null")
            row["is_identical"] = (
                "identical" in str(v.get("reason_if_null", "")).lower()
            )

            # Evidence spans as pipe-delimited string
            spans = v.get("evidence_spans", [])
            if spans:
                row["evidence_spans"] = "|".join(
                    f"{s.get('role', '?')}:{s.get('span', '?')}"
                    for s in spans if isinstance(s, dict)
                )
            else:
                row["evidence_spans"] = ""

            # Uncertainty flags as pipe-delimited
            flags = v.get("uncertainty_flags", [])
            row["uncertainty_flags"] = "|".join(flags) if flags else ""
            row["n_uncertainty_flags"] = len(flags)

            row["notes"] = v.get("notes")

            # The full counterfactual text (can be large — optional)
            row["counterfactual_text"] = v.get("text")

            rows.append(row)

    return rows


def flatten_questions(results: list[dict]) -> list[dict]:
    """One row per question with original demographics and cue interactions."""
    rows = []
    for r in results:
        qid = r.get("question_id", "")
        orig = r.get("original", {})
        cf = r.get("counterfactuals", {})
        orig_demo = cf.get("original_demographics", {})
        cues = cf.get("clinical_cue_interactions", {})

        n_variants = 0
        n_valid = 0
        n_null = 0
        variants = cf.get("variants", [])
        if isinstance(variants, list):
            n_variants = len(variants)
            for v in variants:
                if isinstance(v, dict):
                    if v.get("text") is not None:
                        n_valid += 1
                    else:
                        n_null += 1

        rows.append({
            "question_id": qid,
            "original_question": orig.get("question", ""),
            "gold_answer": orig.get("answer", ""),
            "gold_answer_idx": orig.get("answer_idx", ""),
            "orig_gender": orig_demo.get("gender"),
            "orig_age": orig_demo.get("age"),
            "orig_race_ethnicity": orig_demo.get("race_ethnicity"),
            "orig_sexual_orientation": orig_demo.get("sexual_orientation"),
            "orig_relationship_status": orig_demo.get("relationship_status"),
            "orig_social_context_cues": "|".join(
                orig_demo.get("social_context_cues", [])
            ),
            "orig_names": "|".join(orig_demo.get("names_present", [])),
            "orig_pronouns": "|".join(orig_demo.get("pronouns_used", [])),
            # Clinical cue interactions
            "has_phenotypic_cues": bool(cues.get("phenotypic_cues")),
            "has_sex_specific_features": bool(
                cues.get("sex_specific_clinical_features")
            ),
            "has_sexual_behavior_cues": bool(cues.get("sexual_behavior_cues")),
            "has_family_history": bool(cues.get("family_history_patterns")),
            "has_epidemiologic_assoc": bool(
                cues.get("epidemiologic_associations")
            ),
            "has_social_context_interactions": bool(
                cues.get("social_context_clinical_interactions")
            ),
            # Variant counts
            "n_variants": n_variants,
            "n_with_text": n_valid,
            "n_null": n_null,
        })

    return rows


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary(variant_rows: list[dict], question_rows: list[dict]) -> str:
    """Generate a human-readable summary report."""
    buf = StringIO()
    p = lambda *a, **kw: print(*a, **kw, file=buf)

    n_q = len(question_rows)
    n_v = len(variant_rows)

    p("=" * 70)
    p(f"  COUNTERFACTUAL DATASET SUMMARY")
    p(f"  {n_q} questions, {n_v} variants")
    p("=" * 70)

    # --- Question-level ---
    p(f"\n--- Questions ---")
    gender_dist = Counter(r["orig_gender"] for r in question_rows)
    p(f"  Gender: {dict(gender_dist.most_common())}")
    p(f"  Has age: {sum(1 for r in question_rows if r['orig_age'] is not None)}/{n_q}")
    p(f"  Has race: {sum(1 for r in question_rows if r['orig_race_ethnicity'] is not None)}/{n_q}")
    p(f"  Has orientation: {sum(1 for r in question_rows if r['orig_sexual_orientation'] is not None)}/{n_q}")
    p(f"  Has sex-specific features: {sum(1 for r in question_rows if r['has_sex_specific_features'])}/{n_q}")
    p(f"  Has epidemiologic assoc: {sum(1 for r in question_rows if r['has_epidemiologic_assoc'])}/{n_q}")

    avg_variants = sum(r["n_variants"] for r in question_rows) / max(n_q, 1)
    avg_text = sum(r["n_with_text"] for r in question_rows) / max(n_q, 1)
    p(f"  Avg variants/question: {avg_variants:.1f} ({avg_text:.1f} with text)")

    # --- Variant-level ---
    p(f"\n--- Variants ({n_v} total) ---")

    has_text = sum(1 for r in variant_rows if r["has_text"])
    is_identical = sum(1 for r in variant_rows if r["is_identical"])
    p(f"  With text: {has_text} ({100*has_text/max(n_v,1):.1f}%)")
    p(f"  Null: {n_v - has_text} ({is_identical} identical to original)")

    # Per-field distributions
    for field in ["analysis_bucket", "intervention_family", "intervention_type",
                   "medical_relevance", "social_bias_salience",
                   "edit_strength", "identity_explicitness",
                   "counterfactual_validity", "clinical_coherence",
                   "gold_answer_invariance", "prior_shift_expected"]:
        ctr = Counter(r.get(field, "?") for r in variant_rows)
        p(f"\n  {field}:")
        for k, v in ctr.most_common():
            p(f"    {k}: {v} ({100*v/max(n_v,1):.1f}%)")

    # Confidence stats
    confs = [r["annotation_confidence"] for r in variant_rows
             if r.get("annotation_confidence") is not None]
    if confs:
        p(f"\n  annotation_confidence:")
        p(f"    mean={sum(confs)/len(confs):.3f} "
          f"median={sorted(confs)[len(confs)//2]:.3f} "
          f"min={min(confs):.2f} max={max(confs):.2f}")

    # --- Analysis classes ---
    c1 = c2 = c3 = 0
    for r in variant_rows:
        if not r["has_text"]:
            if not r["is_identical"]:
                c3 += 1
            continue
        val = r.get("counterfactual_validity")
        coh = r.get("clinical_coherence")
        rel = r.get("medical_relevance")
        inv = r.get("gold_answer_invariance")
        if val == "valid" and coh == "preserved" \
                and inv in ("invariant", "likely_invariant") \
                and rel == "irrelevant":
            c1 += 1
        elif val in ("valid", "questionable") \
                and coh in ("preserved", "weakened") \
                and rel in ("epidemiologic", "mechanistically_causal"):
            c2 += 1
        elif val == "invalid" or coh == "broken":
            c3 += 1

    p(f"\n--- Analysis Classes ---")
    p(f"  Class 1 (clean, main bias analysis):   {c1}")
    p(f"  Class 2 (medically relevant edits):    {c2}")
    p(f"  Class 3 (invalid/broken):              {c3}")
    p(f"  Identical (skipped):                   {is_identical}")
    p(f"  Unclassified:                          "
      f"{has_text - c1 - c2 - (c3 - (n_v - has_text - is_identical))}")

    # --- Cross-tab: medical_relevance × social_bias_salience ---
    cross = Counter()
    for r in variant_rows:
        if not r["has_text"]:
            continue
        cross[(r.get("medical_relevance", "?"),
               r.get("social_bias_salience", "?"))] += 1

    p(f"\n--- medical_relevance × social_bias_salience (text variants only) ---")
    p(f"  {'':>25s}  low    mod    high")
    for mr in ["irrelevant", "epidemiologic", "mechanistically_causal", "ambiguous"]:
        row = [cross.get((mr, s), 0) for s in ["low", "moderate", "high"]]
        p(f"  {mr:>25s}  {row[0]:<6d} {row[1]:<6d} {row[2]:<6d}")

    # --- Ladder coverage ---
    LADDER_TYPES = {"sex", "race_ethnicity", "pronoun", "name",
                    "sexual_orientation", "kinship_role"}
    ladder_cov = {}
    for r in variant_rows:
        if not r["has_text"]:
            continue
        it = r.get("intervention_type", "?")
        s = r.get("edit_strength", "?")
        if it in LADDER_TYPES:
            ladder_cov.setdefault(it, Counter())[s] += 1

    p(f"\n--- Intervention Ladder Coverage ---")
    for it in sorted(ladder_cov):
        sc = ladder_cov[it]
        parts = [f"{s}={sc.get(s, 0)}" for s in ["minimal", "moderate", "strong"]]
        p(f"  {it}: {', '.join(parts)}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_csv_pandas(rows: list[dict], path: Path, exclude_text: bool = False):
    """Write rows to CSV using pandas."""
    df = pd.DataFrame(rows)
    if exclude_text and "counterfactual_text" in df.columns:
        df = df.drop(columns=["counterfactual_text"])
    if exclude_text and "original_question" in df.columns:
        df = df.drop(columns=["original_question"])
    df.to_csv(path, index=False)
    return df


def write_csv_stdlib(rows: list[dict], path: Path, exclude_text: bool = False):
    """Write rows to CSV using stdlib csv module (no pandas)."""
    if not rows:
        return
    keys = list(rows[0].keys())
    if exclude_text:
        keys = [k for k in keys if k not in ("counterfactual_text", "original_question")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(rows, path, exclude_text=False):
    if HAS_PANDAS:
        return write_csv_pandas(rows, path, exclude_text)
    else:
        write_csv_stdlib(rows, path, exclude_text)
        return None


# ---------------------------------------------------------------------------
# Validation (reuses logic from generator, standalone here)
# ---------------------------------------------------------------------------

VALID_ENUMS = {
    "intervention_type": {
        "sex", "age", "race_ethnicity", "pronoun", "name",
        "pregnancy_status", "reproductive_status",
        "neutral_rework", "irrelevant_surface",
        "sexual_orientation", "gender_identity", "kinship_role",
        "honorific", "disability_identity",
        "insurance_status", "housing_status", "occupation",
        "marital_status", "socioeconomic_status", "family_structure",
        "nationality", "religion",
    },
    "analysis_bucket": {"core_bias", "identity_bias", "structural_context", "control"},
    "medical_relevance": {"irrelevant", "epidemiologic", "mechanistically_causal", "ambiguous"},
    "social_bias_salience": {"low", "moderate", "high"},
    "counterfactual_validity": {"valid", "questionable", "invalid"},
    "clinical_coherence": {"preserved", "weakened", "broken"},
    "gold_answer_invariance": {"invariant", "likely_invariant", "uncertain", "likely_changed"},
    "edit_strength": {"minimal", "moderate", "strong", "single"},
}


def validate_rows(rows: list[dict]) -> list[str]:
    """Validate flattened variant rows."""
    warnings = []
    for i, r in enumerate(rows):
        qid = r.get("question_id", "?")
        vid = r.get("variant_id", "?")

        for field, valid in VALID_ENUMS.items():
            val = r.get(field)
            if val is not None and val not in valid:
                warnings.append(f"{qid}/{vid}: invalid {field}={val!r}")

        # Control checks
        itype = r.get("intervention_type", "")
        if itype in ("neutral_rework", "irrelevant_surface"):
            if r.get("medical_relevance") != "irrelevant":
                warnings.append(f"{qid}/{vid}: control has medical_relevance={r.get('medical_relevance')!r}")
            if r.get("social_bias_salience") != "low":
                warnings.append(f"{qid}/{vid}: control has social_bias_salience={r.get('social_bias_salience')!r}")

        # Explicitness checks
        if itype == "name" and r.get("identity_explicitness") != "implicit":
            warnings.append(f"{qid}/{vid}: name should be implicit, got {r.get('identity_explicitness')!r}")
        if itype == "pronoun" and r.get("identity_explicitness") != "linguistic":
            warnings.append(f"{qid}/{vid}: pronoun should be linguistic, got {r.get('identity_explicitness')!r}")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flatten and analyze MedQA counterfactual JSON output"
    )
    parser.add_argument("input_files", type=str, nargs="+",
                        help="JSON file(s) from generate_counterfactuals_v4.py")
    parser.add_argument("--outdir", type=str, default="analysis_output",
                        help="Output directory for CSVs and report")
    parser.add_argument("--summary_only", action="store_true",
                        help="Print summary only, no CSV output")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks and print warnings")
    parser.add_argument("--no_text", action="store_true",
                        help="Exclude full question/variant text from CSVs "
                             "(much smaller files)")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "parquet", "both"],
                        help="Output format (parquet requires pandas+pyarrow)")
    args = parser.parse_args()

    # --- Load and merge input files ---
    all_results = []
    for fpath in args.input_files:
        print(f"Loading {fpath}...")
        with open(fpath) as f:
            data = json.load(f)
        print(f"  {len(data)} questions")
        all_results.extend(data)

    print(f"\nTotal: {len(all_results)} questions across {len(args.input_files)} file(s)")

    # --- Flatten ---
    print("Flattening variants...")
    variant_rows = flatten_variants(all_results)
    question_rows = flatten_questions(all_results)
    print(f"  {len(variant_rows)} variant rows")
    print(f"  {len(question_rows)} question rows")

    # --- Summary ---
    report = generate_summary(variant_rows, question_rows)
    print(report)

    if args.summary_only:
        return

    # --- Validate ---
    if args.validate:
        print("\n--- Validation ---")
        warnings = validate_rows(variant_rows)
        if warnings:
            print(f"  {len(warnings)} warnings:")
            for w in warnings[:100]:
                print(f"    {w}")
            if len(warnings) > 100:
                print(f"    ... +{len(warnings) - 100} more")
        else:
            print("  ✓ All rows pass validation.")
        if args.validate and not args.outdir:
            return

    # --- Write outputs ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Variants CSV
    vpath = outdir / "variants.csv"
    print(f"\nWriting {vpath}...")
    write_csv(variant_rows, vpath, exclude_text=args.no_text)

    # Questions CSV
    qpath = outdir / "questions.csv"
    print(f"Writing {qpath}...")
    write_csv(question_rows, qpath, exclude_text=args.no_text)

    # Parquet if requested
    if args.format in ("parquet", "both") and HAS_PANDAS:
        try:
            vdf = pd.DataFrame(variant_rows)
            qdf = pd.DataFrame(question_rows)
            vdf.to_parquet(outdir / "variants.parquet", index=False)
            qdf.to_parquet(outdir / "questions.parquet", index=False)
            print(f"Writing parquet files...")
        except Exception as e:
            print(f"  Parquet write failed (need pyarrow): {e}")

    # Summary report
    rpath = outdir / "summary_report.txt"
    with open(rpath, "w") as f:
        f.write(report)
    print(f"Writing {rpath}")

    # --- Quick size check ---
    vsize = vpath.stat().st_size / 1024
    qsize = qpath.stat().st_size / 1024
    print(f"\nFile sizes: variants.csv={vsize:.0f}KB, questions.csv={qsize:.0f}KB")
    print(f"\n→ Output in {outdir}/")

    # --- Print R loading snippet ---
    print(f"""
─── R loading snippet ───────────────────────────────────
library(tidyverse)

variants <- read_csv("{vpath}")
questions <- read_csv("{qpath}")

# Filter to Class 1 (clean counterfactuals, main bias analysis)
class1 <- variants %>%
  filter(
    has_text == TRUE,
    counterfactual_validity == "valid",
    clinical_coherence == "preserved",
    gold_answer_invariance %in% c("invariant", "likely_invariant"),
    medical_relevance == "irrelevant"
  )

# Cross-tab: medical_relevance × social_bias_salience
variants %>%
  filter(has_text) %>%
  count(medical_relevance, social_bias_salience) %>%
  pivot_wider(names_from = social_bias_salience, values_from = n, values_fill = 0)

# Ladder dose-response: group by intervention_type and edit_strength
variants %>%
  filter(has_text, ladder_applicable == TRUE) %>%
  count(intervention_type, edit_strength) %>%
  pivot_wider(names_from = edit_strength, values_from = n, values_fill = 0)

# By analysis bucket
variants %>%
  filter(has_text) %>%
  count(analysis_bucket, counterfactual_validity) %>%
  pivot_wider(names_from = counterfactual_validity, values_from = n, values_fill = 0)
──────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()