

"""
Multi-Pass Annotation Reconciliation
=======================================
Takes 2+ annotation pass files, computes inter-pass agreement on key
fields, and produces a tiered benchmark where:
 
  Tier 1 (high confidence core):
    - Labels agree across all passes on critical fields
    - Mean annotation_confidence >= 0.8
    - No uncertainty flags on critical fields
    - edit_locality = minimal or sentence_level
 
  Tier 2 (plausible but ambiguous):
    - Partial agreement OR medium confidence
    - Some uncertainty flags
 
  Tier 3 (stress test / excluded):
    - Disagreement on critical fields
    - Low confidence
    - Invalid or broken
 
Usage:
  # Run two annotation passes first:
  python generate_counterfactuals.py --pass_id pass_A --prompt_variant A --output pass_A.json
  python generate_counterfactuals.py --pass_id pass_B --prompt_variant B --output pass_B.json
 
  # Or with a different model:
  python generate_counterfactuals.py --pass_id pass_C --prompt_variant A --model gpt-4o-mini --output pass_C.json
 
  # Reconcile:
  python reconcile_annotations.py pass_A.json pass_B.json --output benchmark.json
  python reconcile_annotations.py pass_A.json pass_B.json pass_C.json --output benchmark.json
"""
 
import json
import argparse
import statistics
from collections import Counter
from pathlib import Path
 
 
# ---------------------------------------------------------------------------
# Critical fields for agreement (these determine tier assignment)
# ---------------------------------------------------------------------------
 
# Fields where disagreement is most consequential
CRITICAL_FIELDS = [
    "counterfactual_validity",
    "target_attribute_role",
    "gold_answer_invariance",
]
 
# Fields where disagreement is informative but less decisive
SECONDARY_FIELDS = [
    "clinical_coherence",
    "prior_shift_expected",
    "edit_locality",
]
 
# Uncertainty flags that block Tier 1 on critical fields
CRITICAL_UNCERTAINTY = {
    "uncertain_medical_relevance",
    "uncertain_answer_invariance",
    "uncertain_causal_mechanism",
}
 
 
# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------
 
def load_passes(filepaths: list[str]) -> list[list[dict]]:
    """Load multiple pass files."""
    passes = []
    for fp in filepaths:
        with open(fp) as f:
            data = json.load(f)
        passes.append(data)
        print(f"Loaded {len(data)} records from {fp}")
    return passes
 
 
def align_passes(passes: list[list[dict]]) -> dict:
    """Align records by question_id across passes.
 
    Returns: {question_id: [record_pass1, record_pass2, ...]}
    Only includes questions present in ALL passes.
    """
    # Index each pass by question_id
    indexed = []
    for p in passes:
        idx = {r["question_id"]: r for r in p}
        indexed.append(idx)
 
    # Intersect
    common_ids = set(indexed[0].keys())
    for idx in indexed[1:]:
        common_ids &= set(idx.keys())
 
    aligned = {}
    for qid in sorted(common_ids):
        aligned[qid] = [idx[qid] for idx in indexed]
 
    print(f"\nAligned {len(aligned)} questions across {len(passes)} passes")
    return aligned
 
 
# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------
 
def get_variant_data(record: dict, category: str, label: str) -> dict | None:
    """Extract a specific variant's annotation from a record."""
    variants = record.get("counterfactuals", {}).get("variants", {})
    cat_data = variants.get(category, {})
    vdata = cat_data.get(label, {})
    return vdata if isinstance(vdata, dict) else None
 
 
def compute_field_agreement(values: list) -> dict:
    """Compute agreement stats for a list of values across passes."""
    if not values:
        return {"unanimous": False, "majority": None, "values": []}
 
    counter = Counter(values)
    most_common_val, most_common_count = counter.most_common(1)[0]
    n = len(values)
 
    return {
        "unanimous": most_common_count == n,
        "majority": most_common_val,
        "majority_frac": most_common_count / n,
        "values": values,
        "n_unique": len(counter),
    }
 
 
def reconcile_variant(variant_data_per_pass: list[dict | None]) -> dict:
    """Reconcile annotations for a single variant across passes.
 
    Returns a reconciled record with agreement metadata.
    """
    active = [v for v in variant_data_per_pass if v is not None]
    n_passes = len(variant_data_per_pass)
    n_active = len(active)
 
    if n_active == 0:
        return {"tier": "excluded", "reason": "no_data_in_any_pass"}
 
    # --- Text agreement (did all passes produce text or null?) ---
    text_states = []
    for v in active:
        t = v.get("text")
        reason = str(v.get("reason_if_null", "")).lower()
        if t is None and "identical" in reason:
            text_states.append("identical")
        elif t is None:
            text_states.append("null")
        else:
            text_states.append("has_text")
 
    text_agreement = compute_field_agreement(text_states)
 
    # If passes disagree on whether the variant even exists, it's ambiguous
    if not text_agreement["unanimous"] and text_agreement["n_unique"] > 1:
        if "null" in text_states and "has_text" in text_states:
            # Some passes say invalid, others produce text
            pass  # handled below in field agreement
 
    # --- Field agreement on critical and secondary fields ---
    field_agreements = {}
    for field in CRITICAL_FIELDS + SECONDARY_FIELDS:
        values = [v.get(field) for v in active if v.get(field) is not None]
        field_agreements[field] = compute_field_agreement(values)
 
    # --- Confidence aggregation ---
    confidences = [v.get("annotation_confidence", 0.5) for v in active
                   if v.get("annotation_confidence") is not None]
    mean_confidence = statistics.mean(confidences) if confidences else 0.5
 
    # --- Uncertainty flag aggregation ---
    all_flags = set()
    for v in active:
        for flag in v.get("uncertainty_flags", []):
            all_flags.add(flag)
 
    critical_uncertainty = all_flags & CRITICAL_UNCERTAINTY
 
    # --- Evidence span collection ---
    all_evidence = []
    seen_spans = set()
    for v in active:
        for span in v.get("evidence_spans", []):
            key = (span.get("span", ""), span.get("role", ""))
            if key not in seen_spans:
                all_evidence.append(span)
                seen_spans.add(key)
 
    # --- Rationale collection ---
    rationales = [v.get("rationale", "") for v in active
                  if v.get("rationale")]
 
    # --- Tier assignment ---
    tier, tier_reason = assign_tier(
        text_agreement=text_agreement,
        field_agreements=field_agreements,
        mean_confidence=mean_confidence,
        critical_uncertainty=critical_uncertainty,
        active=active,
    )
 
    # --- Build reconciled labels (use majority vote) ---
    reconciled_labels = {}
    for field in CRITICAL_FIELDS + SECONDARY_FIELDS:
        fa = field_agreements[field]
        reconciled_labels[field] = fa.get("majority")
 
    # Use text from the first pass that has it (they should be very similar)
    reconciled_text = None
    for v in active:
        if v.get("text") is not None:
            reconciled_text = v["text"]
            break
 
    return {
        "tier": tier,
        "tier_reason": tier_reason,
        "reconciled_text": reconciled_text,
        "reconciled_labels": reconciled_labels,
        "mean_confidence": round(mean_confidence, 3),
        "n_passes": n_passes,
        "n_active": n_active,
        "text_agreement": text_agreement["unanimous"],
        "critical_field_agreement": {
            f: field_agreements[f]["unanimous"] for f in CRITICAL_FIELDS
        },
        "secondary_field_agreement": {
            f: field_agreements[f]["unanimous"] for f in SECONDARY_FIELDS
        },
        "all_uncertainty_flags": sorted(all_flags),
        "critical_uncertainty_flags": sorted(critical_uncertainty),
        "evidence_spans": all_evidence,
        "rationales": rationales,
        "per_pass_labels": [
            {field: v.get(field) for field in CRITICAL_FIELDS + SECONDARY_FIELDS +
             ["annotation_confidence", "uncertainty_flags"]}
            for v in active
        ],
    }
 
 
def assign_tier(text_agreement, field_agreements, mean_confidence,
                critical_uncertainty, active) -> tuple[str, str]:
    """Assign tier based on agreement, confidence, and uncertainty."""
 
    # Check if all passes say identical or null-invalid
    if text_agreement["majority"] in ("identical",):
        return "identical", "all passes agree variant is identical to original"
 
    if text_agreement["majority"] == "null":
        return "tier3", "all/most passes say edit is invalid"
 
    # Check critical field unanimity
    critical_unanimous = all(
        field_agreements[f]["unanimous"] for f in CRITICAL_FIELDS
    )
 
    # Check what the agreed-upon labels are
    agreed_validity = field_agreements["counterfactual_validity"].get("majority")
    agreed_coherence = field_agreements["clinical_coherence"].get("majority")
    agreed_role = field_agreements["target_attribute_role"].get("majority")
    agreed_invariance = field_agreements["gold_answer_invariance"].get("majority")
 
    # --- Tier 3: invalid or broken ---
    if agreed_validity == "invalid" or agreed_coherence == "broken":
        return "tier3", "agreed invalid/broken"
 
    # --- Tier 1: high-confidence core ---
    if (critical_unanimous
            and mean_confidence >= 0.8
            and not critical_uncertainty
            and agreed_validity == "valid"
            and agreed_coherence == "preserved"):
        # Further check: role must be appropriate for bias analysis
        if agreed_role in ("irrelevant", "socially_loaded"):
            if agreed_invariance in ("invariant", "likely_invariant"):
                return "tier1", "unanimous valid + high confidence + answer invariant"
 
        # Tier 1 for medically relevant items (separate analysis track)
        if agreed_role in ("epidemiologic", "mechanistically_causal"):
            return "tier1_medical", ("unanimous labels + high confidence + "
                                     "medically relevant (separate analysis)")
 
    # --- Tier 2: plausible but ambiguous ---
    # Majority agreement on critical fields
    majority_agreement = all(
        field_agreements[f].get("majority_frac", 0) >= 0.5
        for f in CRITICAL_FIELDS
    )
 
    if majority_agreement and agreed_validity in ("valid", "questionable"):
        if mean_confidence >= 0.5:
            return "tier2", "majority agreement + medium confidence"
 
    # --- Fallback ---
    if mean_confidence < 0.5:
        return "tier3", "low confidence across passes"
 
    return "tier2", "partial agreement or edge case"
 
 
# ---------------------------------------------------------------------------
# Full reconciliation
# ---------------------------------------------------------------------------
 
CATEGORIES = {
    "gender": ["male", "female", "neutral"],
    "age": ["young_adult", "middle_aged", "elderly"],
    "race_ethnicity": ["White", "Black/African American", "Hispanic/Latino",
                       "Asian", "no_race_specified"],
    "control": ["neutral_rework", "irrelevant_surface"],
}
 
 
def reconcile_all(aligned: dict) -> list[dict]:
    """Reconcile all questions and variants across passes."""
    results = []
 
    for qid, pass_records in aligned.items():
        # Use the original from the first pass
        original = pass_records[0]["original"]
 
        # Reconcile clinical_cue_interactions (union across passes)
        all_cues = {}
        for pr in pass_records:
            cci = pr.get("counterfactuals", {}).get("clinical_cue_interactions", {})
            for field, cues in cci.items():
                if field not in all_cues:
                    all_cues[field] = set()
                for c in (cues or []):
                    all_cues[field].add(c)
        merged_cues = {k: sorted(v) for k, v in all_cues.items()}
 
        # Reconcile each variant
        reconciled_variants = {}
        for category, labels in CATEGORIES.items():
            reconciled_variants[category] = {}
            for label in labels:
                variant_per_pass = [
                    get_variant_data(pr, category, label)
                    for pr in pass_records
                ]
                reconciled_variants[category][label] = reconcile_variant(
                    variant_per_pass
                )
 
        results.append({
            "question_id": qid,
            "original": original,
            "clinical_cue_interactions": merged_cues,
            "variants": reconciled_variants,
        })
 
    return results
 
 
# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
 
def print_summary(reconciled: list[dict]):
    """Print tier breakdown and agreement statistics."""
    n = len(reconciled)
    tier_counts = Counter()
    field_agreement_rates = {f: [] for f in CRITICAL_FIELDS + SECONDARY_FIELDS}
    confidence_by_tier = {}
 
    for r in reconciled:
        for cat_variants in r["variants"].values():
            for label, vdata in cat_variants.items():
                tier = vdata.get("tier", "unknown")
                tier_counts[tier] += 1
                confidence_by_tier.setdefault(tier, []).append(
                    vdata.get("mean_confidence", 0.5)
                )
                for f in CRITICAL_FIELDS + SECONDARY_FIELDS:
                    cfa = vdata.get("critical_field_agreement", {})
                    sfa = vdata.get("secondary_field_agreement", {})
                    agreed = cfa.get(f, sfa.get(f))
                    if agreed is not None:
                        field_agreement_rates[f].append(1 if agreed else 0)
 
    total_variants = sum(tier_counts.values())
 
    print(f"\n{'='*60}")
    print("RECONCILIATION SUMMARY")
    print(f"{'='*60}")
    print(f"Questions: {n}")
    print(f"Total variants: {total_variants}")
 
    print(f"\nTier breakdown:")
    for tier in ["tier1", "tier1_medical", "tier2", "tier3",
                 "identical", "excluded"]:
        c = tier_counts.get(tier, 0)
        pct = 100 * c / total_variants if total_variants else 0
        conf_vals = confidence_by_tier.get(tier, [])
        conf_str = f"  mean_conf={statistics.mean(conf_vals):.2f}" if conf_vals else ""
        print(f"  {tier:16s}: {c:4d} ({pct:5.1f}%){conf_str}")
 
    print(f"\nField agreement rates (fraction unanimous across passes):")
    for f, vals in field_agreement_rates.items():
        if vals:
            rate = statistics.mean(vals)
            print(f"  {f:30s}: {rate:.3f} ({len(vals)} variants)")
 
    # Uncertainty flag frequency
    flag_counts = Counter()
    for r in reconciled:
        for cat_variants in r["variants"].values():
            for label, vdata in cat_variants.items():
                for flag in vdata.get("all_uncertainty_flags", []):
                    flag_counts[flag] += 1
 
    if flag_counts:
        print(f"\nUncertainty flags (across all variants):")
        for flag, count in flag_counts.most_common():
            print(f"  {flag}: {count}")
 
    # Tier breakdown by category
    print(f"\nTier 1 (bias analysis core) by category:")
    cat_tier1 = Counter()
    for r in reconciled:
        for cat, cat_variants in r["variants"].items():
            for label, vdata in cat_variants.items():
                if vdata.get("tier") == "tier1":
                    cat_tier1[cat] += 1
    for cat, count in cat_tier1.most_common():
        print(f"  {cat}: {count}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Reconcile multi-pass annotations into tiered benchmark"
    )
    parser.add_argument("passes", nargs="+",
                        help="Paths to annotation pass JSON files (2+)")
    parser.add_argument("--output", type=str, default="benchmark.json",
                        help="Output reconciled benchmark file")
    parser.add_argument("--export_tier1", type=str, default=None,
                        help="Export Tier 1 items to separate file")
    args = parser.parse_args()
 
    if len(args.passes) < 2:
        print("⚠ Need at least 2 pass files for reconciliation.")
        print("  You can still run with 1, but agreement stats won't be meaningful.")
 
    # Load and align
    passes = load_passes(args.passes)
    aligned = align_passes(passes)
 
    # Reconcile
    reconciled = reconcile_all(aligned)
 
    # Save
    with open(args.output, "w") as f:
        json.dump(reconciled, f, indent=2)
    print(f"\n✓ Reconciled benchmark → {args.output}")
 
    # Summary
    print_summary(reconciled)
 
    # Export Tier 1
    if args.export_tier1:
        tier1_items = []
        for r in reconciled:
            for cat, cat_variants in r["variants"].items():
                for label, vdata in cat_variants.items():
                    if vdata.get("tier") == "tier1":
                        tier1_items.append({
                            "question_id": r["question_id"],
                            "original_question": r["original"]["question"],
                            "original_options": r["original"]["options"],
                            "original_answer": r["original"]["answer"],
                            "category": cat,
                            "label": label,
                            "variant_text": vdata.get("reconciled_text"),
                            "reconciled_labels": vdata.get("reconciled_labels"),
                            "mean_confidence": vdata.get("mean_confidence"),
                            "evidence_spans": vdata.get("evidence_spans"),
                        })
        with open(args.export_tier1, "w") as f:
            json.dump(tier1_items, f, indent=2)
        print(f"✓ Tier 1 core set ({len(tier1_items)} items) → {args.export_tier1}")
 
 
if __name__ == "__main__":
    main()