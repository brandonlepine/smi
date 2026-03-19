"""
Counterfactual Validation & Inspection (v3 — formal schema)
==============================================================
Validates output against counterfactual_schema.json and the annotation
protocol. Checks:
  1. JSON Schema validation (structural + enum correctness)
  2. Edit distance to flag clinical content drift
  3. Protocol §5 classification into analysis classes
  4. Clinical cue interaction analysis
  5. Cross-tabulations (role × invariance, role × prior_shift, etc.)
  6. Side-by-side diffs for spot-checking

Usage:
  python validate_counterfactuals.py counterfactuals.json
  python validate_counterfactuals.py counterfactuals.json --schema counterfactual_schema.json
  python validate_counterfactuals.py counterfactuals.json --export_classes classes.json --show_diffs 5
"""

import json
import difflib
import argparse
import statistics
from collections import Counter
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_against_schema(records: list[dict], schema_path: str) -> list[str]:
    """Validate entire output against the JSON Schema."""
    if not HAS_JSONSCHEMA:
        return ["⚠ jsonschema not installed — skipping schema validation. "
                "Install with: pip install jsonschema"]

    with open(schema_path) as f:
        schema = json.load(f)

    errors = []
    validator = jsonschema.Draft202012Validator(schema)
    for error in validator.iter_errors(records):
        path = " → ".join(str(p) for p in error.absolute_path)
        errors.append(f"  {path}: {error.message}")
    return errors


# ---------------------------------------------------------------------------
# Structural checks (fallback if no jsonschema)
# ---------------------------------------------------------------------------

VALID_VALIDITY = {"valid", "questionable", "invalid"}
VALID_COHERENCE = {"preserved", "weakened", "broken"}
VALID_ROLE = {"irrelevant", "epidemiologic", "mechanistically_causal",
              "sex_specific", "socially_loaded"}
VALID_INVARIANCE = {"invariant", "likely_invariant", "uncertain", "likely_changed"}
VALID_PRIOR_SHIFT = {"none", "mild", "moderate", "strong"}
VALID_LOCALITY = {"minimal", "sentence_level", "multi_sentence"}

VARIANT_FIELDS = {
    "text", "reason_if_null", "counterfactual_validity", "clinical_coherence",
    "target_attribute_role", "gold_answer_invariance", "prior_shift_expected",
    "edit_locality", "notes"
}

EXPECTED_KEYS = {
    "gender": {"male", "female", "neutral"},
    "age": {"young_adult", "middle_aged", "elderly"},
    "race_ethnicity": {"White", "Black/African American", "Hispanic/Latino",
                       "Asian", "no_race_specified"},
    "control": {"neutral_rework", "irrelevant_surface"},
}


def check_completeness(record: dict) -> list[str]:
    """Structural checks without jsonschema."""
    issues = []
    qid = record.get("question_id", "?")
    cf = record.get("counterfactuals", {})
    variants = cf.get("variants", {})

    if "original_demographics" not in cf:
        issues.append(f"{qid}: missing original_demographics")
    if "clinical_cue_interactions" not in cf:
        issues.append(f"{qid}: missing clinical_cue_interactions")

    for category, expected in EXPECTED_KEYS.items():
        cat_data = variants.get(category, {})
        missing = expected - set(cat_data.keys())
        if missing:
            issues.append(f"{qid}/{category}: missing keys {missing}")

        for key in expected & set(cat_data.keys()):
            vdata = cat_data[key]
            if not isinstance(vdata, dict):
                issues.append(f"{qid}/{category}/{key}: not a dict")
                continue
            missing_fields = VARIANT_FIELDS - set(vdata.keys())
            if missing_fields:
                issues.append(f"{qid}/{category}/{key}: missing {missing_fields}")

            # Enum checks
            for field, allowed in [
                ("counterfactual_validity", VALID_VALIDITY),
                ("clinical_coherence", VALID_COHERENCE),
                ("target_attribute_role", VALID_ROLE),
                ("gold_answer_invariance", VALID_INVARIANCE),
                ("prior_shift_expected", VALID_PRIOR_SHIFT),
                ("edit_locality", VALID_LOCALITY),
            ]:
                val = vdata.get(field)
                if val is not None and val not in allowed:
                    issues.append(f"{qid}/{category}/{key}: "
                                  f"{field}='{val}' not in {allowed}")
    return issues


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def compute_edit_ratio(original: str, variant: str) -> float:
    if variant is None:
        return 0.0
    sm = difflib.SequenceMatcher(None, original, variant)
    return 1.0 - sm.ratio()


def flag_high_edits(records: list[dict], threshold: float) -> list[str]:
    flags = []
    for r in records:
        qid = r["question_id"]
        orig = r["original"]["question"]
        variants = r.get("counterfactuals", {}).get("variants", {})
        for cat, cat_v in variants.items():
            for label, vdata in cat_v.items():
                if not isinstance(vdata, dict):
                    continue
                text = vdata.get("text")
                if text is None:
                    continue
                ratio = compute_edit_ratio(orig, text)
                if ratio > threshold:
                    flags.append(f"{qid}/{cat}/{label}: "
                                 f"edit_ratio={ratio:.2f}")
    return flags


# ---------------------------------------------------------------------------
# Classification per protocol §5
# ---------------------------------------------------------------------------

def classify_variant(vdata: dict) -> str:
    """Classify into class1, class2, class3, identical, or unclassified."""
    if not isinstance(vdata, dict):
        return "malformed"

    text = vdata.get("text")
    reason = str(vdata.get("reason_if_null", "")).lower()
    v = vdata.get("counterfactual_validity")
    c = vdata.get("clinical_coherence")
    role = vdata.get("target_attribute_role")
    inv = vdata.get("gold_answer_invariance")

    if text is None:
        return "identical" if "identical" in reason else "class3"

    # Class 1: valid + preserved + invariant/likely + irrelevant/socially_loaded
    if (v == "valid" and c == "preserved"
            and inv in ("invariant", "likely_invariant")
            and role in ("irrelevant", "socially_loaded")):
        return "class1"

    # Class 2: valid|questionable + preserved|weakened + epidemiologic|mechanistically_causal
    if (v in ("valid", "questionable")
            and c in ("preserved", "weakened")
            and role in ("epidemiologic", "mechanistically_causal")):
        return "class2"

    # Class 3: invalid | broken
    if v == "invalid" or c == "broken":
        return "class3"

    # Borderline: valid + preserved but role is sex_specific with text
    if role == "sex_specific":
        return "class3"

    return "unclassified"


def build_class_export(records: list[dict]) -> dict:
    classes = {}
    for r in records:
        qid = r["question_id"]
        variants = r.get("counterfactuals", {}).get("variants", {})
        for cat, cat_v in variants.items():
            for label, vdata in cat_v.items():
                if not isinstance(vdata, dict):
                    continue
                cls = classify_variant(vdata)
                classes.setdefault(cls, []).append({
                    "question_id": qid,
                    "category": cat,
                    "label": label,
                    "class": cls,
                    "counterfactual_validity": vdata.get("counterfactual_validity"),
                    "clinical_coherence": vdata.get("clinical_coherence"),
                    "target_attribute_role": vdata.get("target_attribute_role"),
                    "gold_answer_invariance": vdata.get("gold_answer_invariance"),
                    "prior_shift_expected": vdata.get("prior_shift_expected"),
                    "edit_locality": vdata.get("edit_locality"),
                    "has_text": vdata.get("text") is not None,
                })
    return classes


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_diff(original: str, variant_text: str | None, label: str):
    if variant_text is None:
        return
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        variant_text.splitlines(keepends=True),
        fromfile="original", tofile=label, n=2
    )
    diff_str = "".join(diff)
    if diff_str:
        for line in diff_str.splitlines():
            print(f"      {line}")


def demographic_summary(records):
    gender_counts = Counter()
    age_values = []
    race_present = 0
    for r in records:
        od = r.get("counterfactuals", {}).get("original_demographics", {})
        gender_counts[od.get("gender", "unknown")] += 1
        if od.get("age") is not None:
            age_values.append(od["age"])
        if od.get("race_ethnicity") is not None:
            race_present += 1

    n = len(records)
    print(f"\n{'='*60}")
    print("ORIGINAL DEMOGRAPHICS")
    print(f"{'='*60}")
    print(f"Total: {n}")
    for g, c in gender_counts.most_common():
        print(f"  {g}: {c} ({100*c/n:.1f}%)")
    print(f"  Age mentioned: {len(age_values)}/{n}")
    if age_values:
        print(f"    mean={statistics.mean(age_values):.1f} "
              f"median={statistics.median(age_values):.0f} "
              f"range={min(age_values)}–{max(age_values)}")
    print(f"  Race mentioned: {race_present}/{n}")


def clinical_cue_summary(records):
    fields = ["phenotypic_cues", "sex_specific_clinical_features",
              "sexual_behavior_cues", "family_history_patterns",
              "epidemiologic_associations"]
    print(f"\n{'='*60}")
    print("CLINICAL CUE INTERACTIONS")
    print(f"{'='*60}")
    for field in fields:
        counts = Counter()
        q_with = 0
        for r in records:
            cues = r.get("counterfactuals", {}) \
                    .get("clinical_cue_interactions", {}).get(field, [])
            if cues:
                q_with += 1
                for c in cues:
                    counts[c] += 1
        print(f"\n  {field}: {q_with}/{len(records)} questions")
        for cue, count in counts.most_common(8):
            print(f"    {cue}: {count}")


# ---------------------------------------------------------------------------
# Cross-tabulations
# ---------------------------------------------------------------------------

def cross_tabulate(records):
    """Print key cross-tabs: role × invariance, role × prior_shift."""
    role_inv = Counter()
    role_prior = Counter()
    cat_role = Counter()

    for r in records:
        variants = r.get("counterfactuals", {}).get("variants", {})
        for cat, cat_v in variants.items():
            for label, vdata in cat_v.items():
                if not isinstance(vdata, dict) or vdata.get("text") is None:
                    continue
                role = vdata.get("target_attribute_role", "?")
                inv = vdata.get("gold_answer_invariance", "?")
                ps = vdata.get("prior_shift_expected", "?")
                role_inv[(role, inv)] += 1
                role_prior[(role, ps)] += 1
                cat_role[(cat, role)] += 1

    print(f"\n{'='*60}")
    print("CROSS-TABULATIONS (non-null variants only)")
    print(f"{'='*60}")

    print("\n  target_attribute_role × gold_answer_invariance:")
    for (role, inv), count in sorted(role_inv.items()):
        print(f"    {role:25s} × {inv:18s} = {count}")

    print("\n  target_attribute_role × prior_shift_expected:")
    for (role, ps), count in sorted(role_prior.items()):
        print(f"    {role:25s} × {ps:10s} = {count}")

    print("\n  category × target_attribute_role:")
    for (cat, role), count in sorted(cat_role.items()):
        print(f"    {cat:18s} × {role:25s} = {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate counterfactuals against formal annotation protocol"
    )
    parser.add_argument("input", help="Path to counterfactuals.json")
    parser.add_argument("--schema", type=str,
                        default="counterfactual_schema.json",
                        help="Path to JSON Schema file")
    parser.add_argument("--show_diffs", type=int, default=3)
    parser.add_argument("--edit_threshold", type=float, default=0.35)
    parser.add_argument("--export_classes", type=str, default=None,
                        help="Export class assignments to JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {args.input}\n")

    # --- JSON Schema validation ---
    print("JSON SCHEMA VALIDATION")
    print("-" * 40)
    if Path(args.schema).exists():
        schema_errors = validate_against_schema(records, args.schema)
        if schema_errors:
            print(f"  ⚠ {len(schema_errors)} schema violations:")
            for e in schema_errors[:20]:
                print(f"    {e}")
            if len(schema_errors) > 20:
                print(f"    ... and {len(schema_errors) - 20} more")
        else:
            print("  ✓ All records pass schema validation")
    else:
        print(f"  ⚠ Schema file not found: {args.schema}")
        print("    Falling back to structural checks...")
        all_issues = []
        for r in records:
            all_issues.extend(check_completeness(r))
        if all_issues:
            print(f"  ⚠ {len(all_issues)} issues:")
            for i in all_issues[:20]:
                print(f"    {i}")
        else:
            print("  ✓ Structural checks pass")

    # --- Edit distance ---
    print(f"\nEDIT DISTANCE (threshold={args.edit_threshold})")
    print("-" * 40)
    flags = flag_high_edits(records, args.edit_threshold)
    if flags:
        print(f"  ⚠ {len(flags)} high-edit variants:")
        for f_ in flags[:15]:
            print(f"    {f_}")
    else:
        print("  ✓ No variants exceed threshold")

    # Edit ratios by category
    edit_by_cat = {}
    for r in records:
        orig = r["original"]["question"]
        variants = r.get("counterfactuals", {}).get("variants", {})
        for cat, cat_v in variants.items():
            for label, vdata in cat_v.items():
                if isinstance(vdata, dict) and vdata.get("text"):
                    edit_by_cat.setdefault(cat, []).append(
                        compute_edit_ratio(orig, vdata["text"])
                    )
    print("\n  Mean edit ratios by category:")
    for cat, ratios in sorted(edit_by_cat.items()):
        print(f"    {cat:18s}: mean={statistics.mean(ratios):.3f} "
              f"max={max(ratios):.3f} n={len(ratios)}")

    # --- Demographics ---
    demographic_summary(records)

    # --- Clinical cues ---
    clinical_cue_summary(records)

    # --- Classification ---
    print(f"\n{'='*60}")
    print("ANALYSIS CLASSES (protocol §5)")
    print(f"{'='*60}")
    class_data = build_class_export(records)
    for cls in ["class1", "class2", "class3", "identical", "unclassified"]:
        items = class_data.get(cls, [])
        if items:
            by_cat = Counter(i["category"] for i in items)
            cat_str = ", ".join(f"{c}={n}" for c, n in by_cat.most_common())
            print(f"  {cls:16s}: {len(items):4d}  ({cat_str})")

    c1 = len(class_data.get("class1", []))
    c2 = len(class_data.get("class2", []))
    c3 = len(class_data.get("class3", []))
    print(f"\n  → Class 1 (main bias analysis): {c1}")
    print(f"  → Class 2 (appropriate sensitivity): {c2}")
    print(f"  → Class 3 (robustness probes): {c3}")

    # --- Cross-tabulations ---
    cross_tabulate(records)

    # --- Export ---
    if args.export_classes:
        with open(args.export_classes, "w") as f:
            json.dump(class_data, f, indent=2)
        print(f"\n✓ Class assignments → {args.export_classes}")

    # --- Sample diffs ---
    if args.show_diffs > 0:
        print(f"\n{'='*60}")
        print(f"SAMPLE DIFFS ({args.show_diffs} questions)")
        print(f"{'='*60}")
        for r in records[:args.show_diffs]:
            print(f"\n{'─'*60}")
            print(f"ID: {r['question_id']}")
            print(f"Q:  {r['original']['question'][:140]}...")
            od = r.get("counterfactuals", {}).get("original_demographics", {})
            print(f"Demo: {od}")
            cci = r.get("counterfactuals", {}) \
                   .get("clinical_cue_interactions", {})
            nonempty = {k: v for k, v in cci.items() if v}
            if nonempty:
                print(f"Cues: {nonempty}")

            variants = r.get("counterfactuals", {}).get("variants", {})
            for cat in ["gender", "age", "race_ethnicity", "control"]:
                print(f"\n  --- {cat} ---")
                for label, vdata in variants.get(cat, {}).items():
                    if not isinstance(vdata, dict):
                        continue
                    text = vdata.get("text")
                    cls = classify_variant(vdata)
                    v = vdata.get("counterfactual_validity", "?")
                    c = vdata.get("clinical_coherence", "?")
                    role = vdata.get("target_attribute_role", "?")
                    inv = vdata.get("gold_answer_invariance", "?")
                    ps = vdata.get("prior_shift_expected", "?")
                    loc = vdata.get("edit_locality", "?")

                    if text is None:
                        reason = vdata.get("reason_if_null", "?")
                        print(f"    [{label}] NULL ({reason}) → {cls}")
                    else:
                        print(f"    [{label}] v={v} c={c} role={role} "
                              f"inv={inv} prior={ps} loc={loc} → {cls}")
                        print_diff(r["original"]["question"], text, label)


if __name__ == "__main__":
    main()