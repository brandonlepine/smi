#!/usr/bin/env python3
"""
One-time manifest repair for interrupted + resumed extraction runs.

Rebuilds a complete manifest.pt that covers all shard files on disk,
reconstructing pair_metadata for shards that were written before the
manifest existed (i.e., pre-resume shards).

Usage:
  python scripts/repair_manifest.py \
    --extraction_dir extractions_v6 \
    --data_path cf_v6_balanced.json \
    --model_path models/llama2-7b
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Minimal copies of helpers from mechanistic_head_tracing.py
# (so this script is self-contained)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Question:
{question}

Answer choices:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

INTERVENTION_TYPE_TO_CATEGORY = {
    "sex": ("gender", "gender"),
    "sex_gender": ("gender", "gender"),
    "age": ("age", "age"),
    "race_ethnicity": ("race_ethnicity", "race"),
    "pronoun": ("pronoun", "pronoun"),
    "name": ("name", "name"),
    "pregnancy_status": ("pregnancy_status", "pregnancy"),
    "reproductive_status": ("reproductive_status", "reproductive"),
    "neutral_rework": ("control", "control"),
    "irrelevant_surface": ("control", "control"),
    "sexual_orientation": ("sexual_orientation", "sexual_orientation"),
    "gender_identity": ("gender_identity", "gender_identity"),
    "kinship_role": ("kinship_role", "kinship"),
    "disability_identity": ("disability_identity", "disability"),
    "insurance_status": ("insurance_status", "insurance"),
    "housing_status": ("housing_status", "housing"),
    "occupation": ("occupation", "occupation"),
    "marital_status": ("marital_status", "marital"),
    "socioeconomic_status": ("socioeconomic_status", "socioeconomic"),
    "family_structure": ("family_structure", "family"),
    "nationality": ("nationality", "nationality"),
    "religion": ("religion", "religion"),
}


def format_prompt(question_text, options):
    return PROMPT_TEMPLATE.format(
        question=question_text.strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


def classify_for_analysis(vdata):
    if not isinstance(vdata, dict):
        return "excluded"
    text = vdata.get("text")
    reason = str(vdata.get("reason_if_null", "")).lower()
    validity = vdata.get("counterfactual_validity")
    coherence = vdata.get("clinical_coherence")
    role = vdata.get("target_attribute_role")
    inv = vdata.get("gold_answer_invariance")
    if text is None:
        return "identical" if "identical" in reason else "excluded"
    if validity == "valid" and coherence == "preserved" \
            and inv in ("invariant", "likely_invariant") \
            and role in ("irrelevant", "socially_loaded"):
        return "class1"
    if validity in ("valid", "questionable") \
            and coherence in ("preserved", "weakened") \
            and role in ("epidemiologic", "mechanistically_causal"):
        return "class2"
    if validity == "invalid" or coherence == "broken":
        return "class3"
    return "class2"


def build_pair_lookup(data_path):
    """Build a dict: pair_key -> variant metadata, plus qid -> original."""
    with open(data_path) as f:
        records = json.load(f)

    pair_lookup = {}   # pair_key -> (record, vdata)
    orig_lookup = {}   # qid -> record

    for record in records:
        qid = record["question_id"]
        orig_lookup[qid] = record
        variants = record.get("counterfactuals", {}).get("variants", [])
        for vdata in variants:
            if not isinstance(vdata, dict):
                continue
            itype = vdata.get("intervention_type", "")
            cat_info = INTERVENTION_TYPE_TO_CATEGORY.get(itype)
            if cat_info is None:
                continue
            category, attribute_type = cat_info
            label = str(vdata.get("attribute_value_counterfactual", itype))
            pair_key = f"{qid}__{category}__{label}"
            pair_lookup[pair_key] = (record, vdata, category, attribute_type, label)

    return orig_lookup, pair_lookup


def build_char_to_token_map(text, token_ids, tokenizer):
    char_map = [None] * len(text)
    decoded = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids]
    char_pos = 0
    for tok_idx, tok_text in enumerate(decoded):
        clean = tok_text.replace("▁", " ").replace("Ġ", " ")
        if not clean:
            continue
        for ch in clean:
            if char_pos >= len(text):
                break
            char_map[char_pos] = tok_idx
            char_pos += 1
    return char_map


def find_stem_offset(full_ids, stem_ids):
    for start in range(len(full_ids) - len(stem_ids) + 1):
        if full_ids[start:start + len(stem_ids)] == stem_ids:
            return start
    for trim in range(1, min(5, len(stem_ids))):
        sub = stem_ids[trim:]
        for start in range(len(full_ids) - len(sub) + 1):
            if full_ids[start:start + len(sub)] == sub:
                return start - trim
    return None


def get_diff_info(tokenizer, orig_question, cf_question, orig_prompt, cf_prompt):
    import difflib
    char_sm = difflib.SequenceMatcher(None, orig_question, cf_question)
    char_edit_regions = []
    total_char_edits = 0
    for op, i1, i2, j1, j2 in char_sm.get_opcodes():
        if op == "equal":
            continue
        char_edit_regions.append((op, i1, i2, j1, j2))
        total_char_edits += max(i2 - i1, j2 - j1)

    stem_orig_ids = tokenizer.encode(orig_question, add_special_tokens=False)
    stem_cf_ids = tokenizer.encode(cf_question, add_special_tokens=False)
    orig_c2t = build_char_to_token_map(orig_question, stem_orig_ids, tokenizer)
    cf_c2t = build_char_to_token_map(cf_question, stem_cf_ids, tokenizer)

    stem_orig_edit_tokens = set()
    stem_cf_edit_tokens = set()
    primary_orig = set()
    primary_cf = set()

    for _, oi1, oi2, ci1, ci2 in char_edit_regions:
        for c in range(oi1, oi2):
            if c < len(orig_c2t) and orig_c2t[c] is not None:
                stem_orig_edit_tokens.add(orig_c2t[c])
        for c in range(ci1, ci2):
            if c < len(cf_c2t) and cf_c2t[c] is not None:
                stem_cf_edit_tokens.add(cf_c2t[c])

    if char_edit_regions:
        biggest_idx = max(range(len(char_edit_regions)),
                          key=lambda i: max(char_edit_regions[i][2] - char_edit_regions[i][1],
                                            char_edit_regions[i][4] - char_edit_regions[i][3]))
        _, oi1, oi2, ci1, ci2 = char_edit_regions[biggest_idx]
        for c in range(oi1, oi2):
            if c < len(orig_c2t) and orig_c2t[c] is not None:
                primary_orig.add(orig_c2t[c])
        for c in range(ci1, ci2):
            if c < len(cf_c2t) and cf_c2t[c] is not None:
                primary_cf.add(cf_c2t[c])

    full_orig_ids = tokenizer.encode(orig_prompt, add_special_tokens=True)
    full_cf_ids = tokenizer.encode(cf_prompt, add_special_tokens=True)
    orig_offset = find_stem_offset(full_orig_ids, stem_orig_ids)
    cf_offset = find_stem_offset(full_cf_ids, stem_cf_ids)
    alignment_failed = orig_offset is None or cf_offset is None

    if alignment_failed:
        orig_edit_positions = []
        cf_edit_positions = []
        largest_orig_positions = []
        largest_cf_positions = []
    else:
        orig_edit_positions = sorted(p + orig_offset for p in stem_orig_edit_tokens
                                      if 0 <= p + orig_offset < len(full_orig_ids))
        cf_edit_positions = sorted(p + cf_offset for p in stem_cf_edit_tokens
                                    if 0 <= p + cf_offset < len(full_cf_ids))
        largest_orig_positions = sorted(p + orig_offset for p in primary_orig
                                         if 0 <= p + orig_offset < len(full_orig_ids))
        largest_cf_positions = sorted(p + cf_offset for p in primary_cf
                                       if 0 <= p + cf_offset < len(full_cf_ids))

    return {
        "orig_edit_positions": orig_edit_positions,
        "cf_edit_positions": cf_edit_positions,
        "largest_orig_positions": largest_orig_positions,
        "largest_cf_positions": largest_cf_positions,
        "alignment_failed": alignment_failed,
        "n_tokens_changed": len(stem_orig_edit_tokens | stem_cf_edit_tokens),
        "n_largest_region_tokens": len(primary_orig | primary_cf),
        "n_edit_regions": len(char_edit_regions),
        "n_orig_tokens": len(full_orig_ids),
        "n_cf_tokens": len(full_cf_ids),
        "stem_orig_len": len(stem_orig_ids),
        "stem_cf_len": len(stem_cf_ids),
        "char_edit_distance": total_char_edits,
        "token_edit_ratio": len(stem_orig_edit_tokens | stem_cf_edit_tokens)
                            / max(len(stem_orig_ids), len(stem_cf_ids), 1),
        "orig_edit_positions": orig_edit_positions,
        "cf_edit_positions": cf_edit_positions,
        "n_orig_edit_positions": len(orig_edit_positions),
        "n_cf_edit_positions": len(cf_edit_positions),
    }


# ---------------------------------------------------------------------------
# Main repair logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True,
                        help="Tokenizer path — only the tokenizer is loaded, not the model weights")
    args = parser.parse_args()

    extraction_dir = Path(args.extraction_dir)
    manifest_path = extraction_dir / "manifest.pt"

    # Load existing (incomplete) manifest as base for model_config / answer_token_ids
    print("Loading existing manifest...")
    existing_manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)

    # Load tokenizer (no model weights needed)
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    # Build pair lookup from source data
    print(f"Building pair lookup from {args.data_path}...")
    orig_lookup, pair_lookup = build_pair_lookup(args.data_path)
    print(f"  {len(orig_lookup)} questions, {len(pair_lookup)} pairs")

    # Scan all shard files on disk
    shard_files = sorted(extraction_dir.glob("extractions_shard_*.pt"))
    print(f"Scanning {len(shard_files)} shard files...")

    # Keys already covered by existing manifest's pair_metadata
    existing_pair_keys = {m["pair_key"] for m in existing_manifest["pair_metadata"]}
    print(f"  Existing pair_metadata entries: {len(existing_pair_keys)}")

    new_shards = []
    new_pair_metadata = []
    skipped_keys = []

    for i, sf in enumerate(shard_files):
        idx = int(sf.stem.split("_")[-1])
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(shard_files)}] shard_{idx:04d}")

        try:
            shard = torch.load(sf, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Warning: could not load {sf.name}: {e}")
            continue

        new_shards.append({
            "shard_idx": idx,
            "filename": sf.name,
            "n_original_results": len(shard.get("original_results", {})),
            "n_cf_results": len(shard.get("cf_results", {})),
        })

        # Reconstruct pair_metadata for CF keys not in existing manifest
        for pair_key in shard.get("cf_results", {}).keys():
            if pair_key in existing_pair_keys:
                continue

            lookup = pair_lookup.get(pair_key)
            if lookup is None:
                skipped_keys.append(pair_key)
                continue

            record, vdata, category, attribute_type, label = lookup
            qid = record["question_id"]
            original = record["original"]
            options = original["options"]
            gold_answer = original.get("answer_idx", "")
            if not gold_answer:
                for k, v in options.items():
                    if v == original["answer"]:
                        gold_answer = k
                        break

            orig_q = original["question"]
            cf_q = vdata.get("text", "")
            orig_prompt = format_prompt(orig_q, options)
            cf_prompt = format_prompt(cf_q, options)

            control_subtype = vdata.get("intervention_type", "") if category == "control" else "none"
            analysis_class = classify_for_analysis(vdata)
            if category == "control":
                analysis_class = "control"

            diff_info = get_diff_info(tokenizer, orig_q, cf_q, orig_prompt, cf_prompt)

            new_pair_metadata.append({
                "question_id": qid,
                "category": category,
                "label": label,
                "attribute_type": attribute_type,
                "control_subtype": control_subtype,
                "gold_answer": gold_answer,
                "analysis_class": analysis_class,
                "counterfactual_validity": vdata.get("counterfactual_validity"),
                "clinical_coherence": vdata.get("clinical_coherence"),
                "target_attribute_role": vdata.get("target_attribute_role", vdata.get("medical_relevance")),
                "gold_answer_invariance": vdata.get("gold_answer_invariance"),
                "prior_shift_expected": vdata.get("prior_shift_expected"),
                "edit_locality": vdata.get("edit_locality", vdata.get("edit_scope")),
                "annotation_confidence": vdata.get("annotation_confidence"),
                "intervention_type": vdata.get("intervention_type"),
                "intervention_family": vdata.get("intervention_family"),
                "analysis_bucket": vdata.get("analysis_bucket"),
                "edit_scope": vdata.get("edit_scope"),
                "edit_strength": vdata.get("edit_strength"),
                "attribute_value_counterfactual": vdata.get("attribute_value_counterfactual"),
                "medical_relevance": vdata.get("medical_relevance"),
                "social_bias_salience": vdata.get("social_bias_salience"),
                "pair_key": pair_key,
                "orig_edit_key": f"{qid}__edit__{pair_key}",
                "orig_largest_key": None,
                "cf_largest_key": None,
                **diff_info,
            })

        del shard

    # Merge: new reconstructed metadata + existing manifest metadata
    all_pair_metadata = new_pair_metadata + existing_manifest["pair_metadata"]
    all_shards = sorted(new_shards, key=lambda s: s["shard_idx"])

    repaired_manifest = {
        "model_config": existing_manifest["model_config"],
        "answer_token_ids": existing_manifest["answer_token_ids"],
        "pair_metadata": all_pair_metadata,
        "shards": all_shards,
    }

    backup_path = extraction_dir / "manifest_backup.pt"
    import shutil
    shutil.copy(manifest_path, backup_path)
    print(f"\nBacked up original manifest → {backup_path.name}")

    torch.save(repaired_manifest, manifest_path)
    print(f"Saved repaired manifest → {manifest_path.name}")
    print(f"  Shards: {len(all_shards)}")
    print(f"  Pair metadata: {len(all_pair_metadata)} total")
    print(f"    {len(new_pair_metadata)} reconstructed from pre-resume shards")
    print(f"    {len(existing_manifest['pair_metadata'])} from resumed run")

    if skipped_keys:
        print(f"\nWarning: {len(skipped_keys)} CF keys not found in source data:")
        for k in skipped_keys[:10]:
            print(f"  {k}")

    # Sanity: count unique analysis classes
    from collections import Counter
    cc = Counter(m["analysis_class"] for m in all_pair_metadata)
    print(f"\nAnalysis class distribution: {dict(cc)}")


if __name__ == "__main__":
    main()
