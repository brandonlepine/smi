"""
Stage 1 Extraction v3: Token-Localized Hidden States + Attention Head Summaries
================================================================================
Extends v2.1 with compact attention-head extraction for later causal tracing.

Retained from v2.1:
  - Schema-normalized intervention metadata
  - Token-localized hidden state extraction
  - Format auto-detection (flat-list vs nested-dict)

New in v3:
  - Optional extraction of per-head attention summaries (--extract_attention)
  - Per-layer, per-head attention summaries for:
      * final token query position
      * edited-token query region (mean across edit positions)
      * largest contiguous edit region query
  - Top-k attended source positions per layer/head
  - Attention mass directed to:
      * edit region
      * largest edit region
      * question stem
  - Per-head attention entropy (Shannon entropy of distribution)

Performance:
  - Groups processing by question_id so each original prompt gets ONE
    forward pass, reused across all its counterfactual pairs
  - Eliminates duplicate forward passes for __largest extractions
  - Vectorized attention summary (batched tensor ops, no per-head loops)

Checkpointing:
  - Saves progress every --checkpoint_every pairs (default 100)
  - Use --resume to continue from an existing checkpoint
  - Checkpoint file: {output}.ckpt (auto-cleaned on completion)

Important:
  - This does NOT save full attention tensors
  - It saves compact summaries suitable for downstream causal-tracing
    candidate-head discovery
  - Enabling attention extraction increases runtime and memory use
    (output_attentions=True in the forward pass)

Usage:
  python extract_representations.py \
    --model_path /path/to/llama2-13b \
    --data_path /path/to/pass_A.json \
    --output stage1_extractions_v3.pt \
    --device auto \
    --extract_attention \
    --attention_topk 10 \
    --checkpoint_every 100 \
    --resume
"""

import json
import argparse
import time
import difflib
import warnings
import gc
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Prompt template
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


def format_prompt(question_text: str, options: dict) -> str:
    return PROMPT_TEMPLATE.format(
        question=(question_text or "").strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

OLD_TO_NEW_TYPE = {
    # older naming / shorthand / drifted names
    "gender": "sex",
    "race": "race_ethnicity",
    "housing": "housing_status",
    "insurance": "insurance_status",
    "marital": "marital_status",
    "kinship": "kinship_role",
}

CORE_BIAS_TYPES = {
    "sex",
    "age",
    "race_ethnicity",
    "pronoun",
    "name",
    "pregnancy_status",
    "reproductive_status",
}

IDENTITY_BIAS_TYPES = {
    "gender_identity",
    "sexual_orientation",
    "honorific",
    "kinship_role",
    "disability_identity",
}

STRUCTURAL_TYPES = {
    "insurance_status",
    "housing_status",
    "occupation",
    "marital_status",
    "socioeconomic_status",
    "family_structure",
    "nationality",
    "religion",
}

CONTROL_TYPES = {
    "neutral_rework",
    "irrelevant_surface",
    "control",
}


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_intervention_type(raw_type: str) -> str:
    raw = safe_str(raw_type)
    return OLD_TO_NEW_TYPE.get(raw, raw)


def normalize_group(
    intervention_type: str,
    analysis_bucket: str | None = None,
    analysis_class: str | None = None,
) -> str:
    """
    Stable grouping for downstream analyses.

    Priority:
      1. explicit control handling
      2. explicit analysis_bucket if trustworthy
      3. normalized intervention type sets
      4. fallback on analysis_class
    """
    itype = normalize_intervention_type(intervention_type)
    bucket = safe_str(analysis_bucket)

    if analysis_class == "control" or itype in CONTROL_TYPES:
        return "control"

    if bucket in {"core_bias", "identity_bias", "structural_context", "control"}:
        return bucket

    if itype in CORE_BIAS_TYPES:
        return "core_bias"
    if itype in IDENTITY_BIAS_TYPES:
        return "identity_bias"
    if itype in STRUCTURAL_TYPES:
        return "structural_context"

    if analysis_class == "control":
        return "control"

    return "unknown"


def normalize_control_subtype(vdata: dict, intervention_type: str) -> str:
    """
    Recover control subtype from a variety of possible fields.
    """
    candidates = [
        safe_str(vdata.get("control_subtype")),
        safe_str(vdata.get("variant_subtype")),
        safe_str(vdata.get("subtype")),
        safe_str(vdata.get("control_type")),
        safe_str(vdata.get("label")),
        safe_str(intervention_type),
    ]
    for c in candidates:
        if c in {"neutral_rework", "irrelevant_surface"}:
            return c
    if normalize_intervention_type(intervention_type) == "control":
        return "unknown_control"
    return "none"


def normalize_attribute_type(intervention_type: str) -> str:
    """
    Attribute type used in downstream analysis plots/tables.
    """
    itype = normalize_intervention_type(intervention_type)
    mapping = {
        "sex": "gender",
        "age": "age",
        "race_ethnicity": "race",
        "pronoun": "pronoun",
        "name": "name",
        "pregnancy_status": "pregnancy",
        "reproductive_status": "reproductive",
        "sexual_orientation": "sexual_orientation",
        "gender_identity": "gender_identity",
        "kinship_role": "kinship",
        "honorific": "honorific",
        "disability_identity": "disability",
        "insurance_status": "insurance",
        "housing_status": "housing",
        "occupation": "occupation",
        "marital_status": "marital",
        "socioeconomic_status": "socioeconomic",
        "family_structure": "family",
        "nationality": "nationality",
        "religion": "religion",
        "neutral_rework": "control",
        "irrelevant_surface": "control",
        "control": "control",
    }
    return mapping.get(itype, itype)


def normalize_edit_locality(vdata: dict) -> str | None:
    """
    Preserve a stable locality-like field for downstream analyses.

    Preference:
      1. explicit edit_locality if present
      2. edit_strength (minimal/moderate/strong/single)
      3. edit_scope (token/phrase/sentence/multi_sentence)
    """
    edit_locality = safe_str(vdata.get("edit_locality"))
    if edit_locality:
        return edit_locality

    edit_strength = safe_str(vdata.get("edit_strength"))
    if edit_strength in {"minimal", "moderate", "strong", "single"}:
        return edit_strength

    edit_scope = safe_str(vdata.get("edit_scope"))
    if edit_scope:
        return edit_scope

    return None


# ---------------------------------------------------------------------------
# Token-level diff between original and counterfactual
# ---------------------------------------------------------------------------

def find_edited_token_positions(
    tokenizer,
    orig_question: str,
    cf_question: str,
    orig_prompt: str,
    cf_prompt: str,
) -> dict:
    """
    Find edited token positions using character-level alignment.
    """
    orig_question = orig_question or ""
    cf_question = cf_question or ""

    # --- Step 1: Character-level diff on stems ---
    char_sm = difflib.SequenceMatcher(None, orig_question, cf_question)
    char_edit_regions = []
    total_char_edits = 0

    for op, i1, i2, j1, j2 in char_sm.get_opcodes():
        if op == "equal":
            continue
        char_edit_regions.append((op, i1, i2, j1, j2))
        total_char_edits += max(i2 - i1, j2 - j1)

    # --- Step 2: Map character spans -> token positions in stems ---
    stem_orig_ids = tokenizer.encode(orig_question, add_special_tokens=False)
    stem_cf_ids = tokenizer.encode(cf_question, add_special_tokens=False)

    def build_char_to_token_map(text: str, token_ids: list[int], tokenizer) -> list[int | None]:
        char_map = [None] * len(text)
        char_pos = 0

        for tok_idx, tid in enumerate(token_ids):
            decoded = tokenizer.decode([tid])
            clean = decoded.lstrip("▁Ġ")
            if clean != decoded:
                if char_pos < len(text) and text[char_pos] == " ":
                    char_map[char_pos] = tok_idx
                    char_pos += 1
                clean_len = len(clean)
            else:
                clean_len = len(decoded)

            for _ in range(clean_len):
                if char_pos < len(text):
                    char_map[char_pos] = tok_idx
                    char_pos += 1

        return char_map

    orig_c2t = build_char_to_token_map(orig_question, stem_orig_ids, tokenizer)
    cf_c2t = build_char_to_token_map(cf_question, stem_cf_ids, tokenizer)

    stem_orig_edit_tokens = set()
    stem_cf_edit_tokens = set()

    for _, oi1, oi2, ci1, ci2 in char_edit_regions:
        for c in range(oi1, oi2):
            if c < len(orig_c2t) and orig_c2t[c] is not None:
                stem_orig_edit_tokens.add(orig_c2t[c])
        for c in range(ci1, ci2):
            if c < len(cf_c2t) and cf_c2t[c] is not None:
                stem_cf_edit_tokens.add(cf_c2t[c])

    # --- Step 2b: Largest contiguous region ---
    primary_orig = set()
    primary_cf = set()
    if char_edit_regions:
        biggest_idx = max(
            range(len(char_edit_regions)),
            key=lambda i: max(
                char_edit_regions[i][2] - char_edit_regions[i][1],
                char_edit_regions[i][4] - char_edit_regions[i][3],
            ),
        )
        _, oi1, oi2, ci1, ci2 = char_edit_regions[biggest_idx]
        for c in range(oi1, oi2):
            if c < len(orig_c2t) and orig_c2t[c] is not None:
                primary_orig.add(orig_c2t[c])
        for c in range(ci1, ci2):
            if c < len(cf_c2t) and cf_c2t[c] is not None:
                primary_cf.add(cf_c2t[c])

    # --- Step 3: Map stem token positions -> full prompt positions ---
    full_orig_ids = tokenizer.encode(orig_prompt, add_special_tokens=True)
    full_cf_ids = tokenizer.encode(cf_prompt, add_special_tokens=True)

    def find_stem_offset(full_ids: list[int], stem_ids: list[int]) -> int | None:
        for start in range(len(full_ids) - len(stem_ids) + 1):
            if full_ids[start:start + len(stem_ids)] == stem_ids:
                return start
        for trim in range(1, min(5, len(stem_ids))):
            sub = stem_ids[trim:]
            for start in range(len(full_ids) - len(sub) + 1):
                if full_ids[start:start + len(sub)] == sub:
                    return start - trim
        return None

    orig_offset = find_stem_offset(full_orig_ids, stem_orig_ids)
    cf_offset = find_stem_offset(full_cf_ids, stem_cf_ids)
    alignment_failed = (orig_offset is None or cf_offset is None)

    if alignment_failed:
        warnings.warn(
            f"Stem-to-prompt alignment failed: "
            f"orig_offset={'FAIL' if orig_offset is None else orig_offset}, "
            f"cf_offset={'FAIL' if cf_offset is None else cf_offset}. "
            f"Edit positions will be empty for this pair."
        )
        orig_edit_positions = []
        cf_edit_positions = []
        largest_orig_positions = []
        largest_cf_positions = []
    else:
        orig_edit_positions = sorted(
            p + orig_offset for p in stem_orig_edit_tokens
            if 0 <= p + orig_offset < len(full_orig_ids)
        )
        cf_edit_positions = sorted(
            p + cf_offset for p in stem_cf_edit_tokens
            if 0 <= p + cf_offset < len(full_cf_ids)
        )
        largest_orig_positions = sorted(
            p + orig_offset for p in primary_orig
            if 0 <= p + orig_offset < len(full_orig_ids)
        )
        largest_cf_positions = sorted(
            p + cf_offset for p in primary_cf
            if 0 <= p + cf_offset < len(full_cf_ids)
        )

    # --- Step 4: Debug context ---
    debug_context = {}
    for tag, positions, full_ids in [
        ("orig", orig_edit_positions, full_orig_ids),
        ("cf", cf_edit_positions, full_cf_ids),
    ]:
        if positions:
            center = positions[len(positions) // 2]
            ws = max(0, center - 5)
            we = min(len(full_ids), center + 6)
            debug_context[f"{tag}_window"] = tokenizer.decode(full_ids[ws:we])
            debug_context[f"{tag}_window_tokens"] = [
                tokenizer.decode([tid]) for tid in full_ids[ws:we]
            ]
            debug_context[f"{tag}_edit_center"] = center

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
        "token_edit_ratio": (
            len(stem_orig_edit_tokens | stem_cf_edit_tokens)
            / max(len(stem_orig_ids), len(stem_cf_ids), 1)
        ),
        "orig_offset": orig_offset,
        "cf_offset": cf_offset,
        "debug_context": debug_context,
    }


# ---------------------------------------------------------------------------
# Legacy format support
# ---------------------------------------------------------------------------

CATEGORIES = {
    "gender": ["male", "female", "neutral"],
    "age": ["young_adult", "middle_aged", "elderly"],
    "race_ethnicity": [
        "White",
        "Black/African American",
        "Hispanic/Latino",
        "Asian",
        "no_race_specified",
    ],
    "control": ["neutral_rework", "irrelevant_surface"],
}

CATEGORY_TO_ATTR = {
    "gender": "gender",
    "age": "age",
    "race_ethnicity": "race",
    "control": "control",
}


@dataclass
class InferencePair:
    question_id: str
    category: str
    label: str
    attribute_type: str
    control_subtype: str

    raw_intervention_type: str
    normalized_intervention_type: str
    normalized_group: str
    analysis_bucket: str | None

    original_prompt: str
    counterfactual_prompt: str
    original_question: str
    cf_question: str
    gold_answer: str
    options: dict

    analysis_class: str
    counterfactual_validity: str | None
    clinical_coherence: str | None
    target_attribute_role: str | None
    gold_answer_invariance: str | None
    prior_shift_expected: str | None
    edit_locality: str | None
    raw_edit_locality: str | None
    annotation_confidence: float | None


def classify_for_analysis(vdata: dict) -> str:
    """
    Backward-compatible analysis class derivation.

    class1:
      valid/preserved + invariant-ish + medically irrelevant/socially loaded
    class2:
      valid/questionable + preserved/weakened + epidemiologic/mechanistically causal/ambiguous
    class3:
      invalid/broken
    """
    if not isinstance(vdata, dict):
        return "excluded"

    text = vdata.get("text")
    reason = safe_str(vdata.get("reason_if_null")).lower()
    validity = safe_str(vdata.get("counterfactual_validity"))
    coherence = safe_str(vdata.get("clinical_coherence"))

    role = safe_str(vdata.get("target_attribute_role"))
    if not role:
        role = safe_str(vdata.get("medical_relevance"))

    invariance = safe_str(vdata.get("gold_answer_invariance"))

    if text is None:
        return "identical" if "identical" in reason else "excluded"

    if (
        validity == "valid"
        and coherence == "preserved"
        and invariance in {"invariant", "likely_invariant"}
        and role in {"irrelevant", "socially_loaded"}
    ):
        return "class1"

    if (
        validity in {"valid", "questionable"}
        and coherence in {"preserved", "weakened"}
        and role in {"epidemiologic", "mechanistically_causal", "ambiguous"}
    ):
        return "class2"

    if validity == "invalid" or coherence == "broken":
        return "class3"

    return "class2"


def _load_pairs_legacy(records, pairs):
    for record in records:
        qid = record["question_id"]
        original = record["original"]
        options = original["options"]

        gold_answer = original.get("answer_idx", "")
        if not gold_answer:
            for k, v in options.items():
                if v == original["answer"]:
                    gold_answer = k
                    break

        orig_prompt = format_prompt(original["question"], options)
        variants = record.get("counterfactuals", {}).get("variants", {})

        for category, labels in CATEGORIES.items():
            for label in labels:
                vdata = variants.get(category, {}).get(label, {})
                if not isinstance(vdata, dict):
                    continue
                text = vdata.get("text")
                if text is None:
                    continue

                analysis_class = classify_for_analysis(vdata)
                if category == "control" and text is not None:
                    analysis_class = "control"

                raw_intervention_type = category if category != "control" else label
                normalized_type = normalize_intervention_type(raw_intervention_type)
                analysis_bucket = safe_str(vdata.get("analysis_bucket")) or None
                normalized_group = normalize_group(
                    normalized_type,
                    analysis_bucket=analysis_bucket,
                    analysis_class=analysis_class,
                )
                control_subtype = (
                    label if category == "control"
                    else normalize_control_subtype(vdata, raw_intervention_type)
                )

                raw_edit_locality = safe_str(vdata.get("edit_locality")) or None
                edit_locality = normalize_edit_locality(vdata)

                cf_prompt = format_prompt(text, options)

                pairs.append(
                    InferencePair(
                        question_id=qid,
                        category=category,
                        label=label,
                        attribute_type=CATEGORY_TO_ATTR[category],
                        control_subtype=control_subtype,
                        raw_intervention_type=raw_intervention_type,
                        normalized_intervention_type=normalized_type,
                        normalized_group=normalized_group,
                        analysis_bucket=analysis_bucket,
                        original_prompt=orig_prompt,
                        counterfactual_prompt=cf_prompt,
                        original_question=original["question"],
                        cf_question=text,
                        gold_answer=gold_answer,
                        options=options,
                        analysis_class=analysis_class,
                        counterfactual_validity=vdata.get("counterfactual_validity"),
                        clinical_coherence=vdata.get("clinical_coherence"),
                        target_attribute_role=vdata.get("target_attribute_role"),
                        gold_answer_invariance=vdata.get("gold_answer_invariance"),
                        prior_shift_expected=vdata.get("prior_shift_expected"),
                        edit_locality=edit_locality,
                        raw_edit_locality=raw_edit_locality,
                        annotation_confidence=vdata.get("annotation_confidence"),
                    )
                )


def _load_pairs_flat(records, pairs):
    for record in records:
        qid = record["question_id"]
        original = record["original"]
        options = original["options"]

        gold_answer = original.get("answer_idx", "")
        if not gold_answer:
            for k, v in options.items():
                if v == original["answer"]:
                    gold_answer = k
                    break

        orig_prompt = format_prompt(original["question"], options)
        variants = record.get("counterfactuals", {}).get("variants", [])

        for vdata in variants:
            if not isinstance(vdata, dict):
                continue

            text = vdata.get("text")
            if text is None:
                continue

            raw_intervention_type = safe_str(vdata.get("intervention_type"))
            normalized_type = normalize_intervention_type(raw_intervention_type)
            attribute_type = normalize_attribute_type(normalized_type)

            label = safe_str(vdata.get("attribute_value_counterfactual")) or normalized_type

            analysis_class = classify_for_analysis(vdata)
            analysis_bucket = safe_str(vdata.get("analysis_bucket")) or None
            normalized_group = normalize_group(
                normalized_type,
                analysis_bucket=analysis_bucket,
                analysis_class=analysis_class,
            )

            if normalized_group == "control":
                analysis_class = "control"

            control_subtype = normalize_control_subtype(vdata, raw_intervention_type)

            raw_edit_locality = (
                safe_str(vdata.get("edit_locality"))
                or safe_str(vdata.get("edit_strength"))
                or safe_str(vdata.get("edit_scope"))
                or None
            )
            edit_locality = normalize_edit_locality(vdata)

            category = normalized_type

            cf_prompt = format_prompt(text, options)

            pairs.append(
                InferencePair(
                    question_id=qid,
                    category=category,
                    label=label,
                    attribute_type=attribute_type,
                    control_subtype=control_subtype,
                    raw_intervention_type=raw_intervention_type,
                    normalized_intervention_type=normalized_type,
                    normalized_group=normalized_group,
                    analysis_bucket=analysis_bucket,
                    original_prompt=orig_prompt,
                    counterfactual_prompt=cf_prompt,
                    original_question=original["question"],
                    cf_question=text,
                    gold_answer=gold_answer,
                    options=options,
                    analysis_class=analysis_class,
                    counterfactual_validity=vdata.get("counterfactual_validity"),
                    clinical_coherence=vdata.get("clinical_coherence"),
                    target_attribute_role=(
                        vdata.get("target_attribute_role")
                        or vdata.get("medical_relevance")
                    ),
                    gold_answer_invariance=vdata.get("gold_answer_invariance"),
                    prior_shift_expected=vdata.get("prior_shift_expected"),
                    edit_locality=edit_locality,
                    raw_edit_locality=raw_edit_locality,
                    annotation_confidence=vdata.get("annotation_confidence"),
                )
            )


def load_pairs(data_path: str) -> list[InferencePair]:
    with open(data_path) as f:
        records = json.load(f)

    pairs = []

    first_variants = None
    for r in records:
        first_variants = r.get("counterfactuals", {}).get("variants")
        if first_variants is not None:
            break

    if isinstance(first_variants, list):
        print("Detected flat-list variant format")
        _load_pairs_flat(records, pairs)
    elif isinstance(first_variants, dict):
        print("Detected legacy nested-dict variant format")
        _load_pairs_legacy(records, pairs)
    else:
        print("Warning: could not detect variant format cleanly, trying flat-list")
        _load_pairs_flat(records, pairs)

    cc = Counter(p.analysis_class for p in pairs)
    gc = Counter(p.normalized_group for p in pairs)
    tc = Counter(p.normalized_intervention_type for p in pairs)
    lc = Counter(p.edit_locality for p in pairs)
    ctrlc = Counter(p.control_subtype for p in pairs if p.normalized_group == "control")

    print(f"Loaded {len(records)} questions -> {len(pairs)} active pairs")
    print(f"  Analysis classes: {dict(cc)}")
    print(f"  Normalized groups: {dict(gc)}")
    print(f"  Intervention types: {dict(tc)}")
    print(f"  Edit locality: {dict(lc)}")
    if ctrlc:
        print(f"  Control subtypes: {dict(ctrlc)}")

    unknown_types = sorted({p.raw_intervention_type for p in pairs if p.normalized_group == "unknown"})
    if unknown_types:
        print(f"  WARNING: unknown normalized groups for intervention types: {unknown_types}")

    return pairs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: str,
    dtype=torch.float16,
    extract_attention: bool = False,
):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model (dtype={dtype})...")
    if extract_attention:
        print("  Attention extraction requested: forcing attention implementation to 'eager'")

    kwargs = dict(
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    if extract_attention:
        kwargs["attn_implementation"] = "eager"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError as e:
        # Older transformers may not accept attn_implementation; fall back gracefully.
        if extract_attention and "attn_implementation" in str(e):
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
            elif hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
        else:
            raise
    if device == "cpu":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, hidden_size={hidden_dim}")

    print("\n  Verifying hidden_states indexing...")
    dummy = tokenizer("test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    hs = out.hidden_states
    print(f"  output_hidden_states returns {len(hs)} tensors")
    print(f"  Index 0 shape: {hs[0].shape} (embedding output)")
    print(f"  Index {len(hs)-1} shape: {hs[-1].shape} (final layer)")
    assert len(hs) == n_layers + 1, \
        f"Expected {n_layers + 1} hidden states, got {len(hs)}"
    print(f"  ✓ Confirmed: hs[0]=embedding, hs[1..{n_layers}]=transformer blocks")
    print(f"  ✓ Using hs[1..{n_layers}] for layer analysis (skipping embedding)")

    return model, tokenizer


def get_answer_token_ids(tokenizer) -> dict:
    answer_ids = {}
    for letter in ["A", "B", "C", "D"]:
        candidates = [
            tokenizer.encode(letter, add_special_tokens=False),
            tokenizer.encode(f" {letter}", add_special_tokens=False),
        ]
        token_id = None
        for tokens in candidates:
            if tokens:
                token_id = tokens[-1]
                decoded = tokenizer.decode([token_id]).strip()
                if letter in decoded:
                    break
        if token_id is None:
            raise ValueError(f"Could not resolve token for '{letter}'")
        answer_ids[letter] = token_id
        print(f"  '{letter}' -> id={token_id} ('{tokenizer.decode([token_id])}')")
    return answer_ids


# ---------------------------------------------------------------------------
# Attention summary helpers
# ---------------------------------------------------------------------------

def summarize_attention_heads(
    attentions: tuple,
    n_tokens: int,
    topk: int,
    edit_positions: list[int] | None = None,
    largest_edit_positions: list[int] | None = None,
    stem_positions: list[int] | None = None,
) -> dict:
    """
    Build compact per-layer, per-head attention summaries (vectorized).

    Parameters
    ----------
    attentions : tuple of (1, n_heads, seq_len, seq_len) tensors, one per layer
    n_tokens : sequence length
    topk : number of top-attended positions to store per head
    edit_positions : token indices of the full edit region
    largest_edit_positions : token indices of the largest contiguous edit region
    stem_positions : token indices of the question stem

    Returns
    -------
    dict with keys:
        final_token:  per-layer, per-head summary from the final query position
        edit_region:  per-layer, per-head summary from the mean of edit query positions
        largest_edit: per-layer, per-head summary from the mean of largest edit query positions
    Each summary contains:
        topk_positions: (n_layers, n_heads, topk) int16
        topk_values:    (n_layers, n_heads, topk) float16
        edit_mass:      (n_layers, n_heads) float32
        largest_edit_mass: (n_layers, n_heads) float32
        stem_mass:      (n_layers, n_heads) float32
        entropy:        (n_layers, n_heads) float32
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[-1]

    # Precompute validated position lists (same for all layers)
    valid_edit = [p for p in (edit_positions or []) if 0 <= p < seq_len]
    valid_largest = [p for p in (largest_edit_positions or []) if 0 <= p < seq_len]
    valid_stem = [p for p in (stem_positions or []) if 0 <= p < seq_len]
    k = min(topk, seq_len)

    def _summarize_query(query_selector):
        """
        query_selector: callable(attn_layer_tensor) -> (n_heads, seq_len)
        All ops are batched across heads — no per-head Python loop.
        """
        topk_pos = np.zeros((n_layers, n_heads, topk), dtype=np.int16)
        topk_val = np.zeros((n_layers, n_heads, topk), dtype=np.float16)
        edit_mass_arr = np.zeros((n_layers, n_heads), dtype=np.float32)
        largest_mass_arr = np.zeros((n_layers, n_heads), dtype=np.float32)
        stem_mass_arr = np.zeros((n_layers, n_heads), dtype=np.float32)
        entropy_arr = np.zeros((n_layers, n_heads), dtype=np.float32)

        for layer_idx in range(n_layers):
            # query_attn: (n_heads, seq_len) — one distribution per head
            query_attn = query_selector(attentions[layer_idx]).float()

            # Top-k across all heads at once
            vals, idxs = query_attn.topk(k, dim=-1)  # both (n_heads, k)
            topk_pos[layer_idx, :, :k] = idxs.cpu().numpy().astype(np.int16)
            topk_val[layer_idx, :, :k] = vals.cpu().numpy().astype(np.float16)

            # Attention mass — vectorized gather+sum
            if valid_edit:
                edit_mass_arr[layer_idx] = query_attn[:, valid_edit].sum(dim=-1).cpu().numpy()
            if valid_largest:
                largest_mass_arr[layer_idx] = query_attn[:, valid_largest].sum(dim=-1).cpu().numpy()
            if valid_stem:
                stem_mass_arr[layer_idx] = query_attn[:, valid_stem].sum(dim=-1).cpu().numpy()

            # Entropy — vectorized: -sum(p * log(p)), clamping to avoid log(0)
            safe_p = query_attn.clamp(min=1e-30)
            entropy_arr[layer_idx] = -(query_attn * safe_p.log()).sum(dim=-1).cpu().numpy()

        return {
            "topk_positions": topk_pos,
            "topk_values": topk_val,
            "edit_mass": edit_mass_arr,
            "largest_edit_mass": largest_mass_arr,
            "stem_mass": stem_mass_arr,
            "entropy": entropy_arr,
        }

    result = {}

    # 1. Final token query
    result["final_token"] = _summarize_query(
        lambda attn: attn[0, :, -1, :]  # (n_heads, seq_len)
    )

    # 2. Edit region query (mean attention across edit query positions)
    if valid_edit:
        result["edit_region"] = _summarize_query(
            lambda attn: attn[0, :, valid_edit, :].mean(dim=1)
        )

    # 3. Largest contiguous edit region query
    if valid_largest:
        result["largest_edit"] = _summarize_query(
            lambda attn: attn[0, :, valid_largest, :].mean(dim=1)
        )

    return result


def find_stem_positions(
    tokenizer,
    question_text: str,
    full_prompt: str,
) -> list[int] | None:
    """Find token positions of the question stem within the full prompt."""
    if not question_text:
        return None
    stem_ids = tokenizer.encode(question_text.strip(), add_special_tokens=False)
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=True)

    # Reuse the same offset-finding logic
    for start in range(len(full_ids) - len(stem_ids) + 1):
        if full_ids[start:start + len(stem_ids)] == stem_ids:
            return list(range(start, start + len(stem_ids)))

    # Fallback with trimming
    for trim in range(1, min(5, len(stem_ids))):
        sub = stem_ids[trim:]
        for start in range(len(full_ids) - len(sub) + 1):
            if full_ids[start:start + len(sub)] == sub:
                actual_start = start - trim
                if actual_start >= 0:
                    return list(range(actual_start, actual_start + len(stem_ids)))

    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _forward_pass(model, tokenizer, prompt, extract_attention=False, max_length=2048):
    """Run model forward pass once. Returns (outputs, n_tokens)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=extract_attention,
        )
    return outputs, n_tokens


def _extract_from_outputs(
    outputs,
    n_tokens: int,
    answer_token_ids: dict,
    edit_positions: list[int] | None = None,
    largest_edit_positions: list[int] | None = None,
    stem_positions: list[int] | None = None,
    extract_attention: bool = False,
    attention_topk: int = 10,
) -> dict:
    """
    Extract packed result dict from cached forward pass outputs.

    Can be called multiple times on the same outputs with different
    position sets — avoids duplicate forward passes.
    """
    final_logits = outputs.logits[0, -1, :]
    logits_abcd = np.array(
        [
            final_logits[answer_token_ids[k]].float().cpu().item()
            for k in ["A", "B", "C", "D"]
        ],
        dtype=np.float32,
    )

    hs = outputs.hidden_states
    n_layers = len(hs) - 1

    # Store as float16 to keep memory/disk manageable on large runs.
    hidden_final = np.stack(
        [hs[l][0, -1, :].to(torch.float16).cpu().numpy() for l in range(1, n_layers + 1)]
    )

    d = {
        "logits_abcd": logits_abcd,
        "hidden_final": hidden_final,
        "n_tokens": n_tokens,
    }

    if edit_positions:
        valid_pos = [p for p in edit_positions if 0 <= p < n_tokens]
        if valid_pos:
            d["hidden_at_edit"] = np.stack(
                [
                    hs[l][0, valid_pos, :].to(torch.float16).cpu().mean(dim=0).numpy()
                    for l in range(1, n_layers + 1)
                ]
            )

    attentions = getattr(outputs, "attentions", None)
    if extract_attention and attentions is not None and len(attentions) > 0:
        d["attention_summary"] = summarize_attention_heads(
            attentions=attentions,
            n_tokens=n_tokens,
            topk=attention_topk,
            edit_positions=edit_positions,
            largest_edit_positions=largest_edit_positions,
            stem_positions=stem_positions,
        )

    return d


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(path, original_results, cf_results, token_diffs,
                     completed_qids, pairs_completed):
    """Save extraction progress to a checkpoint file (atomic write)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "original_results": original_results,
            "cf_results": cf_results,
            "token_diffs": token_diffs,
            "completed_qids": list(completed_qids),
            "pairs_completed": pairs_completed,
        },
        tmp_path,
    )
    # Atomic replace to avoid corrupting checkpoint on process kill.
    tmp_path.replace(path)
    fsize = path.stat().st_size / (1024 ** 2)
    print(f"  Checkpoint saved: {fsize:.1f} MB, "
          f"{len(completed_qids)} qids, {pairs_completed} pairs done")


def _load_checkpoint(path):
    """Load checkpoint. Returns (orig_results, cf_results, token_diffs, completed_qids, pairs_completed)."""
    path = Path(path)
    try:
        ckpt = torch.load(path, weights_only=False)
    except (EOFError, RuntimeError, ValueError) as e:
        # Common when killed mid-write: "EOFError: Ran out of input"
        ts = time.strftime("%Y%m%d-%H%M%S")
        bad_path = path.with_suffix(path.suffix + f".corrupt-{ts}")
        try:
            path.replace(bad_path)
            print(f"WARNING: Checkpoint was corrupt ({e}). Moved to: {bad_path}")
        except Exception:
            print(f"WARNING: Checkpoint was corrupt ({e}). Could not move it aside.")
        raise
    return (
        ckpt["original_results"],
        ckpt["cf_results"],
        ckpt["token_diffs"],
        set(ckpt["completed_qids"]),
        ckpt["pairs_completed"],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_all(
    model,
    tokenizer,
    pairs,
    answer_token_ids,
    extract_attention: bool = False,
    attention_topk: int = 10,
    max_length: int = 2048,
    attention_max_length: int = 512,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 100,
    resume: bool = False,
) -> dict:
    # --- Group pairs by question_id for single-pass optimization ---
    # Each original gets ONE forward pass reused for all its CF pairs.
    pairs_by_qid = {}
    for idx, p in enumerate(pairs):
        if p.question_id not in pairs_by_qid:
            pairs_by_qid[p.question_id] = []
        pairs_by_qid[p.question_id].append((idx, p))

    qid_order = list(pairs_by_qid.keys())

    # --- Load checkpoint if resuming ---
    original_results = {}
    cf_results = {}
    token_diffs = {}
    completed_qids = set()
    pairs_completed = 0

    if checkpoint_path and resume and Path(checkpoint_path).exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        try:
            original_results, cf_results, token_diffs, completed_qids, pairs_completed = \
                _load_checkpoint(checkpoint_path)
            print(f"  {len(completed_qids)} qids / {pairs_completed} pairs already done")
        except Exception:
            print("  WARNING: Could not resume from checkpoint; starting fresh.")
            original_results, cf_results, token_diffs, completed_qids, pairs_completed = \
                {}, {}, {}, set(), 0

    remaining_qids = [q for q in qid_order if q not in completed_qids]
    total_remaining = sum(len(pairs_by_qid[q]) for q in remaining_qids)

    print(f"\nTotal: {len(qid_order)} unique questions, {len(pairs)} pairs")
    if completed_qids:
        print(f"Remaining: {len(remaining_qids)} questions, {total_remaining} pairs")
    if extract_attention:
        print(f"  Attention extraction ENABLED (topk={attention_topk})")
        print(f"  Attention max_length={attention_max_length} (quadratic memory)")
    else:
        print(f"  max_length={max_length}")

    t0 = time.time()
    pairs_since_ckpt = 0

    for qid_idx, qid in enumerate(remaining_qids):
        qid_pairs = pairs_by_qid[qid]
        first_pair = qid_pairs[0][1]

        # --- Single forward pass for this original ---
        orig_stem_pos = None
        if extract_attention:
            orig_stem_pos = find_stem_positions(
                tokenizer, first_pair.original_question, first_pair.original_prompt
            )

        orig_outputs, orig_n_tokens = _forward_pass(
            model,
            tokenizer,
            first_pair.original_prompt,
            extract_attention,
            max_length=attention_max_length if extract_attention else max_length,
        )

        # Base original extraction (no edit positions)
        original_results[qid] = _extract_from_outputs(
            orig_outputs, orig_n_tokens, answer_token_ids,
            stem_positions=orig_stem_pos,
            extract_attention=extract_attention,
            attention_topk=attention_topk,
        )

        # --- Process each CF pair for this qid ---
        for _, pair in qid_pairs:
            pair_key = f"{pair.question_id}__{pair.normalized_intervention_type}__{pair.label}"

            done_total = pairs_completed + pairs_since_ckpt
            if (done_total + 1) % 25 == 0 or done_total == 0:
                elapsed = time.time() - t0
                rate = (pairs_since_ckpt + 1) / elapsed if elapsed > 0 else 0
                remaining = total_remaining - pairs_since_ckpt - 1
                eta = remaining / rate / 60 if rate > 0 else 0
                print(f"  [{done_total+1}/{len(pairs)}] {pair_key} "
                      f"({rate:.1f}/s, ETA {eta:.1f}m)")

            # Token diff
            diff_info = find_edited_token_positions(
                tokenizer, pair.original_question, pair.cf_question,
                pair.original_prompt, pair.counterfactual_prompt,
            )
            token_diffs[pair_key] = diff_info

            if pairs_since_ckpt < 5 and diff_info.get("debug_context"):
                dc = diff_info["debug_context"]
                print(f"    DEBUG: n_changed={diff_info['n_tokens_changed']}, "
                      f"orig_offset={diff_info['orig_offset']}, "
                      f"cf_offset={diff_info['cf_offset']}")
                if "orig_window" in dc:
                    print(f"    ORIG window: ...{dc['orig_window']}...")
                if "cf_window" in dc:
                    print(f"    CF   window: ...{dc['cf_window']}...")

            cf_edit_pos = diff_info["cf_edit_positions"] if not diff_info["alignment_failed"] else None
            cf_largest_pos = diff_info["largest_cf_positions"] if not diff_info["alignment_failed"] else None

            cf_stem_pos = None
            if extract_attention:
                cf_stem_pos = find_stem_positions(
                    tokenizer, pair.cf_question, pair.counterfactual_prompt
                )

            # --- Single forward pass for this CF ---
            cf_outputs, cf_n_tokens = _forward_pass(
                model,
                tokenizer,
                pair.counterfactual_prompt,
                extract_attention,
                max_length=attention_max_length if extract_attention else max_length,
            )

            # Primary CF extraction
            cf_results[pair_key] = _extract_from_outputs(
                cf_outputs, cf_n_tokens, answer_token_ids,
                edit_positions=cf_edit_pos,
                largest_edit_positions=cf_largest_pos,
                stem_positions=cf_stem_pos,
                extract_attention=extract_attention,
                attention_topk=attention_topk,
            )

            # Largest CF extraction — reuses same forward pass, no duplicate!
            if cf_largest_pos and cf_largest_pos != cf_edit_pos:
                cf_results[f"{pair_key}__largest"] = _extract_from_outputs(
                    cf_outputs, cf_n_tokens, answer_token_ids,
                    edit_positions=cf_largest_pos,
                    largest_edit_positions=cf_largest_pos,
                    stem_positions=cf_stem_pos,
                    extract_attention=extract_attention,
                    attention_topk=attention_topk,
                )

            del cf_outputs
            if torch.backends.mps.is_available() and (pairs_since_ckpt % 25 == 0):
                torch.mps.empty_cache()
                gc.collect()

            # Original with edit positions — reuses cached orig forward pass!
            if not diff_info["alignment_failed"]:
                orig_edit_key = f"{qid}__edit__{pair_key}"
                original_results[orig_edit_key] = _extract_from_outputs(
                    orig_outputs, orig_n_tokens, answer_token_ids,
                    edit_positions=diff_info["orig_edit_positions"],
                    largest_edit_positions=diff_info["largest_orig_positions"],
                    stem_positions=orig_stem_pos,
                    extract_attention=extract_attention,
                    attention_topk=attention_topk,
                )

                orig_largest_pos = diff_info["largest_orig_positions"]
                if orig_largest_pos and orig_largest_pos != diff_info["orig_edit_positions"]:
                    orig_largest_key = f"{qid}__largest__{pair_key}"
                    original_results[orig_largest_key] = _extract_from_outputs(
                        orig_outputs, orig_n_tokens, answer_token_ids,
                        edit_positions=orig_largest_pos,
                        largest_edit_positions=orig_largest_pos,
                        stem_positions=orig_stem_pos,
                        extract_attention=extract_attention,
                        attention_topk=attention_topk,
                    )

            pairs_since_ckpt += 1

        # Done with this qid — free original forward pass outputs
        del orig_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()

        completed_qids.add(qid)

        # Save checkpoint periodically
        if checkpoint_path and checkpoint_every > 0 and pairs_since_ckpt >= checkpoint_every:
            _save_checkpoint(
                checkpoint_path, original_results, cf_results, token_diffs,
                completed_qids, pairs_completed + pairs_since_ckpt,
            )
            pairs_completed += pairs_since_ckpt
            pairs_since_ckpt = 0
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                gc.collect()

    # Final checkpoint
    if checkpoint_path and checkpoint_every > 0 and pairs_since_ckpt > 0:
        pairs_completed += pairs_since_ckpt
        _save_checkpoint(
            checkpoint_path, original_results, cf_results, token_diffs,
            completed_qids, pairs_completed,
        )

    # --- Build metadata ---
    pair_metadata = []
    for p in pairs:
        pk = f"{p.question_id}__{p.normalized_intervention_type}__{p.label}"
        td = token_diffs[pk]
        pair_metadata.append(
            {
                "question_id": p.question_id,
                "category": p.category,
                "label": p.label,
                "attribute_type": p.attribute_type,
                "control_subtype": p.control_subtype,

                "raw_intervention_type": p.raw_intervention_type,
                "normalized_intervention_type": p.normalized_intervention_type,
                "normalized_group": p.normalized_group,
                "analysis_bucket": p.analysis_bucket,

                "gold_answer": p.gold_answer,
                "analysis_class": p.analysis_class,
                "counterfactual_validity": p.counterfactual_validity,
                "clinical_coherence": p.clinical_coherence,
                "target_attribute_role": p.target_attribute_role,
                "gold_answer_invariance": p.gold_answer_invariance,
                "prior_shift_expected": p.prior_shift_expected,
                "edit_locality": p.edit_locality,
                "raw_edit_locality": p.raw_edit_locality,
                "annotation_confidence": p.annotation_confidence,

                "pair_key": pk,
                "orig_edit_key": f"{p.question_id}__edit__{pk}",
                "orig_largest_key": f"{p.question_id}__largest__{pk}",
                "cf_largest_key": f"{pk}__largest",

                "alignment_failed": td["alignment_failed"],
                "n_tokens_changed": td["n_tokens_changed"],
                "n_largest_region_tokens": td["n_largest_region_tokens"],
                "n_edit_regions": td["n_edit_regions"],
                "char_edit_distance": td["char_edit_distance"],
                "token_edit_ratio": td["token_edit_ratio"],
                "n_orig_tokens": td["n_orig_tokens"],
                "n_cf_tokens": td["n_cf_tokens"],
                "stem_orig_len": td["stem_orig_len"],
                "stem_cf_len": td["stem_cf_len"],

                "orig_edit_positions": td["orig_edit_positions"],
                "cf_edit_positions": td["cf_edit_positions"],
                "largest_orig_positions": td["largest_orig_positions"],
                "largest_cf_positions": td["largest_cf_positions"],
                "n_orig_edit_positions": len(td["orig_edit_positions"]),
                "n_cf_edit_positions": len(td["cf_edit_positions"]),

                "debug_context": td.get("debug_context", {}),
            }
        )

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    model_config = {
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "n_heads": n_heads,
        "note": (
            "hidden_final is (n_layers, hidden_dim) = transformer "
            "block outputs only (no embedding layer). "
            f"L0=block1 output ... L{n_layers-1}=block{n_layers} output."
        ),
    }

    if extract_attention:
        model_config["attention_topk"] = attention_topk
        model_config["attention_note"] = (
            "attention_summary contains per-query-type dicts "
            "(final_token, edit_region, largest_edit). "
            "Each has: topk_positions (n_layers, n_heads, topk) int16, "
            "topk_values (n_layers, n_heads, topk) float16, "
            "edit_mass/largest_edit_mass/stem_mass (n_layers, n_heads) float32, "
            "entropy (n_layers, n_heads) float32."
        )

    return {
        "original_results": original_results,
        "cf_results": cf_results,
        "pair_metadata": pair_metadata,
        "answer_token_ids": answer_token_ids,
        "model_config": model_config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 v3: Extract logits + localized hidden states + attention head summaries"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="stage1_extractions_v3.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument(
        "--extract_attention",
        action="store_true",
        help="Extract per-head attention summaries (increases runtime/memory)",
    )
    parser.add_argument(
        "--attention_topk",
        type=int,
        default=10,
        help="Number of top-attended source positions to store per head (default: 10)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Tokenizer max_length when NOT extracting attention (default: 2048).",
    )
    parser.add_argument(
        "--attention_max_length",
        type=int,
        default=512,
        help="Tokenizer max_length when extracting attention (default: 512).",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save checkpoint every N pairs (default: 100). Set to 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    pairs = load_pairs(args.data_path)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to {len(pairs)} pairs")

    model, tokenizer = load_model(
        args.model_path, device, extract_attention=args.extract_attention
    )

    print("\nAnswer token IDs:")
    answer_token_ids = get_answer_token_ids(tokenizer)

    # Checkpoint path
    checkpoint_path = None
    if args.checkpoint_every > 0:
        checkpoint_path = str(Path(args.output).with_suffix(".ckpt"))

    results = extract_all(
        model,
        tokenizer,
        pairs,
        answer_token_ids,
        extract_attention=args.extract_attention,
        attention_topk=args.attention_topk,
        max_length=args.max_length,
        attention_max_length=args.attention_max_length,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    print(f"\nSaving to {args.output}...")
    torch.save(results, args.output)
    fsize = Path(args.output).stat().st_size / (1024 ** 2)
    print(f"Done. {fsize:.1f} MB")

    # Clean up checkpoint after successful completion
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"Checkpoint cleaned up: {checkpoint_path}")


if __name__ == "__main__":
    main()