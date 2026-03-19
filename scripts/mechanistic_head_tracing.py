#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Memory-safe Stage 1 Extraction
==============================

Goals
-----
1. Avoid storing full attention tensors.
2. Save only compact summaries needed for candidate-head discovery.
3. Write results incrementally in shards instead of holding everything in RAM.
4. Preserve compatibility with downstream Stage 1 / Stage 2 analyses.

What this extracts
------------------
For each original / counterfactual prompt:
- logits_abcd: answer logits for A/B/C/D
- hidden_final: final-token hidden states for transformer blocks only
- hidden_at_edit: pooled hidden states at edit-token positions (optional)
- hidden_at_largest: pooled hidden states at largest edit region (optional)
- attention_summary:
    {
      "final_token": {
         "edit_mass": (layers, heads),
         "largest_edit_mass": (layers, heads),
         "stem_mass": (layers, heads),
         "entropy": (layers, heads),
         "topk_positions": (layers, heads, k),
         "topk_values": (layers, heads, k),
      },
      "edit_region": {...} optional,
      "largest_edit": {...} optional,
    }

Key memory protections
----------------------
- Uses output_attentions=True only during a single forward pass.
- Summarizes each layer's attention immediately.
- Never stores (layers, heads, seq, seq) arrays.
- Writes shards every N records.
- Supports chunking via --start_idx / --end_idx / --max_pairs.

Usage
-----
python extract_representations_memory_safe.py \
  --model_path /path/to/model \
  --data_path /path/to/data.json \
  --output_dir /path/to/extractions_memsafe \
  --device auto \
  --shard_size 64 \
  --attn_topk 8

Notes
-----
- For LLaMA-family models, hidden_states[0] is embedding output.
- We keep hidden_states[1:] only, so L0 = block1 output.
"""

import os
import gc
import json
import time
import math
import argparse
import difflib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional

import numpy as np
import torch
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
        question=question_text.strip(),
        A=options.get("A", ""),
        B=options.get("B", ""),
        C=options.get("C", ""),
        D=options.get("D", ""),
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InferencePair:
    question_id: str
    category: str
    label: str
    attribute_type: str
    control_subtype: str
    original_prompt: str
    counterfactual_prompt: str
    original_question: str
    cf_question: str
    gold_answer: str
    options: dict
    analysis_class: str
    counterfactual_validity: Optional[str]
    clinical_coherence: Optional[str]
    target_attribute_role: Optional[str]
    gold_answer_invariance: Optional[str]
    prior_shift_expected: Optional[str]
    edit_locality: Optional[str]
    annotation_confidence: Optional[float]
    intervention_type: Optional[str] = None
    intervention_family: Optional[str] = None
    analysis_bucket: Optional[str] = None
    edit_scope: Optional[str] = None
    edit_strength: Optional[str] = None
    attribute_value_counterfactual: Optional[str] = None
    medical_relevance: Optional[str] = None
    social_bias_salience: Optional[str] = None


@dataclass
class ExtractionResult:
    logits_abcd: np.ndarray
    hidden_final: np.ndarray
    hidden_at_edit: Optional[np.ndarray]
    hidden_at_largest: Optional[np.ndarray]
    n_tokens: int
    attention_summary: Optional[dict]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

CATEGORIES = {
    "gender": ["male", "female", "neutral"],
    "age": ["young_adult", "middle_aged", "elderly"],
    "race_ethnicity": [
        "White", "Black/African American", "Hispanic/Latino",
        "Asian", "no_race_specified"
    ],
    "control": ["neutral_rework", "irrelevant_surface"],
}

CATEGORY_TO_ATTR = {
    "gender": "gender",
    "age": "age",
    "race_ethnicity": "race",
    "control": "control",
}

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


def classify_for_analysis(vdata: dict) -> str:
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

                control_subtype = label if category == "control" else "none"
                cf_prompt = format_prompt(text, options)

                pairs.append(InferencePair(
                    question_id=qid,
                    category=category,
                    label=label,
                    attribute_type=CATEGORY_TO_ATTR[category],
                    control_subtype=control_subtype,
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
                    edit_locality=vdata.get("edit_locality"),
                    annotation_confidence=vdata.get("annotation_confidence"),
                ))


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

            itype = vdata.get("intervention_type", "")
            cat_info = INTERVENTION_TYPE_TO_CATEGORY.get(itype)
            if cat_info is None:
                continue
            category, attribute_type = cat_info

            label = str(vdata.get("attribute_value_counterfactual", itype))
            analysis_class = classify_for_analysis(vdata)
            if category == "control" and text is not None:
                analysis_class = "control"

            control_subtype = itype if category == "control" else "none"
            cf_prompt = format_prompt(text, options)

            pairs.append(InferencePair(
                question_id=qid,
                category=category,
                label=label,
                attribute_type=attribute_type,
                control_subtype=control_subtype,
                original_prompt=orig_prompt,
                counterfactual_prompt=cf_prompt,
                original_question=original["question"],
                cf_question=text,
                gold_answer=gold_answer,
                options=options,
                analysis_class=analysis_class,
                counterfactual_validity=vdata.get("counterfactual_validity"),
                clinical_coherence=vdata.get("clinical_coherence"),
                target_attribute_role=vdata.get(
                    "target_attribute_role",
                    vdata.get("medical_relevance")
                ),
                gold_answer_invariance=vdata.get("gold_answer_invariance"),
                prior_shift_expected=vdata.get("prior_shift_expected"),
                edit_locality=vdata.get("edit_locality", vdata.get("edit_scope")),
                annotation_confidence=vdata.get("annotation_confidence"),
                intervention_type=vdata.get("intervention_type"),
                intervention_family=vdata.get("intervention_family"),
                analysis_bucket=vdata.get("analysis_bucket"),
                edit_scope=vdata.get("edit_scope"),
                edit_strength=vdata.get("edit_strength"),
                attribute_value_counterfactual=vdata.get("attribute_value_counterfactual"),
                medical_relevance=vdata.get("medical_relevance"),
                social_bias_salience=vdata.get("social_bias_salience"),
            ))


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
        print("Warning: unknown format, trying flat-list loader")
        _load_pairs_flat(records, pairs)

    cc = Counter(p.analysis_class for p in pairs)
    print(f"Loaded {len(records)} questions -> {len(pairs)} active pairs")
    print(f"  Classes: {dict(cc)}")
    return pairs


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def build_char_to_token_map(text: str, token_ids: list[int], tokenizer) -> list[Optional[int]]:
    char_map = [None] * len(text)
    if not text:
        return char_map

    decoded_tokens = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids]
    char_pos = 0

    for tok_idx, tok_text in enumerate(decoded_tokens):
        clean = tok_text.replace("▁", " ").replace("Ġ", " ")
        if not clean:
            continue

        for ch in clean:
            if char_pos >= len(text):
                break
            char_map[char_pos] = tok_idx
            char_pos += 1

    return char_map


def find_stem_offset(full_ids: list[int], stem_ids: list[int]) -> Optional[int]:
    for start in range(len(full_ids) - len(stem_ids) + 1):
        if full_ids[start:start + len(stem_ids)] == stem_ids:
            return start
    for trim in range(1, min(5, len(stem_ids))):
        sub = stem_ids[trim:]
        for start in range(len(full_ids) - len(sub) + 1):
            if full_ids[start:start + len(sub)] == sub:
                return start - trim
    return None


def find_edited_token_positions(tokenizer, orig_question, cf_question, orig_prompt, cf_prompt) -> dict:
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

    for _, oi1, oi2, ci1, ci2 in char_edit_regions:
        for c in range(oi1, oi2):
            if c < len(orig_c2t) and orig_c2t[c] is not None:
                stem_orig_edit_tokens.add(orig_c2t[c])
        for c in range(ci1, ci2):
            if c < len(cf_c2t) and cf_c2t[c] is not None:
                stem_cf_edit_tokens.add(cf_c2t[c])

    primary_orig = set()
    primary_cf = set()
    if char_edit_regions:
        biggest_idx = max(
            range(len(char_edit_regions)),
            key=lambda i: max(
                char_edit_regions[i][2] - char_edit_regions[i][1],
                char_edit_regions[i][4] - char_edit_regions[i][3]
            )
        )
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
    alignment_failed = (orig_offset is None or cf_offset is None)

    if alignment_failed:
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
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str, dtype):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model...")
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        kwargs["device_map"] = "auto"
    elif device == "cpu":
        kwargs["device_map"] = None
    elif device == "mps":
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager", **kwargs)
    if device in {"cpu", "mps"}:
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  layers={n_layers}, hidden_size={hidden_dim}")

    dummy = tokenizer("test", return_tensors="pt")
    dummy = {k: v.to(model.device) for k, v in dummy.items()}
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    assert len(hs) == n_layers + 1, f"Expected {n_layers + 1} hidden states, got {len(hs)}"
    print(f"  hidden_states verified: hs[0]=embedding, hs[1..{n_layers}]=blocks")
    return model, tokenizer


def get_answer_token_ids(tokenizer) -> dict:
    answer_ids = {}
    for letter in ["A", "B", "C", "D"]:
        candidates = [
            tokenizer.encode(letter, add_special_tokens=False),
            tokenizer.encode(f" {letter}", add_special_tokens=False),
        ]
        token_id = None
        for toks in candidates:
            if toks:
                token_id = toks[-1]
                decoded = tokenizer.decode([token_id]).strip()
                if letter in decoded:
                    break
        if token_id is None:
            raise ValueError(f"Could not resolve token for {letter}")
        answer_ids[letter] = token_id
        print(f"  {letter} -> {token_id} ({tokenizer.decode([token_id])!r})")
    return answer_ids


# ---------------------------------------------------------------------------
# Attention summaries
# ---------------------------------------------------------------------------

def attention_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def summarize_single_query_attention(
    attn_vec: np.ndarray,
    edit_positions: Optional[list[int]],
    largest_positions: Optional[list[int]],
    question_positions: Optional[list[int]],
    topk: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    seq_len = attn_vec.shape[0]

    def mass_at(pos_list):
        if not pos_list:
            return 0.0
        valid = [p for p in pos_list if 0 <= p < seq_len]
        if not valid:
            return 0.0
        return float(attn_vec[valid].sum())

    edit_mass = mass_at(edit_positions)
    largest_mass = mass_at(largest_positions)
    stem_mass = mass_at(question_positions)
    ent = attention_entropy(attn_vec)

    k = min(topk, seq_len)
    top_idx = np.argpartition(attn_vec, -k)[-k:]
    top_idx = top_idx[np.argsort(attn_vec[top_idx])[::-1]]
    top_vals = attn_vec[top_idx]

    if k < topk:
        pad_n = topk - k
        top_idx = np.concatenate([top_idx, -1 * np.ones(pad_n, dtype=np.int64)])
        top_vals = np.concatenate([top_vals, np.zeros(pad_n, dtype=np.float32)])

    return edit_mass, largest_mass, stem_mass, ent, top_idx.astype(np.int64), top_vals.astype(np.float32)


def summarize_attention_for_queries(
    attentions,
    query_positions: list[int],
    edit_positions: Optional[list[int]],
    largest_positions: Optional[list[int]],
    question_positions: Optional[list[int]],
    topk: int,
) -> Optional[dict]:
    """
    attentions is a tuple of length n_layers.
    each tensor is (batch=1, heads, tgt_len, src_len)
    """
    if not attentions or not query_positions:
        return None

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    edit_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    largest_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    stem_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    entropy = np.zeros((n_layers, n_heads), dtype=np.float32)
    topk_positions = np.full((n_layers, n_heads, topk), -1, dtype=np.int64)
    topk_values = np.zeros((n_layers, n_heads, topk), dtype=np.float32)

    for l, layer_attn in enumerate(attentions):
        # (1, heads, tgt_len, src_len)
        layer_np = layer_attn[0].detach().float().cpu().numpy()

        valid_q = [q for q in query_positions if 0 <= q < layer_np.shape[1]]
        if not valid_q:
            continue

        # mean over chosen query positions -> (heads, src_len)
        head_src = layer_np[:, valid_q, :].mean(axis=1)

        for h in range(n_heads):
            em, lm, sm, ent, tidx, tvals = summarize_single_query_attention(
                head_src[h],
                edit_positions=edit_positions,
                largest_positions=largest_positions,
                question_positions=question_positions,
                topk=topk,
            )
            edit_mass[l, h] = em
            largest_mass[l, h] = lm
            stem_mass[l, h] = sm
            entropy[l, h] = ent
            topk_positions[l, h] = tidx
            topk_values[l, h] = tvals

    return {
        "edit_mass": edit_mass,
        "largest_edit_mass": largest_mass,
        "stem_mass": stem_mass,
        "entropy": entropy,
        "topk_positions": topk_positions,
        "topk_values": topk_values,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def to_numpy_stack_hidden(hidden_states) -> np.ndarray:
    # skip embedding hs[0]
    n_layers = len(hidden_states) - 1
    return np.stack([
        hidden_states[l][0].detach().float().cpu().numpy()
        for l in range(1, n_layers + 1)
    ])


def pooled_hidden_at_positions(hidden_states, positions: list[int]) -> Optional[np.ndarray]:
    if not positions:
        return None

    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[1].shape[1]
    valid = [p for p in positions if 0 <= p < seq_len]
    if not valid:
        return None

    return np.stack([
        hidden_states[l][0, valid, :].detach().float().cpu().mean(dim=0).numpy()
        for l in range(1, n_layers + 1)
    ])


def run_inference(
    model,
    tokenizer,
    prompt: str,
    answer_token_ids: dict,
    extract_attention: bool = False,
    attn_topk: int = 8,
    edit_positions: Optional[list[int]] = None,
    largest_positions: Optional[list[int]] = None,
    question_positions: Optional[list[int]] = None,
) -> ExtractionResult:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=extract_attention,
            use_cache=False,
        )

    final_logits = outputs.logits[0, -1, :]
    logits_abcd = np.array([
        final_logits[answer_token_ids[k]].float().cpu().item()
        for k in ["A", "B", "C", "D"]
    ], dtype=np.float32)

    hs = outputs.hidden_states
    n_layers = len(hs) - 1

    hidden_final = np.stack([
        hs[l][0, -1, :].detach().float().cpu().numpy()
        for l in range(1, n_layers + 1)
    ])

    hidden_at_edit = pooled_hidden_at_positions(hs, edit_positions or [])
    hidden_at_largest = pooled_hidden_at_positions(hs, largest_positions or [])

    attention_summary = None
    if extract_attention and outputs.attentions is not None:
        final_query_positions = [n_tokens - 1]
        attention_summary = {
            "final_token": summarize_attention_for_queries(
                outputs.attentions,
                query_positions=final_query_positions,
                edit_positions=edit_positions,
                largest_positions=largest_positions,
                question_positions=question_positions,
                topk=attn_topk,
            )
        }

        if edit_positions:
            attention_summary["edit_region"] = summarize_attention_for_queries(
                outputs.attentions,
                query_positions=edit_positions,
                edit_positions=edit_positions,
                largest_positions=largest_positions,
                question_positions=question_positions,
                topk=attn_topk,
            )

        if largest_positions:
            attention_summary["largest_edit"] = summarize_attention_for_queries(
                outputs.attentions,
                query_positions=largest_positions,
                edit_positions=edit_positions,
                largest_positions=largest_positions,
                question_positions=question_positions,
                topk=attn_topk,
            )

    # aggressively clear large tensors
    del outputs
    if model.device.type == "cuda":
        torch.cuda.empty_cache()

    return ExtractionResult(
        logits_abcd=logits_abcd,
        hidden_final=hidden_final,
        hidden_at_edit=hidden_at_edit,
        hidden_at_largest=hidden_at_largest,
        n_tokens=n_tokens,
        attention_summary=attention_summary,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def pack_extraction_result(r: ExtractionResult) -> dict:
    out = {
        "logits_abcd": r.logits_abcd,
        "hidden_final": r.hidden_final,
        "n_tokens": r.n_tokens,
    }
    if r.hidden_at_edit is not None:
        out["hidden_at_edit"] = r.hidden_at_edit
    if r.hidden_at_largest is not None:
        out["hidden_at_largest"] = r.hidden_at_largest
    if r.attention_summary is not None:
        out["attention_summary"] = r.attention_summary
    return out


def save_shard(output_dir: Path, shard_idx: int, payload: dict):
    path = output_dir / f"extractions_shard_{shard_idx:04d}.pt"
    torch.save(payload, path)
    print(f"  saved {path.name}")


def load_processed_keys(output_dir: Path) -> tuple[set, set]:
    """Scan existing shards and return (processed_orig_keys, processed_cf_keys)."""
    shard_files = sorted(output_dir.glob("extractions_shard_*.pt"))
    if not shard_files:
        return set(), set()

    processed_orig = set()
    processed_cf = set()
    print(f"Resuming: scanning {len(shard_files)} existing shards for processed keys...")
    for path in shard_files:
        try:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            processed_orig.update(shard.get("original_results", {}).keys())
            processed_cf.update(shard.get("cf_results", {}).keys())
        except Exception as e:
            print(f"  Warning: could not read {path.name}: {e}")
    print(f"  Found {len(processed_orig)} processed originals, {len(processed_cf)} processed CFs")
    return processed_orig, processed_cf


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_all(
    model,
    tokenizer,
    pairs: list[InferencePair],
    answer_token_ids: dict,
    output_dir: Path,
    shard_size: int,
    extract_attention: bool,
    attn_topk: int,
    resume: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_orig_keys: set = set()
    processed_cf_keys: set = set()
    if resume:
        processed_orig_keys, processed_cf_keys = load_processed_keys(output_dir)

    unique_originals = {}
    for p in pairs:
        if p.question_id not in unique_originals:
            unique_originals[p.question_id] = p

    original_qids = list(unique_originals.keys())
    total_items = len(original_qids) + len(pairs)

    manifest = {
        "model_config": {
            "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "note": (
                "hidden_final is (n_layers, hidden_dim), transformer block outputs only; "
                f"L0=block1 output ... L{model.config.num_hidden_layers - 1}=last block output."
            ),
            "extract_attention": extract_attention,
            "attn_topk": attn_topk,
        },
        "answer_token_ids": answer_token_ids,
        "pair_metadata": [],
        "shards": [],
    }

    # On resume, pre-populate shards list with existing shard files so the
    # final manifest covers all data, not just the resumed portion.
    if resume:
        for sf in sorted(output_dir.glob("extractions_shard_*.pt")):
            idx = int(sf.stem.split("_")[-1])
            try:
                sd = torch.load(sf, map_location="cpu", weights_only=False)
                manifest["shards"].append({
                    "shard_idx": idx,
                    "filename": sf.name,
                    "n_original_results": len(sd.get("original_results", {})),
                    "n_cf_results": len(sd.get("cf_results", {})),
                })
            except Exception as e:
                print(f"  Warning: could not pre-register {sf.name}: {e}")

    current_orig = {}
    current_cf = {}
    existing_shards = sorted(output_dir.glob("extractions_shard_*.pt"))
    shard_idx = len(existing_shards)
    items_in_shard = 0

    def flush_current_shard():
        nonlocal current_orig, current_cf, shard_idx, items_in_shard
        if not current_orig and not current_cf:
            return
        payload = {
            "original_results": current_orig,
            "cf_results": current_cf,
        }
        save_shard(output_dir, shard_idx, payload)
        manifest["shards"].append({
            "shard_idx": shard_idx,
            "filename": f"extractions_shard_{shard_idx:04d}.pt",
            "n_original_results": len(current_orig),
            "n_cf_results": len(current_cf),
        })
        current_orig = {}
        current_cf = {}
        shard_idx += 1
        items_in_shard = 0
        gc.collect()
        if model.device.type == "cuda":
            torch.cuda.empty_cache()

    # originals
    print("\n--- extracting originals ---")
    t0 = time.time()
    for i, qid in enumerate(original_qids):
        if qid in processed_orig_keys:
            continue
        pair = unique_originals[qid]

        stem_ids = tokenizer.encode(pair.original_question, add_special_tokens=False)
        full_ids = tokenizer.encode(pair.original_prompt, add_special_tokens=True)
        stem_offset = find_stem_offset(full_ids, stem_ids)
        if stem_offset is None:
            question_positions = []
        else:
            question_positions = list(range(stem_offset, stem_offset + len(stem_ids)))

        res = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=pair.original_prompt,
            answer_token_ids=answer_token_ids,
            extract_attention=extract_attention,
            attn_topk=attn_topk,
            edit_positions=None,
            largest_positions=None,
            question_positions=question_positions,
        )
        current_orig[qid] = pack_extraction_result(res)
        items_in_shard += 1

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"  originals [{i + 1}/{len(original_qids)}] ({rate:.2f}/s)")

        if items_in_shard >= shard_size:
            flush_current_shard()

    # counterfactuals
    print("\n--- extracting counterfactuals ---")
    t1 = time.time()
    for i, pair in enumerate(pairs):
        pair_key = f"{pair.question_id}__{pair.category}__{pair.label}"

        if pair_key in processed_cf_keys:
            continue

        diff_info = find_edited_token_positions(
            tokenizer,
            orig_question=pair.original_question,
            cf_question=pair.cf_question,
            orig_prompt=pair.original_prompt,
            cf_prompt=pair.counterfactual_prompt,
        )

        cf_question_ids = tokenizer.encode(pair.cf_question, add_special_tokens=False)
        cf_full_ids = tokenizer.encode(pair.counterfactual_prompt, add_special_tokens=True)
        cf_offset = find_stem_offset(cf_full_ids, cf_question_ids)
        if cf_offset is None:
            cf_question_positions = []
        else:
            cf_question_positions = list(range(cf_offset, cf_offset + len(cf_question_ids)))

        cf_res = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=pair.counterfactual_prompt,
            answer_token_ids=answer_token_ids,
            extract_attention=extract_attention,
            attn_topk=attn_topk,
            edit_positions=diff_info["cf_edit_positions"],
            largest_positions=diff_info["largest_cf_positions"],
            question_positions=cf_question_positions,
        )
        current_cf[pair_key] = pack_extraction_result(cf_res)
        items_in_shard += 1

        # original with edit positions
        if not diff_info["alignment_failed"]:
            orig_edit_key = f"{pair.question_id}__edit__{pair_key}"
            if orig_edit_key not in current_orig:
                orig_question_ids = tokenizer.encode(pair.original_question, add_special_tokens=False)
                orig_full_ids = tokenizer.encode(pair.original_prompt, add_special_tokens=True)
                orig_offset = find_stem_offset(orig_full_ids, orig_question_ids)
                if orig_offset is None:
                    orig_question_positions = []
                else:
                    orig_question_positions = list(range(orig_offset, orig_offset + len(orig_question_ids)))

                orig_edit_res = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=pair.original_prompt,
                    answer_token_ids=answer_token_ids,
                    extract_attention=extract_attention,
                    attn_topk=attn_topk,
                    edit_positions=diff_info["orig_edit_positions"],
                    largest_positions=diff_info["largest_orig_positions"],
                    question_positions=orig_question_positions,
                )
                current_orig[orig_edit_key] = pack_extraction_result(orig_edit_res)
                items_in_shard += 1

        manifest["pair_metadata"].append({
            "question_id": pair.question_id,
            "category": pair.category,
            "label": pair.label,
            "attribute_type": pair.attribute_type,
            "control_subtype": pair.control_subtype,
            "gold_answer": pair.gold_answer,
            "analysis_class": pair.analysis_class,
            "counterfactual_validity": pair.counterfactual_validity,
            "clinical_coherence": pair.clinical_coherence,
            "target_attribute_role": pair.target_attribute_role,
            "gold_answer_invariance": pair.gold_answer_invariance,
            "prior_shift_expected": pair.prior_shift_expected,
            "edit_locality": pair.edit_locality,
            "annotation_confidence": pair.annotation_confidence,
            "intervention_type": pair.intervention_type,
            "intervention_family": pair.intervention_family,
            "analysis_bucket": pair.analysis_bucket,
            "edit_scope": pair.edit_scope,
            "edit_strength": pair.edit_strength,
            "attribute_value_counterfactual": pair.attribute_value_counterfactual,
            "medical_relevance": pair.medical_relevance,
            "social_bias_salience": pair.social_bias_salience,
            "pair_key": pair_key,
            "orig_edit_key": f"{pair.question_id}__edit__{pair_key}",
            "orig_largest_key": None,  # pooled in orig_edit extraction; can add separate run later if needed
            "cf_largest_key": None,
            "alignment_failed": diff_info["alignment_failed"],
            "n_tokens_changed": diff_info["n_tokens_changed"],
            "n_largest_region_tokens": diff_info["n_largest_region_tokens"],
            "n_edit_regions": diff_info["n_edit_regions"],
            "char_edit_distance": diff_info["char_edit_distance"],
            "token_edit_ratio": diff_info["token_edit_ratio"],
            "n_orig_tokens": diff_info["n_orig_tokens"],
            "n_cf_tokens": diff_info["n_cf_tokens"],
            "stem_orig_len": diff_info["stem_orig_len"],
            "stem_cf_len": diff_info["stem_cf_len"],
            "orig_edit_positions": diff_info["orig_edit_positions"],
            "cf_edit_positions": diff_info["cf_edit_positions"],
            "largest_orig_positions": diff_info["largest_orig_positions"],
            "largest_cf_positions": diff_info["largest_cf_positions"],
            "n_orig_edit_positions": len(diff_info["orig_edit_positions"]),
            "n_cf_edit_positions": len(diff_info["cf_edit_positions"]),
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(pairs) - i - 1) / max(rate, 1e-6) / 60.0
            print(f"  counterfactuals [{i + 1}/{len(pairs)}] ({rate:.2f}/s, ETA {eta:.1f}m)")

        if items_in_shard >= shard_size:
            flush_current_shard()

    flush_current_shard()

    manifest_path = output_dir / "manifest.pt"
    torch.save(manifest, manifest_path)
    print(f"\nSaved manifest: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory-safe extraction with compact attention summaries")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--extract_attention", action="store_true")
    parser.add_argument("--attn_topk", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Skip already-processed pairs found in output_dir")
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Device: {device}")
    print(f"DType: {dtype}")

    pairs = load_pairs(args.data_path)

    if args.start_idx or args.end_idx is not None:
        pairs = pairs[args.start_idx:args.end_idx]
        print(f"Sliced pairs to [{args.start_idx}:{args.end_idx}] -> {len(pairs)}")

    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to max_pairs={args.max_pairs} -> {len(pairs)}")

    model, tokenizer = load_model(args.model_path, device, dtype)

    print("\nAnswer token IDs:")
    answer_token_ids = get_answer_token_ids(tokenizer)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    extract_all(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        answer_token_ids=answer_token_ids,
        output_dir=outdir,
        shard_size=args.shard_size,
        extract_attention=args.extract_attention,
        attn_topk=args.attn_topk,
        resume=args.resume,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()