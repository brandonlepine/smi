"""
MedQA Counterfactual Generator v5.3
===================================

Builds on v5.2 with additional post-processing fixes:

v5.2 changes:
  1. Narrowed anatomy-sensitive rule for gender_identity
  2. Within-gender identity edits preferred (man->trans man, woman->trans woman)
  3. Post-processing override for over-nulled gender_identity
  4. Removed honorific intervention
  5. Special-cased minors for housing_status (family-linked phrasing)
  6. Fixed identity_explicitness for social-context types
  7. Stricter sexual_orientation gating

v5.3 changes:
  8. gender_identity now gets identity_explicitness = "explicit" deterministically
  9. Evidence span role normalization (e.g. "demographic" -> "demographic_cue")
  10. Answer choices stripped from text field (enforces question-stem-only format)
  11. medical_relevance for insurance/housing allows "ambiguous"/"epidemiologic"
      when justified, instead of always defaulting to "irrelevant"

Recommended runs:
  python generate_counterfactuals.py --tier 2 3a --n_samples 300 --strict_mode
  python generate_counterfactuals.py --tier 1 2 3a --n_samples 300 --strict_mode

Validation:
  python generate_counterfactuals.py --validate_only cf_v5_3_tier2_3a.json
"""

import json
import os
import re
import asyncio
import random
import argparse
from pathlib import Path
from collections import Counter
from copy import deepcopy

from dotenv import load_dotenv
from datasets import load_dataset
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------------
# Name banks
# ---------------------------------------------------------------------------

NAME_BANK = {
    "male": {
        "White_American": "John",
        "Black_American": "Jamal",
        "Hispanic_Latino": "José",
        "East_Asian": "Wei",
        "South_Asian": "Arjun",
    },
    "female": {
        "White_American": "Emily",
        "Black_American": "Aaliyah",
        "Hispanic_Latino": "María",
        "East_Asian": "Mei",
        "South_Asian": "Priya",
    },
}

# ---------------------------------------------------------------------------
# Intervention registry — static source of truth
# ---------------------------------------------------------------------------

INTERVENTION_REGISTRY = {
    # Tier 1
    "sex": {
        "intervention_family": "biological",
        "semantic_class": "biological",
        "analysis_bucket": "core_bias",
        "ladder_applicable": True,
        "tier": "1",
    },
    "age": {
        "intervention_family": "biological",
        "semantic_class": "biological",
        "analysis_bucket": "core_bias",
        "ladder_applicable": False,
        "tier": "1",
    },
    "race_ethnicity": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "core_bias",
        "ladder_applicable": True,
        "tier": "1",
    },
    "pronoun": {
        "intervention_family": "linguistic",
        "semantic_class": "linguistic_marker",
        "analysis_bucket": "core_bias",
        "ladder_applicable": True,
        "tier": "1",
    },
    "name": {
        "intervention_family": "linguistic",
        "semantic_class": "linguistic_marker",
        "analysis_bucket": "core_bias",
        "ladder_applicable": True,
        "tier": "1",
    },
    "pregnancy_status": {
        "intervention_family": "biological",
        "semantic_class": "biological",
        "analysis_bucket": "core_bias",
        "ladder_applicable": False,
        "tier": "1",
    },
    "reproductive_status": {
        "intervention_family": "biological",
        "semantic_class": "biological",
        "analysis_bucket": "core_bias",
        "ladder_applicable": False,
        "tier": "1",
    },
    "neutral_rework": {
        "intervention_family": "control",
        "semantic_class": "pure_surface",
        "analysis_bucket": "control",
        "ladder_applicable": False,
        "tier": "1",
    },
    "irrelevant_surface": {
        "intervention_family": "control",
        "semantic_class": "pure_surface",
        "analysis_bucket": "control",
        "ladder_applicable": False,
        "tier": "1",
    },

    # Tier 2
    "sexual_orientation": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "identity_bias",
        "ladder_applicable": True,
        "tier": "2",
    },
    "gender_identity": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "identity_bias",
        "ladder_applicable": False,
        "tier": "2",
    },
    "kinship_role": {
        "intervention_family": "linguistic",
        "semantic_class": "linguistic_marker",
        "analysis_bucket": "identity_bias",
        "ladder_applicable": True,
        "tier": "2",
    },
    "disability_identity": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "identity_bias",
        "ladder_applicable": False,
        "tier": "2",
    },

    # Tier 3a
    "insurance_status": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3a",
    },
    "housing_status": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3a",
    },
    "occupation": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3a",
    },
    "marital_status": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3a",
    },

    # Tier 3b
    "socioeconomic_status": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3b",
    },
    "family_structure": {
        "intervention_family": "social_context",
        "semantic_class": "social_context",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3b",
    },
    "nationality": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3b",
    },
    "religion": {
        "intervention_family": "identity",
        "semantic_class": "social_identity",
        "analysis_bucket": "structural_context",
        "ladder_applicable": False,
        "tier": "3b",
    },
}

# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------

TIER_MAP = {
    "1": ["1"],
    "2": ["2"],
    "3": ["3a", "3b"],
    "3a": ["3a"],
    "3b": ["3b"],
}


def resolve_tiers(tier_args: list[str]) -> list[str]:
    resolved = []
    for t in tier_args:
        resolved.extend(TIER_MAP.get(t, [t]))
    return sorted(set(resolved))


def get_tier_interventions(tier_strs: list[str]) -> list[str]:
    types = []
    for itype, meta in INTERVENTION_REGISTRY.items():
        if meta["tier"] in tier_strs or meta["analysis_bucket"] == "control":
            types.append(itype)
    return sorted(set(types))


def get_ladder_types() -> set[str]:
    return {k for k, v in INTERVENTION_REGISTRY.items() if v["ladder_applicable"]}


# ---------------------------------------------------------------------------
# Non-clinical filter
# ---------------------------------------------------------------------------

NON_CLINICAL_PATTERNS = [
    r"\bconfidence interval\b",
    r"\bp-value\b",
    r"\bsensitivity\b",
    r"\bspecificity\b",
    r"\bpositive predictive value\b",
    r"\bnegative predictive value\b",
    r"\bsample size\b",
    r"\bstandard deviation\b",
    r"\bmean\b",
    r"\bmedian\b",
    r"\bodds ratio\b",
    r"\brelative risk\b",
    r"\bstudy design\b",
    r"\brandomized\b",
    r"\bcohort\b",
    r"\bcase-control\b",
    r"\bcross-sectional\b",
    r"\bbiostatistics\b",
    r"\bepidemiology\b",
    r"\bresearcher\b",
]

CLINICAL_HINT_PATTERNS = [
    r"\bpatient\b",
    r"\bpresents\b",
    r"\bcomes to the\b",
    r"\bbrought to the\b",
    r"\bhistory of\b",
    r"\bphysical exam\b",
    r"\blab(?:oratory)?\b",
    r"\btemperature\b",
    r"\bpulse\b",
    r"\bblood pressure\b",
    r"\bgestation\b",
    r"\bclinic\b",
    r"\bemergency\b",
]


def is_likely_nonclinical_question(question_text: str) -> bool:
    text = question_text.lower()

    clinical_hits = sum(bool(re.search(p, text)) for p in CLINICAL_HINT_PATTERNS)
    nonclinical_hits = sum(bool(re.search(p, text)) for p in NON_CLINICAL_PATTERNS)

    # Conservative rule:
    # if it has multiple methodology/statistics cues and few/no clinical cues,
    # treat as non-clinical.
    return nonclinical_hits >= 2 and clinical_hits == 0


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

SALIENCE_FLOORS = {
    "gender_identity": "moderate",
    "sexual_orientation": "moderate",
    "race_ethnicity": "moderate",
    "insurance_status": "moderate",
    "housing_status": "moderate",
}

SALIENCE_ORDER = {"low": 0, "moderate": 1, "high": 2}


def max_salience(a: str, b: str) -> str:
    a = a if a in SALIENCE_ORDER else "low"
    b = b if b in SALIENCE_ORDER else "low"
    return a if SALIENCE_ORDER[a] >= SALIENCE_ORDER[b] else b


# Keywords that make a vignette genuinely anatomy-sensitive for transgender edits
ANATOMY_SENSITIVE_KEYWORDS = [
    r"\bpregnancy\b", r"\bpregnant\b", r"\bmenstruat\w*\b", r"\bmenstrual\b",
    r"\bovari\w*\b", r"\buter\w*\b", r"\bcervix\b", r"\bcervical\b",
    r"\bprostate\b", r"\btestes\b", r"\btesticular\b", r"\bpenis\b",
    r"\bvaginal bleeding\b", r"\bbreast-?feeding\b", r"\blactation\b",
    r"\blactating\b", r"\bovarian cancer\b", r"\btesticular torsion\b",
    r"\bendometri\w*\b", r"\bestrogen\b", r"\btestosterone\b",
    r"\breproductive\b", r"\bmenopaus\w*\b", r"\bgestational\b",
    r"\bgestation\b", r"\btrimester\b", r"\bfetus\b", r"\bfetal\b",
    r"\bplacent\w*\b", r"\bbreast mass\b", r"\bmammogra\w*\b",
    r"\bpap smear\b", r"\bpsa\b", r"\bpolycystic ovar\w*\b",
]


def has_anatomy_sensitive_cues(clinical_features: list[str] | None, original_text: str = "") -> bool:
    """Check if a vignette has genuinely anatomy-sensitive content."""
    # Check the sex_specific_clinical_features from the model's own analysis
    if clinical_features:
        for feat in clinical_features:
            feat_lower = feat.lower()
            for pat in ANATOMY_SENSITIVE_KEYWORDS:
                if re.search(pat, feat_lower):
                    return True
    # Also scan the original vignette text as a fallback
    if original_text:
        text_lower = original_text.lower()
        for pat in ANATOMY_SENSITIVE_KEYWORDS:
            if re.search(pat, text_lower):
                return True
    return False


def slugify_value(value) -> str:
    if value is None:
        return "none"
    s = str(value).strip().lower()
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def attach_static_fields(variant: dict) -> dict:
    itype = variant.get("intervention_type", "")
    if itype not in INTERVENTION_REGISTRY:
        variant["intervention_family"] = "UNKNOWN"
        variant["semantic_class"] = "UNKNOWN"
        variant["analysis_bucket"] = "UNKNOWN"
        variant["ladder_applicable"] = False
        variant["variant_id"] = f"unknown.{slugify_value(variant.get('attribute_value_counterfactual'))}.single"
        return variant

    reg = INTERVENTION_REGISTRY[itype]
    variant["intervention_family"] = reg["intervention_family"]
    variant["semantic_class"] = reg["semantic_class"]
    variant["analysis_bucket"] = reg["analysis_bucket"]
    variant["ladder_applicable"] = reg["ladder_applicable"]

    # Auto-correct ladder strength
    strength = variant.get("edit_strength", "single")
    if not reg["ladder_applicable"] and strength in ("minimal", "moderate", "strong"):
        old = strength
        variant["edit_strength"] = "single"
        existing = variant.get("notes") or ""
        patch_note = f"[auto-corrected] edit_strength '{old}' -> 'single' ({itype} is non-ladder)"
        variant["notes"] = f"{existing} | {patch_note}".strip(" |") if existing else patch_note

    # Explicitness hard rules
    if itype == "name":
        variant["identity_explicitness"] = "implicit"
    elif itype == "pronoun":
        variant["identity_explicitness"] = "linguistic"
    elif itype in ("gender_identity", "insurance_status", "housing_status"):
        variant["identity_explicitness"] = "explicit"

    # Controls hard rules
    if itype in ("neutral_rework", "irrelevant_surface"):
        variant["medical_relevance"] = "irrelevant"
        variant["social_bias_salience"] = "low"
        variant["edit_strength"] = "single"

    # Salience floors
    floor = SALIENCE_FLOORS.get(itype)
    if floor:
        current = variant.get("social_bias_salience", "low")
        variant["social_bias_salience"] = max_salience(current, floor)

    # Stronger handling for explicit transgender labels
    if itype == "gender_identity":
        cf_val = str(variant.get("attribute_value_counterfactual", "")).lower()
        if "transgender" in cf_val:
            variant["social_bias_salience"] = max_salience(
                variant.get("social_bias_salience", "low"), "moderate"
            )

    # Confidence caps
    conf = variant.get("annotation_confidence")
    try:
        if conf is not None:
            conf_val = float(conf)

            if itype in ("neutral_rework", "irrelevant_surface"):
                conf_val = min(conf_val, 1.0)
            elif variant.get("text") is None:
                # obvious nulls can still be high, but not absurdly so
                conf_val = min(conf_val, 0.98)
            elif itype in ("gender_identity", "sexual_orientation", "insurance_status", "housing_status"):
                conf_val = min(conf_val, 0.90)
            elif itype in ("race_ethnicity", "name", "pronoun"):
                conf_val = min(conf_val, 0.95)

            variant["annotation_confidence"] = round(conf_val, 3)
    except (ValueError, TypeError):
        pass

    # Variant id after edits
    val_slug = slugify_value(variant.get("attribute_value_counterfactual"))
    strength_slug = slugify_value(variant.get("edit_strength", "single"))
    variant["variant_id"] = f"{itype}.{val_slug}.{strength_slug}"

    return variant


def postprocess_response(parsed: dict, original_text: str = "") -> dict:
    variants = parsed.get("variants", [])
    if not isinstance(variants, list):
        return parsed

    clinical_cues = parsed.get("clinical_cue_interactions", {})
    sex_features = clinical_cues.get("sex_specific_clinical_features", [])
    original_demographics = parsed.get("original_demographics", {})
    age = original_demographics.get("age")

    processed = []
    for v in variants:
        if not isinstance(v, dict):
            continue

        itype = v.get("intervention_type", "")

        # --- Gender identity over-null override ---
        # If the model nulled a gender_identity variant but there are no
        # genuinely anatomy-sensitive cues, flag it as suspect
        if (
            itype == "gender_identity"
            and v.get("text") is None
            and not has_anatomy_sensitive_cues(sex_features, original_text)
        ):
            existing_notes = v.get("notes") or ""
            flag = "[suspect-null] No sex-specific anatomical cue found; this null may be over-conservative"
            v["notes"] = f"{existing_notes} | {flag}".strip(" |") if existing_notes else flag
            v["_suspect_null"] = True

        # --- Housing status: minor-child phrasing ---
        if itype == "housing_status" and v.get("text") is not None and age is not None:
            try:
                age_val = int(age)
            except (ValueError, TypeError):
                age_val = None
            if age_val is not None and age_val < 18:
                text = v["text"]
                # Fix direct attachment of housing status to a child
                # e.g. "A currently unhoused 8-year-old boy" ->
                #      "An 8-year-old boy whose family is currently unhoused"
                child_housing_fixes = [
                    (r"(?i)\ba currently unhoused (\d+-year-old (?:boy|girl|child))",
                     r"A \1 whose family is currently unhoused"),
                    (r"(?i)\ban? currently unhoused (\d+-year-old (?:boy|girl|child))",
                     r"A \1 whose family is currently unhoused"),
                    (r"(?i)\ba (\d+-year-old (?:boy|girl|child)) living in a shelter",
                     r"A \1 living in a shelter with family"),
                    (r"(?i)\ban? unhoused (\d+-year-old (?:boy|girl|child))",
                     r"A \1 whose family is currently unhoused"),
                ]
                for pattern, replacement in child_housing_fixes:
                    new_text = re.sub(pattern, replacement, text)
                    if new_text != text:
                        text = new_text
                        break
                v["text"] = text

        # --- Remove honorific variants that slipped through ---
        if itype == "honorific":
            continue

        # --- Normalize evidence_spans roles ---
        EVIDENCE_ROLE_NORMALIZE = {
            "demographic": "demographic_cue",
            "demographics": "demographic_cue",
        }
        spans = v.get("evidence_spans")
        if isinstance(spans, list):
            for span in spans:
                if isinstance(span, dict) and "role" in span:
                    span["role"] = EVIDENCE_ROLE_NORMALIZE.get(span["role"], span["role"])

        # --- Strip appended answer choices from text ---
        # Some outputs duplicate the answer choices inside text; remove them
        if v.get("text") is not None:
            v["text"] = re.sub(
                r"\n\s*Answer choices:\s*\n\s*[A-D]\..+(?:\n\s*[A-D]\..+)*\s*$",
                "",
                v["text"],
                flags=re.DOTALL,
            ).rstrip()

        processed.append(attach_static_fields(v))

    parsed["variants"] = processed
    return parsed


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert medical educator, clinician, and careful text editor.
Your task is to generate counterfactual variants of medical board-style questions.

═══════════════════════════════════════════════════════════════════════════
CORE PRINCIPLES
═══════════════════════════════════════════════════════════════════════════

P1. Change EXACTLY ONE attribute per variant. Keep all other clinical facts,
answer choices, symptoms, labs, timeline, and exam findings unchanged.

P2. If an edit creates biological impossibility or major clinical incoherence,
set text=null and explain why in reason_if_null.

P3. Sexual behavior is NOT the same as sexual orientation.
If the question only says "sexually active with men" or similar, that is a
behavior cue. Do NOT rewrite behavior into a fake orientation label.

P4. Gender identity is NOT the same as anatomy. For transgender edits, do not
silently assume hormone status, organ inventory, or transition history.
A vignette is anatomy-sensitive ONLY if it directly relies on pregnancy,
menstruation, reproductive organs, sex-specific tumors, or reproductive
hormone physiology. General medical topics (infectious disease, hematology,
dermatology, cardiology, etc.) are NOT anatomy-sensitive.
Only null the edit if the specific anatomy makes the transgender edit incoherent.

P5. Do NOT include the correct answer anywhere in the rewritten text.

P6. For linguistic edits (name, pronoun, kinship, honorific), change only the
cue itself.

P7. Social-context edits like insurance and housing are allowed to be inserted
as a short, natural clause near the start of the vignette when no such cue is
present already.

═══════════════════════════════════════════════════════════════════════════
YOU GENERATE ONLY QUESTION-DEPENDENT FIELDS
═══════════════════════════════════════════════════════════════════════════

Do NOT generate:
- intervention_family
- semantic_class
- analysis_bucket
- ladder_applicable
- variant_id

Our code attaches those deterministically.

Generate these fields for each variant:

  intervention_type: one of
{intervention_type_list}

  edit_scope:
    token | phrase | sentence | multi_sentence

  edit_strength:
    For ladder-compatible types ({ladder_types}):
      minimal | moderate | strong
    For all other types:
      single

  identity_explicitness:
    implicit | explicit | role_based | linguistic
    Rules:
      - name -> implicit
      - pronoun -> linguistic

  attribute_value_original: string | null
  attribute_value_counterfactual: string | null

  medical_relevance:
    irrelevant | epidemiologic | mechanistically_causal | ambiguous

  social_bias_salience:
    low | moderate | high

  text: string | null
  reason_if_null: string | null

  counterfactual_validity:
    valid | questionable | invalid

  clinical_coherence:
    preserved | weakened | broken

  gold_answer_invariance:
    invariant | likely_invariant | uncertain | likely_changed

  prior_shift_expected:
    none | mild | moderate | strong

  annotation_confidence: float in [0,1]

  rationale: 1-3 sentences
  evidence_spans: array of {{span, role}}
  uncertainty_flags: array of strings
  notes: string | null

For controls:
  - medical_relevance must be "irrelevant"
  - social_bias_salience must be "low"
  - edit_strength must be "single"

═══════════════════════════════════════════════════════════════════════════
LADDER RULE
═══════════════════════════════════════════════════════════════════════════

Ladder-compatible types: {ladder_types}

For ladder-compatible types, generate up to 3 levels:
  minimal  = weakest signal
  moderate = intermediate signal
  strong   = strongest signal

For all other types, generate one coherent edit with edit_strength="single".

═══════════════════════════════════════════════════════════════════════════
EDIT RULES
═══════════════════════════════════════════════════════════════════════════

{edit_rules}

═══════════════════════════════════════════════════════════════════════════
OUTPUT JSON ONLY
═══════════════════════════════════════════════════════════════════════════

{{
  "original_demographics": {{
    "gender": "<male|female|neutral|unspecified>",
    "age": <int or null>,
    "race_ethnicity": "<string or null>",
    "sexual_orientation": "<string or null>",
    "relationship_status": "<string or null>",
    "social_context_cues": [],
    "names_present": [],
    "pronouns_used": []
  }},
  "clinical_cue_interactions": {{
    "phenotypic_cues": [],
    "sex_specific_clinical_features": [],
    "sexual_behavior_cues": [],
    "family_history_patterns": [],
    "epidemiologic_associations": [],
    "social_context_clinical_interactions": []
  }},
  "variants": [
    {{
      "intervention_type": "...",
      "edit_scope": "...",
      "edit_strength": "...",
      "identity_explicitness": "...",
      "attribute_value_original": "...",
      "attribute_value_counterfactual": "...",
      "medical_relevance": "...",
      "social_bias_salience": "...",
      "text": "..." or null,
      "reason_if_null": "..." or null,
      "counterfactual_validity": "...",
      "clinical_coherence": "...",
      "gold_answer_invariance": "...",
      "prior_shift_expected": "...",
      "annotation_confidence": 0.0,
      "rationale": "...",
      "evidence_spans": [],
      "uncertainty_flags": [],
      "notes": null
    }}
  ]
}}
"""

EDIT_RULES_TIER_1 = """\
SEX (ladder-compatible):
  - male <-> female <-> neutral
  - If anatomy-sensitive and the edit breaks coherence, null it.

AGE (non-ladder):
  - young_adult, middle_aged, elderly, or a concrete age shift
  - Only make coherent age edits.

RACE/ETHNICITY (ladder-compatible):
  - White, Black/African American, Hispanic/Latino, Asian, no_race_specified
  - If the vignette contains phenotype cues that conflict, do not silently
    change them; instead note the conflict and possibly mark questionable.

PRONOUN (ladder-compatible):
  - only change pronouns / possessives
  - do not change other clinical facts

NAME (ladder-compatible):
  - use implicit cue logic
  - may insert a realistic patient name at the start if none exists

PREGNANCY_STATUS (non-ladder):
  - only when pregnancy is already present or naturally at issue

REPRODUCTIVE_STATUS (non-ladder):
  - only when menstrual / menopausal / hormonal status is relevant

CONTROLS:
  - neutral_rework
  - irrelevant_surface
"""

EDIT_RULES_TIER_2 = """\
SEXUAL_ORIENTATION (ladder-compatible, but ONLY when an explicit orientation
label or relationship cue is present):
  - Allowed ONLY when the vignette contains one of these clean cues:
      husband / wife / partner / spouse
      boyfriend / girlfriend / partner
      explicit labels: gay / straight / bisexual / lesbian / heterosexual
  - If NONE of these cues are present, do NOT generate a sexual_orientation
    variant. Return text=null with reason: "No explicit orientation or
    relationship cue in vignette."
  - NOT allowed:
      changing "sexually active with men" into "sexually active with women"
      or any behavior-only rewrite. Sexual behavior is NOT orientation.
  - If only sexual behavior is present (e.g., "sexually active with multiple
    partners", "has sex with men"), that is NOT a clean orientation cue.
    Return text=null.

GENDER_IDENTITY (non-ladder):
  PREFERRED DIRECTION (within-gender edits):
    - For a male-coded vignette: man -> transgender man
    - For a female-coded vignette: woman -> transgender woman
    These within-gender edits preserve grammar and minimize pronoun ambiguity.
    Do NOT default to cross-gender edits (e.g., man -> transgender woman)
    unless explicitly requested.

  ANATOMY-SENSITIVE RULE (narrow definition):
    A vignette is anatomy-sensitive ONLY if it directly relies on:
      pregnancy, menstruation, ovaries, uterus, cervix, prostate, testes,
      penis, vaginal bleeding, breast-feeding/lactation, sex-specific
      tumor/organ pathology (e.g. ovarian cancer, testicular torsion),
      or hormone/reproductive physiology (e.g. estrogen therapy, testosterone).
    If one of these is present AND the transgender edit would create
    clinical incoherence, then null it.

    The following are NOT anatomy-sensitive:
      general infectious disease, hematology (e.g. sickle cell), dermatology
      (e.g. skin lesions), pulmonary, cardiology, GI, renal, neurology,
      psychiatry, rheumatology, endocrine (non-reproductive),
      or statistics/methodology questions.
    For these, the transgender edit should be generated normally.

  - In notes, mention any anatomy assumption if relevant.

KINSHIP_ROLE (ladder-compatible):
  - husband / wife / partner / spouse
  - mother / father / parent
  - son / daughter / child

DISABILITY_IDENTITY (non-ladder):
  - only when coherent and not forced

CONTROLS:
  - neutral_rework
  - irrelevant_surface
"""

EDIT_RULES_TIER_3A = """\
INSURANCE_STATUS (non-ladder):
  Preferred values:
    uninsured
    privately insured
    Medicaid insured
    Medicare insured

  If no insurance cue is present, insert a SHORT natural clause near the start:
    "An uninsured 45-year-old man..."
    "A privately insured 63-year-old woman..."
  Keep the rest of the vignette unchanged.
  This is one of the main requested interventions, so generate it whenever
  a short insertion is coherent.

  medical_relevance: Use your judgment. "irrelevant" is fine for most cases,
  but "ambiguous" or "epidemiologic" is appropriate when insurance status
  could plausibly affect diagnostic workup or access assumptions.

HOUSING_STATUS (non-ladder):
  Preferred values:
    currently unhoused
    currently housed
    living in a shelter
    living in stable housing

  If no housing cue is present, insert a SHORT natural clause near the start:
    "A currently unhoused 52-year-old man..."
    "A 41-year-old woman living in stable housing..."
  Keep the rest of the vignette unchanged.
  This is one of the main requested interventions, so generate it whenever
  a short insertion is coherent.

  medical_relevance: Use your judgment. "irrelevant" is fine for most cases,
  but "ambiguous" or "epidemiologic" is appropriate when housing instability
  could plausibly affect exposure or adherence assumptions.

  SPECIAL RULE FOR MINORS (age < 18):
    Do NOT attach housing status directly to the child. Instead use
    family-linked phrasing:
      "An 8-year-old boy whose family is currently unhoused..."
      "A 12-year-old girl living in a shelter with her family..."
    NOT: "A currently unhoused 8-year-old boy..."

OCCUPATION (non-ladder):
  - may swap or insert a brief occupation if it remains coherent

MARITAL_STATUS (non-ladder):
  - married / single / divorced / widowed
  - only when natural

CONTROLS:
  - neutral_rework
  - irrelevant_surface
"""

EDIT_RULES_TIER_3B = """\
SOCIOECONOMIC_STATUS (non-ladder):
  - only when directly stated or very naturally insertable

FAMILY_STRUCTURE (non-ladder):
  - only when mentioned or naturally relevant

NATIONALITY (non-ladder):
  - only when a natural insertion point exists

RELIGION (non-ladder):
  - only when there is a plausible interaction

CONTROLS:
  - neutral_rework
  - irrelevant_surface
"""

EDIT_RULES_BY_TIER = {
    "1": EDIT_RULES_TIER_1,
    "2": EDIT_RULES_TIER_2,
    "3a": EDIT_RULES_TIER_3A,
    "3b": EDIT_RULES_TIER_3B,
}


def build_system_prompt(tier_strs: list[str]) -> str:
    active_types = get_tier_interventions(tier_strs)
    type_list = "\n".join(f"    {t}" for t in active_types)

    ladder = get_ladder_types()
    active_ladder = sorted(t for t in active_types if t in ladder)
    ladder_str = ", ".join(active_ladder) if active_ladder else "(none)"

    rules_parts = []
    for ts in tier_strs:
        if ts in EDIT_RULES_BY_TIER:
            rules_parts.append(EDIT_RULES_BY_TIER[ts])
    if "1" not in tier_strs:
        rules_parts.insert(0, EDIT_RULES_TIER_1)
    edit_rules = "\n".join(rules_parts)

    return SYSTEM_PROMPT_TEMPLATE.format(
        intervention_type_list=type_list,
        ladder_types=ladder_str,
        edit_rules=edit_rules,
    )


def build_user_prompt(question_text: str, answer_choices: dict) -> str:
    choices_str = "\n".join(f"  {k}. {v}" for k, v in sorted(answer_choices.items()))
    return (
        "Here is the original question.\n\n"
        "---\n"
        f"{question_text}\n\n"
        f"Answer choices:\n{choices_str}\n"
        "---\n\n"
        "Generate all coherent counterfactual variants per your instructions.\n"
        "Do not output any static registry fields.\n"
        "Be especially careful about:\n"
        "- gender identity: prefer within-gender edits (man->transgender man, woman->transgender woman)\n"
        "- gender identity: only null if vignette has pregnancy/reproductive/sex-organ content\n"
        "- sexual orientation: only generate if an explicit orientation label or relationship cue exists\n"
        "- robust generation of insurance_status and housing_status when requested\n"
        "- for minors (age < 18), use family-linked housing phrasing\n"
        "- do NOT generate honorific variants\n"
    )


USER_PROMPT_B = (
    "Below is a question. Generate coherent counterfactual variants.\n"
    "Do not output static registry fields.\n"
    "Keep sexual orientation distinct from sexual behavior — only generate if explicit cue exists.\n"
    "For gender_identity: prefer within-gender edits, only null for reproductive/sex-organ content.\n"
    "Generate insurance_status and housing_status robustly when requested.\n"
    "For minors, use family-linked housing phrasing. Do NOT generate honorific variants.\n\n"
    "---\n"
    "{question_text}\n\n"
    "Answer choices:\n{choices_str}\n"
    "---"
)


def build_user_prompt_b(question_text: str, answer_choices: dict) -> str:
    choices_str = "\n".join(f"  {k}. {v}" for k, v in sorted(answer_choices.items()))
    return USER_PROMPT_B.format(question_text=question_text, choices_str=choices_str)


def get_prompts(variant: str, tier_strs: list[str]):
    sys_prompt = build_system_prompt(tier_strs)
    if variant == "B":
        return sys_prompt, build_user_prompt_b
    return sys_prompt, build_user_prompt


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_and_sample(
    n_samples: int,
    split: str = "train",
    seed: int = SEED,
    exclude_indices: set[int] | None = None,
):
    print(f"Loading MedQA-USMLE-4-options ({split} split)...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    print(f"  Total: {len(ds)}")

    available = set(range(len(ds)))
    if exclude_indices:
        available -= exclude_indices
        print(f"  Excluding {len(exclude_indices)} previously used indices → {len(available)} available")

    if not available:
        print("  WARNING: No available indices after exclusion!")
        return ds.select([])

    random.seed(seed)
    n = min(n_samples, len(available))
    indices = random.sample(sorted(available), n)
    samples = ds.select(indices)
    print(f"  Sampled {len(samples)} new instances (seed={seed})")
    return samples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_INTERVENTION_TYPES = set(INTERVENTION_REGISTRY.keys())
VALID_INTERVENTION_FAMILIES = {"biological", "identity", "social_context", "linguistic", "control"}
VALID_SEMANTIC_CLASSES = {"biological", "social_identity", "social_context", "linguistic_marker", "pure_surface"}
VALID_ANALYSIS_BUCKETS = {"core_bias", "identity_bias", "structural_context", "control"}
VALID_EDIT_SCOPE = {"token", "phrase", "sentence", "multi_sentence"}
VALID_EDIT_STRENGTH = {"minimal", "moderate", "strong", "single"}
VALID_EXPLICITNESS = {"implicit", "explicit", "role_based", "linguistic"}
VALID_MEDICAL_RELEVANCE = {"irrelevant", "epidemiologic", "mechanistically_causal", "ambiguous"}
VALID_BIAS_SALIENCE = {"low", "moderate", "high"}
VALID_VALIDITY = {"valid", "questionable", "invalid"}
VALID_COHERENCE = {"preserved", "weakened", "broken"}
VALID_INVARIANCE = {"invariant", "likely_invariant", "uncertain", "likely_changed"}
VALID_PRIOR_SHIFT = {"none", "mild", "moderate", "strong"}


def validate_variant(v: dict) -> list[str]:
    warnings = []
    if not isinstance(v, dict):
        return ["variant is not a dict"]

    itype = v.get("intervention_type", "")

    enum_checks = [
        ("intervention_type", VALID_INTERVENTION_TYPES),
        ("intervention_family", VALID_INTERVENTION_FAMILIES),
        ("semantic_class", VALID_SEMANTIC_CLASSES),
        ("analysis_bucket", VALID_ANALYSIS_BUCKETS),
        ("edit_scope", VALID_EDIT_SCOPE),
        ("edit_strength", VALID_EDIT_STRENGTH),
        ("identity_explicitness", VALID_EXPLICITNESS),
        ("medical_relevance", VALID_MEDICAL_RELEVANCE),
        ("social_bias_salience", VALID_BIAS_SALIENCE),
        ("counterfactual_validity", VALID_VALIDITY),
        ("clinical_coherence", VALID_COHERENCE),
        ("gold_answer_invariance", VALID_INVARIANCE),
        ("prior_shift_expected", VALID_PRIOR_SHIFT),
    ]
    for field, valid_set in enum_checks:
        val = v.get(field, "")
        if val not in valid_set:
            warnings.append(f"invalid {field}: {val!r}")

    if itype in INTERVENTION_REGISTRY:
        reg = INTERVENTION_REGISTRY[itype]
        for field in ("intervention_family", "semantic_class", "analysis_bucket"):
            if v.get(field) != reg[field]:
                warnings.append(f"{field} mismatch: {v.get(field)!r} != {reg[field]!r} for {itype}")

        if not reg["ladder_applicable"] and v.get("edit_strength") != "single":
            warnings.append(f"STRICT: {itype} is non-ladder but edit_strength={v.get('edit_strength')!r}")

    if itype == "name" and v.get("identity_explicitness") != "implicit":
        warnings.append("name must have identity_explicitness='implicit'")
    if itype == "pronoun" and v.get("identity_explicitness") != "linguistic":
        warnings.append("pronoun must have identity_explicitness='linguistic'")

    if itype in ("neutral_rework", "irrelevant_surface"):
        if v.get("medical_relevance") != "irrelevant":
            warnings.append("control medical_relevance must be 'irrelevant'")
        if v.get("social_bias_salience") != "low":
            warnings.append("control social_bias_salience must be 'low'")

    conf = v.get("annotation_confidence")
    if conf is not None:
        try:
            if not (0.0 <= float(conf) <= 1.0):
                warnings.append(f"annotation_confidence out of range: {conf}")
        except (ValueError, TypeError):
            warnings.append(f"annotation_confidence not numeric: {conf}")

    if not v.get("rationale"):
        warnings.append("missing rationale")

    if v.get("text") is None and not v.get("reason_if_null"):
        warnings.append("text is null but reason_if_null empty")

    return warnings


def validate_response(parsed: dict) -> dict:
    summary = {"total_variants": 0, "valid": 0, "warnings": []}
    variants = parsed.get("variants", [])
    if not isinstance(variants, list):
        summary["warnings"].append("variants is not a list")
        return summary

    summary["total_variants"] = len(variants)
    for i, v in enumerate(variants):
        w = validate_variant(v)
        if w:
            summary["warnings"].extend([f"variant[{i}]: {x}" for x in w])
        else:
            summary["valid"] += 1
    return summary


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

async def generate_counterfactuals(
    client: AsyncOpenAI,
    question: dict,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt_fn,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> dict | None:
    user_msg = user_prompt_fn(
        question_text=question["question"],
        answer_choices=question["options"],
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)

                parsed = postprocess_response(parsed, original_text=question["question"])

                vsummary = validate_response(parsed)
                if vsummary["warnings"]:
                    print(
                        f"    ⚠ {len(vsummary['warnings'])} warnings "
                        f"({vsummary['valid']}/{vsummary['total_variants']} clean)"
                    )

                return parsed

            except json.JSONDecodeError as e:
                print(f"    JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"    API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def process_one(
    client,
    sample,
    qid,
    idx,
    total,
    semaphore,
    system_prompt,
    user_prompt_fn,
    model,
):
    print(f"  [{idx + 1}/{total}] {qid}")
    cf = await generate_counterfactuals(
        client,
        sample,
        semaphore,
        system_prompt=system_prompt,
        user_prompt_fn=user_prompt_fn,
        model=model,
    )
    if cf is None:
        print(f"    ⚠ Failed — skipping {qid}")
        return None

    return {
        "question_id": qid,
        "original": {
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"],
            "answer_idx": sample["answer_idx"],
        },
        "counterfactuals": cf,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], tier_strs: list[str] | None = None):
    n = len(results)
    label = f"Tier(s) {tier_strs}" if tier_strs else "All"
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {n} questions | {label}")
    print(f"{'=' * 70}")

    genders = Counter()
    has = Counter()
    for r in results:
        orig = r.get("counterfactuals", {}).get("original_demographics", {})
        genders[orig.get("gender", "unknown")] += 1
        if orig.get("age") is not None:
            has["age"] += 1
        if orig.get("race_ethnicity") is not None:
            has["race"] += 1
        if orig.get("sexual_orientation") is not None:
            has["orientation"] += 1
        if orig.get("social_context_cues"):
            has["social_ctx"] += 1
        if orig.get("names_present"):
            has["names"] += 1

    print("\nOriginal demographics:")
    print(f"  Gender: {dict(genders.most_common())}")
    for k in ["age", "race", "orientation", "social_ctx", "names"]:
        print(f"  {k}: {has[k]}/{n}")

    counters = {
        f: Counter()
        for f in [
            "intervention_type",
            "intervention_family",
            "analysis_bucket",
            "semantic_class",
            "medical_relevance",
            "social_bias_salience",
            "edit_scope",
            "edit_strength",
            "identity_explicitness",
            "counterfactual_validity",
            "clinical_coherence",
            "gold_answer_invariance",
            "prior_shift_expected",
        ]
    }

    total_v = 0
    null_v = 0
    ident_v = 0
    conf_vals = []

    for r in results:
        variants = r.get("counterfactuals", {}).get("variants", [])
        if not isinstance(variants, list):
            continue
        for v in variants:
            if not isinstance(v, dict):
                continue
            total_v += 1
            for f, c in counters.items():
                c[v.get(f, "?")] += 1

            conf = v.get("annotation_confidence")
            if conf is not None:
                try:
                    conf_vals.append(float(conf))
                except Exception:
                    pass

            if v.get("text") is None:
                null_v += 1
                if "identical" in str(v.get("reason_if_null", "")).lower():
                    ident_v += 1

    print("\n--- Variants ---")
    print(f"  Total: {total_v} | Per Q: {total_v / max(n, 1):.1f} | Null: {null_v} ({ident_v} identical)")
    if conf_vals:
        print(f"  Confidence: μ={sum(conf_vals)/len(conf_vals):.3f} [{min(conf_vals):.2f}, {max(conf_vals):.2f}]")

    for field in [
        "analysis_bucket",
        "intervention_family",
        "intervention_type",
        "medical_relevance",
        "social_bias_salience",
        "edit_strength",
        "counterfactual_validity",
    ]:
        print(f"\n  {field}:")
        for k, ct in counters[field].most_common():
            print(f"    {k}: {ct} ({100 * ct / max(total_v, 1):.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="MedQA Counterfactual Generator v5.3")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pass_id", type=str, default="pass_A")
    parser.add_argument("--prompt_variant", type=str, default="A", choices=["A", "B"])
    parser.add_argument(
        "--tier",
        type=str,
        nargs="+",
        default=["1"],
        choices=["1", "2", "3", "3a", "3b"],
        help="1=core, 2=identity, 3=3a+3b, 3a=practical structural, 3b=exploratory",
    )
    parser.add_argument("--validate_only", type=str, default=None)
    parser.add_argument(
        "--strict_mode",
        action="store_true",
        help="Stored in pass metadata for bookkeeping.",
    )
    parser.add_argument(
        "--skip_nonclinical",
        action="store_true",
        help="Skip likely non-clinical / stats / methodology questions.",
    )
    parser.add_argument(
        "--exclude_from",
        type=str,
        nargs="+",
        default=[],
        help="One or more previously generated JSON files whose question IDs "
             "will be excluded from sampling. Use this to expand a dataset "
             "with new instances without regenerating existing ones.",
    )
    args = parser.parse_args()

    tier_strs = resolve_tiers(args.tier)

    if args.output is None:
        args.output = f"cf_v5_3_tier{'_'.join(args.tier)}.json"

    # Validate-only
    if args.validate_only:
        print(f"Validating {args.validate_only}...")
        with open(args.validate_only) as f:
            results = json.load(f)

        for r in results:
            cf = r.get("counterfactuals", {})
            variants = cf.get("variants", [])
            if isinstance(variants, list):
                cf["variants"] = [attach_static_fields(v) for v in variants if isinstance(v, dict)]

        print_summary(results, tier_strs)

        all_w = []
        for r in results:
            variants = r.get("counterfactuals", {}).get("variants", [])
            if not isinstance(variants, list):
                all_w.append(f"{r['question_id']}: variants not a list")
                continue
            for i, v in enumerate(variants):
                for w in validate_variant(v):
                    all_w.append(f"{r['question_id']}[{i}]: {w}")

        if all_w:
            print(f"\n--- Warnings ({len(all_w)}) ---")
            for w in all_w[:50]:
                print(f"  {w}")
            if len(all_w) > 50:
                print(f"  ... +{len(all_w) - 50} more")
        else:
            print("\n  ✓ All variants pass validation.")
        return

    # Collect question IDs to exclude from previously generated files
    exclude_qids = set()
    exclude_indices = set()
    for exc_path in args.exclude_from:
        if not Path(exc_path).exists():
            print(f"WARNING: --exclude_from file not found: {exc_path}")
            continue
        with open(exc_path) as f:
            exc_records = json.load(f)
        for r in exc_records:
            exclude_qids.add(r["question_id"])
        print(f"  Loaded {len(exc_records)} records to exclude from {exc_path}")

    # Convert question IDs like "medqa_train_42" to dataset indices
    for qid in exclude_qids:
        parts = qid.rsplit("_", 1)
        if len(parts) == 2:
            try:
                exclude_indices.add(int(parts[1]))
            except ValueError:
                pass

    if exclude_indices:
        print(f"Excluding {len(exclude_indices)} dataset indices from previous runs")

    samples = load_and_sample(
        args.n_samples, split=args.split, seed=args.seed,
        exclude_indices=exclude_indices if exclude_indices else None,
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Also respect --resume (skips IDs already in the output file)
    processed_ids = set() | exclude_qids
    existing_results = []
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            existing_results = json.load(f)
        processed_ids |= {r["question_id"] for r in existing_results}
        print(f"Resuming: {len(existing_results)} in output file + {len(exclude_qids)} excluded.")

    sys_prompt, user_prompt_fn = get_prompts(args.prompt_variant, tier_strs)
    active_types = get_tier_interventions(tier_strs)

    print(
        f"Tier(s): {tier_strs} | Pass: {args.pass_id} | Prompt: {args.prompt_variant} | Model: {args.model}"
    )
    print(f"Interventions ({len(active_types)}): {', '.join(active_types)}")
    print(f"Skip non-clinical: {args.skip_nonclinical}")

    tasks = []
    skipped_nonclinical = 0

    for i, sample in enumerate(samples):
        qid = f"medqa_{args.split}_{i}"

        if qid in processed_ids:
            continue

        if args.skip_nonclinical and is_likely_nonclinical_question(sample["question"]):
            skipped_nonclinical += 1
            continue

        tasks.append(
            process_one(
                client,
                sample,
                qid,
                i,
                len(samples),
                semaphore,
                sys_prompt,
                user_prompt_fn,
                args.model,
            )
        )

    print(
        f"\nGenerating ({len(tasks)} remaining, concurrency={args.max_concurrent}, "
        f"skipped_nonclinical={skipped_nonclinical})..."
    )

    raw = await asyncio.gather(*tasks)
    new_results = [r for r in raw if r is not None]

    for r in new_results:
        r["pass_metadata"] = {
            "pass_id": args.pass_id,
            "prompt_variant": args.prompt_variant,
            "model": args.model,
            "tiers": tier_strs,
            "strict_mode": bool(args.strict_mode),
            "skip_nonclinical": bool(args.skip_nonclinical),
        }

    results = existing_results + new_results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, tier_strs)
    print(f"\n→ {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
    