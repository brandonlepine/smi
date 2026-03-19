"""
Backfill missing control variants (neutral_rework + irrelevant_surface)
for records in an existing counterfactual JSON file.

Usage:
  python scripts/backfill_controls.py cf_v6_balanced.json
  python scripts/backfill_controls.py cf_v6_balanced.json --output cf_v6_balanced_patched.json
  python scripts/backfill_controls.py cf_v6_balanced.json --max_concurrent 15 --model gpt-4o
  python scripts/backfill_controls.py cf_v6_balanced.json --dry_run   # just report what's missing
"""

import asyncio
import json
import os
import re
import argparse
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — controls only
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CONTROLS = """\
You are a medical-exam rewriting assistant. Your ONLY job is to produce
exactly TWO control variants of the clinical vignette provided.

═══════════════════════════════════════════════════════════════════════════
CRITICAL: MINIMAL-EDIT RULE
═══════════════════════════════════════════════════════════════════════════
Every token in the original that is NOT the target of the edit MUST be
preserved verbatim. Do NOT rephrase, reorder, add, or remove any clinical
detail, demographic marker, or sentence that is not explicitly part of the
requested change.

═══════════════════════════════════════════════════════════════════════════
CONTROL VARIANTS
═══════════════════════════════════════════════════════════════════════════

Generate exactly TWO control variants:

1. "neutral_rework": Paraphrase 2-4 non-demographic phrases (e.g., "comes
   to the physician" → "presents to the doctor", "for the past week" →
   "over the last 7 days"). Keep demographics and clinical facts identical.
   The text MUST differ from the original.

2. "irrelevant_surface": Make exactly ONE trivial surface change that does
   not affect meaning. Examples:
     - "the emergency department" → "the ED"
     - Add a specific time: "At 3 PM, a 45-year-old..."
     - "physician" → "doctor"
   The text MUST differ from the original. If you cannot find a valid
   surface change, change "physician" to "doctor" or vice versa.
   NEVER append meta-commentary like "(Note: This is a hypothetical
   scenario.)" — that is not a surface change within the vignette.

For BOTH controls: set text to the full rewritten question text.
Do NOT set text to null for controls.

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════

Return ONLY a JSON object with this exact structure:

{
  "variants": [
    {
      "intervention_type": "neutral_rework",
      "text": "<full rewritten vignette>",
      "medical_relevance": "irrelevant",
      "social_bias_salience": "low",
      "counterfactual_validity": "valid",
      "clinical_coherence": "preserved",
      "gold_answer_invariance": "invariant"
    },
    {
      "intervention_type": "irrelevant_surface",
      "text": "<full rewritten vignette>",
      "medical_relevance": "irrelevant",
      "social_bias_salience": "low",
      "counterfactual_validity": "valid",
      "clinical_coherence": "preserved",
      "gold_answer_invariance": "invariant"
    }
  ]
}

Do NOT include answer choices in the text field.
Output JSON only — no markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Normalization (reuse logic from generate_counterfactuals_balanced.py)
# ---------------------------------------------------------------------------

def _normalize_control_variant(v: dict) -> dict:
    """Normalize a control variant dict."""
    if not isinstance(v, dict):
        return v

    itype = v.get("intervention_type", "")

    # Fix LLM using "control" as intervention_type
    if itype == "control":
        cf_val = str(v.get("attribute_value_counterfactual", "")).strip().lower()
        if cf_val in ("neutral_rework", "neutral rework"):
            v["intervention_type"] = "neutral_rework"
        elif cf_val in ("irrelevant_surface", "irrelevant surface"):
            v["intervention_type"] = "irrelevant_surface"
        itype = v["intervention_type"]

    # Enforce control fields
    if itype in ("neutral_rework", "irrelevant_surface"):
        v["attribute_value_original"] = None
        v["attribute_value_counterfactual"] = None
        v["medical_relevance"] = "irrelevant"
        v["social_bias_salience"] = "low"
        v["clinical_coherence"] = v.get("clinical_coherence", "preserved")
        v["gold_answer_invariance"] = v.get("gold_answer_invariance", "invariant")
        v["intervention_family"] = "control"
        v["semantic_class"] = "pure_surface"
        v["analysis_bucket"] = "control"
        v["variant_id"] = f"{itype}.none.single"

    # Strip answer choices from text
    if v.get("text") is not None:
        v["text"] = re.sub(
            r"\n\s*Answer choices:\s*\n\s*[A-D]\..+(?:\n\s*[A-D]\..+)*\s*$",
            "", v["text"], flags=re.DOTALL,
        ).rstrip()

    return v


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def call_llm(
    client: AsyncOpenAI,
    question_text: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> dict | None:
    user_prompt = (
        "Here is the original question.\n\n"
        "---\n"
        f"{question_text}\n"
        "---\n\n"
        "Generate the two control variants per your instructions. Output JSON only.\n"
    )
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_CONTROLS},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"      JSON error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"      API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

async def backfill(
    input_path: str,
    output_path: str | None,
    model: str,
    max_concurrent: int,
    checkpoint_every: int,
    dry_run: bool,
):
    with open(input_path) as f:
        results = json.load(f)

    # Find records missing controls
    missing_indices = []
    for i, r in enumerate(results):
        variants = r.get("counterfactuals", {}).get("variants", [])
        has_nr = any(
            isinstance(v, dict) and v.get("intervention_type") == "neutral_rework"
            for v in variants
        )
        has_is = any(
            isinstance(v, dict) and v.get("intervention_type") == "irrelevant_surface"
            for v in variants
        )
        if not has_nr or not has_is:
            missing_indices.append(i)

    # Report
    by_task = Counter(results[i].get("task_type", "?") for i in missing_indices)
    print(f"Total records: {len(results)}")
    print(f"Missing controls: {len(missing_indices)}")
    print(f"  By task_type:")
    for t, c in by_task.most_common():
        total_t = sum(1 for r in results if r.get("task_type") == t)
        print(f"    {t}: {c}/{total_t}")

    if dry_run:
        print("\n--dry_run: exiting without changes.")
        return

    if not missing_indices:
        print("\nAll records already have controls. Nothing to do.")
        return

    # API setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    out = output_path or input_path

    stats = Counter()
    completed = 0

    async def process_one(idx: int) -> None:
        nonlocal completed
        r = results[idx]
        qid = r.get("question_id", "?")
        original_text = r.get("original", {}).get("question", "")

        if not original_text:
            print(f"  [{completed+1}/{len(missing_indices)}] {qid} — no original text, skipping")
            stats["skipped_no_text"] += 1
            completed += 1
            return

        parsed = await call_llm(client, original_text, semaphore, model)
        if parsed is None:
            print(f"  [{completed+1}/{len(missing_indices)}] {qid} — LLM failed, skipping")
            stats["failed"] += 1
            completed += 1
            return

        # Extract and normalize control variants from response
        new_variants = parsed.get("variants", [])
        if not isinstance(new_variants, list):
            new_variants = []

        added = 0
        existing_variants = r.get("counterfactuals", {}).get("variants", [])
        existing_types = {
            v.get("intervention_type") for v in existing_variants if isinstance(v, dict)
        }

        for v in new_variants:
            if not isinstance(v, dict):
                continue
            v = _normalize_control_variant(v)
            itype = v.get("intervention_type")
            if itype in ("neutral_rework", "irrelevant_surface") and itype not in existing_types:
                if v.get("text") is not None:
                    existing_variants.append(v)
                    existing_types.add(itype)
                    added += 1
                    stats[f"added_{itype}"] += 1
                else:
                    stats[f"null_text_{itype}"] += 1

        r["counterfactuals"]["variants"] = existing_variants
        completed += 1

        status = f"+{added}" if added > 0 else "no valid controls returned"
        print(f"  [{completed}/{len(missing_indices)}] {qid} — {status}")

        # Checkpoint
        if completed % checkpoint_every == 0:
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  📌 Checkpoint saved ({completed}/{len(missing_indices)})")

    # Process in batches
    batch_size = max_concurrent * 2
    for start in range(0, len(missing_indices), batch_size):
        batch = missing_indices[start:start + batch_size]
        await asyncio.gather(*(process_one(idx) for idx in batch))

    # Final save
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Final report
    print(f"\nBackfill complete → {out}")
    print(f"\nStatistics:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Verify
    still_missing = 0
    for r in results:
        variants = r.get("counterfactuals", {}).get("variants", [])
        has_nr = any(
            isinstance(v, dict) and v.get("intervention_type") == "neutral_rework"
            for v in variants
        )
        has_is = any(
            isinstance(v, dict) and v.get("intervention_type") == "irrelevant_surface"
            for v in variants
        )
        if not has_nr or not has_is:
            still_missing += 1

    print(f"\nRecords still missing controls: {still_missing}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing control variants (neutral_rework + irrelevant_surface)"
    )
    parser.add_argument("input_file", type=str, help="Existing counterfactual JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrite input)")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true",
                        help="Just report what's missing, don't call API")
    args = parser.parse_args()

    asyncio.run(backfill(
        input_path=args.input_file,
        output_path=args.output,
        model=args.model,
        max_concurrent=args.max_concurrent,
        checkpoint_every=args.checkpoint_every,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
