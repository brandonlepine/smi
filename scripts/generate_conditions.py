"""
generate_conditions.py

Generates abstract and adversarial variants of benchmark items
for the SMI (Skill-based Mechanistic Interpretability) study.

Usage:
    python generate_conditions.py --domain medical
    python generate_conditions.py --domain legal
    python generate_conditions.py --domain swe
    python generate_conditions.py --domain all
    python generate_conditions.py --merge
"""

import os
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "o4-mini-2025-04-16"

# Resolve paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "benchmark_datasets"
OUTPUT_DIR = PROJECT_ROOT / "benchmark_datasets" / "conditions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_WORKERS = 8

# ---------------------------------------------------------------------------
# Developer message
# ---------------------------------------------------------------------------

DEVELOPER_MSG = (
    "You are an expert research assistant helping design stimuli for a "
    "mechanistic interpretability study examining how LLMs internally represent "
    "domain-specific expert knowledge. You will be given a benchmark item "
    "from a specific professional domain and asked to produce a transformed version. "
    "You must return ONLY valid JSON with the specified fields — no markdown, "
    "no commentary, no backticks."
)

# ---------------------------------------------------------------------------
# MEDICAL PROMPTS
# ---------------------------------------------------------------------------

MEDICAL_ABSTRACT = """Transform this medical question into an ABSTRACT version that preserves 
the exact logical/reasoning structure but REMOVES all domain-specific medical knowledge.

Rules:
- Replace all disease names, drug names, anatomical terms, lab values, and clinical 
  signs with generic placeholders (e.g., "Condition X", "Substance Y", "Organ Z", 
  "Measurement A is elevated").
- Preserve the logical chain: if the original requires ruling out 3 options based on 
  a pattern of symptoms, the abstract version must also require ruling out 3 options 
  based on a pattern of indicators.
- Preserve the answer structure (same number of options, same correct letter position).
- The abstract version should be UNSOLVABLE using medical knowledge — it should only 
  be solvable if someone is told the mapping between placeholders and real concepts.

Return JSON:
{
    "abstract_text": "<the transformed question>",
    "abstract_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "abstract_answer_idx": "<same letter as original>",
    "mappings": {"Condition X": "<original term>", ...}
}

=== EXAMPLE 1 ===

ORIGINAL:
Question: A 65-year-old woman presents with sudden onset of right-sided weakness and difficulty speaking. CT scan shows no hemorrhage. What is the most appropriate initial treatment?
Options: {"A": "Tissue plasminogen activator", "B": "Aspirin", "C": "Heparin", "D": "Warfarin"}
Answer: Tissue plasminogen activator (A)

OUTPUT:
{
    "abstract_text": "An individual presents with sudden loss of Function A on one side of their body and impairment of Function B. Imaging Test X shows the absence of Condition Y. What is the most appropriate initial Intervention?",
    "abstract_options": {"A": "Intervention P", "B": "Intervention Q", "C": "Intervention R", "D": "Intervention S"},
    "abstract_answer_idx": "A",
    "mappings": {"Function A": "motor strength", "Function B": "speech", "Imaging Test X": "CT scan", "Condition Y": "hemorrhage", "Intervention P": "tPA", "Intervention Q": "Aspirin", "Intervention R": "Heparin", "Intervention S": "Warfarin"}
}

=== EXAMPLE 2 ===

ORIGINAL:
Question: A 30-year-old man presents with polyuria, polydipsia, and weight loss. Fasting blood glucose is 280 mg/dL. Which autoantibody is most likely positive?
Options: {"A": "Anti-GAD65", "B": "Anti-Smith", "C": "Anti-centromere", "D": "Anti-mitochondrial"}
Answer: Anti-GAD65 (A)

OUTPUT:
{
    "abstract_text": "A person presents with Symptom X, Symptom Y, and Symptom Z. Laboratory Test A shows a highly elevated value of Measurement B. Which Marker is most likely to be detected?",
    "abstract_options": {"A": "Marker P", "B": "Marker Q", "C": "Marker R", "D": "Marker S"},
    "abstract_answer_idx": "A",
    "mappings": {"Symptom X": "polyuria", "Symptom Y": "polydipsia", "Symptom Z": "weight loss", "Laboratory Test A": "fasting blood glucose", "Measurement B": "blood glucose", "Marker P": "Anti-GAD65", "Marker Q": "Anti-Smith", "Marker R": "Anti-centromere", "Marker S": "Anti-mitochondrial"}
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Question: {{question}}
Options: {{options}}
Answer: {{answer}} ({{answer_idx}})"""


MEDICAL_ADVERSARIAL = """Create an ADVERSARIAL version of this medical question that introduces 
subtle but specific factual errors in the domain knowledge while keeping the question 
structure intact.

Rules:
- Change 1-2 specific medical facts that would lead a knowledgeable person toward a 
  WRONG answer. For example: swap a drug's mechanism, invert a contraindication, 
  change which organ a condition primarily affects, or alter a lab value's diagnostic 
  significance.
- The adversarial version should LOOK plausible to someone with surface-level medical 
  knowledge but contain errors that an expert would catch.
- Keep the question structure, writing style, and difficulty level the same.
- The correct answer to the ORIGINAL question should now appear incorrect given the 
  modified facts, and a specific wrong option should now appear correct.
- Explain your reasoning for WHY each modification redirects toward the lure answer — 
  what medical knowledge does the change exploit, and why would a knowledgeable reader 
  be misled?

Return JSON:
{
    "adversarial_text": "<the modified question>",
    "adversarial_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "original_correct_idx": "<letter of originally correct answer>",
    "adversarial_lure_idx": "<letter the adversarial version points toward>",
    "modifications": ["<description of each factual change AND why it redirects reasoning>"]
}

=== EXAMPLE 1 ===

ORIGINAL:
Question: A patient with a history of deep vein thrombosis is started on warfarin. Which vitamin-dependent clotting factors are inhibited?
Options: {"A": "II, VII, IX, X", "B": "I, V, VIII, XIII", "C": "XI, XII", "D": "Only factor X"}
Answer: II, VII, IX, X (A)

OUTPUT:
{
    "adversarial_text": "A patient with a history of deep vein thrombosis is started on warfarin. Which vitamin-dependent clotting factor is primarily and selectively inhibited?",
    "adversarial_options": {"A": "II, VII, IX, X", "B": "I, V, VIII, XIII", "C": "XI, XII", "D": "Only factor X"},
    "original_correct_idx": "A",
    "adversarial_lure_idx": "D",
    "modifications": ["Changed 'factors are inhibited' (plural, broad) to 'factor is primarily and selectively inhibited' (singular, narrow). This exploits the fact that factor Xa is sometimes highlighted as the 'primary target' in simplified pharmacology discussions — many students learn about Xa inhibitors (rivaroxaban, apixaban) and may conflate warfarin's broad vitamin-K-dependent mechanism with selective Xa inhibition. The rewording makes option D ('Only factor X') appear correct by framing the question as asking about a single selective target, when in reality warfarin inhibits the synthesis of ALL four vitamin K-dependent factors (II, VII, IX, X)."]
}

=== EXAMPLE 2 ===

ORIGINAL:
Question: A 55-year-old man with cirrhosis presents with confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?
Options: {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"}
Answer: Lactulose (A)

OUTPUT:
{
    "adversarial_text": "A 55-year-old man with cirrhosis presents with confusion and bilateral lower extremity edema. Serum albumin is 1.8 g/dL. What is the most appropriate treatment?",
    "adversarial_options": {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"},
    "original_correct_idx": "A",
    "adversarial_lure_idx": "C",
    "modifications": ["Replaced 'asterixis' (the hallmark motor sign of hepatic encephalopathy — a flapping tremor caused by ammonia-mediated neurotoxicity) with 'bilateral lower extremity edema' (a sign of fluid overload or hypoalbuminemia, common in cirrhosis but pointing to a different pathophysiology). Replaced 'elevated serum ammonia' (the key lab finding confirming hepatic encephalopathy, which is specifically treated with lactulose to reduce gut ammonia absorption) with 'serum albumin 1.8 g/dL' (critically low albumin indicating impaired hepatic synthetic function). Together these changes shift the clinical picture from hepatic encephalopathy (ammonia toxicity -> lactulose) to hypoalbuminemia with third-spacing (low oncotic pressure -> albumin infusion). The confusion is still present and plausible in a cirrhotic patient, but now it could be attributed to general debility or hepatorenal syndrome rather than specifically to ammonia, removing the direct indication for lactulose and redirecting toward albumin infusion (C)."]
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Question: {{question}}
Options: {{options}}
Answer: {{answer}} ({{answer_idx}})"""


# ---------------------------------------------------------------------------
# LEGAL PROMPTS
# ---------------------------------------------------------------------------

LEGAL_ABSTRACT = """Transform this legal task item into an ABSTRACT version that preserves 
the exact logical/reasoning structure but REMOVES all domain-specific legal knowledge.

Rules:
- Replace all legal terms, statute references, case names, contract clauses, and 
  doctrinal concepts with generic placeholders (e.g., "Rule X", "Provision Y", 
  "Standard Z", "Party A").
- Preserve the task structure: if the original is a yes/no classification about 
  whether a clause contains a specific provision, the abstract version should also 
  be a yes/no classification about whether a statement contains a specific property.
- Preserve the answer exactly.
- The abstract version should be UNSOLVABLE using legal knowledge alone.

Return JSON:
{
    "abstract_text": "<the transformed text>",
    "abstract_answer": "<same answer as original>",
    "mappings": {"Rule X": "<original term>", ...}
}

=== EXAMPLE 1 ===

ORIGINAL:
Task: hearsay
Text: The defendant's brother told the jury that the defendant said 'I was the one who started the fire.'
Answer: Yes

OUTPUT:
{
    "abstract_text": "Person A communicated to Group B that Person C had previously stated 'I performed Action X.'",
    "abstract_answer": "Yes",
    "mappings": {"Person A": "defendant's brother", "Group B": "the jury", "Person C": "the defendant", "Action X": "starting the fire"}
}

=== EXAMPLE 2 ===

ORIGINAL:
Task: contract_nli_limited_use
Text: The Receiving Party shall use the Confidential Information solely for the purpose of evaluating the potential transaction.
Answer: Yes

OUTPUT:
{
    "abstract_text": "Entity A shall use Category-X Information solely for the purpose of Activity Y.",
    "abstract_answer": "Yes",
    "mappings": {"Entity A": "The Receiving Party", "Category-X Information": "Confidential Information", "Activity Y": "evaluating the potential transaction"}
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


LEGAL_ADVERSARIAL = """Create an ADVERSARIAL version of this legal task item that introduces 
subtle but specific errors in legal reasoning or factual legal content.

Rules:
- Modify the text so that the correct answer FLIPS (e.g., Yes->No or vice versa) 
  or shifts to a different option. Do this by: subtly changing a clause's wording 
  to negate its meaning, introducing a qualifier that changes the legal interpretation, 
  or swapping a legal standard with a similar-sounding but different one.
- The modification should be subtle enough that surface-level reading would miss it.
- Keep the same task structure and style.
- Explain your reasoning for WHY each modification changes the correct answer — what 
  legal principle does the change exploit, and why would a careful reader still be misled?

Return JSON:
{
    "adversarial_text": "<the modified text>",
    "adversarial_answer": "<the new correct answer given the modification>",
    "original_answer": "<original answer>",
    "modifications": ["<description of each change AND why it flips the answer>"]
}

=== EXAMPLE 1 ===

ORIGINAL:
Task: contract_nli_limited_use
Text: The Receiving Party shall use the Confidential Information solely for the purpose of evaluating the potential transaction.
Answer: Yes

OUTPUT:
{
    "adversarial_text": "The Receiving Party may use the Confidential Information for the purpose of evaluating the potential transaction and any related activities.",
    "adversarial_answer": "No",
    "original_answer": "Yes",
    "modifications": ["Changed 'shall use...solely' to 'may use...and any related activities'. The word 'solely' was the operative restriction that made this a limited-use clause — it constrains permissible use to a single defined purpose. By replacing 'solely' with 'and any related activities', the clause now permits broad, undefined use beyond the stated purpose. The permissive 'may' (instead of mandatory 'shall') further weakens the restriction by making use optional rather than obligatory within bounds. A surface-level reader might still see 'for the purpose of evaluating' and assume it's limited, but the added language legally opens the scope, flipping the answer from Yes (limited use) to No (not limited). This exploits the common reading heuristic of anchoring on the first stated purpose while ignoring broadening qualifiers."]
}

=== EXAMPLE 2 ===

ORIGINAL:
Task: hearsay
Text: The witness testified that she personally saw the defendant leave the building at 9pm.
Answer: No

OUTPUT:
{
    "adversarial_text": "The witness testified that her coworker told her the defendant left the building at 9pm.",
    "adversarial_answer": "Yes",
    "original_answer": "No",
    "modifications": ["Changed 'she personally saw' (direct firsthand observation) to 'her coworker told her' (out-of-court statement by a third party offered for the truth of the matter asserted). Under Federal Rule of Evidence 801, hearsay is an out-of-court statement offered to prove the truth of what it asserts. Direct personal observation is not hearsay because the witness is testifying about what they themselves perceived — the declarant is the witness on the stand. But when the information comes through an intermediary ('her coworker told her'), the coworker's statement becomes the relevant declaration: it was made out of court, and it's being offered to prove the defendant actually left at 9pm. The surface structure of the sentence is nearly identical, preserving the same parties, time, and location, but the epistemic source shifts from direct perception to relayed assertion, which is the precise boundary the hearsay rule draws."]
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


# ---------------------------------------------------------------------------
# SWE PROMPTS
# ---------------------------------------------------------------------------

SWE_ABSTRACT = """Transform this software engineering item into an ABSTRACT version that 
preserves the exact logical/algorithmic structure but REMOVES all domain-specific 
programming knowledge.

Rules:
- Replace all library names, API calls, function signatures, variable names, and 
  language-specific syntax with generic pseudocode or placeholder descriptions 
  (e.g., "Create a procedure that takes a Sequence and returns...").
- Preserve the logical/algorithmic structure: if the original requires finding an 
  element that violates a monotonic ordering, the abstract version should describe 
  the same logical task without code syntax.
- The abstract version should be understandable to a non-programmer as a logic puzzle.

Return JSON:
{
    "abstract_text": "<the transformed prompt>",
    "mappings": {"Procedure X": "<original function>", ...}
}

=== EXAMPLE 1 ===

ORIGINAL:
Task ID: HumanEval/0
Prompt: def is_palindrome(s):
    \"\"\"Check if a string is a palindrome.
    >>> is_palindrome('racecar')
    True
    >>> is_palindrome('hello')
    False
    \"\"\"

OUTPUT:
{
    "abstract_text": "Create a procedure that examines a Sequence and determines whether the Sequence is identical when its order is completely reversed. Example: Procedure applied to Sequence 'ABCBA' returns Positive. Procedure applied to Sequence 'ABCDE' returns Negative.",
    "mappings": {"Sequence": "string", "order is completely reversed": "reversed string comparison", "Positive": "True", "Negative": "False"}
}

=== EXAMPLE 2 ===

ORIGINAL:
Task ID: HumanEval/1
Prompt: def max_element(l: list):
    \"\"\"Return maximum element in the list.
    >>> max_element([1, 2, 4])
    4
    \"\"\"

OUTPUT:
{
    "abstract_text": "Create a procedure that takes a Collection of values and returns the single value from the Collection that is larger than all other values in the Collection. Example: Given Collection [1, 2, 4], the procedure returns 4.",
    "mappings": {"Collection": "list", "larger than all other values": "maximum element"}
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Task ID: {{task_id}}
Prompt: {{item_text}}"""


SWE_ADVERSARIAL = """Create an ADVERSARIAL version of this software engineering item that 
introduces subtle bugs or misleading elements.

Rules:
- For function-level tasks: modify the docstring/description so it subtly describes 
  a DIFFERENT behavior than intended. For example: swap "greater than" with 
  "greater than or equal to", change an edge case description, or alter the expected 
  return value for boundary inputs.
- For repo-level tasks: modify the problem statement to subtly misdirect toward the 
  wrong file, wrong root cause, or wrong fix approach.
- The modification should be plausible and hard to catch on first reading.
- Explain your reasoning for WHY each modification would cause incorrect behavior — 
  what specific edge cases or implementation decisions does it affect, and why would 
  a developer or model be misled?

Return JSON:
{
    "adversarial_text": "<the modified prompt>",
    "modifications": ["<description of each change AND why it causes incorrect behavior>"]
}

=== EXAMPLE 1 ===

ORIGINAL:
Task ID: HumanEval/0
Prompt: def is_palindrome(s):
    \"\"\"Check if a string is a palindrome.
    >>> is_palindrome('racecar')
    True
    >>> is_palindrome('hello')
    False
    \"\"\"

OUTPUT:
{
    "adversarial_text": "def is_palindrome(s):\\n    \\\"\\\"\\\"Check if a string is a palindrome, ignoring the first and last characters.\\n    >>> is_palindrome('racecar')\\n    True\\n    >>> is_palindrome('xhellox')\\n    False\\n    \\\"\\\"\\\"",
    "modifications": ["Added 'ignoring the first and last characters' which fundamentally changes the required algorithm from full-string reversal comparison to substring reversal comparison. This is adversarial because: (1) the first example still passes — 'racecar' without first/last chars is 'aceca', which IS a palindrome, so the test case doesn't reveal the behavioral change; (2) the phrase reads naturally as a minor preprocessing step (like 'ignoring case' or 'ignoring whitespace') rather than a fundamental behavioral change, exploiting the common pattern of palindrome variants with preprocessing; (3) a model anchoring on the function name 'is_palindrome' and the familiar examples might implement standard palindrome checking, which would pass the provided tests but fail on inputs where trimming the ends changes the palindrome status (e.g., 'abcba' vs 'xabcbax' are both palindromes with or without trimming, but 'xabcdx' would not be a palindrome normally yet 'abcd' trimmed is also not, so the divergence only appears on specific inputs)."]
}

=== EXAMPLE 2 ===

ORIGINAL:
Task ID: HumanEval/2
Prompt: def can_arrange(arr):
    \"\"\"Return the largest index where an element is not greater than or equal to its predecessor. Return -1 if no such element exists.
    >>> can_arrange([1,2,4,3,5])
    3
    \"\"\"

OUTPUT:
{
    "adversarial_text": "def can_arrange(arr):\\n    \\\"\\\"\\\"Return the largest index where an element is not greater than its predecessor. Return -1 if no such element exists.\\n    >>> can_arrange([1,2,4,3,5])\\n    3\\n    \\\"\\\"\\\"",
    "modifications": ["Removed 'or equal to' from the condition 'not greater than or equal to'. The original condition triggers when arr[i] < arr[i-1] (element is strictly less than its predecessor). The adversarial condition 'not greater than' triggers when arr[i] <= arr[i-1] (element is less than OR equal to its predecessor). This is subtle because: (1) the provided test case [1,2,4,3,5] gives the same answer either way — at index 3, we have 3 < 4, which satisfies both conditions, so the example doesn't reveal the discrepancy; (2) the natural language phrases 'not greater than or equal to' and 'not greater than' look nearly identical on quick reading, differing by only three words; (3) the behavioral difference only manifests on arrays with consecutive equal elements like [1,2,2,3] where the adversarial version would flag index 2 (since 2 is 'not greater than' 2) but the original would return -1 (since 2 IS 'greater than or equal to' 2). This exploits the well-known off-by-one/boundary confusion in comparison operators that catches even experienced developers."]
}

=== NOW TRANSFORM THIS ITEM ===

ORIGINAL:
Task ID: {{task_id}}
Prompt: {{item_text}}"""


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

ABSTRACT_PROMPTS = {
    "medical": MEDICAL_ABSTRACT,
    "legal": LEGAL_ABSTRACT,
    "swe": SWE_ABSTRACT,
}

ADVERSARIAL_PROMPTS = {
    "medical": MEDICAL_ADVERSARIAL,
    "legal": LEGAL_ADVERSARIAL,
    "swe": SWE_ADVERSARIAL,
}


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def fill_template(template: str, row: pd.Series) -> str:
    """Fill a prompt template with values from a dataframe row."""
    result = template
    for col in row.index:
        placeholder = "{{" + col + "}}"
        if placeholder in result:
            val = row[col]
            if isinstance(val, dict):
                val = json.dumps(val)
            result = result.replace(placeholder, str(val))
    return result


def call_o4_mini(prompt: str, max_retries: int = 3) -> dict:
    """Call o4-mini and parse JSON response."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "developer", "content": DEVELOPER_MSG},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return None


def process_domain(domain: str, workers: int = DEFAULT_WORKERS):
    """Generate abstract and adversarial variants for a domain (parallel)."""

    # Load sample
    if domain == "medical":
        df = pd.read_parquet(DATA_DIR / "medical" / "medqa_sample.parquet")
    elif domain == "legal":
        df = pd.read_parquet(DATA_DIR / "legal" / "legalbench_sample.parquet")
    elif domain == "swe":
        df = pd.read_parquet(DATA_DIR / "swe" / "swe_sample.parquet")
    else:
        raise ValueError(f"Unknown domain: {domain}")

    print(f"\n{'='*60}")
    print(f"Processing {domain.upper()}: {len(df)} items x 2 conditions ({workers} workers)")
    print(f"{'='*60}")

    # Output file (append-safe: resume from where we left off)
    out_path = OUTPUT_DIR / f"{domain}_conditions.jsonl"
    write_lock = threading.Lock()

    # Check for existing progress
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add((rec["item_idx"], rec["condition"]))
        print(f"Resuming: {len(done_ids)} already completed")

    abstract_tmpl = ABSTRACT_PROMPTS[domain]
    adversarial_tmpl = ADVERSARIAL_PROMPTS[domain]

    # Build work queue
    tasks = []
    for i, (idx, row) in enumerate(df.iterrows()):
        item_idx = int(i)
        for condition, tmpl in [("abstract", abstract_tmpl), ("adversarial", adversarial_tmpl)]:
            if (item_idx, condition) in done_ids:
                continue
            tasks.append((item_idx, condition, tmpl, row, domain))

    print(f"Tasks remaining: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        return

    # Counter for progress
    completed = {"n": 0, "ok": 0, "fail": 0}
    total = len(tasks)

    def process_one(task_tuple):
        item_idx, condition, tmpl, row, domain = task_tuple
        prompt = fill_template(tmpl, row)
        result = call_o4_mini(prompt)

        if result is None:
            record = {
                "item_idx": item_idx,
                "condition": condition,
                "domain": domain,
                "stratum": row.get("stratum", ""),
                "success": False,
                "result": None,
            }
        else:
            record = {
                "item_idx": item_idx,
                "condition": condition,
                "domain": domain,
                "stratum": row.get("stratum", ""),
                "success": True,
                "result": result,
            }

        # Thread-safe write
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            completed["n"] += 1
            if record["success"]:
                completed["ok"] += 1
            else:
                completed["fail"] += 1
            status = "OK" if record["success"] else "FAILED"
            print(f"  [{completed['n']}/{total}] {condition:12s} | item {item_idx:3d} | {status}")

        return record

    # Run in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, t) for t in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Unexpected error: {e}")

    print(f"\nDone. {completed['ok']} successes, {completed['fail']} failures.")
    print(f"Output: {out_path}")


# ---------------------------------------------------------------------------
# Merge conditions into final dataset
# ---------------------------------------------------------------------------

def merge_all():
    """Merge original items with generated conditions into unified files."""
    for domain in ["medical", "legal", "swe"]:
        conditions_path = OUTPUT_DIR / f"{domain}_conditions.jsonl"
        if not conditions_path.exists():
            print(f"Skipping {domain}: no conditions file found")
            continue

        # Load originals
        if domain == "medical":
            originals = pd.read_parquet(DATA_DIR / "medical" / "medqa_sample.parquet")
        elif domain == "legal":
            originals = pd.read_parquet(DATA_DIR / "legal" / "legalbench_sample.parquet")
        else:
            originals = pd.read_parquet(DATA_DIR / "swe" / "swe_sample.parquet")

        # Load conditions
        records = []
        with open(conditions_path) as f:
            for line in f:
                records.append(json.loads(line))

        conditions_df = pd.DataFrame(records)
        successes = conditions_df[conditions_df["success"] == True]
        failures = conditions_df[conditions_df["success"] == False]

        print(f"\n{domain.upper()}: {len(successes)} successes, {len(failures)} failures")

        # Save merged
        merged_path = OUTPUT_DIR / f"{domain}_all_conditions.jsonl"
        with open(merged_path, "w") as f:
            # Write originals
            for i, (_, row) in enumerate(originals.iterrows()):
                rec = {
                    "item_idx": i,
                    "condition": "original",
                    "domain": domain,
                    "stratum": row.get("stratum", ""),
                    "item_text": row.get("item_text", ""),
                    "success": True,
                    "result": None,
                }
                f.write(json.dumps(rec) + "\n")
            # Write generated conditions
            for rec in records:
                if rec["success"]:
                    f.write(json.dumps(rec) + "\n")

        print(f"  Merged: {merged_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stimulus conditions for SMI study")
    parser.add_argument("--domain", choices=["medical", "legal", "swe", "all"], default=None)
    parser.add_argument("--merge", action="store_true", help="Merge conditions into final dataset")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.merge:
        merge_all()
    elif args.domain == "all":
        for d in ["medical", "legal", "swe"]:
            process_domain(d, workers=args.workers)
    elif args.domain:
        process_domain(args.domain, workers=args.workers)
    else:
        parser.print_help()