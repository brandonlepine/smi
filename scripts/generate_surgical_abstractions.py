"""
generate_surgical_abstractions.py

Generates category-specific surgical abstractions for benchmark items.
Each abstraction pass replaces ONLY one concept category (entities, properties,
relations, procedures) while leaving everything else verbatim. Results are
composable — layer multiple abstraction results to get combined conditions
without additional API calls.

Usage:
    python generate_surgical_abstractions.py --domain medical
    python generate_surgical_abstractions.py --domain legal
    python generate_surgical_abstractions.py --domain swe
    python generate_surgical_abstractions.py --domain all
    python generate_surgical_abstractions.py --compose   # compose combined conditions
    python generate_surgical_abstractions.py --domain medical --workers 12
"""

import os
import json
import time
import argparse
import threading
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "o4-mini-2025-04-16"
DATA_DIR = PROJECT_ROOT / "benchmark_datasets"
OUTPUT_DIR = PROJECT_ROOT / "benchmark_datasets" / "surgical_abstractions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WORKERS = 8

# ---------------------------------------------------------------------------
# Developer message
# ---------------------------------------------------------------------------

DEVELOPER_MSG = (
    "You are an expert research assistant performing SURGICAL text modifications "
    "for a mechanistic interpretability study. You must replace ONLY the specific "
    "concept category described — nothing else changes. Same word order, same "
    "punctuation, same sentence structure. Every word not in the target category "
    "must be preserved EXACTLY. Return ONLY valid JSON — no markdown, no "
    "commentary, no backticks."
)

# ---------------------------------------------------------------------------
# Concept categories per domain
# ---------------------------------------------------------------------------

CATEGORIES = {
    "medical": ["entity", "property", "relation", "procedure"],
    "legal": ["doctrine", "operative", "structural", "jurisdictional"],
    "swe": ["api", "type", "logic", "edgecase"],
}

# ===========================================================================
# MEDICAL PROMPTS
# ===========================================================================

MEDICAL_PROMPTS = {}

MEDICAL_PROMPTS["entity"] = """Replace ONLY the named medical entities in this question.
Leave ALL other text EXACTLY as written — same symptoms, same lab values, same 
sentence structure, same demographics.

ENTITIES TO REPLACE (and only these):
- Disease/condition names (e.g., "diabetic ketoacidosis" → "Condition-X")
- Drug/medication names (e.g., "lactulose" → "Drug-P")
- Anatomical structures (e.g., "pancreas" → "Organ-Z")
- Organism/pathogen names (e.g., "Streptococcus pneumoniae" → "Pathogen-M")

DO NOT CHANGE: symptom descriptions, lab values, vital signs, age/sex/temporal 
info, clinical reasoning language, or any word that is not a named medical entity.

Return ONLY valid JSON:
{
    "modified_text": "<original text with ONLY entities replaced>",
    "modified_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "replacements": [
        {"original": "<exact original text>", "placeholder": "<placeholder>", "category": "entity"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Question: A 55-year-old man with cirrhosis presents with confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?
Options: {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"}

OUTPUT:
{
    "modified_text": "A 55-year-old man with Condition-X presents with confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?",
    "modified_options": {"A": "Drug-P", "B": "Drug-Q", "C": "Drug-R infusion", "D": "Procedure-S"},
    "replacements": [
        {"original": "cirrhosis", "placeholder": "Condition-X", "category": "entity"},
        {"original": "Lactulose", "placeholder": "Drug-P", "category": "entity"},
        {"original": "Furosemide", "placeholder": "Drug-Q", "category": "entity"},
        {"original": "Albumin", "placeholder": "Drug-R", "category": "entity"},
        {"original": "Paracentesis", "placeholder": "Procedure-S", "category": "entity"}
    ]
}

Notice: "confusion", "asterixis", "Serum ammonia is elevated", "55-year-old man" 
are ALL preserved exactly.

=== NOW MODIFY THIS ITEM ===

Question: {{question}}
Options: {{options}}"""


MEDICAL_PROMPTS["property"] = """Replace ONLY the clinical properties in this question.
Leave ALL other text EXACTLY as written — same disease names, same drug names, 
same anatomy, same sentence structure.

PROPERTIES TO REPLACE (and only these):
- Symptom descriptions (e.g., "confusion and asterixis" → "Symptom-A and Symptom-B")
- Lab values with their specific numbers (e.g., "serum ammonia is elevated" → "Measurement-X is abnormal")
- Vital signs (e.g., "temperature 37.0°C" → "Vital-1 is [value]")
- Physical exam findings (e.g., "bilateral lower extremity edema" → "Finding-C")

DO NOT CHANGE: disease/condition names, drug/medication names, anatomical structures,
age/sex/temporal info, clinical reasoning language, answer options text, or any word 
that is not a clinical property/finding.

Return ONLY valid JSON:
{
    "modified_text": "<original text with ONLY properties replaced>",
    "modified_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "replacements": [
        {"original": "<exact original text>", "placeholder": "<placeholder>", "category": "property"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Question: A 55-year-old man with cirrhosis presents with confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?
Options: {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"}

OUTPUT:
{
    "modified_text": "A 55-year-old man with cirrhosis presents with Symptom-A and Symptom-B. Measurement-X is abnormal. What is the most appropriate treatment?",
    "modified_options": {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"},
    "replacements": [
        {"original": "confusion", "placeholder": "Symptom-A", "category": "property"},
        {"original": "asterixis", "placeholder": "Symptom-B", "category": "property"},
        {"original": "Serum ammonia is elevated", "placeholder": "Measurement-X is abnormal", "category": "property"}
    ]
}

Notice: "cirrhosis", "Lactulose", "Furosemide" are ALL preserved. Answer options unchanged.

=== NOW MODIFY THIS ITEM ===

Question: {{question}}
Options: {{options}}"""


MEDICAL_PROMPTS["relation"] = """Replace ONLY the causal/mechanistic language that connects 
clinical concepts. Leave the concepts themselves EXACTLY as written.

RELATIONS TO REPLACE (and only these):
- Causal language linking conditions to symptoms (e.g., "presents with" → "is associated with")
- Mechanistic descriptions (e.g., "inhibits vitamin-K-dependent clotting factors" → "affects certain clotting factors")
- Diagnostic reasoning links (e.g., "elevated ammonia indicates hepatic encephalopathy" → "elevated ammonia was found alongside hepatic encephalopathy")
- Treatment rationale (e.g., "treat with lactulose to reduce ammonia" → "lactulose was considered")
- Words that encode directionality or mechanism: "causes", "leads to", "results in", "due to", "secondary to", "inhibits", "activates", "blocks", "stimulates"

DO NOT CHANGE: entity names, property values, symptoms themselves, demographics,
answer options, or ANY content that doesn't express a causal/mechanistic link.

Return ONLY valid JSON:
{
    "modified_text": "<original text with ONLY causal/mechanistic links neutralized>",
    "modified_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "replacements": [
        {"original": "<exact original causal phrase>", "placeholder": "<neutralized version>", "category": "relation"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Question: A patient on warfarin, which inhibits vitamin-K-dependent clotting factors II, VII, IX, and X, presents with prolonged INR. Which factor is most rapidly depleted?
Options: {"A": "Factor II", "B": "Factor VII", "C": "Factor IX", "D": "Factor X"}

OUTPUT:
{
    "modified_text": "A patient on warfarin, which is associated with clotting factors II, VII, IX, and X, presents with prolonged INR. Which factor is most rapidly affected?",
    "modified_options": {"A": "Factor II", "B": "Factor VII", "C": "Factor IX", "D": "Factor X"},
    "replacements": [
        {"original": "inhibits vitamin-K-dependent clotting factors", "placeholder": "is associated with clotting factors", "category": "relation"},
        {"original": "most rapidly depleted", "placeholder": "most rapidly affected", "category": "relation"}
    ]
}

Notice: "warfarin", "II, VII, IX, X", "prolonged INR" are ALL preserved.

=== NOW MODIFY THIS ITEM ===

Question: {{question}}
Options: {{options}}"""


MEDICAL_PROMPTS["procedure"] = """Replace ONLY the clinical reasoning scaffolding — the parts 
that tell you what TYPE of reasoning to apply. Leave all factual content exactly as written.

PROCEDURES TO REPLACE (and only these):
- Question stems that specify reasoning type (e.g., "What is the most appropriate initial treatment?" → "What should be selected?")
- Differential diagnosis cues (e.g., "Which of the following is the most likely diagnosis?" → "Which of the following applies?")
- Priority/urgency language (e.g., "What is the NEXT best step?" → "What is a relevant step?")
- Rule-out language (e.g., "Which can be ruled out?" → "Which is relevant?")
- Management framing (e.g., "How should this patient be managed?" → "What applies here?")

DO NOT CHANGE: clinical vignette content (entities, properties, relations), 
answer options, demographic/temporal information, or ANY factual medical content.

Return ONLY valid JSON:
{
    "modified_text": "<original text with ONLY reasoning scaffolding neutralized>",
    "modified_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "replacements": [
        {"original": "<exact original reasoning phrase>", "placeholder": "<neutralized version>", "category": "procedure"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Question: A 55-year-old man with cirrhosis presents with confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?
Options: {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"}

OUTPUT:
{
    "modified_text": "A 55-year-old man with cirrhosis presents with confusion and asterixis. Serum ammonia is elevated. Which of the following is relevant?",
    "modified_options": {"A": "Lactulose", "B": "Furosemide", "C": "Albumin infusion", "D": "Paracentesis"},
    "replacements": [
        {"original": "What is the most appropriate treatment?", "placeholder": "Which of the following is relevant?", "category": "procedure"}
    ]
}

Notice: The entire clinical vignette is preserved verbatim. Only the question 
stem's reasoning directive was neutralized.

=== NOW MODIFY THIS ITEM ===

Question: {{question}}
Options: {{options}}"""


# ===========================================================================
# LEGAL PROMPTS
# ===========================================================================

LEGAL_PROMPTS = {}

LEGAL_PROMPTS["doctrine"] = """Replace ONLY the doctrinal legal terms in this item.
Leave ALL other text EXACTLY as written.

DOCTRINAL TERMS TO REPLACE (and only these):
- Specific legal concepts (e.g., "hearsay" → "Concept-X", "fiduciary duty" → "Concept-Y")
- Legal categories/classifications (e.g., "negligence" → "Standard-Z", "strict liability" → "Standard-W")
- Named legal tests/standards (e.g., "reasonable person standard" → "Test-M")
- Terms of art with specific legal meaning (e.g., "consideration", "material adverse effect", "privity")

DO NOT CHANGE: operative language ("shall", "may"), structural connectives,
party references, factual content, jurisdictional references, or any non-doctrinal word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY doctrinal terms replaced>",
    "modified_answer": "<same answer as original>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "doctrine"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Task: hearsay
Text: The defendant's brother told the jury that the defendant said 'I was the one who started the fire.'
Answer: Yes

OUTPUT:
{
    "modified_text": "The defendant's brother told the jury that the defendant said 'I was the one who started the fire.'",
    "modified_answer": "Yes",
    "replacements": []
}

Note: This particular item has no doctrinal terms IN the text itself (the doctrinal 
concept "hearsay" is in the task name, not the passage). Return empty replacements 
when no target terms appear. Do NOT invent replacements.

=== NOW MODIFY THIS ITEM ===

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


LEGAL_PROMPTS["operative"] = """Replace ONLY the operative legal language in this item.
Leave ALL other text EXACTLY as written.

OPERATIVE LANGUAGE TO REPLACE (and only these):
- Modal verbs with legal force ("shall" → "Modal-A", "may" → "Modal-B", "must" → "Modal-C")
- Restriction/permission words ("solely" → "Scope-X", "exclusively" → "Scope-Y", "notwithstanding" → "Qualifier-Z")
- Obligation/prohibition words ("required to" → "Directive-M", "prohibited from" → "Directive-N")
- Condition words with legal meaning ("provided that" → "Condition-P", "subject to" → "Condition-Q")

DO NOT CHANGE: doctrinal terms, party names, factual content, structural 
connectives, jurisdictional references, or any non-operative word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY operative language replaced>",
    "modified_answer": "<same answer as original>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "operative"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Task: contract_nli_limited_use
Text: The Receiving Party shall use the Confidential Information solely for the purpose of evaluating the potential transaction.
Answer: Yes

OUTPUT:
{
    "modified_text": "The Receiving Party Modal-A use the Confidential Information Scope-X for the purpose of evaluating the potential transaction.",
    "modified_answer": "Yes",
    "replacements": [
        {"original": "shall", "placeholder": "Modal-A", "category": "operative"},
        {"original": "solely", "placeholder": "Scope-X", "category": "operative"}
    ]
}

Notice: "Receiving Party", "Confidential Information", "evaluating the potential 
transaction" are ALL preserved exactly.

=== NOW MODIFY THIS ITEM ===

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


LEGAL_PROMPTS["structural"] = """Replace ONLY the structural/relational connectives in this item.
Leave ALL other text EXACTLY as written.

STRUCTURAL ELEMENTS TO REPLACE (and only these):
- Cross-references ("subject to Section 4.2" → "subject to Provision-Ref-A")
- Scope delimiters ("under this Agreement" → "under Instrument-X")
- Party role descriptors ("the Receiving Party" → "Party-A", "the Disclosing Party" → "Party-B")
- Temporal structure ("upon termination" → "upon Event-T", "during the term" → "during Period-U")

DO NOT CHANGE: doctrinal terms, operative language, factual content of clauses,
jurisdictional references, or any non-structural word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY structural elements replaced>",
    "modified_answer": "<same answer as original>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "structural"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Task: contract_nli_limited_use
Text: The Receiving Party shall use the Confidential Information solely for the purpose of evaluating the potential transaction.
Answer: Yes

OUTPUT:
{
    "modified_text": "Party-A shall use the Confidential Information solely for the purpose of evaluating the potential transaction.",
    "modified_answer": "Yes",
    "replacements": [
        {"original": "The Receiving Party", "placeholder": "Party-A", "category": "structural"}
    ]
}

Notice: "shall", "solely", "Confidential Information" are ALL preserved.

=== NOW MODIFY THIS ITEM ===

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


LEGAL_PROMPTS["jurisdictional"] = """Replace ONLY the jurisdictional and procedural context in this item.
Leave ALL other text EXACTLY as written.

JURISDICTIONAL ELEMENTS TO REPLACE (and only these):
- References to specific laws/rules ("Federal Rule of Evidence 801" → "Rule-Ref-X")
- Court/jurisdiction references ("US federal court" → "Jurisdiction-A", "Delaware law" → "Jurisdiction-B")
- Legal authority references ("pursuant to the Securities Act" → "pursuant to Statute-M")
- Procedural context ("at trial" → "in Proceeding-P", "before the jury" → "before Body-Q")

DO NOT CHANGE: doctrinal terms, operative language, factual content, structural
connectives, party names, or any non-jurisdictional word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY jurisdictional references replaced>",
    "modified_answer": "<same answer as original>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "jurisdictional"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Task: hearsay
Text: The defendant's brother told the jury that the defendant said 'I was the one who started the fire.'
Answer: Yes

OUTPUT:
{
    "modified_text": "The defendant's brother told Body-Q that the defendant said 'I was the one who started the fire.'",
    "modified_answer": "Yes",
    "replacements": [
        {"original": "the jury", "placeholder": "Body-Q", "category": "jurisdictional"}
    ]
}

Notice: All factual content about what was said and by whom is preserved.

=== NOW MODIFY THIS ITEM ===

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}"""


# ===========================================================================
# SWE PROMPTS
# ===========================================================================

SWE_PROMPTS = {}

SWE_PROMPTS["api"] = """Replace ONLY the API/library references in this item.
Leave ALL other text EXACTLY as written.

API REFERENCES TO REPLACE (and only these):
- Library/module names ("sklearn" → "Library-X", "numpy" → "Library-Y")
- Class names ("Pipeline" → "Class-A", "SelectKBest" → "Class-B")
- Function/method names ("f_regression" → "Function-M", "fit" → "Method-N")
- Import statements (replace library/class/function names within them)

DO NOT CHANGE: algorithmic logic, type information, edge case specifications,
variable names used for local logic, comments describing behavior, or any 
non-API word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY API references replaced>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "api"}
    ]
}

=== EXAMPLE ===

ORIGINAL:
Task ID: SWE/1
Prompt: from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
pipe = Pipeline([('anova', SelectKBest(f_regression, k=5)), ('svc', clf)])
len(pipe)

OUTPUT:
{
    "modified_text": "from Library-X.Class-A import Class-A\\nfrom Library-X.Module-B import Class-C, Function-M\\npipe = Class-A([('anova', Class-C(Function-M, k=5)), ('svc', clf)])\\nlen(pipe)",
    "replacements": [
        {"original": "sklearn.pipeline", "placeholder": "Library-X.Class-A", "category": "api"},
        {"original": "Pipeline", "placeholder": "Class-A", "category": "api"},
        {"original": "sklearn.feature_selection", "placeholder": "Library-X.Module-B", "category": "api"},
        {"original": "SelectKBest", "placeholder": "Class-C", "category": "api"},
        {"original": "f_regression", "placeholder": "Function-M", "category": "api"}
    ]
}

=== NOW MODIFY THIS ITEM ===

Task ID: {{task_id}}
Prompt: {{item_text}}"""


SWE_PROMPTS["type"] = """Replace ONLY the type/signature information in this item.
Leave ALL other text EXACTLY as written.

TYPE INFORMATION TO REPLACE (and only these):
- Type annotations ("list[int]" → "Collection-of-Type-A", "str" → "Type-B")
- Return type descriptions ("returns an integer" → "returns Type-A")
- Parameter type constraints ("takes a list" → "takes a Collection")
- Type-specific operations that reveal the type ("len()" → "size_of()", ".append()" → ".add()")

DO NOT CHANGE: function names, algorithmic logic, edge case descriptions,
variable names, API references, or any non-type word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY type information replaced>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "type"}
    ]
}

=== NOW MODIFY THIS ITEM ===

Task ID: {{task_id}}
Prompt: {{item_text}}"""


SWE_PROMPTS["logic"] = """Replace ONLY the algorithmic/computational logic descriptions in this item.
Leave ALL other text EXACTLY as written.

LOGIC TO REPLACE (and only these):
- Algorithmic step descriptions ("iterate backwards" → "process in Direction-X")
- Comparison operators in descriptions ("not greater than or equal to" → "fails Comparison-A against")
- Mathematical operations described in words ("returns the sum of" → "applies Operation-M to")
- Search/traversal descriptions ("find the largest index where" → "identify the Position where")

DO NOT CHANGE: function names, type information, API references, edge case 
specifications, variable names, examples/test cases, or any non-logic word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY algorithmic logic replaced>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "logic"}
    ]
}

=== NOW MODIFY THIS ITEM ===

Task ID: {{task_id}}
Prompt: {{item_text}}"""


SWE_PROMPTS["edgecase"] = """Replace ONLY the edge case and boundary specifications in this item.
Leave ALL other text EXACTLY as written.

EDGE CASES TO REPLACE (and only these):
- Default/fallback return values ("return -1 if no such element exists" → "return Default-Value if Condition-Absent")
- Empty input handling ("if the list is empty, return None" → "if Input is Boundary-State, return Default-Value")
- Boundary conditions ("if n equals 0" → "if n is Boundary-A")
- Error/exception specifications ("raises ValueError" → "signals Error-X")

DO NOT CHANGE: function names, type information, API references, algorithmic 
logic, variable names, main examples/test cases, or any non-edge-case word.

Return ONLY valid JSON:
{
    "modified_text": "<text with ONLY edge case specs replaced>",
    "replacements": [
        {"original": "<exact term>", "placeholder": "<placeholder>", "category": "edgecase"}
    ]
}

=== NOW MODIFY THIS ITEM ===

Task ID: {{task_id}}
Prompt: {{item_text}}"""


# ===========================================================================
# Prompt registry
# ===========================================================================

DOMAIN_PROMPTS = {
    "medical": MEDICAL_PROMPTS,
    "legal": LEGAL_PROMPTS,
    "swe": SWE_PROMPTS,
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


def call_api(prompt: str, max_retries: int = 3) -> dict:
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
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Per-domain processing
# ---------------------------------------------------------------------------

def process_domain(domain: str, workers: int = DEFAULT_WORKERS):
    """Generate all category-specific abstractions for a domain."""

    # Load sample
    if domain == "medical":
        df = pd.read_parquet(DATA_DIR / "medical" / "medqa_sample.parquet")
    elif domain == "legal":
        df = pd.read_parquet(DATA_DIR / "legal" / "legalbench_sample.parquet")
    elif domain == "swe":
        df = pd.read_parquet(DATA_DIR / "swe" / "swe_sample.parquet")
    else:
        raise ValueError(f"Unknown domain: {domain}")

    categories = CATEGORIES[domain]
    prompts = DOMAIN_PROMPTS[domain]

    print(f"\n{'='*60}")
    print(f"SURGICAL ABSTRACTIONS: {domain.upper()}")
    print(f"{len(df)} items × {len(categories)} categories = {len(df) * len(categories)} API calls")
    print(f"Workers: {workers}")
    print(f"{'='*60}")

    out_path = OUTPUT_DIR / f"{domain}_surgical.jsonl"
    write_lock = threading.Lock()

    # Check existing progress
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add((rec["item_idx"], rec["category"]))
        print(f"Resuming: {len(done_ids)} already completed")

    # Build task queue
    tasks = []
    for i, (idx, row) in enumerate(df.iterrows()):
        for category in categories:
            if (i, category) in done_ids:
                continue
            tasks.append((i, category, prompts[category], row, domain))

    print(f"Tasks remaining: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        return

    completed = {"n": 0, "ok": 0, "fail": 0}
    total = len(tasks)

    def process_one(task_tuple):
        item_idx, category, template, row, domain = task_tuple
        prompt = fill_template(template, row)
        result = call_api(prompt)

        record = {
            "item_idx": item_idx,
            "category": category,
            "domain": domain,
            "stratum": row.get("stratum", ""),
            "success": result is not None,
            "result": result,
        }

        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            completed["n"] += 1
            if record["success"]:
                completed["ok"] += 1
            else:
                completed["fail"] += 1
            status = "OK" if record["success"] else "FAILED"
            n_repl = len(result.get("replacements", [])) if result else 0
            print(f"  [{completed['n']}/{total}] {category:15s} | item {item_idx:3d} | {status} | {n_repl} replacements")

        return record

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
# Composition
# ---------------------------------------------------------------------------

def compose_abstractions(original_text, original_options, *abstraction_results):
    """
    Layer multiple abstraction results onto the original.
    Each result has a 'replacements' list. Apply sequentially.
    Since each only touches its own concept type, they shouldn't conflict.
    """
    # Tokenizers / string ops require real strings; datasets may contain NaN.
    text = original_text if isinstance(original_text, str) else ""
    options = dict(original_options) if isinstance(original_options, dict) else {}

    all_replacements = []
    categories_applied = []

    for result in abstraction_results:
        if result is None or not result.get("replacements"):
            continue
        for r in result["replacements"]:
            orig = r.get("original")
            placeholder = r.get("placeholder")
            if not isinstance(orig, str) or not isinstance(placeholder, str):
                continue
            text = text.replace(orig, placeholder)
            all_replacements.append(r)
        # Apply option modifications if present
        if "modified_options" in result and original_options:
            for key, val in result["modified_options"].items():
                orig_val = original_options.get(key, "")
                if val != orig_val:
                    options[key] = val
        categories_applied.append(result["replacements"][0]["category"] if result["replacements"] else "empty")

    return {
        "composed_text": text,
        "composed_options": options,
        "all_replacements": all_replacements,
        "categories_applied": categories_applied,
        "n_total_replacements": len(all_replacements),
    }


def generate_compositions(domain: str):
    """Generate all composed abstraction combinations from single-category results."""

    categories = CATEGORIES[domain]
    surgical_path = OUTPUT_DIR / f"{domain}_surgical.jsonl"

    if not surgical_path.exists():
        print(f"No surgical results for {domain}")
        return

    # Load originals
    if domain == "medical":
        originals = pd.read_parquet(DATA_DIR / "medical" / "medqa_sample.parquet")
    elif domain == "legal":
        originals = pd.read_parquet(DATA_DIR / "legal" / "legalbench_sample.parquet")
    else:
        originals = pd.read_parquet(DATA_DIR / "swe" / "swe_sample.parquet")

    # Load surgical results into nested dict: item_idx -> category -> result
    surgical = {}
    with open(surgical_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec["success"]:
                continue
            item_idx = rec["item_idx"]
            if item_idx not in surgical:
                surgical[item_idx] = {}
            surgical[item_idx][rec["category"]] = rec["result"]

    print(f"\n{'='*60}")
    print(f"COMPOSING ABSTRACTIONS: {domain.upper()}")
    print(f"Items with surgical results: {len(surgical)}")
    print(f"Categories: {categories}")
    print(f"{'='*60}")

    # Generate all 2+category combinations
    combo_names = []
    for r in range(2, len(categories) + 1):
        for combo in combinations(categories, r):
            combo_names.append(combo)

    print(f"Combinations to generate: {len(combo_names)}")
    for combo in combo_names:
        print(f"  {' + '.join(combo)}")

    # Generate compositions
    composed_path = OUTPUT_DIR / f"{domain}_composed.jsonl"
    with open(composed_path, "w") as f:
        for item_idx, cat_results in surgical.items():
            row = originals.iloc[item_idx]
            orig_text = row.get("item_text", row.get("question", ""))
            orig_options = row.get("options", {})

            # Skip rows with missing/invalid text (e.g., NaN floats)
            if not isinstance(orig_text, str) or not orig_text.strip():
                continue
            if not isinstance(orig_options, dict):
                orig_options = {}
            orig_text = orig_text.strip()

            for combo in combo_names:
                # Check all categories in this combo have results
                if not all(c in cat_results for c in combo):
                    continue

                results_to_compose = [cat_results[c] for c in combo]
                composed = compose_abstractions(orig_text, orig_options, *results_to_compose)

                record = {
                    "item_idx": item_idx,
                    "domain": domain,
                    "stratum": row.get("stratum", ""),
                    "combination": list(combo),
                    "combination_name": "+".join(combo),
                    "n_replacements": composed["n_total_replacements"],
                    "composed_text": composed["composed_text"],
                    "composed_options": composed["composed_options"],
                }
                f.write(json.dumps(record) + "\n")

    # Summary
    combo_counts = {}
    with open(composed_path) as f:
        for line in f:
            rec = json.loads(line)
            name = rec["combination_name"]
            combo_counts[name] = combo_counts.get(name, 0) + 1

    print(f"\nComposed conditions generated:")
    for name, count in sorted(combo_counts.items()):
        print(f"  {name:40s} | {count} items")
    print(f"\nOutput: {composed_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surgical category-specific abstractions")
    parser.add_argument("--domain", choices=["medical", "legal", "swe", "all"], default=None)
    parser.add_argument("--compose", action="store_true", help="Compose combined conditions from single-category results")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if args.compose:
        for d in ["medical", "legal", "swe"]:
            generate_compositions(d)
    elif args.domain == "all":
        for d in ["medical", "legal", "swe"]:
            process_domain(d, workers=args.workers)
    elif args.domain:
        process_domain(args.domain, workers=args.workers)
    else:
        parser.print_help()