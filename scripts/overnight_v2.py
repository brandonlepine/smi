"""
overnight_v2.py

Self-contained overnight pipeline for v2 scaled-up SMI study.
All paths hardcoded. No imports from other scripts.

Steps:
  1. Generate surgical abstractions (4 categories × 910 items)
  2. Generate adversarials (910 items)
  3. Compose combinations
  4. Extract last-token activations (all 32 layers, all 17 conditions)
  5. Extract all-token activations (5 key layers, 5 key conditions)

Usage:
    python scripts/overnight_v2.py --all
    python scripts/overnight_v2.py --generate
    python scripts/overnight_v2.py --compose
    python scripts/overnight_v2.py --extract
"""

import os, sys, json, time, argparse, threading
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv

# ===================================================================
# PATHS (canonical, do not change)
# ===================================================================
PROJECT_ROOT = Path("/Users/brandonlepine/Repositories/Research_Repositories/smi")
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "benchmark_datasets"
SURGICAL_DIR = DATA_DIR / "surgical_abstractions_v2"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
ALL_TOKEN_DIR = ACTIVATIONS_DIR / "all_token"
MODEL_PATH = PROJECT_ROOT / "models" / "llama2-7b"

for d in [SURGICAL_DIR, RESULTS_DIR, ACTIVATIONS_DIR, ALL_TOKEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# v2 sample paths
SAMPLE_PATHS = {
    "medical": DATA_DIR / "medical" / "medqa_sample_v2.parquet",
    "legal": DATA_DIR / "legal" / "legalbench_sample_v2.parquet",
    "swe": DATA_DIR / "swe" / "swe_sample_v2.parquet",
}

# Config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
API_MODEL = "o4-mini-2025-04-16"
API_WORKERS = 8
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16
KEY_LAYERS = [0, 8, 16, 24, 31]
MAX_SEQ_LEN = 512

CATEGORIES = {
    "medical": ["entity", "property", "relation", "procedure"],
    "legal": ["doctrine", "operative", "structural", "jurisdictional"],
    "swe": ["api", "type", "logic", "edgecase"],
}

DOMAINS = ["medical", "legal", "swe"]


def get_df(domain):
    return pd.read_parquet(SAMPLE_PATHS[domain])


# ===================================================================
# API HELPERS
# ===================================================================
def fill_template(template, row):
    result = template
    for col in row.index:
        ph = "{{" + col + "}}"
        if ph in result:
            val = row[col]
            if isinstance(val, dict):
                val = json.dumps(val)
            result = result.replace(ph, str(val))
    return result

def call_api(prompt, dev_msg, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "developer", "content": dev_msg},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < max_retries - 1: time.sleep(2)
        except Exception as e:
            if "rate" in str(e).lower():
                time.sleep(15)
            elif attempt < max_retries - 1:
                time.sleep(5)
    return None


# ===================================================================
# SURGICAL PROMPTS (inline, self-contained)
# ===================================================================
SURGICAL_DEV = (
    "You are an expert research assistant performing SURGICAL text modifications "
    "for a mechanistic interpretability study. Replace ONLY the specific concept "
    "category described — nothing else changes. "
    "\n\nCRITICAL FORMATTING RULE: All placeholders MUST use the format CATEGORY_N "
    "where CATEGORY is the uppercase concept type and N is a sequential number. "
    "Examples: ENTITY_1, ENTITY_2, PROPERTY_1, RELATION_1, DOCTRINE_1, API_1. "
    "NEVER use brackets, angle brackets, or any other wrapper around placeholders. "
    "NEVER use lowercase or mixed case in placeholders. "
    "NEVER use subcategory names like DISEASE_1 or DRUG_1 — always use the main "
    "category name (ENTITY_1, not DISEASE_1). "
    "\n\nReturn ONLY valid JSON — no markdown, no commentary, no backticks."
)

ADVERSARIAL_DEV = (
    "You are an expert research assistant helping design stimuli for a mechanistic "
    "interpretability study. You must return ONLY valid JSON — no markdown, no "
    "commentary, no backticks."
)

# --- Medical surgical prompts ---
MED_SURGICAL = {
"entity": """Replace ONLY named medical entities. Leave ALL other text EXACTLY as written.
Replace: disease/condition names, drug names, anatomical structures, pathogen names.
DO NOT change: symptoms, lab values, vitals, demographics, reasoning language, answer options.
Placeholders MUST be: ENTITY_1, ENTITY_2, ENTITY_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_options": {"A":"...","B":"...","C":"...","D":"..."}, "replacements": [{"original":"...","placeholder":"ENTITY_1","category":"entity"}]}

Question: {{question}}
Options: {{options}}""",

"property": """Replace ONLY clinical properties. Leave ALL other text EXACTLY as written.
Replace: symptom descriptions, lab values with numbers, vital signs, physical exam findings.
DO NOT change: disease names, drug names, anatomy, demographics, answer options text.
Placeholders MUST be: PROPERTY_1, PROPERTY_2, PROPERTY_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_options": {"A":"...","B":"...","C":"...","D":"..."}, "replacements": [{"original":"...","placeholder":"PROPERTY_1","category":"property"}]}

Question: {{question}}
Options: {{options}}""",

"relation": """Replace ONLY causal/mechanistic language connecting concepts. Leave concepts themselves EXACTLY as written.
Replace: causal verbs (causes, leads to, inhibits, activates), diagnostic links (indicates, suggests), treatment rationale.
DO NOT change: entity names, property values, demographics, answer options.
Placeholders MUST be: RELATION_1, RELATION_2, RELATION_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_options": {"A":"...","B":"...","C":"...","D":"..."}, "replacements": [{"original":"...","placeholder":"RELATION_1","category":"relation"}]}

Question: {{question}}
Options: {{options}}""",

"procedure": """Replace ONLY clinical reasoning scaffolding. Leave all factual content EXACTLY as written.
Replace: question stems specifying reasoning type ("most appropriate treatment" → "which is relevant"), priority language, differential cues.
DO NOT change: clinical vignette content, answer options, demographics.
Placeholders MUST be: PROCEDURE_1, PROCEDURE_2, PROCEDURE_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_options": {"A":"...","B":"...","C":"...","D":"..."}, "replacements": [{"original":"...","placeholder":"PROCEDURE_1","category":"procedure"}]}

Question: {{question}}
Options: {{options}}""",
}

# --- Legal surgical prompts ---
LEGAL_SURGICAL = {
"doctrine": """Replace ONLY doctrinal legal terms. Leave ALL other text EXACTLY as written.
Replace: legal concepts (hearsay, negligence, fiduciary duty), legal tests/standards, terms of art.
DO NOT change: operative language, structural connectives, factual content, party references.
Placeholders MUST be: DOCTRINE_1, DOCTRINE_2, DOCTRINE_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_answer": "...", "replacements": [{"original":"...","placeholder":"DOCTRINE_1","category":"doctrine"}]}

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}""",

"operative": """Replace ONLY operative legal language. Leave ALL other text EXACTLY as written.
Replace: modal verbs with legal force (shall, may, must), restriction words (solely, exclusively), obligation words.
DO NOT change: doctrinal terms, party names, factual content, structural connectives.
Placeholders MUST be: OPERATIVE_1, OPERATIVE_2, OPERATIVE_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_answer": "...", "replacements": [{"original":"...","placeholder":"OPERATIVE_1","category":"operative"}]}

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}""",

"structural": """Replace ONLY structural/relational connectives. Leave ALL other text EXACTLY as written.
Replace: cross-references, scope delimiters, party role descriptors, temporal structure.
DO NOT change: doctrinal terms, operative language, factual content of clauses.
Placeholders MUST be: STRUCTURAL_1, STRUCTURAL_2, STRUCTURAL_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_answer": "...", "replacements": [{"original":"...","placeholder":"STRUCTURAL_1","category":"structural"}]}

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}""",

"jurisdictional": """Replace ONLY jurisdictional/procedural context. Leave ALL other text EXACTLY as written.
Replace: references to specific laws/rules, court/jurisdiction references, legal authority references.
DO NOT change: doctrinal terms, operative language, factual content, structural connectives.
Placeholders MUST be: JURISDICTIONAL_1, JURISDICTIONAL_2, JURISDICTIONAL_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "modified_answer": "...", "replacements": [{"original":"...","placeholder":"JURISDICTIONAL_1","category":"jurisdictional"}]}

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}""",
}

# --- SWE surgical prompts ---
SWE_SURGICAL = {
"api": """Replace ONLY API/library references. Leave ALL other text EXACTLY as written.
Replace: library/module names, class names, function/method names, import statements.
DO NOT change: algorithmic logic, type info, edge cases, variable names for local logic.
Placeholders MUST be: API_1, API_2, API_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "replacements": [{"original":"...","placeholder":"API_1","category":"api"}]}

Task ID: {{task_id}}
Prompt: {{item_text}}""",

"type": """Replace ONLY type/signature information. Leave ALL other text EXACTLY as written.
Replace: type annotations, return type descriptions, parameter type constraints.
DO NOT change: function names, algorithmic logic, edge cases, API references.
Placeholders MUST be: TYPE_1, TYPE_2, TYPE_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "replacements": [{"original":"...","placeholder":"TYPE_1","category":"type"}]}

Task ID: {{task_id}}
Prompt: {{item_text}}""",

"logic": """Replace ONLY algorithmic/computational logic descriptions. Leave ALL other text EXACTLY as written.
Replace: algorithmic step descriptions, comparison operators in words, mathematical operations, search/traversal descriptions.
DO NOT change: function names, type info, API references, edge cases, examples.
Placeholders MUST be: LOGIC_1, LOGIC_2, LOGIC_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "replacements": [{"original":"...","placeholder":"LOGIC_1","category":"logic"}]}

Task ID: {{task_id}}
Prompt: {{item_text}}""",

"edgecase": """Replace ONLY edge case and boundary specifications. Leave ALL other text EXACTLY as written.
Replace: default/fallback return values, empty input handling, boundary conditions, error specs.
DO NOT change: function names, type info, API references, algorithmic logic, main examples.
Placeholders MUST be: EDGECASE_1, EDGECASE_2, EDGECASE_3, etc. No brackets, no lowercase.
Return JSON: {"modified_text": "...", "replacements": [{"original":"...","placeholder":"EDGECASE_1","category":"edgecase"}]}

Task ID: {{task_id}}
Prompt: {{item_text}}""",
}

# --- Adversarial prompts ---
ADV_PROMPTS = {
"medical": """Create an ADVERSARIAL version: change 1-2 medical facts to redirect toward a WRONG answer.
Keep structure, style, difficulty identical. Explain WHY each change redirects reasoning.
Return JSON: {"adversarial_text": "...", "adversarial_options": {"A":"...","B":"...","C":"...","D":"..."}, "original_correct_idx": "...", "adversarial_lure_idx": "...", "modifications": ["..."]}

Question: {{question}}
Options: {{options}}
Answer: {{answer}} ({{answer_idx}})""",

"legal": """Create an ADVERSARIAL version: subtly modify text to FLIP the correct answer.
Change wording to negate meaning, introduce qualifiers, or swap legal standards.
Return JSON: {"adversarial_text": "...", "adversarial_answer": "...", "original_answer": "...", "modifications": ["..."]}

Task: {{task_name}}
Text: {{item_text}}
Answer: {{answer}}""",

"swe": """Create an ADVERSARIAL version: introduce subtle bugs or misleading elements.
Swap comparison operators, change edge cases, alter expected returns for boundaries.
Return JSON: {"adversarial_text": "...", "modifications": ["..."]}

Task ID: {{task_id}}
Prompt: {{item_text}}""",
}

DOMAIN_SURGICAL = {
    "medical": MED_SURGICAL,
    "legal": LEGAL_SURGICAL,
    "swe": SWE_SURGICAL,
}


# ===================================================================
# STEP 1: GENERATE
# ===================================================================
def generate_all():
    print("\n" + "=" * 60)
    print("GENERATING SURGICAL + ADVERSARIAL")
    print("=" * 60)

    for domain in DOMAINS:
        df = get_df(domain)
        cats = CATEGORIES[domain]
        surg_prompts = DOMAIN_SURGICAL[domain]
        adv_prompt = ADV_PROMPTS[domain]

        surg_path = SURGICAL_DIR / f"{domain}_surgical.jsonl"
        adv_path = SURGICAL_DIR / f"{domain}_adversarial.jsonl"
        write_lock = threading.Lock()

        # Check done
        done_surg = set()
        if surg_path.exists():
            with open(surg_path) as f:
                for line in f:
                    r = json.loads(line)
                    done_surg.add((r["item_idx"], r["category"]))

        done_adv = set()
        if adv_path.exists():
            with open(adv_path) as f:
                for line in f:
                    r = json.loads(line)
                    done_adv.add(r["item_idx"])

        # Build tasks
        tasks = []
        for i, (_, row) in enumerate(df.iterrows()):
            for cat in cats:
                if (i, cat) not in done_surg:
                    tasks.append(("surg", i, cat, surg_prompts[cat], row, domain))
            if i not in done_adv:
                tasks.append(("adv", i, None, adv_prompt, row, domain))

        print(f"\n--- {domain.upper()}: {len(df)} items, {len(tasks)} tasks remaining ---")
        if not tasks:
            continue

        completed = {"n": 0, "ok": 0}
        total = len(tasks)

        def normalize_placeholders(result, category):
            """Normalize placeholder format to CATEGORY_N regardless of what the LLM returned."""
            if not result or "replacements" not in result:
                return result
            import re
            cat_upper = category.upper()
            text = result.get("modified_text", "")
            options = result.get("modified_options", result.get("modified_answer", None))
            counter = 1
            for repl in result["replacements"]:
                old_ph = repl["placeholder"]
                new_ph = f"{cat_upper}_{counter}"
                if old_ph != new_ph:
                    text = text.replace(old_ph, new_ph)
                    if isinstance(options, dict):
                        for k in options:
                            options[k] = options[k].replace(old_ph, new_ph)
                    repl["placeholder"] = new_ph
                counter += 1
            result["modified_text"] = text
            if isinstance(options, dict):
                if "modified_options" in result:
                    result["modified_options"] = options
            return result

        def process(t):
            ttype, idx, cat, tmpl, row, dom = t
            prompt = fill_template(tmpl, row)
            dev_msg = SURGICAL_DEV if ttype == "surg" else ADVERSARIAL_DEV
            result = call_api(prompt, dev_msg)

            if ttype == "surg" and result is not None:
                result = normalize_placeholders(result, cat)

            if ttype == "surg":
                rec = {"item_idx": idx, "category": cat, "domain": dom,
                       "stratum": row.get("stratum", ""), "success": result is not None,
                       "result": result}
                with write_lock:
                    with open(surg_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            else:
                rec = {"item_idx": idx, "condition": "adversarial", "domain": dom,
                       "stratum": row.get("stratum", ""), "success": result is not None,
                       "result": result}
                with write_lock:
                    with open(adv_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

            with write_lock:
                completed["n"] += 1
                if rec["success"]: completed["ok"] += 1
                if completed["n"] % 100 == 0:
                    pct = completed["n"] / total * 100
                    print(f"  [{completed['n']}/{total}] ({pct:.0f}%) {completed['ok']} ok | {domain}")

        with ThreadPoolExecutor(max_workers=API_WORKERS) as ex:
            futures = [ex.submit(process, t) for t in tasks]
            for f in as_completed(futures):
                try: f.result()
                except Exception as e: print(f"  Error: {e}")

        print(f"  {domain.upper()} done: {completed['ok']}/{total}")


# ===================================================================
# STEP 2: COMPOSE
# ===================================================================
def compose_all():
    print("\n" + "=" * 60)
    print("COMPOSING COMBINATIONS")
    print("=" * 60)

    for domain in DOMAINS:
        cats = CATEGORIES[domain]
        surg_path = SURGICAL_DIR / f"{domain}_surgical.jsonl"
        if not surg_path.exists():
            print(f"  Skipping {domain}")
            continue

        df = get_df(domain)
        surgical = {}
        with open(surg_path) as f:
            for line in f:
                r = json.loads(line)
                if not r["success"]: continue
                idx = r["item_idx"]
                if idx not in surgical: surgical[idx] = {}
                surgical[idx][r["category"]] = r["result"]

        combos = []
        for r in range(2, len(cats) + 1):
            for c in combinations(cats, r):
                combos.append(c)

        composed_path = SURGICAL_DIR / f"{domain}_composed.jsonl"
        count = 0
        with open(composed_path, "w") as f:
            for item_idx, cat_results in surgical.items():
                if item_idx >= len(df): continue
                row = df.iloc[item_idx]
                orig_text = row.get("item_text", row.get("question", ""))
                orig_options = row.get("options", {})

                # Guard against NaN / non-string rows (string ops require str)
                if not isinstance(orig_text, str) or not orig_text.strip():
                    continue
                if not isinstance(orig_options, dict):
                    orig_options = {}
                orig_text = orig_text.strip()

                for combo in combos:
                    if not all(c in cat_results for c in combo):
                        continue
                    text = orig_text
                    options = dict(orig_options) if isinstance(orig_options, dict) else {}
                    for c in combo:
                        for repl in cat_results[c].get("replacements", []):
                            orig = repl.get("original")
                            placeholder = repl.get("placeholder")
                            if not isinstance(orig, str) or not isinstance(placeholder, str):
                                continue
                            text = text.replace(orig, placeholder)
                        if "modified_options" in cat_results[c] and orig_options:
                            for k, v in cat_results[c]["modified_options"].items():
                                if v != orig_options.get(k, ""):
                                    options[k] = v
                    rec = {"item_idx": item_idx, "domain": domain,
                           "stratum": row.get("stratum", ""),
                           "combination_name": "+".join(combo),
                           "composed_text": text, "composed_options": options}
                    f.write(json.dumps(rec) + "\n")
                    count += 1

        print(f"  {domain.upper()}: {count} composed conditions from {len(surgical)} items")


# ===================================================================
# STEP 3: EXTRACT ACTIVATIONS
# ===================================================================
def load_model():
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model.eval()
    nl = model.config.num_hidden_layers
    hd = model.config.hidden_size
    print(f"Loaded: {nl} layers, {hd} dim, device={DEVICE}")
    return model, tokenizer, nl, hd


def extract_acts(model, tokenizer, text, n_layers, key_layers_set):
    """Run forward pass, return last-token (all layers) + all-token (key layers)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = inputs.input_ids.to(DEVICE)
    seq_len = input_ids.shape[1]

    last_tok = {}
    all_tok = {}
    hooks = []

    def make_hook(li):
        def fn(mod, inp, out):
            h = out[0]
            if h.dim() == 2: h = h.unsqueeze(0)
            hc = h[0].detach().cpu().to(torch.float16)
            last_tok[li] = hc[-1].numpy()
            if li in key_layers_set:
                all_tok[li] = hc.numpy()
        return fn

    for i in range(n_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks: h.remove()

    lt = np.stack([last_tok[i] for i in range(n_layers)])  # (n_layers, hidden)
    return lt, all_tok, seq_len


def build_conditions(domain):
    """Build all condition texts. Returns (conditions_dict, originals_df)."""
    df = get_df(domain)
    cats = CATEGORIES[domain]

    conds = {"original": {}}
    for i, (_, row) in enumerate(df.iterrows()):
        t = row.get("item_text", row.get("question", ""))
        if isinstance(t, str) and t.strip():
            conds["original"][i] = t.strip()

    # Single surgical
    sp = SURGICAL_DIR / f"{domain}_surgical.jsonl"
    if sp.exists():
        with open(sp) as f:
            for line in f:
                r = json.loads(line)
                if not r["success"]: continue
                cn = f"abstract_{r['category']}"
                if cn not in conds: conds[cn] = {}
                mt = (r.get("result") or {}).get("modified_text", "")
                if isinstance(mt, str) and mt.strip():
                    conds[cn][r["item_idx"]] = mt.strip()

    # Composed
    cp = SURGICAL_DIR / f"{domain}_composed.jsonl"
    if cp.exists():
        with open(cp) as f:
            for line in f:
                r = json.loads(line)
                cn = f"abstract_{r['combination_name']}"
                if cn not in conds: conds[cn] = {}
                ct = r.get("composed_text", "")
                if isinstance(ct, str) and ct.strip():
                    conds[cn][r["item_idx"]] = ct.strip()

    # Adversarial
    ap = SURGICAL_DIR / f"{domain}_adversarial.jsonl"
    if ap.exists():
        with open(ap) as f:
            for line in f:
                r = json.loads(line)
                if r.get("success"):
                    if "adversarial" not in conds: conds["adversarial"] = {}
                    at = (r.get("result") or {}).get("adversarial_text", "")
                    if isinstance(at, str) and at.strip():
                        conds["adversarial"][r["item_idx"]] = at.strip()

    return conds, df


def extract_domain(domain, model, tokenizer, n_layers, hidden_dim):
    print(f"\n{'='*60}")
    print(f"EXTRACTING: {domain.upper()}")
    print("=" * 60)

    conds, df = build_conditions(domain)
    cats = CATEGORIES[domain]
    key_conds = ["original"] + [f"abstract_{c}" for c in cats] + ["adversarial"]
    key_conds = [c for c in key_conds if c in conds]
    all_cond_names = sorted(conds.keys())
    key_layers_set = set(KEY_LAYERS)

    # Items with all conditions
    valid = set(conds["original"].keys())
    for cn in all_cond_names:
        valid &= set(conds[cn].keys())
    valid = sorted(valid)

    # Items with key conditions (for all-token)
    key_valid = set(conds["original"].keys())
    for cn in key_conds:
        key_valid &= set(conds[cn].keys())
    key_valid = sorted(key_valid)

    n_items = len(valid)
    n_conds = len(all_cond_names)
    print(f"  All conditions: {n_conds} | Items with all: {n_items}")
    print(f"  Key conditions: {len(key_conds)} | Items with key: {len(key_valid)}")
    for cn in all_cond_names:
        print(f"    {cn:50s} | {len(conds[cn])} items")

    # ---- Phase 1: Last-token (all layers, all conditions) ----
    print(f"\n  Phase 1: Last-token ({n_items} × {n_conds} = {n_items * n_conds} fwd)")
    lt_acts = np.zeros((n_items, n_conds, n_layers, hidden_dim), dtype=np.float16)
    metadata = []

    t0 = time.time()
    fwd = 0
    total_fwd = n_items * n_conds

    for i, item_idx in enumerate(valid):
        row = df.iloc[item_idx]
        metadata.append({"item_idx": item_idx, "stratum": row.get("stratum", "")})

        for j, cn in enumerate(all_cond_names):
            text = conds[cn].get(item_idx, "")
            if not text: continue
            lt, _, _ = extract_acts(model, tokenizer, text, n_layers, set())
            lt_acts[i, j] = lt
            fwd += 1

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = fwd / elapsed if elapsed > 0 else 0
            eta = (total_fwd - fwd) / rate / 60 if rate > 0 else 0
            print(f"    [{i+1}/{n_items}] {fwd}/{total_fwd} | {rate:.1f} fps | ETA: {eta:.0f} min")

        if DEVICE.type == "mps" and (i + 1) % 30 == 0:
            torch.mps.empty_cache()

    lt_path = ACTIVATIONS_DIR / f"{domain}_last_token.npz"
    np.savez_compressed(lt_path, activations=lt_acts,
        condition_names=np.array(all_cond_names),
        item_indices=np.array([m["item_idx"] for m in metadata]),
        strata=np.array([m["stratum"] for m in metadata]))
    print(f"    Saved: {lt_path} ({lt_path.stat().st_size / 1e6:.0f} MB)")

    # ---- Phase 2: All-token (key layers, key conditions) ----
    n_key = len(key_valid)
    n_key_c = len(key_conds)
    print(f"\n  Phase 2: All-token ({n_key} × {n_key_c} = {n_key * n_key_c} fwd)")
    print(f"    Key layers: {KEY_LAYERS}")
    print(f"    Key conditions: {key_conds}")

    # Save per-item .npz files to avoid loading everything into RAM
    t1 = time.time()
    fwd2 = 0
    total_fwd2 = n_key * n_key_c

    item_dir = ALL_TOKEN_DIR / domain
    item_dir.mkdir(exist_ok=True)

    for i, item_idx in enumerate(key_valid):
        item_acts = {}  # cond_name -> {layer -> (seq_len, hidden_dim)}

        for cn in key_conds:
            text = conds[cn].get(item_idx, "")
            if not text: continue
            _, at, sl = extract_acts(model, tokenizer, text, n_layers, key_layers_set)
            item_acts[cn] = at
            fwd2 += 1

        # Save this item's all-token activations
        save_dict = {"item_idx": item_idx, "stratum": df.iloc[item_idx].get("stratum", "")}
        for cn, layer_dict in item_acts.items():
            for layer, acts in layer_dict.items():
                save_dict[f"{cn}_L{layer}"] = acts  # (seq_len, hidden_dim)

        np.savez_compressed(item_dir / f"item_{item_idx:04d}.npz", **save_dict)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t1
            rate = fwd2 / elapsed if elapsed > 0 else 0
            eta = (total_fwd2 - fwd2) / rate / 60 if rate > 0 else 0
            print(f"    [{i+1}/{n_key}] {fwd2}/{total_fwd2} | {rate:.1f} fps | ETA: {eta:.0f} min")

        if DEVICE.type == "mps" and (i + 1) % 15 == 0:
            torch.mps.empty_cache()

    total_size = sum(f.stat().st_size for f in item_dir.glob("*.npz")) / 1e9
    print(f"    Saved: {item_dir}/ ({len(list(item_dir.glob('*.npz')))} files, {total_size:.1f} GB)")
    print(f"    Total time for {domain}: {(time.time() - t0)/60:.1f} min")


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.generate = args.compose = args.extract = True

    t_global = time.time()

    if args.generate:
        generate_all()

    if args.compose:
        compose_all()

    if args.extract:
        model, tokenizer, nl, hd = load_model()
        for d in DOMAINS:
            extract_domain(d, model, tokenizer, nl, hd)

    hrs = (time.time() - t_global) / 3600
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {hrs:.1f} hours")
    print("=" * 60)