#!/usr/bin/env python3
"""
Audit the MedQA dataset for demographic attribute mentions.

Scans every question in GBaker/MedQA-USMLE-4-options for mentions of:
  - Sex/gender cues (man, woman, male, female, boy, girl, etc.)
  - Age mentions (with extraction of numeric age)
  - Race/ethnicity (White, Black, Hispanic, Asian, Native American, etc.)
  - Sexual orientation cues (gay, lesbian, bisexual, husband, wife, partner, etc.)
  - Gender identity cues (transgender, nonbinary, etc.)
  - Insurance status cues (uninsured, Medicaid, Medicare, private insurance, etc.)
  - Housing status cues (homeless, shelter, etc.)
  - Occupation cues (teacher, construction worker, etc.)
  - Pregnancy/reproductive cues
  - Disability cues
  - Religion cues
  - Marital/family cues (married, divorced, single, etc.)
  - Nationality/immigration cues

Outputs:
  - Console summary of counts per attribute
  - CSV with per-question attribute flags
  - Detailed breakdown suitable for planning counterfactual generation

Usage:
  python audit_medqa_demographics.py --split train --output_dir ./medqa_audit
  python audit_medqa_demographics.py --split train --output_dir ./medqa_audit --all_splits
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from datasets import load_dataset

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Sex / gender (captures the actual term matched)
SEX_GENDER_PATTERNS = {
    "male": r'\b(\d+[ -]?year[ -]?old\s+)?(male|man|boy|gentleman)\b',
    "female": r'\b(\d+[ -]?year[ -]?old\s+)?(female|woman|girl|lady)\b',
    "he_him": r'\b(he|him|his)\b',
    "she_her": r'\b(she|her|hers)\b',
}

# More specific sex/gender (to distinguish from pronoun-only)
SEX_EXPLICIT = {
    "male_explicit": r'\b(male|man|boy|gentleman)\b',
    "female_explicit": r'\b(female|woman|girl|lady)\b',
}

# Age
AGE_PATTERN = r'\b(\d{1,3})[ -]?(year|yr|y/?o|month|mo)[ -]?(old|s)?\b'
AGE_GROUP_TERMS = {
    "infant": r'\b(infant|newborn|neonate|baby)\b',
    "child": r'\b(child|pediatric|toddler|preschool)\b',
    "adolescent": r'\b(adolescent|teenager|teen|puberty)\b',
    "elderly": r'\b(elderly|geriatric|older adult|senior citizen)\b',
}

# Race / ethnicity
RACE_PATTERNS = {
    "White/Caucasian": r'\b(white|caucasian|european[ -]?american)\b',
    "Black/African American": r'\b(black|african[ -]?american|african american)\b',
    "Hispanic/Latino": r'\b(hispanic|latino|latina|latinx|mexican[ -]?american|puerto rican|cuban)\b',
    "Asian": r'\b(asian|chinese|japanese|korean|vietnamese|filipino|indian|south asian|east asian|southeast asian)\b',
    "Native American": r'\b(native american|american indian|indigenous|alaska native|first nations)\b',
    "Pacific Islander": r'\b(pacific islander|hawaiian|samoan|polynesian)\b',
    "Middle Eastern": r'\b(middle eastern|arab|persian|iranian|iraqi|syrian|lebanese|turkish)\b',
    "Ashkenazi Jewish": r'\b(ashkenazi|jewish)\b',
    "Amish": r'\b(amish)\b',
}

# Sexual orientation
SEXUAL_ORIENTATION_PATTERNS = {
    "gay": r'\b(gay|homosexual)\b',
    "lesbian": r'\b(lesbian)\b',
    "bisexual": r'\b(bisexual)\b',
    "heterosexual": r'\b(heterosexual|straight)\b',
    "partner_same_sex_cue": r'\b(his (husband|boyfriend|male partner)|her (wife|girlfriend|female partner))\b',
    "relationship_cue": r'\b(husband|wife|spouse|partner|boyfriend|girlfriend|significant other|fiancé|fiancée)\b',
}

# Gender identity
GENDER_IDENTITY_PATTERNS = {
    "transgender": r'\b(transgender|trans man|trans woman|transman|transwoman|trans male|trans female)\b',
    "nonbinary": r'\b(nonbinary|non-binary|genderqueer|gender[ -]?fluid|agender)\b',
    "hormone_therapy_cue": r'\b(hormone replacement therapy|HRT|testosterone therapy|estrogen therapy|gender[ -]?affirming)\b',
}

# Insurance
INSURANCE_PATTERNS = {
    "uninsured": r'\b(uninsured|no insurance|lacks insurance|without insurance|no health insurance)\b',
    "Medicaid": r'\b(medicaid)\b',
    "Medicare": r'\b(medicare)\b',
    "private_insurance": r'\b(private insurance|privately insured|employer[ -]?sponsored|PPO|HMO|Blue Cross|Aetna|Cigna|UnitedHealth)\b',
    "insurance_general": r'\b(insurance|insured|coverage)\b',
}

# Housing
HOUSING_PATTERNS = {
    "homeless": r'\b(homeless|unhoused|living on the street|shelter|no fixed address|lives in a car)\b',
    "group_home": r'\b(group home|nursing home|assisted living|long[ -]?term care|skilled nursing)\b',
    "incarcerated": r'\b(incarcerated|prison|jail|correctional|inmate|detained)\b',
}

# Pregnancy / reproductive
PREGNANCY_PATTERNS = {
    "pregnant": r'\b(pregnant|pregnancy|gravid|gestation|trimester|prenatal|antenatal|postpartum|peripartum)\b',
    "reproductive": r'\b(menopause|menopausal|amenorrhea|menstrual|gynecologic|obstetric|contracepti)\b',
    "gravida_para": r'\b(G\d+P\d+|gravida|para |nulliparous|multiparous|primigravida|multigravida)\b',
}

# Disability
DISABILITY_PATTERNS = {
    "wheelchair": r'\b(wheelchair|paralyzed|paraplegic|quadriplegic|paraplegia|quadriplegia)\b',
    "blind_deaf": r'\b(blind|deaf|hearing impaired|visually impaired|hearing loss)\b',
    "intellectual": r'\b(intellectual disability|down syndrome|developmental delay|autism|autistic)\b',
    "mental_health": r'\b(schizophren|bipolar|major depress|psychiatric history|psych ward)\b',
}

# Religion
RELIGION_PATTERNS = {
    "Christian": r'\b(christian|catholic|protestant|evangelical|baptist|methodist|lutheran|presbyterian|church)\b',
    "Muslim": r'\b(muslim|islamic|islam|mosque|ramadan|halal)\b',
    "Jewish_religious": r'\b(jewish|judaism|synagogue|rabbi|kosher|sabbath|orthodox jewish)\b',
    "Hindu": r'\b(hindu|hinduism|temple)\b',
    "Buddhist": r'\b(buddhist|buddhism)\b',
    "Jehovah_Witness": r"\b(jehovah'?s? witness)\b",
    "religious_general": r'\b(religious|spiritual|prayer|faith|church|temple|mosque)\b',
}

# Marital / family
MARITAL_PATTERNS = {
    "married": r'\b(married|marriage)\b',
    "divorced": r'\b(divorced|divorce|separation)\b',
    "widowed": r'\b(widowed|widow|widower)\b',
    "single": r'\b(single|unmarried|never married)\b',
    "children_mentioned": r'\b(children|kids|son|daughter|child|mother of|father of)\b',
}

# Occupation (select high-signal ones)
OCCUPATION_PATTERNS = {
    "healthcare_worker": r'\b(nurse|physician|doctor|paramedic|EMT|pharmacist|dentist|surgeon|resident|intern)\b',
    "manual_labor": r'\b(construction worker|farmer|factory worker|miner|welder|plumber|electrician|mechanic|laborer)\b',
    "military": r'\b(military|veteran|army|navy|marine|air force|soldier|deployed|combat)\b',
    "sex_worker": r'\b(sex worker|prostitut|commercial sex)\b',
    "office_worker": r'\b(office worker|accountant|lawyer|teacher|professor|engineer|programmer|executive)\b',
    "student": r'\b(student|college|university|medical student|nursing student)\b',
    "unemployed": r'\b(unemployed|jobless|out of work)\b',
}

# Nationality / immigration
NATIONALITY_PATTERNS = {
    "immigrant": r'\b(immigrant|emigrated|migrated|moved from|recently arrived from|refugee|asylum)\b',
    "country_of_origin": r'\b(from (Mexico|China|India|Nigeria|Haiti|Honduras|Guatemala|El Salvador|Brazil|Colombia|Philippines|Vietnam|Korea|Japan|Somalia|Ethiopia|Sudan|Syria|Iraq|Iran|Afghanistan|Pakistan))\b',
    "travel_history": r'\b(recently traveled to|returned from|travel history|visiting from)\b',
}

# Substance use (bonus — relevant to bias)
SUBSTANCE_PATTERNS = {
    "alcohol": r'\b(alcohol|drinks?( per| a) (day|week|night)|beer|wine|liquor|ethanol|ETOH)\b',
    "tobacco": r'\b(smok(es?|ing|er)|tobacco|cigarette|pack[ -]?year|nicotine)\b',
    "IV_drugs": r'\b(IV drug|intravenous drug|injection drug|IVDU|heroin|needle sharing)\b',
    "marijuana": r'\b(marijuana|cannabis|THC|weed)\b',
    "cocaine": r'\b(cocaine|crack)\b',
}


def scan_question(text: str) -> dict:
    """Scan a question for all demographic attributes. Returns dict of findings."""
    text_lower = text.lower()
    results = {}

    # Sex/gender
    sex_found = []
    for label, pat in SEX_EXPLICIT.items():
        if re.search(pat, text_lower):
            sex_found.append(label.replace("_explicit", ""))
    results["sex_explicit"] = sex_found if sex_found else None
    results["has_sex_cue"] = bool(sex_found)

    # Age
    age_matches = re.findall(AGE_PATTERN, text_lower)
    if age_matches:
        ages = []
        for m in age_matches:
            try:
                val = int(m[0])
                unit = m[1].lower()
                if "month" in unit or "mo" in unit:
                    ages.append(val / 12.0)
                else:
                    ages.append(val)
            except (ValueError, IndexError):
                pass
        results["ages_found"] = ages
        results["has_age"] = True
        if ages:
            primary_age = ages[0]
            if primary_age < 1:
                results["age_group"] = "infant"
            elif primary_age < 13:
                results["age_group"] = "child"
            elif primary_age < 18:
                results["age_group"] = "adolescent"
            elif primary_age < 30:
                results["age_group"] = "young_adult"
            elif primary_age < 50:
                results["age_group"] = "middle_aged"
            elif primary_age < 65:
                results["age_group"] = "older_adult"
            else:
                results["age_group"] = "elderly"
    else:
        results["ages_found"] = []
        results["has_age"] = False
        results["age_group"] = None

    # Age group terms
    for label, pat in AGE_GROUP_TERMS.items():
        if re.search(pat, text_lower):
            if results["age_group"] is None:
                results["age_group"] = label
            results["has_age"] = True

    # Race/ethnicity
    races_found = []
    for label, pat in RACE_PATTERNS.items():
        if re.search(pat, text_lower):
            races_found.append(label)
    results["races_found"] = races_found
    results["has_race"] = bool(races_found)

    # Sexual orientation
    so_found = []
    for label, pat in SEXUAL_ORIENTATION_PATTERNS.items():
        if re.search(pat, text_lower):
            so_found.append(label)
    results["sexual_orientation_cues"] = so_found
    results["has_sexual_orientation_cue"] = bool(so_found)
    results["has_relationship_cue"] = "relationship_cue" in so_found

    # Gender identity
    gi_found = []
    for label, pat in GENDER_IDENTITY_PATTERNS.items():
        if re.search(pat, text_lower):
            gi_found.append(label)
    results["gender_identity_cues"] = gi_found
    results["has_gender_identity_cue"] = bool(gi_found)

    # Insurance
    ins_found = []
    for label, pat in INSURANCE_PATTERNS.items():
        if re.search(pat, text_lower):
            ins_found.append(label)
    results["insurance_cues"] = ins_found
    results["has_insurance_cue"] = bool(ins_found)

    # Housing
    housing_found = []
    for label, pat in HOUSING_PATTERNS.items():
        if re.search(pat, text_lower):
            housing_found.append(label)
    results["housing_cues"] = housing_found
    results["has_housing_cue"] = bool(housing_found)

    # Pregnancy / reproductive
    preg_found = []
    for label, pat in PREGNANCY_PATTERNS.items():
        if re.search(pat, text_lower):
            preg_found.append(label)
    results["pregnancy_cues"] = preg_found
    results["has_pregnancy_cue"] = bool(preg_found)

    # Disability
    dis_found = []
    for label, pat in DISABILITY_PATTERNS.items():
        if re.search(pat, text_lower):
            dis_found.append(label)
    results["disability_cues"] = dis_found
    results["has_disability_cue"] = bool(dis_found)

    # Religion
    rel_found = []
    for label, pat in RELIGION_PATTERNS.items():
        if re.search(pat, text_lower):
            rel_found.append(label)
    results["religion_cues"] = rel_found
    results["has_religion_cue"] = bool(rel_found)

    # Marital / family
    mar_found = []
    for label, pat in MARITAL_PATTERNS.items():
        if re.search(pat, text_lower):
            mar_found.append(label)
    results["marital_cues"] = mar_found
    results["has_marital_cue"] = bool(mar_found)

    # Occupation
    occ_found = []
    for label, pat in OCCUPATION_PATTERNS.items():
        if re.search(pat, text_lower):
            occ_found.append(label)
    results["occupation_cues"] = occ_found
    results["has_occupation_cue"] = bool(occ_found)

    # Nationality / immigration
    nat_found = []
    for label, pat in NATIONALITY_PATTERNS.items():
        if re.search(pat, text_lower):
            nat_found.append(label)
    results["nationality_cues"] = nat_found
    results["has_nationality_cue"] = bool(nat_found)

    # Substance use
    sub_found = []
    for label, pat in SUBSTANCE_PATTERNS.items():
        if re.search(pat, text_lower):
            sub_found.append(label)
    results["substance_cues"] = sub_found
    results["has_substance_cue"] = bool(sub_found)

    return results


def run_audit(split: str, output_dir: str, all_splits: bool = False):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    splits_to_scan = [split]
    if all_splits:
        splits_to_scan = ["train", "test"]

    all_rows = []
    for sp in splits_to_scan:
        print(f"\nLoading MedQA split: {sp}...")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=sp)
        print(f"  {len(ds)} questions")

        for idx, record in enumerate(ds):
            question = record["question"]
            findings = scan_question(question)
            findings["split"] = sp
            findings["idx"] = idx
            findings["question_id"] = f"medqa_{sp}_{idx}"
            findings["question_length"] = len(question)
            findings["question_text"] = question[:200]  # truncated for CSV
            all_rows.append(findings)

    print(f"\nTotal questions scanned: {len(all_rows)}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DEMOGRAPHIC ATTRIBUTE AUDIT — MedQA")
    print(f"{'='*70}")

    n_total = len(all_rows)

    # High-level presence counts
    attributes = [
        ("has_sex_cue", "Sex/Gender (explicit term)"),
        ("has_age", "Age mention"),
        ("has_race", "Race/Ethnicity"),
        ("has_sexual_orientation_cue", "Sexual Orientation (any cue)"),
        ("has_relationship_cue", "Relationship mention (husband/wife/partner)"),
        ("has_gender_identity_cue", "Gender Identity"),
        ("has_insurance_cue", "Insurance Status"),
        ("has_housing_cue", "Housing Status"),
        ("has_pregnancy_cue", "Pregnancy/Reproductive"),
        ("has_disability_cue", "Disability"),
        ("has_religion_cue", "Religion"),
        ("has_marital_cue", "Marital/Family Status"),
        ("has_occupation_cue", "Occupation"),
        ("has_nationality_cue", "Nationality/Immigration"),
        ("has_substance_cue", "Substance Use"),
    ]

    print(f"\n  {'Attribute':<45} {'Count':>6} {'%':>7}")
    print(f"  {'-'*60}")
    for key, label in attributes:
        count = sum(1 for r in all_rows if r.get(key))
        print(f"  {label:<45} {count:>6} {count/n_total:>7.1%}")

    # Detailed breakdowns
    print(f"\n{'='*70}")
    print("DETAILED BREAKDOWNS")
    print(f"{'='*70}")

    # Race/ethnicity
    print(f"\n  Race/Ethnicity mentions:")
    race_counter = Counter()
    for r in all_rows:
        for race in r.get("races_found", []):
            race_counter[race] += 1
    for race, count in race_counter.most_common():
        print(f"    {race:<30} {count:>5} ({count/n_total:.1%})")

    # Age groups
    print(f"\n  Age group distribution:")
    age_counter = Counter(r["age_group"] for r in all_rows if r["age_group"])
    for ag, count in age_counter.most_common():
        print(f"    {ag:<20} {count:>5} ({count/n_total:.1%})")

    # Sex
    print(f"\n  Sex/gender explicit mentions:")
    sex_counter = Counter()
    for r in all_rows:
        for s in (r.get("sex_explicit") or []):
            sex_counter[s] += 1
    for s, count in sex_counter.most_common():
        print(f"    {s:<20} {count:>5} ({count/n_total:.1%})")

    # Sexual orientation
    print(f"\n  Sexual orientation cues:")
    so_counter = Counter()
    for r in all_rows:
        for cue in r.get("sexual_orientation_cues", []):
            so_counter[cue] += 1
    for cue, count in so_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Insurance
    print(f"\n  Insurance mentions:")
    ins_counter = Counter()
    for r in all_rows:
        for cue in r.get("insurance_cues", []):
            ins_counter[cue] += 1
    for cue, count in ins_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Housing
    print(f"\n  Housing mentions:")
    housing_counter = Counter()
    for r in all_rows:
        for cue in r.get("housing_cues", []):
            housing_counter[cue] += 1
    for cue, count in housing_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Pregnancy
    print(f"\n  Pregnancy/reproductive mentions:")
    preg_counter = Counter()
    for r in all_rows:
        for cue in r.get("pregnancy_cues", []):
            preg_counter[cue] += 1
    for cue, count in preg_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Religion
    print(f"\n  Religion mentions:")
    rel_counter = Counter()
    for r in all_rows:
        for cue in r.get("religion_cues", []):
            rel_counter[cue] += 1
    for cue, count in rel_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Marital
    print(f"\n  Marital/family mentions:")
    mar_counter = Counter()
    for r in all_rows:
        for cue in r.get("marital_cues", []):
            mar_counter[cue] += 1
    for cue, count in mar_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Occupation
    print(f"\n  Occupation mentions:")
    occ_counter = Counter()
    for r in all_rows:
        for cue in r.get("occupation_cues", []):
            occ_counter[cue] += 1
    for cue, count in occ_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Nationality
    print(f"\n  Nationality/immigration mentions:")
    nat_counter = Counter()
    for r in all_rows:
        for cue in r.get("nationality_cues", []):
            nat_counter[cue] += 1
    for cue, count in nat_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Disability
    print(f"\n  Disability mentions:")
    dis_counter = Counter()
    for r in all_rows:
        for cue in r.get("disability_cues", []):
            dis_counter[cue] += 1
    for cue, count in dis_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # Substance use
    print(f"\n  Substance use mentions:")
    sub_counter = Counter()
    for r in all_rows:
        for cue in r.get("substance_cues", []):
            sub_counter[cue] += 1
    for cue, count in sub_counter.most_common():
        print(f"    {cue:<30} {count:>5} ({count/n_total:.1%})")

    # -----------------------------------------------------------------------
    # Co-occurrence analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CO-OCCURRENCE: Questions with MULTIPLE demographic attributes")
    print(f"{'='*70}")

    bool_keys = [k for k, _ in attributes]
    for r in all_rows:
        r["n_attributes"] = sum(1 for k in bool_keys if r.get(k))

    attr_count_dist = Counter(r["n_attributes"] for r in all_rows)
    for n_attr in sorted(attr_count_dist.keys()):
        count = attr_count_dist[n_attr]
        print(f"  {n_attr} attributes: {count:>5} ({count/n_total:.1%})")

    # Questions with race + other attributes
    race_questions = [r for r in all_rows if r["has_race"]]
    if race_questions:
        print(f"\n  Among questions WITH race mention (n={len(race_questions)}):")
        for key, label in attributes:
            if key == "has_race":
                continue
            count = sum(1 for r in race_questions if r.get(key))
            print(f"    + {label:<40} {count:>5} ({count/len(race_questions):.1%})")

    # -----------------------------------------------------------------------
    # Swappability analysis — how many questions are good candidates?
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SWAPPABILITY ANALYSIS — Counterfactual Generation Potential")
    print(f"{'='*70}")

    # Race: can swap if race is mentioned
    print(f"\n  Race/ethnicity swappable (has explicit race mention):")
    for race, count in race_counter.most_common():
        print(f"    {race:<30} {count:>5} — can generate CF by swapping to other races")

    # Sex: can swap if explicit sex term present
    n_sex = sum(1 for r in all_rows if r["has_sex_cue"])
    print(f"\n  Sex swappable (explicit man/woman/male/female): {n_sex}")

    # Relationship cues for sexual orientation
    n_rel = sum(1 for r in all_rows if r["has_relationship_cue"])
    print(f"  Has relationship cue (husband/wife/partner — SO-swappable): {n_rel}")

    # Questions with NO demographic info (hardest to generate CF for)
    n_no_demo = sum(1 for r in all_rows if r["n_attributes"] == 0)
    print(f"\n  Questions with NO demographic attributes detected: {n_no_demo} ({n_no_demo/n_total:.1%})")

    # Questions mentioning only age + sex (very common clinical vignette pattern)
    n_age_sex_only = sum(1 for r in all_rows
                         if r["has_age"] and r["has_sex_cue"]
                         and not r["has_race"] and not r["has_insurance_cue"]
                         and not r["has_housing_cue"] and not r["has_religion_cue"])
    print(f"  Questions with ONLY age + sex (no other demographics): {n_age_sex_only} ({n_age_sex_only/n_total:.1%})")

    # -----------------------------------------------------------------------
    # Per-split breakdown
    # -----------------------------------------------------------------------
    if all_splits and len(splits_to_scan) > 1:
        print(f"\n{'='*70}")
        print("PER-SPLIT BREAKDOWN")
        print(f"{'='*70}")
        for sp in splits_to_scan:
            sp_rows = [r for r in all_rows if r["split"] == sp]
            print(f"\n  {sp} (n={len(sp_rows)}):")
            for key, label in attributes[:6]:  # top 6
                count = sum(1 for r in sp_rows if r.get(key))
                print(f"    {label:<40} {count:>5} ({count/max(len(sp_rows),1):.1%})")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    if HAS_PANDAS:
        # Flatten list fields for CSV
        csv_rows = []
        for r in all_rows:
            flat = {
                "question_id": r["question_id"],
                "split": r["split"],
                "idx": r["idx"],
                "question_length": r["question_length"],
                "question_preview": r["question_text"],
                "n_attributes": r["n_attributes"],
            }
            for key, label in attributes:
                flat[key] = r.get(key, False)

            flat["sex_labels"] = "|".join(r.get("sex_explicit") or [])
            flat["age_group"] = r.get("age_group", "")
            flat["ages"] = "|".join(str(a) for a in r.get("ages_found", []))
            flat["races"] = "|".join(r.get("races_found", []))
            flat["so_cues"] = "|".join(r.get("sexual_orientation_cues", []))
            flat["gi_cues"] = "|".join(r.get("gender_identity_cues", []))
            flat["insurance_cues"] = "|".join(r.get("insurance_cues", []))
            flat["housing_cues"] = "|".join(r.get("housing_cues", []))
            flat["pregnancy_cues"] = "|".join(r.get("pregnancy_cues", []))
            flat["disability_cues"] = "|".join(r.get("disability_cues", []))
            flat["religion_cues"] = "|".join(r.get("religion_cues", []))
            flat["marital_cues"] = "|".join(r.get("marital_cues", []))
            flat["occupation_cues"] = "|".join(r.get("occupation_cues", []))
            flat["nationality_cues"] = "|".join(r.get("nationality_cues", []))
            flat["substance_cues"] = "|".join(r.get("substance_cues", []))
            csv_rows.append(flat)

        df = pd.DataFrame(csv_rows)
        csv_path = outdir / "medqa_demographic_audit.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved per-question audit CSV: {csv_path}")

        # Also save summary
        summary_rows = []
        for key, label in attributes:
            count = sum(1 for r in all_rows if r.get(key))
            summary_rows.append({"attribute": label, "key": key, "count": count,
                                 "pct": count / n_total})
        pd.DataFrame(summary_rows).to_csv(outdir / "medqa_audit_summary.csv", index=False)
        print(f"Saved summary CSV: {outdir / 'medqa_audit_summary.csv'}")

    # Save detailed JSON for downstream use
    # (just the flags, not the full question text)
    json_out = []
    for r in all_rows:
        jr = {k: v for k, v in r.items() if k != "question_text"}
        # Convert lists to serializable form
        for k, v in jr.items():
            if isinstance(v, list):
                jr[k] = v
        json_out.append(jr)

    json_path = outdir / "medqa_demographic_audit.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"Saved detailed JSON: {json_path}")

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Audit MedQA dataset for demographic attribute mentions")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to scan")
    parser.add_argument("--output_dir", type=str, default="./medqa_audit")
    parser.add_argument("--all_splits", action="store_true", help="Scan all splits (train, validation, test)")
    args = parser.parse_args()
    run_audit(args.split, args.output_dir, args.all_splits)


if __name__ == "__main__":
    main()
