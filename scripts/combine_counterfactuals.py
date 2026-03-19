#!/usr/bin/env python3
"""Combine two or more counterfactual JSON datasets into one.

Deduplicates by question_id — if the same question appears in multiple
files, the version from the *later* file on the command line wins.

Usage
-----
python scripts/combine_counterfactuals.py \
    cf_v5_3_tier1_2_3.json cf_v5_3_tier2_3a.json \
    -o cf_v5_3_combined.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Legacy nested-dict format: values are the records
        return list(data.values())
    raise ValueError(f"Unexpected top-level type in {path}: {type(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge counterfactual JSON files, deduplicating by question_id."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Two or more counterfactual JSON files to combine.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path for the combined output JSON file.",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("Need at least two input files.", file=sys.stderr)
        sys.exit(1)

    seen: dict[str, dict] = {}  # question_id → record (last wins)
    total_loaded = 0

    for path in args.inputs:
        records = load_records(path)
        total_loaded += len(records)
        for rec in records:
            qid = rec.get("question_id")
            if qid is None:
                print(f"Warning: skipping record without question_id in {path}",
                      file=sys.stderr)
                continue
            seen[qid] = rec
        print(f"  Loaded {len(records):>5} records from {path}")

    combined = list(seen.values())
    duplicates = total_loaded - len(combined)

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined: {len(combined)} unique records "
          f"({duplicates} duplicates removed) → {args.output}")


if __name__ == "__main__":
    main()
