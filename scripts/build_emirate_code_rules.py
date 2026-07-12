#!/usr/bin/env python
"""
Build emirate-specific plate code rules from the recovered TRAIN split only.

The rules are used only to split a recognised OCR string into:
    code + number

Example:
    Sharjah OCR text 3222566 -> code 3, number 222566
    Abu Dhabi OCR text 1141153 -> code 11, number 41153

The script does not use validation or test filenames, so it does not leak test
labels into the parser.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<code>[A-Za-z0-9]{1,3})-(?P<number>\d{2,7})_jpg\.rf\.",
    re.IGNORECASE,
)


def friendly(name: str) -> str:
    return str(name).replace("_", " ").strip()


def main() -> int:
    project_root = Path.cwd().resolve()
    source_csv = (
        project_root
        / "results"
        / "emirate_recovery"
        / "all_recovered_annotations.csv"
    )
    output_path = project_root / "results" / "emirate_code_rules.json"

    if not source_csv.is_file():
        raise FileNotFoundError(f"Recovered annotations not found: {source_csv}")

    data = pd.read_csv(source_csv, dtype=str).fillna("")
    required = {"split", "target_emirate", "current_image_path"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    train = data[data["split"] == "train"].copy()

    code_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parsed_rows = 0
    skipped_rows = 0

    for row in train.itertuples(index=False):
        filename = PureWindowsPath(str(row.current_image_path)).name
        match = FILENAME_PATTERN.match(filename)
        if match is None:
            skipped_rows += 1
            continue

        emirate = friendly(row.target_emirate)
        code = match.group("code").upper()
        code_counts[emirate][code] += 1
        parsed_rows += 1

    rules = {}
    for emirate, counts in sorted(code_counts.items()):
        numeric_counts = {
            code: count for code, count in counts.items() if code.isdigit()
        }
        letter_counts = {
            code: count for code, count in counts.items() if code.isalpha()
        }

        numeric_codes = sorted(
            numeric_counts,
            key=lambda code: (-len(code), -numeric_counts[code], code),
        )
        letter_codes = sorted(
            letter_counts,
            key=lambda code: (-len(code), -letter_counts[code], code),
        )

        numeric_length_counts = Counter(
            {length: 0 for length in range(1, 4)}
        )
        for code, count in numeric_counts.items():
            numeric_length_counts[len(code)] += count

        preferred_numeric_length = None
        if numeric_counts:
            preferred_numeric_length = max(
                numeric_length_counts,
                key=lambda length: numeric_length_counts[length],
            )

        rules[emirate] = {
            "numeric_codes": numeric_codes,
            "numeric_code_counts": numeric_counts,
            "letter_codes": letter_codes,
            "letter_code_counts": letter_counts,
            "preferred_numeric_code_length": preferred_numeric_length,
            "total_parsed_train_rows": int(sum(counts.values())),
        }

    payload = {
        "source": str(source_csv),
        "source_split": "train",
        "parsed_train_rows": parsed_rows,
        "skipped_train_rows": skipped_rows,
        "rules": rules,
        "note": (
            "Rules are derived only from recovered training filenames and are "
            "used to split an OCR string into code and number. They do not alter "
            "the OCR prediction."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Emirate code rules created")
    print("--------------------------")
    print(f"Parsed train rows:  {parsed_rows}")
    print(f"Skipped train rows: {skipped_rows}")
    print(f"Saved:              {output_path}")
    print()

    for emirate, rule in rules.items():
        numeric = ", ".join(rule["numeric_codes"][:20]) or "(none)"
        letters = ", ".join(rule["letter_codes"][:20]) or "(none)"
        print(f"{emirate}")
        print(f"  Numeric codes: {numeric}")
        print(f"  Letter codes:  {letters}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
