#!/usr/bin/env python
"""
Finalize the recovered emirate dataset by mapping:
    AM_UNVERIFIED -> Umm_Al_Quwain

The preview inspection confirmed that source classes new_am and old_am are
Umm Al Quwain plates. This script renames class folders and updates all
generated CSV/JSON metadata without rebuilding crops.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


OLD_LABEL = "AM_UNVERIFIED"
NEW_LABEL = "Umm_Al_Quwain"


def update_csv(path: Path) -> int:
    if not path.is_file():
        return 0

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return 0
        rows = list(reader)

    changed = 0
    for row in rows:
        if row.get("target_emirate") == OLD_LABEL:
            row["target_emirate"] = NEW_LABEL
            changed += 1

        crop_path = row.get("crop_path", "")
        if OLD_LABEL in crop_path:
            row["crop_path"] = crop_path.replace(OLD_LABEL, NEW_LABEL)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return changed


def rename_class_folders(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        old_dir = split_root / OLD_LABEL
        new_dir = split_root / NEW_LABEL

        if old_dir.is_dir() and new_dir.exists():
            raise FileExistsError(
                f"Both old and new class folders exist: {old_dir} and {new_dir}"
            )
        if old_dir.is_dir():
            old_dir.rename(new_dir)
            print(f"Renamed: {old_dir} -> {new_dir}")
        elif new_dir.is_dir():
            print(f"Already renamed: {new_dir}")
        else:
            raise FileNotFoundError(
                f"Neither class folder exists for split '{split}': "
                f"{old_dir} or {new_dir}"
            )


def update_summary(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Recovery summary not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    total_counts = data.get("total_class_counts", {})
    if OLD_LABEL in total_counts:
        total_counts[NEW_LABEL] = total_counts.pop(OLD_LABEL)

    split_counts = data.get("split_class_counts", {})
    for counts in split_counts.values():
        if OLD_LABEL in counts:
            counts[NEW_LABEL] = counts.pop(OLD_LABEL)

    mapping = data.get("mapping", {})
    for source_name in ("new_am", "old_am"):
        if source_name in mapping:
            mapping[source_name] = NEW_LABEL

    data["important_note"] = (
        "Source classes new_am and old_am were confirmed by visual inspection "
        "to represent Umm Al Quwain plates."
    )

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    project_root = Path.cwd().resolve()
    dataset_root = project_root / "datasets" / "uae_lp_emirate"
    results_root = project_root / "results" / "emirate_recovery"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rename_class_folders(dataset_root)

    changed_total = 0
    for split in ("train", "val", "test"):
        changed_total += update_csv(dataset_root / split / "annotations.csv")

    changed_total += update_csv(results_root / "all_recovered_annotations.csv")

    excluded_path = results_root / "excluded_records.csv"
    if excluded_path.is_file():
        # No target_emirate column is expected here, but leave the file untouched.
        pass

    update_summary(results_root / "recovery_summary.json")

    old_preview = results_root / "previews" / f"{OLD_LABEL}.jpg"
    new_preview = results_root / "previews" / f"{NEW_LABEL}.jpg"
    if old_preview.is_file() and not new_preview.exists():
        old_preview.rename(new_preview)
        print(f"Renamed preview: {new_preview}")

    confirmation_path = results_root / "umm_al_quwain_confirmation.txt"
    confirmation_path.write_text(
        "Source classes new_am and old_am were confirmed by visual inspection "
        "to represent Umm Al Quwain plates.\n",
        encoding="utf-8",
    )

    print()
    print("Emirate label finalization complete")
    print("-----------------------------------")
    print(f"Updated CSV rows: {changed_total}")
    print(f"Final class name: {NEW_LABEL}")
    print(f"Confirmation:     {confirmation_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
