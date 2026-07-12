#!/usr/bin/env python
"""
Build a second OCR dataset that adds numeric-prefix Abu Dhabi plates to the
existing UAE OCR dataset.

Why:
The first OCR dataset accepted only labels beginning with letters, such as
S-10198 -> S10198. Many Abu Dhabi plates instead use numeric category codes,
such as 7-59700 -> 759700 or 10-14909 -> 1014909.

This script:
- preserves the existing OCR train/val/test samples;
- recovers strict numeric Abu Dhabi labels from original filenames;
- excludes ambiguous and multi-plate images;
- preserves the frozen train/val/test split;
- creates merged train/val/test annotations for fine-tuning and evaluation;
- creates a separate Abu Dhabi numeric test set for honest reporting.

It does not train or modify any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


NUMERIC_FILENAME_PATTERN = re.compile(
    r"^(?P<code>\d{1,2})-(?P<number>\d{2,7})_jpg\.rf\.",
    re.IGNORECASE,
)

EXPECTED_ORIGINAL_COUNTS = {
    "train": 3350,
    "val": 1055,
    "test": 1017,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OCR v2 dataset with numeric Abu Dhabi labels."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--original-ocr-root",
        type=Path,
        default=Path("datasets/uae_lp_ocr"),
    )
    parser.add_argument(
        "--recovered-annotations",
        type=Path,
        default=Path("results/emirate_recovery/all_recovered_annotations.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/uae_lp_ocr_v2"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def resolve_original_image(annotations_path: Path, value: str) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()

    candidate = (annotations_path.parent / candidate).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Original OCR image not found for annotation '{value}' in {annotations_path}"
    )


def resolve_recovered_crop(value: str, project_root: Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_file():
        return candidate.resolve()

    normalized = str(value).replace("\\", "/")
    marker = "/datasets/uae_lp_emirate/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = (
            project_root / "datasets" / "uae_lp_emirate" / relative
        ).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Recovered crop not found: {value}")


def parse_numeric_label(current_image_path: str) -> str | None:
    filename = PureWindowsPath(str(current_image_path)).name
    match = NUMERIC_FILENAME_PATTERN.match(filename)
    if match is None:
        return None

    label = match.group("code") + match.group("number")
    if not label.isdigit() or not 3 <= len(label) <= 9:
        return None
    return label


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def copy_and_record(
    source: Path,
    destination: Path,
    plate_text: str,
    source_type: str,
    source_reference: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "image_path": str(destination.resolve()),
        "plate_text": plate_text,
        "source_type": source_type,
        "source_reference": source_reference,
    }


def write_annotations(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "plate_text",
                "source_type",
                "source_reference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_training_annotations(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    FastPlateOCR only needs image_path and plate_text. Keep a second audit CSV
    with provenance, while training annotations stay minimal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "plate_text"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"],
                }
            )


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_preview(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    count: int = 24,
) -> None:
    rows = rows[:count]
    if not rows:
        return

    columns = 4
    tile_width = 320
    tile_height = 170
    title_height = 48
    row_count = (len(rows) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * tile_width, title_height + row_count * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black", font=load_font(24))
    label_font = load_font(14)

    for index, row in enumerate(rows):
        grid_x = index % columns
        grid_y = index // columns
        x0 = grid_x * tile_width
        y0 = title_height + grid_y * tile_height

        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 20, tile_height - 46))
            paste_x = x0 + (tile_width - image.width) // 2
            paste_y = y0 + 28 + (tile_height - 36 - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))

        draw.rectangle(
            (x0 + 2, y0 + 2, x0 + tile_width - 3, y0 + tile_height - 3),
            outline=(170, 170, 170),
            width=1,
        )
        draw.text(
            (x0 + 8, y0 + 7),
            f"Label: {row['plate_text']}",
            fill="black",
            font=label_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    original_root = resolve(project_root, args.original_ocr_root)
    recovered_path = resolve(project_root, args.recovered_annotations)
    output_root = resolve(project_root, args.output_root)

    if not original_root.is_dir():
        raise FileNotFoundError(f"Original OCR root not found: {original_root}")
    if not recovered_path.is_file():
        raise FileNotFoundError(
            f"Recovered annotations not found: {recovered_path}"
        )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite only when intentionally rebuilding it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    recovered = pd.read_csv(recovered_path, dtype=str).fillna("")
    required_columns = {
        "split",
        "current_image_path",
        "crop_path",
        "target_emirate",
    }
    missing = required_columns - set(recovered.columns)
    if missing:
        raise ValueError(
            f"Recovered annotations are missing columns: {sorted(missing)}"
        )

    recovered = recovered[
        recovered["target_emirate"].isin(["Abu_Dhabi", "Abu Dhabi"])
    ].copy()

    # A filename can safely label only one plate crop.
    per_image_counts = recovered["current_image_path"].value_counts()
    recovered = recovered[
        recovered["current_image_path"].map(per_image_counts) == 1
    ].copy()

    recovered["plate_text"] = recovered["current_image_path"].map(
        parse_numeric_label
    )
    recovered = recovered[recovered["plate_text"].notna()].copy()

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    numeric_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Copy original OCR data.
    for split in ("train", "val", "test"):
        annotations_path = original_root / split / "annotations.csv"
        if not annotations_path.is_file():
            raise FileNotFoundError(annotations_path)

        original = pd.read_csv(
            annotations_path,
            dtype={"image_path": str, "plate_text": str},
        )
        if len(original) != EXPECTED_ORIGINAL_COUNTS[split]:
            raise RuntimeError(
                f"Original OCR {split} count is {len(original)}, expected "
                f"{EXPECTED_ORIGINAL_COUNTS[split]}."
            )

        original_dest = output_root / split / "images"
        for index, row in original.iterrows():
            source = resolve_original_image(
                annotations_path,
                str(row["image_path"]),
            )
            destination = (
                original_dest
                / f"orig_{index:06d}_{safe_stem(source.name)}"
            )
            split_rows[split].append(
                copy_and_record(
                    source=source,
                    destination=destination,
                    plate_text=str(row["plate_text"]).strip(),
                    source_type="original_ocr",
                    source_reference=str(source),
                )
            )

    # Copy strict numeric Abu Dhabi crops into the same frozen split.
    for split in ("train", "val", "test"):
        subset = recovered[recovered["split"] == split].copy()
        subset = subset.sort_values(
            ["current_image_path", "crop_path"]
        ).reset_index(drop=True)

        destination_root = output_root / split / "images"
        for index, row in subset.iterrows():
            source = resolve_recovered_crop(
                str(row["crop_path"]),
                project_root,
            )
            destination = (
                destination_root
                / f"adnum_{index:06d}_{safe_stem(source.name)}"
            )
            record = copy_and_record(
                source=source,
                destination=destination,
                plate_text=str(row["plate_text"]),
                source_type="abu_dhabi_numeric",
                source_reference=str(row["current_image_path"]),
            )
            split_rows[split].append(record)
            numeric_rows[split].append(record)

    # Write merged and numeric-only datasets.
    for split in ("train", "val", "test"):
        split_dir = output_root / split
        write_training_annotations(
            split_dir / "annotations.csv",
            split_rows[split],
        )
        write_annotations(
            split_dir / "annotations_with_source.csv",
            split_rows[split],
        )

        numeric_dir = output_root / f"abu_dhabi_numeric_{split}"
        numeric_only_records: list[dict[str, Any]] = []
        numeric_image_dir = numeric_dir / "images"
        for index, row in enumerate(numeric_rows[split]):
            source = Path(row["image_path"])
            destination = (
                numeric_image_dir
                / f"{index:06d}_{safe_stem(source.name)}"
            )
            numeric_only_records.append(
                copy_and_record(
                    source=source,
                    destination=destination,
                    plate_text=row["plate_text"],
                    source_type=row["source_type"],
                    source_reference=row["source_reference"],
                )
            )

        write_training_annotations(
            numeric_dir / "annotations.csv",
            numeric_only_records,
        )
        write_annotations(
            numeric_dir / "annotations_with_source.csv",
            numeric_only_records,
        )

    make_preview(
        numeric_rows["train"],
        output_root / "abu_dhabi_numeric_train_preview.jpg",
        "Abu Dhabi numeric OCR training samples",
    )
    make_preview(
        numeric_rows["test"],
        output_root / "abu_dhabi_numeric_test_preview.jpg",
        "Abu Dhabi numeric OCR test samples",
    )

    summary = {
        "original_ocr_counts": EXPECTED_ORIGINAL_COUNTS,
        "abu_dhabi_numeric_counts": {
            split: len(numeric_rows[split])
            for split in ("train", "val", "test")
        },
        "merged_counts": {
            split: len(split_rows[split])
            for split in ("train", "val", "test")
        },
        "label_rule": (
            "Strict numeric filenames only: one/two digit code, hyphen, "
            "two-to-seven digit number."
        ),
        "multi_plate_policy": (
            "Images with more than one recovered plate row were excluded "
            "because one filename cannot safely label multiple crops."
        ),
        "split_policy": (
            "The existing frozen train/validation/test split was preserved."
        ),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("OCR v2 dataset created")
    print("----------------------")
    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: original={EXPECTED_ORIGINAL_COUNTS[split]} "
            f"abu_dhabi_numeric={len(numeric_rows[split])} "
            f"merged={len(split_rows[split])}"
        )
    print(f"\nOutput: {output_root}")
    print(
        "Next: validate the merged train/val annotations and fine-tune from "
        "the existing best.keras checkpoint."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
