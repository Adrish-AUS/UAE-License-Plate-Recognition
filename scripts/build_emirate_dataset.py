#!/usr/bin/env python
"""
Recover UAE emirate labels from the original 51-class Roboflow export.

The raw Roboflow split is NOT used for the final classifier. The script scans
all raw folders only to recover annotations, then follows the frozen
train/val/test membership in datasets/uae_lp_v2_yolo.

Output:
datasets/uae_lp_emirate/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg
    train/annotations.csv
    val/annotations.csv
    test/annotations.csv

results/emirate_recovery/
    recovery_summary.json
    all_recovered_annotations.csv
    excluded_records.csv
    previews/<class>.jpg

The two source classes new_am and old_am are deliberately mapped to
AM_UNVERIFIED until their meaning is confirmed visually.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Source class IDs from the 51-class data.yaml.
SOURCE_CLASS_NAMES = {
    37: "new_DUBAI",
    38: "new_RAK",
    39: "new_abudabi",
    40: "new_ajman",
    41: "new_am",
    42: "new_fujairah",
    43: "old_DUBAI",
    44: "old_RAK",
    45: "old_abudabi",
    46: "old_ajman",
    47: "old_am",
    48: "old_fujira",
    49: "old_sharka",
}

TARGET_CLASS_BY_SOURCE_ID = {
    37: "Dubai",
    38: "Ras_Al_Khaimah",
    39: "Abu_Dhabi",
    40: "Ajman",
    41: "AM_UNVERIFIED",
    42: "Fujairah",
    43: "Dubai",
    44: "Ras_Al_Khaimah",
    45: "Abu_Dhabi",
    46: "Ajman",
    47: "AM_UNVERIFIED",
    48: "Fujairah",
    49: "Sharjah",
}

RAW_PLATE_CLASS_ID = 50
CURRENT_PLATE_CLASS_ID = 0


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            self.cx - half_w,
            self.cy - half_h,
            self.cx + half_w,
            self.cy + half_h,
        )


@dataclass(frozen=True)
class RawRecord:
    image_path: Path
    label_path: Path
    raw_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover emirate-labelled plate crops from the raw 51-class export."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/UAE_raw_51class"),
        help="Raw 51-class Roboflow export.",
    )
    parser.add_argument(
        "--current-root",
        type=Path,
        default=Path("datasets/uae_lp_v2_yolo"),
        help="Current one-class dataset whose split membership must be preserved.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/uae_lp_emirate"),
        help="Recovered classification dataset output.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/emirate_recovery"),
        help="Audit tables and preview output.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding around each current full-plate box.",
    )
    parser.add_argument(
        "--min-plate-iou",
        type=float,
        default=0.80,
        help="Minimum IoU when matching a current plate box to raw class-50 plate box.",
    )
    parser.add_argument(
        "--min-emirate-overlap",
        type=float,
        default=0.50,
        help="Minimum fraction of an emirate box covered by a matched plate box.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=24,
        help="Maximum preview crops per recovered class.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild existing output folders.",
    )
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def source_key(filename: str) -> str:
    """
    Produce a fallback key that ignores Roboflow's .rf.<hash> suffix.

    Example:
      S-10198_jpg.rf.oP95QxogiQdY4gHguzjV.jpg -> s-10198_jpg
    """
    stem = Path(filename).stem.casefold()
    stem = re.sub(r"\.rf\.[^.]+$", "", stem)
    return stem


def find_split_dirs_current(current_root: Path) -> dict[str, tuple[Path, Path]]:
    layouts = [
        {
            "train": (current_root / "images" / "train", current_root / "labels" / "train"),
            "val": (current_root / "images" / "val", current_root / "labels" / "val"),
            "test": (current_root / "images" / "test", current_root / "labels" / "test"),
        },
        {
            "train": (current_root / "train" / "images", current_root / "train" / "labels"),
            "val": (current_root / "valid" / "images", current_root / "valid" / "labels"),
            "test": (current_root / "test" / "images", current_root / "test" / "labels"),
        },
    ]

    for layout in layouts:
        if all(image_dir.is_dir() and label_dir.is_dir() for image_dir, label_dir in layout.values()):
            return {split: (image_dir.resolve(), label_dir.resolve()) for split, (image_dir, label_dir) in layout.items()}

    details = []
    for layout in layouts:
        details.extend(
            f"{split}: {image_dir} | {label_dir}"
            for split, (image_dir, label_dir) in layout.items()
        )
    raise FileNotFoundError(
        "Could not identify the current dataset train/val/test layout. Tried:\n"
        + "\n".join(details)
    )


def discover_raw_records(raw_root: Path) -> list[RawRecord]:
    records: list[RawRecord] = []

    # Roboflow layout: raw_root/train/images + raw_root/train/labels.
    for split_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Raw image has no matching label: {image_path}")
            records.append(
                RawRecord(
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve(),
                    raw_split=split_dir.name,
                )
            )

    if not records:
        raise FileNotFoundError(
            f"No raw split folders containing images/ and labels/ were found below {raw_root}"
        )
    return records


def read_yolo_labels(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 values, found {len(parts)}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO row") from exc

        if any(value < 0.0 or value > 1.0 for value in (cx, cy, width, height)):
            raise ValueError(f"{path}:{line_number}: coordinates outside [0,1]")
        boxes.append(YoloBox(class_id, cx, cy, width, height))
    return boxes


def intersection_area(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def box_area(box: YoloBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def iou(a: YoloBox, b: YoloBox) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner: YoloBox, outer: YoloBox) -> bool:
    x1, y1, x2, y2 = outer.xyxy
    return x1 <= inner.cx <= x2 and y1 <= inner.cy <= y2


def overlap_fraction(inner: YoloBox, outer: YoloBox) -> float:
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def match_raw_record(
    image_name: str,
    exact_index: dict[str, list[RawRecord]],
    fallback_index: dict[str, list[RawRecord]],
) -> tuple[RawRecord | None, str]:
    exact = exact_index.get(image_name.casefold(), [])
    if len(exact) == 1:
        return exact[0], "exact_filename"
    if len(exact) > 1:
        return None, "duplicate_exact_raw_filename"

    fallback = fallback_index.get(source_key(image_name), [])
    if len(fallback) == 1:
        return fallback[0], "roboflow_base_key"
    if len(fallback) > 1:
        return None, "duplicate_fallback_raw_key"
    return None, "raw_image_not_found"


def crop_box_pixels(
    box: YoloBox,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.xyxy
    pad_x = box.width * padding
    pad_y = box.height * padding

    px1 = max(0, math.floor((x1 - pad_x) * image_width))
    py1 = max(0, math.floor((y1 - pad_y) * image_height))
    px2 = min(image_width, math.ceil((x2 + pad_x) * image_width))
    py2 = min(image_height, math.ceil((y2 + pad_y) * image_height))
    return px1, py1, px2, py2


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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


def create_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    title: str,
    columns: int = 4,
    tile_width: int = 320,
    tile_height: int = 170,
) -> None:
    if not image_paths:
        return

    count = min(len(image_paths), len(labels))
    rows = math.ceil(count / columns)
    title_height = 48
    canvas = Image.new(
        "RGB",
        (columns * tile_width, title_height + rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(14)
    draw.text((10, 10), title, fill="black", font=title_font)

    for index in range(count):
        grid_x = index % columns
        grid_y = index // columns
        x0 = grid_x * tile_width
        y0 = title_height + grid_y * tile_height

        with Image.open(image_paths[index]) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 20, tile_height - 48))
            paste_x = x0 + (tile_width - image.width) // 2
            paste_y = y0 + 30 + (tile_height - 40 - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))

        draw.rectangle(
            (x0 + 2, y0 + 2, x0 + tile_width - 3, y0 + tile_height - 3),
            outline=(170, 170, 170),
            width=1,
        )
        draw.text(
            (x0 + 7, y0 + 7),
            labels[index][:42],
            fill="black",
            font=label_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_root = resolve(project_root, args.raw_root)
    current_root = resolve(project_root, args.current_root)
    output_root = resolve(project_root, args.output_root)
    results_root = resolve(project_root, args.results_root)

    if not 0.0 <= args.padding <= 0.5:
        raise ValueError("--padding must be between 0 and 0.5")
    if not 0.0 <= args.min_plate_iou <= 1.0:
        raise ValueError("--min-plate-iou must be between 0 and 1")
    if not 0.0 <= args.min_emirate_overlap <= 1.0:
        raise ValueError("--min-emirate-overlap must be between 0 and 1")

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not current_root.is_dir():
        raise FileNotFoundError(f"Current dataset root not found: {current_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Inspect it or rerun with --overwrite."
            )
        shutil.rmtree(output_root)

    if results_root.exists() and args.overwrite:
        shutil.rmtree(results_root)

    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    current_splits = find_split_dirs_current(current_root)
    raw_records = discover_raw_records(raw_root)

    exact_index: dict[str, list[RawRecord]] = defaultdict(list)
    fallback_index: dict[str, list[RawRecord]] = defaultdict(list)
    raw_split_counts = Counter()

    for record in raw_records:
        exact_index[record.image_path.name.casefold()].append(record)
        fallback_index[source_key(record.image_path.name)].append(record)
        raw_split_counts[record.raw_split] += 1

    recovered_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in current_splits
    }
    match_methods = Counter()
    exclusion_reasons = Counter()
    preview_paths: dict[str, list[Path]] = defaultdict(list)
    preview_labels: dict[str, list[str]] = defaultdict(list)

    current_image_counts: dict[str, int] = {}
    current_box_counts: dict[str, int] = {}

    for split, (image_dir, label_dir) in current_splits.items():
        split_output = output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        split_rows: list[dict[str, object]] = []

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        current_image_counts[split] = len(images)
        current_box_counts[split] = 0

        for image_path in images:
            current_label_path = label_dir / f"{image_path.stem}.txt"
            if not current_label_path.is_file():
                excluded_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "plate_index": "",
                        "reason": "current_label_missing",
                        "details": str(current_label_path),
                    }
                )
                exclusion_reasons["current_label_missing"] += 1
                continue

            current_boxes = [
                box
                for box in read_yolo_labels(current_label_path)
                if box.class_id == CURRENT_PLATE_CLASS_ID
            ]
            current_box_counts[split] += len(current_boxes)

            raw_record, match_method = match_raw_record(
                image_path.name,
                exact_index,
                fallback_index,
            )
            if raw_record is None:
                for plate_index in range(1, len(current_boxes) + 1):
                    excluded_rows.append(
                        {
                            "split": split,
                            "image_path": str(image_path),
                            "plate_index": plate_index,
                            "reason": match_method,
                            "details": "",
                        }
                    )
                    exclusion_reasons[match_method] += 1
                continue

            match_methods[match_method] += 1
            raw_boxes = read_yolo_labels(raw_record.label_path)
            raw_plate_boxes = [
                box for box in raw_boxes if box.class_id == RAW_PLATE_CLASS_ID
            ]
            raw_emirate_boxes = [
                box for box in raw_boxes if box.class_id in TARGET_CLASS_BY_SOURCE_ID
            ]

            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image_width, image_height = image.size

                for plate_index, current_plate in enumerate(current_boxes, start=1):
                    if not raw_plate_boxes:
                        excluded_rows.append(
                            {
                                "split": split,
                                "image_path": str(image_path),
                                "plate_index": plate_index,
                                "reason": "raw_plate_box_missing",
                                "details": str(raw_record.label_path),
                            }
                        )
                        exclusion_reasons["raw_plate_box_missing"] += 1
                        continue

                    scored_raw_plates = sorted(
                        (
                            (iou(current_plate, raw_plate), raw_plate)
                            for raw_plate in raw_plate_boxes
                        ),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    best_plate_iou, matched_raw_plate = scored_raw_plates[0]

                    if best_plate_iou < args.min_plate_iou:
                        excluded_rows.append(
                            {
                                "split": split,
                                "image_path": str(image_path),
                                "plate_index": plate_index,
                                "reason": "raw_plate_iou_below_threshold",
                                "details": f"{best_plate_iou:.6f}",
                            }
                        )
                        exclusion_reasons["raw_plate_iou_below_threshold"] += 1
                        continue

                    candidate_emirates: list[tuple[float, YoloBox]] = []
                    for emirate_box in raw_emirate_boxes:
                        overlap = overlap_fraction(emirate_box, matched_raw_plate)
                        if center_inside(emirate_box, matched_raw_plate) or overlap >= args.min_emirate_overlap:
                            candidate_emirates.append((overlap, emirate_box))

                    if not candidate_emirates:
                        excluded_rows.append(
                            {
                                "split": split,
                                "image_path": str(image_path),
                                "plate_index": plate_index,
                                "reason": "no_emirate_annotation_inside_plate",
                                "details": f"raw_emirate_boxes={len(raw_emirate_boxes)}",
                            }
                        )
                        exclusion_reasons["no_emirate_annotation_inside_plate"] += 1
                        continue

                    target_classes = {
                        TARGET_CLASS_BY_SOURCE_ID[box.class_id]
                        for _, box in candidate_emirates
                    }
                    if len(target_classes) != 1:
                        detail = "|".join(
                            sorted(
                                f"{SOURCE_CLASS_NAMES[box.class_id]}:{overlap:.3f}"
                                for overlap, box in candidate_emirates
                            )
                        )
                        excluded_rows.append(
                            {
                                "split": split,
                                "image_path": str(image_path),
                                "plate_index": plate_index,
                                "reason": "conflicting_emirate_annotations",
                                "details": detail,
                            }
                        )
                        exclusion_reasons["conflicting_emirate_annotations"] += 1
                        continue

                    candidate_emirates.sort(key=lambda item: item[0], reverse=True)
                    emirate_overlap, chosen_emirate_box = candidate_emirates[0]
                    target_class = TARGET_CLASS_BY_SOURCE_ID[chosen_emirate_box.class_id]
                    source_class_name = SOURCE_CLASS_NAMES[chosen_emirate_box.class_id]

                    x1, y1, x2, y2 = crop_box_pixels(
                        current_plate,
                        image_width,
                        image_height,
                        args.padding,
                    )
                    if x2 <= x1 or y2 <= y1:
                        excluded_rows.append(
                            {
                                "split": split,
                                "image_path": str(image_path),
                                "plate_index": plate_index,
                                "reason": "invalid_crop_coordinates",
                                "details": f"{x1},{y1},{x2},{y2}",
                            }
                        )
                        exclusion_reasons["invalid_crop_coordinates"] += 1
                        continue

                    class_dir = split_output / target_class
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = (
                        f"{safe_name(image_path.stem)}"
                        f"__plate{plate_index:02d}.jpg"
                    )
                    crop_path = class_dir / crop_name

                    crop = image.crop((x1, y1, x2, y2))
                    crop.save(crop_path, quality=95)

                    row = {
                        "split": split,
                        "current_image_path": str(image_path.resolve()),
                        "current_label_path": str(current_label_path.resolve()),
                        "raw_image_path": str(raw_record.image_path),
                        "raw_label_path": str(raw_record.label_path),
                        "raw_split": raw_record.raw_split,
                        "raw_match_method": match_method,
                        "plate_index": plate_index,
                        "crop_path": str(crop_path.resolve()),
                        "target_emirate": target_class,
                        "source_class_id": chosen_emirate_box.class_id,
                        "source_class_name": source_class_name,
                        "plate_match_iou": best_plate_iou,
                        "emirate_overlap_fraction": emirate_overlap,
                        "crop_x1": x1,
                        "crop_y1": y1,
                        "crop_x2": x2,
                        "crop_y2": y2,
                    }
                    recovered_rows.append(row)
                    split_rows.append(row)
                    split_counts[split][target_class] += 1

                    if len(preview_paths[target_class]) < args.preview_count:
                        preview_paths[target_class].append(crop_path)
                        preview_labels[target_class].append(
                            f"{split} | {source_class_name}"
                        )

        split_columns = [
            "split",
            "current_image_path",
            "current_label_path",
            "raw_image_path",
            "raw_label_path",
            "raw_split",
            "raw_match_method",
            "plate_index",
            "crop_path",
            "target_emirate",
            "source_class_id",
            "source_class_name",
            "plate_match_iou",
            "emirate_overlap_fraction",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
        ]
        write_csv(split_output / "annotations.csv", split_rows, split_columns)

    recovered_columns = [
        "split",
        "current_image_path",
        "current_label_path",
        "raw_image_path",
        "raw_label_path",
        "raw_split",
        "raw_match_method",
        "plate_index",
        "crop_path",
        "target_emirate",
        "source_class_id",
        "source_class_name",
        "plate_match_iou",
        "emirate_overlap_fraction",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
    ]
    excluded_columns = [
        "split",
        "image_path",
        "plate_index",
        "reason",
        "details",
    ]

    write_csv(
        results_root / "all_recovered_annotations.csv",
        recovered_rows,
        recovered_columns,
    )
    write_csv(
        results_root / "excluded_records.csv",
        excluded_rows,
        excluded_columns,
    )

    for target_class in sorted(preview_paths):
        create_contact_sheet(
            preview_paths[target_class],
            preview_labels[target_class],
            results_root / "previews" / f"{target_class}.jpg",
            title=f"{target_class} recovered plate crops",
        )

    total_class_counts = Counter(
        str(row["target_emirate"]) for row in recovered_rows
    )

    summary = {
        "raw_root": str(raw_root),
        "current_root": str(current_root),
        "output_root": str(output_root),
        "results_root": str(results_root),
        "raw_split_counts": dict(raw_split_counts),
        "current_image_counts": current_image_counts,
        "current_plate_box_counts": current_box_counts,
        "raw_match_methods": dict(match_methods),
        "recovered_crop_count": len(recovered_rows),
        "excluded_plate_count": len(excluded_rows),
        "exclusion_reasons": dict(exclusion_reasons),
        "total_class_counts": dict(total_class_counts),
        "split_class_counts": {
            split: dict(counter)
            for split, counter in split_counts.items()
        },
        "mapping": {
            SOURCE_CLASS_NAMES[source_id]: TARGET_CLASS_BY_SOURCE_ID[source_id]
            for source_id in sorted(SOURCE_CLASS_NAMES)
        },
        "important_note": (
            "AM_UNVERIFIED is intentionally not assigned to an emirate. "
            "Inspect results/emirate_recovery/previews/AM_UNVERIFIED.jpg before renaming it."
        ),
    }

    summary_path = results_root / "recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Emirate recovery completed")
    print("--------------------------")
    print(f"Raw records scanned:      {len(raw_records)}")
    print(f"Recovered crops:          {len(recovered_rows)}")
    print(f"Excluded plate records:   {len(excluded_rows)}")
    print()
    print("Recovered class counts:")
    for class_name, count in total_class_counts.most_common():
        print(f"  {class_name:20s} {count}")
    print()
    print("Split counts:")
    for split in ("train", "val", "test"):
        if split not in split_counts:
            continue
        formatted = ", ".join(
            f"{name}={count}"
            for name, count in sorted(split_counts[split].items())
        )
        print(f"  {split}: {formatted}")
    print()
    print(f"Summary:  {summary_path}")
    print(f"Previews: {results_root / 'previews'}")
    print()
    print(
        "Do not train yet. First inspect every preview, especially "
        "AM_UNVERIFIED.jpg, and review excluded_records.csv."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
