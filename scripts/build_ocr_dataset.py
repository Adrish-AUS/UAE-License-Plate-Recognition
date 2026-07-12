import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATASET = (
    ROOT
    / "datasets"
    / "uae_lp_v2_rfdetr_coco"
)

OUTPUT_DATASET = (
    ROOT
    / "datasets"
    / "uae_lp_ocr"
)

SUMMARY_FILE = (
    ROOT
    / "results"
    / "ocr_training"
    / "ocr_dataset_summary.json"
)

# Input split name -> output split name
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

# Add a small amount of context around the COCO box.
HORIZONTAL_PADDING_RATIO = 0.05
VERTICAL_PADDING_RATIO = 0.10

# Very tiny annotations are unlikely to contain usable text.
MINIMUM_BOX_WIDTH = 30
MINIMUM_BOX_HEIGHT = 12

PREVIEW_SAMPLES = 36
RANDOM_SEED = 486


def remove_roboflow_suffix(filename: str) -> str:
    """
    Convert a Roboflow-exported filename back to its original stem.

    Example:
        S-10198_jpg.rf.ABC123.jpg
    becomes:
        S-10198
    """
    stem = Path(filename).stem

    # Remove ".rf.randomHash".
    stem = re.sub(
        r"\.rf\.[A-Za-z0-9]+$",
        "",
        stem,
    )

    # Remove repeated export suffixes.
    suffixes = [
        "_jpeg_jpg",
        "_png_jpg",
        "_jpg_jpg",
        "_jpeg",
        "_png",
        "_jpg",
    ]

    changed = True

    while changed:
        changed = False

        for suffix in suffixes:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break

    return stem


def parse_strict_plate_label(
    filename: str,
) -> str | None:
    """
    Accept only filenames shaped exactly like:

        S-10198
        A_10005
        K 3483

    The result becomes:

        S10198
        A10005
        K3483

    Filenames containing source numbers, dates, WhatsApp names,
    hashes or unrelated words are rejected.
    """
    cleaned = remove_roboflow_suffix(filename)

    match = re.fullmatch(
        r"([A-Za-z]{1,3})[-_ ]+(\d{2,7})",
        cleaned,
    )

    if match is None:
        return None

    code = match.group(1).upper()
    number = match.group(2)

    return f"{code}{number}"


def clamp_crop_box(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a COCO XYWH box to a padded, clamped XYXY crop.
    """
    x, y, width, height = map(float, bbox)

    horizontal_padding = (
        width * HORIZONTAL_PADDING_RATIO
    )

    vertical_padding = (
        height * VERTICAL_PADDING_RATIO
    )

    x1 = max(
        0,
        int(round(x - horizontal_padding)),
    )

    y1 = max(
        0,
        int(round(y - vertical_padding)),
    )

    x2 = min(
        image_width,
        int(round(x + width + horizontal_padding)),
    )

    y2 = min(
        image_height,
        int(round(y + height + vertical_padding)),
    )

    return x1, y1, x2, y2


def make_preview(
    split_name: str,
    examples: list[dict],
    output_directory: Path,
) -> None:
    """
    Create a contact sheet showing several generated crops and labels.
    """
    if not examples:
        return

    random_generator = random.Random(
        RANDOM_SEED
    )

    selected = random_generator.sample(
        examples,
        min(PREVIEW_SAMPLES, len(examples)),
    )

    columns = 4
    cell_width = 320
    cell_height = 150

    rows = (
        len(selected) + columns - 1
    ) // columns

    canvas = np.full(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for index, example in enumerate(selected):
        row = index // columns
        column = index % columns

        crop = cv2.imread(
            str(example["absolute_image_path"])
        )

        if crop is None:
            continue

        available_width = cell_width - 20
        available_height = cell_height - 45

        scale = min(
            available_width / crop.shape[1],
            available_height / crop.shape[0],
        )

        scale = min(scale, 1.5)

        resized_width = max(
            1,
            int(crop.shape[1] * scale),
        )

        resized_height = max(
            1,
            int(crop.shape[0] * scale),
        )

        resized = cv2.resize(
            crop,
            (
                resized_width,
                resized_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        origin_x = (
            column * cell_width
            + (cell_width - resized_width) // 2
        )

        origin_y = (
            row * cell_height
            + 8
        )

        canvas[
            origin_y:origin_y + resized_height,
            origin_x:origin_x + resized_width,
        ] = resized

        text_x = column * cell_width + 10
        text_y = (
            row * cell_height
            + cell_height
            - 12
        )

        cv2.putText(
            canvas,
            example["plate_text"],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    preview_file = (
        output_directory
        / f"{split_name}_preview.jpg"
    )

    cv2.imwrite(
        str(preview_file),
        canvas,
    )


def process_split(
    source_split: str,
    output_split: str,
) -> dict:
    """
    Build one OCR split using strict filename labels and COCO boxes.
    """
    source_directory = (
        SOURCE_DATASET
        / source_split
    )

    annotation_file = (
        source_directory
        / "_annotations.coco.json"
    )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Missing COCO file: {annotation_file}"
        )

    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    output_directory = (
        OUTPUT_DATASET
        / output_split
    )

    image_output_directory = (
        output_directory
        / "images"
    )

    image_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:
        annotations_by_image[
            annotation["image_id"]
        ].append(annotation)

    counts = Counter()
    rows = []
    preview_examples = []

    image_records = sorted(
        coco["images"],
        key=lambda record: record["id"],
    )

    for image_record in image_records:
        counts["images_seen"] += 1

        filename = Path(
            image_record["file_name"]
        ).name

        plate_text = parse_strict_plate_label(
            filename
        )

        if plate_text is None:
            counts[
                "skipped_unreliable_filename"
            ] += 1
            continue

        counts["strict_filename"] += 1

        annotations = annotations_by_image.get(
            image_record["id"],
            [],
        )

        # A filename normally describes one plate only. Do not assign
        # that same text to several boxes in a multi-plate image.
        if len(annotations) != 1:
            counts[
                "skipped_not_exactly_one_box"
            ] += 1
            continue

        annotation = annotations[0]

        _, _, box_width, box_height = map(
            float,
            annotation["bbox"],
        )

        if (
            box_width < MINIMUM_BOX_WIDTH
            or box_height < MINIMUM_BOX_HEIGHT
        ):
            counts["skipped_tiny_box"] += 1
            continue

        source_image = (
            source_directory
            / filename
        )

        if not source_image.exists():
            counts["skipped_missing_image"] += 1
            continue

        image = cv2.imread(
            str(source_image)
        )

        if image is None:
            counts["skipped_unreadable_image"] += 1
            continue

        actual_height, actual_width = image.shape[:2]

        x1, y1, x2, y2 = clamp_crop_box(
            annotation["bbox"],
            actual_width,
            actual_height,
        )

        if x2 <= x1 or y2 <= y1:
            counts["skipped_invalid_crop"] += 1
            continue

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            counts["skipped_empty_crop"] += 1
            continue

        safe_label = re.sub(
            r"[^A-Z0-9]",
            "",
            plate_text,
        )

        output_filename = (
            f"{image_record['id']:06d}_"
            f"{safe_label}.jpg"
        )

        output_image = (
            image_output_directory
            / output_filename
        )

        saved = cv2.imwrite(
            str(output_image),
            crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not saved:
            counts["skipped_write_failure"] += 1
            continue

        relative_image_path = (
            Path("images")
            / output_filename
        ).as_posix()

        rows.append(
            {
                "image_path": relative_image_path,
                "plate_text": plate_text,
            }
        )

        preview_examples.append(
            {
                "absolute_image_path": output_image,
                "plate_text": plate_text,
            }
        )

        counts["saved"] += 1

    annotations_csv = (
        output_directory
        / "annotations.csv"
    )

    with annotations_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "plate_text",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    make_preview(
        output_split,
        preview_examples,
        output_directory,
    )

    return {
        "source_split": source_split,
        "output_split": output_split,
        "annotations_csv": str(
            annotations_csv
        ),
        **dict(counts),
    }


def main() -> None:
    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "source_dataset": str(
            SOURCE_DATASET
        ),
        "output_dataset": str(
            OUTPUT_DATASET
        ),
        "label_rule": (
            "One to three Latin letters, separator, "
            "then two to seven digits"
        ),
        "single_box_images_only": True,
        "splits": {},
    }

    print("Building custom UAE OCR dataset...\n")

    for source_split, output_split in SPLITS.items():
        split_result = process_split(
            source_split,
            output_split,
        )

        summary["splits"][
            output_split
        ] = split_result

        print(
            f"{output_split}: "
            f"saved={split_result.get('saved', 0)}, "
            f"strict filenames="
            f"{split_result.get('strict_filename', 0)}, "
            f"multi/no box skipped="
            f"{split_result.get('skipped_not_exactly_one_box', 0)}, "
            f"tiny skipped="
            f"{split_result.get('skipped_tiny_box', 0)}"
        )

    summary["total_saved"] = sum(
        split_result.get("saved", 0)
        for split_result in summary[
            "splits"
        ].values()
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nOCR dataset creation complete.")
    print(
        f"Total crops saved: "
        f"{summary['total_saved']}"
    )

    print("\nDataset location:")
    print(OUTPUT_DATASET)

    print("\nSummary:")
    print(SUMMARY_FILE)

    print(
        "\nOpen each preview image and confirm that "
        "the displayed label matches the visible plate."
    )


if __name__ == "__main__":
    main()