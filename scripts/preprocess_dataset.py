"""Audit the frozen release or reproduce it from the original multiclass export.

The accepted membership is immutable and is defined by
``reports/dataset_manifest.csv``.  Audit mode regenerates derived evidence without
changing membership.  Raw-build mode writes only to a caller-supplied staging
directory and never replaces the accepted dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preprocessing_utils import (
    ACCEPTED_COUNTS,
    BOUND_TOLERANCE,
    COCO_CATEGORY_ID,
    COCO_PARITY_TOLERANCE_PX,
    LICENSE_ID,
    LICENSE_NAME,
    SOURCE_URL,
    SPLITS,
    TARGET_CLASS_ID,
    TARGET_CLASS_NAME,
    collect_dataset_records,
    count_source_labels,
    discover_source_splits,
    find_image_files,
    find_label_files,
    load_yolo_labels,
    read_class_names,
    read_csv,
    read_image_size,
    sha256_file,
    split_counts,
    write_csv,
    write_json,
    yolo_to_coco_bbox,
)

RELEASE_DATE = "2026-07-10"
RELEASE_VERSION = "2.0.1"
FROZEN_MEMBERSHIP = (
    "Frozen project-controlled approximately 70/15/15 membership recorded by "
    "reports/dataset_manifest.csv. The original membership-generation seed and exact "
    "grouping algorithm are not independently reconstructable from the committed evidence."
)
CLASS_MAPPING_COLUMNS = [
    "source_class_id",
    "source_class_name",
    "source_box_count",
    "decision",
    "target_class_id",
    "target_class_name",
    "boxes_kept",
    "boxes_removed",
    "reason",
]
EXCLUSION_COLUMNS = [
    "image_relative_path",
    "original_split",
    "reason_category",
    "decision_status",
    "decision_authority",
    "decision_date",
    "notes",
]
MANIFEST_COLUMNS = [
    "split",
    "image_relative_path",
    "label_relative_path",
    "image_width",
    "image_height",
    "image_size_bytes",
    "label_size_bytes",
    "box_count",
    "image_sha256",
    "label_sha256",
]
SIMILARITY_NOTE = (
    "Previously identified by automated cross-split scene-similarity screening and conservatively "
    "excluded by project-owner decision. This does not claim that the image is corrupt, mislabeled, "
    "or an exact duplicate."
)
OMISSION_NOTE = (
    "The original omission reason is not recoverable from the available project evidence. The image "
    "remains excluded solely to preserve the frozen, previously validated release membership. This "
    "does not claim that the image is corrupt, duplicated, unreadable, or mislabeled."
)
SIMILARITY_EXCLUSIONS = {
    "datasets/uae_lp_v2_yolo/images/test/1-shortmobileview-K13300_png_jpg.rf.OyEfZmZXOf7fY1UkWvtL.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-shortmobileview-Q6818600_png_jpg.rf.XcUKmna5NRPf8Jcbq97B.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-shortmobileview-Q6909600_png_jpg.rf.dBraxAvqqy08ac0HVnTa.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-shortview-K13300_png_jpg.rf.cTJQVCG7cSQDoNf7ilIu.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-view-Q2121610_png_jpg.rf.jeWAcuPvFoPxGbD4elEj.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-view-Q6909610_png_jpg.rf.Q7VIbtzAMzYpYB1TbaY0.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-viewmobile-K13300_png_jpg.rf.QXPtXdgTwTf10aesHz6w.jpg",
    "datasets/uae_lp_v2_yolo/images/test/1-viewmobile-K13310_png_jpg.rf.F5dky1xZp6o9WqtRQQq8.jpg",
    "datasets/uae_lp_v2_yolo/images/test/Z-44_jpg.rf.XmEzTI85SsaZVMPOfaP7.jpg",
    "datasets/uae_lp_v2_yolo/images/val/1-shortview-K8446600_jpg.rf.xUF71EM7PG4zHMmyMelQ.jpg",
    "datasets/uae_lp_v2_yolo/images/val/1-viewmobile-Q6909610_png_jpg.rf.Q833TzVdsyjeX8Jj9e35.jpg",
}
OMISSION_EXCLUSIONS = {
    "datasets/uae_lp_v2_yolo/images/test/3006_jpg.rf.sCzCUEjkCy5XVzovs42Z.jpg",
    "datasets/uae_lp_v2_yolo/images/test/C-8_jpg.rf.dcmfBOoSvKkOgBvm7n2q.jpg",
    "datasets/uae_lp_v2_yolo/images/test/WhatsApp-Image-2022-01-25-at-10-15-37-PM_jpeg_jpg.rf.0ajYLrBY8L7ZTZN95Gtg.jpg",
    "datasets/uae_lp_v2_yolo/images/val/WhatsApp-Image-2022-01-23-at-10-05-04-PM_jpeg_jpg.rf.MCf4q806wxYlPi3azIiI.jpg",
    "datasets/uae_lp_v2_yolo/images/val/WhatsApp-Image-2022-01-25-at-10-05-34-PM_jpeg_jpg.rf.GhXFKFMeqGRCZCCBJNrT.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-existing", action="store_true")
    mode.add_argument("--build-from-raw", action="store_true")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    return parser.parse_args()


def _resolve(base: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _expected_exclusions() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in SIMILARITY_EXCLUSIONS:
        rows[path] = {
            "image_relative_path": path,
            "original_split": path.split("/")[3],
            "reason_category": "conservative_cross_split_scene_similarity_exclusion",
            "decision_status": "EXCLUDED_BY_PROJECT_DECISION",
            "decision_authority": "project_owner",
            "decision_date": RELEASE_DATE,
            "notes": SIMILARITY_NOTE,
        }
    for path in OMISSION_EXCLUSIONS:
        rows[path] = {
            "image_relative_path": path,
            "original_split": path.split("/")[3],
            "reason_category": "frozen_release_omission_provenance_unavailable",
            "decision_status": "EXCLUDED_BY_PROJECT_DECISION",
            "decision_authority": "project_owner",
            "decision_date": RELEASE_DATE,
            "notes": OMISSION_NOTE,
        }
    return rows


def validate_exclusions(path: Path, manifest_paths: set[str]) -> list[dict[str, str]]:
    rows = read_csv(path)
    if len(rows) != 16 or (rows and list(rows[0]) != EXCLUSION_COLUMNS):
        raise ValueError(f"{path} must contain the exact seven columns and 16 decision rows")
    expected = _expected_exclusions()
    observed = {row["image_relative_path"].replace("\\", "/"): row for row in rows}
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(path for path in set(expected) & set(observed) if expected[path] != observed[path])
        raise ValueError(f"Exclusion ledger differs from project decisions: missing={missing}, extra={extra}, mismatched={mismatched}")
    overlap = sorted(set(observed) & manifest_paths)
    if overlap:
        raise ValueError(f"Excluded paths remain active: {overlap}")
    return [observed[path] for path in sorted(observed, key=str.casefold)]


def _target_source_id(names: list[str]) -> int:
    matches = [index for index, name in enumerate(names) if name.casefold() == "plate"]
    if len(matches) != 1:
        raise ValueError(f"Source must contain one exact class named plate; found IDs {matches}")
    return matches[0]


def _source_plate_candidate_count(split_dirs: dict[str, tuple[Path, Path]], target_id: int) -> int:
    labels = [path for _, label_dir in split_dirs.values() for path in find_label_files(label_dir)]

    def contains_plate(path: Path) -> bool:
        return any(box.class_id == target_id for box in load_yolo_labels(path, allow_empty=True))

    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sum(executor.map(contains_plate, labels))


def _decode_source_images(split_dirs: dict[str, tuple[Path, Path]]) -> int:
    images = [path for image_dir, _ in split_dirs.values() for path in find_image_files(image_dir)]

    def decode(path: Path) -> bool:
        try:
            read_image_size(path, fully_decode=True)
            return True
        except Exception:
            return False

    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded = sum(executor.map(decode, images))
    return len(images) - decoded


def validate_source(source_root: Path, *, decode_images: bool) -> dict[str, object]:
    split_dirs = discover_source_splits(source_root)
    names = read_class_names(source_root / "data.yaml")
    target_id = _target_source_id(names)
    class_counts, totals = count_source_labels(split_dirs, len(names))
    plate_candidates = _source_plate_candidate_count(split_dirs, target_id)
    if len(names) != 51 or target_id != 50:
        raise ValueError(f"Expected 51 source classes with plate at ID 50; observed {len(names)} classes and ID {target_id}")
    expected_totals = {"images": 9985, "labels": 9985, "boxes": 86294, "orphan_images": 0, "orphan_labels": 0}
    if totals != expected_totals:
        raise ValueError(f"Raw-source totals differ: observed={totals}, expected={expected_totals}")
    if class_counts[target_id] != 12468 or plate_candidates != 9626:
        raise ValueError(
            f"Raw plate accounting differs: plate_boxes={class_counts[target_id]}, candidate_images={plate_candidates}"
        )
    unreadable = _decode_source_images(split_dirs) if decode_images else "NOT_RUN"
    if isinstance(unreadable, int) and unreadable:
        raise ValueError(f"Raw source contains {unreadable} unreadable images")
    return {
        "split_dirs": split_dirs,
        "names": names,
        "target_id": target_id,
        "class_counts": class_counts,
        "totals": totals,
        "plate_candidates": plate_candidates,
        "unreadable": unreadable,
    }


def build_class_mapping(names: list[str], counts: Counter[int], target_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id, name in enumerate(names):
        source_count = counts[class_id]
        keep = class_id == target_id
        rows.append(
            {
                "source_class_id": class_id,
                "source_class_name": name,
                "source_box_count": source_count,
                "decision": "KEEP_AND_MAP" if keep else "REMOVE_NON_TARGET",
                "target_class_id": TARGET_CLASS_ID if keep else "",
                "target_class_name": TARGET_CLASS_NAME if keep else "",
                "boxes_kept": ACCEPTED_COUNTS["total"]["boxes"] if keep else 0,
                "boxes_removed": source_count - ACCEPTED_COUNTS["total"]["boxes"] if keep else source_count,
                "reason": (
                    "The exact source class plate is remapped to class 0; release membership is frozen by the active manifest."
                    if keep
                    else "OCR character, expiration, emirate and style annotations are outside the one-class full-plate target."
                ),
            }
        )
    return rows


def _validate_final_records(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts = split_counts(records)
    for record in records:
        boxes = list(record["boxes"])
        if not boxes:
            raise ValueError(f"Accepted release contains an empty label: {record['label_path']}")
        if any(box.class_id != TARGET_CLASS_ID for box in boxes):
            raise ValueError(f"Accepted release contains a nonzero class: {record['label_path']}")
    for split in (*SPLITS, "total"):
        observed = {"images": counts[split]["images"], "boxes": counts[split]["boxes"]}
        if observed != ACCEPTED_COUNTS[split]:
            raise ValueError(f"Accepted counts differ for {split}: {observed} != {ACCEPTED_COUNTS[split]}")
    return counts


def write_manifest(package_root: Path, records: list[dict[str, object]]) -> Path:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "split": record["split"],
                "image_relative_path": Path(record["image_path"]).relative_to(package_root).as_posix(),
                "label_relative_path": Path(record["label_path"]).relative_to(package_root).as_posix(),
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "image_size_bytes": record["image_size_bytes"],
                "label_size_bytes": record["label_size_bytes"],
                "box_count": record["box_count"],
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
            }
        )
    path = package_root / "reports" / "dataset_manifest.csv"
    write_csv(path, MANIFEST_COLUMNS, rows)
    return path


def generate_coco(package_root: Path, records: list[dict[str, object]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    image_id = 1
    annotation_id = 1
    for split in SPLITS:
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        for record in (item for item in records if item["split"] == split):
            width = int(record["image_width"])
            height = int(record["image_height"])
            images.append(
                {
                    "id": image_id,
                    "license": LICENSE_ID,
                    "file_name": f"images/{split}/{Path(record['image_path']).name}",
                    "width": width,
                    "height": height,
                }
            )
            for box in record["boxes"]:
                bbox = yolo_to_coco_bbox(box, width, height)
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": COCO_CATEGORY_ID,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                annotation_id += 1
            image_id += 1
        payload = {
            "info": {
                "description": "Frozen UAE full-plate release converted from accepted YOLO labels",
                "version": RELEASE_VERSION,
                "date_created": RELEASE_DATE,
                "source_url": SOURCE_URL,
                "license": LICENSE_NAME,
            },
            "licenses": [
                {"id": LICENSE_ID, "name": LICENSE_NAME, "url": "https://creativecommons.org/licenses/by/4.0/"}
            ],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": COCO_CATEGORY_ID, "name": TARGET_CLASS_NAME, "supercategory": TARGET_CLASS_NAME}
            ],
        }
        path = package_root / "annotations" / "coco" / f"{split}.json"
        write_json(path, payload)
        paths[split] = path
    return paths


def write_audit(package_root: Path, source_unreadable: int | str) -> Path:
    rows = [
        ("source_images", 9985, "Observed raw image files."),
        ("source_unreadable_images", source_unreadable, "Present-run full decode of the supplied raw export."),
        ("no_plate_images_removed", 359, "Raw labels containing no source plate class."),
        ("plate_only_candidate_images", 9626, "Raw images containing at least one source plate box."),
        ("release_decision_images_excluded", 16, "Project-owner decisions recorded in reports/excluded_images.csv."),
        ("final_images", 9610, "Frozen accepted release membership."),
        ("source_boxes", 86294, "Observed raw YOLO rows."),
        ("non_target_boxes_removed", 73826, "All source classes except the exact plate class."),
        ("source_plate_boxes", 12468, "Observed source class-50 plate boxes."),
        ("plate_boxes_excluded_with_release_decisions", 17, "Plate boxes belonging to the 16 excluded images."),
        ("final_plate_boxes", 12451, "Observed accepted class-0 boxes."),
        ("total_boxes_not_in_final_release", 73843, "Non-target rows plus plate boxes in excluded images."),
        ("final_unreadable_images", 0, "Present-run full decode of all accepted images."),
        ("final_orphan_images", 0, "Accepted image-to-label stem comparison."),
        ("final_orphan_labels", 0, "Accepted label-to-image stem comparison."),
    ]
    path = package_root / "reports" / "preprocessing_audit.csv"
    write_csv(path, ["metric", "value", "evidence"], ({"metric": key, "value": value, "evidence": evidence} for key, value, evidence in rows))
    return path


def write_release(package_root: Path, manifest_path: Path, coco_paths: dict[str, Path]) -> Path:
    release = {
        "release_name": "uae_lp_v2_yolo",
        "semantic_version": RELEASE_VERSION,
        "creation_date": RELEASE_DATE,
        "revision_date": RELEASE_DATE,
        "package_type": "github_source_without_images",
        "source_dataset_name": "UAE",
        "source_url": SOURCE_URL,
        "license": LICENSE_NAME,
        "target_class": {"id": TARGET_CLASS_ID, "name": TARGET_CLASS_NAME},
        "split_counts": {split: ACCEPTED_COUNTS[split]["images"] for split in SPLITS}
        | {"total": ACCEPTED_COUNTS["total"]["images"]},
        "box_counts": {split: ACCEPTED_COUNTS[split]["boxes"] for split in SPLITS}
        | {"total": ACCEPTED_COUNTS["total"]["boxes"]},
        "membership_policy": FROZEN_MEMBERSHIP,
        "manifest_sha256": sha256_file(manifest_path),
        "coco_annotation_sha256": {split: sha256_file(coco_paths[split]) for split in SPLITS},
        "augmentation_policy_filename": "configs/augmentation_policy.yaml",
        "augmentation_applied_offline": False,
        "crop_heavy_training_samples_retained": True,
        "proposed_models": ["YOLO", "RT-DETR", "RF-DETR"],
        "known_limitations": [
            "Images are intentionally omitted from the GitHub source repository and distributed separately.",
            "The exact historical reason for five frozen-release omissions is unavailable.",
            "The local raw export metadata used a different Roboflow workspace URL from the required canonical attribution URL.",
            "The augmentation policy is a preprocessing handoff and is not claimed as integrated into model training.",
            "The repository proposes YOLO, RT-DETR and RF-DETR but does not claim those models are implemented here.",
        ],
    }
    path = package_root / "dataset_release.json"
    write_json(path, release)
    return path


def audit_existing(package_root: Path, dataset_root: Path, source_root: Path | None) -> None:
    records = collect_dataset_records(dataset_root, decode_images=True, include_hashes=True)
    counts = _validate_final_records(records)
    manifest_paths = {Path(record["image_path"]).relative_to(package_root).as_posix() for record in records}
    validate_exclusions(package_root / "reports" / "excluded_images.csv", manifest_paths)
    if source_root is not None:
        source = validate_source(source_root, decode_images=True)
        names = list(source["names"])
        class_counts = Counter(source["class_counts"])
        target_id = int(source["target_id"])
        source_unreadable: int | str = int(source["unreadable"])
    else:
        existing = read_csv(package_root / "reports" / "class_mapping.csv")
        if len(existing) != 51:
            raise ValueError("Without --source-root, the committed class mapping must contain 51 rows")
        names = [row["source_class_name"] for row in existing]
        class_counts = Counter({int(row["source_class_id"]): int(row["source_box_count"]) for row in existing})
        target_id = _target_source_id(names)
        source_unreadable = "NOT_RUN_SOURCE_NOT_SUPPLIED"
    mapping_rows = build_class_mapping(names, class_counts, target_id)
    write_csv(package_root / "reports" / "class_mapping.csv", CLASS_MAPPING_COLUMNS, mapping_rows)
    manifest_path = write_manifest(package_root, records)
    coco_paths = generate_coco(package_root, records)
    audit_path = write_audit(package_root, source_unreadable)
    release_path = write_release(package_root, manifest_path, coco_paths)
    print(f"Audit PASS: {counts['total']['images']} images, {counts['total']['boxes']} boxes")
    print(f"Wrote {manifest_path.relative_to(package_root)}")
    print(f"Wrote {audit_path.relative_to(package_root)}")
    print(f"Wrote {release_path.relative_to(package_root)}")


def _source_index(source: dict[str, object]) -> dict[str, list[tuple[Path, Path]]]:
    index: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_dir, label_dir in source["split_dirs"].values():
        for image_path in find_image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Raw image has no matching label: {image_path}")
            index[image_path.name.casefold()].append((image_path, label_path))
    return index


def _boxes_equal(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if a.class_id != b.class_id:
            return False
        values_a = (a.x_center, a.y_center, a.width, a.height)
        values_b = (b.x_center, b.y_center, b.width, b.height)
        if any(not math.isclose(x, y, rel_tol=0.0, abs_tol=BOUND_TOLERANCE) for x, y in zip(values_a, values_b, strict=True)):
            return False
    return True


def _load_coco_index(package_root: Path) -> dict[tuple[str, str], tuple[dict, list[dict]]]:
    result: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for split in SPLITS:
        data = json.loads((package_root / "annotations" / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in data["annotations"]:
            by_image[int(annotation["image_id"])].append(annotation)
        for image in data["images"]:
            result[(split, Path(image["file_name"]).name.casefold())] = (
                image,
                sorted(by_image[int(image["id"])], key=lambda row: int(row["id"])),
            )
    return result


def build_from_raw(package_root: Path, dataset_root: Path, source_root: Path, staging_root: Path) -> None:
    manifest_path = package_root / "reports" / "dataset_manifest.csv"
    manifest = read_csv(manifest_path)  # Read before creating staging output.
    if len(manifest) != ACCEPTED_COUNTS["total"]["images"]:
        raise ValueError(f"Active manifest must contain {ACCEPTED_COUNTS['total']['images']} rows")
    if staging_root == dataset_root or dataset_root in staging_root.parents:
        raise ValueError("Staging output must not be the accepted dataset or one of its descendants")
    if staging_root.exists() and any(staging_root.iterdir()):
        raise FileExistsError(f"Staging directory must be new or empty: {staging_root}")
    source = validate_source(source_root, decode_images=False)
    index = _source_index(source)
    target_id = int(source["target_id"])
    coco_index = _load_coco_index(package_root)
    total_boxes = 0
    for row in manifest:
        split = row["split"]
        image_name = Path(row["image_relative_path"]).name
        matches = index.get(image_name.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f"Expected one raw match for {image_name}; found {len(matches)}")
        source_image, source_label = matches[0]
        raw_target = [box for box in load_yolo_labels(source_label, allow_empty=True) if box.class_id == target_id]
        if not raw_target:
            raise ValueError(f"Active manifest image has no source plate box: {image_name}")
        staged_image = staging_root / "images" / split / image_name
        staged_label = staging_root / "labels" / split / Path(row["label_relative_path"]).name
        staged_image.parent.mkdir(parents=True, exist_ok=True)
        staged_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, staged_image)
        lines = [
            f"0 {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
            for box in raw_target
        ]
        staged_label.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        accepted_label = dataset_root / "labels" / split / staged_label.name
        accepted_boxes = load_yolo_labels(accepted_label, allow_empty=False)
        staged_boxes = load_yolo_labels(staged_label, allow_empty=False)
        if not _boxes_equal(staged_boxes, accepted_boxes):
            raise ValueError(f"Reconstructed label differs beyond normalized tolerance: {image_name}")
        if sha256_file(staged_image) != row["image_sha256"]:
            raise ValueError(f"Reconstructed image SHA-256 differs: {image_name}")
        if sha256_file(accepted_label) != row["label_sha256"]:
            raise ValueError(f"Committed label SHA-256 differs from manifest: {accepted_label.name}")
        if sha256_file(staged_label) != row["label_sha256"]:
            raise ValueError(f"Reconstructed label bytes differ from the committed normalized form: {image_name}")
        image, annotations = coco_index[(split, image_name.casefold())]
        if int(image["width"]) != int(row["image_width"]) or int(image["height"]) != int(row["image_height"]):
            raise ValueError(f"COCO dimensions differ from the manifest: {image_name}")
        if len(annotations) != len(staged_boxes):
            raise ValueError(f"COCO box count differs from reconstructed YOLO: {image_name}")
        for box, annotation in zip(staged_boxes, annotations, strict=True):
            expected = yolo_to_coco_bbox(box, int(row["image_width"]), int(row["image_height"]))
            if any(abs(float(a) - float(b)) > COCO_PARITY_TOLERANCE_PX for a, b in zip(expected, annotation["bbox"], strict=True)):
                raise ValueError(f"YOLO-COCO parity differs by more than 0.01 pixel: {image_name}")
        total_boxes += len(staged_boxes)
    if total_boxes != ACCEPTED_COUNTS["total"]["boxes"]:
        raise ValueError(f"Staged box total differs: {total_boxes}")
    (staging_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: license_plate\n",
        encoding="utf-8",
        newline="\n",
    )
    for split in SPLITS:
        images = find_image_files(staging_root / "images" / split)
        labels = find_label_files(staging_root / "labels" / split)
        if len(images) != ACCEPTED_COUNTS[split]["images"] or len(labels) != ACCEPTED_COUNTS[split]["images"]:
            raise ValueError(f"Staged membership count differs for {split}")
    print(
        "Raw reconstruction PASS: staged output exactly reproduces "
        f"{len(manifest)} images and {total_boxes} boxes without replacing the accepted dataset."
    )
    print(f"Staging root: {staging_root}")


def main() -> None:
    args = parse_args()
    invocation_root = Path.cwd().resolve()
    package_root = _resolve(invocation_root, args.package_root)
    dataset_root = _resolve(package_root, args.dataset_root)
    source_root = _resolve(invocation_root, args.source_root)
    staging_root = _resolve(invocation_root, args.staging_root)
    assert package_root is not None and dataset_root is not None
    if args.audit_existing:
        audit_existing(package_root, dataset_root, source_root)
        return
    if source_root is None:
        raise ValueError("--build-from-raw requires --source-root")
    if staging_root is None:
        raise ValueError("--build-from-raw requires --staging-root")
    build_from_raw(package_root, dataset_root, source_root, staging_root)


if __name__ == "__main__":
    main()
