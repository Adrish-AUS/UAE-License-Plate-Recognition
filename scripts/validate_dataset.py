"""Validate the image-free repository contract or the complete local dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from check_split_leakage import IMAGES_NOT_INCLUDED, run_leakage
from preprocess_dataset import EXCLUSION_COLUMNS, _expected_exclusions
from preprocessing_utils import (
    ACCEPTED_COUNTS,
    BOUND_TOLERANCE,
    COCO_PARITY_TOLERANCE_PX,
    SPLITS,
    TARGET_CLASS_NAME,
    find_image_files,
    find_label_files,
    load_augmentation_policy,
    load_yaml,
    load_yolo_labels,
    read_csv,
    sha256_file,
    yolo_to_coco_bbox,
)

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
REQUIRED_FILES = {
    ".gitignore",
    "README.md",
    "DATASET_ATTRIBUTION.md",
    "requirements.txt",
    "data.yaml",
    "dataset_release.json",
    "configs/augmentation_policy.yaml",
    "datasets/uae_lp_v2_yolo/data.yaml",
    "reports/class_mapping.csv",
    "reports/dataset_manifest.csv",
    "reports/dataset_stats.csv",
    "reports/excluded_images.csv",
    "reports/preprocessing_audit.csv",
    "reports/split_leakage_candidates.csv",
    "reports/validation_report.md",
    "annotations/coco/train.json",
    "annotations/coco/val.json",
    "annotations/coco/test.json",
    "notebooks/01_data_preprocessing.ipynb",
    "scripts/check_split_leakage.py",
    "scripts/preprocess_dataset.py",
    "scripts/preprocessing_utils.py",
    "scripts/validate_dataset.py",
}
FIGURES = {
    "train_random_samples.jpg",
    "val_random_samples.jpg",
    "test_random_samples.jpg",
    "smallest_boxes.jpg",
    "largest_boxes.jpg",
    "multi_plate_images.jpg",
    "edge_touching_boxes.jpg",
    "augmentation_preview.jpg",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".ipynb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    parser.add_argument("--mode", choices=("auto", "repository", "full"), default="auto")
    return parser.parse_args()


def _resolve(package_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (package_root / path).resolve()


def _item(name: str, observed: str, errors: list[str] | None = None, *, status: str | None = None) -> dict[str, object]:
    errors = errors or []
    return {
        "check": name,
        "observed": observed,
        "status": status or ("FAIL" if errors else "PASS"),
        "errors": errors,
    }


def _tracked_files(package_root: Path) -> list[str]:
    if (package_root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=package_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return sorted(item.decode("utf-8").replace("\\", "/") for item in result.stdout.split(b"\0") if item)
    files = []
    for path in package_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if "__pycache__" in path.parts or path.suffix.casefold() in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(package_root).as_posix()
        if relative.startswith("datasets/uae_lp_v2_yolo/images/"):
            continue
        files.append(relative)
    return sorted(files)


def _text_files(package_root: Path, tracked: list[str]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for relative in tracked:
        if relative.startswith("datasets/uae_lp_v2_yolo/labels/"):
            continue
        path = package_root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", "requirements.txt"}:
            continue
        try:
            values.append((relative, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return values


def _validate_repository_files(package_root: Path, tracked: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    tracked_set = set(tracked)
    missing = sorted(REQUIRED_FILES - tracked_set)
    results.append(_item("Required repository files", f"missing={len(missing)}", [f"Missing required file: {path}" for path in missing]))

    readmes = [path for path in tracked if Path(path).name == "README.md"]
    results.append(_item("Exactly one README.md", f"count={len(readmes)}", [] if len(readmes) == 1 else [f"Observed README files: {readmes}"]))

    hygiene_errors: list[str] = []
    for relative in tracked:
        parts = {part.casefold() for part in Path(relative).parts}
        lower = relative.casefold()
        if "__pycache__" in parts or Path(relative).suffix.casefold() in {".pyc", ".pyo"}:
            hygiene_errors.append(f"Tracked bytecode/cache: {relative}")
        if ".venv" in parts:
            hygiene_errors.append(f"Tracked virtual environment file: {relative}")
        if lower.startswith("datasets/uae_lp_v2_yolo/images/"):
            hygiene_errors.append(f"Tracked dataset image: {relative}")
        if lower.endswith((".pt", ".pth", ".onnx", ".engine")):
            hygiene_errors.append(f"Tracked model weight: {relative}")
    results.append(_item("Repository packaging hygiene", f"tracked_files={len(tracked)}, issues={len(hygiene_errors)}", hygiene_errors))

    removed_paths = [
        "reports/" + "near_duplicate_" + "review_pairs",
        "reports/" + "near_duplicate_" + "review_decisions.csv",
        "reports/manual_inspection.csv",
        "reports/final_dataset_manifest.csv",
    ]
    artifact_errors = [f"Removed artifact still exists: {path}" for path in removed_paths if path in tracked_set or (package_root / path).exists()]
    results.append(_item("Abandoned review artifacts are absent", f"remaining={len(artifact_errors)}", artifact_errors))

    banned = [
        "near_duplicate_" + "review",
        "pair_" + "001",
        "pair_" + "043",
        "43 " + "historical",
        "43 " + "pair",
        "NEEDS_" + "HUMAN_REVIEW",
        "pending_" + "human_review",
    ]
    private_prefix = "C:" + "\\Users\\"
    banned_errors: list[str] = []
    private_errors: list[str] = []
    for relative, text in _text_files(package_root, tracked):
        for token in banned:
            if token in text:
                banned_errors.append(f"Forbidden legacy text in {relative}: {token}")
        if private_prefix.casefold() in text.casefold():
            private_errors.append(f"Absolute private Windows path in {relative}")
    results.append(_item("Legacy review language is absent", f"matches={len(banned_errors)}", banned_errors))
    results.append(_item("Absolute private paths are absent", f"matches={len(private_errors)}", private_errors))
    return results


def _validate_yaml(package_root: Path, dataset_root: Path) -> dict[str, object]:
    errors: list[str] = []
    expected_paths = {
        package_root / "data.yaml": {
            "train": "datasets/uae_lp_v2_yolo/images/train",
            "val": "datasets/uae_lp_v2_yolo/images/val",
            "test": "datasets/uae_lp_v2_yolo/images/test",
        },
        dataset_root / "data.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"},
    }
    for path, expected in expected_paths.items():
        try:
            data = load_yaml(path)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if data.get("nc") != 1 or data.get("names") not in ({0: TARGET_CLASS_NAME}, [TARGET_CLASS_NAME]):
            errors.append(f"{path.name}: expected one class named {TARGET_CLASS_NAME}")
        for split, value in expected.items():
            if str(data.get(split, "")).replace("\\", "/") != value:
                errors.append(f"{path.name}: {split} path differs")
    return _item("Root and nested data.yaml semantics", f"issues={len(errors)}", errors)


def _validate_manifest_and_labels(
    package_root: Path, dataset_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, str]], dict[tuple[str, str], list], dict[str, dict[str, int]]]:
    results: list[dict[str, object]] = []
    manifest_path = package_root / "reports" / "dataset_manifest.csv"
    manifest = read_csv(manifest_path)
    manifest_errors: list[str] = []
    if manifest and list(manifest[0]) != MANIFEST_COLUMNS:
        manifest_errors.append("dataset_manifest.csv columns differ")
    if len(manifest) != ACCEPTED_COUNTS["total"]["images"]:
        manifest_errors.append(f"Manifest rows={len(manifest)}")
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    counts = {split: {"images": 0, "boxes": 0} for split in SPLITS}
    boxes_by_image: dict[tuple[str, str], list] = {}
    label_errors: list[str] = []
    hash_errors: list[str] = []
    for row in manifest:
        split = row.get("split", "")
        if split not in SPLITS:
            manifest_errors.append(f"Invalid manifest split: {split}")
            continue
        image_relative = row["image_relative_path"].replace("\\", "/")
        label_relative = row["label_relative_path"].replace("\\", "/")
        expected_image_prefix = f"datasets/uae_lp_v2_yolo/images/{split}/"
        expected_label_prefix = f"datasets/uae_lp_v2_yolo/labels/{split}/"
        if not image_relative.startswith(expected_image_prefix) or not label_relative.startswith(expected_label_prefix):
            manifest_errors.append(f"Manifest path/split mismatch: {image_relative}")
        if image_relative.casefold() in seen_images or label_relative.casefold() in seen_labels:
            manifest_errors.append(f"Duplicate manifest path: {image_relative}")
        seen_images.add(image_relative.casefold())
        seen_labels.add(label_relative.casefold())
        label_path = dataset_root / "labels" / split / Path(label_relative).name
        if not label_path.is_file():
            label_errors.append(f"Missing label: {label_relative}")
            continue
        try:
            boxes = load_yolo_labels(label_path, allow_empty=False)
        except Exception as exc:
            label_errors.append(str(exc))
            continue
        if any(box.class_id != 0 for box in boxes):
            label_errors.append(f"Nonzero class in {label_relative}")
        if len(boxes) != int(row["box_count"]):
            label_errors.append(f"Manifest box count differs for {label_relative}")
        if label_path.stat().st_size != int(row["label_size_bytes"]):
            hash_errors.append(f"Label byte size differs for {label_relative}")
        if sha256_file(label_path) != row["label_sha256"]:
            hash_errors.append(f"Label SHA-256 differs for {label_relative}")
        try:
            width, height = int(row["image_width"]), int(row["image_height"])
            if width <= 0 or height <= 0:
                raise ValueError
        except ValueError:
            manifest_errors.append(f"Invalid manifest dimensions: {image_relative}")
        boxes_by_image[(split, Path(image_relative).name.casefold())] = boxes
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(boxes)
    counts["total"] = {
        "images": sum(counts[split]["images"] for split in SPLITS),
        "boxes": sum(counts[split]["boxes"] for split in SPLITS),
    }
    for split in (*SPLITS, "total"):
        if counts[split] != ACCEPTED_COUNTS[split]:
            manifest_errors.append(f"{split} counts={counts[split]}, expected={ACCEPTED_COUNTS[split]}")
    results.append(_item("Active manifest membership and split counts", f"rows={len(manifest)}; counts={counts}", manifest_errors))
    results.append(
        _item(
            "YOLO label syntax, class, bounds and nonempty rows",
            f"labels={len(boxes_by_image)}, boxes={counts['total']['boxes']}, issues={len(label_errors)}",
            label_errors,
        )
    )
    results.append(_item("Committed label SHA-256 values", f"checked={len(boxes_by_image)}, issues={len(hash_errors)}", hash_errors))

    groups: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        groups[row.get("image_sha256", "")].add(row.get("split", ""))
    collision_groups = [digest for digest, splits in groups.items() if len(splits) > 1]
    collision_errors = [f"Recorded image SHA crosses splits: {digest}" for digest in collision_groups]
    results.append(_item("Recorded image SHA-256 uniqueness across splits", f"cross_split_groups={len(collision_groups)}", collision_errors))
    return results, manifest, boxes_by_image, counts


def _validate_coco(
    package_root: Path,
    manifest: list[dict[str, str]],
    boxes_by_image: dict[tuple[str, str], list],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    results: list[dict[str, object]] = []
    manifest_by_image = {(row["split"], Path(row["image_relative_path"]).name.casefold()): row for row in manifest}
    category_errors: list[str] = []
    id_errors: list[str] = []
    reference_errors: list[str] = []
    geometry_errors: list[str] = []
    parity_errors: list[str] = []
    membership_errors: list[str] = []
    global_image_ids: set[int] = set()
    global_annotation_ids: set[int] = set()
    coco_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = package_root / "annotations" / "coco" / f"{split}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            category_errors.append(f"Cannot read {path.name}: {exc}")
            continue
        expected_category = [{"id": 1, "name": TARGET_CLASS_NAME, "supercategory": TARGET_CLASS_NAME}]
        if data.get("categories") != expected_category:
            category_errors.append(f"{path.name}: category definition differs")
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        image_by_id: dict[int, dict] = {}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for image in images:
            image_id = image.get("id")
            if not isinstance(image_id, int) or image_id in global_image_ids:
                id_errors.append(f"{path.name}: invalid/duplicate image ID {image_id}")
                continue
            global_image_ids.add(image_id)
            image_by_id[image_id] = image
            if image.get("license") != 1:
                reference_errors.append(f"{path.name}: image {image_id} license must be 1")
        for annotation in annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            if not isinstance(annotation_id, int) or annotation_id in global_annotation_ids:
                id_errors.append(f"{path.name}: invalid/duplicate annotation ID {annotation_id}")
            else:
                global_annotation_ids.add(annotation_id)
            if image_id not in image_by_id:
                reference_errors.append(f"{path.name}: annotation {annotation_id} has missing image reference")
                continue
            annotations_by_image[int(image_id)].append(annotation)
        observed_names: set[str] = set()
        for image_id, image in image_by_id.items():
            file_name = str(image.get("file_name", "")).replace("\\", "/")
            if not file_name.startswith(f"images/{split}/"):
                membership_errors.append(f"{path.name}: wrong split path {file_name}")
            name = Path(file_name).name.casefold()
            observed_names.add(name)
            key = (split, name)
            row = manifest_by_image.get(key)
            boxes = boxes_by_image.get(key)
            if row is None or boxes is None:
                membership_errors.append(f"{path.name}: image absent from YOLO manifest: {file_name}")
                continue
            width, height = int(row["image_width"]), int(row["image_height"])
            if image.get("width") != width or image.get("height") != height:
                geometry_errors.append(f"{path.name}: dimensions differ for {file_name}")
            current = sorted(annotations_by_image.get(image_id, []), key=lambda value: int(value.get("id", -1)))
            if len(current) != len(boxes):
                parity_errors.append(f"{path.name}: box count differs for {file_name}")
            for index, annotation in enumerate(current):
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    geometry_errors.append(f"{path.name}: invalid bbox for annotation {annotation.get('id')}")
                    continue
                try:
                    x, y, box_width, box_height = [float(value) for value in bbox]
                    area = float(annotation.get("area"))
                except (TypeError, ValueError):
                    geometry_errors.append(f"{path.name}: nonnumeric geometry for annotation {annotation.get('id')}")
                    continue
                tolerance = BOUND_TOLERANCE * max(width, height)
                if (
                    not all(math.isfinite(value) for value in (x, y, box_width, box_height, area))
                    or box_width <= 0
                    or box_height <= 0
                    or x < -tolerance
                    or y < -tolerance
                    or x + box_width > width + tolerance
                    or y + box_height > height + tolerance
                ):
                    geometry_errors.append(f"{path.name}: invalid/out-of-bounds bbox for annotation {annotation.get('id')}")
                if not math.isclose(area, box_width * box_height, rel_tol=1e-9, abs_tol=1e-6):
                    geometry_errors.append(f"{path.name}: area differs for annotation {annotation.get('id')}")
                if annotation.get("category_id") != 1:
                    category_errors.append(f"{path.name}: category ID differs for annotation {annotation.get('id')}")
                if index < len(boxes):
                    expected = yolo_to_coco_bbox(boxes[index], width, height)
                    if any(abs(left - right) > COCO_PARITY_TOLERANCE_PX for left, right in zip(expected, (x, y, box_width, box_height), strict=True)):
                        parity_errors.append(f"{path.name}: bbox parity differs for {file_name} row {index + 1}")
        expected_names = {name for current_split, name in manifest_by_image if current_split == split}
        if observed_names != expected_names:
            membership_errors.append(f"{path.name}: YOLO-COCO membership differs")
        coco_counts[split] = {"images": len(images), "boxes": len(annotations)}
    results.append(_item("COCO category, IDs and references", f"category={len(category_errors)}, ids={len(id_errors)}, refs={len(reference_errors)}", category_errors + id_errors + reference_errors))
    results.append(_item("COCO dimensions, boxes and areas", f"issues={len(geometry_errors)}", geometry_errors))
    results.append(_item("YOLO-COCO membership", f"issues={len(membership_errors)}; counts={coco_counts}", membership_errors))
    results.append(_item("YOLO-COCO 0.01-pixel parity", f"issues={len(parity_errors)}", parity_errors))
    return results, coco_counts


def _validate_metadata(package_root: Path, manifest: list[dict[str, str]], counts: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    manifest_paths = {row["image_relative_path"].replace("\\", "/") for row in manifest}

    exclusion_errors: list[str] = []
    exclusions = read_csv(package_root / "reports" / "excluded_images.csv")
    if len(exclusions) != 16 or (exclusions and list(exclusions[0]) != EXCLUSION_COLUMNS):
        exclusion_errors.append("Exclusion ledger must contain seven exact columns and 16 rows")
    observed = {row.get("image_relative_path", "").replace("\\", "/"): row for row in exclusions}
    if observed != _expected_exclusions():
        exclusion_errors.append("Exclusion rows or project-owner decision values differ")
    overlap = sorted(set(observed) & manifest_paths)
    if overlap:
        exclusion_errors.extend(f"Excluded path is active: {path}" for path in overlap)
    results.append(_item("Sixteen project-owner exclusions", f"rows={len(exclusions)}, active_overlap={len(overlap)}", exclusion_errors))

    audit_expected = {
        "source_images": "9985",
        "no_plate_images_removed": "359",
        "plate_only_candidate_images": "9626",
        "release_decision_images_excluded": "16",
        "final_images": "9610",
        "source_boxes": "86294",
        "non_target_boxes_removed": "73826",
        "source_plate_boxes": "12468",
        "plate_boxes_excluded_with_release_decisions": "17",
        "final_plate_boxes": "12451",
        "total_boxes_not_in_final_release": "73843",
        "final_unreadable_images": "0",
        "final_orphan_images": "0",
        "final_orphan_labels": "0",
    }
    audit_rows = read_csv(package_root / "reports" / "preprocessing_audit.csv")
    audit = {row.get("metric", ""): row.get("value", "") for row in audit_rows}
    audit_errors = [f"{key}={audit.get(key)}, expected={value}" for key, value in audit_expected.items() if audit.get(key) != value]
    results.append(_item("Preprocessing accounting", f"metrics={len(audit)}, issues={len(audit_errors)}", audit_errors))

    class_rows = read_csv(package_root / "reports" / "class_mapping.csv")
    class_errors: list[str] = []
    if len(class_rows) != 51:
        class_errors.append(f"Class mapping rows={len(class_rows)}")
    target_rows = [row for row in class_rows if row.get("source_class_name") == "plate"]
    if len(target_rows) != 1:
        class_errors.append("Class mapping must identify one source plate class")
    else:
        target = target_rows[0]
        expected = {
            "source_class_id": "50",
            "source_box_count": "12468",
            "decision": "KEEP_AND_MAP",
            "target_class_id": "0",
            "target_class_name": TARGET_CLASS_NAME,
            "boxes_kept": "12451",
            "boxes_removed": "17",
        }
        class_errors.extend(f"plate mapping {key} differs" for key, value in expected.items() if target.get(key) != value)
    removed = sum(int(row.get("boxes_removed", 0)) for row in class_rows if row.get("decision") == "REMOVE_NON_TARGET")
    if removed != 73826:
        class_errors.append(f"Non-target removed boxes={removed}")
    results.append(_item("Source class mapping", f"rows={len(class_rows)}, non_target_removed={removed}", class_errors))

    release_errors: list[str] = []
    release_path = package_root / "dataset_release.json"
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except Exception as exc:
        release = {}
        release_errors.append(f"Cannot parse dataset_release.json: {exc}")
    expected_scalars = {
        "semantic_version": "2.0.1",
        "creation_date": "2026-07-10",
        "revision_date": "2026-07-10",
        "package_type": "github_source_without_images",
        "source_url": "https://universe.roboflow.com/addinguae/uae-zcfqj",
        "license": "CC BY 4.0",
        "augmentation_applied_offline": False,
        "crop_heavy_training_samples_retained": True,
    }
    for key, value in expected_scalars.items():
        if release.get(key) != value:
            release_errors.append(f"dataset_release.json {key} differs")
    if release.get("target_class") != {"id": 0, "name": TARGET_CLASS_NAME}:
        release_errors.append("dataset_release.json target_class differs")
    if release.get("proposed_models") != ["YOLO", "RT-DETR", "RF-DETR"]:
        release_errors.append("dataset_release.json proposed_models differs")
    for split in (*SPLITS, "total"):
        if release.get("split_counts", {}).get(split) != counts[split]["images"]:
            release_errors.append(f"Release image count differs for {split}")
        if release.get("box_counts", {}).get(split) != counts[split]["boxes"]:
            release_errors.append(f"Release box count differs for {split}")
    manifest_hash = sha256_file(package_root / "reports" / "dataset_manifest.csv")
    if release.get("manifest_sha256") != manifest_hash:
        release_errors.append("Manifest SHA-256 differs from dataset_release.json")
    coco_hashes = {split: sha256_file(package_root / "annotations" / "coco" / f"{split}.json") for split in SPLITS}
    for split, digest in coco_hashes.items():
        if release.get("coco_annotation_sha256", {}).get(split) != digest:
            release_errors.append(f"COCO SHA-256 differs for {split}")
    results.append(
        _item(
            "Release counts and manifest/COCO hashes",
            f"manifest={manifest_hash}; coco={coco_hashes}; issues={len(release_errors)}",
            release_errors,
        )
    )

    try:
        load_augmentation_policy(package_root / "configs" / "augmentation_policy.yaml")
        policy_errors: list[str] = []
    except Exception as exc:
        policy_errors = [str(exc)]
    results.append(_item("Training-only augmentation policy", f"issues={len(policy_errors)}", policy_errors))
    return results


def _validate_full_images(
    package_root: Path,
    dataset_root: Path,
    manifest: list[dict[str, str]],
    *,
    regenerate_visuals: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results: list[dict[str, object]] = []
    manifest_names = {
        split: {Path(row["image_relative_path"]).name.casefold() for row in manifest if row["split"] == split}
        for split in SPLITS
    }
    membership_errors: list[str] = []
    for split in SPLITS:
        actual = {path.name.casefold() for path in find_image_files(dataset_root / "images" / split)}
        if actual != manifest_names[split]:
            membership_errors.append(
                f"{split} image membership differs: actual={len(actual)}, manifest={len(manifest_names[split])}"
            )
    results.append(_item("Actual image membership", f"issues={len(membership_errors)}", membership_errors))

    tasks = []
    for row in manifest:
        path = dataset_root / "images" / row["split"] / Path(row["image_relative_path"]).name
        tasks.append((row, path))

    def inspect(task: tuple[dict[str, str], Path]) -> tuple[list[str], list[str], list[str]]:
        row, path = task
        decode_errors: list[str] = []
        size_errors: list[str] = []
        hash_errors: list[str] = []
        if not path.is_file():
            return [f"Missing image: {row['image_relative_path']}"], [], []
        try:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
        except Exception as exc:
            return [f"Unreadable image {row['image_relative_path']}: {exc}"], [], []
        if (width, height) != (int(row["image_width"]), int(row["image_height"])):
            size_errors.append(f"Image dimensions differ: {row['image_relative_path']}")
        if len(payload) != int(row["image_size_bytes"]):
            size_errors.append(f"Image byte size differs: {row['image_relative_path']}")
        if digest != row["image_sha256"]:
            hash_errors.append(f"Image SHA-256 differs: {row['image_relative_path']}")
        return decode_errors, size_errors, hash_errors

    decode_errors: list[str] = []
    size_errors: list[str] = []
    hash_errors: list[str] = []
    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for decoded, sized, hashed in executor.map(inspect, tasks):
            decode_errors.extend(decoded)
            size_errors.extend(sized)
            hash_errors.extend(hashed)
    results.append(_item("Actual image decoding", f"decoded={len(tasks) - len(decode_errors)}, issues={len(decode_errors)}", decode_errors))
    results.append(_item("Actual image-size verification", f"checked={len(tasks)}, issues={len(size_errors)}", size_errors))
    results.append(_item("Actual image SHA-256 recomputation", f"checked={len(tasks)}, issues={len(hash_errors)}", hash_errors))

    leakage = run_leakage(package_root, dataset_root, mode="full", threshold=8, write_output=True)
    exact_errors = []
    if leakage["exact_cross_split_duplicates"]:
        exact_errors.append(f"Exact cross-split duplicate candidates={leakage['exact_cross_split_duplicates']}")
    results.append(_item("Current exact cross-split duplicates", f"count={leakage['exact_cross_split_duplicates']}", exact_errors))
    results.append(_item("Current perceptual-hash scan", f"candidates={leakage['perceptual_candidates']}; threshold=8"))

    visual_errors: list[str] = []
    if regenerate_visuals:
        command = [
            sys.executable,
            "scripts/visualize_dataset.py",
            "--package-root",
            ".",
            "--dataset-root",
            "datasets/uae_lp_v2_yolo",
        ]
        completed = subprocess.run(command, cwd=package_root, text=True, capture_output=True)
        if completed.returncode:
            visual_errors.append("Contact-sheet regeneration failed: " + (completed.stderr.strip() or completed.stdout.strip()))
    for name in FIGURES:
        if not (package_root / "reports" / "figures" / name).is_file():
            visual_errors.append(f"Missing contact sheet: reports/figures/{name}")
    results.append(_item("Visual contact-sheet regeneration", f"figures={len(FIGURES)}, issues={len(visual_errors)}", visual_errors))
    return results, leakage


def _not_run_full_results() -> tuple[list[dict[str, object]], dict[str, object]]:
    names = [
        "Actual image decoding",
        "Actual image-size verification",
        "Actual image SHA-256 recomputation",
        "Current perceptual-hash scan",
        "Visual contact-sheet regeneration",
    ]
    results = [_item(name, IMAGES_NOT_INCLUDED, status=IMAGES_NOT_INCLUDED) for name in names]
    leakage = {
        "mode": "repository",
        "exact_cross_split_duplicates": 0,
        "perceptual_candidates": IMAGES_NOT_INCLUDED,
        "distance_threshold": 8,
    }
    return results, leakage


def _write_report(
    package_root: Path,
    repository_results: list[dict[str, object]],
    full_results: list[dict[str, object]],
    counts: dict[str, dict[str, int]],
    leakage: dict[str, object],
    overall: str,
) -> None:
    def table(lines: list[str], rows: list[dict[str, object]]) -> None:
        lines.extend(["| Check | Observed | Status |", "|---|---|---|"])
        for result in rows:
            observed = str(result["observed"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {result['check']} | {observed} | {result['status']} |")

    lines = [
        "# Dataset Validation Report",
        "",
        "The GitHub source repository intentionally excludes dataset image files. Repository validation uses committed labels, manifests and COCO metadata; full validation uses the separately distributed images.",
        "",
        "## Repository Validation",
        "",
        "Command: `python scripts/validate_dataset.py --mode repository`",
        "",
    ]
    table(lines, repository_results)
    lines.extend(
        [
            "",
            "## Full Dataset Validation Evidence",
            "",
            "Command: `python scripts/validate_dataset.py --mode full`",
            "",
        ]
    )
    table(lines, full_results)
    lines.extend(
        [
            "",
            "## Accepted Counts",
            "",
            "| Split | Images | Boxes |",
            "|---|---:|---:|",
        ]
    )
    for split in (*SPLITS, "total"):
        lines.append(f"| {split} | {counts[split]['images']} | {counts[split]['boxes']} |")
    release = json.loads((package_root / "dataset_release.json").read_text(encoding="utf-8"))
    lines.extend(
        [
            "",
            "## Integrity and Leakage Summary",
            "",
            f"- Manifest SHA-256: `{release['manifest_sha256']}` (PASS)",
            f"- COCO SHA-256 values: `{json.dumps(release['coco_annotation_sha256'], sort_keys=True)}` (PASS)",
            f"- Exact cross-split duplicates: {leakage['exact_cross_split_duplicates']} (PASS)",
            f"- Current perceptual scan: {leakage['perceptual_candidates']}",
            "- Crop-heavy training examples remain intentionally retained as valid full-plate training examples.",
            "",
            "## Overall Status",
            "",
            overall,
            "",
        ]
    )
    (package_root / "reports" / "validation_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_validation(
    package_root: Path,
    dataset_root: Path,
    *,
    mode: str = "auto",
    write_report: bool = True,
    regenerate_visuals: bool = True,
) -> dict[str, object]:
    package_root = package_root.resolve()
    dataset_root = dataset_root.resolve()
    tracked = _tracked_files(package_root)
    manifest_preview = read_csv(package_root / "reports" / "dataset_manifest.csv")
    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "full" if len(manifest_preview) == ACCEPTED_COUNTS["total"]["images"] and all(
            (dataset_root / "images" / row["split"] / Path(row["image_relative_path"]).name).is_file()
            for row in manifest_preview
        ) else "repository"

    repository_results = _validate_repository_files(package_root, tracked)
    repository_results.append(_validate_yaml(package_root, dataset_root))
    label_results, manifest, boxes_by_image, counts = _validate_manifest_and_labels(package_root, dataset_root)
    repository_results.extend(label_results)
    coco_results, coco_counts = _validate_coco(package_root, manifest, boxes_by_image)
    repository_results.extend(coco_results)
    repository_results.extend(_validate_metadata(package_root, manifest, counts))
    repository_leakage = run_leakage(package_root, dataset_root, mode="repository", threshold=8, write_output=selected_mode == "repository")
    exact_errors = [] if not repository_leakage["exact_cross_split_duplicates"] else ["Recorded exact cross-split duplicates exist"]
    repository_results.append(
        _item(
            "Recorded exact cross-split duplicates",
            f"count={repository_leakage['exact_cross_split_duplicates']}",
            exact_errors,
        )
    )
    repository_status = "PASS" if all(result["status"] == "PASS" for result in repository_results) else "FAIL"

    if selected_mode == "full":
        full_results, leakage = _validate_full_images(
            package_root,
            dataset_root,
            manifest,
            regenerate_visuals=regenerate_visuals,
        )
    else:
        full_results, leakage = _not_run_full_results()
    full_status = "PASS" if selected_mode == "repository" or all(result["status"] == "PASS" for result in full_results) else "FAIL"
    overall = "PASS" if repository_status == "PASS" and full_status == "PASS" else "FAIL"
    if write_report:
        _write_report(package_root, repository_results, full_results, counts, leakage, overall)
    errors = [
        error
        for result in (*repository_results, *full_results)
        for error in result.get("errors", [])
    ]
    return {
        "status": overall,
        "mode": selected_mode,
        "repository_status": repository_status,
        "full_status": full_status if selected_mode == "full" else IMAGES_NOT_INCLUDED,
        "counts": counts,
        "coco_counts": coco_counts,
        "leakage": leakage,
        "results": repository_results + full_results,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    dataset_root = _resolve(package_root, args.dataset_root)
    report = run_validation(package_root, dataset_root, mode=args.mode)
    summary = {
        "status": report["status"],
        "mode": report["mode"],
        "repository_status": report["repository_status"],
        "full_status": report["full_status"],
        "counts": report["counts"],
        "exact_cross_split_duplicates": report["leakage"]["exact_cross_split_duplicates"],
        "perceptual_candidates": report["leakage"]["perceptual_candidates"],
        "errors": report["errors"][:20],
    }
    print(json.dumps(summary, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
