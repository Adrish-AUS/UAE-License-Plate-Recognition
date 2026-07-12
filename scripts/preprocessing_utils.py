"""Shared, deterministic preprocessing helpers for the UAE plate handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import yaml
from PIL import Image, ImageDraw, ImageFont

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "license_plate"
COCO_CATEGORY_ID = 1
LICENSE_ID = 1
LICENSE_NAME = "CC BY 4.0"
SOURCE_URL = "https://universe.roboflow.com/addinguae/uae-zcfqj"
BOUND_TOLERANCE = 1e-6
COCO_PARITY_TOLERANCE_PX = 0.01
VISUALIZATION_SEED = 486
ACCEPTED_COUNTS = {
    "train": {"images": 6738, "boxes": 9415},
    "val": {"images": 1440, "boxes": 1525},
    "test": {"images": 1432, "boxes": 1511},
    "total": {"images": 9610, "boxes": 12451},
}


@dataclass(frozen=True)
class YoloBox:
    """One validated normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def normalized_xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.x_center + self.width / 2,
            self.y_center + self.height / 2,
        )


def find_image_files(directory: Path) -> list[Path]:
    """Return image files in stable, case-insensitive filename order."""

    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (path.name.casefold(), path.name),
    )


def find_label_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.txt"), key=lambda path: (path.name.casefold(), path.name))


def required_dataset_directories(dataset_root: Path) -> list[Path]:
    return [dataset_root / kind / split for kind in ("images", "labels") for split in SPLITS]


def require_dataset_layout(dataset_root: Path) -> None:
    missing = [path for path in required_dataset_directories(dataset_root) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required dataset directories: " + ", ".join(str(path) for path in missing))


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def read_class_names(data_yaml: Path) -> list[str]:
    """Read class names and IDs from an actual YOLO data.yaml."""

    data = load_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, list):
        result = [str(value) for value in names]
    elif isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda value: int(value))
            if [int(key) for key in keys] != list(range(len(keys))):
                raise ValueError("Class IDs must be sequential from zero")
            result = [str(names[key]) for key in keys]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid names mapping in {data_yaml}: {exc}") from exc
    else:
        raise ValueError(f"Missing or invalid names in {data_yaml}")
    nc = data.get("nc")
    if nc is not None and int(nc) != len(result):
        raise ValueError(f"nc={nc} does not match {len(result)} names in {data_yaml}")
    if not result:
        raise ValueError(f"No class names were found in {data_yaml}")
    return result


def discover_source_splits(source_root: Path) -> dict[str, tuple[Path, Path]]:
    """Discover present Roboflow/YOLO split pairs without inventing a missing split."""

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not (source_root / "data.yaml").is_file():
        raise FileNotFoundError(f"Source data.yaml does not exist: {source_root / 'data.yaml'}")
    aliases = {"train": ("train",), "val": ("val", "valid"), "test": ("test",)}
    found: dict[str, tuple[Path, Path]] = {}
    for normalized, candidates in aliases.items():
        for candidate in candidates:
            image_dir = source_root / candidate / "images"
            label_dir = source_root / candidate / "labels"
            if image_dir.exists() or label_dir.exists():
                if not image_dir.is_dir() or not label_dir.is_dir():
                    raise FileNotFoundError(
                        f"Source split {candidate} must contain both images and labels directories: {source_root}"
                    )
                found[normalized] = (image_dir, label_dir)
                break
    if not found:
        raise FileNotFoundError(f"No source image/label split pairs were found below {source_root}")
    return found


def parse_yolo_line(
    line: str,
    path: Path,
    line_number: int,
    *,
    tolerance: float = BOUND_TOLERANCE,
) -> YoloBox:
    tokens = line.split()
    if len(tokens) != 5:
        raise ValueError(f"{path}:{line_number}: expected exactly five numeric values")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: all five values must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: values must be finite")
    if not values[0].is_integer():
        raise ValueError(f"{path}:{line_number}: class ID must be an integer")
    class_id = int(values[0])
    x_center, y_center, width, height = values[1:]
    if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
        raise ValueError(f"{path}:{line_number}: center coordinates must be in [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}:{line_number}: width and height must be positive")
    x_min, y_min = x_center - width / 2, y_center - height / 2
    x_max, y_max = x_center + width / 2, y_center + height / 2
    if x_min < -tolerance or y_min < -tolerance or x_max > 1 + tolerance or y_max > 1 + tolerance:
        raise ValueError(
            f"{path}:{line_number}: complete box must remain inside normalized image bounds "
            f"(floating-point tolerance {tolerance:g})"
        )
    return YoloBox(class_id, x_center, y_center, width, height)


def load_yolo_labels(path: Path, *, allow_empty: bool = True) -> list[YoloBox]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    if path.stat().st_size == 0:
        if allow_empty:
            return []
        raise ValueError(f"Zero-byte label file: {path}")
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        boxes.append(parse_yolo_line(line, path, line_number))
    if not boxes and not allow_empty:
        raise ValueError(f"Label file contains no boxes: {path}")
    return boxes


def read_image_size(path: Path, *, fully_decode: bool = True) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Zero-byte image file: {path}")
    with Image.open(path) as image:
        if fully_decode:
            image.load()
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid decoded dimensions for {path}: {width}x{height}")
    return width, height


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_to_coco_bbox(box: YoloBox, image_width: int, image_height: int) -> list[float]:
    width = box.width * image_width
    height = box.height * image_height
    x_min = box.x_center * image_width - width / 2
    y_min = box.y_center * image_height - height / 2
    return [x_min, y_min, width, height]


def collect_dataset_records(
    dataset_root: Path,
    *,
    decode_images: bool = True,
    include_hashes: bool = False,
) -> list[dict[str, object]]:
    """Collect one validated record per final image."""

    require_dataset_layout(dataset_root)
    tasks: list[tuple[str, Path, Path]] = []
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        for image_path in find_image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            tasks.append((split, image_path, label_path))

    def load_record(task: tuple[str, Path, Path]) -> dict[str, object]:
        split, image_path, label_path = task
        boxes = load_yolo_labels(label_path, allow_empty=True)
        width, height = read_image_size(image_path, fully_decode=decode_images)
        record: dict[str, object] = {
            "split": split,
            "image_path": image_path,
            "label_path": label_path,
            "image_width": width,
            "image_height": height,
            "image_size_bytes": image_path.stat().st_size,
            "label_size_bytes": label_path.stat().st_size,
            "boxes": boxes,
            "box_count": len(boxes),
        }
        if include_hashes:
            record["image_sha256"] = sha256_file(image_path)
            record["label_sha256"] = sha256_file(label_path)
        return record

    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load_record, tasks))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def summarize(values: Sequence[float | int]) -> dict[str, float | str]:
    if not values:
        return {"min": "NOT_AVAILABLE", "median": "NOT_AVAILABLE", "mean": "NOT_AVAILABLE", "max": "NOT_AVAILABLE"}
    numeric = [float(value) for value in values]
    return {"min": min(numeric), "median": median(numeric), "mean": mean(numeric), "max": max(numeric)}


def pixel_xyxy(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = box.normalized_xyxy()
    return x_min * width, y_min * height, x_max * width, y_max * height


def _font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(image_path: Path, boxes: Sequence[YoloBox], caption: str, tile_size: tuple[int, int] = (480, 390)) -> Image.Image:
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        original_width, original_height = source.size
        caption_height = 54
        scale = min(tile_size[0] / original_width, (tile_size[1] - caption_height) / original_height)
        rendered = source.resize((max(1, round(original_width * scale)), max(1, round(original_height * scale))))
    tile = Image.new("RGB", tile_size, "white")
    offset_x = (tile_size[0] - rendered.width) // 2
    tile.paste(rendered, (offset_x, 0))
    draw = ImageDraw.Draw(tile)
    label_font = _font(14)
    for box in boxes:
        x1, y1, x2, y2 = pixel_xyxy(box, original_width, original_height)
        coords = (offset_x + x1 * scale, y1 * scale, offset_x + x2 * scale, y2 * scale)
        draw.rectangle(coords, outline=(255, 30, 30), width=4)
        text_x = max(0, min(coords[0], tile_size[0] - 100))
        text_y = max(0, coords[1] - 17)
        draw.rectangle((text_x, text_y, text_x + 100, text_y + 17), fill=(255, 30, 30))
        draw.text((text_x + 2, text_y), TARGET_CLASS_NAME, fill="white", font=label_font)
    draw.text((6, tile_size[1] - caption_height + 4), caption[:82], fill="black", font=label_font)
    draw.text((6, tile_size[1] - 24), image_path.name[:76], fill="black", font=_font(12))
    return tile


def make_contact_sheet(
    records: Sequence[dict[str, object]],
    output_path: Path,
    *,
    title: str,
    columns: int = 4,
) -> None:
    if not records:
        raise ValueError(f"No valid samples exist for requested figure: {output_path.name}")
    tile_size = (480, 390)
    title_height = 42
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (tile_size[0] * columns, tile_size[1] * rows + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill="black", font=_font(22))
    for index, record in enumerate(records):
        tile = draw_boxes(
            Path(record["image_path"]),
            list(record["boxes"]),
            f"{record['split']} | boxes={record['box_count']}",
            tile_size,
        )
        x = index % columns * tile_size[0]
        y = index // columns * tile_size[1] + title_height
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def load_augmentation_policy(path: Path) -> dict:
    policy = load_yaml(path)
    required = {
        "enabled_for_train",
        "enabled_for_validation",
        "enabled_for_test",
        "rotation_degrees",
        "scale_range",
        "translation_fraction",
        "brightness_factor",
        "contrast_factor",
        "gaussian_blur_probability",
        "gaussian_blur_radius",
        "constrained_crop_probability",
        "minimum_bbox_visibility",
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "random_seed",
    }
    missing = sorted(required - policy.keys())
    if missing:
        raise ValueError(f"Augmentation policy is missing fields: {', '.join(missing)}")
    if policy["enabled_for_validation"] or policy["enabled_for_test"]:
        raise ValueError("Validation and test random augmentation must be disabled")
    if float(policy["horizontal_flip_probability"]) != 0 or float(policy["vertical_flip_probability"]) != 0:
        raise ValueError("Horizontal and vertical flips must remain disabled")
    if float(policy["minimum_bbox_visibility"]) != 0.70:
        raise ValueError("minimum_bbox_visibility must be 0.70")
    return policy


def augment_training_sample(
    rgb_image,
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    policy: dict,
    rng: random.Random,
    *,
    force_constrained_crop: bool = False,
):
    """Apply a mild training-only transform and update every box; return None if visibility is insufficient."""

    import cv2
    import numpy as np

    if not boxes_xyxy:
        return None
    height, width = rgb_image.shape[:2]
    angle = rng.uniform(*[float(value) for value in policy["rotation_degrees"]])
    scale = rng.uniform(*[float(value) for value in policy["scale_range"]])
    translation = float(policy["translation_fraction"])
    tx = rng.uniform(-translation, translation) * width
    ty = rng.uniform(-translation, translation) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale).astype(np.float64)
    matrix[:, 2] += (tx, ty)

    use_crop = force_constrained_crop or rng.random() < float(policy["constrained_crop_probability"])
    if use_crop:
        crop_fraction = rng.uniform(0.0, min(0.08, translation))
        crop_x = crop_fraction * width
        crop_y = crop_fraction * height
        crop_matrix = np.array(
            [[width / (width - 2 * crop_x), 0, -crop_x * width / (width - 2 * crop_x)],
             [0, height / (height - 2 * crop_y), -crop_y * height / (height - 2 * crop_y)]],
            dtype=np.float64,
        )
        homogeneous = np.vstack([matrix, [0, 0, 1]])
        matrix = (np.vstack([crop_matrix, [0, 0, 1]]) @ homogeneous)[:2]

    transformed = cv2.warpAffine(
        rgb_image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    output_boxes: list[tuple[float, float, float, float]] = []
    minimum_visibility = float(policy["minimum_bbox_visibility"])
    for x1, y1, x2, y2 in boxes_xyxy:
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
        points = corners @ matrix.T
        raw_x1, raw_y1 = points[:, 0].min(), points[:, 1].min()
        raw_x2, raw_y2 = points[:, 0].max(), points[:, 1].max()
        raw_area = max(0.0, raw_x2 - raw_x1) * max(0.0, raw_y2 - raw_y1)
        clipped = (max(0.0, raw_x1), max(0.0, raw_y1), min(float(width), raw_x2), min(float(height), raw_y2))
        clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        visibility = clipped_area / raw_area if raw_area else 0.0
        if visibility < minimum_visibility or clipped_area <= 0:
            return None
        output_boxes.append(clipped)

    brightness = rng.uniform(*[float(value) for value in policy["brightness_factor"]])
    contrast = rng.uniform(*[float(value) for value in policy["contrast_factor"]])
    float_image = transformed.astype(np.float32)
    float_image = (float_image - 127.5) * contrast + 127.5
    float_image *= brightness
    transformed = np.clip(float_image, 0, 255).astype(np.uint8)
    if rng.random() < float(policy["gaussian_blur_probability"]):
        radius = rng.uniform(*[float(value) for value in policy["gaussian_blur_radius"]])
        transformed = cv2.GaussianBlur(transformed, (0, 0), sigmaX=radius, sigmaY=radius)
    return transformed, output_boxes


def split_counts(records: Sequence[dict[str, object]]) -> dict[str, dict[str, int]]:
    result = {split: {"images": 0, "labels": 0, "boxes": 0} for split in SPLITS}
    for record in records:
        split = str(record["split"])
        result[split]["images"] += 1
        result[split]["labels"] += 1
        result[split]["boxes"] += int(record["box_count"])
    result["total"] = {
        key: sum(result[split][key] for split in SPLITS) for key in ("images", "labels", "boxes")
    }
    return result


def group_by_sha(records: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["image_sha256"]), []).append(record)
    return groups


def count_source_labels(
    split_dirs: dict[str, tuple[Path, Path]], class_count: int
) -> tuple[Counter[int], dict[str, int]]:
    """Validate and count all source rows from actual label files."""

    class_counts: Counter[int] = Counter()
    totals = {"images": 0, "labels": 0, "boxes": 0, "orphan_images": 0, "orphan_labels": 0}
    all_labels: list[Path] = []
    for image_dir, label_dir in split_dirs.values():
        images = find_image_files(image_dir)
        labels = find_label_files(label_dir)
        all_labels.extend(labels)
        totals["images"] += len(images)
        totals["labels"] += len(labels)
        image_stems = Counter(path.stem.casefold() for path in images)
        label_stems = Counter(path.stem.casefold() for path in labels)
        totals["orphan_images"] += sum((image_stems - label_stems).values())
        totals["orphan_labels"] += sum((label_stems - image_stems).values())
    def read_source_label(label_path: Path) -> Counter[int]:
        local: Counter[int] = Counter()
        for box in load_yolo_labels(label_path, allow_empty=True):
            if box.class_id < 0 or box.class_id >= class_count:
                raise ValueError(f"{label_path}: source class ID {box.class_id} is absent from data.yaml")
            local[box.class_id] += 1
        return local

    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for local in executor.map(read_source_label, all_labels):
            class_counts.update(local)
            totals["boxes"] += sum(local.values())
    return class_counts, totals
