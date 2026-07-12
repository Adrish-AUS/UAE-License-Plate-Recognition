#!/usr/bin/env python
"""
Train RT-DETR-L on the exact UAE licence-plate YOLO split.

Default comparison settings:
- COCO-pretrained rtdetr-l.pt
- 576 x 576 input
- 20 epochs
- physical batch size 4
- nominal batch size 16 (Ultralytics gradient accumulation target)
- AdamW, lr=1e-4
- seed 486
- patience 4
- no horizontal or vertical flips
- AMP disabled because the official RT-DETR trainer documentation warns that
  AMP can produce NaNs during bipartite matching
- deterministic=False because grid_sample in RT-DETR does not support fully
  deterministic training

The script validates the split counts before training and writes an absolute
dataset YAML so Windows relative paths cannot silently point to the wrong data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

EXPECTED_COUNTS = {
    "train": 6738,
    "valid": 1440,
    "test": 1432,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR-L on the UAE plate dataset.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/uae_lp_v2_yolo"),
        help="YOLO-format dataset folder, relative to project root unless absolute.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=576)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--nbs",
        type=int,
        default=16,
        help="Nominal batch size. With batch 4, this targets accumulation to effective batch 16.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="rtdetr_l_main")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate environment, dataset, and pretrained model without training.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an interrupted RT-DETR last.pt checkpoint.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Continue even if image counts differ from the frozen project split.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Do not evaluate best.pt on the untouched test split after training.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def identify_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    aliases = ["valid", "val"] if split == "valid" else [split]
    candidates: list[tuple[Path, Path]] = []

    # Layout A: dataset/train/images + dataset/train/labels
    for alias in aliases:
        candidates.append((dataset_root / alias / "images", dataset_root / alias / "labels"))

    # Layout B: dataset/images/train + dataset/labels/train
    for alias in aliases:
        candidates.append((dataset_root / "images" / alias, dataset_root / "labels" / alias))

    for image_dir, label_dir in candidates:
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir.resolve(), label_dir.resolve()

    tried = "\n".join(
        f"  - images: {image_dir} | labels: {label_dir}"
        for image_dir, label_dir in candidates
    )
    raise FileNotFoundError(
        f"Could not find the {split} split. Tried:\n{tried}"
    )


def validate_labels(images: Iterable[Path], labels_dir: Path) -> tuple[int, int]:
    missing_labels = 0
    invalid_rows = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue

        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                invalid_rows += 1
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                invalid_rows += 1
                continue

            if class_id != 0 or any(value < 0.0 or value > 1.0 for value in coords):
                invalid_rows += 1

    return missing_labels, invalid_rows


def write_dataset_yaml(dataset_root: Path, split_dirs: dict[str, tuple[Path, Path]]) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is missing. It is normally installed with Ultralytics."
        ) from exc

    yaml_path = dataset_root.parent / "uae_lp_v2_rtdetr.yaml"
    payload = {
        "path": str(dataset_root),
        "train": str(split_dirs["train"][0]),
        "val": str(split_dirs["valid"][0]),
        "test": str(split_dirs["test"][0]),
        "nc": 1,
        "names": {0: "license_plate"},
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path.resolve()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)

    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if args.epochs <= 0 or args.batch <= 0 or args.nbs <= 0 or args.imgsz <= 0:
        raise ValueError("epochs, batch, nbs, and imgsz must all be positive.")

    try:
        import torch
        import ultralytics
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not available in this environment. Activate rfdetr_env, "
            "save pip freeze, and install ultralytics before running this script."
        ) from exc

    print("Environment")
    print("-----------")
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Ultralytics:   {ultralytics.__version__}")
    print(f"CUDA:          {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:    {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    else:
        raise RuntimeError("CUDA is not available. Do not start the overnight run on CPU.")

    split_dirs = {
        split: identify_split_dirs(dataset_root, split)
        for split in ("train", "valid", "test")
    }

    split_summary: dict[str, dict[str, int | str]] = {}
    print("\nDataset validation")
    print("------------------")
    for split, (image_dir, label_dir) in split_dirs.items():
        images = image_files(image_dir)
        missing_labels, invalid_rows = validate_labels(images, label_dir)
        expected = EXPECTED_COUNTS[split]

        split_summary[split] = {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "images": len(images),
            "expected_images": expected,
            "missing_label_files": missing_labels,
            "invalid_label_rows": invalid_rows,
        }

        print(
            f"{split:5s}: images={len(images)} expected={expected} "
            f"missing_labels={missing_labels} invalid_rows={invalid_rows}"
        )

        if len(images) != expected and not args.allow_count_mismatch:
            raise RuntimeError(
                f"{split} contains {len(images)} images instead of the frozen count {expected}. "
                "Stop and verify the split, or pass --allow-count-mismatch only when intentional."
            )
        if missing_labels:
            raise RuntimeError(f"{split} has {missing_labels} images without label files.")
        if invalid_rows:
            raise RuntimeError(f"{split} has {invalid_rows} invalid YOLO label rows.")

    dataset_yaml = write_dataset_yaml(dataset_root, split_dirs)
    print(f"\nGenerated dataset YAML: {dataset_yaml}")

    metadata_path = project_root / "results" / "rtdetr_training_setup.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(dataset_yaml),
        "split_summary": split_summary,
        "model": "rtdetr-l.pt",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "physical_batch_size": args.batch,
        "nominal_batch_size": args.nbs,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "weight_decay": 0.001,
        "seed": 486,
        "patience": 4,
        "amp": False,
        "deterministic": False,
        "augmentation": {
            "hsv_h": 0.0,
            "hsv_s": 0.2,
            "hsv_v": 0.2,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.10,
            "shear": 0.0,
            "perspective": 0.0005,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved setup metadata: {metadata_path.resolve()}")

    if args.resume is not None:
        resume_path = resolve_path(project_root, args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\nResuming from: {resume_path}")
        model = RTDETR(str(resume_path))
        model.train(resume=True)
        return 0

    print("\nLoading COCO-pretrained RT-DETR-L...")
    model = RTDETR("rtdetr-l.pt")
    model.info()

    if args.check_only:
        print("\nCHECK PASSED. No training was started.")
        return 0

    print("\nStarting RT-DETR-L training")
    print("---------------------------")
    print("AMP is disabled for stability.")
    print("Horizontal and vertical flips are disabled for readable licence plates.")
    print("The validation set is used during training; the test set remains untouched.")

    train_results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        nbs=args.nbs,
        device=args.device,
        workers=args.workers,
        project=str((project_root / "runs").resolve()),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        weight_decay=0.001,
        warmup_epochs=1.0,
        cos_lr=True,
        patience=4,
        seed=486,
        deterministic=False,
        amp=False,
        cache=False,
        val=True,
        save=True,
        plots=True,
        single_cls=True,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        verbose=True,
    )

    print("\nTraining finished.")

    trainer = getattr(model, "trainer", None)
    best_value = getattr(trainer, "best", None) if trainer is not None else None
    if best_value is None:
        raise RuntimeError(
            "Training completed, but the best.pt path could not be obtained from the trainer."
        )

    best_path = Path(str(best_value)).resolve()
    if not best_path.is_file():
        raise FileNotFoundError(f"Training finished, but best.pt was not found: {best_path}")

    print(f"Best checkpoint: {best_path}")

    if args.skip_test_eval:
        print("Test evaluation was skipped by request.")
        return 0

    print("\nEvaluating best.pt on the untouched test split")
    print("----------------------------------------------")
    best_model = RTDETR(str(best_path))
    metrics = best_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str((project_root / "runs").resolve()),
        name="rtdetr_l_test",
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        raise RuntimeError("Ultralytics validation did not return box metrics.")

    speed = getattr(metrics, "speed", {})
    test_payload = {
        "checkpoint": str(best_path),
        "dataset_yaml": str(dataset_yaml),
        "split": "test",
        "num_test_images_expected": EXPECTED_COUNTS["test"],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "map_50_95": float(box_metrics.map),
        "map_50": float(box_metrics.map50),
        "map_75": float(box_metrics.map75),
        "mean_precision": float(box_metrics.mp),
        "mean_recall": float(box_metrics.mr),
        "speed_ms_per_image": {
            str(key): float(value) for key, value in dict(speed).items()
        },
        "note": (
            "Ultralytics validation precision and recall use its validation operating point. "
            "For a threshold-controlled comparison, select confidence on validation and freeze "
            "it before computing test precision, recall, and F1."
        ),
    }

    test_metrics_path = project_root / "results" / "rtdetr_l_test_metrics.json"
    test_metrics_path.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")

    print("\nRT-DETR-L test summary")
    print("----------------------")
    print(f"mAP@50:95: {test_payload['map_50_95']:.6f}")
    print(f"mAP@50:    {test_payload['map_50']:.6f}")
    print(f"mAP@75:    {test_payload['map_75']:.6f}")
    print(f"Precision: {test_payload['mean_precision']:.6f}")
    print(f"Recall:    {test_payload['mean_recall']:.6f}")
    print(f"Saved:     {test_metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
