#!/usr/bin/env python
"""
Train and evaluate a seven-class UAE emirate classifier on recovered plate crops.

Model:
- ImageNet-pretrained ResNet18
- Input size: 224 x 224
- Balanced train sampling
- Mild plate-preserving augmentation
- Early stopping on validation macro F1
- Automatic evaluation on the untouched test split

Outputs:
runs/emirate_resnet18_main/
    best.pt
    last.pt
    training_history.csv
    training_curves.png
    test_metrics.json
    test_predictions.csv
    classification_report.csv
    confusion_matrix_counts.png
    confusion_matrix_normalized.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ImageFile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXPECTED_CLASSES = {
    "Abu_Dhabi",
    "Ajman",
    "Dubai",
    "Fujairah",
    "Ras_Al_Khaimah",
    "Sharjah",
    "Umm_Al_Quwain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a seven-class UAE emirate classifier."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/uae_lp_emirate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/emirate_resnet18_main"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=486)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                        saturation=0.10,
                    )
                ],
                p=0.60,
            ),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=2,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def class_counts(dataset: datasets.ImageFolder) -> Counter[int]:
    return Counter(target for _, target in dataset.samples)


def build_balanced_sampler(
    dataset: datasets.ImageFolder,
    seed: int,
) -> WeightedRandomSampler:
    counts = class_counts(dataset)
    sample_weights = [
        1.0 / counts[target]
        for _, target in dataset.samples
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def validate_dataset_structure(data_root: Path) -> dict[str, Any]:
    train_transform, eval_transform = make_transforms()
    split_datasets = {
        "train": datasets.ImageFolder(data_root / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_root / "val", transform=eval_transform),
        "test": datasets.ImageFolder(data_root / "test", transform=eval_transform),
    }

    class_sets = {
        split: set(dataset.classes)
        for split, dataset in split_datasets.items()
    }
    if any(classes != EXPECTED_CLASSES for classes in class_sets.values()):
        raise RuntimeError(
            "Class folders do not match the expected seven emirates:\n"
            + "\n".join(
                f"{split}: {sorted(classes)}"
                for split, classes in class_sets.items()
            )
        )

    reference_mapping = split_datasets["train"].class_to_idx
    for split in ("val", "test"):
        if split_datasets[split].class_to_idx != reference_mapping:
            raise RuntimeError(
                f"{split} class_to_idx differs from train."
            )

    summary: dict[str, Any] = {
        "classes": split_datasets["train"].classes,
        "class_to_idx": reference_mapping,
        "splits": {},
    }

    for split, dataset in split_datasets.items():
        counts = class_counts(dataset)
        summary["splits"][split] = {
            dataset.classes[class_index]: int(counts[class_index])
            for class_index in range(len(dataset.classes))
        }

    return {
        "datasets": split_datasets,
        "summary": summary,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    return {
        "loss": average_loss,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            all_targets,
            all_predictions,
        ),
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
        "targets": all_targets,
        "predictions": all_predictions,
    }


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    columns = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "train_weighted_f1",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[0, 1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    axes[1, 0].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation")
    axes[1, 0].set_title("Macro F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(
        epochs,
        [row["val_balanced_accuracy"] for row in history],
        label="Validation balanced accuracy",
    )
    axes[1, 1].plot(
        epochs,
        [row["val_weighted_f1"] for row in history],
        label="Validation weighted F1",
    )
    axes[1, 1].set_title("Validation metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=[name.replace("_", " ") for name in class_names],
        yticklabels=[name.replace("_", " ") for name in class_names],
        ylabel="Ground truth",
        xlabel="Prediction",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    dataset: datasets.ImageFolder,
    device: torch.device,
) -> tuple[list[int], list[int], list[float], list[str]]:
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []
    paths_all: list[str] = []

    sample_offset = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        batch_size = len(targets)
        batch_paths = [
            dataset.samples[index][0]
            for index in range(sample_offset, sample_offset + batch_size)
        ]
        sample_offset += batch_size

        targets_all.extend(targets.tolist())
        predictions_all.extend(predictions.cpu().tolist())
        confidences_all.extend(confidences.cpu().tolist())
        paths_all.extend(batch_paths)

    return targets_all, predictions_all, confidences_all, paths_all


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path.cwd().resolve()
    data_root = (
        args.data_root.resolve()
        if args.data_root.is_absolute()
        else (project_root / args.data_root).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (project_root / args.output_dir).resolve()
    )

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    validated = validate_dataset_structure(data_root)
    split_datasets: dict[str, datasets.ImageFolder] = validated["datasets"]
    dataset_summary = validated["summary"]

    print("Emirate dataset")
    print("----------------")
    print(f"Root: {data_root}")
    for split in ("train", "val", "test"):
        counts = dataset_summary["splits"][split]
        print(f"{split}: total={sum(counts.values())}")
        for class_name, count in counts.items():
            print(f"  {class_name:20s} {count}")

    if args.check_only:
        print("\nCHECK PASSED. No training was started.")
        return 0

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only when intentionally restarting."
            )
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. Do not train this model on CPU.")

    sampler = build_balanced_sampler(split_datasets["train"], args.seed)
    pin_memory = True

    train_loader = DataLoader(
        split_datasets["train"],
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        split_datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        split_datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(split_datasets["train"].classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    run_config = {
        "model": "resnet18",
        "pretrained_weights": "ImageNet ResNet18 default weights",
        "input_size": [224, 224],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "balanced_sampling": True,
        "loss": "CrossEntropyLoss(label_smoothing=0.05)",
        "selection_metric": "validation macro F1",
        "classes": split_datasets["train"].classes,
        "class_to_idx": split_datasets["train"].class_to_idx,
        "dataset_summary": dataset_summary,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_macro_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nTraining")
    print("--------")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
        )

        scheduler.step(val_metrics["macro_f1"])
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        save_history_csv(output_dir / "training_history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": split_datasets["train"].class_to_idx,
            "classes": split_datasets["train"].classes,
            "val_metrics": {
                key: value
                for key, value in val_metrics.items()
                if key not in {"targets", "predictions"}
            },
            "run_config": run_config,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        improved = val_metrics["macro_f1"] > best_macro_f1 + 1e-4
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"macroF1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} | "
            f"lr={learning_rate:.2e} | {elapsed:.1f}s"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    plot_training_curves(
        history,
        output_dir / "training_curves.png",
    )

    best_checkpoint = torch.load(
        output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    targets, predictions, confidences, paths = predict_test(
        model,
        test_loader,
        split_datasets["test"],
        device,
    )

    class_names = split_datasets["test"].classes
    test_metrics = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "num_test_samples": len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "classes": class_names,
        "class_to_idx": split_datasets["test"].class_to_idx,
        "limitation": (
            "Umm Al Quwain has only 7 validation and 15 test examples; its "
            "per-class metrics have high sampling uncertainty."
        ),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_rows.append(
                {
                    "class": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
    with (output_dir / "classification_report.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    with (output_dir / "test_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "ground_truth",
                "prediction",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for path, target, prediction, confidence in zip(
            paths,
            targets,
            predictions,
            confidences,
        ):
            writer.writerow(
                {
                    "image_path": path,
                    "ground_truth": class_names[target],
                    "prediction": class_names[prediction],
                    "confidence": confidence,
                    "correct": target == prediction,
                }
            )

    counts_matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )
    normalized_matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        normalize="true",
    )

    save_confusion_matrix(
        counts_matrix,
        class_names,
        output_dir / "confusion_matrix_counts.png",
        title="Emirate classifier confusion matrix (counts)",
        normalized=False,
    )
    save_confusion_matrix(
        normalized_matrix,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        title="Emirate classifier confusion matrix (row normalized)",
        normalized=True,
    )

    print("\nUntouched test results")
    print("----------------------")
    print(f"Best epoch:         {test_metrics['best_epoch']}")
    print(f"Test samples:       {test_metrics['num_test_samples']}")
    print(f"Accuracy:           {test_metrics['accuracy']:.6f}")
    print(f"Balanced accuracy:  {test_metrics['balanced_accuracy']:.6f}")
    print(f"Macro F1:           {test_metrics['macro_f1']:.6f}")
    print(f"Weighted F1:        {test_metrics['weighted_f1']:.6f}")
    print(f"Outputs:            {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
