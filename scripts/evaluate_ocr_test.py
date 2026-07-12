#!/usr/bin/env python
"""
Evaluate the fine-tuned UAE licence-plate OCR model without using model.evaluate().

Why this script exists:
- The FastPlateOCR CLI validation path can fail because Keras tries to process
  string labels under some backend/version combinations.
- This script sends only numeric image batches to model.predict().
- All metrics are then calculated in NumPy.

Outputs:
- metrics.json
- predictions.csv
- character_confusions.csv
- examples.csv
- correct_examples.jpg
- incorrect_examples.jpg
- examples/correct/*
- examples/incorrect/*
- optional demo_prediction.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# This must be set before importing keras or fast_plate_ocr.
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a FastPlateOCR .keras model on a labelled OCR crop dataset."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/ocr_uae_main"),
        help="Root containing timestamped OCR training runs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Exact path to best.keras. If omitted, the newest best.keras under --run-root is used.",
    )
    parser.add_argument(
        "--plate-config",
        type=Path,
        default=None,
        help="Matching plate_config.yaml. If omitted, it is located beside the model.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/uae_lp_ocr/test/annotations.csv"),
        help="Untouched test annotations CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ocr_uae_test"),
        help="Directory for metrics, predictions, and example images.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-correct-examples", type=int, default=12)
    parser.add_argument("--num-incorrect-examples", type=int, default=12)
    parser.add_argument(
        "--demo-image",
        type=Path,
        default=None,
        help="Optional single crop to test after the dataset evaluation.",
    )
    parser.add_argument(
        "--demo-expected",
        type=str,
        default=None,
        help="Optional expected text for --demo-image, for example S10198.",
    )
    return parser.parse_args()


def find_newest_best_model(run_root: Path) -> Path:
    if not run_root.exists():
        raise FileNotFoundError(f"OCR run root does not exist: {run_root.resolve()}")

    candidates = sorted(
        run_root.rglob("best.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    other_keras = sorted(
        [p for p in run_root.rglob("*.keras") if p.name.lower() != "last.keras"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    details = "\n".join(f"  - {p.resolve()}" for p in other_keras[:20])
    raise FileNotFoundError(
        "No best.keras was found. The script will not silently use last.keras.\n"
        "Non-last .keras candidates found:\n"
        f"{details if details else '  (none)'}\n"
        "Pass the correct checkpoint explicitly with --model."
    )


def find_matching_config(model_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        requested = requested.resolve()
        if not requested.is_file():
            raise FileNotFoundError(f"Plate config not found: {requested}")
        return requested

    direct = model_path.parent / "plate_config.yaml"
    if direct.is_file():
        return direct.resolve()

    candidates = sorted(model_path.parent.glob("*plate*config*.yaml"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple plate config files were found beside the model. "
            "Pass the matching one with --plate-config:\n"
            + "\n".join(f"  - {p.resolve()}" for p in candidates)
        )

    raise FileNotFoundError(
        f"No plate_config.yaml was found beside {model_path}. "
        "Pass it explicitly with --plate-config."
    )


def locate_model_config(model_path: Path) -> Path | None:
    direct = model_path.parent / "model_config.yaml"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(model_path.parent.glob("*model*config*.yaml"))
    return candidates[0].resolve() if len(candidates) == 1 else None


def load_fastplate_components():
    try:
        from fast_plate_ocr.core.process import (
            postprocess_output,
            read_and_resize_plate_image,
        )
        from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
        from fast_plate_ocr.train.utilities.utils import load_keras_model
    except Exception as exc:
        raise RuntimeError(
            "Could not import FastPlateOCR training utilities. Activate rfdetr_env "
            "and confirm fast-plate-ocr[train] is installed."
        ) from exc

    return (
        postprocess_output,
        read_and_resize_plate_image,
        load_plate_config_from_yaml,
        load_keras_model,
    )


def resolve_image_path(csv_path: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return (csv_path.parent / candidate).resolve()


def extract_plate_output(raw_output: Any, model: Any) -> np.ndarray:
    if isinstance(raw_output, dict):
        if "plate" not in raw_output:
            raise KeyError(f"Model returned a dict without a 'plate' output: {raw_output.keys()}")
        return np.asarray(raw_output["plate"])

    if isinstance(raw_output, (list, tuple)):
        output_names = list(getattr(model, "output_names", []))
        if "plate" in output_names:
            return np.asarray(raw_output[output_names.index("plate")])
        return np.asarray(raw_output[0])

    return np.asarray(raw_output)


def select_quantile_examples(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """Choose low-, medium-, and high-confidence examples deterministically."""
    if count <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= count:
        return df.copy()

    ordered = df.sort_values(["mean_visible_confidence", "image_path"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, num=count)
    indices = sorted(set(int(round(i)) for i in indices))
    selected = ordered.iloc[indices]
    if len(selected) < count:
        missing = count - len(selected)
        extras = ordered.drop(index=selected.index).head(missing)
        selected = pd.concat([selected, extras], ignore_index=True)
    return selected.head(count)


def make_contact_sheet(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    columns: int = 3,
    tile_width: int = 420,
    tile_height: int = 190,
) -> None:
    if rows.empty:
        return

    header_height = 48
    margin = 10
    rows_count = int(np.ceil(len(rows) / columns))
    title_height = 45

    canvas = np.full(
        (title_height + rows_count * tile_height, columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for sheet_index, row in enumerate(rows.itertuples(index=False)):
        grid_row = sheet_index // columns
        grid_col = sheet_index % columns
        x0 = grid_col * tile_width
        y0 = title_height + grid_row * tile_height

        cv2.rectangle(
            canvas,
            (x0 + 2, y0 + 2),
            (x0 + tile_width - 3, y0 + tile_height - 3),
            (180, 180, 180),
            1,
        )

        label1 = f"GT: {row.ground_truth}   PRED: {row.prediction}"
        label2 = (
            f"Exact: {bool(row.correct)}   "
            f"Length: {bool(row.length_correct)}   "
            f"Conf: {float(row.mean_visible_confidence):.3f}"
        )
        cv2.putText(
            canvas,
            label1[:58],
            (x0 + margin, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label2[:65],
            (x0 + margin, y0 + 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        image = cv2.imread(str(row.resolved_image_path), cv2.IMREAD_COLOR)
        if image is None:
            cv2.putText(
                canvas,
                "Image could not be read",
                (x0 + margin, y0 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 180),
                1,
                cv2.LINE_AA,
            )
            continue

        available_w = tile_width - 2 * margin
        available_h = tile_height - header_height - 2 * margin
        scale = min(available_w / image.shape[1], available_h / image.shape[0])
        new_w = max(1, int(round(image.shape[1] * scale)))
        new_h = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        paste_x = x0 + (tile_width - new_w) // 2
        paste_y = y0 + header_height + (available_h - new_h) // 2
        canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save contact sheet: {output_path}")


def copy_examples(
    selected: pd.DataFrame,
    category: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    category_dir = output_dir / "examples" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(selected.itertuples(index=False), start=1):
        source = Path(row.resolved_image_path)
        suffix = source.suffix.lower() or ".jpg"
        safe_gt = str(row.ground_truth).replace("/", "_").replace("\\", "_")
        safe_pred = str(row.prediction).replace("/", "_").replace("\\", "_") or "EMPTY"
        destination = category_dir / f"{index:02d}_GT-{safe_gt}_PRED-{safe_pred}{suffix}"
        shutil.copy2(source, destination)

        records.append(
            {
                "category": category,
                "source_image": str(source),
                "copied_image": str(destination.resolve()),
                "ground_truth": row.ground_truth,
                "prediction": row.prediction,
                "correct": bool(row.correct),
                "length_correct": bool(row.length_correct),
                "slot_char_accuracy": float(row.slot_char_accuracy),
                "mean_visible_confidence": float(row.mean_visible_confidence),
            }
        )
    return records


def run_single_image(
    image_path: Path,
    expected: str | None,
    model: Any,
    plate_config: Any,
    read_and_resize_plate_image: Any,
    postprocess_output: Any,
    output_dir: Path,
) -> None:
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Demo image not found: {image_path}")

    image = read_and_resize_plate_image(
        image_path=image_path,
        img_height=plate_config.img_height,
        img_width=plate_config.img_width,
        image_color_mode=plate_config.image_color_mode,
        keep_aspect_ratio=plate_config.keep_aspect_ratio,
        interpolation_method=plate_config.interpolation,
        padding_color=plate_config.padding_color,
    )
    raw = extract_plate_output(model.predict(np.expand_dims(image, 0), verbose=0), model)
    pred = postprocess_output(
        model_output=raw,
        max_plate_slots=plate_config.max_plate_slots,
        model_alphabet=plate_config.alphabet,
        pad_char=plate_config.pad_char,
        remove_pad_char=True,
        return_confidence=True,
    )[0]

    result = {
        "image_path": str(image_path),
        "prediction": pred.plate,
        "expected": expected,
        "correct": None if expected is None else pred.plate == expected,
        "character_confidences": (
            [float(v) for v in pred.char_probs[: len(pred.plate)]]
            if pred.char_probs is not None
            else None
        ),
    }
    with open(output_dir / "demo_prediction.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\nDemo crop prediction")
    print("--------------------")
    print(f"Image:      {image_path}")
    print(f"Prediction: {pred.plate}")
    if expected is not None:
        print(f"Expected:   {expected}")
        print(f"Correct:    {pred.plate == expected}")


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    model_path = args.model.resolve() if args.model is not None else find_newest_best_model(args.run_root)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    plate_config_path = find_matching_config(model_path, args.plate_config)
    model_config_path = locate_model_config(model_path)
    annotations_path = args.annotations.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_path}")

    (
        postprocess_output,
        read_and_resize_plate_image,
        load_plate_config_from_yaml,
        load_keras_model,
    ) = load_fastplate_components()

    print("Using files")
    print("-----------")
    print(f"Model:        {model_path}")
    print(f"Plate config: {plate_config_path}")
    print(f"Model config: {model_config_path if model_config_path else '(not found beside checkpoint)'}")
    print(f"Annotations:  {annotations_path}")
    print(f"Output:       {output_dir}")

    plate_config = load_plate_config_from_yaml(plate_config_path)
    model = load_keras_model(model_path, plate_config)

    annotations = pd.read_csv(annotations_path, dtype={"image_path": str, "plate_text": str})
    required_columns = {"image_path", "plate_text"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Annotations CSV is missing columns: {sorted(missing_columns)}")
    if annotations.empty:
        raise ValueError("Annotations CSV contains no rows.")
    if annotations[["image_path", "plate_text"]].isna().any().any():
        raise ValueError("Annotations CSV contains missing image_path or plate_text values.")

    annotations["plate_text"] = annotations["plate_text"].astype(str).str.strip()
    annotations["resolved_image_path"] = annotations["image_path"].map(
        lambda p: str(resolve_image_path(annotations_path, p))
    )

    missing_images = [
        p for p in annotations["resolved_image_path"].tolist() if not Path(p).is_file()
    ]
    if missing_images:
        preview = "\n".join(f"  - {p}" for p in missing_images[:20])
        raise FileNotFoundError(
            f"{len(missing_images)} annotated images were not found. First entries:\n{preview}"
        )

    alphabet = plate_config.alphabet
    alphabet_to_index = {character: index for index, character in enumerate(alphabet)}
    max_slots = int(plate_config.max_plate_slots)
    vocab_size = len(alphabet)
    pad_char = plate_config.pad_char
    pad_index = int(plate_config.pad_idx)

    invalid_labels: list[str] = []
    for label in annotations["plate_text"]:
        if len(label) > max_slots or any(character not in alphabet for character in label):
            invalid_labels.append(label)
    if invalid_labels:
        raise ValueError(
            "Test labels are incompatible with the matching plate config. "
            f"First invalid labels: {invalid_labels[:20]}"
        )

    result_rows: list[dict[str, Any]] = []
    confusion_counter: Counter[tuple[str, str]] = Counter()

    num_samples = len(annotations)
    for start in range(0, num_samples, args.batch_size):
        batch_df = annotations.iloc[start : start + args.batch_size]
        batch_images: list[np.ndarray] = []

        for image_path in batch_df["resolved_image_path"]:
            image = read_and_resize_plate_image(
                image_path=image_path,
                img_height=plate_config.img_height,
                img_width=plate_config.img_width,
                image_color_mode=plate_config.image_color_mode,
                keep_aspect_ratio=plate_config.keep_aspect_ratio,
                interpolation_method=plate_config.interpolation,
                padding_color=plate_config.padding_color,
            )
            batch_images.append(image)

        batch_array = np.asarray(batch_images, dtype=np.uint8)
        raw_output = model.predict(batch_array, batch_size=len(batch_array), verbose=0)
        plate_output = extract_plate_output(raw_output, model)
        predictions_3d = plate_output.reshape((-1, max_slots, vocab_size))
        predicted_indices = np.argmax(predictions_3d, axis=-1)
        predicted_confidences = np.max(predictions_3d, axis=-1)

        decoded = postprocess_output(
            model_output=plate_output,
            max_plate_slots=max_slots,
            model_alphabet=alphabet,
            pad_char=pad_char,
            remove_pad_char=True,
            return_confidence=True,
        )

        for local_index, (_, row) in enumerate(batch_df.iterrows()):
            ground_truth = str(row["plate_text"])
            prediction = decoded[local_index].plate

            ground_truth_padded = ground_truth.ljust(max_slots, pad_char)
            ground_truth_indices = np.asarray(
                [alphabet_to_index[character] for character in ground_truth_padded],
                dtype=np.int64,
            )
            pred_indices = predicted_indices[local_index]
            slot_matches = pred_indices == ground_truth_indices

            predicted_length = int(np.count_nonzero(pred_indices != pad_index))
            ground_truth_length = len(ground_truth)
            correct = prediction == ground_truth
            length_correct = predicted_length == ground_truth_length

            visible_mask = pred_indices != pad_index
            if np.any(visible_mask):
                mean_visible_confidence = float(
                    np.mean(predicted_confidences[local_index][visible_mask])
                )
                min_visible_confidence = float(
                    np.min(predicted_confidences[local_index][visible_mask])
                )
            else:
                mean_visible_confidence = float(
                    np.mean(predicted_confidences[local_index])
                )
                min_visible_confidence = float(
                    np.min(predicted_confidences[local_index])
                )

            for slot, is_match in enumerate(slot_matches):
                if not is_match:
                    true_char = alphabet[int(ground_truth_indices[slot])]
                    pred_char = alphabet[int(pred_indices[slot])]
                    confusion_counter[(true_char, pred_char)] += 1

            result_rows.append(
                {
                    "image_path": row["image_path"],
                    "resolved_image_path": row["resolved_image_path"],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": bool(correct),
                    "ground_truth_length": ground_truth_length,
                    "predicted_length": predicted_length,
                    "length_correct": bool(length_correct),
                    "correct_character_slots": int(np.sum(slot_matches)),
                    "max_plate_slots": max_slots,
                    "slot_char_accuracy": float(np.mean(slot_matches)),
                    "mean_visible_confidence": mean_visible_confidence,
                    "min_visible_confidence": min_visible_confidence,
                }
            )

        completed = min(start + args.batch_size, num_samples)
        print(f"\rProcessed {completed}/{num_samples} crops", end="", flush=True)

    print()
    predictions_df = pd.DataFrame(result_rows)

    exact_matches = int(predictions_df["correct"].sum())
    incorrect_plates = int(len(predictions_df) - exact_matches)
    exact_match_accuracy = exact_matches / len(predictions_df)
    character_accuracy = float(
        predictions_df["correct_character_slots"].sum()
        / (len(predictions_df) * max_slots)
    )
    length_accuracy = float(predictions_df["length_correct"].mean())

    metrics = {
        "model_path": str(model_path),
        "plate_config_path": str(plate_config_path),
        "model_config_path": str(model_config_path) if model_config_path else None,
        "annotations_path": str(annotations_path),
        "num_test_crops": int(len(predictions_df)),
        "exact_matches": exact_matches,
        "incorrect_plates": incorrect_plates,
        "full_plate_exact_match_accuracy": exact_match_accuracy,
        "character_accuracy": character_accuracy,
        "character_accuracy_definition": (
            "Mean categorical accuracy across all fixed character slots, including "
            "correct padding slots. This matches FastPlateOCR char_acc."
        ),
        "plate_length_accuracy": length_accuracy,
        "max_plate_slots": max_slots,
        "alphabet": alphabet,
        "pad_char": pad_char,
        "keras_backend": os.environ.get("KERAS_BACKEND"),
    }

    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    confusion_rows = [
        {
            "ground_truth_character": "<PAD>" if true == pad_char else true,
            "predicted_character": "<PAD>" if pred == pad_char else pred,
            "count": count,
        }
        for (true, pred), count in confusion_counter.most_common()
    ]
    pd.DataFrame(
        confusion_rows,
        columns=["ground_truth_character", "predicted_character", "count"],
    ).to_csv(output_dir / "character_confusions.csv", index=False)

    correct_selected = select_quantile_examples(
        predictions_df[predictions_df["correct"]],
        args.num_correct_examples,
    )
    incorrect_selected = (
        predictions_df[~predictions_df["correct"]]
        .sort_values(
            ["correct_character_slots", "mean_visible_confidence", "image_path"],
            ascending=[True, True, True],
        )
        .head(args.num_incorrect_examples)
    )

    example_records: list[dict[str, Any]] = []
    example_records.extend(copy_examples(correct_selected, "correct", output_dir))
    example_records.extend(copy_examples(incorrect_selected, "incorrect", output_dir))
    pd.DataFrame(example_records).to_csv(output_dir / "examples.csv", index=False)

    make_contact_sheet(
        correct_selected,
        output_dir / "correct_examples.jpg",
        title="OCR correct test examples",
    )
    make_contact_sheet(
        incorrect_selected,
        output_dir / "incorrect_examples.jpg",
        title="OCR incorrect test examples",
    )

    print("\nTest metrics")
    print("------------")
    print(f"Test crops:                  {len(predictions_df)}")
    print(f"Exact matches:               {exact_matches}")
    print(f"Incorrect plates:            {incorrect_plates}")
    print(f"Full-plate exact accuracy:   {exact_match_accuracy:.6f}")
    print(f"Character accuracy:          {character_accuracy:.6f}")
    print(f"Plate-length accuracy:       {length_accuracy:.6f}")
    print(f"Predictions CSV:             {predictions_path}")
    print(f"Metrics JSON:                {output_dir / 'metrics.json'}")
    print(f"Correct contact sheet:       {output_dir / 'correct_examples.jpg'}")
    print(f"Incorrect contact sheet:     {output_dir / 'incorrect_examples.jpg'}")

    if args.demo_image is not None:
        run_single_image(
            image_path=args.demo_image,
            expected=args.demo_expected,
            model=model,
            plate_config=plate_config,
            read_and_resize_plate_image=read_and_resize_plate_image,
            postprocess_output=postprocess_output,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
