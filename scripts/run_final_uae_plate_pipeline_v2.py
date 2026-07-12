#!/usr/bin/env python
"""
Final UAE licence-plate recognition pipeline v2:

full vehicle image
-> RF-DETR Medium plate detection
-> padded plate crop
-> OCR v2 for complete plate text
-> ResNet18 emirate classifier
-> train-derived emirate code rules for code/number splitting
-> annotated image + JSON

The parser does not change OCR text. It only separates a recognised string,
for example:
    Sharjah: 3222566 -> code 3, number 222566
    Abu Dhabi: 1141153 -> code 11, number 41153
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRMedium
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


DEFAULT_DETECTION_THRESHOLD = 0.8414926
LETTER_CODE_PATTERN = re.compile(r"^([A-Z]{1,3})([0-9]{2,7})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final RF-DETR + OCR v2 + emirate classifier pipeline."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("runs/rfdetr_medium_main/checkpoint_best_total.pth"),
    )
    parser.add_argument(
        "--ocr-run-root",
        type=Path,
        default=Path("runs/ocr_uae_v2_retry"),
    )
    parser.add_argument("--ocr-model", type=Path, default=None)
    parser.add_argument("--plate-config", type=Path, default=None)
    parser.add_argument(
        "--emirate-checkpoint",
        type=Path,
        default=Path("runs/emirate_resnet18_main/best.pt"),
    )
    parser.add_argument(
        "--code-rules",
        type=Path,
        default=Path("results/emirate_code_rules.json"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
    )
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_pipeline_demo_v2"),
    )
    return parser.parse_args()


def find_newest_best_model(run_root: Path) -> Path:
    root = run_root.resolve()
    candidates = sorted(
        root.rglob("best.keras"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best.keras found under {root}")
    return candidates[0].resolve()


def find_plate_config(model_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        path = requested.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    direct = model_path.parent / "plate_config.yaml"
    if direct.is_file():
        return direct.resolve()

    candidates = list(model_path.parent.glob("*plate*config*.yaml"))
    if len(candidates) == 1:
        return candidates[0].resolve()

    raise FileNotFoundError(
        f"Could not determine plate_config.yaml beside {model_path}"
    )


def load_ocr_components():
    from fast_plate_ocr.core.process import (
        postprocess_output,
        read_and_resize_plate_image,
    )
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    return (
        postprocess_output,
        read_and_resize_plate_image,
        load_plate_config_from_yaml,
        load_keras_model,
    )


def extract_plate_output(raw_output: Any, model: Any) -> np.ndarray:
    if isinstance(raw_output, dict):
        return np.asarray(raw_output["plate"])

    if isinstance(raw_output, (list, tuple)):
        output_names = list(getattr(model, "output_names", []))
        if "plate" in output_names:
            return np.asarray(raw_output[output_names.index("plate")])
        return np.asarray(raw_output[0])

    return np.asarray(raw_output)


def padded_box(
    box_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction

    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(image_width, int(np.ceil(x2 + pad_x))),
        min(image_height, int(np.ceil(y2 + pad_y))),
    )


def friendly_class_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def load_code_rules(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Code rules not found: {path}\n"
            "Run scripts/build_emirate_code_rules.py first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules")
    if not isinstance(rules, dict):
        raise ValueError(f"Invalid code rules file: {path}")
    return rules


def split_plate_text(
    text: str,
    emirate: str,
    code_rules: dict[str, Any],
) -> tuple[str | None, str | None, str]:
    cleaned = text.strip().upper()

    # Standard letter-coded plates.
    letter_match = LETTER_CODE_PATTERN.fullmatch(cleaned)
    if letter_match:
        return letter_match.group(1), letter_match.group(2), "letter_regex"

    # Numeric-only plates. Select the longest known training code that forms
    # a prefix and leaves at least two number digits.
    if cleaned.isdigit():
        emirate_rule = code_rules.get(emirate, {})
        numeric_codes = [
            str(code)
            for code in emirate_rule.get("numeric_codes", [])
        ]

        candidates = [
            code
            for code in numeric_codes
            if cleaned.startswith(code) and len(cleaned) - len(code) >= 2
        ]
        if candidates:
            code = max(candidates, key=len)
            return code, cleaned[len(code):], "train_known_numeric_code"

        preferred_length = emirate_rule.get(
            "preferred_numeric_code_length"
        )
        if (
            isinstance(preferred_length, int)
            and preferred_length > 0
            and len(cleaned) - preferred_length >= 2
        ):
            return (
                cleaned[:preferred_length],
                cleaned[preferred_length:],
                "train_preferred_numeric_length_fallback",
            )

        return None, None, "numeric_code_rule_missing"

    return None, None, "unrecognized_format"


def load_emirate_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], transforms.Compose]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    classes = checkpoint.get("classes")
    if not classes:
        class_to_idx = checkpoint.get("class_to_idx")
        if not class_to_idx:
            raise KeyError("Emirate checkpoint has no class metadata.")
        classes = [
            name
            for name, _ in sorted(
                class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )
    return model, list(classes), transform


@torch.no_grad()
def classify_emirate(
    crop_bgr: np.ndarray,
    model: nn.Module,
    classes: list[str],
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[str, float, list[dict[str, float | str]]]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = probabilities.max(dim=0)

    top_count = min(3, len(classes))
    top_probs, top_indices = torch.topk(probabilities, k=top_count)
    top_predictions = [
        {
            "emirate": friendly_class_name(classes[int(class_index)]),
            "confidence": float(probability),
        }
        for probability, class_index in zip(
            top_probs.cpu().tolist(),
            top_indices.cpu().tolist(),
        )
    ]

    return (
        friendly_class_name(classes[int(index)]),
        float(confidence),
        top_predictions,
    )


def draw_label(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    lines: list[str],
) -> None:
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54
    thickness = 2
    line_gap = 7
    padding = 7

    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in lines
    ]
    label_width = max(size[0] for size in sizes) + 2 * padding
    line_height = max(size[1] for size in sizes) + line_gap
    label_height = len(lines) * line_height + padding

    label_x1 = max(0, min(x1, image.shape[1] - label_width))
    preferred_y1 = y1 - label_height - 4
    label_y1 = (
        preferred_y1
        if preferred_y1 >= 0
        else min(y2 + 4, image.shape[0] - label_height)
    )

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(
        image,
        (label_x1, label_y1),
        (label_x1 + label_width, label_y1 + label_height),
        (0, 0, 0),
        -1,
    )

    first_y = label_y1 + padding + sizes[0][1]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (label_x1 + padding, first_y + index * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()

    image_path = args.image.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    emirate_checkpoint = args.emirate_checkpoint.resolve()
    code_rules_path = args.code_rules.resolve()

    ocr_model_path = (
        args.ocr_model.resolve()
        if args.ocr_model is not None
        else find_newest_best_model(args.ocr_run_root)
    )
    plate_config_path = find_plate_config(
        ocr_model_path,
        args.plate_config,
    )

    for label, path in {
        "input image": image_path,
        "RF-DETR checkpoint": detector_checkpoint,
        "OCR v2 checkpoint": ocr_model_path,
        "plate config": plate_config_path,
        "emirate checkpoint": emirate_checkpoint,
        "code rules": code_rules_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    code_rules = load_code_rules(code_rules_path)

    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_height, image_width = source_image.shape[:2]
    annotated = source_image.copy()

    run_dir = args.output_dir.resolve() / image_path.stem
    crop_dir = run_dir / "crops"
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading RF-DETR...")
    detector = RFDETRMedium(
        pretrain_weights=str(detector_checkpoint)
    )

    print("Loading OCR v2...")
    print(ocr_model_path)
    (
        postprocess_output,
        read_and_resize_plate_image,
        load_plate_config_from_yaml,
        load_keras_model,
    ) = load_ocr_components()
    plate_config = load_plate_config_from_yaml(plate_config_path)
    ocr_model = load_keras_model(ocr_model_path, plate_config)

    print("Loading emirate classifier...")
    (
        emirate_model,
        emirate_classes,
        emirate_transform,
    ) = load_emirate_model(emirate_checkpoint, device)

    print("Running final pipeline v2...")
    detections = detector.predict(
        str(image_path),
        threshold=float(args.threshold),
    )

    boxes = np.asarray(getattr(detections, "xyxy", []), dtype=float)
    confidences = np.asarray(
        getattr(detections, "confidence", []),
        dtype=float,
    )

    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes.reshape(1, 4)
    if confidences.ndim == 0 and confidences.size == 1:
        confidences = confidences.reshape(1)

    order = (
        np.argsort(-confidences)
        if len(confidences)
        else np.asarray([], dtype=int)
    )

    results: list[dict[str, Any]] = []

    for output_index, detection_index in enumerate(order, start=1):
        box = boxes[detection_index]
        detection_confidence = float(confidences[detection_index])

        x1 = max(0, min(int(round(box[0])), image_width - 1))
        y1 = max(0, min(int(round(box[1])), image_height - 1))
        x2 = max(0, min(int(round(box[2])), image_width - 1))
        y2 = max(0, min(int(round(box[3])), image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = padded_box(
            box,
            image_width,
            image_height,
            float(args.crop_padding),
        )
        crop = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        crop_path = crop_dir / f"plate_{output_index:02d}.png"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"Could not save crop: {crop_path}")

        emirate, emirate_confidence, emirate_top3 = classify_emirate(
            crop,
            emirate_model,
            emirate_classes,
            emirate_transform,
            device,
        )

        ocr_input = read_and_resize_plate_image(
            image_path=crop_path,
            img_height=plate_config.img_height,
            img_width=plate_config.img_width,
            image_color_mode=plate_config.image_color_mode,
            keep_aspect_ratio=plate_config.keep_aspect_ratio,
            interpolation_method=plate_config.interpolation,
            padding_color=plate_config.padding_color,
        )
        raw_output = ocr_model.predict(
            np.expand_dims(ocr_input, axis=0),
            verbose=0,
        )
        plate_output = extract_plate_output(raw_output, ocr_model)

        decoded = postprocess_output(
            model_output=plate_output,
            max_plate_slots=plate_config.max_plate_slots,
            model_alphabet=plate_config.alphabet,
            pad_char=plate_config.pad_char,
            remove_pad_char=True,
            return_confidence=True,
        )[0]

        plate_text = decoded.plate.strip().upper()
        character_confidences = (
            np.asarray(
                decoded.char_probs[: len(plate_text)],
                dtype=float,
            )
            if decoded.char_probs is not None and plate_text
            else np.asarray([], dtype=float)
        )
        ocr_confidence = (
            float(np.mean(character_confidences))
            if character_confidences.size
            else None
        )

        code, number, parsing_method = split_plate_text(
            plate_text,
            emirate,
            code_rules,
        )

        draw_label(
            annotated,
            (x1, y1, x2, y2),
            [
                f"{emirate} ({emirate_confidence:.3f})",
                f"Plate: {plate_text}  Det: {detection_confidence:.3f}",
                (
                    f"Code: {code or 'unparsed'}  "
                    f"Number: {number or 'unparsed'}  "
                    f"OCR: {ocr_confidence:.3f}"
                    if ocr_confidence is not None
                    else
                    f"Code: {code or 'unparsed'}  "
                    f"Number: {number or 'unparsed'}  OCR: N/A"
                ),
            ],
        )

        results.append(
            {
                "plate_index": output_index,
                "detection_confidence": detection_confidence,
                "detection_box_xyxy": [x1, y1, x2, y2],
                "padded_crop_box_xyxy": [
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                ],
                "crop_path": str(crop_path.resolve()),
                "emirate": emirate,
                "emirate_confidence": emirate_confidence,
                "emirate_top3": emirate_top3,
                "combined_plate_text": plate_text,
                "code": code,
                "number": number,
                "parsing_method": parsing_method,
                "format_valid": code is not None and number is not None,
                "ocr_mean_character_confidence": ocr_confidence,
                "ocr_character_confidences": [
                    float(value)
                    for value in character_confidences
                ],
            }
        )

    annotated_path = run_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    json_path = run_dir / "result.json"
    payload = {
        "source_image": str(image_path),
        "detector_checkpoint": str(detector_checkpoint),
        "ocr_v2_checkpoint": str(ocr_model_path),
        "emirate_checkpoint": str(emirate_checkpoint),
        "code_rules": str(code_rules_path),
        "detection_threshold": float(args.threshold),
        "plates_detected": len(results),
        "annotated_image": str(annotated_path.resolve()),
        "plates": results,
        "limitations": [
            "Code rules are derived only from recovered training filenames.",
            "The parser separates code and number but does not alter OCR text.",
            "A second RF-DETR box in a single-plate image should be treated as a detector false positive.",
            "End-to-end performance may be lower than crop-level component metrics.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nFinal-pipeline v2 result")
    print("------------------------")
    print(f"Plates detected: {len(results)}")
    for result in results:
        print(
            f"Plate {result['plate_index']}: "
            f"emirate={result['emirate']} "
            f"({result['emirate_confidence']:.4f}) | "
            f"text={result['combined_plate_text']} | "
            f"code={result['code']} | "
            f"number={result['number']} | "
            f"parse={result['parsing_method']} | "
            f"det={result['detection_confidence']:.4f} | "
            f"ocr={result['ocr_mean_character_confidence']}"
        )

    print(f"\nAnnotated image: {annotated_path}")
    print(f"JSON result:     {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
