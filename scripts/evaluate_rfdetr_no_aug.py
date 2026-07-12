import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRMedium

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    ROOT
    / "runs"
    / "rfdetr_medium_no_aug"
    / "checkpoint_best_ema.pth"
)

TEST_DIR = (
    ROOT
    / "datasets"
    / "uae_lp_v2_rfdetr_coco"
    / "test"
)

ANNOTATION_FILE = TEST_DIR / "_annotations.coco.json"

OUTPUT_DIR = ROOT / "results" / "rfdetr_medium_no_aug_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "coco_predictions.json"
METRICS_FILE = OUTPUT_DIR / "test_metrics.json"

# Reduce to 2 only if CUDA runs out of memory.
BATCH_SIZE = 4

# Keep low-confidence predictions for proper COCO AP calculation.
PREDICTION_THRESHOLD = 0.001

def calculate_iou(box_a, box_b):
    """Calculate IoU between two XYXY boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

def calculate_best_f1(coco_data, predictions, iou_threshold=0.5):
    """
    Sweep confidence thresholds and find the best detection-level F1 score
    using IoU >= 0.50.
    """
    ground_truth_by_image = defaultdict(list)

    for annotation in coco_data["annotations"]:
        x, y, width, height = annotation["bbox"]

        ground_truth_by_image[annotation["image_id"]].append(
            [x, y, x + width, y + height]
        )

    predictions_by_image = defaultdict(list)

    for prediction in predictions:
        x, y, width, height = prediction["bbox"]

        predictions_by_image[prediction["image_id"]].append(
            {
                "score": prediction["score"],
                "box": [x, y, x + width, y + height],
            }
        )

    scored_predictions = []
    total_ground_truth = len(coco_data["annotations"])

    for image_id, image_predictions in predictions_by_image.items():
        ground_truth_boxes = ground_truth_by_image.get(image_id, [])
        matched_ground_truth = set()

        image_predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        for prediction in image_predictions:
            best_iou = 0.0
            best_ground_truth_index = None

            for index, ground_truth_box in enumerate(ground_truth_boxes):
                if index in matched_ground_truth:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    ground_truth_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_index = index

            if (
                best_ground_truth_index is not None
                and best_iou >= iou_threshold
            ):
                matched_ground_truth.add(best_ground_truth_index)
                true_positive = 1
                false_positive = 0
            else:
                true_positive = 0
                false_positive = 1

            scored_predictions.append(
                (
                    prediction["score"],
                    true_positive,
                    false_positive,
                )
            )

    scored_predictions.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    scores = np.array([item[0] for item in scored_predictions])
    true_positives = np.cumsum(
        [item[1] for item in scored_predictions]
    )
    false_positives = np.cumsum(
        [item[2] for item in scored_predictions]
    )

    precision = true_positives / np.maximum(
        true_positives + false_positives,
        1,
    )
    recall = true_positives / max(total_ground_truth, 1)

    f1 = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )

    best_index = int(np.argmax(f1))

    return {
        "best_confidence_threshold": float(scores[best_index]),
        "precision_at_best_f1": float(precision[best_index]),
        "recall_at_best_f1": float(recall[best_index]),
        "best_f1_iou50": float(f1[best_index]),
        "iou_threshold": iou_threshold,
    }

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Test annotations missing: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    categories = coco_data["categories"]

    if len(categories) != 1:
        raise ValueError(
            f"Expected one class, found {len(categories)}."
        )

    license_plate_category_id = categories[0]["id"]

    image_records = sorted(
        coco_data["images"],
        key=lambda item: item["id"],
    )

    print("Loading checkpoint:")
    print(CHECKPOINT)

    model = RFDETRMedium(
        pretrain_weights=str(CHECKPOINT)
    )

    predictions = []

    for start in range(0, len(image_records), BATCH_SIZE):
        batch_records = image_records[
            start : start + BATCH_SIZE
        ]

        batch_paths = [
            str(TEST_DIR / Path(record["file_name"]).name)
            for record in batch_records
        ]

        for image_path in batch_paths:
            if not Path(image_path).exists():
                raise FileNotFoundError(
                    f"Missing test image: {image_path}"
                )

        batch_detections = model.predict(
            batch_paths,
            threshold=PREDICTION_THRESHOLD,
        )

        if not isinstance(batch_detections, list):
            batch_detections = [batch_detections]

        for record, detections in zip(
            batch_records,
            batch_detections,
        ):
            image_width = record["width"]
            image_height = record["height"]

            for box, confidence in zip(
                detections.xyxy,
                detections.confidence,
            ):
                x1, y1, x2, y2 = map(float, box)

                # Keep coordinates inside the original image.
                x1 = max(0.0, min(x1, image_width))
                y1 = max(0.0, min(y1, image_height))
                x2 = max(0.0, min(x2, image_width))
                y2 = max(0.0, min(y2, image_height))

                width = x2 - x1
                height = y2 - y1

                if width <= 0 or height <= 0:
                    continue

                predictions.append(
                    {
                        "image_id": record["id"],
                        "category_id": license_plate_category_id,
                        "bbox": [x1, y1, width, height],
                        "score": float(confidence),
                    }
                )

        completed = min(
            start + BATCH_SIZE,
            len(image_records),
        )

        print(
            f"Processed {completed}/{len(image_records)} images"
        )

    with PREDICTIONS_FILE.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2)

    print("\nRunning official COCO evaluation...")

    coco_ground_truth = COCO(str(ANNOTATION_FILE))
    coco_predictions = coco_ground_truth.loadRes(
        str(PREDICTIONS_FILE)
    )

    evaluator = COCOeval(
        coco_ground_truth,
        coco_predictions,
        iouType="bbox",
    )

    evaluator.params.imgIds = [
        record["id"] for record in image_records
    ]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "checkpoint": str(CHECKPOINT),
        "test_images": len(image_records),
        "test_boxes": len(coco_data["annotations"]),
        "predictions_retained": len(predictions),
        "mAP_50_95": float(evaluator.stats[0]),
        "mAP_50": float(evaluator.stats[1]),
        "mAP_75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_100": float(evaluator.stats[8]),
    }

    metrics.update(
        calculate_best_f1(
            coco_data,
            predictions,
            iou_threshold=0.5,
        )
    )

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nFinal test metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nSaved metrics to:\n{METRICS_FILE}")
    print(f"\nSaved predictions to:\n{PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()