import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rfdetr import RFDETRMedium


ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    ROOT
    / "runs"
    / "rfdetr_medium_main"
    / "checkpoint_best_total.pth"
)

DATASET_DIR = (
    ROOT
    / "datasets"
    / "uae_lp_v2_rfdetr_coco"
)

VALID_DIR = DATASET_DIR / "valid"
VALID_ANNOTATIONS = VALID_DIR / "_annotations.coco.json"

TEST_DIR = DATASET_DIR / "test"
TEST_ANNOTATIONS = TEST_DIR / "_annotations.coco.json"

TEST_PREDICTIONS = (
    ROOT
    / "results"
    / "rfdetr_medium_test"
    / "coco_predictions.json"
)

OUTPUT_DIR = ROOT / "results" / "rfdetr_medium_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_PREDICTIONS = OUTPUT_DIR / "validation_predictions.json"
OUTPUT_METRICS = OUTPUT_DIR / "fixed_threshold_metrics.json"

BATCH_SIZE = 4
LOW_THRESHOLD = 0.001
IOU_THRESHOLD = 0.50


def calculate_iou(box_a, box_b):
    """Calculate IoU between two XYXY boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(
        0.0,
        y2 - y1,
    )

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0,
        box_a[3] - box_a[1],
    )

    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0,
        box_b[3] - box_b[1],
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def xywh_to_xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


def run_validation_inference():
    """Generate low-threshold predictions for the validation split."""
    with VALID_ANNOTATIONS.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    image_records = sorted(
        coco_data["images"],
        key=lambda item: item["id"],
    )

    category_id = coco_data["categories"][0]["id"]

    print("Loading RF-DETR checkpoint:")
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
            str(
                VALID_DIR
                / Path(record["file_name"]).name
            )
            for record in batch_records
        ]

        detections_batch = model.predict(
            batch_paths,
            threshold=LOW_THRESHOLD,
        )

        if not isinstance(detections_batch, list):
            detections_batch = [detections_batch]

        for record, detections in zip(
            batch_records,
            detections_batch,
        ):
            image_width = record["width"]
            image_height = record["height"]

            for box, confidence in zip(
                detections.xyxy,
                detections.confidence,
            ):
                x1, y1, x2, y2 = map(float, box)

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
                        "category_id": category_id,
                        "bbox": [x1, y1, width, height],
                        "score": float(confidence),
                    }
                )

        completed = min(
            start + BATCH_SIZE,
            len(image_records),
        )

        print(
            f"Validation images processed: "
            f"{completed}/{len(image_records)}"
        )

    with VALID_PREDICTIONS.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(predictions, file, indent=2)

    return coco_data, predictions


def prepare_ground_truth(coco_data):
    ground_truth = defaultdict(list)

    for annotation in coco_data["annotations"]:
        ground_truth[annotation["image_id"]].append(
            xywh_to_xyxy(annotation["bbox"])
        )

    return ground_truth


def find_best_validation_threshold(
    coco_data,
    predictions,
):
    """
    Find the confidence threshold with the highest validation F1
    using IoU >= 0.50.
    """
    ground_truth = prepare_ground_truth(coco_data)
    predictions_by_image = defaultdict(list)

    for prediction in predictions:
        predictions_by_image[
            prediction["image_id"]
        ].append(
            {
                "score": prediction["score"],
                "box": xywh_to_xyxy(
                    prediction["bbox"]
                ),
            }
        )

    scored_predictions = []

    for image_id, image_predictions in (
        predictions_by_image.items()
    ):
        ground_truth_boxes = ground_truth.get(
            image_id,
            [],
        )

        matched_ground_truth = set()

        image_predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        for prediction in image_predictions:
            best_iou = 0.0
            best_index = None

            for index, ground_truth_box in enumerate(
                ground_truth_boxes
            ):
                if index in matched_ground_truth:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    ground_truth_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if (
                best_index is not None
                and best_iou >= IOU_THRESHOLD
            ):
                matched_ground_truth.add(best_index)
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

    scores = np.array(
        [item[0] for item in scored_predictions]
    )

    true_positives = np.cumsum(
        [item[1] for item in scored_predictions]
    )

    false_positives = np.cumsum(
        [item[2] for item in scored_predictions]
    )

    total_ground_truth = len(
        coco_data["annotations"]
    )

    precision = true_positives / np.maximum(
        true_positives + false_positives,
        1,
    )

    recall = true_positives / max(
        total_ground_truth,
        1,
    )

    f1 = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12,
        )
    )

    best_index = int(np.argmax(f1))

    return {
        "threshold": float(scores[best_index]),
        "validation_precision": float(
            precision[best_index]
        ),
        "validation_recall": float(
            recall[best_index]
        ),
        "validation_f1": float(f1[best_index]),
    }


def evaluate_at_fixed_threshold(
    coco_data,
    predictions,
    confidence_threshold,
):
    """
    Evaluate predictions at one fixed confidence threshold.
    """
    ground_truth = prepare_ground_truth(coco_data)
    predictions_by_image = defaultdict(list)

    for prediction in predictions:
        if prediction["score"] < confidence_threshold:
            continue

        predictions_by_image[
            prediction["image_id"]
        ].append(
            {
                "score": prediction["score"],
                "box": xywh_to_xyxy(
                    prediction["bbox"]
                ),
            }
        )

    true_positives = 0
    false_positives = 0

    for image_id, image_predictions in (
        predictions_by_image.items()
    ):
        ground_truth_boxes = ground_truth.get(
            image_id,
            [],
        )

        matched_ground_truth = set()

        image_predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        for prediction in image_predictions:
            best_iou = 0.0
            best_index = None

            for index, ground_truth_box in enumerate(
                ground_truth_boxes
            ):
                if index in matched_ground_truth:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    ground_truth_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if (
                best_index is not None
                and best_iou >= IOU_THRESHOLD
            ):
                matched_ground_truth.add(best_index)
                true_positives += 1
            else:
                false_positives += 1

    total_ground_truth = len(
        coco_data["annotations"]
    )

    false_negatives = (
        total_ground_truth - true_positives
    )

    precision = (
        true_positives
        / max(
            true_positives + false_positives,
            1,
        )
    )

    recall = (
        true_positives
        / max(
            true_positives + false_negatives,
            1,
        )
    )

    f1 = (
        2 * precision * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {CHECKPOINT}"
        )

    if not TEST_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Test predictions missing: "
            f"{TEST_PREDICTIONS}"
        )

    validation_data, validation_predictions = (
        run_validation_inference()
    )

    validation_result = (
        find_best_validation_threshold(
            validation_data,
            validation_predictions,
        )
    )

    print("\nValidation-selected threshold:")
    for key, value in validation_result.items():
        print(f"{key}: {value}")

    with TEST_ANNOTATIONS.open(
        "r",
        encoding="utf-8",
    ) as file:
        test_data = json.load(file)

    with TEST_PREDICTIONS.open(
        "r",
        encoding="utf-8",
    ) as file:
        test_predictions = json.load(file)

    test_result = evaluate_at_fixed_threshold(
        test_data,
        test_predictions,
        validation_result["threshold"],
    )

    final_result = {
        "checkpoint": str(CHECKPOINT),
        "iou_threshold": IOU_THRESHOLD,
        "confidence_threshold_source": "validation split",
        **validation_result,
        **test_result,
    }

    with OUTPUT_METRICS.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_result, file, indent=2)

    print("\nFinal test metrics using the frozen threshold:")
    for key, value in final_result.items():
        print(f"{key}: {value}")

    print("\nSaved to:")
    print(OUTPUT_METRICS)


if __name__ == "__main__":
    main()