# UAE Licence Plate Detection and Recognition

A computer-vision pipeline for UAE licence plate detection, plate-text recognition, and emirate classification.

The project compares multiple detector architectures on the same frozen train/validation/test split and extends the detector with OCR and seven-emirate classification.

## Project Overview

The final application pipeline is:

```text
Vehicle image
    ↓
RF-DETR Medium plate detector
    ↓
Expanded plate crop
    ↓
OCR v2 plate-text recognition
    ↓
Seven-class ResNet18 emirate classifier
    ↓
Train-derived code/number parser
    ↓
Annotated image + JSON output
```

Detector models evaluated:

- RF-DETR Medium
- RF-DETR Medium without augmentation
- RT-DETR-L
- YOLO baseline developed by another team member

## Dataset

The frozen detector dataset contains:

| Split | Images | Licence-plate boxes |
|---|---:|---:|
| Train | 6,738 | 9,415 |
| Validation | 1,440 | 1,525 |
| Test | 1,432 | 1,511 |
| **Total** | **9,610** | **12,451** |

The detector task uses one class:

```text
license_plate
```

Cross-split duplicate checks found:

- Exact duplicates: 0
- Flagged perceptual duplicate candidates: 0

Dataset files are not included in this repository. See `docs/DATASET_ATTRIBUTION.md` and `configs/` for dataset structure and attribution.

## Main Results

### Detector comparison

| Model | mAP@50:95 | mAP@50 | mAP@75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| RF-DETR Medium | 0.8664 | 0.9965 | 0.9844 | 0.9914* | 0.9921* |
| RT-DETR-L | 0.8639 | 0.9944 | 0.9806 | 0.9774 | 0.9921 |
| YOLO baseline | Add teammate result | Add teammate result | Add teammate result | Add teammate result | Add teammate result |

\*RF-DETR precision and recall were measured at a confidence threshold selected on the validation set and frozen before test evaluation.

### RF-DETR augmentation ablation

| Configuration | mAP@50:95 | mAP@50 | mAP@75 | AR@100 |
|---|---:|---:|---:|---:|
| With augmentation | 0.8664 | 0.9965 | 0.9844 | 0.9027 |
| Without augmentation | 0.8666 | 0.9903 | 0.9777 | 0.9060 |

The overall mAP@50:95 difference was negligible. The augmented model performed better at IoU 0.50 and 0.75, while the non-augmented model produced slightly higher average recall.

### OCR v1

| Metric | Result |
|---|---:|
| Test crops | 1,017 |
| Exact full-plate accuracy | 97.94% |
| Character accuracy | 99.72% |
| Plate-length accuracy | 99.71% |

OCR v2 was later introduced to improve numeric-prefix Abu Dhabi plates. Exact OCR v2 results are stored under:

```text
results/metrics/ocr/
```

### Emirate classifier

| Metric | Result |
|---|---:|
| Test crops | 1,466 |
| Accuracy | 99.59% |
| Balanced accuracy | 97.71% |
| Macro F1 | 98.54% |
| Weighted F1 | 99.59% |

The classifier predicts:

- Dubai
- Abu Dhabi
- Sharjah
- Ajman
- Fujairah
- Ras Al Khaimah
- Umm Al Quwain

Because the dataset is imbalanced, balanced accuracy and macro F1 are more informative than accuracy alone.

## Repository Structure

```text
UAE-License-Plate-Recognition-Release/
├── README.md
├── START_HERE.md
├── requirements.txt
├── .gitignore
├── configs/
├── docs/
├── scripts/
└── results/
    ├── metrics/
    ├── figures/
    └── examples/
```

### Important scripts

```text
scripts/
├── preprocess_dataset.py
├── preprocessing_utils.py
├── validate_dataset.py
├── check_split_leakage.py
├── prepare_rfdetr_dataset.py
├── train_rfdetr_main.py
├── train_rfdetr_no_aug.py
├── evaluate_rfdetr_test.py
├── evaluate_rfdetr_no_aug.py
├── evaluate_fixed_threshold.py
├── benchmark_rfdetr.py
├── export_rfdetr_curves.py
├── train_rtdetr_l.py
├── build_ocr_dataset.py
├── build_ocr_v2_dataset.py
├── evaluate_ocr_test.py
├── build_emirate_dataset.py
├── finalize_emirate_labels.py
├── train_emirate_classifier.py
├── build_emirate_code_rules.py
└── run_final_uae_plate_pipeline_v2.py
```

## Environment Setup

Create and activate a Python environment before installing the dependencies.

### Windows PowerShell

```powershell
python -m venv rfdetr_env
.\rfdetr_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify PyTorch and CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Expected Dataset Structure

The release does not contain the dataset. The main YOLO-format dataset should follow:

```text
datasets/
└── uae_lp_v2_yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

RF-DETR uses the corresponding COCO-format dataset produced by:

```powershell
python ".\scripts\prepare_rfdetr_dataset.py"
```

## Dataset Validation

```powershell
python ".\scripts\validate_dataset.py"
python ".\scripts\check_split_leakage.py"
```

## RF-DETR Training

### Main augmented model

```powershell
python ".\scripts\train_rfdetr_main.py"
```

### No-augmentation ablation

```powershell
python ".\scripts\train_rfdetr_no_aug.py"
```

### Test evaluation

```powershell
python ".\scripts\evaluate_rfdetr_test.py"
python ".\scripts\evaluate_fixed_threshold.py"
python ".\scripts\benchmark_rfdetr.py"
```

## RT-DETR-L Training

Validate the dataset and model setup:

```powershell
python ".\scripts\train_rtdetr_l.py" --check-only
```

Train and automatically evaluate the best checkpoint on the test split:

```powershell
python ".\scripts\train_rtdetr_l.py"
```

Main RT-DETR-L configuration:

- Input size: 576 × 576
- Maximum epochs: 20
- Physical batch size: 4
- Optimizer: AdamW
- Initial learning rate: 1e-4
- Early-stopping patience: 4
- Seed: 486

## OCR

### Build OCR v1 data

```powershell
python ".\scripts\build_ocr_dataset.py"
```

### Build OCR v2 data

```powershell
python ".\scripts\build_ocr_v2_dataset.py"
```

OCR v2 preserves the frozen split and adds strict numeric-prefix Abu Dhabi examples.

### Evaluate OCR

```powershell
python ".\scripts\evaluate_ocr_test.py" --help
```

See `docs/OCR_END_TO_END_REPORT_ADDENDUM.md` for the exact evaluation subsets and interpretation.

## Emirate Classification

```powershell
python ".\scripts\build_emirate_dataset.py"
python ".\scripts\finalize_emirate_labels.py"
python ".\scripts\train_emirate_classifier.py"
python ".\scripts\build_emirate_code_rules.py"
```

## Final Demo

The final application uses RF-DETR Medium, OCR v2, the seven-class emirate classifier, and train-derived parsing rules.

```powershell
python ".\scripts\run_final_uae_plate_pipeline_v2.py" `
    --image "C:\full\path\to\vehicle_image.jpg"
```

Expected outputs:

- Annotated vehicle image
- Detected plate crop
- Detector confidence
- OCR text
- Parsed plate code
- Parsed registration number
- Predicted emirate
- Emirate confidence
- JSON result

Selected examples are available under:

```text
results/examples/final_pipeline/
```

## Checkpoints

Large checkpoints are not committed directly to normal Git history.

Required checkpoints:

```text
RF-DETR augmented:
runs/rfdetr_medium_main/checkpoint_best_total.pth

RF-DETR no augmentation:
runs/rfdetr_medium_no_aug/checkpoint_best_ema.pth

RT-DETR-L:
runs/rtdetr_l_main/weights/best.pt

OCR v2:
newest best.keras under runs/ocr_uae_v2_retry/

Emirate ResNet18:
runs/emirate_resnet18_main/best.pt
```

See `docs/CHECKPOINTS.md` for download and placement instructions.

## Results and Report Material

Exact metrics:

```text
results/metrics/
```

Report-ready figures:

```text
results/figures/
```

Selected qualitative examples:

```text
results/examples/
```

Technical report sections and handoff notes:

```text
docs/
```

## Important Evaluation Notes

- Detector mAP, OCR accuracy, and emirate-classification F1 measure different tasks and should not be compared directly.
- Final-pipeline examples are qualitative demonstrations, not a full end-to-end test accuracy.
- RF-DETR precision and recall use a validation-selected confidence threshold that was frozen before test evaluation.
- RT-DETR speed was reported by the Ultralytics validation pipeline, while RF-DETR speed used a custom batch-size-1 benchmark. Speed values are therefore implementation-dependent.
- Some source annotations contain mistakes, including non-plate objects labelled as plates.
- The emirate dataset is strongly imbalanced toward Dubai.
- Umm Al Quwain has few test examples, so its class-level result is less statistically reliable.
- OCR quality can decrease when detector crops are loose, shifted, blurred, or incomplete.

## Team Handoff

A one-page starting guide is provided in:

```text
START_HERE.md
```

The complete contribution and report-writing handoff is available in:

```text
docs/FINAL_TEAM_HANDOFF.md
```
