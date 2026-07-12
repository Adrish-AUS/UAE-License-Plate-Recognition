# RF-DETR Experiment Summary

## Dataset

- Task: UAE licence plate detection
- Classes: 1 (`license_plate`)
- Total images: 9,610
- Total bounding boxes: 12,451
- Training images: 6,738
- Validation images: 1,440
- Test images: 1,432
- Training boxes: 9,415
- Validation boxes: 1,525
- Test boxes: 1,511
- Exact cross-split duplicates: 0
- Perceptual duplicate candidates: 0

## Model Configuration

- Model: RF-DETR Medium
- Pretraining: Official pretrained RF-DETR Medium weights
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB VRAM
- Maximum epochs: 20
- Early stopping patience: 4 validation checks
- Learning rate: 1e-4
- Physical batch size: 4
- Gradient accumulation steps: 4
- Effective batch size: 16
- Random seed: 486
- Number of parameters: 33.37 million

## Training Augmentations

The main model used the following training-only augmentations:

- Random brightness and contrast changes
- Gaussian blur
- Motion blur
- Rotation up to 5 degrees
- Mild perspective distortion

Horizontal flipping was not used because mirrored licence-plate text is unrealistic.

## Main RF-DETR Test Results

The augmented RF-DETR model was evaluated on the untouched 1,432-image test set.

| Metric | Result |
|---|---:|
| mAP@50:95 | 0.8664 |
| mAP@50 | 0.9965 |
| mAP@75 | 0.9844 |
| AP medium | 0.8495 |
| AP large | 0.8736 |
| AR@100 | 0.9027 |

COCO reported AP small as -1 because the test set contained no ground-truth objects within the COCO small-object area range.

## Fixed-Threshold Precision, Recall and F1

The confidence threshold was selected using the validation set and then frozen before test evaluation.

- Selected confidence threshold: 0.8415
- Validation precision: 0.9895
- Validation recall: 0.9895
- Validation F1: 0.9895

Test results at the frozen threshold:

| Metric | Result |
|---|---:|
| True positives | 1,499 |
| False positives | 13 |
| False negatives | 12 |
| Precision | 0.9914 |
| Recall | 0.9921 |
| F1 | 0.9917 |

## Augmentation Ablation

| Configuration | mAP@50:95 | mAP@50 | mAP@75 | AR@100 |
|---|---:|---:|---:|---:|
| With augmentation | 0.8664 | 0.9965 | 0.9844 | 0.9027 |
| Without augmentation | 0.8666 | 0.9903 | 0.9777 | 0.9060 |

Removing augmentation changed mAP@50:95 by only 0.00012. The difference is negligible. The augmented model achieved higher mAP@50 and mAP@75, while the non-augmented model produced slightly higher average recall. The ablation therefore does not show a clear aggregate mAP advantage from the selected augmentations.

## Inference Speed

Batch-size-1 inference was measured over 200 test images after 10 warm-up images.

| Mode | Mean latency | Median latency | P95 latency | FPS |
|---|---:|---:|---:|---:|
| Standard FP32 | 95.68 ms | 95.20 ms | 108.34 ms | 10.45 |
| Optimized FP32 | 87.12 ms | 85.52 ms | 102.54 ms | 11.48 |

Inference optimization reduced mean latency by approximately 9%.

## Qualitative Findings

The model generally produced tight boxes for clear foreground plates. Difficult cases included distant plates, blur, glare, partial visibility and unusual viewing angles. Manual inspection also identified incorrect ground-truth annotations, including a road exit sign labelled as a licence plate. These annotation errors can lower measured recall and IoU despite correct model behaviour.

## Important Files

- Augmented checkpoint:
  `runs/rfdetr_medium_main/checkpoint_best_total.pth`

- No-augmentation checkpoint:
  `runs/rfdetr_medium_no_aug/checkpoint_best_ema.pth`

- Augmented test metrics:
  `results/rfdetr_medium_test/test_metrics.json`

- No-augmentation test metrics:
  `results/rfdetr_medium_no_aug_test/test_metrics.json`

- Fixed-threshold metrics:
  `results/rfdetr_medium_test/fixed_threshold_metrics.json`

- Standard speed benchmark:
  `results/rfdetr_medium_test/speed_metrics.json`

- Optimized speed benchmark:
  `results/rfdetr_medium_test/speed_metrics_optimized_fp32.json`

- Training curves:
  `results/rfdetr_training_curves/`

- Selected qualitative examples:
  `results/final_examples/`