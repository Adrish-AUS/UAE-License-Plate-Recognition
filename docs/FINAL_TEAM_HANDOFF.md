
# FINAL TEAM HANDOFF - UAE Licence Plate Project

## 1. What I completed

### Dataset and split integrity
- Used the frozen one-class detector dataset with 9,610 images and 12,451 plate boxes.
- Preserved the same split for detector comparison:
  - Train: 6,738 images, 9,415 boxes
  - Validation: 1,440 images, 1,525 boxes
  - Test: 1,432 images, 1,511 boxes
- Confirmed no exact cross-split duplicates and no flagged perceptual duplicate candidates.
- Converted the YOLO annotations to COCO format for RF-DETR.

### RF-DETR Medium
- Fine-tuned the main augmented RF-DETR Medium model.
- Ran a controlled no-augmentation ablation with the same model, split, seed, learning rate, batch configuration, and evaluation protocol.
- Selected the confidence threshold on validation and froze it before test evaluation.
- Benchmarked standard and optimized FP32 inference.

Verified RF-DETR test results:
- mAP@50:95: 0.8664
- mAP@50: 0.9965
- mAP@75: 0.9844
- Precision at frozen threshold: 0.9914
- Recall at frozen threshold: 0.9921
- F1 at frozen threshold: 0.9917
- Optimized FP32 latency: 87.12 ms/image
- Optimized FP32 throughput: 11.48 FPS

Ablation:
- With augmentation mAP@50:95: 0.8664
- Without augmentation mAP@50:95: 0.8666
- Difference: negligible

### OCR
- Built the first OCR crop dataset from reliable filename transcripts.
- Fine-tuned CCT-S v1 and evaluated it on 1,017 ground-truth plate crops.
- Initial OCR result:
  - Exact plate accuracy: 97.94%
  - Character accuracy: 99.72%
  - Plate-length accuracy: 99.71%
- Found a real format gap: numeric-prefix Abu Dhabi plates, such as 7-59700, were underrepresented because the first OCR dataset mainly accepted letter-prefix labels.
- Built OCR v2 by adding strict numeric-prefix Abu Dhabi labels while preserving the frozen split.
- Fine-tuned OCR v2 from the original best checkpoint.
- Verified the previously failing Abu Dhabi examples now read the leading 7 correctly.
- Use the exact OCR v2 JSON files for final numeric results. Do not invent or copy the old 97.94% value as the OCR v2 combined result unless the JSON confirms it.

### Emirate classifier
- Recovered emirate-specific labels from the original 51-class source annotations.
- Mapped the source labels to all seven emirates:
  Dubai, Abu Dhabi, Sharjah, Ajman, Fujairah, Ras Al Khaimah, and Umm Al Quwain.
- Built 10,452 plate crops while preserving the frozen detector split.
- Trained an ImageNet-pretrained ResNet18 with 224 x 224 crops and balanced sampling.

Verified emirate classifier test results:
- Test crops: 1,466
- Accuracy: 99.59%
- Balanced accuracy: 97.71%
- Macro F1: 98.54%
- Weighted F1: 99.59%
- Best epoch: 4

Limitation:
- Umm Al Quwain has only 15 test crops, so its class-level result has high sampling uncertainty.

### Final recognition pipeline
Final application:
RF-DETR Medium -> predicted plate crop -> OCR v2 -> emirate ResNet18 -> train-derived code/number parser -> annotated image and JSON.

Use only:
`scripts/run_final_uae_plate_pipeline_v2.py`

Do not use the older demo scripts for the final submission.

### RT-DETR-L comparison
- Trained Ultralytics RT-DETR-L on the same frozen YOLO-format split.
- Evaluated on the untouched 1,432-image test set with 1,511 plate instances.

Verified RT-DETR-L test results:
- mAP@50:95: 0.863869
- mAP@50: 0.994354
- mAP@75: 0.980566
- Precision: 0.977432
- Recall: 0.992058
- Inference: 13.6 ms/image
- Approximate inference-only throughput: 73.5 FPS

RT-DETR-L is an additional detector comparison. It is not the final application detector.

## 2. Why these additions were made

- RF-DETR versus YOLO provides the required main detector comparison.
- The augmentation on/off RF-DETR experiment satisfies the ablation requirement.
- RT-DETR-L adds another transformer detector under the same split.
- OCR changes the output from "a plate exists here" to a readable code and number.
- The emirate classifier adds a UAE-specific output using recovered verified annotations.
- OCR v2 fixes a documented dataset-format weakness rather than hiding the Abu Dhabi failures.
- The final pipeline gives a stronger demo while keeping detector, OCR, and emirate metrics separate.

## 3. What the report writer must still do

1. Add the final YOLO test row using the same 1,432-image test split.
2. Create one detector comparison table with:
   - YOLO
   - RF-DETR Medium
   - RT-DETR-L
3. Compare mAP@50:95, mAP@50, mAP@75, precision, recall, and speed.
4. Explain that speed values are implementation- and framework-dependent:
   RF-DETR was benchmarked using a custom batch-size-1 timer, while the RT-DETR value comes from Ultralytics validation.
5. Include the RF-DETR augmentation ablation table.
6. Include separate component results:
   - OCR v1 and OCR v2
   - Seven-class emirate classifier
7. Present final-pipeline examples as qualitative end-to-end demonstrations, not as a full end-to-end test accuracy.
8. Include limitations:
   - one-class detector dataset
   - source annotation errors
   - strong Dubai imbalance
   - few Umm Al Quwain test crops
   - OCR format dependence
   - predicted detector crops can reduce recognition quality
9. Write the weekly contribution section required by the project brief.
10. Ensure every figure and table is numbered and referenced in the text.

## 4. Recommended report figures

Keep the report near 8 pages. Suggested figures:
- Dataset/preprocessing or split summary
- RF-DETR training curve
- RF-DETR qualitative predictions/failures
- RF-DETR augmentation ablation table
- Three-detector comparison table or plot
- Emirate confusion matrix
- OCR correct/incorrect examples
- One final annotated end-to-end example

Do not fill the report with every generated plot.

## 5. Required README commands

The README must explain:
- environment installation
- dataset location and expected structure
- RF-DETR training and test commands
- RF-DETR no-augmentation ablation command
- RT-DETR-L training and test command
- OCR dataset preparation, fine-tuning, and evaluation
- emirate recovery, training, and evaluation
- final pipeline inference command

Final demo command:
```powershell
python ".\scripts\run_final_uae_plate_pipeline_v2.py" --image "C:\full\path\to\image.jpg"
```

## 6. Files to treat as final

### Main code
- `scripts/preprocess_dataset.py`
- `scripts/preprocessing_utils.py`
- `scripts/validate_dataset.py`
- `scripts/check_split_leakage.py`
- `scripts/prepare_rfdetr_dataset.py`
- `scripts/train_rfdetr_main.py`
- `scripts/train_rfdetr_no_aug.py`
- `scripts/evaluate_rfdetr_test.py`
- `scripts/evaluate_rfdetr_no_aug.py`
- `scripts/evaluate_fixed_threshold.py`
- `scripts/benchmark_rfdetr.py`
- `scripts/export_rfdetr_curves.py`
- `scripts/build_ocr_dataset.py`
- `scripts/build_ocr_v2_dataset.py`
- `scripts/evaluate_ocr_test.py`
- `scripts/build_emirate_dataset.py`
- `scripts/finalize_emirate_labels.py`
- `scripts/train_emirate_classifier.py`
- `scripts/build_emirate_code_rules.py`
- `scripts/train_rtdetr_l_fixed.py`
- `scripts/run_final_uae_plate_pipeline_v2.py`

### Final results
- `results/rfdetr_medium_test/`
- `results/rfdetr_medium_no_aug_test/`
- `results/rfdetr_training_curves/`
- `results/final_examples/`
- `results/ocr_uae_test/`
- `results/ocr_uae_v2_original_test/`
- `results/ocr_uae_v2_abu_dhabi_test/`
- `results/ocr_uae_v2_combined_test/` if present
- `results/emirate_recovery/recovery_summary.json`
- `results/emirate_recovery/previews/`
- `results/emirate_code_rules.json`
- `results/final_pipeline_demo_v2/`
- `results/rtdetr_l_test_metrics.json`
- `results/rtdetr_training_setup.json`
- `runs/emirate_resnet18_main/test_metrics.json`
- `runs/emirate_resnet18_main/classification_report.csv`
- `runs/emirate_resnet18_main/confusion_matrix_normalized.png`
- `runs/emirate_resnet18_main/training_curves.png`
- `runs/rtdetr_l_main/results.csv`
- `runs/rtdetr_l_main/results.png`
- `runs/rtdetr_l_test/`

## 7. Files that are legacy or one-time debugging artifacts

Move these to `archive/` or exclude them from the public repo:
- `scripts/run_full_uae_plate_demo.py`
- `scripts/run_full_uae_plate_demo_v2.py`
- `scripts/run_final_uae_plate_pipeline.py`
- `scripts/run_rfdetr_ocr_demo.py`
- `scripts/demo_rfdetr_ocr.py`
- `scripts/evaluate_abu_dhabi_numeric_ocr.py`
- the old broken `scripts/train_rtdetr_l.py`
- `scripts/build_notebook.py`
- `scripts/fix_ocr_v2_paths.py`
- `results/end_to_end_demo/`
- `results/full_pipeline_demo/`
- `results/full_pipeline_demo_v2/`
- `results/final_pipeline_demo/`
- `results/ocr_demo/`

Do not delete them until the submission is complete. Archive them first.

## 8. Checkpoints to share separately

Do not commit large checkpoints directly to normal Git history.

- RF-DETR:
  `runs/rfdetr_medium_main/checkpoint_best_total.pth`
- RF-DETR no augmentation:
  `runs/rfdetr_medium_no_aug/checkpoint_best_ema.pth`
- RT-DETR-L:
  `runs/rtdetr_l_main/weights/best.pt`
- OCR v2:
  newest `best.keras` under `runs/ocr_uae_v2_retry/`
- Emirate classifier:
  `runs/emirate_resnet18_main/best.pt`

Upload them to Drive, OneDrive, a GitHub Release, or Git LFS and link them from README.

## 9. Important reporting rules

- RF-DETR, RT-DETR-L, and YOLO are detector models.
- OCR and the emirate classifier are downstream shared modules, not part of RF-DETR itself.
- Do not compare detector mAP directly with OCR accuracy or emirate macro F1.
- Do not claim a full end-to-end accuracy unless the whole test set was evaluated end to end.
- Use the raw JSON/CSV values for all tables.
- State clearly that OCR v2 was introduced after numeric Abu Dhabi failures were observed.
