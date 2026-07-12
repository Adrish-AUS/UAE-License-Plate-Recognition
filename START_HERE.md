# START HERE — Team Handoff

This folder is the clean release of the UAE licence-plate project.

## What is already complete

- Frozen dataset split and leakage checks
- RF-DETR Medium training and untouched test evaluation
- RF-DETR no-augmentation ablation
- Validation-selected confidence threshold and fixed-threshold test metrics
- RF-DETR speed benchmark
- OCR v1 evaluation
- OCR v2 improvement for numeric-prefix Abu Dhabi plates
- Seven-class emirate classifier
- Final RF-DETR + OCR v2 + emirate-classifier pipeline
- RT-DETR-L training and untouched test evaluation
- Curated metrics, figures, examples, and report notes

## What the report writer must still do

1. Add the teammate's exact YOLO test results to the detector comparison table.
2. Do not guess or estimate the YOLO row.
3. Use the same untouched 1,432-image test split when reporting YOLO.
4. Compare YOLO, RF-DETR Medium, and RT-DETR-L using:
   - mAP@50:95
   - mAP@50
   - mAP@75
   - precision
   - recall
   - speed
5. Explain that speed measurements are framework-dependent:
   - RF-DETR used a custom batch-size-1 timer.
   - RT-DETR-L used Ultralytics validation timing.
6. Include the RF-DETR augmentation ablation.
7. Report OCR and emirate results separately from detector mAP.
8. Present final-pipeline examples as qualitative demonstrations only.
9. Do not claim a full end-to-end test accuracy unless the entire test set is evaluated end to end.
10. Add weekly contribution details for every team member.

## Main verified results

### RF-DETR Medium
- mAP@50:95: 0.8664
- mAP@50: 0.9965
- mAP@75: 0.9844
- Precision: 0.9914
- Recall: 0.9921
- F1: 0.9917

### RT-DETR-L
- mAP@50:95: 0.863869
- mAP@50: 0.994354
- mAP@75: 0.980566
- Precision: 0.977432
- Recall: 0.992058
- Inference: 13.6 ms/image

### OCR v1
- Exact full-plate accuracy: 97.94%
- Character accuracy: 99.72%
- Plate-length accuracy: 99.71%

Use the OCR v2 metric JSON files for final OCR v2 numbers. Do not reuse the OCR v1 value as the OCR v2 result.

### Emirate classifier
- Accuracy: 99.59%
- Balanced accuracy: 97.71%
- Macro F1: 98.54%
- Weighted F1: 99.59%

## Important files

- `README.md` — setup, training, evaluation, and demo commands
- `docs/FINAL_TEAM_HANDOFF.md` — full technical handoff
- `docs/RFDETR_REPORT_SECTION.md` — RF-DETR report material
- `docs/OCR_END_TO_END_REPORT_ADDENDUM.md` — OCR and pipeline material
- `docs/EMIRATE_REPORT_ADDENDUM.md` — emirate classifier material
- `results/metrics/` — exact JSON and CSV results
- `results/figures/` — selected report-ready figures
- `results/examples/` — selected qualitative examples

## Final demo

The final application script is:

```powershell
python ".\scripts\run_final_uae_plate_pipeline_v2.py" --image "C:\full\path\to\vehicle_image.jpg"
```

Do not replace it with older demo scripts.

## Large checkpoints

Large checkpoints are shared separately and are not committed to the normal Git repository. See `docs/CHECKPOINTS.md` after the checkpoint links are added.
