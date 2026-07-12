# REPO CLEANUP AND PUBLISH PLAN

## Deliver three separate things

### A. GitHub repository
Purpose: reproducible code, README, small sample outputs, and report-ready figures.

Include:
- source code
- configs
- dataset preparation scripts
- train/evaluate commands
- small JSON/CSV metrics
- selected figures
- selected demo examples
- attribution and licence information

Exclude:
- raw datasets
- generated crop datasets
- all training caches
- full TensorBoard logs
- duplicate demo folders
- large checkpoints unless Git LFS or Releases are used

### B. Team handoff ZIP
Purpose: give the report writer everything needed without searching the repo.

Include:
- final handoff document
- report sections
- exact metrics JSON/CSV files
- final figures
- selected qualitative examples
- essential scripts
- environment snapshot
- checkpoint path manifest

### C. Checkpoints archive
Purpose: reproducibility and demo.

Share separately because it may be several GB.

## Suggested public repository layout

```text
UAE-License-Plate-Recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── data.yaml
├── DATASET_ATTRIBUTION.md
├── configs/
├── scripts/
│   ├── data/
│   ├── rfdetr/
│   ├── rtdetr/
│   ├── ocr/
│   ├── emirate/
│   └── demo/
├── results/
│   ├── metrics/
│   ├── figures/
│   └── examples/
├── docs/
│   ├── FINAL_TEAM_HANDOFF.md
│   └── report_sections/
└── archive/                 # optional locally; usually not committed
```

A full reorganization is optional this close to submission. At minimum:
- keep only one final demo script in the README;
- rename `train_rtdetr_l_fixed.py` to `train_rtdetr_l.py` after moving the broken version to archive;
- move old demo scripts to `scripts/legacy/`;
- move duplicate demo outputs to `results/archive/`.

## Suggested .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
rfdetr_env/

# Datasets and generated crops
datasets/
annotations/
data/raw/
data/processed/

# Large model files
*.pt
*.pth
*.keras
*.onnx
*.engine

# Training logs and caches
tensorboard_logs/
logs/
wandb/
runs/**/weights/
*.cache

# Local/editor files
.vscode/
.idea/
.DS_Store
Thumbs.db

# Generated handoff archives
team_handoff_pack/
team_handoff_pack.zip
```

Do not ignore the small final JSON/CSV metrics and selected figures that the report needs.

## Current report documents that need updating

1. `RFDETR_REPORT_SECTION.md`
   - Keep the RF-DETR methodology, ablation, fixed-threshold metrics, and speed.
   - Replace the final paragraph saying OCR cannot be evaluated. OCR and emirate recognition were later implemented and evaluated separately.

2. `OCR_END_TO_END_REPORT_ADDENDUM.md`
   - It currently documents OCR v1 and the S10198 example.
   - Add OCR v2, numeric Abu Dhabi motivation, exact v2 test metrics, and the final seven-emirate pipeline.

3. `EMIRATE_REPORT_ADDENDUM.md`
   - Results are usable.
   - Replace "CCT-S v1" with the final OCR v2 checkpoint.
   - Mention the train-derived emirate code parser.

4. Add a new RT-DETR-L section.
   - Include configuration, same split, 20 epochs, 576 input, test metrics, and speed.
   - Present it as a detector comparison, not as the final demo model.

5. Add one final detector comparison table after the YOLO teammate supplies their exact row.
