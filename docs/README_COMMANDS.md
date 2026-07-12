# README COMMANDS - READY TO ADAPT

## Environment

```powershell
& "$env:USERPROFILE\rfdetr_env\Scripts\Activate.ps1"
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install dependencies using the final pinned requirements file.

## Dataset validation

```powershell
python ".\scripts\validate_dataset.py"
python ".\scripts\check_split_leakage.py"
```

## RF-DETR training

```powershell
python ".\scripts\train_rfdetr_main.py"
```

## RF-DETR no-augmentation ablation

```powershell
python ".\scripts\train_rfdetr_no_aug.py"
```

## RF-DETR test evaluation

```powershell
python ".\scripts\evaluate_rfdetr_test.py"
python ".\scripts\evaluate_fixed_threshold.py"
python ".\scripts\benchmark_rfdetr.py"
```

## RT-DETR-L training and test

Rename the working file to `train_rtdetr_l.py` in the clean repo.

```powershell
python ".\scripts\train_rtdetr_l.py" --check-only
python ".\scripts\train_rtdetr_l.py"
```

The script trains and then evaluates the best checkpoint on the untouched test split.

## OCR v2 preparation

```powershell
python ".\scripts\build_ocr_v2_dataset.py"
```

OCR fine-tuning used FastPlateOCR CLI and the previous best checkpoint. Copy the exact final command from the training log or project notes into README.

## OCR evaluation

```powershell
python ".\scripts\evaluate_ocr_test.py" --help
```

Document the exact commands for:
- original OCR test
- Abu Dhabi numeric test
- combined OCR v2 test

## Emirate dataset and training

```powershell
python ".\scripts\build_emirate_dataset.py"
python ".\scripts\finalize_emirate_labels.py"
python ".\scripts\train_emirate_classifier.py"
python ".\scripts\build_emirate_code_rules.py"
```

## Final application demo

```powershell
python ".\scripts\run_final_uae_plate_pipeline_v2.py" --image "C:\full\path\to\vehicle_image.jpg"
```

Output:
- annotated image
- detected plate crops
- JSON containing detector confidence, OCR text, parsed code/number, emirate, and confidence
