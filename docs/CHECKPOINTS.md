# Checkpoints and Model Files

Large model files are not stored directly in the normal Git repository. Share them separately using Google Drive, OneDrive, GitHub Releases, or Git LFS.

## Required files

### RF-DETR Medium — augmented main model

Source path in the working project:

```text
runs/rfdetr_medium_main/checkpoint_best_total.pth
```

Recommended placement after download:

```text
runs/rfdetr_medium_main/checkpoint_best_total.pth
```

Used by the final application pipeline.

Download link:

```text
ADD_LINK_HERE
```

---

### RF-DETR Medium — no-augmentation ablation

Source path:

```text
runs/rfdetr_medium_no_aug/checkpoint_best_ema.pth
```

Recommended placement:

```text
runs/rfdetr_medium_no_aug/checkpoint_best_ema.pth
```

Used only for the controlled augmentation ablation.

Download link:

```text
ADD_LINK_HERE
```

---

### RT-DETR-L

Source path:

```text
runs/rtdetr_l_main/weights/best.pt
```

Recommended placement:

```text
runs/rtdetr_l_main/weights/best.pt
```

Used for the additional detector comparison.

Download link:

```text
ADD_LINK_HERE
```

---

### OCR v2

Source path:

```text
runs/ocr_uae_v2_retry/<latest-run>/best.keras
```

Recommended placement:

```text
runs/ocr_uae_v2_retry/<latest-run>/best.keras
```

Use the newest completed OCR v2 run that fixed numeric-prefix Abu Dhabi plates.

Download link:

```text
ADD_LINK_HERE
```

---

### Emirate ResNet18 classifier

Source path:

```text
runs/emirate_resnet18_main/best.pt
```

Recommended placement:

```text
runs/emirate_resnet18_main/best.pt
```

Used by the final seven-emirate pipeline.

Download link:

```text
ADD_LINK_HERE
```

## Important notes

- Do not commit these files directly into ordinary Git history.
- Keep the exact filenames unchanged.
- Preserve the folder structure expected by the scripts.
- The final demo also requires:

```text
results/emirate_code_rules.json
```

- The dataset itself is not included in the release repository.
- After adding the download links, send this file to the report writer together with the release ZIP.
