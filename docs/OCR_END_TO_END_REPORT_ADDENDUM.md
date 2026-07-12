# OCR and End-to-End Recognition Results

## OCR v1 crop evaluation

A CCT-S v1 FastPlateOCR model was fine-tuned with UAE licence-plate crops which allowed their transcripts to be retrieved using filename patterns.
Only those crops containing one labelled plate were kept, since one filename could not reliably correspond to multiple plates.

The first model was tested on 1,017 test crops with ground truth labels. The model gave 996 exact full plate matches and 21 mistakes.
Exact match accuracy was 97.94%, character accuracy was 99.72%, and plate length accuracy was 99.71%.

Character accuracy was calculated for all positions in the model's fixed output, including correct padding positions.
The exact match accuracy is thus the more stringent OCR criterion.

| OCR v1 metric | Test result |
|---|---:|
| Test crops | 1,017 |
| Exact full-plate matches | 996 |
| Incorrect plates | 21 |
| Full-plate exact-match accuracy | 97.94% |
| Character accuracy | 99.72% |
| Plate-length accuracy | 99.71% |

## Numeric-prefix Abu Dhabi limitation and OCR v2

Labels starting with one to three letters followed by digits, such as `S-10198` → `S10198`, formed the primary input into the first OCR dataset.
During end-to-end testing, this led to a format dependency with regards to Abu Dhabi plates where the category code consists of digits. 
Thus, the visible plate `7-59700` was initially classified with a letter being at the front position.

A digit only post-process function was tried but was later scrapped because it occasionally replaced the first digit by either `1` or `0`. Finally, a model-based correction was used. Strictly numeric prefix Abu Dhabi labels were recovered from file names, combined with the OCR output data and assigned the same train/validation/test set splits as those of the OCR dataset. The OCR v2 was then fine-tuned from the OCR v1 `best.keras` checkpoint, with batch size 64, learning rate `5e-5`, max epoch 15 and early stopping patience 4; training terminated after epoch 13.

The OCR v2 dataset retained the original OCR samples and added strict numeric-prefix Abu Dhabi samples while preserving the existing split.

| Split | Original OCR crops | Added Abu Dhabi numeric crops | OCR v2 total |
|---|---:|---:|---:|
| Train | 3,350 | 595 | 3,945 |
| Val | 1,055 | 145 | 1,200 |
| Test | 1,017 | 173 | 1,190 |

## OCR v2 evaluation

The original OCR test, the numeric-prefix Abu Dhabi test, and the combined OCR v2 test are reported separately. This allows distinction between performance retention on the original label format and the improvement of performance on the new label format.

| Evaluation subset | Test crops | Full-plate exact accuracy | Character accuracy | Plate-length accuracy |
|---|---:|---:|---:|---:|
| OCR v2 on original test | 1,017 | 98.23% | 99.77% | 99.90% |
| OCR v2 on Abu Dhabi numeric test | 173 | 86.13% | 97.56% | 95.38% |
| OCR v2 on combined test | Not evaluated | Not available | Not available | Not available |


### Missing evaluation note

- `results/ocr_uae_v2_combined_test/metrics.json` was not found, so no combined OCR v2 accuracy is claimed.

This last configuration employs the OCR v2 checkpoint. The OCR v1 outputs must be preserved as the starting point and must not be used as the final OCR v2 output.

## Emirate-aware code and number parsing

The OCR neural network predicts the full alphanumeric string visible to it.
A second parser component divides the string into the plate code and number using code patterns learned from recovered training file names.

Examples include:

- `S10198` -> code `S`, number `10198`
- `759700` -> code `7`, number `59700`
- `1141153` -> code `11`, number `41153`
- `3222566` -> code `3`, number `222566`

The parser does not replace OCR characters. It only separates a recognized
string into fields after the emirate classifier identifies the plate design.

## Final end-to-end pipeline

The final application uses the following sequence:

1. RF-DETR Medium detects the full licence plate in a vehicle image.
2. The predicted box is expanded by 8% before cropping.
3. OCR v2 recognizes the visible plate code and number.
4. A ResNet18 classifier predicts one of the seven UAE emirates.
5. Training-derived code rules separate the recognized string into code and
   number fields.
6. The system saves an annotated image, plate crops, and a JSON result.

The final inference script is:

`python scripts/run_final_uae_plate_pipeline_v2.py --image <image-path>`

End-to-end examples are a qualitative proof of the concept rather than a full end-to-end test accuracy. The detector mAP, OCR crop accuracy, and emirate classification metrics are stated independently, as they represent different stages of the process. The OCR result on the detector's predicted crops could be lower than the crop-level test accuracy if the detected bounding box was loose or incomplete.

## Files

### OCR v1
- Checkpoint: `runs/ocr_uae_main/2026-07-11_23-56-19/best.keras`
- Metrics: `results/ocr_uae_test/metrics.json`
- Predictions: `results/ocr_uae_test/predictions.csv`
- Correct examples: `results/ocr_uae_test/correct_examples.jpg`
- Incorrect examples: `results/ocr_uae_test/incorrect_examples.jpg`

### OCR v2
- Checkpoint: newest `best.keras` under `runs/ocr_uae_v2_retry/`
- Original-test metrics: `results/ocr_uae_v2_original_test/metrics.json`
- Abu Dhabi numeric metrics:
  `results/ocr_uae_v2_abu_dhabi_test/metrics.json`
- Combined-test metrics:
  `results/ocr_uae_v2_combined_test/metrics.json`

### Final pipeline
- Script: `scripts/run_final_uae_plate_pipeline_v2.py`
- Code rules: `results/emirate_code_rules.json`
- Outputs: `results/final_pipeline_demo_v2/`
