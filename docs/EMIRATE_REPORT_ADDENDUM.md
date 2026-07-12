# Emirate Classification Results

Training of a seven class emirate classifier involved using 10,452 recovered licence plate crops. The original 51 class source annotations included emirate specific labels that were stripped off during one class detector pre-processing. This information was recovered and mapped into Dubai, Abu Dhabi, Sharjah, Fujairah, Ajman, Ras Al Khaimah, and Umm Al Quwain keeping the same train/validation/test split of the detector.

An ImageNet pre-trained ResNet18 network with input images sized 224 x 224 was used for classification. Balancing was applied during training since Dubai comprised most of the recovered crops. Light augmentation such as brightness, blur, affine transformation, scale, and translations were applied during training. However, horizontal flipping was not done since reflected text layout is unrealistic. The best checkpoint was picked using validation macro F1.

The classifier reached 99.59%, 97.71%, 98.54%, and 99.59% accuracy, balanced accuracy, macro F1, and weighted F1 respectively when tested on the raw 1,466 crop test set. This was realized at epoch 4.

| Metric | Test result |
|---|---:|
| Test crops | 1,466 |
| Accuracy | 99.59% |
| Balanced accuracy | 97.71% |
| Macro F1 | 98.54% |
| Weighted F1 | 99.59% |
| Best epoch | 4 |

However, alone, accuracy is not enough because of the high number of samples belonging to the class Dubai.
For this reason, balanced accuracy and macro F1 are used as the main performance measures.
Umm Al Quwain includes just 15 samples, hence its class-level performance is less reliable compared to those of Dubai, Abu Dhabi, or Sharjah.

In the final pipeline, the use of RF-DETR Medium for plate localization, OCR v2 for Latin code and number detection, ResNet18 for seven classes of emirates identification, and trained emirate format rules to split the recognized string into code and number.
