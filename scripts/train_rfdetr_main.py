from pathlib import Path

import torch
from rfdetr import RFDETRMedium


ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "datasets" / "uae_lp_v2_rfdetr_coco"
OUTPUT_DIR = ROOT / "runs" / "rfdetr_medium_main"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training would run on CPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATASET_DIR)
    print("Output:", OUTPUT_DIR)

    model = RFDETRMedium()

    model.train(
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),

        # Maximum training duration.
        # Early stopping may finish before epoch 20.
        epochs=20,

        # Effective batch size = 4 x 4 = 16.
        batch_size=4,
        grad_accum_steps=4,

        lr=1e-4,
        device="cuda",
        seed=486,

        eval_interval=1,
        checkpoint_interval=2,

        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,

        tensorboard=True,
        progress_bar="tqdm",

        # Plate-specific training augmentations.
        # No horizontal flip because mirrored plate text is unrealistic.
        aug_config={
            "RandomBrightnessContrast": {
                "brightness_limit": 0.20,
                "contrast_limit": 0.20,
                "p": 0.40,
            },
            "GaussianBlur": {
                "blur_limit": (3, 5),
                "p": 0.12,
            },
            "MotionBlur": {
                "blur_limit": 5,
                "p": 0.08,
            },
            "Rotate": {
                "limit": 5,
                "p": 0.20,
            },
            "Perspective": {
                "scale": (0.02, 0.05),
                "p": 0.15,
            },
        },
    )

if __name__ == "__main__":
    main()