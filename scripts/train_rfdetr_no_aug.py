from pathlib import Path

import torch
from rfdetr import RFDETRMedium

# Repository root folder
ROOT = Path(__file__).resolve().parent.parent

# Same dataset used by the main augmented experiment
DATASET_DIR = (
    ROOT
    / "datasets"
    / "uae_lp_v2_rfdetr_coco"
)

# Separate output folder so the main checkpoints are not overwritten
OUTPUT_DIR = (
    ROOT
    / "runs"
    / "rfdetr_medium_no_aug"
)

def main() -> None:
    """Train RF-DETR Medium without data augmentation."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training would run on the CPU."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Experiment: RF-DETR Medium without augmentation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATASET_DIR)
    print("Output:", OUTPUT_DIR)

    # Load the official pretrained RF-DETR Medium model.
    model = RFDETRMedium()

    model.train(
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),

        # Same maximum as the augmented experiment.
        # Early stopping can end training before epoch 20.
        epochs=20,

        # Effective batch size:
        # 4 images x 4 gradient accumulation steps = 16
        batch_size=4,
        grad_accum_steps=4,

        # Same learning rate and seed as the main experiment.
        lr=1e-4,
        seed=486,
        device="cuda",

        # Evaluate after every epoch.
        eval_interval=1,

        # Save a periodic checkpoint every two epochs.
        checkpoint_interval=2,

        # Stop when validation mAP does not improve sufficiently
        # for four consecutive validation checks.
        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,

        # Save TensorBoard training logs.
        tensorboard=True,
        progress_bar="tqdm",

        # This is the only intentional difference from the main run.
        # No brightness, blur, rotation, or perspective transformations.
        aug_config={},
    )

if __name__ == "__main__":
    main()