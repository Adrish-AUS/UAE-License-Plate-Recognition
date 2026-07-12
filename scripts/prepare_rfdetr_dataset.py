import json
import os
import shutil
from pathlib import Path


# Repository root: one level above the scripts folder
ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = ROOT / "datasets" / "uae_lp_v2_yolo" / "images"
SOURCE_ANNOTATIONS = ROOT / "annotations" / "coco"

OUTPUT_DATASET = ROOT / "datasets" / "uae_lp_v2_rfdetr_coco"

# Existing split name -> RF-DETR split name
SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


def link_or_copy(source: Path, destination: Path) -> None:
    """
    Try to create a hard link so the image does not consume extra space.
    Fall back to copying if hard links are unavailable.
    """
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_split(source_split: str, output_split: str) -> None:
    source_image_dir = SOURCE_IMAGES / source_split
    source_json = SOURCE_ANNOTATIONS / f"{source_split}.json"

    output_dir = OUTPUT_DATASET / output_split
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_image_dir.exists():
        raise FileNotFoundError(
            f"Missing image folder: {source_image_dir}"
        )

    if not source_json.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {source_json}"
        )

    with source_json.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    missing_images = []
    used_names = set()

    for image_record in coco.get("images", []):
        original_name = Path(image_record["file_name"]).name

        if original_name in used_names:
            raise ValueError(
                f"Duplicate image filename in {source_split}: {original_name}"
            )

        used_names.add(original_name)

        source_image = source_image_dir / original_name
        destination_image = output_dir / original_name

        if not source_image.exists():
            missing_images.append(str(source_image))
            continue

        link_or_copy(source_image, destination_image)

        # RF-DETR will find the image directly inside the split folder.
        image_record["file_name"] = original_name

    if missing_images:
        print(f"\nMissing images in {source_split}:")
        for path in missing_images[:20]:
            print(f"  {path}")

        raise FileNotFoundError(
            f"{len(missing_images)} referenced images are missing."
        )

    output_json = output_dir / "_annotations.coco.json"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2)

    print(
        f"{output_split:5s} | "
        f"images: {len(coco.get('images', [])):5d} | "
        f"boxes: {len(coco.get('annotations', [])):5d} | "
        f"classes: {len(coco.get('categories', []))}"
    )


def main() -> None:
    print(f"Creating RF-DETR dataset at:\n{OUTPUT_DATASET}\n")

    for source_split, output_split in SPLITS.items():
        prepare_split(source_split, output_split)

    print("\nRF-DETR dataset prepared successfully.")


if __name__ == "__main__":
    main()