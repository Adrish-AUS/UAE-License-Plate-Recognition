"""Check the accepted release for current cross-split leakage candidates.

Repository mode uses the image SHA-256 values recorded in the active manifest.
Full mode recomputes image hashes and performs the current 16x16 difference-hash
scan.  This script reports candidates; it never deletes or moves dataset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from PIL import Image

from preprocessing_utils import ACCEPTED_COUNTS, SPLITS, read_csv, write_csv

OUTPUT_COLUMNS = [
    "candidate_id",
    "check_type",
    "distance",
    "distance_threshold",
    "split_a",
    "image_a",
    "split_b",
    "image_b",
    "exact_duplicate",
    "sha256",
]
IMAGES_NOT_INCLUDED = "NOT_RUN_IMAGES_NOT_INCLUDED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    parser.add_argument("--mode", choices=("auto", "repository", "full"), default="auto")
    parser.add_argument("--threshold", type=int, default=8)
    return parser.parse_args()


def _resolve(package_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (package_root / path).resolve()


def _all_images_present(package_root: Path, manifest: list[dict[str, str]]) -> bool:
    if len(manifest) != ACCEPTED_COUNTS["total"]["images"]:
        return False
    return all((package_root / row["image_relative_path"]).is_file() for row in manifest)


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        image.load()
        pixels = list(image.convert("L").resize((17, 16)).tobytes())
    value = 0
    for row in range(16):
        offset = row * 17
        for column in range(16):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _records(package_root: Path, manifest: list[dict[str, str]], mode: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in manifest:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Manifest contains invalid split: {split}")
        relative = row["image_relative_path"].replace("\\", "/")
        record: dict[str, object] = {"split": split, "relative": relative}
        if mode == "repository":
            digest = row.get("image_sha256", "").strip().lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"Manifest has invalid image SHA-256: {relative}")
            record["sha256"] = digest
        else:
            path = package_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Full mode requires image: {relative}")
            record["path"] = path
        records.append(record)
    return records


def _exact_rows(records: list[dict[str, object]], threshold: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record["sha256"])].append(record)
    rows: list[dict[str, object]] = []
    counter = 1
    for digest, group in sorted(groups.items()):
        if len({str(item["split"]) for item in group}) < 2:
            continue
        for left, right in combinations(group, 2):
            if left["split"] == right["split"]:
                continue
            rows.append(
                {
                    "candidate_id": f"exact_{counter:03d}",
                    "check_type": "sha256",
                    "distance": 0,
                    "distance_threshold": threshold,
                    "split_a": left["split"],
                    "image_a": left["relative"],
                    "split_b": right["split"],
                    "image_b": right["relative"],
                    "exact_duplicate": "true",
                    "sha256": digest,
                }
            )
            counter += 1
    return rows


def _perceptual_rows(records: list[dict[str, object]], threshold: int) -> list[dict[str, object]]:
    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = executor.map(lambda item: difference_hash(Path(item["path"])), records)
        for record, digest in zip(records, hashes, strict=True):
            record["dhash"] = digest
    by_split = {split: [record for record in records if record["split"] == split] for split in SPLITS}
    rows: list[dict[str, object]] = []
    counter = 1
    for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
        for left in by_split[split_a]:
            left_hash = int(left["dhash"])
            for right in by_split[split_b]:
                distance = (left_hash ^ int(right["dhash"])).bit_count()
                if distance > threshold:
                    continue
                rows.append(
                    {
                        "candidate_id": f"dhash_{counter:03d}",
                        "check_type": "difference_hash",
                        "distance": distance,
                        "distance_threshold": threshold,
                        "split_a": split_a,
                        "image_a": left["relative"],
                        "split_b": split_b,
                        "image_b": right["relative"],
                        "exact_duplicate": "false",
                        "sha256": "",
                    }
                )
                counter += 1
    return sorted(rows, key=lambda row: (int(row["distance"]), str(row["image_a"]), str(row["image_b"])))


def run_leakage(
    package_root: Path,
    dataset_root: Path,
    *,
    mode: str = "auto",
    threshold: int = 8,
    write_output: bool = True,
) -> dict[str, object]:
    package_root = package_root.resolve()
    dataset_root = dataset_root.resolve()
    if threshold < 0:
        raise ValueError("Distance threshold must be nonnegative")
    manifest_path = package_root / "reports" / "dataset_manifest.csv"
    manifest = read_csv(manifest_path)
    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "full" if _all_images_present(package_root, manifest) else "repository"
    records = _records(package_root, manifest, selected_mode)
    if selected_mode == "full":
        workers = min(32, max(4, (os.cpu_count() or 4) * 2))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            hashes = executor.map(lambda item: _sha256(Path(item["path"])), records)
            for record, digest in zip(records, hashes, strict=True):
                record["sha256"] = digest
    rows = _exact_rows(records, threshold)
    exact_count = len(rows)
    if selected_mode == "full":
        perceptual = _perceptual_rows(records, threshold)
        rows.extend(perceptual)
        perceptual_result: int | str = len(perceptual)
    else:
        perceptual_result = IMAGES_NOT_INCLUDED
    if write_output:
        write_csv(package_root / "reports" / "split_leakage_candidates.csv", OUTPUT_COLUMNS, rows)
    return {
        "mode": selected_mode,
        "images_considered": len(records),
        "exact_cross_split_duplicates": exact_count,
        "perceptual_candidates": perceptual_result,
        "distance_threshold": threshold,
        "candidates": rows,
    }


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    dataset_root = _resolve(package_root, args.dataset_root)
    result = run_leakage(
        package_root,
        dataset_root,
        mode=args.mode,
        threshold=args.threshold,
        write_output=True,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    if result["exact_cross_split_duplicates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
