import json
import time
from pathlib import Path

import numpy as np
import torch
from rfdetr import RFDETRMedium


ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    ROOT
    / "runs"
    / "rfdetr_medium_main"
    / "checkpoint_best_total.pth"
)

TEST_DIR = (
    ROOT
    / "datasets"
    / "uae_lp_v2_rfdetr_coco"
    / "test"
)

ANNOTATIONS = TEST_DIR / "_annotations.coco.json"

OUTPUT_FILE = (
    ROOT
    / "results"
    / "rfdetr_medium_test"
    / "speed_metrics_optimized_fp32.json"
)

WARMUP_IMAGES = 10
BENCHMARK_IMAGES = 200


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    with ANNOTATIONS.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    image_paths = [
        TEST_DIR / Path(record["file_name"]).name
        for record in coco["images"]
    ]

    model = RFDETRMedium(
        pretrain_weights=str(CHECKPOINT)
    )

    print("Optimizing model for inference...", flush=True)

    model.optimize_for_inference(
        compile=True,
        batch_size=1,
        dtype=torch.float32,
    )

    print("Optimization complete.", flush=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Warm-up images:", WARMUP_IMAGES)
    print("Benchmark images:", BENCHMARK_IMAGES)

    # Warm up the GPU before timing.
    for image_path in image_paths[:WARMUP_IMAGES]:
        model.predict(str(image_path), threshold=0.5)

    torch.cuda.synchronize()

    latencies = []

    for index, image_path in enumerate(
        image_paths[
            WARMUP_IMAGES:
            WARMUP_IMAGES + BENCHMARK_IMAGES
        ],
        start=1,
    ):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        model.predict(str(image_path), threshold=0.5)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        latencies.append(elapsed * 1000)

        if index % 25 == 0:
            print(f"Processed {index}/{BENCHMARK_IMAGES}")

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_latency

    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.model.parameters()
    )

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "images_tested": len(latencies),
        "batch_size": 1,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "fps": fps,
        "parameters": parameter_count,
        "parameters_millions": parameter_count / 1_000_000,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSpeed results:")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()