from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "with_augmentation": ROOT / "runs" / "rfdetr_medium_main",
    "without_augmentation": ROOT / "runs" / "rfdetr_medium_no_aug",
}

OUTPUT_DIR = ROOT / "results" / "rfdetr_training_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_scalars(run_directory: Path):
    """
    Find every TensorBoard event file in one experiment and collect
    all scalar values. If the same step appears more than once,
    the latest recorded value is retained.
    """
    event_files = sorted(
        run_directory.rglob("events.out.tfevents.*")
    )

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in:\n{run_directory}"
        )

    collected = defaultdict(dict)

    for event_file in event_files:
        print(f"Reading: {event_file}")

        accumulator = EventAccumulator(
            str(event_file),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()

        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                collected[tag][event.step] = event.value

    return collected

def save_all_scalars(experiment_name, scalar_data):
    """
    Save all recorded scalar metrics into a CSV file.
    """
    rows = []

    for tag, step_values in scalar_data.items():
        for step, value in sorted(step_values.items()):
            rows.append(
                {
                    "experiment": experiment_name,
                    "tag": tag,
                    "step": step,
                    "value": value,
                }
            )

    dataframe = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR / f"{experiment_name}_all_scalars.csv"
    )
    dataframe.to_csv(output_file, index=False)

    print(f"Saved scalar data: {output_file}")

def find_tag(all_tags, required_terms, excluded_terms=None):
    """
    Find a TensorBoard tag containing every required term while
    excluding ambiguous tags such as mAP_50_95 when requesting mAP_50.
    """
    required_terms = [
        term.lower() for term in required_terms
    ]

    excluded_terms = [
        term.lower()
        for term in (excluded_terms or [])
    ]

    for tag in all_tags:
        lower_tag = tag.lower()

        contains_required = all(
            term in lower_tag
            for term in required_terms
        )

        contains_excluded = any(
            term in lower_tag
            for term in excluded_terms
        )

        if contains_required and not contains_excluded:
            return tag

    return None

def plot_comparison(
    metric_name,
    experiment_data,
    tag_terms,
    y_label,
    excluded_terms=None,
):
    """
    Plot the same metric for augmented and non-augmented runs.
    """
    plt.figure(figsize=(8, 5))

    plotted_anything = False

    for experiment_name, scalar_data in experiment_data.items():
        tag = find_tag(
        list(scalar_data.keys()),
        tag_terms,
        excluded_terms,
        )   

        if tag is None:
            print(
                f"Could not find {metric_name} tag for "
                f"{experiment_name}"
            )
            continue

        step_values = scalar_data[tag]
        steps = sorted(step_values)
        values = [step_values[step] for step in steps]

        plt.plot(
            steps,
            values,
            marker="o",
            label=experiment_name.replace("_", " ").title(),
        )

        plotted_anything = True

        print(
            f"{experiment_name} {metric_name} tag: {tag}"
        )

    if not plotted_anything:
        print(f"Skipped plot: {metric_name}")
        plt.close()
        return

    plt.xlabel("Training step / epoch")
    plt.ylabel(y_label)
    plt.title(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{metric_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved plot: {output_file}")

def main():
    experiment_data = {}

    for experiment_name, run_directory in RUNS.items():
        print(f"\nLoading experiment: {experiment_name}")
        scalar_data = load_scalars(run_directory)

        experiment_data[experiment_name] = scalar_data

        print("\nAvailable scalar tags:")
        for tag in sorted(scalar_data):
            print(f"  {tag}")

        save_all_scalars(
            experiment_name,
            scalar_data,
        )

    # These searches are intentionally flexible because RF-DETR and
    # Lightning versions may use slightly different TensorBoard names.
    plots = [
    (
        "Validation mAP 50-95",
        ["val", "map_50_95"],
        [],
        "mAP@50:95",
    ),
    (
        "Validation mAP 50",
        ["val", "map_50"],
        ["map_50_95"],
        "mAP@50",
    ),
    (
        "Validation Loss",
        ["val", "loss"],
        [],
        "Validation loss",
    ),
    (
        "Training Loss",
        ["train", "loss"],
        [],
        "Training loss",
    ),
    ]

    for (
    metric_name,
    tag_terms,
    excluded_terms,
    y_label,
    ) in plots:
        plot_comparison(
            metric_name,
            experiment_data,
            tag_terms,
            y_label,
            excluded_terms,
    )

    print("\nCurve export complete.")
    print(f"Results saved to:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()