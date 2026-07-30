import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "feature_set",
    "architecture",
    "weight_decay",
    "selected_total",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "validation_B_accuracy",
    "validation_P_accuracy",
    "test_B_accuracy",
    "test_P_accuracy",
]


def accuracy(metrics: dict, split: str, label: str | None = None) -> str:
    split_metrics = metrics.get(split, {})
    if split_metrics.get("count", 0) == 0:
        return ""
    if label is None:
        return str(split_metrics.get("accuracy", ""))
    return str(split_metrics.get("per_class", {}).get(label, {}).get("accuracy", ""))


def summarize_report(path: Path) -> dict[str, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    architecture = report["architecture"]
    if isinstance(architecture, dict):
        if "input_shape" in architecture:
            architecture_text = f"{architecture['input_shape']}->{architecture.get('conv_channels', [])}->{architecture.get('classes', [])}"
        else:
            architecture_text = json.dumps(architecture, sort_keys=True)
    else:
        architecture_text = "->".join(str(item) for item in architecture)
    return {
        "run": str(path.parent),
        "feature_set": report["feature_set"],
        "architecture": architecture_text,
        "weight_decay": str(report.get("weight_decay", "")),
        "selected_total": str(report["sample_counts"]["selected_total"]),
        "train_accuracy": accuracy(metrics, "train"),
        "validation_accuracy": accuracy(metrics, "validation"),
        "test_accuracy": accuracy(metrics, "test"),
        "validation_B_accuracy": accuracy(metrics, "validation", "B"),
        "validation_P_accuracy": accuracy(metrics, "validation", "P"),
        "test_B_accuracy": accuracy(metrics, "test", "B"),
        "test_P_accuracy": accuracy(metrics, "test", "P"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ML baseline training reports.")
    parser.add_argument("--root", type=Path, default=Path("experiments/ml_baseline"))
    parser.add_argument("--out", type=Path, default=Path("experiments/ml_baseline/summary.csv"))
    args = parser.parse_args()

    reports = sorted(args.root.glob("*/training_report.json"))
    rows = [summarize_report(path) for path in reports]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {args.out}")
    for row in rows:
        print(
            f"{row['run']}: val={row['validation_accuracy']} "
            f"test={row['test_accuracy']} features={row['feature_set']}"
        )


if __name__ == "__main__":
    main()
