from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--x_metric", required=True)
    parser.add_argument("--y_metric", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()

    xs, ys, labels = [], [], []
    with Path(args.merged_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != args.dataset:
                continue
            xs.append(float(row[f"{args.x_metric}_mean"]))
            ys.append(float(row[f"{args.y_metric}_mean"]))
            labels.append(f"{row['model']}\n{row['method']}")

    if not xs:
        raise ValueError(f"No rows found for dataset={args.dataset}")

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel(args.x_metric.replace("_", " ").title())
    plt.ylabel(args.y_metric.replace("_", " ").title())
    plt.title(f"{args.dataset}: {args.x_metric} vs {args.y_metric}")
    plt.tight_layout()

    output = Path(args.output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
