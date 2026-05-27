from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()
    methods, values = [], []
    with Path(args.merged_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != args.dataset or not row["method"].startswith("efsc"):
                continue
            methods.append(row["method"])
            values.append(float(row[f"{args.metric}_mean"]))
    if not methods:
        raise ValueError(f"No EFSC rows found for dataset={args.dataset}")
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(methods)), values)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.title(f"{args.dataset}: Ablation Comparison")
    plt.tight_layout()
    output = Path(args.output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote ablation plot to {output}")


if __name__ == "__main__":
    main()
