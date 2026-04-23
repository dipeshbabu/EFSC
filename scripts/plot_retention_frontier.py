from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(obj, key: str) -> float:
    value = obj[key]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention_files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()
    if len(args.retention_files) != len(args.labels):
        raise ValueError("--retention_files and --labels must have the same length")

    xs = []
    ys = []
    for path in args.retention_files:
        row = load_json(Path(path))
        xs.append(metric(row, "quantized_structure_retention"))
        ys.append(metric(row, "quantized_safety_retention"))

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, args.labels):
        plt.annotate(label, (x, y))
    plt.xlabel("Quantized structure retention")
    plt.ylabel("Quantized safety retention")
    plt.title("Retention frontier")
    plt.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
