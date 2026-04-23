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
    parser.add_argument("--fp_metrics", nargs="+", required=True)
    parser.add_argument("--quant_metrics", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()
    if not (len(args.fp_metrics) == len(args.quant_metrics) == len(args.labels)):
        raise ValueError("Metric lists and labels must have the same length")

    xs = []
    ys = []
    for fp_path, q_path in zip(args.fp_metrics, args.quant_metrics):
        fp = load_json(Path(fp_path))
        q = load_json(Path(q_path))
        fp_su = metric(fp, "safety_utility_score")
        q_su = metric(q, "safety_utility_score")
        xs.append(fp_su)
        ys.append(fp_su - q_su)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, args.labels):
        plt.annotate(label, (x, y))
    plt.xlabel("FP safety utility")
    plt.ylabel("Safety utility drop after quantization")
    plt.title("Quantization tradeoff")
    plt.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
