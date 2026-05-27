from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

NUMERIC_FIELDS = [
    "accuracy",
    "macro_f1",
    "harmful_refusal_rate",
    "under_refusal_rate",
    "benign_retention_rate",
    "over_refusal_rate",
    "safe_answer_accuracy",
    "clarify_accuracy",
    "ambiguity_calibration",
    "authorized_benign_success",
    "educational_benign_success",
    "hard_benign_success",
    "family_consistency_score",
    "counterfactual_action_stability",
    "counterfactual_action_sensitivity",
    "refusal_precision",
    "refusal_recall",
    "safety_utility_score",
]


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_name(path: Path) -> Dict[str, str]:
    parts = path.stem.split("__")
    if len(parts) < 4:
        return {"model": "unknown", "method": "unknown", "seed": "unknown", "dataset": path.stem}
    return {"model": parts[0], "method": parts[1], "seed": parts[2], "dataset": "__".join(parts[3:])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in Path(args.metrics_dir).glob("*.json"):
        meta = parse_name(path)
        key = (meta["model"], meta["method"], meta["dataset"])
        grouped[key].append(load_json(path))

    if not grouped:
        raise ValueError(f"No metric JSON files found in {args.metrics_dir}")

    rows = []
    for (model, method, dataset), objs in grouped.items():
        row = {"model": model, "method": method, "dataset": dataset, "num_seeds": len(objs)}
        for field in NUMERIC_FIELDS:
            vals = [float(obj[field]) for obj in objs if field in obj]
            row[f"{field}_mean"] = mean(vals)
            row[f"{field}_std"] = std(vals)
        rows.append(row)
    rows = sorted(rows, key=lambda row: (row["dataset"], row["model"], row["method"]))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote merged seed metrics to {output}")


if __name__ == "__main__":
    main()
