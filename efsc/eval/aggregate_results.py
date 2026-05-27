from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


MAIN_COLUMNS = [
    "run",
    "method",
    "model",
    "benchmark",
    "macro_f1",
    "benign_answer_rate",
    "harmful_refusal_rate",
    "over_refusal_rate",
    "under_refusal_rate",
    "clarification_accuracy",
    "safe_answer_accuracy",
    "family_consistency_score",
    "counterfactual_refusal_gap",
    "trainable_params",
    "gpu_hours",
    "latency_per_example_sec",
]


def flatten_metrics(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {column: "" for column in MAIN_COLUMNS}
    row["run"] = path.parent.name
    row.update({key: data.get(key, "") for key in MAIN_COLUMNS if key in data})
    row.setdefault("benchmark", data.get("source", ""))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate EFSC metrics JSON files into a paper-table CSV.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: List[Dict] = []
    for pattern in args.inputs:
        for path in Path().glob(pattern):
            rows.append(flatten_metrics(path))
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
