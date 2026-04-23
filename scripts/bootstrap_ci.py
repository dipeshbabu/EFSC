from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.maintrack_metrics import compute_maintrack_metrics


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[int(p * (len(xs) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    rows = load_jsonl(Path(args.predictions))
    if not rows:
        raise ValueError(f"No prediction rows found in {args.predictions}")
    metric_names = [
        "macro_f1",
        "harmful_refusal_rate",
        "benign_retention_rate",
        "over_refusal_rate",
        "ambiguity_calibration",
        "safety_utility_score",
        "family_consistency_score",
        "counterfactual_action_stability",
        "counterfactual_action_sensitivity",
    ]
    sampled = {name: [] for name in metric_names}
    n = len(rows)
    for _ in range(args.num_samples):
        boot = [rows[random.randrange(n)] for _ in range(n)]
        metrics = compute_maintrack_metrics(boot)
        for name in metric_names:
            sampled[name].append(float(metrics[name]))
    output = {
        name: {
            "mean_bootstrap": sum(values) / max(len(values), 1),
            "ci_low_95": percentile(values, 0.025),
            "ci_high_95": percentile(values, 0.975),
        }
        for name, values in sampled.items()
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
