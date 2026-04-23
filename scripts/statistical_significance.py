from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.eval.maintrack_metrics import compute_maintrack_metrics


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in rows}


def metric(rows: List[Dict[str, Any]], name: str) -> float:
    return float(compute_maintrack_metrics(rows)[name])


def paired_rows(primary: Dict[str, Dict[str, Any]], baseline: Dict[str, Dict[str, Any]]):
    common_ids = sorted(set(primary) & set(baseline))
    primary_rows = [dict(primary[ex_id]) for ex_id in common_ids]
    baseline_rows = [dict(baseline[ex_id]) for ex_id in common_ids]
    return common_ids, primary_rows, baseline_rows


def random_swap_predictions(primary_rows, baseline_rows, rng: random.Random):
    a_rows = [dict(row) for row in primary_rows]
    b_rows = [dict(row) for row in baseline_rows]
    for a, b in zip(a_rows, b_rows):
        if rng.random() < 0.5:
            a["pred_action"], b["pred_action"] = b["pred_action"], a["pred_action"]
    return a_rows, b_rows


def paired_permutation_test(primary_rows, baseline_rows, metric_name: str, n_permutations: int, seed: int):
    rng = random.Random(seed)
    observed = metric(primary_rows, metric_name) - metric(baseline_rows, metric_name)
    more_extreme = 0
    for _ in range(n_permutations):
        a_rows, b_rows = random_swap_predictions(primary_rows, baseline_rows, rng)
        diff = metric(a_rows, metric_name) - metric(b_rows, metric_name)
        if abs(diff) >= abs(observed):
            more_extreme += 1
    p_value = (more_extreme + 1) / (n_permutations + 1)
    return {"observed_diff": observed, "p_value": p_value, "n_permutations": n_permutations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_preds", required=True)
    parser.add_argument("--baseline_preds", required=True)
    parser.add_argument("--metrics", nargs="+", default=["safety_utility_score", "accuracy", "macro_f1"])
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary = index_by_id(load_jsonl(Path(args.primary_preds)))
    baseline = index_by_id(load_jsonl(Path(args.baseline_preds)))
    common_ids, primary_rows, baseline_rows = paired_rows(primary, baseline)
    if not common_ids:
        raise ValueError("No overlapping example ids between prediction files.")

    results = {
        "primary_preds": args.primary_preds,
        "baseline_preds": args.baseline_preds,
        "num_common_examples": len(common_ids),
        "tests": {},
    }
    for metric_name in args.metrics:
        results["tests"][metric_name] = paired_permutation_test(
            primary_rows=primary_rows,
            baseline_rows=baseline_rows,
            metric_name=metric_name,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
