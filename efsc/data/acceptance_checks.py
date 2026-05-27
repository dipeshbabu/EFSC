from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List

from efsc.data.dataset import validate_prompt_example
from efsc.utils import load_jsonl, save_json


def check_precedence(row: Dict) -> List[str]:
    errors = []
    if row["harm_label"] == "HIGH" and row["legit_label"] == "LOW" and row["action_label"] != "REFUSE":
        errors.append("HIGH harm + LOW legitimacy must REFUSE")
    if row["legit_label"] == "MEDIUM" and row["uncertainty_label"] == "AMBIGUOUS" and row["action_label"] != "CLARIFY":
        errors.append("MEDIUM legitimacy + AMBIGUOUS uncertainty must CLARIFY")
    if row["legit_label"] == "HIGH" and row["harm_label"] == "MEDIUM" and row["action_label"] == "ANSWER":
        errors.append("MEDIUM harm + HIGH legitimacy should SAFE_ANSWER, not ANSWER")
    if row["harm_label"] != "HIGH" and row["legit_label"] == "HIGH" and row["action_label"] == "REFUSE":
        errors.append("benign legitimate prompts must not REFUSE")
    return errors


def run_checks(
    rows: List[Dict],
    required_actions: List[str] | None = None,
    required_harm_labels: List[str] | None = None,
    min_rows: int = 0,
) -> Dict:
    errors = []
    ids = set()
    family_splits = defaultdict(set)
    action_counts = Counter()
    harm_counts = Counter()
    source_counts = Counter()
    for index, row in enumerate(rows):
        try:
            validate_prompt_example(row)
        except Exception as exc:
            errors.append({"index": index, "id": row.get("id"), "error": str(exc)})
            continue
        if row["id"] in ids:
            errors.append({"index": index, "id": row["id"], "error": "duplicate id"})
        ids.add(row["id"])
        action_counts[row["action_label"]] += 1
        harm_counts[row["harm_label"]] += 1
        source_counts[row["source"]] += 1
        family_splits[row["family_id"]].add(row["split"])
        for error in check_precedence(row):
            errors.append({"index": index, "id": row["id"], "error": error})
    for family_id, splits in family_splits.items():
        if len(splits) > 1:
            errors.append({"family_id": family_id, "error": f"family crosses splits: {sorted(splits)}"})
    if min_rows and len(rows) < min_rows:
        errors.append({"error": f"expected at least {min_rows} rows, found {len(rows)}"})
    for action in required_actions or []:
        if action_counts[action] == 0:
            errors.append({"error": f"missing required action label: {action}"})
    for label in required_harm_labels or []:
        if harm_counts[label] == 0:
            errors.append({"error": f"missing required harm label: {label}"})
    return {
        "passed": not errors,
        "num_rows": len(rows),
        "num_errors": len(errors),
        "action_counts": dict(action_counts),
        "harm_counts": dict(harm_counts),
        "source_counts": dict(source_counts),
        "errors": errors[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EFSC dataset acceptance checks before training.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-actions", nargs="*", default=[])
    parser.add_argument("--require-harm-labels", nargs="*", default=[])
    parser.add_argument("--min-rows", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(load_jsonl(path))
    report = run_checks(
        rows,
        required_actions=args.require_actions,
        required_harm_labels=args.require_harm_labels,
        min_rows=args.min_rows,
    )
    save_json(report, args.output)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
