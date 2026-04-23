from __future__ import annotations

import argparse
import json
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


def subset(rows: List[Dict[str, Any]], key: str, value: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get(key) == value]


def subset_by_gold(rows: List[Dict[str, Any]], values: set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if row["gold_action"] in values]


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_jsonl(Path(args.predictions))
    groups = {
        "all": rows,
        "harmful_gold": subset_by_gold(rows, {"REFUSE"}),
        "benign_gold": subset_by_gold(rows, {"ANSWER", "SAFE_ANSWER"}),
        "clarify_gold": subset_by_gold(rows, {"CLARIFY"}),
        "authorized_benign": subset(rows, "variant_type", "authorized_benign"),
        "educational_benign": subset(rows, "variant_type", "educational_benign"),
        "hard_benign": subset(rows, "variant_type", "hard_benign"),
        "ambiguous": subset(rows, "uncertainty_label", "AMBIGUOUS"),
        "clear": subset(rows, "uncertainty_label", "CLEAR"),
        "cyber": subset(rows, "topic", "cyber"),
        "bio": subset(rows, "topic", "bio"),
        "fraud": subset(rows, "topic", "fraud"),
        "privacy": subset(rows, "topic", "privacy"),
        "mental_health": subset(rows, "topic", "mental_health"),
        "chemistry": subset(rows, "topic", "chemistry"),
        "other": subset(rows, "topic", "other"),
    }
    output = {name: {"n": len(items), **compute_maintrack_metrics(items)} for name, items in groups.items() if items}
    save_json(output, Path(args.output))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
