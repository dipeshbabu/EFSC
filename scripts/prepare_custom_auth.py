from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.data.create_custom_auth_set import build_examples
from efsc.utils import save_jsonl


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_list(value: str, fallback: List[str]) -> List[str]:
    if not value:
        return fallback
    return json.loads(value)


def csv_rows_to_jsonl_rows(path: Path, split: str) -> List[Dict[str, Any]]:
    rows = []
    for raw in load_csv(path):
        action = raw["action_label"]
        rows.append({
            "id": f"custom_auth::{raw['id']}",
            "source": "custom_auth",
            "source_split": split,
            "split": split,
            "family_id": raw.get("family_id", ""),
            "prompt": raw["prompt"].strip(),
            "language": raw.get("language", "en"),
            "topic": raw.get("topic", "general"),
            "variant_type": raw["variant_type"],
            "harm_label": raw["harm_label"],
            "legit_label": raw["legit_label"],
            "uncertainty_label": raw["uncertainty_label"],
            "action_label": action,
            "preferred_over": parse_json_list(raw.get("preferred_over", ""), [action]),
            "dispreferred_over": parse_json_list(raw.get("dispreferred_over", ""), ["REFUSE"]),
            "metadata": {"raw_id": raw["id"], "raw_label": None, "notes": raw.get("notes")},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-examples", type=int, default=None, help="Generate a templated custom set when input_csv is omitted.")
    parser.add_argument("--test-examples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_csv:
        rows = csv_rows_to_jsonl_rows(Path(args.input_csv), args.split)
    else:
        rows = build_examples(args.num_examples or 1800, args.test_examples, args.seed)
    save_jsonl(rows, Path(args.output))
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
