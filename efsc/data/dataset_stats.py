from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from efsc.utils import load_jsonl, save_json


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def compute_stats(rows: List[Dict]) -> Dict:
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    return {
        "num_rows": len(rows),
        "num_families": len(families),
        "by_source": counter_to_dict(Counter(row.get("source", "unknown") for row in rows)),
        "by_split": counter_to_dict(Counter(row.get("split", "unknown") for row in rows)),
        "by_action": counter_to_dict(Counter(row["action_label"] for row in rows)),
        "by_harm": counter_to_dict(Counter(row["harm_label"] for row in rows)),
        "by_legitimacy": counter_to_dict(Counter(row["legit_label"] for row in rows)),
        "by_uncertainty": counter_to_dict(Counter(row["uncertainty_label"] for row in rows)),
        "by_variant": counter_to_dict(Counter(row["variant_type"] for row in rows)),
        "by_topic": counter_to_dict(Counter(row["topic"] for row in rows)),
        "family_size": {
            "min": min((len(items) for items in families.values()), default=0),
            "max": max((len(items) for items in families.values()), default=0),
            "mean": sum(len(items) for items in families.values()) / max(1, len(families)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute normalized EFSC dataset statistics.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(load_jsonl(path))
    save_json(compute_stats(rows), args.output)


if __name__ == "__main__":
    main()
