from __future__ import annotations

import argparse
import json
from typing import Dict, List

from efsc.utils import load_jsonl, save_jsonl


def build_pairs(rows: List[Dict]) -> List[Dict]:
    pairs = []
    for row in rows:
        preferred = row.get("preferred_over") or [row["action_label"]]
        dispreferred = row.get("dispreferred_over") or ["REFUSE"]
        for pos in preferred:
            for neg in dispreferred:
                if pos == neg:
                    continue
                pairs.append({
                    "id": f"{row['id']}::{pos}>{neg}",
                    "source_id": row["id"],
                    "source": row.get("source", "unknown"),
                    "family_id": row["family_id"],
                    "prompt": row["prompt"],
                    "preferred_action": pos,
                    "dispreferred_action": neg,
                    "action_label": row["action_label"],
                    "metadata": {"topic": row.get("topic"), "variant_type": row.get("variant_type")},
                })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline structured-action preference pairs for EFSC stage 3.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    pairs = build_pairs(rows)
    save_jsonl(pairs, args.output)
    print(json.dumps({"input_rows": len(rows), "preference_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
