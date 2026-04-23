from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

from efsc.utils import load_jsonl, save_jsonl


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"[^a-z0-9 ]+", "", prompt)
    return prompt


def deduplicate(rows: List[Dict]) -> List[Dict]:
    seen = set()
    kept = []
    for row in rows:
        key = (normalize_prompt(row["prompt"]), row.get("source"), row.get("split"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate normalized EFSC JSONL files by prompt/source/split.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(load_jsonl(path))
    kept = deduplicate(rows)
    save_jsonl(kept, args.output)
    print(json.dumps({"input_rows": len(rows), "output_rows": len(kept), "removed": len(rows) - len(kept)}, indent=2))


if __name__ == "__main__":
    main()
