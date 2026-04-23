from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.utils import load_jsonl, save_jsonl


def canonicalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prompt_key(prompt: str) -> str:
    return hashlib.md5(canonicalize(prompt).encode("utf-8")).hexdigest()


def deduplicate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen = {}
    kept, removed = [], []
    for row in rows:
        key = prompt_key(row["prompt"])
        if key in seen:
            removed.append(row)
        else:
            seen[key] = row["id"]
            kept.append(row)
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--removed_output", required=True)
    args = parser.parse_args()
    kept, removed = deduplicate(load_jsonl(Path(args.input)))
    save_jsonl(kept, Path(args.output))
    save_jsonl(removed, Path(args.removed_output))
    print(json.dumps({"kept": len(kept), "removed": len(removed)}, indent=2))


if __name__ == "__main__":
    main()
