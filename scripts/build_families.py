from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.utils import load_jsonl, save_json, save_jsonl


def lexical_signature(prompt: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", prompt.lower())
    tokens = [token for token in text.split() if len(token) > 3]
    return " ".join(sorted(set(tokens[:12])))


def assign_family_ids(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["topic"], lexical_signature(row["prompt"])[:80])].append(row)
    manifest = {}
    for index, members in enumerate(grouped.values()):
        family_id = f"fam_{index:06d}"
        for member in members:
            member["family_id"] = family_id
        manifest[family_id] = {
            "size": len(members),
            "topics": sorted({member["topic"] for member in members}),
            "sources": sorted({member["source"] for member in members}),
            "ids": [member["id"] for member in members],
        }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest_output", required=True)
    args = parser.parse_args()
    rows, manifest = assign_family_ids(load_jsonl(Path(args.input)))
    save_jsonl(rows, Path(args.output))
    save_json(manifest, Path(args.manifest_output))
    print(json.dumps({"rows": len(rows), "families": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
