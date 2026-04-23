from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.utils import load_jsonl, save_json, save_jsonl


def family_split(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    family_ids = list(families)
    random.Random(seed).shuffle(family_ids)
    n_val = max(1, int(len(family_ids) * val_ratio)) if family_ids else 0
    val_families = set(family_ids[:n_val])
    train_rows, val_rows = [], []
    for family_id, members in families.items():
        target = val_rows if family_id in val_families else train_rows
        split = "val" if family_id in val_families else "train"
        for member in members:
            member["split"] = split
            target.append(member)
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--manifest_output", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_rows, val_rows = family_split(load_jsonl(Path(args.input)), args.val_ratio, args.seed)
    manifest = {
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "train_families": sorted({row["family_id"] for row in train_rows}),
        "val_families": sorted({row["family_id"] for row in val_rows}),
    }
    save_jsonl(train_rows, Path(args.train_output))
    save_jsonl(val_rows, Path(args.val_output))
    save_json(manifest, Path(args.manifest_output))
    print(json.dumps({"train": len(train_rows), "val": len(val_rows)}, indent=2))


if __name__ == "__main__":
    main()
