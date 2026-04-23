from __future__ import annotations

import argparse
import json
from pathlib import Path

from efsc.data.prepare_data import assign_counterfactual_families, split_train_val_by_family, write_manifests
from efsc.utils import ensure_dir, load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign EFSC counterfactual family IDs and make family-safe train/val splits.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--family-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    for row in rows:
        row["split"] = "train_pool"
    assign_counterfactual_families(rows, family_size=args.family_size)
    train_rows, val_rows = split_train_val_by_family(rows, args.val_fraction, args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_jsonl(train_rows, output_dir / "train.jsonl")
    save_jsonl(val_rows, output_dir / "val.jsonl")
    write_manifests(train_rows, val_rows, Path(output_dir))
    print(json.dumps({"train": len(train_rows), "val": len(val_rows)}, indent=2))


if __name__ == "__main__":
    main()
