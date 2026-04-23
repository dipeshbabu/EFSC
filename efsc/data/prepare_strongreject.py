from __future__ import annotations

import argparse
import json

from efsc.data.prepare_data import convert_strongreject, load_hf_dataset
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize StrongREJECT into the EFSC JSONL schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--dataset", default="AlignmentResearch/StrongREJECT")
    parser.add_argument("--fallback-dataset", default="Machlovi/strongreject-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.input_jsonl:
        raw = load_jsonl(args.input_jsonl)
    else:
        try:
            raw = load_hf_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
        except Exception:
            raw = load_hf_dataset(args.fallback_dataset, split="train", cache_dir=args.cache_dir)
    rows = convert_strongreject(raw, limit=args.limit)
    save_jsonl(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
