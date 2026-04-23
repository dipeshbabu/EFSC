from __future__ import annotations

import argparse
import json

from efsc.data.prepare_data import convert_xstest, load_hf_dataset
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize XSTest into the EFSC JSONL schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--dataset", default="walledai/XSTest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raw = load_jsonl(args.input_jsonl) if args.input_jsonl else load_hf_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    rows = convert_xstest(raw, limit=args.limit)
    save_jsonl(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
