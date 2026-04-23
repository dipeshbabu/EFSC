from __future__ import annotations

import argparse
import json

from efsc.data.prepare_data import convert_orbench, load_hf_dataset
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize OR-Bench hard or toxic split into the EFSC JSONL schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--dataset", default="bench-llms/or-bench")
    parser.add_argument("--subset", choices=["or-bench-hard-1k", "or-bench-toxic"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raw = load_jsonl(args.input_jsonl) if args.input_jsonl else load_hf_dataset(args.dataset, args.subset, split=args.split, cache_dir=args.cache_dir)
    source = "orbench_toxic" if args.subset == "or-bench-toxic" else "orbench_hard"
    rows = convert_orbench(raw, source=source, limit=args.limit)
    save_jsonl(rows, args.output)
    print(json.dumps({"rows": len(rows), "source": source, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
