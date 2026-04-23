from __future__ import annotations

import argparse
import json

from efsc.data.prepare_data import convert_wildjailbreak, load_hf_dataset
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize WildJailbreak into the EFSC JSONL schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-jsonl", default=None, help="Optional local JSONL export. If omitted, Hugging Face is used.")
    parser.add_argument("--dataset", default="allenai/wildjailbreak")
    parser.add_argument("--config", default="train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.input_jsonl:
        raw = load_jsonl(args.input_jsonl)
    else:
        raw = load_hf_dataset(args.dataset, args.config, split=args.split, delimiter="\t", keep_default_na=False, cache_dir=args.cache_dir)
    rows = convert_wildjailbreak(raw, split="train_pool", limit=args.limit)
    save_jsonl(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
