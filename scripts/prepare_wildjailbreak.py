from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.data.prepare_data import convert_wildjailbreak
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw WildJailbreak JSONL")
    parser.add_argument("--output", required=True, help="Normalized JSONL")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()
    rows = convert_wildjailbreak(load_jsonl(Path(args.input)), split=args.split)
    save_jsonl(rows, Path(args.output))
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
