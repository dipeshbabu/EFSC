from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.data.dataset_stats import compute_stats
from efsc.utils import load_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    save_json(compute_stats(load_jsonl(Path(args.input))), Path(args.output))


if __name__ == "__main__":
    main()
