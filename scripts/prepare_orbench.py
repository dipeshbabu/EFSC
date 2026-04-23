from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.data.prepare_data import convert_orbench
from efsc.utils import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_input", required=True)
    parser.add_argument("--toxic_input", required=True)
    parser.add_argument("--hard_output", required=True)
    parser.add_argument("--toxic_output", required=True)
    args = parser.parse_args()
    hard = convert_orbench(load_jsonl(Path(args.hard_input)), source="orbench_hard")
    toxic = convert_orbench(load_jsonl(Path(args.toxic_input)), source="orbench_toxic")
    save_jsonl(hard, Path(args.hard_output))
    save_jsonl(toxic, Path(args.toxic_output))
    print(json.dumps({"orbench_hard": len(hard), "orbench_toxic": len(toxic)}, indent=2))


if __name__ == "__main__":
    main()
