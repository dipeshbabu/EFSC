from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.data.prepare_data import preference_lists
from efsc.utils import load_jsonl, save_jsonl


def add_preferences(row: Dict[str, Any]) -> Dict[str, Any]:
    preferred, dispreferred = preference_lists(row["action_label"])
    row["preferred_over"] = preferred
    row["dispreferred_over"] = dispreferred
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [add_preferences(row) for row in load_jsonl(Path(args.input))]
    save_jsonl(rows, Path(args.output))
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
