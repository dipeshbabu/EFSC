from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--metric_name", default="macro_f1")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    groups: Dict[str, Dict] = {}
    for path in Path(args.metrics_dir).glob("*.json"):
        obj = load_json(path)
        parts = path.stem.split("__")
        if len(parts) < 3:
            continue
        group_key = "__".join(parts[:-1])
        metric_value = obj.get(args.metric_name)
        if metric_value is None:
            continue
        if group_key not in groups or metric_value > groups[group_key]["score"]:
            groups[group_key] = {"file": path.name, "score": metric_value}

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(f"Wrote best checkpoint summary to {output}")


if __name__ == "__main__":
    main()
