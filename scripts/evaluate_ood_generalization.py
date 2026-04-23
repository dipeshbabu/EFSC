from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_metrics", required=True)
    parser.add_argument("--ood_metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    id_metrics = load_json(Path(args.id_metrics))
    ood_metrics = [load_json(Path(path)) for path in args.ood_metrics]
    keys = ["macro_f1", "harmful_refusal_rate", "benign_retention_rate", "over_refusal_rate", "ambiguity_calibration", "safety_utility_score"]
    result = {"id": {}, "ood_avg": {}, "drops": {}}
    for key in keys:
        if key not in id_metrics:
            continue
        id_val = float(id_metrics[key])
        vals = [float(metrics[key]) for metrics in ood_metrics if key in metrics]
        ood_avg = sum(vals) / max(len(vals), 1)
        result["id"][key] = id_val
        result["ood_avg"][key] = ood_avg
        result["drops"][key] = id_val - ood_avg
    save_json(result, Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
