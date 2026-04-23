from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efsc.constants import ID2ACTION
from src.eval.metrics import compute_all_metrics


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{"):
        data = json.loads(text)
        rows = data.get("predictions", [])
    else:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [normalize_prediction_row(row) for row in rows]


def normalize_action(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ID2ACTION[int(value)]


def normalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    normalized["gold_action"] = normalize_action(row.get("gold_action_label", row.get("gold_action")))
    normalized["pred_action"] = normalize_action(row.get("pred_action_label", row.get("pred_action")))
    normalized.setdefault("prompt", row.get("prompt", ""))
    normalized.setdefault("topic", row.get("topic", "unknown"))
    normalized.setdefault("family_id", row.get("family_id", ""))
    return normalized


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_prediction_rows(Path(args.predictions))
    metrics = compute_all_metrics(rows)
    save_json(metrics, Path(args.output))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
