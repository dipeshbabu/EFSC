from __future__ import annotations
import argparse
import json
from pathlib import Path
from efsc.eval.compute_family_metrics import compute_family_metrics
from efsc.eval.metrics import selective_refusal_metrics
from efsc.utils import load_jsonl, save_json

def load_rows(path: str):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{"):
        return json.loads(text).get("predictions", [])
    return load_jsonl(path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    y_true = [int(row["gold_action"]) for row in rows if "gold_action" in row]
    y_pred = [int(row["pred_action"]) for row in rows if "pred_action" in row]
    benign_mask = [row.get("variant_type") not in {"harmful_malicious", "jailbreak", "toxic_harmful"} for row in rows if "pred_action" in row]
    harmful_mask = [row.get("variant_type") in {"harmful_malicious", "jailbreak", "toxic_harmful"} for row in rows if "pred_action" in row]
    metrics = {"num_rows": len(rows)}
    if y_true and len(y_true) == len(y_pred):
        metrics.update(selective_refusal_metrics(y_true, y_pred, benign_mask, harmful_mask))
    metrics.update(compute_family_metrics(rows))
    save_json(metrics, Path(args.output))

if __name__ == "__main__":
    main()
