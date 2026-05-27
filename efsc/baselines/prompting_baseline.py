from __future__ import annotations
import argparse
from pathlib import Path
from efsc.config import load_config
from efsc.constants import ACTION2ID
from efsc.eval.metrics import classification_metrics, selective_refusal_metrics
from efsc.utils import load_jsonl, save_json

def heuristic_action(prompt: str) -> int:
    text = prompt.lower()
    if any(k in text for k in ["kill", "weapon", "phish", "malware", "explosive"]):
        if any(k in text for k in ["explain", "detect", "defend", "prevent", "authorized", "train employees"]): return ACTION2ID["SAFE_ANSWER"]
        return ACTION2ID["REFUSE"]
    if any(k in text for k in ["clarify", "unclear", "not sure"]): return ACTION2ID["CLARIFY"]
    return ACTION2ID["ANSWER"]

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); rows = load_jsonl(config["test_path"])
    y_true, y_pred, benign_mask, harmful_mask, preds = [], [], [], [], []
    for row in rows:
        pred = heuristic_action(row["prompt"]); y_pred.append(pred); y_true.append(ACTION2ID[row["action_label"]]); benign_mask.append(row["harm_label"] != "HIGH"); harmful_mask.append(row["harm_label"] == "HIGH"); preds.append({"id": row["id"], "pred_action": pred})
    metrics = classification_metrics(y_true, y_pred); metrics.update(selective_refusal_metrics(y_true, y_pred, benign_mask, harmful_mask))
    out = Path(config["output_dir"]); out.mkdir(parents=True, exist_ok=True); save_json(metrics, out / "metrics.json"); save_json({"predictions": preds}, out / "predictions.json")

if __name__ == "__main__":
    main()
