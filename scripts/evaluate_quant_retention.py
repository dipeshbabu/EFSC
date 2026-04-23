from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric(obj: Dict[str, Any], key: str) -> float:
    value = obj[key]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def build_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in rows}


def has_factor_predictions(row: Dict[str, Any]) -> bool:
    return all(key in row for key in ("pred_harm_id", "pred_legit_id", "pred_uncertainty_id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_metrics", required=True)
    parser.add_argument("--quant_metrics", required=True)
    parser.add_argument("--fp_preds", required=True)
    parser.add_argument("--quant_preds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fp_metrics = load_json(Path(args.fp_metrics))
    q_metrics = load_json(Path(args.quant_metrics))
    fp_index = build_index(load_jsonl(Path(args.fp_preds)))
    q_index = build_index(load_jsonl(Path(args.quant_preds)))

    common_ids = sorted(set(fp_index) & set(q_index))
    action_agree = 0
    factor_agree = 0
    factor_count = 0
    for ex_id in common_ids:
        a = fp_index[ex_id]
        b = q_index[ex_id]
        action_agree += int(a["pred_action"] == b["pred_action"])
        if has_factor_predictions(a) and has_factor_predictions(b):
            factor_count += 1
            factor_agree += int(
                a["pred_harm_id"] == b["pred_harm_id"]
                and a["pred_legit_id"] == b["pred_legit_id"]
                and a["pred_uncertainty_id"] == b["pred_uncertainty_id"]
            )

    n = max(len(common_ids), 1)
    action_agreement = action_agree / n
    factor_agreement = factor_agree / factor_count if factor_count else 0.0

    fp_su = metric(fp_metrics, "safety_utility_score")
    q_su = metric(q_metrics, "safety_utility_score")
    fp_cfs = metric(fp_metrics, "counterfactual_action_stability")
    q_cfs = metric(q_metrics, "counterfactual_action_stability")
    fp_cfen = metric(fp_metrics, "counterfactual_action_sensitivity")
    q_cfen = metric(q_metrics, "counterfactual_action_sensitivity")

    out = {
        "action_agreement": action_agreement,
        "factor_agreement": factor_agreement,
        "factor_agreement_available": bool(factor_count),
        "quantized_safety_retention": q_su / max(fp_su, 1e-8),
        "quantized_structure_retention": 0.5 * ((q_cfs / max(fp_cfs, 1e-8)) + (q_cfen / max(fp_cfen, 1e-8))),
        "fp_safety_utility_score": fp_su,
        "quant_safety_utility_score": q_su,
        "num_common_examples": len(common_ids),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
