from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def build_registry() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    models = [
        ("roberta-base", "small"),
        ("mistral-7b-instruct", "medium"),
        ("llama-3.2-3b-instruct", "medium"),
    ]
    methods = [
        "direct_classifier",
        "efsc_stage1",
        "efsc_stage1_stage2",
        "efsc_full",
        "efsc_no_harm_head",
        "efsc_no_legit_head",
        "efsc_no_uncertainty_head",
        "efsc_no_counterfactual",
        "efsc_no_preference",
        "lora_baseline",
    ]
    for model_name, scale in models:
        for method in methods:
            for seed in ("1", "2", "3"):
                rows.append({
                    "run_id": f"{model_name}__{method}__seed{seed}",
                    "model_name": model_name,
                    "model_scale": scale,
                    "method": method,
                    "seed": seed,
                    "status": "planned",
                    "notes": "",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    rows = build_registry()
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
