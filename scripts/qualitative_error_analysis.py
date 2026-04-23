from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in rows}


def correct(row: Dict[str, Any]) -> bool:
    return row["gold_action"] == row["pred_action"]


def render_row(row: Dict[str, Any], primary=None, baseline=None, quant=None) -> str:
    lines = [
        f"- id: `{row.get('id', '')}`",
        f"  - topic: `{row.get('topic', 'unknown')}`; variant: `{row.get('variant_type', 'unknown')}`; uncertainty: `{row.get('uncertainty_label', 'unknown')}`",
        f"  - gold: `{row.get('gold_action', '')}`",
    ]
    if primary is not None:
        lines.append(f"  - primary: `{primary.get('pred_action', '')}`")
    if baseline is not None:
        lines.append(f"  - baseline: `{baseline.get('pred_action', '')}`")
    if quant is not None:
        lines.append(f"  - quantized: `{quant.get('pred_action', '')}`")
    prompt = row.get("prompt", "").replace("\n", " ").strip()
    if len(prompt) > 500:
        prompt = prompt[:497] + "..."
    lines.append(f"  - prompt: {prompt}")
    return "\n".join(lines)


def section(title: str, rows: List[str]) -> str:
    if not rows:
        return f"## {title}\n\nNo examples found.\n"
    return f"## {title}\n\n" + "\n\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_preds", required=True)
    parser.add_argument("--baseline_preds")
    parser.add_argument("--quant_preds")
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--max_examples", type=int, default=12)
    args = parser.parse_args()

    primary = index_by_id(load_jsonl(Path(args.primary_preds)))
    baseline = index_by_id(load_jsonl(Path(args.baseline_preds))) if args.baseline_preds else {}
    quant = index_by_id(load_jsonl(Path(args.quant_preds))) if args.quant_preds else {}

    primary_beats_baseline = []
    baseline_beats_primary = []
    quant_changed = []
    quant_fixed = []
    quant_broke = []

    for ex_id, p in primary.items():
        b = baseline.get(ex_id)
        q = quant.get(ex_id)
        if b is not None:
            if correct(p) and not correct(b) and len(primary_beats_baseline) < args.max_examples:
                primary_beats_baseline.append(render_row(p, primary=p, baseline=b))
            if correct(b) and not correct(p) and len(baseline_beats_primary) < args.max_examples:
                baseline_beats_primary.append(render_row(p, primary=p, baseline=b))
        if q is not None and p["pred_action"] != q["pred_action"] and len(quant_changed) < args.max_examples:
            quant_changed.append(render_row(p, primary=p, quant=q))
        if q is not None and correct(q) and not correct(p) and len(quant_fixed) < args.max_examples:
            quant_fixed.append(render_row(p, primary=p, quant=q))
        if q is not None and correct(p) and not correct(q) and len(quant_broke) < args.max_examples:
            quant_broke.append(render_row(p, primary=p, quant=q))

    text = "\n".join([
        "# Qualitative Error Analysis",
        "",
        f"Primary predictions: `{args.primary_preds}`",
        f"Baseline predictions: `{args.baseline_preds or ''}`",
        f"Quantized predictions: `{args.quant_preds or ''}`",
        "",
        section("Primary Correct, Baseline Wrong", primary_beats_baseline),
        section("Baseline Correct, Primary Wrong", baseline_beats_primary),
        section("Quantization Changed Action", quant_changed),
        section("Quantization Fixed FP Error", quant_fixed),
        section("Quantization Broke FP Correct Prediction", quant_broke),
    ])

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote qualitative analysis to {output}")


if __name__ == "__main__":
    main()
