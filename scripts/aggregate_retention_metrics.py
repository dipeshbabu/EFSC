from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


FIELDS = [
    "action_agreement",
    "factor_agreement",
    "quantized_safety_retention",
    "quantized_structure_retention",
    "fp_safety_utility_score",
    "quant_safety_utility_score",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(vals):
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": mean(vals), "std": stdev(vals), "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention_files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [load_json(Path(path)) for path in args.retention_files]
    out = {}
    for field in FIELDS:
        vals = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        if vals:
            out[field] = summarize(vals)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
