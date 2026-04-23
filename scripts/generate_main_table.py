from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(stat, key: str) -> str:
    if key not in stat:
        return "--"
    value = stat[key]
    if isinstance(value, dict):
        return f"{float(value['mean']):.3f} $\\pm$ {float(value.get('std', 0.0)):.3f}"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_tex", required=True)
    args = parser.parse_args()
    if len(args.rows) != len(args.labels):
        raise ValueError("--rows and --labels must have the same length")

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Safety Utility & Harmful Refusal & Benign Retention & Ambiguity Cal. \\",
        r"\midrule",
    ]
    for label, path in zip(args.labels, args.rows):
        stat = load_json(Path(path))
        lines.append(
            f"{label} & {fmt(stat, 'safety_utility_score')} & {fmt(stat, 'harmful_refusal_rate')} & "
            f"{fmt(stat, 'benign_retention_rate')} & {fmt(stat, 'ambiguity_calibration')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out = Path(args.output_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
