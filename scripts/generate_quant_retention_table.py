from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(obj, key: str) -> float:
    value = obj[key]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


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
        r"Method & QSR & QStR & Action Agree & Factor Agree \\",
        r"\midrule",
    ]
    for label, path in zip(args.labels, args.rows):
        row = load_json(Path(path))
        lines.append(
            f"{label} & {scalar(row, 'quantized_safety_retention'):.3f} & "
            f"{scalar(row, 'quantized_structure_retention'):.3f} & "
            f"{scalar(row, 'action_agreement'):.3f} & {scalar(row, 'factor_agreement'):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out = Path(args.output_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
