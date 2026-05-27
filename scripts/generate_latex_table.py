from __future__ import annotations

import argparse
import csv
from pathlib import Path


def fmt(mean_val: str, std_val: str) -> str:
    try:
        return f"{float(mean_val):.3f} $\\pm$ {float(std_val):.3f}"
    except Exception:
        return "--"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_tex", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.merged_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == args.dataset:
                rows.append(row)

    lines = [
        "\\begin{tabular}{l l c c c c}",
        "\\toprule",
        "Model & Method & Acc. & Macro F1 & Harmful Refusal & Over Refusal \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['method']} & "
            f"{fmt(row['accuracy_mean'], row['accuracy_std'])} & "
            f"{fmt(row['macro_f1_mean'], row['macro_f1_std'])} & "
            f"{fmt(row['harmful_refusal_rate_mean'], row['harmful_refusal_rate_std'])} & "
            f"{fmt(row['over_refusal_rate_mean'], row['over_refusal_rate_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    output = Path(args.output_tex)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {output}")


if __name__ == "__main__":
    main()
