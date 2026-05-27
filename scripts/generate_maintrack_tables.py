from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(row: Dict[str, str], key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if mean_key not in row or std_key not in row or row[mean_key] == "":
        return "--"
    return f"{float(row[mean_key]):.3f} $\\pm$ {float(row[std_key]):.3f}"


def write_table(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_main_table(rows: List[Dict[str, str]], dataset: str) -> List[str]:
    selected = [row for row in rows if row["dataset"] == dataset]
    lines = [
        "\\begin{tabular}{l l c c c c c}",
        "\\toprule",
        "Model & Method & Macro F1 & Harmful Refusal & Benign Retention & Over Refusal & Safety Utility \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['model']} & {row['method']} & {fmt(row, 'macro_f1')} & "
            f"{fmt(row, 'harmful_refusal_rate')} & {fmt(row, 'benign_retention_rate')} & "
            f"{fmt(row, 'over_refusal_rate')} & {fmt(row, 'safety_utility_score')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def build_ambiguity_table(rows: List[Dict[str, str]], dataset: str) -> List[str]:
    selected = [row for row in rows if row["dataset"] == dataset]
    lines = [
        "\\begin{tabular}{l l c c c}",
        "\\toprule",
        "Model & Method & Clarify Acc. & Ambiguity Calib. & Safe Answer Acc. \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(f"{row['model']} & {row['method']} & {fmt(row, 'clarify_accuracy')} & {fmt(row, 'ambiguity_calibration')} & {fmt(row, 'safe_answer_accuracy')} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def build_consistency_table(rows: List[Dict[str, str]], dataset: str) -> List[str]:
    selected = [row for row in rows if row["dataset"] == dataset]
    lines = [
        "\\begin{tabular}{l l c c c}",
        "\\toprule",
        "Model & Method & Family Consistency & CF Stability & CF Sensitivity \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(f"{row['model']} & {row['method']} & {fmt(row, 'family_consistency_score')} & {fmt(row, 'counterfactual_action_stability')} & {fmt(row, 'counterfactual_action_sensitivity')} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = load_csv(Path(args.merged_csv))
    out_dir = Path(args.output_dir)
    write_table(build_main_table(rows, args.dataset), out_dir / f"{args.dataset}_main_table.tex")
    write_table(build_ambiguity_table(rows, args.dataset), out_dir / f"{args.dataset}_ambiguity_table.tex")
    write_table(build_consistency_table(rows, args.dataset), out_dir / f"{args.dataset}_consistency_table.tex")
    print(f"Wrote main track tables to {out_dir}")


if __name__ == "__main__":
    main()
