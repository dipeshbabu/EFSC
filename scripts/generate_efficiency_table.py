from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(obj, key: str, default: float = 0.0) -> float:
    value = obj.get(key, default)
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+")
    parser.add_argument("--efficiency_files", nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_tex", required=True)
    args = parser.parse_args()

    row_paths = args.rows or args.efficiency_files
    if not row_paths:
        raise ValueError("Pass --rows or --efficiency_files")
    if len(row_paths) != len(args.labels):
        raise ValueError("Need one label per efficiency file")

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Trainable Params & Peak GPU MB & Tokens / Sec & Latency ms / ex \\",
        r"\midrule",
    ]
    for label, path in zip(args.labels, row_paths):
        row = load_json(Path(path))
        lines.append(
            f"{label} & {int(scalar(row, 'num_trainable_params'))} & "
            f"{scalar(row, 'peak_gpu_mem_mb'):.1f} & {scalar(row, 'tokens_per_sec'):.1f} & "
            f"{scalar(row, 'latency_ms_per_example'):.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out = Path(args.output_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
