from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, List


def parse_command(line: str) -> Dict[str, str]:
    tokens = shlex.split(line)
    parsed = {"script": tokens[1] if len(tokens) > 1 else ""}
    idx = 2
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            key = token[2:]
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                parsed[key] = tokens[idx + 1]
                idx += 2
            else:
                parsed[key] = "true"
                idx += 1
        else:
            idx += 1
    return parsed


def expected_artifacts(cmd: Dict[str, str], output_root: Path) -> List[Path]:
    run_id = cmd["run_id"]
    dataset = cmd["dataset_name"]
    quant = cmd["quant_mode"]
    artifacts = [
        output_root / "models" / run_id / "best.pt",
        output_root / "preds" / f"{run_id}__fp__{dataset}.jsonl",
        output_root / "preds" / f"{run_id}__{quant}__{dataset}.jsonl",
        output_root / "metrics" / f"{run_id}__fp__{dataset}.json",
        output_root / "metrics" / f"{run_id}__{quant}__{dataset}.json",
        output_root / "efficiency" / f"{run_id}.json",
    ]
    if cmd["script"] == "run_qefsc_quant_experiment.py":
        artifacts.extend([
            output_root / "preds" / f"{run_id}__{quant}_recal__{dataset}.jsonl",
            output_root / "metrics" / f"{run_id}__{quant}_recal__{dataset}.json",
            output_root / "retention" / f"{run_id}__{quant}__before__{dataset}.json",
            output_root / "retention" / f"{run_id}__{quant}__after__{dataset}.json",
        ])
    else:
        artifacts.append(output_root / "retention" / f"{run_id}__{quant}__{dataset}.json")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_script", default="runs/full_maintrack_grid.sh")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--report", required=True)
    parser.add_argument("--fail_on_missing", action="store_true")
    args = parser.parse_args()

    grid = Path(args.grid_script)
    output_root = Path(args.output_root)
    commands = []
    for line in grid.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            commands.append(parse_command(line))

    missing = []
    present = []
    for cmd in commands:
        for path in expected_artifacts(cmd, output_root):
            row = {"run_id": cmd.get("run_id", ""), "artifact": str(path)}
            if path.exists():
                present.append(row)
            else:
                missing.append(row)

    report = {
        "grid_script": str(grid),
        "output_root": str(output_root),
        "num_commands": len(commands),
        "num_present_artifacts": len(present),
        "num_missing_artifacts": len(missing),
        "missing": missing,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_missing and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
