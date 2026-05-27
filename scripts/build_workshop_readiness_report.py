from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPECTED_DATASETS = {
    "custom_auth": "data/processed/test_custom_auth.jsonl",
    "xstest": "data/processed/test_xstest.jsonl",
    "orbench_hard": "data/processed/test_orbench_hard.jsonl",
    "orbench_toxic": "data/processed/test_orbench_toxic.jsonl",
    "strongreject": "data/processed/test_strongreject.jsonl",
}
EXPECTED_METHODS = {"qefsc_fc", "plain_efsc", "direct_policy"}
PRIMARY_METRICS = [
    "safety_utility_score",
    "harmful_refusal_rate",
    "benign_retention_rate",
    "ambiguity_calibration",
    "macro_f1",
]
RETENTION_METRICS = [
    "quantized_safety_retention",
    "quantized_structure_retention",
    "action_agreement",
    "factor_agreement",
]
EFFICIENCY_METRICS = [
    "num_trainable_params",
    "peak_gpu_mem_mb",
    "tokens_per_sec",
    "latency_ms_per_example",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if parsed["script"] == "run_qefsc_quant_experiment.py":
        parsed["method"] = "qefsc_fc"
    else:
        parsed["method"] = parsed.get("baseline_type", "")
    return parsed


def read_grid(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    commands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            commands.append(parse_command(line))
    return commands


def missing_keys(path: Path, keys: Iterable[str]) -> List[str]:
    if not path.exists():
        return list(keys)
    try:
        obj = load_json(path)
    except json.JSONDecodeError:
        return list(keys)
    return [key for key in keys if key not in obj]


def expected_run_artifacts(cmd: Dict[str, str], output_root: Path) -> List[Path]:
    run_id = cmd.get("run_id", "")
    dataset = cmd.get("dataset_name", "")
    quant = cmd.get("quant_mode", "int4")
    artifacts = [
        output_root / "models" / run_id / "best.pt",
        output_root / "preds" / f"{run_id}__fp__{dataset}.jsonl",
        output_root / "preds" / f"{run_id}__{quant}__{dataset}.jsonl",
        output_root / "metrics" / f"{run_id}__fp__{dataset}.json",
        output_root / "metrics" / f"{run_id}__{quant}__{dataset}.json",
        output_root / "efficiency" / f"{run_id}.json",
    ]
    if cmd.get("method") == "qefsc_fc":
        artifacts.extend(
            [
                output_root / "preds" / f"{run_id}__{quant}_recal__{dataset}.jsonl",
                output_root / "metrics" / f"{run_id}__{quant}_recal__{dataset}.json",
                output_root / "retention" / f"{run_id}__{quant}__before__{dataset}.json",
                output_root / "retention" / f"{run_id}__{quant}__after__{dataset}.json",
            ]
        )
    else:
        artifacts.append(output_root / "retention" / f"{run_id}__{quant}__{dataset}.json")
    return artifacts


def paper_placeholder_rows(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if " & --" in line or line.strip().startswith("--"))


def summarize_grid(commands: List[Dict[str, str]]) -> Dict[str, Any]:
    by_dataset = Counter(cmd.get("dataset_name", "") for cmd in commands)
    by_method = Counter(cmd.get("method", "") for cmd in commands)
    by_model = Counter(cmd.get("model_name", "") for cmd in commands)
    method_dataset = defaultdict(set)
    for cmd in commands:
        method_dataset[cmd.get("method", "")].add(cmd.get("dataset_name", ""))
    return {
        "num_commands": len(commands),
        "by_dataset": dict(by_dataset),
        "by_method": dict(by_method),
        "by_model": dict(by_model),
        "method_dataset_coverage": {key: sorted(value) for key, value in method_dataset.items()},
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    grid = Path(args.grid_script)
    paper = Path(args.paper_tex)
    commands = read_grid(grid)

    missing_datasets = [
        {"dataset": name, "path": path}
        for name, path in EXPECTED_DATASETS.items()
        if not Path(path).exists()
    ]
    missing_artifacts = []
    for cmd in commands:
        for artifact in expected_run_artifacts(cmd, output_root):
            if not artifact.exists():
                missing_artifacts.append({"run_id": cmd.get("run_id", ""), "artifact": str(artifact)})

    metric_files = sorted((output_root / "metrics").glob("*.json")) if (output_root / "metrics").exists() else []
    retention_files = sorted((output_root / "retention").glob("*.json")) if (output_root / "retention").exists() else []
    efficiency_files = sorted((output_root / "efficiency").glob("*.json")) if (output_root / "efficiency").exists() else []

    metric_key_gaps = [
        {"file": str(path), "missing_keys": missing_keys(path, PRIMARY_METRICS)}
        for path in metric_files
        if missing_keys(path, PRIMARY_METRICS)
    ]
    retention_key_gaps = [
        {"file": str(path), "missing_keys": missing_keys(path, RETENTION_METRICS)}
        for path in retention_files
        if missing_keys(path, RETENTION_METRICS)
    ]
    efficiency_key_gaps = [
        {"file": str(path), "missing_keys": missing_keys(path, EFFICIENCY_METRICS)}
        for path in efficiency_files
        if missing_keys(path, EFFICIENCY_METRICS)
    ]

    warnings = []
    if missing_datasets:
        warnings.append("Some frozen test datasets are missing.")
    if not commands:
        warnings.append("No run grid commands found.")
    if missing_artifacts:
        warnings.append("Some expected run artifacts are missing.")
    if metric_key_gaps:
        warnings.append("Some metric files lack paper-critical metrics.")
    if retention_key_gaps:
        warnings.append("Some retention files lack quantization-retention metrics.")
    if efficiency_key_gaps:
        warnings.append("Some efficiency files lack deployment-cost metrics.")
    placeholders = paper_placeholder_rows(paper)
    if placeholders:
        warnings.append("Paper still contains placeholder result cells.")

    return {
        "status": "ready" if not warnings else "not_ready",
        "warnings": warnings,
        "grid": str(grid),
        "output_root": str(output_root),
        "paper_tex": str(paper),
        "grid_summary": summarize_grid(commands),
        "missing_datasets": missing_datasets,
        "missing_artifacts_count": len(missing_artifacts),
        "missing_artifacts_sample": missing_artifacts[:50],
        "metric_files": len(metric_files),
        "retention_files": len(retention_files),
        "efficiency_files": len(efficiency_files),
        "metric_key_gaps": metric_key_gaps[:50],
        "retention_key_gaps": retention_key_gaps[:50],
        "efficiency_key_gaps": efficiency_key_gaps[:50],
        "paper_placeholder_rows": placeholders,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Workshop Readiness Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Warnings",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Grid commands: {report['grid_summary']['num_commands']}",
            f"- Metric files: {report['metric_files']}",
            f"- Retention files: {report['retention_files']}",
            f"- Efficiency files: {report['efficiency_files']}",
            f"- Missing artifacts: {report['missing_artifacts_count']}",
            f"- Paper placeholder rows: {report['paper_placeholder_rows']}",
        ]
    )
    if report["missing_datasets"]:
        lines.extend(["", "## Missing Datasets"])
        lines.extend(f"- `{row['dataset']}`: `{row['path']}`" for row in report["missing_datasets"])
    if report["missing_artifacts_sample"]:
        lines.extend(["", "## Missing Artifact Sample"])
        lines.extend(f"- `{row['artifact']}`" for row in report["missing_artifacts_sample"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_script", default="runs/full_maintrack_grid.sh")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--paper_tex", default="paper/Template-2026/colm2026_conference.tex")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md")
    parser.add_argument("--fail_on_not_ready", action="store_true")
    args = parser.parse_args()

    report = build_report(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")

    if args.fail_on_not_ready and report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
