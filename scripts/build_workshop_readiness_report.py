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
EXPECTED_METHODS = {"qefsc_fc", "plain_efsc", "direct_policy", "prompt_classifier"}
EXPECTED_ACTIONS = {"ANSWER", "SAFE_ANSWER", "CLARIFY", "REFUSE"}
EXPECTED_HARM_LABELS = {"LOW", "MEDIUM", "HIGH"}
EXPECTED_DATASET_ACTIONS = {
    "custom_auth": {"ANSWER", "SAFE_ANSWER", "CLARIFY"},
    "xstest": {"ANSWER"},
    "orbench_hard": {"SAFE_ANSWER"},
    "orbench_toxic": {"REFUSE"},
    "strongreject": {"REFUSE"},
}
PRIMARY_METRICS = [
    "num_examples",
    "num_harmful",
    "num_benign",
    "num_gold_refuse",
    "safety_utility_score",
    "harmful_refusal_rate",
    "under_refusal_rate",
    "benign_retention_rate",
    "over_refusal_rate",
    "ambiguity_calibration",
    "macro_f1",
]
RETENTION_METRICS = [
    "quantized_safety_retention",
    "quantized_structure_retention",
    "action_agreement",
    "factor_agreement",
    "action_flip_rate",
    "bad_action_flip_rate",
    "harmful_bad_flip_rate",
    "benign_bad_flip_rate",
    "safety_utility_delta",
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
    elif parsed["script"] == "run_prompt_classifier_experiment.py":
        parsed["method"] = "prompt_classifier"
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_counts(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    return {
        "path": str(path),
        "num_rows": len(rows),
        "actions": dict(Counter(row.get("gold_action", row.get("action_label", "")) for row in rows)),
        "harm": dict(Counter(row.get("harm_label", "") for row in rows)),
    }


def build_label_report() -> Dict[str, Any]:
    split_paths = {
        "train": Path("data/processed/train.jsonl"),
        "val": Path("data/processed/val.jsonl"),
        **{f"test_{name}": Path(path) for name, path in EXPECTED_DATASETS.items()},
    }
    by_split = {name: label_counts(path) for name, path in split_paths.items()}
    train_val_rows = []
    for split in ("train", "val"):
        train_val_rows.extend(load_jsonl(split_paths[split]))
    all_rows = []
    for path in split_paths.values():
        all_rows.extend(load_jsonl(path))
    train_actions = Counter(row.get("gold_action", row.get("action_label", "")) for row in train_val_rows)
    train_harm = Counter(row.get("harm_label", "") for row in train_val_rows)
    all_actions = Counter(row.get("gold_action", row.get("action_label", "")) for row in all_rows)
    all_harm = Counter(row.get("harm_label", "") for row in all_rows)

    warnings = []
    missing_train_actions = sorted(EXPECTED_ACTIONS - set(train_actions))
    if missing_train_actions:
        warnings.append(f"Train/val is missing action labels needed to learn selective refusal: {missing_train_actions}.")
    missing_train_harm = sorted(EXPECTED_HARM_LABELS - set(train_harm))
    if missing_train_harm:
        warnings.append(f"Train/val is missing harm labels needed for factor learning: {missing_train_harm}.")
    missing_all_actions = sorted(EXPECTED_ACTIONS - set(all_actions))
    if missing_all_actions:
        warnings.append(f"Full processed corpus is missing action labels: {missing_all_actions}.")
    missing_all_harm = sorted(EXPECTED_HARM_LABELS - set(all_harm))
    if missing_all_harm:
        warnings.append(f"Full processed corpus is missing harm labels: {missing_all_harm}.")
    for dataset, required_actions in EXPECTED_DATASET_ACTIONS.items():
        counts = by_split[f"test_{dataset}"]["actions"]
        missing = sorted(required_actions - set(counts))
        if missing:
            warnings.append(f"Dataset `{dataset}` is missing expected gold actions: {missing}.")

    return {
        "by_split": by_split,
        "train_val_actions": dict(train_actions),
        "train_val_harm": dict(train_harm),
        "all_actions": dict(all_actions),
        "all_harm": dict(all_harm),
        "warnings": warnings,
    }


def expected_run_artifacts(cmd: Dict[str, str], output_root: Path) -> List[Path]:
    run_id = cmd.get("run_id", "")
    dataset = cmd.get("dataset_name", "")
    quant = cmd.get("quant_mode", "int4")
    if cmd.get("method") == "prompt_classifier":
        return [
            output_root / "preds" / f"{run_id}__fp__{dataset}.jsonl",
            output_root / "preds" / f"{run_id}__{quant}__{dataset}.jsonl",
            output_root / "metrics" / f"{run_id}__fp__{dataset}.json",
            output_root / "metrics" / f"{run_id}__{quant}__{dataset}.json",
            output_root / "retention" / f"{run_id}__{quant}__{dataset}.json",
            output_root / "efficiency" / f"{run_id}.json",
        ]
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
    elif cmd.get("recalibrate") == "true":
        artifacts.extend(
            [
                output_root / "preds" / f"{run_id}__{quant}_recal__{dataset}.jsonl",
                output_root / "metrics" / f"{run_id}__{quant}_recal__{dataset}.json",
                output_root / "retention" / f"{run_id}__{quant}_recal__{dataset}.json",
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
    label_report = build_label_report()

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
    missing_methods = sorted(EXPECTED_METHODS - {cmd.get("method", "") for cmd in commands})
    if missing_methods:
        warnings.append(f"Run grid is missing required top-paper comparison methods: {missing_methods}.")
    unrecalibrated_baselines = [
        cmd
        for cmd in commands
        if cmd.get("method") in {"plain_efsc", "direct_policy"} and cmd.get("recalibrate") != "true"
    ]
    if unrecalibrated_baselines:
        warnings.append("Baseline grid commands do not all request post-quantization recalibration.")
    warnings.extend(label_report["warnings"])
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
        "label_report": label_report,
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
    if report["label_report"]["warnings"]:
        lines.extend(["", "## Label Coverage Warnings"])
        lines.extend(f"- {warning}" for warning in report["label_report"]["warnings"])
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
