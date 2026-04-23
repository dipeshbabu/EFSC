import json
import subprocess
import sys
from pathlib import Path


def write_metric(path: Path, macro_f1: float) -> None:
    path.write_text(
        json.dumps(
            {
                "accuracy": macro_f1,
                "macro_f1": macro_f1,
                "benign_answer_rate": 0.8,
                "over_refusal_rate": 0.2,
                "harmful_refusal_rate": 0.9,
                "under_refusal_rate": 0.1,
                "clarify_accuracy": 0.7,
                "safe_answer_accuracy": 0.6,
                "family_consistency_score": 0.75,
            }
        ),
        encoding="utf-8",
    )


def test_seed_aggregate_and_main_table():
    tmp_path = Path("sample_data") / "result_generation_test"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_metric(metrics_dir / "roberta-base__efsc_full__seed1__xstest.json", 0.5)
    write_metric(metrics_dir / "roberta-base__efsc_full__seed2__xstest.json", 0.7)

    aggregate = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_seed_metrics.py",
            "--metric_files",
            str(metrics_dir / "roberta-base__efsc_full__seed1__xstest.json"),
            str(metrics_dir / "roberta-base__efsc_full__seed2__xstest.json"),
            "--output",
            str(aggregate),
        ],
        check=True,
    )
    assert "macro_f1" in aggregate.read_text(encoding="utf-8")

    table = tmp_path / "table.tex"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_main_table.py",
            "--rows",
            str(aggregate),
            "--labels",
            "EFSC",
            "--output_tex",
            str(table),
        ],
        check=True,
    )
    assert "\\begin{tabular}" in table.read_text(encoding="utf-8")
