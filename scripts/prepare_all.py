from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def append_files(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_handle:
        for path in paths:
            with path.open("r", encoding="utf-8") as in_handle:
                shutil.copyfileobj(in_handle, out_handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    work_dir = Path(args.work_dir)
    manifest_dir = work_dir / "manifests"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    run(["python", "scripts/prepare_wildjailbreak.py", "--input", str(raw_dir / "wildjailbreak_train.jsonl"), "--output", str(work_dir / "wildjailbreak_train.norm.jsonl"), "--split", "train"])
    run(["python", "scripts/prepare_falsereject.py", "--input", str(raw_dir / "falsereject_train.jsonl"), "--output", str(work_dir / "falsereject_train.norm.jsonl"), "--split", "train"])
    run(["python", "scripts/prepare_custom_auth.py", "--input_csv", str(raw_dir / "custom_auth_train.csv"), "--output", str(work_dir / "custom_auth_train.norm.jsonl"), "--split", "train"])

    run(["python", "scripts/prepare_xstest.py", "--input", str(raw_dir / "xstest_test.jsonl"), "--output", str(work_dir / "test_xstest.jsonl")])
    run(["python", "scripts/prepare_orbench.py", "--hard_input", str(raw_dir / "orbench_hard_test.jsonl"), "--toxic_input", str(raw_dir / "orbench_toxic_test.jsonl"), "--hard_output", str(work_dir / "test_orbench_hard.jsonl"), "--toxic_output", str(work_dir / "test_orbench_toxic.jsonl")])
    run(["python", "scripts/prepare_strongreject.py", "--input", str(raw_dir / "strongreject_test.jsonl"), "--output", str(work_dir / "test_strongreject.jsonl")])
    run(["python", "scripts/prepare_custom_auth.py", "--input_csv", str(raw_dir / "custom_auth_test.csv"), "--output", str(work_dir / "test_custom_auth.jsonl"), "--split", "test"])

    merged_train = work_dir / "merged_train.jsonl"
    append_files([
        work_dir / "wildjailbreak_train.norm.jsonl",
        work_dir / "falsereject_train.norm.jsonl",
        work_dir / "custom_auth_train.norm.jsonl",
    ], merged_train)

    run(["python", "scripts/deduplicate_prompts.py", "--input", str(merged_train), "--output", str(work_dir / "merged_train.dedup.jsonl"), "--removed_output", str(work_dir / "removed_duplicates.jsonl")])
    run(["python", "scripts/build_families.py", "--input", str(work_dir / "merged_train.dedup.jsonl"), "--output", str(work_dir / "merged_train.family.jsonl"), "--manifest_output", str(manifest_dir / "family_manifest.json")])
    run(["python", "scripts/build_splits.py", "--input", str(work_dir / "merged_train.family.jsonl"), "--train_output", str(work_dir / "train.jsonl"), "--val_output", str(work_dir / "val.jsonl"), "--manifest_output", str(manifest_dir / "split_manifest.json"), "--val_ratio", str(args.val_ratio), "--seed", str(args.seed)])
    run(["python", "scripts/build_preferences.py", "--input", str(work_dir / "train.jsonl"), "--output", str(work_dir / "train.pref.jsonl")])
    run(["python", "scripts/build_preferences.py", "--input", str(work_dir / "val.jsonl"), "--output", str(work_dir / "val.pref.jsonl")])

    for name in ["train.jsonl", "val.jsonl", "test_xstest.jsonl", "test_orbench_hard.jsonl", "test_orbench_toxic.jsonl", "test_strongreject.jsonl", "test_custom_auth.jsonl"]:
        run(["python", "scripts/summarize_data.py", "--input", str(work_dir / name), "--output", str(work_dir / f"{name}.summary.json")])

    run([
        "python", "scripts/acceptance_checks.py",
        "--input",
        str(work_dir / "train.jsonl"),
        str(work_dir / "val.jsonl"),
        str(work_dir / "test_xstest.jsonl"),
        str(work_dir / "test_orbench_hard.jsonl"),
        str(work_dir / "test_orbench_toxic.jsonl"),
        str(work_dir / "test_strongreject.jsonl"),
        str(work_dir / "test_custom_auth.jsonl"),
        "--output",
        str(work_dir / "acceptance_report.json"),
    ])


if __name__ == "__main__":
    main()
