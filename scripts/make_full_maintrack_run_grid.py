from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_script", required=True)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--no_prompt_baseline", action="store_true")
    parser.add_argument("--no_baseline_recalibration", action="store_true")
    args = parser.parse_args()

    models = [
        ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"),
        ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ]
    datasets = [
        ("custom_auth", "data/processed/test_custom_auth.jsonl"),
        ("xstest", "data/processed/test_xstest.jsonl"),
        ("orbench_hard", "data/processed/test_orbench_hard.jsonl"),
        ("orbench_toxic", "data/processed/test_orbench_toxic.jsonl"),
        ("strongreject", "data/processed/test_strongreject.jsonl"),
    ]
    seeds = [1, 2, 3]

    lines = []
    available_datasets = []
    for dataset_name, dataset_path in datasets:
        if Path(dataset_path).exists() or not args.skip_missing:
            available_datasets.append((dataset_name, dataset_path))
            if not Path(dataset_path).exists():
                print(f"Including expected dataset even though file is currently missing: {dataset_path}")
        else:
            print(f"Skipping missing dataset: {dataset_path}")

    for model_tag, model_name in models:
        for seed in seeds:
            run_id = f"{model_tag}__qefsc_fc__seed{seed}"
            for dataset_name, dataset_path in available_datasets:
                lines.append(
                    f'python run_qefsc_quant_experiment.py '
                    f'--run_id {run_id} '
                    f'--model_name "{model_name}" '
                    f'--variant qefsc_fc '
                    f'--seed {seed} '
                    f'--test_path {dataset_path} '
                    f'--dataset_name {dataset_name} '
                    f'--quant_mode int4 '
                    f'--use_lora'
                )

            for baseline in ["plain_efsc", "direct_policy"]:
                b_run_id = f"{model_tag}__{baseline}__seed{seed}"
                for dataset_name, dataset_path in available_datasets:
                    recalibration_arg = "" if args.no_baseline_recalibration else " --recalibrate"
                    lines.append(
                        f'python run_baseline_quant_experiment.py '
                        f'--run_id {b_run_id} '
                        f'--model_name "{model_name}" '
                        f'--baseline_type {baseline} '
                        f'--seed {seed} '
                        f'--test_path {dataset_path} '
                        f'--dataset_name {dataset_name} '
                        f'--quant_mode int4 '
                        f'--use_lora'
                        f'{recalibration_arg}'
                    )
        if not args.no_prompt_baseline:
            run_id = f"{model_tag}__prompt_classifier__seed1"
            for dataset_name, dataset_path in available_datasets:
                lines.append(
                    f'python run_prompt_classifier_experiment.py '
                    f'--run_id {run_id} '
                    f'--model_name "{model_name}" '
                    f'--test_path {dataset_path} '
                    f'--dataset_name {dataset_name} '
                    f'--quant_mode int4'
                )

    out = Path(args.output_script)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} commands to {out}")


if __name__ == "__main__":
    main()
