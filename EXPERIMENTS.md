# EXPERIMENTS

This runbook is the paper-facing command plan for **QEFSC-FC**.

## Goal

Evaluate whether feature-conditioned factorized safety control preserves safety behavior under low-bit deployment.

Compared methods:

- `qefsc_fc`: proposed multi-layer feature-conditioned factorized controller
- `plain_efsc`: final-hidden-state factorized controller
- `direct_policy`: direct action classifier

Main model:

- `Qwen/Qwen2.5-3B-Instruct`

Validation model:

- `meta-llama/Llama-3.1-8B-Instruct`

## Commands

Copy commands in order. Each command has its own block.

Purpose: materialize the full frozen dataset set in `data/processed` before generating experiment commands.

```bash
python -m efsc.data.prepare_all --output-dir data/processed --custom-auth data/raw/custom_auth/custom_auth.jsonl
```

Purpose: create the active run grid. Missing datasets are skipped unless `--include_missing` is used.

```bash
python scripts/make_full_maintrack_run_grid.py --output_script runs/full_maintrack_grid.sh
```

Purpose: train/evaluate QEFSC-FC on Qwen seed 1, custom_auth, FP plus INT4, then recalibrate the controller after quantization.

```bash
python run_qefsc_quant_experiment.py --run_id qwen25_3b__qefsc_fc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --variant qefsc_fc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: train/evaluate the Plain EFSC baseline with the same Qwen seed 1, custom_auth, FP plus INT4 protocol.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__plain_efsc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type plain_efsc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: train/evaluate the Direct Policy baseline with the same Qwen seed 1, custom_auth, FP plus INT4 protocol.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__direct_policy__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type direct_policy --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: optional fairness run; add post-quantization recalibration to a baseline. Use the same pattern for `direct_policy` if needed.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__plain_efsc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type plain_efsc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora --recalibrate
```

Purpose: check missing artifacts after the smoke runs. Use `--fail_on_missing` in CI or final packaging.

```bash
python scripts/check_maintrack_artifacts.py --grid_script runs/full_maintrack_grid.sh --report outputs/aggregates/artifact_check.json
```

Purpose: aggregate QEFSC-FC FP metrics across three Qwen seeds for custom_auth.

```bash
python scripts/aggregate_seed_metrics.py --metric_files outputs/metrics/qwen25_3b__qefsc_fc__seed1__fp__custom_auth.json outputs/metrics/qwen25_3b__qefsc_fc__seed2__fp__custom_auth.json outputs/metrics/qwen25_3b__qefsc_fc__seed3__fp__custom_auth.json --output outputs/aggregates/qwen25_3b__qefsc_fc__fp__custom_auth.json
```

Purpose: aggregate Direct Policy FP metrics across three Qwen seeds for custom_auth.

```bash
python scripts/aggregate_seed_metrics.py --metric_files outputs/metrics/qwen25_3b__direct_policy__seed1__fp__custom_auth.json outputs/metrics/qwen25_3b__direct_policy__seed2__fp__custom_auth.json outputs/metrics/qwen25_3b__direct_policy__seed3__fp__custom_auth.json --output outputs/aggregates/qwen25_3b__direct_policy__fp__custom_auth.json
```

Purpose: aggregate Plain EFSC FP metrics across three Qwen seeds for custom_auth.

```bash
python scripts/aggregate_seed_metrics.py --metric_files outputs/metrics/qwen25_3b__plain_efsc__seed1__fp__custom_auth.json outputs/metrics/qwen25_3b__plain_efsc__seed2__fp__custom_auth.json outputs/metrics/qwen25_3b__plain_efsc__seed3__fp__custom_auth.json --output outputs/aggregates/qwen25_3b__plain_efsc__fp__custom_auth.json
```

Purpose: aggregate QEFSC-FC INT4 post-recalibration retention across three Qwen seeds.

```bash
python scripts/aggregate_retention_metrics.py --retention_files outputs/retention/qwen25_3b__qefsc_fc__seed1__int4__after__custom_auth.json outputs/retention/qwen25_3b__qefsc_fc__seed2__int4__after__custom_auth.json outputs/retention/qwen25_3b__qefsc_fc__seed3__int4__after__custom_auth.json --output outputs/aggregates/qwen25_3b__qefsc_fc__int4_after_retention__custom_auth.json
```

Purpose: aggregate Direct Policy INT4 retention across three Qwen seeds.

```bash
python scripts/aggregate_retention_metrics.py --retention_files outputs/retention/qwen25_3b__direct_policy__seed1__int4__custom_auth.json outputs/retention/qwen25_3b__direct_policy__seed2__int4__custom_auth.json outputs/retention/qwen25_3b__direct_policy__seed3__int4__custom_auth.json --output outputs/aggregates/qwen25_3b__direct_policy__int4_retention__custom_auth.json
```

Purpose: aggregate Plain EFSC INT4 retention across three Qwen seeds.

```bash
python scripts/aggregate_retention_metrics.py --retention_files outputs/retention/qwen25_3b__plain_efsc__seed1__int4__custom_auth.json outputs/retention/qwen25_3b__plain_efsc__seed2__int4__custom_auth.json outputs/retention/qwen25_3b__plain_efsc__seed3__int4__custom_auth.json --output outputs/aggregates/qwen25_3b__plain_efsc__int4_retention__custom_auth.json
```

Purpose: build a flat CSV for quick result sanity checks and spreadsheet inspection.

```bash
python scripts/build_results_dataframe.py --metrics_dir outputs/metrics --output_csv outputs/aggregates/all_metrics_flat.csv
```

Purpose: run paired permutation significance tests comparing QEFSC-FC against Direct Policy on FP predictions.

```bash
python scripts/statistical_significance.py --primary_preds outputs/preds/qwen25_3b__qefsc_fc__seed1__fp__custom_auth.jsonl --baseline_preds outputs/preds/qwen25_3b__direct_policy__seed1__fp__custom_auth.jsonl --metrics safety_utility_score accuracy macro_f1 --output outputs/aggregates/qwen25_qefsc_vs_direct_seed1_significance.json
```

Purpose: generate qualitative appendix examples comparing QEFSC-FC, Direct Policy, and QEFSC-FC INT4 predictions.

```bash
python scripts/qualitative_error_analysis.py --primary_preds outputs/preds/qwen25_3b__qefsc_fc__seed1__fp__custom_auth.jsonl --baseline_preds outputs/preds/qwen25_3b__direct_policy__seed1__fp__custom_auth.jsonl --quant_preds outputs/preds/qwen25_3b__qefsc_fc__seed1__int4_recal__custom_auth.jsonl --output_md outputs/analysis/qwen25_seed1_qualitative.md
```

Purpose: generate the main FP result LaTeX table from aggregated metrics.

```bash
python scripts/generate_main_table.py --rows outputs/aggregates/qwen25_3b__direct_policy__fp__custom_auth.json outputs/aggregates/qwen25_3b__plain_efsc__fp__custom_auth.json outputs/aggregates/qwen25_3b__qefsc_fc__fp__custom_auth.json --labels "Direct Policy" "Plain EFSC" "QEFSC-FC" --output_tex outputs/tables/main_custom_auth_fp.tex
```

Purpose: generate the INT4 retention LaTeX table from aggregated retention metrics.

```bash
python scripts/generate_quant_retention_table.py --rows outputs/aggregates/qwen25_3b__direct_policy__int4_retention__custom_auth.json outputs/aggregates/qwen25_3b__plain_efsc__int4_retention__custom_auth.json outputs/aggregates/qwen25_3b__qefsc_fc__int4_after_retention__custom_auth.json --labels "Direct Policy" "Plain EFSC" "QEFSC-FC" --output_tex outputs/tables/retention_custom_auth_int4.tex
```

Purpose: generate the seed-1 efficiency LaTeX table.

```bash
python scripts/generate_efficiency_table.py --rows outputs/efficiency/qwen25_3b__direct_policy__seed1.json outputs/efficiency/qwen25_3b__plain_efsc__seed1.json outputs/efficiency/qwen25_3b__qefsc_fc__seed1.json --labels "Direct Policy" "Plain EFSC" "QEFSC-FC" --output_tex outputs/tables/efficiency_qwen_seed1.tex
```

Purpose: plot FP safety utility against safety-utility drop after INT4 quantization.

```bash
python scripts/plot_quant_tradeoff.py --fp_metrics outputs/aggregates/qwen25_3b__direct_policy__fp__custom_auth.json outputs/aggregates/qwen25_3b__plain_efsc__fp__custom_auth.json outputs/aggregates/qwen25_3b__qefsc_fc__fp__custom_auth.json --quant_metrics outputs/aggregates/qwen25_3b__direct_policy__int4__custom_auth.json outputs/aggregates/qwen25_3b__plain_efsc__int4__custom_auth.json outputs/aggregates/qwen25_3b__qefsc_fc__int4__custom_auth.json --labels "Direct Policy" "Plain EFSC" "QEFSC-FC" --output_png outputs/plots/qwen_quant_tradeoff.png
```

Purpose: plot quantized safety retention against quantized structure retention.

```bash
python scripts/plot_retention_frontier.py --retention_files outputs/aggregates/qwen25_3b__direct_policy__int4_retention__custom_auth.json outputs/aggregates/qwen25_3b__plain_efsc__int4_retention__custom_auth.json outputs/aggregates/qwen25_3b__qefsc_fc__int4_after_retention__custom_auth.json --labels "Direct Policy" "Plain EFSC" "QEFSC-FC" --output_png outputs/plots/qwen_retention_frontier.png
```

## Artifact Names

The runners write consistent names:

```text
outputs/models/<run_id>/best.pt
outputs/preds/<run_id>__fp__<dataset>.jsonl
outputs/preds/<run_id>__int4__<dataset>.jsonl
outputs/preds/<run_id>__int4_recal__<dataset>.jsonl
outputs/metrics/<run_id>__fp__<dataset>.json
outputs/metrics/<run_id>__int4__<dataset>.json
outputs/metrics/<run_id>__int4_recal__<dataset>.json
outputs/retention/<run_id>__int4__<dataset>.json
outputs/retention/<run_id>__int4__before__<dataset>.json
outputs/retention/<run_id>__int4__after__<dataset>.json
outputs/efficiency/<run_id>.json
```

For baselines, `int4_recal` artifacts are produced only when `run_baseline_quant_experiment.py` is called with `--recalibrate`.

## Minimal Paper-Ready Run Set

Run Qwen for all three methods with seeds `1`, `2`, and `3` on the full frozen test set:

- `custom_auth`
- `xstest`
- `orbench_hard`
- `orbench_toxic`
- `strongreject`

Llama seed 1 is the first scale-validation run on the same dataset set; Llama seeds 2 and 3 are useful if compute allows.

## What To Inspect First

1. `safety_utility_score`: QEFSC-FC should beat both baselines.
2. `quantized_safety_retention`: QEFSC-FC should lose less under INT4.
3. `quantized_structure_retention`: QEFSC-FC should preserve counterfactual behavior.
4. Recalibration gain: QEFSC-FC `after` should improve over `before`.
5. Efficiency: trainable parameter count and peak GPU memory should support the deployment-constrained claim.
