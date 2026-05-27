# EFSC Final Repo Checklist

## Data and Experiment Setup

- `scripts/validate_data.py` validates required fields, labels, split rules, family leakage, and action precedence.
- `scripts/make_experiment_registry.py` writes the fixed 90-run registry to `runs/experiment_registry.csv`.
- `src/data/dataset_loader.py` provides a training-time JSONL dataset for normalized EFSC rows.
- `src/eval/metrics.py` computes the core paper metrics from prediction rows.
- `scripts/evaluate_predictions.py` converts prediction JSONL into metrics JSON.
- `scripts/error_analysis.py` builds paper-ready failure buckets and example cases.
- `EXPERIMENTS.md` defines the fixed run order and evaluation commands.

## Still Required For Full Paper Runs

- Configure Hugging Face access for WildJailbreak and other public datasets.
- Run the full public dataset normalization pipeline.
- Train EFSC stage 1, stage 2, and stage 3 on the finalized data using either the package trainers or the top-level `train_stage*.py` scripts.
- Train direct classifier and LoRA baselines.
- Add or finalize ablation model switches for no-harm, no-legitimacy, and no-uncertainty heads.
- Run final 3-seed evaluations and aggregate result tables.
- Complete manual annotation QA for the custom authorization and ambiguity set.

## Newly Added End-to-End Scaffold

- `src/models/efsc_model.py`
- `src/train/losses.py`
- `src/train/utils.py`
- `src/train/collators.py`
- `train_stage1.py`
- `train_stage2.py`
- `train_stage3.py`
- `train_direct_classifier.py`
- `run_inference.py`
- `scripts/aggregate_results.py`
- `src/train/peft_utils.py`
- `src/models/model_factory.py`
- `src/train/preference_dataset.py`
- `src/train/dpo_loss.py`
- `train_stage1_lora.py`
- `train_stage3_dpo.py`
- `run_inference_lora.py`

## Medium-Model Readiness

- EFSC can wrap the backbone with LoRA through `build_efsc_model(..., use_lora=True)`.
- The direct classifier baseline can also use LoRA via `train_direct_classifier.py --use_lora`.
- Stage 3 DPO uses a trainable policy model and frozen reference model.
- `scripts/prepare_all.py` exports both `train.pref.jsonl` and `val.pref.jsonl`.

## Paper Result Generation

- `scripts/update_experiment_registry.py` updates run status in `runs/experiment_registry.csv`.
- `scripts/launch_multiseed.py` writes reproducible shell commands for planned runs.
- `scripts/merge_seed_metrics.py` aggregates metric JSONs across seeds.
- `scripts/generate_latex_table.py` creates copy-pasteable LaTeX tables.
- `scripts/plot_results.py` creates paper figures from merged metrics.
- `scripts/select_best_checkpoint.py` selects best metric files by validation metric.
- `scripts/build_final_paper_bundle.py` collects tables, figures, and analysis JSONs into `outputs/paper_bundle`.
- `runs/run_all.sh` can be regenerated from the registry with `scripts/launch_multiseed.py`.

## Main-Track Evaluation And Ablations

- `src/eval/maintrack_metrics.py` computes safety utility, ambiguity, authorization, and counterfactual metrics.
- `scripts/evaluate_maintrack.py` evaluates prediction JSONL files with the main-track metric suite.
- `scripts/evaluate_subsets.py` computes metrics for harmful, benign, ambiguous, topic, and variant slices.
- `scripts/evaluate_ood_generalization.py` compares ID and OOD metric files.
- `scripts/log_efficiency.py` records efficiency metrics for table generation.
- `scripts/generate_maintrack_tables.py` writes main, ambiguity, and consistency LaTeX tables.
- `scripts/generate_efficiency_table.py` writes the efficiency LaTeX table.
- `src/models/efsc_ablation_model.py` implements actual factor/bottleneck ablation variants.
- `train_ablation.py` trains the ablation variants.
- `scripts/run_ablation_suite.py` writes ablation command sheets.
- `run_one_experiment.py` runs train, inference, evaluation, subsets, and efficiency stub logging for one run.
- `scripts/bootstrap_ci.py` computes bootstrap confidence intervals.
- `scripts/plot_pareto_frontier.py` and `scripts/plot_ablation_bars.py` create main-track plots.
- `scripts/build_results_pack.py` bundles metrics, subsets, efficiency logs, tables, and figures.
