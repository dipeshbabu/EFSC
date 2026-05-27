# EFSC

Efficient Factorized Safety Calibration for deployment-constrained language models.

The current main-track system is **QEFSC-FC**: Quantization Robust Efficient Factorized Safety Control with Feature Conditioning. It trains lightweight safety controllers on decoder-only LMs, evaluates full precision and low-bit deployment, and compares against matched `plain_efsc` and `direct_policy` baselines.

## Repository Layout

```text
src/
  data/       metadata-preserving JSONL loader
  models/     QEFSC-FC, decoder EFSC, plain EFSC, direct-policy baselines
  quant/      FP16/BF16/INT8/INT4 backbone loading
  train/      losses, LoRA helpers, tokenizer setup, efficiency tracking
  eval/       main-track safety and counterfactual metrics

scripts/
  data prep, evaluation, retention, statistics, tables, plots, artifact checks

docs/
  architecture, data manifest, labeling, reproducibility, result templates

efsc/
  dataset preparation helpers retained for compatibility with preprocessing scripts
```

The active experiment entrypoints are the root-level QEFSC and baseline runners. The original factorized EFSC formulation is retained as the `plain_efsc` baseline rather than as the main proposed method.
The workshop manuscript lives locally under `paper/Template-2026/` and is intentionally ignored by Git. Keep tracked code, data manifests, runbooks, and reproducibility checks in the repository; keep submission drafts and template assets local unless explicitly requested.

## Key Docs

- Architecture: [docs/architecture_diagram.md](docs/architecture_diagram.md)
- Full runbook: [EXPERIMENTS.md](EXPERIMENTS.md)
- Data manifest: [docs/data_manifest.md](docs/data_manifest.md)
- Labeling guide: [docs/labeling_guidelines.md](docs/labeling_guidelines.md)
- Reproducibility checklist: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- Final repo checklist: [docs/final_repo_checklist.md](docs/final_repo_checklist.md)

## Setup

Use `uv` for a clean, reproducible Linux/macOS/WSL environment. The CUDA extra installs `bitsandbytes`, which is required for INT8/INT4 experiments on CUDA-capable Linux/WSL systems.

Purpose: create an isolated Python environment for the repo.

```bash
uv venv --python 3.11
```

Purpose: activate the environment.

```bash
source .venv/bin/activate
```

Purpose: install EFSC in editable mode with test tools into the repo venv explicitly.

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

Purpose: install the CUDA/quantization extra into the same repo venv when running INT8/INT4 experiments.

```bash
uv pip install --python .venv/bin/python -e ".[dev,cuda]"
```

Purpose: verify the active interpreter and core packages before running experiments.

```bash
which python
python -c "import torch, peft, pytest; print('core ok')"
```

Purpose: verify imports and syntax for the active source and script tree.

```bash
python -m compileall src scripts efsc
```

Purpose: run the fast local unit tests.

```bash
python -m pytest tests
```

## Expected Data

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test_custom_auth.jsonl
data/processed/test_xstest.jsonl
data/processed/test_orbench_hard.jsonl
data/processed/test_orbench_toxic.jsonl
data/processed/test_strongreject.jsonl
```

Purpose: build the full frozen training/validation/test set into `data/processed` from the public benchmark sources plus the local custom authorization set.

```bash
python -m efsc.data.prepare_all --output-dir data/processed --custom-auth data/raw/custom_auth/custom_auth.jsonl
```

The current workspace has `test_custom_auth.jsonl`. `scripts/make_full_maintrack_run_grid.py` now targets the full frozen test set by default. Run the data-prep command above first so every expected test file exists. Use `--skip_missing` only for smoke runs.

## First Experiment Commands

Run these in a CUDA-enabled environment with model access.

Purpose: train/evaluate the proposed QEFSC-FC method on Qwen, run INT4 inference, and recalibrate the controller.

```bash
python run_qefsc_quant_experiment.py --run_id qwen25_3b__qefsc_fc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --variant qefsc_fc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: train/evaluate the final-hidden-state factorized baseline under the same FP and INT4 protocol.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__plain_efsc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type plain_efsc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: train/evaluate the direct action classifier baseline under the same FP and INT4 protocol.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__direct_policy__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type direct_policy --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora
```

Purpose: optionally add post-quantization recalibration for a baseline when you want the fairest quantized comparison.

```bash
python run_baseline_quant_experiment.py --run_id qwen25_3b__plain_efsc__seed1 --model_name "Qwen/Qwen2.5-3B-Instruct" --baseline_type plain_efsc --seed 1 --test_path data/processed/test_custom_auth.jsonl --dataset_name custom_auth --quant_mode int4 --use_lora --recalibrate
```

Purpose: generate the full active command grid for the frozen test set in `docs/data_manifest.md`. Use `--skip_missing` only for smoke runs.

```bash
python scripts/make_full_maintrack_run_grid.py --output_script runs/full_maintrack_grid.sh
```

Purpose: check which expected outputs from the grid are still missing.

```bash
python scripts/check_maintrack_artifacts.py --grid_script runs/full_maintrack_grid.sh --report outputs/aggregates/artifact_check.json
```

Purpose: run the workshop readiness gate before writing or submitting final claims.

```bash
python scripts/build_workshop_readiness_report.py --grid_script runs/full_maintrack_grid.sh --output_root outputs --paper_tex paper/Template-2026/colm2026_conference.tex --output_json outputs/reviewer_readiness/workshop_readiness.json --output_md outputs/reviewer_readiness/workshop_readiness.md --fail_on_not_ready
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for aggregation, significance testing, qualitative analysis, tables, and plots.
