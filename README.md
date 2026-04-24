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
  architecture, data manifest, labeling, reproducibility

efsc/
  dataset preparation helpers retained for compatibility with preprocessing scripts
```

The active experiment entrypoints are the root-level QEFSC and baseline runners. The original factorized EFSC formulation is retained as the `plain_efsc` baseline rather than as the main proposed method.

## Key Docs

- Architecture: [docs/architecture_diagram.md](docs/architecture_diagram.md)
- Full runbook: [EXPERIMENTS.md](EXPERIMENTS.md)
- Data manifest: [docs/data_manifest.md](docs/data_manifest.md)
- Labeling guide: [docs/labeling_guidelines.md](docs/labeling_guidelines.md)

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
```

The current workspace has `test_custom_auth.jsonl`. If `test_xstest.jsonl` is missing, `scripts/make_full_maintrack_run_grid.py` skips it by default.

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

Purpose: generate the full active command grid for available datasets.

```bash
python scripts/make_full_maintrack_run_grid.py --output_script runs/full_maintrack_grid.sh
```

Purpose: check which expected outputs from the grid are still missing.

```bash
python scripts/check_maintrack_artifacts.py --grid_script runs/full_maintrack_grid.sh --report outputs/aggregates/artifact_check.json
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for aggregation, significance testing, qualitative analysis, tables, and plots.
