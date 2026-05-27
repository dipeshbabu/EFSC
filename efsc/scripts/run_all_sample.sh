#!/usr/bin/env bash
set -e
python -m efsc.train.stage1_supervised --config efsc/configs/efsc_stage1.yaml
python -m efsc.train.stage2_counterfactual --config efsc/configs/efsc_stage2.yaml
python -m efsc.train.stage3_dpo --config efsc/configs/efsc_stage3.yaml
python -m efsc.eval.run_eval --config efsc/configs/eval_main.yaml
python -m efsc.baselines.prompting_baseline --config efsc/configs/baseline_prompting.yaml
python -m efsc.baselines.direct_classifier --config efsc/configs/baseline_direct_classifier.yaml
