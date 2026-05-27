# EFSC Reproducibility Checklist

Before training:

- `docs/data_manifest_and_labeling_spec.md` exists.
- Public dataset snapshot date is recorded in `docs/data_manifest.md`.
- `data/processed/train.jsonl`, `val.jsonl`, and all test JSONL files exist.
- `data/manifests/source_manifest.json`, `family_manifest.json`, `split_manifest.json`, and `label_distribution.json` exist.
- `scripts/acceptance_checks.py` passes on all processed files.
- No public test source appears in train or validation.
- No family crosses train, validation, and test.
- `scripts/make_full_maintrack_run_grid.py` has generated `runs/full_maintrack_grid.sh`.

Before reporting:

- Run at least 3 seeds for `qefsc_fc`, `plain_efsc`, and `direct_policy`.
- Save best and last checkpoints.
- Save prediction JSONL, metric JSON, retention JSON, and efficiency JSON files.
- Run `scripts/check_maintrack_artifacts.py` to detect missing outputs.
- Run `scripts/aggregate_seed_metrics.py` and `scripts/aggregate_retention_metrics.py`.
- Run `scripts/statistical_significance.py` for primary method comparisons.
- Run `scripts/qualitative_error_analysis.py` for appendix examples.
- Generate LaTeX tables and plots using the commands in `EXPERIMENTS.md`.
- Run `scripts/build_workshop_readiness_report.py --fail_on_not_ready` before making final COLM workshop claims.
- Keep the COLM manuscript and template assets under local ignored `paper/Template-2026/` unless paper contents are explicitly requested for version control.
