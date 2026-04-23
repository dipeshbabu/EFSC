# EFSC Data Manifest

Snapshot date: 2026-04-22.

This project uses the fixed NeurIPS main-track data scope defined in `docs/data_manifest_and_labeling_spec.md`.

| Source | HF dataset / local path | License | Usage | Notes |
| --- | --- | --- | --- | --- |
| WildJailbreak | `allenai/wildjailbreak` | ODC-BY, gated responsible-use acceptance | train, val family split | Harmful, jailbreak, and benign contrastive prompts. |
| FalseReject | `AmazonScience/FalseReject` | CC-BY-NC-4.0 | train, val family split | Benign but risky-looking prompts for over-refusal reduction. |
| Custom Authorization and Ambiguity | `data/raw/custom_auth/custom_auth.jsonl` | project-local research data | train, val, test | Generated seed set covering `SAFE_ANSWER` and `CLARIFY`. |
| XSTest | `walledai/XSTest` | CC-BY-4.0 for prompts | test only | Exaggerated safety and benign over-refusal. |
| OR Bench Hard 1K | `bench-llms/or-bench`, subset `or-bench-hard-1k` | CC-BY-4.0 | test only | Hard benign over-refusal frontier. |
| OR Bench Toxic 600 | `bench-llms/or-bench`, subset `or-bench-toxic` | CC-BY-4.0 | test only | Harmful side of refusal frontier. |
| StrongREJECT | `AlignmentResearch/StrongREJECT`, fallback `Machlovi/strongreject-dataset` | dataset card dependent | test only | Harmful compliance and under-refusal evaluation. |

The benchmark list is frozen for main experiments. OKTest and MultiBreak are appendix-only candidates and must not enter the main train, validation, or test tables unless this manifest is revised.
