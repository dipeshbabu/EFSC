# EFSC Error Analysis Template

Use this template after each main evaluation run.

| id | source | topic | family_id | prompt_summary | gold_action | pred_action | error_type | likely_cause | fix_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Error types:

- false refusal
- unsafe answer
- missed clarification
- over-cautious safe answer
- wrong factor attribution
- multilingual instability
- family inconsistency

For each benchmark, sample at least 25 errors or all errors if fewer than 25.
