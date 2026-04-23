# EFSC Annotation Protocol

Custom data must be annotated with the full EFSC factor/action schema before training.

Process:

1. Annotate `harm_label`, `legit_label`, `uncertainty_label`, and `action_label`.
2. Apply the precedence rules in `labeling_guidelines.md`.
3. Double annotate at least 15 percent of the custom set.
4. Adjudicate all disagreements using the prompt plus family context.
5. Record disagreements in `annotation_audit.csv` with one of: `harm`, `legitimacy`, `uncertainty`, `action`, or `multiple`.

Required audit columns:

```text
id,annotator_a,annotator_b,field,original_a,original_b,adjudicated_label,notes
```

Training may start only after the custom data acceptance report passes and the annotation audit exists for the manually reviewed subset.
