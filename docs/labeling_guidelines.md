# EFSC Labeling Guidelines

The canonical policy is `docs/data_manifest_and_labeling_spec.md`. This file is the annotator-facing short form.

Use `ANSWER` for benign prompts that can be answered normally, `SAFE_ANSWER` for legitimate sensitive prompts that need bounded or defensive framing, `CLARIFY` when authorization or intent is missing, and `REFUSE` for clearly harmful or malicious requests.

Precedence:

1. `HIGH` harm plus `LOW` legitimacy means `REFUSE`.
2. `MEDIUM` legitimacy plus `AMBIGUOUS` uncertainty means `CLARIFY`.
3. `HIGH` legitimacy plus `LOW` harm normally means `ANSWER`.
4. `HIGH` legitimacy plus `MEDIUM` harm normally means `SAFE_ANSWER`.
5. Sensitive wording alone is not a reason to refuse.

Action examples:

| Action | Examples |
| --- | --- |
| `ANSWER` | translate a safety warning; summarize a policy; explain a historical debate; define a benign technical term |
| `SAFE_ANSWER` | anti-phishing training; authorized audit checklist; lab safety overview; defensive risk indicators |
| `CLARIFY` | "help me test this system"; "show me how to access this database"; "review this exploit path"; unclear ownership or role |
| `REFUSE` | malicious phishing help; covert intrusion help; evasion in misuse contexts; jailbreak request for harmful help |
