from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_FIELDS = {
    "id",
    "source",
    "source_split",
    "split",
    "family_id",
    "prompt",
    "language",
    "topic",
    "variant_type",
    "harm_label",
    "legit_label",
    "uncertainty_label",
    "action_label",
    "preferred_over",
    "dispreferred_over",
    "metadata",
}

ALLOWED_SOURCES = {
    "wildjailbreak",
    "falsereject",
    "xstest",
    "orbench_hard",
    "orbench_toxic",
    "strongreject",
    "custom_auth",
}
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_TOPICS = {"cyber", "bio", "fraud", "privacy", "mental_health", "chemistry", "general", "other"}
ALLOWED_VARIANTS = {
    "harmful_malicious",
    "jailbreak",
    "defensive_benign",
    "educational_benign",
    "authorized_benign",
    "ambiguous_legitimate",
    "hard_benign",
    "toxic_harmful",
}
ALLOWED_HARM = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_LEGIT = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_UNCERTAINTY = {"CLEAR", "AMBIGUOUS"}
ALLOWED_ACTIONS = {"ANSWER", "SAFE_ANSWER", "CLARIFY", "REFUSE"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {exc}") from exc
    return rows


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def check_required_fields(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"row[{i}] missing fields: {sorted(missing)}")


def check_values(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    for i, row in enumerate(rows):
        if row["source"] not in ALLOWED_SOURCES:
            errors.append(f"row[{i}] invalid source: {row['source']}")
        if row["split"] not in ALLOWED_SPLITS:
            errors.append(f"row[{i}] invalid split: {row['split']}")
        if row["topic"] not in ALLOWED_TOPICS:
            errors.append(f"row[{i}] invalid topic: {row['topic']}")
        if row["variant_type"] not in ALLOWED_VARIANTS:
            errors.append(f"row[{i}] invalid variant_type: {row['variant_type']}")
        if row["harm_label"] not in ALLOWED_HARM:
            errors.append(f"row[{i}] invalid harm_label: {row['harm_label']}")
        if row["legit_label"] not in ALLOWED_LEGIT:
            errors.append(f"row[{i}] invalid legit_label: {row['legit_label']}")
        if row["uncertainty_label"] not in ALLOWED_UNCERTAINTY:
            errors.append(f"row[{i}] invalid uncertainty_label: {row['uncertainty_label']}")
        if row["action_label"] not in ALLOWED_ACTIONS:
            errors.append(f"row[{i}] invalid action_label: {row['action_label']}")
        if not isinstance(row["preferred_over"], list):
            errors.append(f"row[{i}] preferred_over must be a list")
        if not isinstance(row["dispreferred_over"], list):
            errors.append(f"row[{i}] dispreferred_over must be a list")
        if not isinstance(row["metadata"], dict):
            errors.append(f"row[{i}] metadata must be a dict")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            errors.append(f"row[{i}] prompt must be a nonempty string")


def check_duplicate_ids(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    counts = Counter(row["id"] for row in rows)
    dups = [key for key, value in counts.items() if value > 1]
    for dup in dups[:20]:
        errors.append(f"duplicate id found: {dup}")
    if len(dups) > 20:
        errors.append(f"... and {len(dups) - 20} more duplicate ids")


def check_action_precedence(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    for i, row in enumerate(rows):
        harm = row["harm_label"]
        legit = row["legit_label"]
        uncertainty = row["uncertainty_label"]
        action = row["action_label"]
        if harm == "HIGH" and legit == "LOW" and action != "REFUSE":
            errors.append(f"row[{i}] expected REFUSE for HIGH harm and LOW legit")
        if legit == "MEDIUM" and uncertainty == "AMBIGUOUS" and action != "CLARIFY":
            errors.append(f"row[{i}] expected CLARIFY for MEDIUM legit and AMBIGUOUS uncertainty")
        if legit == "HIGH" and harm == "LOW" and action not in {"ANSWER", "SAFE_ANSWER"}:
            errors.append(f"row[{i}] expected ANSWER or SAFE_ANSWER for HIGH legit and LOW harm")
        if legit == "HIGH" and harm == "MEDIUM" and action not in {"SAFE_ANSWER", "CLARIFY"}:
            errors.append(f"row[{i}] expected SAFE_ANSWER or CLARIFY for HIGH legit and MEDIUM harm")


def check_split_source_rules(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    test_only = {"xstest", "orbench_hard", "orbench_toxic", "strongreject"}
    for i, row in enumerate(rows):
        if row["source"] in test_only and row["split"] != "test":
            errors.append(f"row[{i}] test-only source {row['source']} appears in split={row['split']}")
        if row["source"] in {"wildjailbreak", "falsereject"} and row["split"] == "test":
            errors.append(f"row[{i}] training source {row['source']} should not appear in test")


def check_family_consistency(rows: List[Dict[str, Any]], errors: List[str]) -> None:
    fam_to_splits: Dict[str, set[str]] = defaultdict(set)
    fam_to_topics: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_id = row.get("family_id", "")
        if family_id:
            fam_to_splits[family_id].add(row["split"])
            fam_to_topics[family_id].add(row["topic"])
    for fam, splits in fam_to_splits.items():
        if len(splits) > 1:
            errors.append(f"family leakage: {fam} appears in multiple splits: {sorted(splits)}")
    for fam, topics in fam_to_topics.items():
        if len(topics) > 3:
            errors.append(f"suspicious family topic spread: {fam} has topics {sorted(topics)}")


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_rows": len(rows),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_topic": dict(Counter(row["topic"] for row in rows)),
        "by_variant_type": dict(Counter(row["variant_type"] for row in rows)),
        "by_harm_label": dict(Counter(row["harm_label"] for row in rows)),
        "by_legit_label": dict(Counter(row["legit_label"] for row in rows)),
        "by_uncertainty_label": dict(Counter(row["uncertainty_label"] for row in rows)),
        "by_action_label": dict(Counter(row["action_label"] for row in rows)),
        "num_families": len({row["family_id"] for row in rows if row["family_id"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more normalized JSONL files")
    parser.add_argument("--report_output", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(load_jsonl(Path(path)))

    errors: List[str] = []
    check_required_fields(rows, errors)
    if not errors:
        check_values(rows, errors)
        check_duplicate_ids(rows, errors)
        check_action_precedence(rows, errors)
        check_split_source_rules(rows, errors)
        check_family_consistency(rows, errors)

    report = {"ok": len(errors) == 0, "num_errors": len(errors), "errors": errors, "summary": build_summary(rows)}
    save_json(report, Path(args.report_output))
    if errors:
        print(f"Validation failed with {len(errors)} errors")
        for error in errors[:50]:
            print("ERROR:", error)
        sys.exit(1)
    print("Validation passed")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
