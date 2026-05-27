from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from efsc.data.dataset import validate_prompt_example
from efsc.utils import ensure_dir, load_jsonl, save_json, save_jsonl

DEFAULT_TRAIN_SOURCES = ("wildjailbreak", "falsereject")
DEFAULT_TEST_SOURCES = ("xstest", "orbench_hard", "orbench_toxic", "strongreject")

TOPIC_KEYWORDS = {
    "cyber": ("phishing", "malware", "password", "server", "exploit", "port", "sql", "ddos", "hack", "network"),
    "bio": ("virus", "pathogen", "bacteria", "bio"),
    "fraud": ("bank", "credit card", "scam", "fraud", "phishing", "identity theft"),
    "privacy": ("dox", "private", "address", "phone number", "personal data"),
    "mental_health": ("stress", "anxiety", "depression", "mental health", "suicide", "self-harm", "self harm"),
    "chemistry": ("chemical", "chemistry", "corrosive", "solvent", "compound", "lab", "toxin"),
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "being",
    "could",
    "give",
    "help",
    "make",
    "should",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def clean_topic(value: Any, prompt: str = "") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text and text not in {"nan", "none", "unknown", "harmful", "safe", "unsafe"}:
        if any(token in text for token in ("cyber", "phishing", "malware", "exploit", "network")):
            return "cyber"
        if any(token in text for token in ("bio", "pathogen", "virus")):
            return "bio"
        if any(token in text for token in ("fraud", "deception", "scam")):
            return "fraud"
        if any(token in text for token in ("privacy", "dox", "personal")):
            return "privacy"
        if any(token in text for token in ("mental", "self_harm", "suicide", "stress")):
            return "mental_health"
        if any(token in text for token in ("chem", "lab", "toxin")):
            return "chemistry"
        return "other"
    prompt_lower = prompt.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return topic
    return "general"


def extract_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row and row["prompt"]:
        return str(row["prompt"]).strip()
    if "messages" in row and row["messages"]:
        messages = row["messages"]
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                return messages.strip()
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    return str(message.get("content", "")).strip()
            if messages and isinstance(messages[-1], dict):
                return str(messages[-1].get("content", "")).strip()
    if "vanilla" in row or "adversarial" in row:
        adversarial = str(row.get("adversarial") or "").strip()
        vanilla = str(row.get("vanilla") or "").strip()
        return adversarial or vanilla
    raise ValueError(f"Cannot find prompt text in row keys={sorted(row.keys())}")


def preference_lists(action_label: str) -> Tuple[List[str], List[str]]:
    if action_label == "ANSWER":
        return ["ANSWER"], ["REFUSE"]
    if action_label == "SAFE_ANSWER":
        return ["SAFE_ANSWER"], ["REFUSE"]
    if action_label == "CLARIFY":
        return ["CLARIFY"], ["REFUSE", "ANSWER"]
    if action_label == "REFUSE":
        return ["REFUSE"], ["ANSWER", "SAFE_ANSWER", "CLARIFY"]
    raise ValueError(f"Unknown action label: {action_label}")


def make_example(
    *,
    source: str,
    split: str,
    prompt: str,
    variant_type: str,
    action_label: str,
    harm_label: str,
    legit_label: str,
    uncertainty_label: str,
    topic: str,
    source_split: str = "unknown",
    language: str = "en",
    metadata: Optional[Dict[str, Any]] = None,
    row_id: Optional[str] = None,
    family_id: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"Empty prompt for source={source}")
    preferred, dispreferred = preference_lists(action_label)
    ex_id = row_id or f"{source}_{stable_hash(prompt)}"
    row = {
        "id": ex_id,
        "source": source,
        "source_split": source_split,
        "split": split,
        "family_id": family_id or f"{source}_pending_{stable_hash(prompt, 10)}",
        "prompt": prompt,
        "language": language,
        "topic": clean_topic(topic, prompt),
        "variant_type": variant_type,
        "harm_label": harm_label,
        "legit_label": legit_label,
        "uncertainty_label": uncertainty_label,
        "action_label": action_label,
        "preferred_over": preferred,
        "dispreferred_over": dispreferred,
        "metadata": metadata or {},
    }
    validate_prompt_example(row)
    return row


def infer_false_reject_variant(prompt: str, category_text: str) -> Tuple[str, str, str, str]:
    text = f"{prompt} {category_text}".lower()
    if any(token in text for token in ("authorized", "permission", "my company", "my lab", "my network")):
        return "authorized_benign", "SAFE_ANSWER", "MEDIUM", "CLEAR"
    if any(token in text for token in ("whether", "should i", "is it okay", "can i legally")):
        return "ambiguous_legitimate", "CLARIFY", "LOW", "AMBIGUOUS"
    return "hard_benign", "SAFE_ANSWER", "LOW", "CLEAR"


def convert_wildjailbreak(rows: Iterable[Dict[str, Any]], split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        data_type = str(row.get("data_type", "")).lower()
        prompt = extract_prompt(row)
        topic = clean_topic(row.get("category") or row.get("risk_category"), prompt)
        metadata = {
            "data_type": data_type,
            "tactics": row.get("tactics", []),
            "has_adversarial_prompt": bool(str(row.get("adversarial") or "").strip()),
        }
        if data_type in {"vanilla_harmful", "adversarial_harmful"}:
            variant = "jailbreak" if data_type.startswith("adversarial") else "harmful_malicious"
            converted.append(make_example(
                source="wildjailbreak",
                split=split,
                source_split="train",
                prompt=prompt,
                variant_type=variant,
                action_label="REFUSE",
                harm_label="HIGH",
                legit_label="LOW",
                uncertainty_label="CLEAR",
                topic=topic,
                metadata=metadata,
                row_id=f"wildjailbreak_{split}_{index}",
            ))
        elif data_type in {"vanilla_benign", "adversarial_benign"}:
            action = "SAFE_ANSWER" if data_type.startswith("adversarial") else "ANSWER"
            converted.append(make_example(
                source="wildjailbreak",
                split=split,
                source_split="train",
                prompt=prompt,
                variant_type="educational_benign",
                action_label=action,
                harm_label="LOW",
                legit_label="HIGH",
                uncertainty_label="CLEAR",
                topic=topic,
                metadata=metadata,
                row_id=f"wildjailbreak_{split}_{index}",
            ))
        if limit and len(converted) >= limit:
            break
    return converted


def convert_falsereject(rows: Iterable[Dict[str, Any]], split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        prompt = extract_prompt(row)
        category_text = str(row.get("category_text") or row.get("category") or "")
        variant, action, harm, uncertainty = infer_false_reject_variant(prompt, category_text)
        converted.append(make_example(
            source="falsereject",
            split=split,
            source_split="train",
            prompt=prompt,
            variant_type=variant,
            action_label=action,
            harm_label=harm,
            legit_label="HIGH" if uncertainty == "CLEAR" else "MEDIUM",
            uncertainty_label=uncertainty,
            topic=category_text,
            metadata={"category": row.get("category"), "category_text": category_text},
            row_id=f"falsereject_{split}_{index}",
        ))
        if limit and len(converted) >= limit:
            break
    return converted


def convert_xstest(rows: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        prompt = extract_prompt(row)
        topic = row.get("note") or row.get("type") or row.get("focus") or "xstest"
        converted.append(make_example(
            source="xstest",
            split="test",
            source_split="test",
            prompt=prompt,
            variant_type="hard_benign",
            action_label="ANSWER",
            harm_label="LOW",
            legit_label="HIGH",
            uncertainty_label="CLEAR",
            topic=topic,
            metadata={"type": row.get("type"), "focus": row.get("focus"), "label": row.get("label")},
            row_id=f"xstest_{index}",
            family_id=f"xstest_{clean_topic(topic, prompt)}_{index}",
        ))
        if limit and len(converted) >= limit:
            break
    return converted


def convert_orbench(rows: Iterable[Dict[str, Any]], source: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    converted = []
    toxic = source == "orbench_toxic"
    for index, row in enumerate(rows):
        prompt = extract_prompt(row)
        category = row.get("category") or ("harmful" if toxic else "orbench")
        converted.append(make_example(
            source=source,
            split="test",
            source_split="test",
            prompt=prompt,
            variant_type="toxic_harmful" if toxic else "hard_benign",
            action_label="REFUSE" if toxic else "SAFE_ANSWER",
            harm_label="HIGH" if toxic else "LOW",
            legit_label="LOW" if toxic else "HIGH",
            uncertainty_label="CLEAR",
            topic=category,
            metadata={"category": category},
            row_id=f"{source}_{index}",
            family_id=f"{source}_{clean_topic(category, prompt)}_{index}",
        ))
        if limit and len(converted) >= limit:
            break
    return converted


def convert_strongreject(rows: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        prompt = extract_prompt(row)
        category = row.get("category") or row.get("source") or "strongreject"
        converted.append(make_example(
            source="strongreject",
            split="test",
            source_split="test",
            prompt=prompt,
            variant_type="harmful_malicious",
            action_label="REFUSE",
            harm_label="HIGH",
            legit_label="LOW",
            uncertainty_label="CLEAR",
            topic=category,
            metadata={"category": row.get("category"), "source": row.get("source"), "prompt_label": row.get("prompt_label")},
            row_id=f"strongreject_{index}",
            family_id=f"strongreject_{clean_topic(category, prompt)}_{index}",
        ))
        if limit and len(converted) >= limit:
            break
    return converted


def convert_custom_auth(path: Path) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    converted = []
    for index, row in enumerate(rows):
        if "action_label" in row:
            normalized = dict(row)
            normalized.setdefault("source", "custom_auth")
            normalized.setdefault("source_split", "unknown")
            normalized.setdefault("split", "train")
            normalized.setdefault("family_id", f"custom_auth_{stable_hash(normalized['prompt'], 10)}")
            normalized.setdefault("preferred_over", preference_lists(normalized["action_label"])[0])
            normalized.setdefault("dispreferred_over", preference_lists(normalized["action_label"])[1])
            normalized.setdefault("metadata", {})
            validate_prompt_example(normalized)
            converted.append(normalized)
            continue
        prompt = extract_prompt(row)
        variant = row.get("variant_type", "authorized_benign")
        if variant == "ambiguous_legitimate":
            action, harm, legit, uncertainty = "CLARIFY", "LOW", "MEDIUM", "AMBIGUOUS"
        else:
            action, harm, legit, uncertainty = "SAFE_ANSWER", "MEDIUM", "HIGH", "CLEAR"
        converted.append(make_example(
            source="custom_auth",
            split=str(row.get("split", "train")),
            source_split=str(row.get("source_split", "unknown")),
            prompt=prompt,
            variant_type=variant,
            action_label=action,
            harm_label=harm,
            legit_label=legit,
            uncertainty_label=uncertainty,
            topic=str(row.get("topic", "authorization")),
            language=str(row.get("language", "en")),
            metadata={k: v for k, v in row.items() if k not in {"prompt", "variant_type", "topic", "language", "split"}},
            row_id=str(row.get("id", f"custom_auth_{index}")),
        ))
    return converted


def salient_signature(prompt: str) -> str:
    tokens = re.findall(r"[a-z0-9]{4,}", prompt.lower())
    tokens = [token for token in tokens if token not in STOPWORDS]
    return "_".join(tokens[:4]) or stable_hash(prompt, 8)


def assign_counterfactual_families(rows: List[Dict[str, Any]], family_size: int = 6) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row["split"] not in {"train", "val", "train_pool"}:
            continue
        grouped.setdefault((row["source"], row["topic"]), []).append(row)

    for (source, topic), items in grouped.items():
        harmful = [row for row in items if row["harm_label"] == "HIGH"]
        benign = [row for row in items if row["harm_label"] != "HIGH"]
        random.Random(stable_hash(f"{source}:{topic}")).shuffle(harmful)
        random.Random(stable_hash(f"{topic}:{source}")).shuffle(benign)
        family_index = 0
        while harmful or benign:
            family_rows: List[Dict[str, Any]] = []
            if harmful:
                family_rows.append(harmful.pop())
            while benign and len(family_rows) < family_size:
                family_rows.append(benign.pop())
            while harmful and len(family_rows) < family_size and not any(row["harm_label"] == "HIGH" for row in family_rows):
                family_rows.append(harmful.pop())
            if not family_rows:
                break
            signature = salient_signature(family_rows[0]["prompt"])
            family_id = f"{source}_{topic}_fam{family_index:05d}_{stable_hash(signature, 6)}"
            sibling_ids = [row["id"] for row in family_rows]
            for row in family_rows:
                row["family_id"] = family_id
                row["sibling_ids"] = [sibling_id for sibling_id in sibling_ids if sibling_id != row["id"]]
            family_index += 1


def split_train_val_by_family(rows: List[Dict[str, Any]], val_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    family_ids = sorted({row["family_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(family_ids)
    val_count = max(1, int(round(len(family_ids) * val_fraction))) if family_ids else 0
    val_families = set(family_ids[:val_count])
    train_rows, val_rows = [], []
    for row in rows:
        if row["family_id"] in val_families:
            row["split"] = "val"
            val_rows.append(row)
        else:
            row["split"] = "train"
            train_rows.append(row)
    return train_rows, val_rows


def write_manifests(train_rows: List[Dict[str, Any]], val_rows: List[Dict[str, Any]], output_dir: Path, test_sets: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
    manifest_dir = ensure_dir(output_dir.parent / "manifests")
    family_manifest = {}
    split_manifest = {"train": [], "val": []}
    source_manifest: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    label_distribution = {"action": Counter(), "harm": Counter(), "legitimacy": Counter(), "uncertainty": Counter(), "variant": Counter()}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            family_manifest.setdefault(row["family_id"], {"split": split, "source": row["source"], "topic": row["topic"], "ids": []})
            family_manifest[row["family_id"]]["ids"].append(row["id"])
            split_manifest[split].append(row["id"])
            source_manifest[row["source"]][split] += 1
            label_distribution["action"][row["action_label"]] += 1
            label_distribution["harm"][row["harm_label"]] += 1
            label_distribution["legitimacy"][row["legit_label"]] += 1
            label_distribution["uncertainty"][row["uncertainty_label"]] += 1
            label_distribution["variant"][row["variant_type"]] += 1
    for rows in (test_sets or {}).values():
        for row in rows:
            split_manifest.setdefault("test", []).append(row["id"])
            source_manifest[row["source"]]["test"] += 1
            label_distribution["action"][row["action_label"]] += 1
            label_distribution["harm"][row["harm_label"]] += 1
            label_distribution["legitimacy"][row["legit_label"]] += 1
            label_distribution["uncertainty"][row["uncertainty_label"]] += 1
            label_distribution["variant"][row["variant_type"]] += 1
    save_json(family_manifest, manifest_dir / "family_manifest.json")
    save_json(split_manifest, manifest_dir / "split_manifest.json")
    save_json({k: dict(v) for k, v in source_manifest.items()}, manifest_dir / "source_manifest.json")
    save_json({k: dict(v) for k, v in label_distribution.items()}, manifest_dir / "label_distribution.json")

    topic_manifest: Dict[str, Dict[str, int]] = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            topic_manifest.setdefault(row["topic"], {"train": 0, "val": 0})
            topic_manifest[row["topic"]][split] += 1
    save_json(topic_manifest, manifest_dir / "topic_split_manifest.json")


def load_hf_dataset(dataset_name: str, *args: Any, split: str = "train", **kwargs: Any) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to download public benchmark sources.") from exc
    return load_dataset(dataset_name, *args, split=split, **kwargs)


def maybe_limit(rows: Sequence[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    return list(rows[:limit] if limit else rows)


def build_processed(args: argparse.Namespace) -> Dict[str, int]:
    output_dir = ensure_dir(args.output_dir)
    train_pool: List[Dict[str, Any]] = []
    test_sets: Dict[str, List[Dict[str, Any]]] = {}

    if "wildjailbreak" in args.sources:
        raw = load_hf_dataset(
            args.wildjailbreak_dataset,
            args.wildjailbreak_config,
            split=args.wildjailbreak_split,
            delimiter="\t",
            keep_default_na=False,
            cache_dir=args.cache_dir,
        )
        train_pool.extend(convert_wildjailbreak(raw, split="train_pool", limit=args.limit_per_source))

    if "falsereject" in args.sources:
        raw = load_hf_dataset(args.falsereject_dataset, split=args.falsereject_split, cache_dir=args.cache_dir)
        train_pool.extend(convert_falsereject(raw, split="train_pool", limit=args.limit_per_source))

    if args.custom_auth:
        custom_rows = convert_custom_auth(Path(args.custom_auth))
        train_pool.extend([row for row in custom_rows if row["split"] in {"train", "val", "train_pool"}])
        custom_test = [row for row in custom_rows if row["split"] == "test"]
        if custom_test:
            test_sets["custom_auth"] = custom_test

    assign_counterfactual_families(train_pool, family_size=args.family_size)
    train_rows, val_rows = split_train_val_by_family(train_pool, args.val_fraction, args.seed)

    if "xstest" in args.sources:
        raw = load_hf_dataset(args.xstest_dataset, split=args.xstest_split, cache_dir=args.cache_dir)
        test_sets["xstest"] = convert_xstest(raw, limit=args.limit_per_source)

    if "orbench_hard" in args.sources:
        raw = load_hf_dataset(args.orbench_dataset, "or-bench-hard-1k", split="train", cache_dir=args.cache_dir)
        test_sets["orbench_hard"] = convert_orbench(raw, source="orbench_hard", limit=args.limit_per_source)

    if "orbench_toxic" in args.sources:
        raw = load_hf_dataset(args.orbench_dataset, "or-bench-toxic", split="train", cache_dir=args.cache_dir)
        test_sets["orbench_toxic"] = convert_orbench(raw, source="orbench_toxic", limit=args.limit_per_source)

    if "strongreject" in args.sources:
        try:
            raw = load_hf_dataset(args.strongreject_dataset, split=args.strongreject_split, cache_dir=args.cache_dir)
        except Exception:
            raw = load_hf_dataset(args.strongreject_fallback_dataset, split="train", cache_dir=args.cache_dir)
        test_sets["strongreject"] = convert_strongreject(raw, limit=args.limit_per_source)

    save_jsonl(train_rows, output_dir / "train.jsonl")
    save_jsonl(val_rows, output_dir / "val.jsonl")
    for name, rows in test_sets.items():
        save_jsonl(rows, output_dir / f"test_{name}.jsonl")
    if "orbench_hard" in test_sets and "orbench_toxic" in test_sets:
        save_jsonl(test_sets["orbench_hard"] + test_sets["orbench_toxic"], output_dir / "test_orbench.jsonl")
    if test_sets:
        merged = []
        for rows in test_sets.values():
            merged.extend(rows)
        save_jsonl(merged, output_dir / "test_all.jsonl")
    write_manifests(train_rows, val_rows, output_dir, test_sets)

    summary = {"train": len(train_rows), "val": len(val_rows)}
    summary.update({f"test_{name}": len(rows) for name, rows in test_sets.items()})
    save_json(summary, output_dir / "summary.json")
    from efsc.data.acceptance_checks import run_checks

    acceptance_rows = [*train_rows, *val_rows]
    for rows in test_sets.values():
        acceptance_rows.extend(rows)
    required_actions = [] if args.skip_maintrack_label_gate else ["ANSWER", "SAFE_ANSWER", "CLARIFY", "REFUSE"]
    required_harm = [] if args.skip_maintrack_label_gate else ["LOW", "MEDIUM", "HIGH"]
    acceptance_report = run_checks(
        acceptance_rows,
        required_actions=required_actions,
        required_harm_labels=required_harm,
        min_rows=0 if args.skip_maintrack_label_gate else 1,
    )
    save_json(acceptance_report, output_dir / "acceptance_report.json")
    if not acceptance_report["passed"]:
        raise RuntimeError(f"Dataset acceptance checks failed; see {output_dir / 'acceptance_report.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the locked EFSC NeurIPS dataset stack.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_TRAIN_SOURCES + DEFAULT_TEST_SOURCES))
    parser.add_argument("--custom-auth", default=None, help="Optional JSONL file for the custom authorization/ambiguity set.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--family-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-source", type=int, default=None, help="Debug limit for smoke tests.")
    parser.add_argument("--skip-maintrack-label-gate", action="store_true", help="Allow incomplete label coverage for smoke/debug data only.")

    parser.add_argument("--wildjailbreak-dataset", default="allenai/wildjailbreak")
    parser.add_argument("--wildjailbreak-config", default="train")
    parser.add_argument("--wildjailbreak-split", default="train")
    parser.add_argument("--falsereject-dataset", default="AmazonScience/FalseReject")
    parser.add_argument("--falsereject-split", default="train")
    parser.add_argument("--xstest-dataset", default="walledai/XSTest")
    parser.add_argument("--xstest-split", default="test")
    parser.add_argument("--orbench-dataset", default="bench-llms/or-bench")
    parser.add_argument("--strongreject-dataset", default="AlignmentResearch/StrongREJECT")
    parser.add_argument("--strongreject-fallback-dataset", default="Machlovi/strongreject-dataset")
    parser.add_argument("--strongreject-split", default="train")
    return parser.parse_args()


def main() -> None:
    summary = build_processed(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
