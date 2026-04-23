from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List
from torch.utils.data import Dataset
from efsc.constants import ACTION2ID, HARM2ID, LEGIT2ID, SOURCES, UNCERTAINTY2ID, VARIANT_TYPES
from efsc.data.schemas import PromptExample
from efsc.utils import load_jsonl

def validate_prompt_example(row: Dict[str, Any]) -> None:
    required = {
        "id",
        "family_id",
        "prompt",
        "language",
        "topic",
        "variant_type",
        "action_label",
        "harm_label",
        "legit_label",
        "uncertainty_label",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"Missing required EFSC fields: {missing}")
    if row["action_label"] not in ACTION2ID:
        raise ValueError(f"Unknown action_label={row['action_label']!r} for id={row['id']}")
    if row["harm_label"] not in HARM2ID:
        raise ValueError(f"Unknown harm_label={row['harm_label']!r} for id={row['id']}")
    if row["legit_label"] not in LEGIT2ID:
        raise ValueError(f"Unknown legit_label={row['legit_label']!r} for id={row['id']}")
    if row["uncertainty_label"] not in UNCERTAINTY2ID:
        raise ValueError(f"Unknown uncertainty_label={row['uncertainty_label']!r} for id={row['id']}")
    if row.get("variant_type") not in VARIANT_TYPES:
        raise ValueError(f"Unknown variant_type={row.get('variant_type')!r} for id={row['id']}")
    if row.get("source", "unknown") not in SOURCES:
        raise ValueError(f"Unknown source={row.get('source')!r} for id={row['id']}")

class PromptDataset(Dataset):
    def __init__(self, path: str):
        rows = load_jsonl(path)
        for row in rows:
            validate_prompt_example(row)
        self.examples = [PromptExample(**row) for row in rows]
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        row = asdict(ex)
        row["action_id"] = ACTION2ID[ex.action_label]
        row["harm_id"] = HARM2ID[ex.harm_label]
        row["legit_id"] = LEGIT2ID[ex.legit_label]
        row["uncertainty_id"] = UNCERTAINTY2ID[ex.uncertainty_label]
        return row

class CounterfactualPairDataset(Dataset):
    def __init__(self, path: str):
        examples = [PromptExample(**row) for row in load_jsonl(path)]
        by_family = {}
        for ex in examples: by_family.setdefault(ex.family_id, []).append(ex)
        self.pairs = []
        for family, fam_examples in by_family.items():
            harmful = [x for x in fam_examples if x.variant_type in {"harmful_malicious", "jailbreak", "toxic_harmful"}]
            benign = [x for x in fam_examples if x.variant_type in {"educational_benign", "defensive_benign", "authorized_benign", "hard_benign", "ambiguous_legitimate", "translation_benign", "paraphrase_benign"}]
            for h in harmful:
                for b in benign:
                    self.pairs.append({"family_id": family, "harmful": h, "benign": b})
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        return {"family_id": item["family_id"], "harmful_prompt": item["harmful"].prompt, "benign_prompt": item["benign"].prompt}

class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        self.rows = load_jsonl(path)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {"prompt": row["prompt"], "preferred_action_ids": [ACTION2ID[x] for x in row.get("preferred_over", [])] or [ACTION2ID[row["action_label"]]], "dispreferred_action_ids": [ACTION2ID[x] for x in row.get("dispreferred_over", [])] or [ACTION2ID["REFUSE"]]}
