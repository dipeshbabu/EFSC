from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from src.data.dataset_loader import ACTION_TO_ID

ACTION_TEXT = {
    "ANSWER": "Provide a direct answer.",
    "SAFE_ANSWER": "Provide a safe, bounded, high level answer.",
    "CLARIFY": "Ask a clarifying question before helping.",
    "REFUSE": "Refuse to provide harmful assistance.",
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PreferenceDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 512) -> None:
        self.rows = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_rejected(self, row: Dict[str, Any]) -> str:
        preferred = row["preferred_over"][0]
        rejected_pool = row.get("dispreferred_over", [])
        if not rejected_pool:
            rejected_pool = [name for name in ACTION_TEXT if name != preferred]
        return random.choice(rejected_pool)

    def _format_input(self, prompt: str, action_name: str) -> str:
        return f"User request:\n{prompt}\n\nPolicy action:\n{ACTION_TEXT[action_name]}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        chosen_action = row["preferred_over"][0]
        rejected_action = self._sample_rejected(row)

        chosen = self.tokenizer(
            self._format_input(row["prompt"], chosen_action),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            self._format_input(row["prompt"], rejected_action),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
            "chosen_action_id": torch.tensor(ACTION_TO_ID[chosen_action], dtype=torch.long),
            "rejected_action_name": rejected_action,
            "gold_action_name": chosen_action,
            "id": row["id"],
            "prompt": row["prompt"],
            "topic": row["topic"],
            "family_id": row["family_id"],
        }


def preference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
        "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
        "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
        "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch]),
        "chosen_action_id": torch.stack([item["chosen_action_id"] for item in batch]),
        "rejected_action_name": [item["rejected_action_name"] for item in batch],
        "gold_action_name": [item["gold_action_name"] for item in batch],
        "id": [item["id"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "topic": [item["topic"] for item in batch],
        "family_id": [item["family_id"] for item in batch],
    }
