from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

ACTION_TO_ID = {"ANSWER": 0, "SAFE_ANSWER": 1, "CLARIFY": 2, "REFUSE": 3}
ID_TO_ACTION = {value: key for key, value in ACTION_TO_ID.items()}
HARM_TO_ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LEGIT_TO_ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
UNCERTAINTY_TO_ID = {"CLEAR": 0, "AMBIGUOUS": 1}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EFSCDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 512) -> None:
        self.rows = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _format_text(self, row: Dict[str, Any]) -> str:
        return f"User request:\n{row['prompt']}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        action_name = row.get("gold_action", row.get("action_label"))
        if action_name is None:
            raise KeyError("Dataset row must contain either `gold_action` or `action_label`.")

        encoded = self.tokenizer(
            self._format_text(row),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "action_label": torch.tensor(ACTION_TO_ID[action_name], dtype=torch.long),
            "harm_label": torch.tensor(HARM_TO_ID[row["harm_label"]], dtype=torch.long),
            "legit_label": torch.tensor(LEGIT_TO_ID[row["legit_label"]], dtype=torch.long),
            "uncertainty_label": torch.tensor(UNCERTAINTY_TO_ID[row["uncertainty_label"]], dtype=torch.long),
            "uncertainty_label_id": torch.tensor(UNCERTAINTY_TO_ID[row["uncertainty_label"]], dtype=torch.long),
            "family_id": row.get("family_id", ""),
            "topic": row.get("topic", "unknown"),
            "source": row.get("source", "unknown"),
            "variant_type": row.get("variant_type", "unknown"),
            "uncertainty_label_name": row["uncertainty_label"],
            "gold_action": action_name,
            "id": row["id"],
            "prompt": row["prompt"],
            "preferred_over": row.get("preferred_over", []),
            "dispreferred_over": row.get("dispreferred_over", []),
        }
