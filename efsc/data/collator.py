from __future__ import annotations
from typing import Any, Dict, List
import torch

class PromptCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [x["prompt"] for x in batch]
        enc = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc["action_labels"] = torch.tensor([x["action_id"] for x in batch], dtype=torch.long)
        enc["harm_labels"] = torch.tensor([x["harm_id"] for x in batch], dtype=torch.long)
        enc["legit_labels"] = torch.tensor([x["legit_id"] for x in batch], dtype=torch.long)
        enc["uncertainty_labels"] = torch.tensor([x["uncertainty_id"] for x in batch], dtype=torch.long)
        enc["ids"] = [x["id"] for x in batch]
        enc["family_ids"] = [x["family_id"] for x in batch]
        enc["sources"] = [x.get("source", "unknown") for x in batch]
        enc["variant_types"] = [x["variant_type"] for x in batch]
        enc["languages"] = [x["language"] for x in batch]
        enc["topics"] = [x["topic"] for x in batch]
        return enc

class PairCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        harm = self.tokenizer([x["harmful_prompt"] for x in batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        ben = self.tokenizer([x["benign_prompt"] for x in batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"harmful_input_ids": harm["input_ids"], "harmful_attention_mask": harm["attention_mask"], "benign_input_ids": ben["input_ids"], "benign_attention_mask": ben["attention_mask"], "family_ids": [x["family_id"] for x in batch]}

class PreferenceCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        enc = self.tokenizer([x["prompt"] for x in batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc["preferred_action_ids"] = [x["preferred_action_ids"] for x in batch]
        enc["dispreferred_action_ids"] = [x["dispreferred_action_ids"] for x in batch]
        return enc
