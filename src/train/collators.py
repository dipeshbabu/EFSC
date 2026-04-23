from __future__ import annotations

from typing import Any, Dict, List

import torch


def efsc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "action_label": torch.stack([item["action_label"] for item in batch]),
        "harm_label": torch.stack([item["harm_label"] for item in batch]),
        "legit_label": torch.stack([item["legit_label"] for item in batch]),
        "uncertainty_label": torch.stack([item["uncertainty_label"] for item in batch]),
        "family_id": [item["family_id"] for item in batch],
        "topic": [item["topic"] for item in batch],
        "source": [item["source"] for item in batch],
        "variant_type": [item["variant_type"] for item in batch],
        "uncertainty_label_name": [item["uncertainty_label_name"] for item in batch],
        "uncertainty_label_text": [item["uncertainty_label_name"] for item in batch],
        "gold_action_text": [item["gold_action"] for item in batch],
        "id": [item["id"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "preferred_over": [item.get("preferred_over", []) for item in batch],
        "dispreferred_over": [item.get("dispreferred_over", []) for item in batch],
    }
