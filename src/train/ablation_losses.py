from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ablation_supervised_loss(
    harm_logits,
    legit_logits,
    uncertainty_logits,
    action_logits,
    harm_labels,
    legit_labels,
    uncertainty_labels,
    action_labels,
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {"harm": 1.0, "legit": 1.0, "uncertainty": 1.0, "action": 1.0}
    total = weights["action"] * F.cross_entropy(action_logits, action_labels)
    if harm_logits is not None:
        total = total + weights["harm"] * F.cross_entropy(harm_logits, harm_labels)
    if legit_logits is not None:
        total = total + weights["legit"] * F.cross_entropy(legit_logits, legit_labels)
    if uncertainty_logits is not None:
        total = total + weights["uncertainty"] * F.cross_entropy(uncertainty_logits, uncertainty_labels)
    return total
