from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def stage1_supervised_loss(
    harm_logits: torch.Tensor,
    legit_logits: torch.Tensor,
    uncertainty_logits: torch.Tensor,
    action_logits: torch.Tensor,
    harm_labels: torch.Tensor,
    legit_labels: torch.Tensor,
    uncertainty_labels: torch.Tensor,
    action_labels: torch.Tensor,
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {"harm": 1.0, "legit": 1.0, "uncertainty": 1.0, "action": 1.0}
    return (
        weights["harm"] * F.cross_entropy(harm_logits, harm_labels)
        + weights["legit"] * F.cross_entropy(legit_logits, legit_labels)
        + weights["uncertainty"] * F.cross_entropy(uncertainty_logits, uncertainty_labels)
        + weights["action"] * F.cross_entropy(action_logits, action_labels)
    )


def stage2_counterfactual_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    same_family: torch.Tensor,
    same_action: torch.Tensor,
    margin_pos: float = 0.5,
    margin_neg: float = 1.0,
) -> torch.Tensor:
    distances = torch.norm(z_a - z_b, dim=-1)
    pos_mask = same_family.float()
    neg_mask = 1.0 - same_family.float()
    pos_loss = pos_mask * torch.relu(distances - margin_pos) ** 2
    neg_loss = neg_mask * torch.relu(margin_neg - distances) ** 2
    action_bonus = same_action.float() * 0.1 * distances
    return (pos_loss + neg_loss + action_bonus).mean()


def stage3_preference_loss(
    action_logits: torch.Tensor,
    preferred_action_ids: torch.Tensor,
    rejected_action_ids: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    rows = torch.arange(action_logits.size(0), device=action_logits.device)
    preferred_scores = action_logits[rows, preferred_action_ids]
    rejected_scores = action_logits[rows, rejected_action_ids]
    return -F.logsigmoid(beta * (preferred_scores - rejected_scores)).mean()
