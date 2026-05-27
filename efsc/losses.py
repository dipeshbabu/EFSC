from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from efsc.modeling import EFSCOutput

def supervised_loss(outputs: EFSCOutput, action_labels: torch.Tensor, harm_labels: torch.Tensor, legit_labels: torch.Tensor, uncertainty_labels: torch.Tensor, weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    action_loss = F.cross_entropy(outputs.action_logits, action_labels)
    harm_loss = F.cross_entropy(outputs.harm_logits, harm_labels)
    legit_loss = F.cross_entropy(outputs.legit_logits, legit_labels)
    uncertainty_loss = F.cross_entropy(outputs.uncertainty_logits, uncertainty_labels)
    total = action_loss + weights.get("harm", 1.0) * harm_loss + weights.get("legit", 1.0) * legit_loss + weights.get("uncertainty", 1.0) * uncertainty_loss
    return total, {"loss": float(total.detach().cpu().item())}

def margin_ranking_loss(harmful_scores: torch.Tensor, benign_scores: torch.Tensor, margin: float) -> torch.Tensor:
    return F.relu(margin - (harmful_scores - benign_scores)).mean()

def consistency_loss(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return ((p1 - p2) ** 2).mean()

def pairwise_dpo_action_loss(action_logits: torch.Tensor, preferred_action_ids: list[list[int]], dispreferred_action_ids: list[list[int]], beta: float = 1.0) -> torch.Tensor:
    losses = []
    for i in range(action_logits.size(0)):
        pref = torch.tensor(preferred_action_ids[i], device=action_logits.device, dtype=torch.long)
        dis = torch.tensor(dispreferred_action_ids[i], device=action_logits.device, dtype=torch.long)
        pref_score = torch.logsumexp(action_logits[i, pref], dim=0)
        dis_score = torch.logsumexp(action_logits[i, dis], dim=0)
        losses.append(-F.logsigmoid(beta * (pref_score - dis_score)))
    return torch.stack(losses).mean()
