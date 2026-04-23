from __future__ import annotations

import torch


def gather_action_logprob(action_logits: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(action_logits, dim=-1)
    return log_probs.gather(1, action_ids.unsqueeze(1)).squeeze(1)


def dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_action_ids: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi_yw = gather_action_logprob(policy_chosen_logits, chosen_action_ids)
    pi_yl = gather_action_logprob(policy_rejected_logits, chosen_action_ids)
    ref_yw = gather_action_logprob(ref_chosen_logits, chosen_action_ids)
    ref_yl = gather_action_logprob(ref_rejected_logits, chosen_action_ids)
    advantage = (pi_yw - pi_yl) - (ref_yw - ref_yl)
    return -torch.log(torch.sigmoid(beta * advantage) + 1e-8).mean()
