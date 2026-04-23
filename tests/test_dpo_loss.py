import torch

from src.train.dpo_loss import dpo_loss, gather_action_logprob


def test_gather_action_logprob_shape():
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    action_ids = torch.tensor([0, 1])
    out = gather_action_logprob(logits, action_ids)
    assert out.shape == (2,)


def test_dpo_loss_prefers_lower_loss_for_better_policy_margin():
    chosen_ids = torch.tensor([0])
    ref_chosen = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ref_rejected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    weak = dpo_loss(
        policy_chosen_logits=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        policy_rejected_logits=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ref_chosen_logits=ref_chosen,
        ref_rejected_logits=ref_rejected,
        chosen_action_ids=chosen_ids,
    )
    strong = dpo_loss(
        policy_chosen_logits=torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
        policy_rejected_logits=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        ref_chosen_logits=ref_chosen,
        ref_rejected_logits=ref_rejected,
        chosen_action_ids=chosen_ids,
    )
    assert strong < weak
