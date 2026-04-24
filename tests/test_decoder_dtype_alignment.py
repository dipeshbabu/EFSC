from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.direct_policy_decoder import DirectPolicyDecoder
from src.models.plain_efsc_decoder import PlainEFSCCausalDecoder


class DummyBackbone(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.hidden = hidden

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict, use_cache):
        return SimpleNamespace(hidden_states=[self.hidden])


def build_plain_model(hidden: torch.Tensor) -> PlainEFSCCausalDecoder:
    model = PlainEFSCCausalDecoder.__new__(PlainEFSCCausalDecoder)
    nn.Module.__init__(model)
    model.backbone = DummyBackbone(hidden)
    model.hidden_size = hidden.size(-1)
    model.proj = nn.Sequential(nn.Linear(hidden.size(-1), 8), nn.GELU(), nn.Linear(8, 4))
    model.harm_head = nn.Linear(4, 3)
    model.legit_head = nn.Linear(4, 3)
    model.uncertainty_head = nn.Linear(4, 2)
    model.action_head = nn.Linear(4 + 3 + 3 + 2, 4)
    return model


def build_direct_policy_model(hidden: torch.Tensor) -> DirectPolicyDecoder:
    model = DirectPolicyDecoder.__new__(DirectPolicyDecoder)
    nn.Module.__init__(model)
    model.backbone = DummyBackbone(hidden)
    model.hidden_size = hidden.size(-1)
    model.action_head = nn.Sequential(nn.Linear(hidden.size(-1), 8), nn.GELU(), nn.Linear(8, 4))
    return model


def test_plain_efsc_casts_backbone_hidden_states_to_controller_dtype():
    hidden = torch.randn(2, 3, 6, dtype=torch.bfloat16)
    model = build_plain_model(hidden)
    input_ids = torch.ones(2, 3, dtype=torch.long)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    out = model(input_ids, attention_mask)

    assert out.harm_logits.dtype == torch.float32
    assert out.action_logits.dtype == torch.float32


def test_direct_policy_casts_backbone_hidden_states_to_controller_dtype():
    hidden = torch.randn(2, 3, 6, dtype=torch.bfloat16)
    model = build_direct_policy_model(hidden)
    input_ids = torch.ones(2, 3, dtype=torch.long)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    out = model(input_ids, attention_mask)

    assert out.action_logits.dtype == torch.float32
