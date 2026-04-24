from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@dataclass
class DirectPolicyOutput:
    action_logits: torch.Tensor


class DirectPolicyDecoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name, trust_remote_code=True)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def masked_last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def _controller_dtype(self) -> torch.dtype:
        return next(self.action_head.parameters()).dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> DirectPolicyOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooled = self.masked_last_token_pool(outputs.hidden_states[-1], attention_mask)
        pooled = pooled.to(dtype=self._controller_dtype())
        return DirectPolicyOutput(action_logits=self.action_head(pooled))
