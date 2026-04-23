from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.model_utils import build_lora_config_for_model
from src.quant.quant_loader import load_quantized_backbone
from src.train.peft_utils import apply_lora


@dataclass
class QuantDirectPolicyOutput:
    action_logits: torch.Tensor


class QuantizedDirectPolicyDecoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        quant_mode: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = load_quantized_backbone(backbone_name, quant_mode)
        if use_lora:
            config = build_lora_config_for_model(backbone_name, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.backbone = apply_lora(self.backbone, config)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.action_head.to(self.input_device)

    @property
    def input_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    def masked_last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuantDirectPolicyOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooled = self.masked_last_token_pool(outputs.hidden_states[-1], attention_mask)
        dtype = next(self.action_head.parameters()).dtype
        pooled = pooled.to(device=self.input_device, dtype=dtype)
        return QuantDirectPolicyOutput(action_logits=self.action_head(pooled))
