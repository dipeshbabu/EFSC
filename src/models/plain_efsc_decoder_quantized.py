from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.model_utils import build_lora_config_for_model
from src.quant.quant_loader import load_quantized_backbone
from src.train.peft_utils import apply_lora


@dataclass
class QuantPlainEFSCOutput:
    harm_logits: torch.Tensor
    legit_logits: torch.Tensor
    uncertainty_logits: torch.Tensor
    action_logits: torch.Tensor


class QuantizedPlainEFSCCausalDecoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        quant_mode: str,
        bottleneck_dim: int = 128,
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
        self.proj = self._head(self.hidden_size, hidden_dim, bottleneck_dim, dropout)
        self.harm_head = self._head(bottleneck_dim, hidden_dim, 3, dropout)
        self.legit_head = self._head(bottleneck_dim, hidden_dim, 3, dropout)
        self.uncertainty_head = self._head(bottleneck_dim, hidden_dim, 2, dropout)
        self.action_head = self._head(bottleneck_dim + 3 + 3 + 2, hidden_dim, 4, dropout)
        self._move_controller_to_input_device()

    @property
    def input_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    def _head(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _move_controller_to_input_device(self) -> None:
        for module in (self.proj, self.harm_head, self.legit_head, self.uncertainty_head, self.action_head):
            module.to(self.input_device)

    def _controller_dtype(self) -> torch.dtype:
        return next(self.proj.parameters()).dtype

    def masked_last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuantPlainEFSCOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooled = self.masked_last_token_pool(outputs.hidden_states[-1], attention_mask)
        pooled = pooled.to(device=self.input_device, dtype=self._controller_dtype())
        z = self.proj(pooled)
        harm_logits = self.harm_head(z)
        legit_logits = self.legit_head(z)
        uncertainty_logits = self.uncertainty_head(z)
        action_in = torch.cat(
            [
                z,
                torch.softmax(harm_logits, dim=-1),
                torch.softmax(legit_logits, dim=-1),
                torch.softmax(uncertainty_logits, dim=-1),
            ],
            dim=-1,
        )
        return QuantPlainEFSCOutput(
            harm_logits=harm_logits,
            legit_logits=legit_logits,
            uncertainty_logits=uncertainty_logits,
            action_logits=self.action_head(action_in),
        )
