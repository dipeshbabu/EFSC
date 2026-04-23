from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from src.models.feature_fusion import build_fusion_module
from src.models.model_utils import build_lora_config_for_model
from src.quant.quant_loader import load_quantized_backbone
from src.train.peft_utils import apply_lora


@dataclass
class QuantQEFSCOutput:
    fused: torch.Tensor
    harm_logits: torch.Tensor
    legit_logits: torch.Tensor
    uncertainty_logits: torch.Tensor
    action_logits: torch.Tensor


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantizedDecoderQEFSCFCModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        quant_mode: str,
        layer_indices: List[int],
        fusion_type: str = "gated",
        fused_dim: int = 256,
        head_hidden_dim: int = 256,
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
        self.layer_indices = layer_indices
        self.fusion = build_fusion_module(fusion_type, self.hidden_size, len(layer_indices), fused_dim, dropout)
        self.harm_head = MLPHead(fused_dim, head_hidden_dim, 3, dropout)
        self.legit_head = MLPHead(fused_dim, head_hidden_dim, 3, dropout)
        self.uncertainty_head = MLPHead(fused_dim, head_hidden_dim, 2, dropout)
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim + 3 + 3 + 2, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 4),
        )
        self._move_controller_to_input_device()

    @property
    def input_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    def _move_controller_to_input_device(self) -> None:
        device = self.input_device
        self.fusion.to(device)
        self.harm_head.to(device)
        self.legit_head.to(device)
        self.uncertainty_head.to(device)
        self.action_head.to(device)

    def _controller_dtype(self) -> torch.dtype:
        return next(self.fusion.parameters()).dtype

    def _resolve_layer_index(self, idx: int, num_hidden_states: int) -> int:
        resolved = idx if idx >= 0 else num_hidden_states + idx
        return min(max(resolved, 0), num_hidden_states - 1)

    def masked_last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuantQEFSCOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        dtype = self._controller_dtype()
        feats = []
        for idx in self.layer_indices:
            resolved = self._resolve_layer_index(idx, len(outputs.hidden_states))
            pooled = self.masked_last_token_pool(outputs.hidden_states[resolved], attention_mask)
            feats.append(pooled.to(device=self.input_device, dtype=dtype))

        fused = self.fusion(feats)
        harm_logits = self.harm_head(fused)
        legit_logits = self.legit_head(fused)
        uncertainty_logits = self.uncertainty_head(fused)
        action_in = torch.cat(
            [
                fused,
                torch.softmax(harm_logits, dim=-1),
                torch.softmax(legit_logits, dim=-1),
                torch.softmax(uncertainty_logits, dim=-1),
            ],
            dim=-1,
        )
        action_logits = self.action_head(action_in)
        return QuantQEFSCOutput(fused, harm_logits, legit_logits, uncertainty_logits, action_logits)
