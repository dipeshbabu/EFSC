from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@dataclass
class DecoderEFSCOutput:
    pooled: torch.Tensor
    bottleneck: torch.Tensor
    harm_logits: Optional[torch.Tensor]
    legit_logits: Optional[torch.Tensor]
    uncertainty_logits: Optional[torch.Tensor]
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


class DecoderEFSCModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        variant: str = "efsc_full",
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name, trust_remote_code=True)
        self.hidden_size = int(self.backbone.config.hidden_size)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if variant == "efsc_no_bottleneck":
            self.proj = nn.Identity()
            self.feature_dim = self.hidden_size
        elif variant == "efsc_small_bottleneck":
            self.proj = nn.Sequential(nn.Linear(self.hidden_size, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 32))
            self.feature_dim = 32
        elif variant == "efsc_large_bottleneck":
            self.proj = nn.Sequential(nn.Linear(self.hidden_size, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 256))
            self.feature_dim = 256
        else:
            self.proj = nn.Sequential(nn.Linear(self.hidden_size, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, bottleneck_dim))
            self.feature_dim = bottleneck_dim

        self.use_harm = variant not in {"efsc_no_harm_head", "efsc_shared_head"}
        self.use_legit = variant not in {"efsc_no_legit_head", "efsc_shared_head"}
        self.use_uncertainty = variant not in {"efsc_no_uncertainty_head", "efsc_shared_head"}

        if variant == "efsc_shared_head":
            self.shared_factor_head = MLPHead(self.feature_dim, hidden_dim, 8, dropout)
        else:
            self.harm_head = MLPHead(self.feature_dim, hidden_dim, 3, dropout) if self.use_harm else None
            self.legit_head = MLPHead(self.feature_dim, hidden_dim, 3, dropout) if self.use_legit else None
            self.uncertainty_head = MLPHead(self.feature_dim, hidden_dim, 2, dropout) if self.use_uncertainty else None

        if variant == "efsc_no_factor_conditioning":
            action_in_dim = self.feature_dim
        else:
            factor_dim = 0
            factor_dim += 3 if self.use_harm or variant == "efsc_shared_head" else 0
            factor_dim += 3 if self.use_legit or variant == "efsc_shared_head" else 0
            factor_dim += 2 if self.use_uncertainty or variant == "efsc_shared_head" else 0
            action_in_dim = self.feature_dim + factor_dim

        self.action_head = nn.Sequential(
            nn.Linear(action_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return outputs.hidden_states[-1]

    def masked_last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> DecoderEFSCOutput:
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        pooled = self.masked_last_token_pool(hidden_states, attention_mask)
        z = self.proj(pooled)
        harm_logits = legit_logits = uncertainty_logits = None

        if self.variant == "efsc_shared_head":
            shared = self.shared_factor_head(z)
            harm_logits = shared[:, 0:3]
            legit_logits = shared[:, 3:6]
            uncertainty_logits = shared[:, 6:8]
        else:
            if self.use_harm and self.harm_head is not None:
                harm_logits = self.harm_head(z)
            if self.use_legit and self.legit_head is not None:
                legit_logits = self.legit_head(z)
            if self.use_uncertainty and self.uncertainty_head is not None:
                uncertainty_logits = self.uncertainty_head(z)

        if self.variant == "efsc_no_factor_conditioning":
            action_in = z
        else:
            parts = [z]
            if harm_logits is not None:
                parts.append(torch.softmax(harm_logits, dim=-1))
            if legit_logits is not None:
                parts.append(torch.softmax(legit_logits, dim=-1))
            if uncertainty_logits is not None:
                parts.append(torch.softmax(uncertainty_logits, dim=-1))
            action_in = torch.cat(parts, dim=-1)

        return DecoderEFSCOutput(
            pooled=pooled,
            bottleneck=z,
            harm_logits=harm_logits,
            legit_logits=legit_logits,
            uncertainty_logits=uncertainty_logits,
            action_logits=self.action_head(action_in),
        )
