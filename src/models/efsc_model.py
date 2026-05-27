from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class EFSCOutput:
    pooled: torch.Tensor
    bottleneck: torch.Tensor
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


class EFSCModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = int(self.backbone.config.hidden_size)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.harm_head = MLPHead(bottleneck_dim, hidden_dim, 3, dropout)
        self.legit_head = MLPHead(bottleneck_dim, hidden_dim, 3, dropout)
        self.uncertainty_head = MLPHead(bottleneck_dim, hidden_dim, 2, dropout)
        self.action_head = nn.Sequential(
            nn.Linear(bottleneck_dim + 3 + 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def get_pooled(self, backbone_outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(backbone_outputs, "pooler_output") and backbone_outputs.pooler_output is not None:
            return backbone_outputs.pooler_output
        mask = attention_mask.unsqueeze(-1).float()
        return (backbone_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EFSCOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.get_pooled(outputs, attention_mask)
        bottleneck = self.proj(pooled)

        harm_logits = self.harm_head(bottleneck)
        legit_logits = self.legit_head(bottleneck)
        uncertainty_logits = self.uncertainty_head(bottleneck)

        factor_concat = torch.cat(
            [
                bottleneck,
                torch.softmax(harm_logits, dim=-1),
                torch.softmax(legit_logits, dim=-1),
                torch.softmax(uncertainty_logits, dim=-1),
            ],
            dim=-1,
        )
        action_logits = self.action_head(factor_concat)

        return EFSCOutput(
            pooled=pooled,
            bottleneck=bottleneck,
            harm_logits=harm_logits,
            legit_logits=legit_logits,
            uncertainty_logits=uncertainty_logits,
            action_logits=action_logits,
        )


class DirectActionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = int(self.backbone.config.hidden_size)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def get_pooled(self, backbone_outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(backbone_outputs, "pooler_output") and backbone_outputs.pooler_output is not None:
            return backbone_outputs.pooler_output
        mask = attention_mask.unsqueeze(-1).float()
        return (backbone_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.get_pooled(outputs, attention_mask)
        return {"pooled": pooled, "action_logits": self.classifier(pooled)}
