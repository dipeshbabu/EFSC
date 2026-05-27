from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from efsc.constants import ACTION2ID, HARM2ID, LEGIT2ID, UNCERTAINTY2ID

@dataclass
class EFSCOutput:
    action_logits: torch.Tensor
    harm_logits: torch.Tensor
    legit_logits: torch.Tensor
    uncertainty_logits: torch.Tensor
    pooled: torch.Tensor
    bottleneck: torch.Tensor

class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

class EFSCModel(nn.Module):
    def __init__(self, model_name: str, bottleneck_dim: int = 128, dropout: float = 0.1, freeze_backbone: bool = True, action_hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.pooler = MeanPooler()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.bottleneck = nn.Sequential(nn.Linear(self.hidden_size, bottleneck_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck_dim, bottleneck_dim))
        self.harm_head = nn.Linear(bottleneck_dim, len(HARM2ID))
        self.legit_head = nn.Linear(bottleneck_dim, len(LEGIT2ID))
        self.uncertainty_head = nn.Linear(bottleneck_dim, len(UNCERTAINTY2ID))
        self.action_mlp = nn.Sequential(nn.Linear(bottleneck_dim + len(HARM2ID) + len(LEGIT2ID) + len(UNCERTAINTY2ID), action_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(action_hidden_dim, len(ACTION2ID)))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EFSCOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        z = self.bottleneck(pooled)
        harm_logits = self.harm_head(z)
        legit_logits = self.legit_head(z)
        uncertainty_logits = self.uncertainty_head(z)
        features = torch.cat([z, F.softmax(harm_logits, dim=-1), F.softmax(legit_logits, dim=-1), F.softmax(uncertainty_logits, dim=-1)], dim=-1)
        action_logits = self.action_mlp(features)
        return EFSCOutput(action_logits, harm_logits, legit_logits, uncertainty_logits, pooled, z)

class DirectActionClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = 128, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooler()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, len(ACTION2ID)))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        return self.head(pooled)
