from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * num_layers, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, layer_feats: List[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(layer_feats, dim=-1))


class GatedWeightedFusion(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * num_layers, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_layers),
        )
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, layer_feats: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(layer_feats, dim=1)
        gate_in = torch.cat(layer_feats, dim=-1)
        weights = torch.softmax(self.gate(gate_in), dim=-1)
        return self.proj((stacked * weights.unsqueeze(-1)).sum(dim=1))


class MeanFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, layer_feats: List[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.stack(layer_feats, dim=1).mean(dim=1))


def build_fusion_module(
    fusion_type: str,
    input_dim: int,
    num_layers: int,
    output_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    if fusion_type == "concat":
        return ConcatFusion(input_dim, num_layers, output_dim, dropout)
    if fusion_type == "gated":
        return GatedWeightedFusion(input_dim, num_layers, output_dim, dropout)
    if fusion_type == "mean":
        return MeanFusion(input_dim, output_dim)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")
