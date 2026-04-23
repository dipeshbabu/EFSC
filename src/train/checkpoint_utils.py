from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def load_checkpoint_state(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_non_backbone_checkpoint(model, checkpoint: str | Path, include_lora: bool = True) -> None:
    state = load_checkpoint_state(checkpoint, map_location="cpu")
    own = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key not in own:
            continue
        if tuple(value.shape) != tuple(own[key].shape):
            continue
        is_lora = "lora_" in key
        is_backbone = key.startswith("backbone.")
        if not is_backbone or (include_lora and is_lora):
            filtered[key] = value
    own.update(filtered)
    model.load_state_dict(own, strict=False)
