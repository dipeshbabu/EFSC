from __future__ import annotations

from src.models.efsc_model import DirectActionClassifier, EFSCModel
from src.train.peft_utils import apply_lora, build_lora_config


def build_efsc_model(
    backbone_name: str,
    freeze_backbone: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    model = EFSCModel(
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone and not use_lora,
    )

    if use_lora:
        config = build_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model.backbone = apply_lora(model.backbone, config)

    return model


def build_direct_model(
    backbone_name: str,
    freeze_backbone: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    model = DirectActionClassifier(
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone and not use_lora,
    )

    if use_lora:
        config = build_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model.backbone = apply_lora(model.backbone, config)

    return model
