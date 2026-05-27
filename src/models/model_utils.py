from __future__ import annotations

from src.models.ablation_factory import build_ablation_model
from src.models.decoder_efsc_model import DecoderEFSCModel
from src.train.peft_utils import apply_lora, build_lora_config


DECODER_HINTS = ["qwen", "llama", "mistral", "phi", "gemma"]


def is_decoder_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(hint in name for hint in DECODER_HINTS)


def build_lora_config_for_model(
    backbone_name: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    name = backbone_name.lower()
    if any(hint in name for hint in ("qwen", "llama", "mistral", "gemma")):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in name:
        targets = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    else:
        targets = ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"]

    return build_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        task_type_name="CAUSAL_LM",
    )


def build_main_model(
    backbone_name: str,
    variant: str,
    freeze_backbone: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if is_decoder_model(backbone_name):
        model = DecoderEFSCModel(
            backbone_name=backbone_name,
            variant=variant,
            freeze_backbone=freeze_backbone and not use_lora,
        )
        if use_lora:
            config = build_lora_config_for_model(
                backbone_name=backbone_name,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            model.backbone = apply_lora(model.backbone, config)
        return model

    return build_ablation_model(
        backbone_name=backbone_name,
        variant=variant,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
