from __future__ import annotations

from src.models.decoder_qefsc_fc import DecoderQEFSCFCModel
from src.models.model_utils import build_lora_config_for_model, is_decoder_model
from src.train.peft_utils import apply_lora


def build_qefsc_model(
    backbone_name: str,
    variant: str,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if variant == "qefsc_fc":
        model = DecoderQEFSCFCModel(backbone_name, layer_indices=[-1, -4, -8, -12], fusion_type="gated")
    elif variant == "qefsc_fc_concat":
        model = DecoderQEFSCFCModel(backbone_name, layer_indices=[-1, -4, -8, -12], fusion_type="concat")
    elif variant == "qefsc_fc_last2":
        model = DecoderQEFSCFCModel(backbone_name, layer_indices=[-1, -2], fusion_type="gated")
    else:
        raise ValueError(f"Unsupported qefsc variant: {variant}")

    if use_lora:
        if not is_decoder_model(backbone_name):
            raise ValueError("QEFSC feature-conditioned variants currently require a decoder-only backbone for LoRA.")
        config = build_lora_config_for_model(backbone_name, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        model.backbone = apply_lora(model.backbone, config)
    return model
