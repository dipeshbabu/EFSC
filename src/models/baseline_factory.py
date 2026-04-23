from __future__ import annotations

from src.models.direct_policy_decoder import DirectPolicyDecoder
from src.models.direct_policy_decoder_quantized import QuantizedDirectPolicyDecoder
from src.models.model_utils import build_lora_config_for_model
from src.models.plain_efsc_decoder import PlainEFSCCausalDecoder
from src.models.plain_efsc_decoder_quantized import QuantizedPlainEFSCCausalDecoder
from src.train.peft_utils import apply_lora


def build_baseline_model(
    model_name: str,
    baseline_type: str,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if baseline_type == "direct_policy":
        model = DirectPolicyDecoder(model_name)
    elif baseline_type == "plain_efsc":
        model = PlainEFSCCausalDecoder(model_name)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    if use_lora:
        config = build_lora_config_for_model(model_name, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        model.backbone = apply_lora(model.backbone, config)
    return model


def build_quantized_baseline_model(
    model_name: str,
    baseline_type: str,
    quant_mode: str,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    kwargs = {
        "backbone_name": model_name,
        "quant_mode": quant_mode,
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
    if baseline_type == "direct_policy":
        return QuantizedDirectPolicyDecoder(**kwargs)
    if baseline_type == "plain_efsc":
        return QuantizedPlainEFSCCausalDecoder(**kwargs)
    raise ValueError(f"Unknown baseline_type: {baseline_type}")
