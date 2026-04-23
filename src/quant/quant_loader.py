from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_quantized_backbone(model_name: str, quant_mode: str):
    quant_mode = quant_mode.lower()

    if quant_mode in {"fp16", "bf16", "full"}:
        dtype = torch.bfloat16 if quant_mode == "bf16" else torch.float16
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )

    if quant_mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )

    if quant_mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )

    raise ValueError(f"Unsupported quant_mode: {quant_mode}")
