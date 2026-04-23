from __future__ import annotations

from typing import Any, Optional

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ModuleNotFoundError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def require_peft() -> None:
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise RuntimeError("LoRA training requires `peft`. Install project requirements with `uv pip install -e \".[dev]\"` or `uv pip install -e \".[dev,cuda]\"`.")


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    task_type_name: str = "FEATURE_EXTRACTION",
) -> Any:
    require_peft()
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "query",
            "key",
            "value",
            "dense",
        ]

    task_type = getattr(TaskType, task_type_name)
    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for _, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    pct = 100.0 * trainable / max(total, 1)
    print(f"trainable params: {trainable}")
    print(f"total params: {total}")
    print(f"trainable%: {pct:.4f}")


def apply_lora(model, config: Any):
    require_peft()
    peft_model = get_peft_model(model, config)
    print_trainable_parameters(peft_model)
    return peft_model
