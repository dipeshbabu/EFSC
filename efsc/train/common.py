from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from transformers import get_linear_schedule_with_warmup
from efsc.config import load_config
from efsc.utils import ensure_dir, device_from_config, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); return parser.parse_args()

def setup_run(config_path: str) -> Tuple[Dict[str, Any], torch.device, Path]:
    config = load_config(config_path); set_seed(config.get("seed", 42)); output_dir = ensure_dir(config["output_dir"]); device = device_from_config(config); return config, device, output_dir

def build_optimizer_scheduler(model: torch.nn.Module, config: Dict[str, Any], total_steps: int):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config.get("learning_rate", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    warmup_steps = int(config.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler

def write_history_csv(history: list[Dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)

def selection_score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics.get("macro_f1", 0.0))
        + float(metrics.get("benign_answer_rate", 0.0))
        + float(metrics.get("harmful_refusal_rate", 0.0))
        - float(metrics.get("over_refusal_rate", 0.0))
        - float(metrics.get("under_refusal_rate", 0.0))
    )
