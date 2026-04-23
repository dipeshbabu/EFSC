from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EfficiencyTracker:
    start_time: float
    peak_gpu_mem_mb: float = 0.0
    total_tokens: int = 0
    total_examples: int = 0

    @classmethod
    def start(cls) -> "EfficiencyTracker":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return cls(start_time=time.time())

    def update_batch(self, input_ids: torch.Tensor) -> None:
        self.total_examples += int(input_ids.size(0))
        self.total_tokens += int(input_ids.numel())
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.peak_gpu_mem_mb = max(self.peak_gpu_mem_mb, peak_bytes / (1024 ** 2))

    def finish(self) -> Dict[str, float]:
        wall_time = time.time() - self.start_time
        return {
            "train_wall_time_sec": wall_time,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "tokens_per_sec": self.total_tokens / max(wall_time, 1e-8),
            "latency_ms_per_example": 1000.0 * wall_time / max(self.total_examples, 1),
        }


def count_parameters(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
    return {
        "num_total_params": total,
        "num_trainable_params": trainable,
        "trainable_fraction": trainable / max(total, 1),
    }
