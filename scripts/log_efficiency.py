from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--num_trainable_params", type=int, required=True)
    parser.add_argument("--num_total_params", type=int, required=True)
    parser.add_argument("--peak_gpu_mem_mb", type=float, required=True)
    parser.add_argument("--train_wall_time_sec", type=float, required=True)
    parser.add_argument("--tokens_per_sec", type=float, default=0.0)
    parser.add_argument("--latency_ms_per_example", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    obj = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "method": args.method,
        "num_trainable_params": args.num_trainable_params,
        "num_total_params": args.num_total_params,
        "trainable_fraction": args.num_trainable_params / max(args.num_total_params, 1),
        "peak_gpu_mem_mb": args.peak_gpu_mem_mb,
        "train_wall_time_sec": args.train_wall_time_sec,
        "tokens_per_sec": args.tokens_per_sec,
        "latency_ms_per_example": args.latency_ms_per_example,
        "logged_at_unix": time.time(),
    }
    save_json(obj, Path(args.output))
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
