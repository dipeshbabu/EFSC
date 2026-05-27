from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from efsc.utils import save_json


def read_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and normalize efficiency metrics for EFSC runs.")
    parser.add_argument("--run-metrics", default=None, help="Existing metrics.json from a training or eval run.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for file size accounting.")
    parser.add_argument("--trainable-params", type=int, default=None)
    parser.add_argument("--wall-clock-hours", type=float, default=None)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--latency-per-example-sec", type=float, default=None)
    parser.add_argument("--throughput", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = read_json(args.run_metrics)
    if args.trainable_params is not None:
        metrics["trainable_params"] = args.trainable_params
    if args.wall_clock_hours is not None:
        metrics["wall_clock_hours"] = args.wall_clock_hours
        metrics["gpu_hours"] = args.wall_clock_hours * args.num_gpus
    if args.latency_per_example_sec is not None:
        metrics["latency_per_example_sec"] = args.latency_per_example_sec
    if args.throughput is not None:
        metrics["throughput_examples_per_sec"] = args.throughput
    elif metrics.get("latency_per_example_sec"):
        metrics["throughput_examples_per_sec"] = 1.0 / max(metrics["latency_per_example_sec"], 1e-12)
    if args.checkpoint:
        path = Path(args.checkpoint)
        metrics["checkpoint_bytes"] = path.stat().st_size if path.exists() else 0
    save_json(metrics, args.output)


if __name__ == "__main__":
    main()
