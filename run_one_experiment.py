from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ABLATION_METHODS = {
    "efsc_full",
    "efsc_no_harm_head",
    "efsc_no_legit_head",
    "efsc_no_uncertainty_head",
    "efsc_shared_head",
    "efsc_no_factor_conditioning",
    "efsc_no_bottleneck",
    "efsc_small_bottleneck",
    "efsc_large_bottleneck",
}


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_use_lora(model_name: str) -> bool:
    return model_name != "roberta-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_path", default="data/processed/train.jsonl")
    parser.add_argument("--val_path", default="data/processed/val.jsonl")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_root", default="outputs")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    model_dir = output_root / "models" / args.run_id
    pred_path = output_root / "preds" / f"{args.run_id}__{args.dataset_name}.jsonl"
    metrics_path = output_root / "metrics" / f"{args.run_id}__{args.dataset_name}.json"
    subset_path = output_root / "subsets" / f"{args.run_id}__{args.dataset_name}_subsets.json"
    efficiency_path = output_root / "efficiency" / f"{args.run_id}.json"
    use_lora = maybe_use_lora(args.model_name)
    start = time.time()

    if args.method in ABLATION_METHODS:
        train_cmd = [
            "python",
            "train_ablation.py",
            "--model_name",
            args.model_name,
            "--variant",
            args.method,
            "--train_path",
            args.train_path,
            "--val_path",
            args.val_path,
            "--output_dir",
            str(model_dir),
            "--seed",
            str(args.seed),
        ]
        if use_lora:
            train_cmd.append("--use_lora")
        run(train_cmd)
        infer_cmd = [
            "python",
            "run_inference_lora.py",
            "--model_type",
            "efsc",
            "--model_name",
            args.model_name,
            "--variant",
            args.method,
            "--checkpoint",
            str(model_dir / "best.pt"),
            "--input_path",
            args.test_path,
            "--output_path",
            str(pred_path),
        ]
        if use_lora:
            infer_cmd.append("--use_lora")
        run(infer_cmd)
    elif args.method == "direct_classifier":
        train_cmd = [
            "python",
            "train_direct_classifier.py",
            "--model_name",
            args.model_name,
            "--train_path",
            args.train_path,
            "--val_path",
            args.val_path,
            "--output_dir",
            str(model_dir),
            "--seed",
            str(args.seed),
        ]
        if use_lora:
            train_cmd.append("--use_lora")
        run(train_cmd)
        infer_cmd = [
            "python",
            "run_inference_lora.py",
            "--model_type",
            "direct",
            "--model_name",
            args.model_name,
            "--checkpoint",
            str(model_dir / "best_direct_classifier.pt"),
            "--input_path",
            args.test_path,
            "--output_path",
            str(pred_path),
        ]
        if use_lora:
            infer_cmd.append("--use_lora")
        run(infer_cmd)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    run(["python", "scripts/evaluate_maintrack.py", "--predictions", str(pred_path), "--output", str(metrics_path)])
    run(["python", "scripts/evaluate_subsets.py", "--predictions", str(pred_path), "--output", str(subset_path)])

    wall_time = time.time() - start
    efficiency_stub = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "method": args.method,
        "num_trainable_params": 0,
        "num_total_params": 0,
        "peak_gpu_mem_mb": 0.0,
        "train_wall_time_sec": wall_time,
        "tokens_per_sec": 0.0,
        "latency_ms_per_example": 0.0,
    }
    efficiency_path.parent.mkdir(parents=True, exist_ok=True)
    efficiency_path.write_text(json.dumps(efficiency_stub, indent=2), encoding="utf-8")
    print(f"Finished run {args.run_id}")


if __name__ == "__main__":
    main()
