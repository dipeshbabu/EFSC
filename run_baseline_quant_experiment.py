from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_if_missing(target: Path, cmd: list[str]) -> None:
    if target.exists():
        print(f"SKIP: {target} already exists")
        return
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_type", required=True, choices=["direct_policy", "plain_efsc"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_path", default="data/processed/train.jsonl")
    parser.add_argument("--val_path", default="data/processed/val.jsonl")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--quant_mode", required=True, choices=["int8", "int4"])
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    model_dir = out_root / "models" / args.run_id
    best_ckpt = model_dir / "best.pt"
    fp_pred = out_root / "preds" / f"{args.run_id}__fp__{args.dataset_name}.jsonl"
    fp_metrics = out_root / "metrics" / f"{args.run_id}__fp__{args.dataset_name}.json"
    q_pred = out_root / "preds" / f"{args.run_id}__{args.quant_mode}__{args.dataset_name}.jsonl"
    q_metrics = out_root / "metrics" / f"{args.run_id}__{args.quant_mode}__{args.dataset_name}.json"
    retention = out_root / "retention" / f"{args.run_id}__{args.quant_mode}__{args.dataset_name}.json"
    recal_dir = out_root / "recalibrated" / f"{args.run_id}__{args.quant_mode}"
    recal_ckpt = recal_dir / "best_recalibrated.pt"
    qr_pred = out_root / "preds" / f"{args.run_id}__{args.quant_mode}_recal__{args.dataset_name}.jsonl"
    qr_metrics = out_root / "metrics" / f"{args.run_id}__{args.quant_mode}_recal__{args.dataset_name}.json"
    retention_recal = out_root / "retention" / f"{args.run_id}__{args.quant_mode}_recal__{args.dataset_name}.json"

    train_cmd = [
        "python", "train_baseline_decoder.py",
        "--model_name", args.model_name,
        "--baseline_type", args.baseline_type,
        "--train_path", args.train_path,
        "--val_path", args.val_path,
        "--output_dir", str(model_dir),
        "--seed", str(args.seed),
        *(["--use_lora"] if args.use_lora else []),
    ]
    if args.force_train and best_ckpt.exists():
        run(train_cmd)
    else:
        run_if_missing(best_ckpt, train_cmd)

    run_if_missing(fp_pred, [
        "python", "run_inference_baseline_decoder.py",
        "--model_name", args.model_name,
        "--baseline_type", args.baseline_type,
        "--checkpoint", str(best_ckpt),
        "--input_path", args.test_path,
        "--output_path", str(fp_pred),
        *(["--use_lora"] if args.use_lora else []),
    ])
    run_if_missing(fp_metrics, ["python", "scripts/evaluate_maintrack.py", "--predictions", str(fp_pred), "--output", str(fp_metrics)])
    run_if_missing(q_pred, [
        "python", "run_quantized_inference_baseline_decoder.py",
        "--model_name", args.model_name,
        "--baseline_type", args.baseline_type,
        "--quant_mode", args.quant_mode,
        "--controller_checkpoint", str(best_ckpt),
        "--input_path", args.test_path,
        "--output_path", str(q_pred),
        *(["--use_lora"] if args.use_lora else []),
    ])
    run_if_missing(q_metrics, ["python", "scripts/evaluate_maintrack.py", "--predictions", str(q_pred), "--output", str(q_metrics)])
    run_if_missing(retention, [
        "python", "scripts/evaluate_quant_retention.py",
        "--fp_metrics", str(fp_metrics),
        "--quant_metrics", str(q_metrics),
        "--fp_preds", str(fp_pred),
        "--quant_preds", str(q_pred),
        "--output", str(retention),
    ])

    if args.recalibrate:
        run_if_missing(recal_ckpt, [
            "python", "recalibrate_quantized_baseline.py",
            "--model_name", args.model_name,
            "--baseline_type", args.baseline_type,
            "--quant_mode", args.quant_mode,
            "--controller_checkpoint", str(best_ckpt),
            "--train_path", args.train_path,
            "--val_path", args.val_path,
            "--output_dir", str(recal_dir),
            *(["--use_lora"] if args.use_lora else []),
        ])
        run_if_missing(qr_pred, [
            "python", "run_quantized_inference_baseline_decoder.py",
            "--model_name", args.model_name,
            "--baseline_type", args.baseline_type,
            "--quant_mode", args.quant_mode,
            "--controller_checkpoint", str(recal_ckpt),
            "--input_path", args.test_path,
            "--output_path", str(qr_pred),
            *(["--use_lora"] if args.use_lora else []),
        ])
        run_if_missing(qr_metrics, ["python", "scripts/evaluate_maintrack.py", "--predictions", str(qr_pred), "--output", str(qr_metrics)])
        run_if_missing(retention_recal, [
            "python", "scripts/evaluate_quant_retention.py",
            "--fp_metrics", str(fp_metrics),
            "--quant_metrics", str(qr_metrics),
            "--fp_preds", str(fp_pred),
            "--quant_preds", str(qr_pred),
            "--output", str(retention_recal),
        ])

    eff_src = model_dir / "efficiency.json"
    if eff_src.exists():
        eff_dst = out_root / "efficiency" / f"{args.run_id}.json"
        eff_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eff_src, eff_dst)

    print(f"Completed baseline experiment {args.run_id} on {args.dataset_name}")


if __name__ == "__main__":
    main()
