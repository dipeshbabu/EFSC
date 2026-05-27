from __future__ import annotations

import argparse
import csv
from pathlib import Path


def quote(value: str) -> str:
    if any(ch.isspace() for ch in value):
        return f'"{value}"'
    return value


def build_command(row: dict, train_path: str, val_path: str, output_root: str) -> str:
    run_id = row["run_id"]
    model_name = row["model_name"]
    method = row["method"]
    seed = row["seed"]
    out_dir = f"{output_root}/{run_id}"

    if method == "direct_classifier":
        use_lora = model_name != "roberta-base"
        lora_flag = " --use_lora" if use_lora else ""
        return (
            f"python train_direct_classifier.py "
            f"--model_name {quote(model_name)} "
            f"--train_path {quote(train_path)} "
            f"--val_path {quote(val_path)} "
            f"--output_dir {quote(out_dir)} "
            f"--seed {seed}{lora_flag}"
        )

    if method == "efsc_stage1":
        use_lora = model_name != "roberta-base"
        script = "train_stage1_lora.py" if use_lora else "train_stage1.py"
        lora_flag = " --use_lora" if use_lora else ""
        return (
            f"python {script} "
            f"--model_name {quote(model_name)} "
            f"--train_path {quote(train_path)} "
            f"--val_path {quote(val_path)} "
            f"--output_dir {quote(out_dir)} "
            f"--seed {seed}{lora_flag}"
        )

    if method == "efsc_stage1_stage2":
        stage1_dir = f"{output_root}/{model_name}__efsc_stage1__seed{seed}"
        return (
            f"python train_stage2.py "
            f"--model_name {quote(model_name)} "
            f"--train_path {quote(train_path)} "
            f"--stage1_ckpt {quote(stage1_dir + '/best_stage1_lora.pt' if model_name != 'roberta-base' else stage1_dir + '/best_stage1.pt')} "
            f"--output_dir {quote(out_dir)} "
            f"--seed {seed}"
        )

    if method == "efsc_full":
        use_lora = model_name != "roberta-base"
        lora_flag = " --use_lora" if use_lora else ""
        init_ckpt = f"{output_root}/{model_name}__efsc_stage1_stage2__seed{seed}/best_stage2.pt"
        return (
            f"python train_stage3_dpo.py "
            f"--model_name {quote(model_name)} "
            f"--train_path data/processed/train.pref.jsonl "
            f"--val_path data/processed/val.pref.jsonl "
            f"--init_ckpt {quote(init_ckpt)} "
            f"--output_dir {quote(out_dir)} "
            f"--seed {seed}{lora_flag}"
        )

    if method == "efsc_no_preference":
        return f"# {run_id}: use stage2 checkpoint directly for no-preference ablation"
    if method == "efsc_no_counterfactual":
        return f"# {run_id}: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation"
    if method in {"efsc_no_harm_head", "efsc_no_legit_head", "efsc_no_uncertainty_head"}:
        return f"# {run_id}: architecture ablation requires corresponding model switch"

    return f"# unsupported method for now: {run_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", required=True)
    parser.add_argument("--train_path", default="data/processed/train.jsonl")
    parser.add_argument("--val_path", default="data/processed/val.jsonl")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--script_output", required=True)
    args = parser.parse_args()

    with Path(args.registry_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    commands = [build_command(row, args.train_path, args.val_path, args.output_root) for row in rows if row["status"] == "planned"]
    output = Path(args.script_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {output}")


if __name__ == "__main__":
    main()
