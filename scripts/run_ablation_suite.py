from __future__ import annotations

import argparse
from pathlib import Path

ABLATIONS = [
    "efsc_full",
    "efsc_no_harm_head",
    "efsc_no_legit_head",
    "efsc_no_uncertainty_head",
    "efsc_no_counterfactual",
    "efsc_no_dpo",
    "efsc_shared_head",
    "efsc_no_factor_conditioning",
    "efsc_no_bottleneck",
    "efsc_small_bottleneck",
    "efsc_large_bottleneck",
]


def build_command(model_name: str, method: str, seed: int, output_root: str) -> str:
    out_dir = f"{output_root}/{model_name}__{method}__seed{seed}"
    use_lora = " --use_lora" if model_name != "roberta-base" else ""
    if method in {"efsc_no_dpo", "efsc_no_counterfactual"}:
        return (
            f"python train_ablation.py --model_name {model_name} --variant efsc_full "
            f"--train_path data/processed/train.jsonl --val_path data/processed/val.jsonl "
            f"--output_dir {out_dir} --seed {seed}{use_lora}"
        )
    return (
        f"python train_ablation.py --model_name {model_name} --variant {method} "
        f"--train_path data/processed/train.jsonl --val_path data/processed/val.jsonl "
        f"--output_dir {out_dir} --seed {seed}{use_lora}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--script_output", required=True)
    args = parser.parse_args()
    lines = [build_command(args.model_name, method, int(seed), args.output_root) for method in ABLATIONS for seed in args.seeds]
    output = Path(args.script_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote ablation commands to {output}")


if __name__ == "__main__":
    main()
