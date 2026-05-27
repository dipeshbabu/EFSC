from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset_loader import EFSCDataset
from src.models.ablation_factory import build_ablation_model
from src.models.model_factory import build_direct_model, build_efsc_model
from src.train.collators import efsc_collate_fn
from src.train.utils import to_device

ID_TO_ACTION = {0: "ANSWER", 1: "SAFE_ANSWER", 2: "CLARIFY", 3: "REFUSE"}


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["efsc", "direct"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--variant", default="efsc_full")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = EFSCDataset(args.input_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

    if args.model_type == "efsc":
        if args.variant == "efsc_full":
            model = build_efsc_model(
                backbone_name=args.model_name,
                use_lora=args.use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            ).to(device)
        else:
            model = build_ablation_model(
                backbone_name=args.model_name,
                variant=args.variant,
                use_lora=args.use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            ).to(device)
    else:
        model = build_direct_model(
            backbone_name=args.model_name,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    rows = []
    for batch in loader:
        metadata = {
            "id": batch["id"],
            "prompt": batch["prompt"],
            "topic": batch["topic"],
            "family_id": batch["family_id"],
            "variant_type": batch["variant_type"],
            "uncertainty_label": batch["uncertainty_label_name"],
            "gold_action": [ID_TO_ACTION[int(label.item())] for label in batch["action_label"]],
        }
        batch = to_device(batch, device)
        output = model(batch["input_ids"], batch["attention_mask"])
        logits = output.action_logits if args.model_type == "efsc" else output["action_logits"]
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for index, pred_id in enumerate(pred_ids):
            rows.append({
                "id": metadata["id"][index],
                "prompt": metadata["prompt"][index],
                "topic": metadata["topic"][index],
                "family_id": metadata["family_id"][index],
                "variant_type": metadata["variant_type"][index],
                "uncertainty_label": metadata["uncertainty_label"][index],
                "gold_action": metadata["gold_action"][index],
                "pred_action": ID_TO_ACTION[pred_id],
            })

    save_jsonl(rows, Path(args.output_path))
    print(f"Wrote {len(rows)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
