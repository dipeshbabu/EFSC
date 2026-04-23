from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import EFSCDataset, ID_TO_ACTION
from src.models.decoder_qefsc_fc_quantized import QuantizedDecoderQEFSCFCModel
from src.train.checkpoint_utils import load_non_backbone_checkpoint
from src.train.collators import efsc_collate_fn
from src.train.tokenizer_utils import load_tokenizer


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--quant_mode", required=True, choices=["fp16", "bf16", "int8", "int4"])
    parser.add_argument("--controller_checkpoint", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name)
    ds = EFSCDataset(args.input_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

    model = QuantizedDecoderQEFSCFCModel(
        backbone_name=args.model_name,
        quant_mode=args.quant_mode,
        layer_indices=[-1, -4, -8, -12],
        fusion_type="gated",
        fused_dim=256,
        head_hidden_dim=256,
        use_lora=args.use_lora,
    )
    load_non_backbone_checkpoint(model, args.controller_checkpoint, include_lora=args.use_lora)
    model.eval()

    rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(model.input_device)
        attention_mask = batch["attention_mask"].to(model.input_device)
        out = model(input_ids, attention_mask)
        pred_ids = out.action_logits.argmax(dim=-1).detach().cpu().tolist()
        harm_ids = out.harm_logits.argmax(dim=-1).detach().cpu().tolist()
        legit_ids = out.legit_logits.argmax(dim=-1).detach().cpu().tolist()
        unc_ids = out.uncertainty_logits.argmax(dim=-1).detach().cpu().tolist()

        for i, pred_id in enumerate(pred_ids):
            rows.append({
                "id": batch["id"][i],
                "prompt": batch["prompt"][i],
                "topic": batch["topic"][i],
                "family_id": batch["family_id"][i],
                "variant_type": batch["variant_type"][i],
                "uncertainty_label": batch["uncertainty_label_text"][i],
                "gold_action": batch["gold_action_text"][i],
                "pred_action": ID_TO_ACTION[pred_id],
                "pred_harm_id": harm_ids[i],
                "pred_legit_id": legit_ids[i],
                "pred_uncertainty_id": unc_ids[i],
                "quant_mode": args.quant_mode,
            })

    save_jsonl(rows, Path(args.output_path))
    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
