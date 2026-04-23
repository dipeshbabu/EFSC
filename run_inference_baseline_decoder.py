from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import EFSCDataset, ID_TO_ACTION
from src.models.baseline_factory import build_baseline_model
from src.train.collators import efsc_collate_fn
from src.train.tokenizer_utils import load_tokenizer
from src.train.utils import to_device


def save_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_type", required=True, choices=["direct_policy", "plain_efsc"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_name)
    ds = EFSCDataset(args.input_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

    model = build_baseline_model(args.model_name, args.baseline_type, use_lora=args.use_lora).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    rows = []
    for batch in loader:
        meta = {
            "id": batch["id"],
            "prompt": batch["prompt"],
            "topic": batch["topic"],
            "family_id": batch["family_id"],
            "variant_type": batch["variant_type"],
            "uncertainty_label": batch["uncertainty_label_text"],
            "gold_action": batch["gold_action_text"],
        }
        batch = to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        pred_ids = out.action_logits.argmax(dim=-1).detach().cpu().tolist()

        harm_ids = legit_ids = unc_ids = None
        if args.baseline_type == "plain_efsc":
            harm_ids = out.harm_logits.argmax(dim=-1).detach().cpu().tolist()
            legit_ids = out.legit_logits.argmax(dim=-1).detach().cpu().tolist()
            unc_ids = out.uncertainty_logits.argmax(dim=-1).detach().cpu().tolist()

        for i, pred_id in enumerate(pred_ids):
            row = {
                "id": meta["id"][i],
                "prompt": meta["prompt"][i],
                "topic": meta["topic"][i],
                "family_id": meta["family_id"][i],
                "variant_type": meta["variant_type"][i],
                "uncertainty_label": meta["uncertainty_label"][i],
                "gold_action": meta["gold_action"][i],
                "pred_action": ID_TO_ACTION[pred_id],
            }
            if harm_ids is not None and legit_ids is not None and unc_ids is not None:
                row["pred_harm_id"] = harm_ids[i]
                row["pred_legit_id"] = legit_ids[i]
                row["pred_uncertainty_id"] = unc_ids[i]
            rows.append(row)

    save_jsonl(rows, Path(args.output_path))
    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
