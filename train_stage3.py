from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset_loader import ACTION_TO_ID, EFSCDataset
from src.models.efsc_model import EFSCModel
from src.train.collators import efsc_collate_fn
from src.train.losses import stage3_preference_loss
from src.train.utils import save_json, set_seed, to_device


def build_preferred_and_rejected_action_ids(batch: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    preferred_ids, rejected_ids = [], []
    for gold, preferred, rejected in zip(batch["action_label"].tolist(), batch["preferred_over"], batch["dispreferred_over"]):
        preferred_name = preferred[0] if preferred else None
        rejected_name = random.choice(rejected) if rejected else None
        preferred_ids.append(ACTION_TO_ID.get(preferred_name, gold))
        if rejected_name is None or rejected_name not in ACTION_TO_ID:
            candidates = [0, 1, 2, 3]
            candidates.remove(gold)
            rejected_ids.append(random.choice(candidates))
        else:
            rejected_ids.append(ACTION_TO_ID[rejected_name])
    return torch.tensor(preferred_ids, dtype=torch.long), torch.tensor(rejected_ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_ds = EFSCDataset(args.train_path, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=efsc_collate_fn)

    model = EFSCModel(backbone_name=args.model_name, freeze_backbone=args.freeze_backbone).to(device)
    model.load_state_dict(torch.load(args.stage2_ckpt, map_location=device))
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"])
            preferred_ids, rejected_ids = build_preferred_and_rejected_action_ids(batch)
            loss = stage3_preference_loss(
                action_logits=out.action_logits,
                preferred_action_ids=preferred_ids.to(device),
                rejected_action_ids=rejected_ids.to(device),
            )
            loss.backward()
            optimizer.step()
            batch_size = batch["input_ids"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        row = {"epoch": epoch + 1, "train_preference_loss": running / max(seen, 1)}
        history.append(row)
        print(row)
        if row["train_preference_loss"] < best_loss:
            best_loss = row["train_preference_loss"]
            torch.save(model.state_dict(), out_dir / "best_stage3.pt")
    torch.save(model.state_dict(), out_dir / "last_stage3.pt")
    save_json({"history": history}, out_dir / "train_log_stage3.json")


if __name__ == "__main__":
    main()
