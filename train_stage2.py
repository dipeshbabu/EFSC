from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset_loader import EFSCDataset
from src.models.efsc_model import EFSCModel
from src.train.collators import efsc_collate_fn
from src.train.losses import stage2_counterfactual_loss
from src.train.utils import save_json, set_seed, to_device


def make_pairs(batch: Dict[str, torch.Tensor | List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    families = batch["family_id"]
    action_labels = batch["action_label"]
    pair_a, pair_b, same_family, same_action = [], [], [], []
    for i, family in enumerate(families):
        j = random.randrange(len(families))
        pair_a.append(i)
        pair_b.append(j)
        same_family.append(1 if family == families[j] and family != "" else 0)
        same_action.append(1 if int(action_labels[i].item()) == int(action_labels[j].item()) else 0)
    return (
        torch.tensor(pair_a, dtype=torch.long),
        torch.tensor(pair_b, dtype=torch.long),
        torch.tensor(same_family, dtype=torch.float),
        torch.tensor(same_action, dtype=torch.float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
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
    model.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
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
            idx_a, idx_b, same_family, same_action = make_pairs(batch)
            idx_a = idx_a.to(device)
            idx_b = idx_b.to(device)
            loss = stage2_counterfactual_loss(
                z_a=out.bottleneck[idx_a],
                z_b=out.bottleneck[idx_b],
                same_family=same_family.to(device),
                same_action=same_action.to(device),
            )
            loss.backward()
            optimizer.step()
            batch_size = batch["input_ids"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        row = {"epoch": epoch + 1, "train_counterfactual_loss": running / max(seen, 1)}
        history.append(row)
        print(row)
        if row["train_counterfactual_loss"] < best_loss:
            best_loss = row["train_counterfactual_loss"]
            torch.save(model.state_dict(), out_dir / "best_stage2.pt")
    torch.save(model.state_dict(), out_dir / "last_stage2.pt")
    save_json({"history": history}, out_dir / "train_log_stage2.json")


if __name__ == "__main__":
    main()
