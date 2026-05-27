from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset_loader import EFSCDataset
from src.models.ablation_factory import build_ablation_model
from src.train.ablation_losses import ablation_supervised_loss
from src.train.collators import efsc_collate_fn
from src.train.utils import save_json, set_seed, to_device


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        loss = ablation_supervised_loss(
            out.harm_logits,
            out.legit_logits,
            out.uncertainty_logits,
            out.action_logits,
            batch["harm_label"],
            batch["legit_label"],
            batch["uncertainty_label"],
            batch["action_label"],
        )
        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
    return {"val_loss": total_loss / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true")
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
    val_ds = EFSCDataset(args.val_path, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=efsc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

    model = build_ablation_model(backbone_name=args.model_name, variant=args.variant, use_lora=args.use_lora).to(device)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = ablation_supervised_loss(
                out.harm_logits,
                out.legit_logits,
                out.uncertainty_logits,
                out.action_logits,
                batch["harm_label"],
                batch["legit_label"],
                batch["uncertainty_label"],
                batch["action_label"],
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_size = batch["input_ids"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), **evaluate(model, val_loader, device)}
        history.append(row)
        print(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "last.pt")
    save_json({"history": history}, out_dir / "train_log.json")


if __name__ == "__main__":
    main()
