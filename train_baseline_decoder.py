from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset_loader import EFSCDataset
from src.models.baseline_factory import build_baseline_model
from src.train.ablation_losses import ablation_supervised_loss
from src.train.collators import efsc_collate_fn
from src.train.efficiency_utils import EfficiencyTracker, count_parameters
from src.train.tokenizer_utils import load_tokenizer
from src.train.utils import save_json, set_seed, to_device


def compute_loss(out, batch, baseline_type: str):
    if baseline_type == "direct_policy":
        return F.cross_entropy(out.action_logits, batch["action_label"])
    if baseline_type == "plain_efsc":
        return ablation_supervised_loss(
            out.harm_logits,
            out.legit_logits,
            out.uncertainty_logits,
            out.action_logits,
            batch["harm_label"],
            batch["legit_label"],
            batch["uncertainty_label"],
            batch["action_label"],
        )
    raise ValueError(baseline_type)


@torch.no_grad()
def evaluate(model, loader, device, baseline_type: str):
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_loss(out, batch, baseline_type)
        bs = batch["input_ids"].size(0)
        total += loss.item() * bs
        seen += bs
    return {"val_loss": total / max(seen, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_type", required=True, choices=["direct_policy", "plain_efsc"])
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    train_ds = EFSCDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_ds = EFSCDataset(args.val_path, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=efsc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

    model = build_baseline_model(
        model_name=args.model_name,
        baseline_type=args.baseline_type,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    params = count_parameters(model)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    tracker = EfficiencyTracker.start()
    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_train = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(out, batch, args.baseline_type)
            loss.backward()
            optimizer.step()
            scheduler.step()
            bs = batch["input_ids"].size(0)
            total_train += loss.item() * bs
            seen += bs
            tracker.update_batch(batch["input_ids"])

        row = {"epoch": epoch + 1, "train_loss": total_train / max(seen, 1), **evaluate(model, val_loader, device, args.baseline_type)}
        history.append(row)
        print(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    save_json({"history": history}, out_dir / "train_log.json")
    save_json({**params, **tracker.finish()}, out_dir / "efficiency.json")


if __name__ == "__main__":
    main()
