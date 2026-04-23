from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset_loader import EFSCDataset
from src.models.decoder_qefsc_fc_quantized import QuantizedDecoderQEFSCFCModel
from src.train.ablation_losses import ablation_supervised_loss
from src.train.checkpoint_utils import load_non_backbone_checkpoint
from src.train.collators import efsc_collate_fn
from src.train.tokenizer_utils import load_tokenizer
from src.train.utils import save_json


@torch.no_grad()
def evaluate(model, loader) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.input_device)
        attention_mask = batch["attention_mask"].to(model.input_device)
        harm = batch["harm_label"].to(model.input_device)
        legit = batch["legit_label"].to(model.input_device)
        unc = batch["uncertainty_label"].to(model.input_device)
        action = batch["action_label"].to(model.input_device)
        out = model(input_ids, attention_mask)
        loss = ablation_supervised_loss(out.harm_logits, out.legit_logits, out.uncertainty_logits, out.action_logits, harm, legit, unc, action)
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        n += bs
    return {"val_loss": total_loss / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--quant_mode", required=True, choices=["int8", "int4"])
    parser.add_argument("--controller_checkpoint", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name)
    train_ds = EFSCDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_ds = EFSCDataset(args.val_path, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=efsc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=efsc_collate_fn)

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

    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("backbone.")

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_train = 0.0
        seen = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(model.input_device)
            attention_mask = batch["attention_mask"].to(model.input_device)
            harm = batch["harm_label"].to(model.input_device)
            legit = batch["legit_label"].to(model.input_device)
            unc = batch["uncertainty_label"].to(model.input_device)
            action = batch["action_label"].to(model.input_device)
            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss = ablation_supervised_loss(out.harm_logits, out.legit_logits, out.uncertainty_logits, out.action_logits, harm, legit, unc, action)
            loss.backward()
            optimizer.step()
            bs = input_ids.size(0)
            total_train += loss.item() * bs
            seen += bs

        row = {"epoch": epoch + 1, "train_loss": total_train / max(seen, 1), **evaluate(model, val_loader)}
        history.append(row)
        print(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(model.state_dict(), out_dir / "best_recalibrated.pt")

    save_json({"history": history}, out_dir / "recalibration_log.json")


if __name__ == "__main__":
    main()
