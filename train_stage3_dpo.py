from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.model_factory import build_efsc_model
from src.train.dpo_loss import dpo_loss
from src.train.preference_dataset import PreferenceDataset, preference_collate_fn
from src.train.utils import save_json, set_seed, to_device


@torch.no_grad()
def evaluate(policy_model, ref_model, loader, device: torch.device, beta: float) -> Dict[str, float]:
    policy_model.eval()
    ref_model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        pi_chosen = policy_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        pi_rejected = policy_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        ref_chosen = ref_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        ref_rejected = ref_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        loss = dpo_loss(
            policy_chosen_logits=pi_chosen.action_logits,
            policy_rejected_logits=pi_rejected.action_logits,
            ref_chosen_logits=ref_chosen.action_logits,
            ref_rejected_logits=ref_rejected.action_logits,
            chosen_action_ids=batch["chosen_action_id"],
            beta=beta,
        )
        batch_size = batch["chosen_input_ids"].size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
    return {"val_dpo_loss": total_loss / max(n, 1)}


def freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--init_ckpt", required=True, help="Stage 1 or Stage 2 checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_ds = PreferenceDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_ds = PreferenceDataset(args.val_path, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=preference_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=preference_collate_fn)

    policy_model = build_efsc_model(
        backbone_name=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    policy_model.load_state_dict(torch.load(args.init_ckpt, map_location=device), strict=False)

    ref_model = build_efsc_model(backbone_name=args.model_name, use_lora=False).to(device)
    ref_model.load_state_dict(torch.load(args.init_ckpt, map_location=device), strict=False)
    freeze_model(ref_model)

    optimizer = AdamW([parameter for parameter in policy_model.parameters() if parameter.requires_grad], lr=args.lr)

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        policy_model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            pi_chosen = policy_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            pi_rejected = policy_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            with torch.no_grad():
                ref_chosen = ref_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                ref_rejected = ref_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = dpo_loss(
                policy_chosen_logits=pi_chosen.action_logits,
                policy_rejected_logits=pi_rejected.action_logits,
                ref_chosen_logits=ref_chosen.action_logits,
                ref_rejected_logits=ref_rejected.action_logits,
                chosen_action_ids=batch["chosen_action_id"],
                beta=args.beta,
            )
            loss.backward()
            optimizer.step()
            batch_size = batch["chosen_input_ids"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        row = {
            "epoch": epoch + 1,
            "train_dpo_loss": running / max(seen, 1),
            **evaluate(policy_model, ref_model, val_loader, device, args.beta),
        }
        history.append(row)
        print(row)
        if row["val_dpo_loss"] < best_val:
            best_val = row["val_dpo_loss"]
            torch.save(policy_model.state_dict(), out_dir / "best_stage3_dpo.pt")

    torch.save(policy_model.state_dict(), out_dir / "last_stage3_dpo.pt")
    save_json({"history": history}, out_dir / "train_log_stage3_dpo.json")


if __name__ == "__main__":
    main()
