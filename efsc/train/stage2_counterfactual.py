from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from efsc.data.collator import PairCollator
from efsc.data.dataset import CounterfactualPairDataset
from efsc.losses import consistency_loss, margin_ranking_loss
from efsc.modeling import EFSCModel
from efsc.train.common import build_optimizer_scheduler, parse_args, setup_run, write_history_csv
from efsc.utils import move_batch_to_device, save_json

def main() -> None:
    args = parse_args()
    config, device, output_dir = setup_run(args.config)
    model = EFSCModel(model_name=config["model_name"], bottleneck_dim=int(config.get("bottleneck_dim", 128)), dropout=float(config.get("dropout", 0.1)), freeze_backbone=bool(config.get("freeze_backbone", True)), action_hidden_dim=int(config.get("action_hidden_dim", 128))).to(device)
    if config.get("init_checkpoint"): model.load_state_dict(torch.load(config["init_checkpoint"], map_location=device), strict=False)
    ds = CounterfactualPairDataset(config["pair_path"])
    collator = PairCollator(model.tokenizer, max_length=int(config.get("max_length", 512)))
    loader = DataLoader(ds, batch_size=int(config.get("batch_size", 8)), shuffle=True, collate_fn=collator)
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    total_steps = int(config.get("epochs", 3)) * max(1, math.ceil(len(loader) / grad_accum))
    optimizer, scheduler = build_optimizer_scheduler(model, config, total_steps)
    margin = float(config.get("margin", 0.5)); history = []; best_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("mixed_precision", False)) and device.type == "cuda")
    for epoch in range(int(config.get("epochs", 3))):
        running = []
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"stage2 epoch {epoch+1}"), start=1):
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                harm_out = model(input_ids=batch["harmful_input_ids"], attention_mask=batch["harmful_attention_mask"])
                benign_out = model(input_ids=batch["benign_input_ids"], attention_mask=batch["benign_attention_mask"])
                harm_rank = margin_ranking_loss(F.softmax(harm_out.harm_logits, dim=-1)[:, 2], F.softmax(benign_out.harm_logits, dim=-1)[:, 2], margin)
                legit_rank = margin_ranking_loss(F.softmax(benign_out.legit_logits, dim=-1)[:, 2], F.softmax(harm_out.legit_logits, dim=-1)[:, 2], margin)
                cons = consistency_loss(F.softmax(harm_out.uncertainty_logits, dim=-1), F.softmax(benign_out.uncertainty_logits, dim=-1))
                loss = (harm_rank + legit_rank + float(config.get("consistency_weight", 0.25)) * cons) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
            running.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "loss": sum(running) / max(1, len(running))})
        save_json({"history": history}, Path(output_dir) / "stage2_history.json")
        write_history_csv(history, Path(output_dir) / "stage2_history.csv")
        torch.save(model.state_dict(), Path(output_dir) / "stage2_last.pt")
        if history[-1]["loss"] < best_loss:
            best_loss = history[-1]["loss"]; torch.save(model.state_dict(), Path(output_dir) / "stage2_best.pt")

if __name__ == "__main__":
    main()
