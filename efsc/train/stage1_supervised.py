from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from efsc.data.collator import PromptCollator
from efsc.data.dataset import PromptDataset
from efsc.eval.metrics import classification_metrics, selective_refusal_metrics
from efsc.losses import supervised_loss
from efsc.modeling import EFSCModel
from efsc.train.common import build_optimizer_scheduler, parse_args, selection_score, setup_run, write_history_csv
from efsc.utils import move_batch_to_device, save_json

def evaluate(model: EFSCModel, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred, benign_mask, harmful_mask = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = outputs.action_logits.argmax(dim=-1)
            y_true.extend(batch["action_labels"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            harms = batch["harm_labels"].cpu().tolist()
            benign_mask.extend([h < 2 for h in harms])
            harmful_mask.extend([h == 2 for h in harms])
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(selective_refusal_metrics(y_true, y_pred, benign_mask, harmful_mask))
    return metrics

def main() -> None:
    args = parse_args()
    config, device, output_dir = setup_run(args.config)
    model = EFSCModel(model_name=config["model_name"], bottleneck_dim=int(config.get("bottleneck_dim", 128)), dropout=float(config.get("dropout", 0.1)), freeze_backbone=bool(config.get("freeze_backbone", True)), action_hidden_dim=int(config.get("action_hidden_dim", 128))).to(device)
    if config.get("resume_from_checkpoint"):
        model.load_state_dict(torch.load(config["resume_from_checkpoint"], map_location=device), strict=False)
    train_ds, val_ds = PromptDataset(config["train_path"]), PromptDataset(config["val_path"])
    collator = PromptCollator(model.tokenizer, max_length=int(config.get("max_length", 512)))
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", 8)), shuffle=False, collate_fn=collator)
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    total_steps = int(config.get("epochs", 3)) * max(1, math.ceil(len(train_loader) / grad_accum))
    optimizer, scheduler = build_optimizer_scheduler(model, config, total_steps)
    weights = config.get("loss_weights", {"harm": 1.0, "legit": 1.0, "uncertainty": 1.0})
    best_metric, bad_epochs, history = -1.0, 0, []
    patience = int(config.get("early_stopping_patience", 0))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("mixed_precision", False)) and device.type == "cuda")
    for epoch in range(int(config.get("epochs", 3))):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"stage1 epoch {epoch+1}"), start=1):
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss, _ = supervised_loss(outputs, batch["action_labels"], batch["harm_labels"], batch["legit_labels"], batch["uncertainty_labels"], weights)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
        metrics = evaluate(model, val_loader, device); metrics["epoch"] = epoch + 1; metrics["selection_score"] = selection_score(metrics); history.append(metrics)
        torch.save(model.state_dict(), Path(output_dir) / "stage1_last.pt")
        if metrics["selection_score"] > best_metric:
            best_metric = metrics["selection_score"]; bad_epochs = 0; torch.save(model.state_dict(), Path(output_dir) / "stage1_best.pt")
        else:
            bad_epochs += 1
        save_json({"history": history}, Path(output_dir) / "stage1_history.json")
        write_history_csv(history, Path(output_dir) / "stage1_history.csv")
        if patience and bad_epochs >= patience:
            break

if __name__ == "__main__":
    main()
