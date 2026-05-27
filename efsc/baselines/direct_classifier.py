from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from efsc.config import load_config
from efsc.data.collator import PromptCollator
from efsc.data.dataset import PromptDataset
from efsc.eval.metrics import classification_metrics
from efsc.modeling import DirectActionClassifier
from efsc.utils import device_from_config, move_batch_to_device, save_json, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); set_seed(config.get("seed", 42)); device = device_from_config(config)
    model = DirectActionClassifier(config["model_name"], freeze_backbone=bool(config.get("freeze_backbone", True))).to(device)
    train_ds, val_ds = PromptDataset(config["train_path"]), PromptDataset(config["val_path"])
    collator = PromptCollator(model.tokenizer, max_length=int(config.get("max_length", 512)))
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", 8)), shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config.get("learning_rate", 2e-4)))
    best = -1.0; output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config.get("epochs", 3))):
        model.train()
        for batch in tqdm(train_loader, desc=f"direct epoch {epoch+1}"):
            batch = move_batch_to_device(batch, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(logits, batch["action_labels"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                preds = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).argmax(dim=-1)
                y_true.extend(batch["action_labels"].cpu().tolist()); y_pred.extend(preds.cpu().tolist())
        metrics = classification_metrics(y_true, y_pred)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]; torch.save(model.state_dict(), output_dir / "best.pt"); save_json(metrics, output_dir / "best_metrics.json")

if __name__ == "__main__":
    main()
