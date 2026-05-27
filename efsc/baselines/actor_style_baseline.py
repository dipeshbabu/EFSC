from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from efsc.config import load_config
from efsc.data.dataset import PromptDataset
from efsc.modeling import EFSCModel
from efsc.utils import device_from_config, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); set_seed(config.get("seed", 42)); device = device_from_config(config)
    model = EFSCModel(model_name=config["model_name"], bottleneck_dim=int(config.get("bottleneck_dim", 128)), freeze_backbone=True).to(device)
    ds = PromptDataset(config["train_path"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config.get("learning_rate", 2e-4)))
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config.get("epochs", 2))):
        model.train()
        for row in tqdm(ds, desc=f"actor style epoch {epoch+1}"):
            encoded = model.tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=int(config.get("max_length", 512))).to(device)
            outputs = model(**encoded)
            loss = torch.nn.functional.cross_entropy(outputs.action_logits, torch.tensor([row["action_id"]], device=device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), output_dir / "actor_style_baseline.pt")

if __name__ == "__main__":
    main()
