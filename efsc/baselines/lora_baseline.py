from __future__ import annotations
import argparse
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from efsc.config import load_config
from efsc.constants import ACTION2ID
from efsc.data.dataset import PromptDataset
from efsc.utils import device_from_config, set_seed

class LoRAClassifier(torch.nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = 128):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        peft_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.backbone = get_peft_model(backbone, peft_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.head = torch.nn.Sequential(torch.nn.Linear(self.backbone.config.hidden_size, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, len(ACTION2ID)))
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state
        pooled = (last * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); set_seed(config.get("seed", 42)); device = device_from_config(config)
    model = LoRAClassifier(config["model_name"]).to(device); ds = PromptDataset(config["train_path"]); tok = model.tokenizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config.get("learning_rate", 1e-4)))
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        for row in tqdm(ds, desc=f"lora epoch {epoch+1}"):
            encoded = tok(row["prompt"], return_tensors="pt", truncation=True, max_length=int(config.get("max_length", 512))).to(device)
            label = torch.tensor([row["action_id"]], device=device)
            loss = torch.nn.functional.cross_entropy(model(**encoded), label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), output_dir / "lora_baseline.pt")

if __name__ == "__main__":
    main()
