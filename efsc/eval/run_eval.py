from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from efsc.config import load_config
from efsc.constants import ACTION2ID, ID2ACTION
from efsc.data.collator import PromptCollator
from efsc.data.dataset import PromptDataset
from efsc.eval.metrics import classification_metrics, selective_refusal_metrics
from efsc.modeling import EFSCModel
from efsc.utils import count_trainable_parameters, device_from_config, move_batch_to_device, save_json, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); set_seed(config.get("seed", 42)); device = device_from_config(config)
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    model = EFSCModel(model_name=config["model_name"], bottleneck_dim=int(config.get("bottleneck_dim", 128)), freeze_backbone=bool(config.get("freeze_backbone", True))).to(device)
    if config.get("checkpoint"): model.load_state_dict(torch.load(config["checkpoint"], map_location=device), strict=False)
    ds = PromptDataset(config["test_path"]); collator = PromptCollator(model.tokenizer, max_length=int(config.get("max_length", 512)))
    loader = DataLoader(ds, batch_size=int(config.get("batch_size", 8)), shuffle=False, collate_fn=collator)
    y_true, y_pred, benign_mask, harmful_mask, rows = [], [], [], [], []
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = move_batch_to_device(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = outputs.action_logits.argmax(dim=-1)
            probs = torch.softmax(outputs.action_logits, dim=-1)
            y_true.extend(batch["action_labels"].cpu().tolist()); y_pred.extend(preds.cpu().tolist())
            harms = batch["harm_labels"].cpu().tolist()
            benign_mask.extend([h < 2 for h in harms]); harmful_mask.extend([h == 2 for h in harms])
            for i, ex_id in enumerate(batch["ids"]):
                gold_action = int(batch["action_labels"][i].cpu().item())
                pred_action = int(preds[i].cpu().item())
                rows.append({
                    "id": ex_id,
                    "family_id": batch["family_ids"][i],
                    "source": batch["sources"][i],
                    "variant_type": batch["variant_types"][i],
                    "language": batch["languages"][i],
                    "topic": batch["topics"][i],
                    "gold_action": gold_action,
                    "pred_action": pred_action,
                    "gold_action_label": ID2ACTION[gold_action],
                    "pred_action_label": ID2ACTION[pred_action],
                    "answer_prob": float(probs[i, ACTION2ID["ANSWER"]].cpu().item()),
                    "safe_answer_prob": float(probs[i, ACTION2ID["SAFE_ANSWER"]].cpu().item()),
                    "clarify_prob": float(probs[i, ACTION2ID["CLARIFY"]].cpu().item()),
                    "refuse_prob": float(probs[i, ACTION2ID["REFUSE"]].cpu().item()),
                })
    elapsed = perf_counter() - start
    metrics = classification_metrics(y_true, y_pred); metrics.update(selective_refusal_metrics(y_true, y_pred, benign_mask, harmful_mask)); metrics["trainable_params"] = count_trainable_parameters(model); metrics["latency_per_example_sec"] = elapsed / max(1, len(ds))
    save_json(metrics, output_dir / "metrics.json"); save_json({"predictions": rows}, output_dir / "predictions.json")

if __name__ == "__main__":
    main()
