python train_direct_classifier.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__direct_classifier__seed1 --seed 1
python train_direct_classifier.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__direct_classifier__seed2 --seed 2
python train_direct_classifier.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__direct_classifier__seed3 --seed 3
python train_stage1.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__efsc_stage1__seed1 --seed 1
python train_stage1.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__efsc_stage1__seed2 --seed 2
python train_stage1.py --model_name roberta-base --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/roberta-base__efsc_stage1__seed3 --seed 3
python train_stage2.py --model_name roberta-base --train_path data/processed/train.jsonl --stage1_ckpt outputs/roberta-base__efsc_stage1__seed1/best_stage1.pt --output_dir outputs/roberta-base__efsc_stage1_stage2__seed1 --seed 1
python train_stage2.py --model_name roberta-base --train_path data/processed/train.jsonl --stage1_ckpt outputs/roberta-base__efsc_stage1__seed2/best_stage1.pt --output_dir outputs/roberta-base__efsc_stage1_stage2__seed2 --seed 2
python train_stage2.py --model_name roberta-base --train_path data/processed/train.jsonl --stage1_ckpt outputs/roberta-base__efsc_stage1__seed3/best_stage1.pt --output_dir outputs/roberta-base__efsc_stage1_stage2__seed3 --seed 3
python train_stage3_dpo.py --model_name roberta-base --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/roberta-base__efsc_stage1_stage2__seed1/best_stage2.pt --output_dir outputs/roberta-base__efsc_full__seed1 --seed 1
python train_stage3_dpo.py --model_name roberta-base --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/roberta-base__efsc_stage1_stage2__seed2/best_stage2.pt --output_dir outputs/roberta-base__efsc_full__seed2 --seed 2
python train_stage3_dpo.py --model_name roberta-base --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/roberta-base__efsc_stage1_stage2__seed3/best_stage2.pt --output_dir outputs/roberta-base__efsc_full__seed3 --seed 3
# roberta-base__efsc_no_harm_head__seed1: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_harm_head__seed2: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_harm_head__seed3: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_legit_head__seed1: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_legit_head__seed2: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_legit_head__seed3: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_uncertainty_head__seed1: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_uncertainty_head__seed2: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_uncertainty_head__seed3: architecture ablation requires corresponding model switch
# roberta-base__efsc_no_counterfactual__seed1: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# roberta-base__efsc_no_counterfactual__seed2: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# roberta-base__efsc_no_counterfactual__seed3: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# roberta-base__efsc_no_preference__seed1: use stage2 checkpoint directly for no-preference ablation
# roberta-base__efsc_no_preference__seed2: use stage2 checkpoint directly for no-preference ablation
# roberta-base__efsc_no_preference__seed3: use stage2 checkpoint directly for no-preference ablation
# unsupported method for now: roberta-base__lora_baseline__seed1
# unsupported method for now: roberta-base__lora_baseline__seed2
# unsupported method for now: roberta-base__lora_baseline__seed3
python train_direct_classifier.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__direct_classifier__seed1 --seed 1 --use_lora
python train_direct_classifier.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__direct_classifier__seed2 --seed 2 --use_lora
python train_direct_classifier.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__direct_classifier__seed3 --seed 3 --use_lora
python train_stage1_lora.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__efsc_stage1__seed1 --seed 1 --use_lora
python train_stage1_lora.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__efsc_stage1__seed2 --seed 2 --use_lora
python train_stage1_lora.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/mistral-7b-instruct__efsc_stage1__seed3 --seed 3 --use_lora
python train_stage2.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/mistral-7b-instruct__efsc_stage1__seed1/best_stage1_lora.pt --output_dir outputs/mistral-7b-instruct__efsc_stage1_stage2__seed1 --seed 1
python train_stage2.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/mistral-7b-instruct__efsc_stage1__seed2/best_stage1_lora.pt --output_dir outputs/mistral-7b-instruct__efsc_stage1_stage2__seed2 --seed 2
python train_stage2.py --model_name mistral-7b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/mistral-7b-instruct__efsc_stage1__seed3/best_stage1_lora.pt --output_dir outputs/mistral-7b-instruct__efsc_stage1_stage2__seed3 --seed 3
python train_stage3_dpo.py --model_name mistral-7b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/mistral-7b-instruct__efsc_stage1_stage2__seed1/best_stage2.pt --output_dir outputs/mistral-7b-instruct__efsc_full__seed1 --seed 1 --use_lora
python train_stage3_dpo.py --model_name mistral-7b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/mistral-7b-instruct__efsc_stage1_stage2__seed2/best_stage2.pt --output_dir outputs/mistral-7b-instruct__efsc_full__seed2 --seed 2 --use_lora
python train_stage3_dpo.py --model_name mistral-7b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/mistral-7b-instruct__efsc_stage1_stage2__seed3/best_stage2.pt --output_dir outputs/mistral-7b-instruct__efsc_full__seed3 --seed 3 --use_lora
# mistral-7b-instruct__efsc_no_harm_head__seed1: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_harm_head__seed2: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_harm_head__seed3: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_legit_head__seed1: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_legit_head__seed2: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_legit_head__seed3: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_uncertainty_head__seed1: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_uncertainty_head__seed2: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_uncertainty_head__seed3: architecture ablation requires corresponding model switch
# mistral-7b-instruct__efsc_no_counterfactual__seed1: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# mistral-7b-instruct__efsc_no_counterfactual__seed2: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# mistral-7b-instruct__efsc_no_counterfactual__seed3: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# mistral-7b-instruct__efsc_no_preference__seed1: use stage2 checkpoint directly for no-preference ablation
# mistral-7b-instruct__efsc_no_preference__seed2: use stage2 checkpoint directly for no-preference ablation
# mistral-7b-instruct__efsc_no_preference__seed3: use stage2 checkpoint directly for no-preference ablation
# unsupported method for now: mistral-7b-instruct__lora_baseline__seed1
# unsupported method for now: mistral-7b-instruct__lora_baseline__seed2
# unsupported method for now: mistral-7b-instruct__lora_baseline__seed3
python train_direct_classifier.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__direct_classifier__seed1 --seed 1 --use_lora
python train_direct_classifier.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__direct_classifier__seed2 --seed 2 --use_lora
python train_direct_classifier.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__direct_classifier__seed3 --seed 3 --use_lora
python train_stage1_lora.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1__seed1 --seed 1 --use_lora
python train_stage1_lora.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1__seed2 --seed 2 --use_lora
python train_stage1_lora.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1__seed3 --seed 3 --use_lora
python train_stage2.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1__seed1/best_stage1_lora.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed1 --seed 1
python train_stage2.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1__seed2/best_stage1_lora.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed2 --seed 2
python train_stage2.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.jsonl --stage1_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1__seed3/best_stage1_lora.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed3 --seed 3
python train_stage3_dpo.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed1/best_stage2.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_full__seed1 --seed 1 --use_lora
python train_stage3_dpo.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed2/best_stage2.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_full__seed2 --seed 2 --use_lora
python train_stage3_dpo.py --model_name llama-3.2-3b-instruct --train_path data/processed/train.pref.jsonl --val_path data/processed/val.pref.jsonl --init_ckpt outputs/llama-3.2-3b-instruct__efsc_stage1_stage2__seed3/best_stage2.pt --output_dir outputs/llama-3.2-3b-instruct__efsc_full__seed3 --seed 3 --use_lora
# llama-3.2-3b-instruct__efsc_no_harm_head__seed1: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_harm_head__seed2: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_harm_head__seed3: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_legit_head__seed1: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_legit_head__seed2: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_legit_head__seed3: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_uncertainty_head__seed1: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_uncertainty_head__seed2: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_uncertainty_head__seed3: architecture ablation requires corresponding model switch
# llama-3.2-3b-instruct__efsc_no_counterfactual__seed1: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# llama-3.2-3b-instruct__efsc_no_counterfactual__seed2: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# llama-3.2-3b-instruct__efsc_no_counterfactual__seed3: run train_stage3_dpo.py from efsc_stage1 checkpoint for no-counterfactual ablation
# llama-3.2-3b-instruct__efsc_no_preference__seed1: use stage2 checkpoint directly for no-preference ablation
# llama-3.2-3b-instruct__efsc_no_preference__seed2: use stage2 checkpoint directly for no-preference ablation
# llama-3.2-3b-instruct__efsc_no_preference__seed3: use stage2 checkpoint directly for no-preference ablation
# unsupported method for now: llama-3.2-3b-instruct__lora_baseline__seed1
# unsupported method for now: llama-3.2-3b-instruct__lora_baseline__seed2
# unsupported method for now: llama-3.2-3b-instruct__lora_baseline__seed3
