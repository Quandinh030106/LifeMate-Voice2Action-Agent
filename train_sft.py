"""SFT Fine-Tuning Script for LifeMate Agent.
Fine-tunes Qwen2.5-3B-Instruct (or 1.5B) on multi-turn function calling trajectories using LoRA.
Supports automatic adapter merging for standalone, offline inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate SFT Fine-Tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model ID or local path (e.g., Qwen/Qwen2.5-3B-Instruct or 1.5B).")
    parser.add_argument("--train_file", type=str, default="data/sft/sft_train.jsonl",
                        help="Path to sft_train.jsonl.")
    parser.add_argument("--val_file", type=str, default="data/sft/sft_val.jsonl",
                        help="Path to sft_val.jsonl.")
    parser.add_argument("--output_dir", type=str, default="models/lifemate-lora",
                        help="Directory to save LoRA checkpoint.")
    parser.add_argument("--merged_dir", type=str, default="models/lifemate-qwen-sft",
                        help="Directory to save merged standalone model.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device train batch size (default: 2).")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4 -> effective batch size 8).")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate (default: 2e-4).")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32).")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048).")
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging LoRA into base model.")
    return parser.parse_args()


def load_and_format_dataset(file_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """Read SFT jsonl file and pre-format conversations using tokenizer chat template."""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            messages = entry["messages"]
            tools = entry.get("tools")

            try:
                if tools:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
            except Exception:
                formatted = ""
                for m in messages:
                    formatted += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"

            texts.append({"text": formatted})

    print(f"Loaded {len(texts)} samples from {file_path}")
    return Dataset.from_list(texts)


def main():
    args = parse_args()
    print("=" * 60)
    print(" 🚀 LifeMate SFT Fine-Tuning Pipeline (LoRA) ")
    print("=" * 60)
    print(f"Base Model:       {args.model_name_or_path}")
    print(f"Train File:       {args.train_file}")
    print(f"Val File:         {args.val_file}")
    print(f"Output LoRA Dir:  {args.output_dir}")
    print(f"Merged Dir:       {args.merged_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Effective Batch:  {args.batch_size * args.grad_accum}")
    print(f"LoRA r/alpha:     {args.lora_r}/{args.lora_alpha}")

    # 1. Load Tokenizer
    print(f"\nLoading Tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Datasets
    print("\nFormatting Datasets with ChatML template...")
    train_dataset = load_and_format_dataset(args.train_file, tokenizer)
    val_dataset = load_and_format_dataset(args.val_file, tokenizer) if Path(args.val_file).is_file() else None

    # 3. Load Model in bfloat16 / float16 with SDPA
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\nLoading Base Model {args.model_name_or_path} ({dtype}, SDPA)...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # 4. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 5. Training Arguments (robust across all transformers versions)
    args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
        "optim": "adamw_torch",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": 42,
    }
    eval_choice = "epoch" if val_dataset else "no"
    try:
        training_args = TrainingArguments(**args_dict, eval_strategy=eval_choice)
    except TypeError:
        training_args = TrainingArguments(**args_dict, evaluation_strategy=eval_choice)

    # 6. Trainer (robust across all trl versions)
    trainer_dict = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "peft_config": peft_config,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "args": training_args,
    }
    try:
        trainer = SFTTrainer(**trainer_dict, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_dict, tokenizer=tokenizer)

    print("\nStarting Training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7. Merge Adapter into Base Model for Standalone Offline Inference
    if not args.no_merge:
        print("\n" + "=" * 60)
        print(" Merging LoRA weights into Base Model for Offline Inference... ")
        print("=" * 60)
        from peft import PeftModel

        del model
        del trainer
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = peft_model.merge_and_unload()

        merged_path = Path(args.merged_dir)
        merged_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged standalone model to {merged_path}...")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        print("\n🎉 Merge completed successfully!")
        print(f"Standalone model ready at: {merged_path.resolve()}")


if __name__ == "__main__":
    main()
