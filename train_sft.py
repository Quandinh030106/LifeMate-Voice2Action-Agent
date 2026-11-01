"""SFT Fine-Tuning Script for LifeMate Agent.
Fine-tunes Qwen3-4B-Instruct (or Qwen2.5-3B) on multi-turn function calling trajectories using LoRA.
Applies Completion-Only Loss Collator to focus 100% of gradient updates on Assistant tool calls.
Supports automatic adapter merging for standalone, offline inference.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType

# Bypass PEFT torchao incompatibility bug on Kaggle Python 3.13
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate SFT Fine-Tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model ID or local path (e.g., Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-4B, or Qwen/Qwen2.5-3B-Instruct).")
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
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device train batch size (default: 1 for 4B model at 4096 context).")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8 -> effective batch size 8).")
    parser.add_argument("--lr", type=float, default=1.5e-4,
                        help="Peak learning rate (default: 1.5e-4).")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank (default: 32).")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (default: 64).")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length (default: 8192 to cover 99%+ of complete tool calling trajectories).")
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging LoRA into base model.")
    return parser.parse_args()


class LifeMateCompletionCollator:
    """Vectorized data collator that computes loss ONLY on assistant tool calls and responses.
    All system prompts, tool schemas, user requests, and tool results are masked with -100.
    Works natively across any PyTorch, Transformers, or TRL version.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Qwen ChatML assistant start tokens: <|im_start|>assistant\n
        self.start_sub = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        # End token: <|im_end|>
        end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.end_token = end_tokens[-1] if end_tokens else tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"][:self.max_length] for item in batch]
        batch_max_len = max(len(ids) for ids in input_ids_list)

        padded_inputs = []
        attention_masks = []
        labels_list = []
        sub_len = len(self.start_sub)

        for seq in input_ids_list:
            seq_len = len(seq)
            pad_len = batch_max_len - seq_len

            # Right padding for causal LM
            p_input = seq + [self.pad_token_id] * pad_len
            p_mask = [1] * seq_len + [0] * pad_len
            labels = [-100] * batch_max_len

            # Locate all assistant response segments
            for i in range(seq_len - sub_len + 1):
                if seq[i:i+sub_len] == self.start_sub:
                    start_pos = i + sub_len
                    end_pos = start_pos
                    while end_pos < seq_len and seq[end_pos] != self.end_token:
                        end_pos += 1
                    # Include the end_token in loss so the model learns proper stop boundary
                    for pos in range(start_pos, min(end_pos + 1, seq_len)):
                        labels[pos] = seq[pos]

            # Edge case safety fallback: if no assistant tag matched, enable the last non-pad token
            if all(l == -100 for l in labels[:seq_len]):
                labels[seq_len - 1] = seq[-1]

            padded_inputs.append(p_input)
            attention_masks.append(p_mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def load_and_tokenize_dataset(file_path: str, tokenizer: AutoTokenizer, max_length: int = 4096) -> Dataset:
    """Read SFT jsonl file, format using ChatML tools template, and pre-tokenize."""
    items = []
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

            tokenized = tokenizer(
                formatted,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False
            )
            items.append({"input_ids": tokenized.input_ids})

    print(f"Loaded and tokenized {len(items)} samples from {file_path}")
    return Dataset.from_list(items)


def main():
    args = parse_args()
    print("=" * 60)
    print(" 🚀 LifeMate High-Capacity SFT Pipeline (LoRA) ")
    print("=" * 60)
    print(f"Base Model:       {args.model_name_or_path}")
    print(f"Train File:       {args.train_file}")
    print(f"Val File:         {args.val_file}")
    print(f"Output LoRA Dir:  {args.output_dir}")
    print(f"Merged Dir:       {args.merged_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Max Seq Length:   {args.max_seq_length}")
    print(f"Per-device Batch: {args.batch_size}")
    print(f"Grad Accum Steps: {args.grad_accum} (Effective Batch = {args.batch_size * args.grad_accum})")
    print(f"LoRA r/alpha:     {args.lora_r}/{args.lora_alpha}")

    # 1. Load Tokenizer with fallback
    print(f"\nLoading Tokenizer from {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
    except Exception as e:
        fallback_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"[Warning] Failed loading tokenizer from {args.model_name_or_path}: {e}")
        print(f"[Fallback] Falling back to {fallback_id}...")
        args.model_name_or_path = fallback_id
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Pre-Tokenized Datasets
    print("\nTokenizing Datasets with ChatML template...")
    train_dataset = load_and_tokenize_dataset(args.train_file, tokenizer, max_length=args.max_seq_length)
    val_dataset = load_and_tokenize_dataset(args.val_file, tokenizer, max_length=args.max_seq_length) if Path(args.val_file).is_file() else None

    # 3. Instantiate Completion-Only Collator
    data_collator = LifeMateCompletionCollator(tokenizer, max_length=args.max_seq_length)

    # 4. Load Base Model in float16/bfloat16 with SDPA
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
    else:
        dtype = torch.float32

    print(f"\nLoading Base Model {args.model_name_or_path} ({dtype}, SDPA)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    except Exception as e:
        fallback_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"[Warning] Failed loading base model {args.model_name_or_path}: {e}")
        print(f"[Fallback] Retrying with {fallback_id}...")
        args.model_name_or_path = fallback_id
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    # Enable Gradient Checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 5. Configure LoRA (High-Capacity r=32, alpha=64 across all linear layers)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 6. Training Arguments
    sig_ta = inspect.signature(TrainingArguments.__init__)
    valid_ta_params = set(sig_ta.parameters.keys())

    args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
        "optim": "adamw_torch",
        "gradient_checkpointing": True,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": 42,
    }
    eval_choice = "epoch" if val_dataset else "no"
    if "eval_strategy" in valid_ta_params:
        args_dict["eval_strategy"] = eval_choice
    elif "evaluation_strategy" in valid_ta_params:
        args_dict["evaluation_strategy"] = eval_choice

    filtered_args = {k: v for k, v in args_dict.items() if k in valid_ta_params}
    training_args = TrainingArguments(**filtered_args)

    # 7. Trainer with Masked Completion Collator
    sig_tr = inspect.signature(SFTTrainer.__init__)
    valid_tr_params = set(sig_tr.parameters.keys())

    trainer_dict = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "peft_config": peft_config,
        "data_collator": data_collator,
        "max_seq_length": args.max_seq_length,
        "args": training_args,
    }
    if "processing_class" in valid_tr_params:
        trainer_dict["processing_class"] = tokenizer
    elif "tokenizer" in valid_tr_params:
        trainer_dict["tokenizer"] = tokenizer

    filtered_trainer_dict = {k: v for k, v in trainer_dict.items() if k in valid_tr_params}
    print(f"[Config] Instantiating SFTTrainer with Completion-Only Masking and LoRA...")
    trainer = SFTTrainer(**filtered_trainer_dict)

    print("\nStarting Training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Merge Adapter into Base Model for Standalone Offline Inference
    if not args.no_merge:
        print("\n" + "=" * 60)
        print(" Merging LoRA weights into Base Model for Standalone Offline Inference... ")
        print("=" * 60)
        from peft import PeftModel

        del model
        del trainer
        if torch.cuda.is_available():
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
