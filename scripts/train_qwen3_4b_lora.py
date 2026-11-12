"""Fine-Tuning Script for Qwen 3-4B on LifeMate (Zalo AI Challenge 2026).
Trains Qwen3-4B-Instruct-2507 with QLoRA (4-bit) on 1,545 ground-truth trajectories from train.jsonl.

Features:
1. Multi-turn ReAct trajectory formatting:
   - System prompt with context & user profile
   - History turns
   - Multi-step tool calls & execution observations
   - Proper stopping on task completion or sys.ask_missing_info
2. Casing & punctuation augmentation (robust to ASR variations: ALL-CAPS, lowercase, punctuated).
3. Loss masking on Assistant tokens only (System, User, and Tool observation tokens are masked with -100).
4. Efficient QLoRA 4-bit training fitting within Kaggle GPU T4/P100 (16GB VRAM, ~4GB usage).
5. Automatic fast-merge to /kaggle/working/models/lifemate-qwen3-4b-merged upon completion.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Bypass PEFT torchao incompatibility on Kaggle Python 3.13
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import (
    build_system_prompt,
    extract_available_tools,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 3-4B for LifeMate on Kaggle")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name or local path (e.g. Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root folder containing train split.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/models/lifemate-qwen3-4b-lora",
                        help="Path to save LoRA adapter checkpoint.")
    parser.add_argument("--merged_dir", type=str, default="/kaggle/working/models/lifemate-qwen3-4b-merged",
                        help="Path to save merged standalone model.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size (default: 2).")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch size = batch_size * grad_accum = 16).")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate (default: 2e-4).")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048).")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank r (default: 16).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32).")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit NormalFloat quantization (requires bitsandbytes). Default: False (native 16-bit LoRA).")
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging LoRA into base model.")
    return parser.parse_args()


def locate_data_files(hint: Optional[str] = None) -> Tuple[Path, Path]:
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates.extend([
        Path("/kaggle/input/datasets/locvt06/zaic2026-track-lifemate"),
        Path("/kaggle/input/zaic2026-track-lifemate"),
        PROJECT_ROOT / "data" / "train+public_test" / "train+public_test",
        PROJECT_ROOT / "data" / "train+public_test",
        PROJECT_ROOT / "data",
    ])
    for cand in candidates:
        tr_file = cand / "train" / "train.jsonl"
        cat_file = cand / "train" / "train_tools_catalog.json"
        if tr_file.is_file() and cat_file.is_file():
            return tr_file.resolve(), cat_file.resolve()

    found_tr = glob.glob("/kaggle/input/**/train.jsonl", recursive=True) + glob.glob("**/train.jsonl", recursive=True)
    if found_tr:
        tr_p = Path(found_tr[0]).resolve()
        cat_p = tr_p.parent / "train_tools_catalog.json"
        return tr_p, cat_p

    raise FileNotFoundError("Could not find train.jsonl and train_tools_catalog.json!")


def augment_user_request(text: str, rng: random.Random) -> str:
    """Randomly apply casing & punctuation variations to match ASR behavior."""
    mode = rng.random()
    if mode < 0.25:
        # ALL CAPS (Gipformer output style)
        return text.upper()
    elif mode < 0.50:
        # Lowercase without punctuation (Whisper output style)
        t = text.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    else:
        # Original text with normal casing and punctuation
        return text


def build_sample_messages(
    item: Dict[str, Any],
    catalog: Dict[str, Any],
    rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert one ground truth sample into multi-turn messages and available tools."""
    tools = extract_available_tools(item, catalog)
    system_prompt = build_system_prompt(item)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    # Prior multi-turn conversation history
    for turn in item.get("history", []):
        role = turn.get("role")
        if role == "user":
            txt = turn.get("audio_text") or turn.get("text") or ""
            messages.append({"role": "user", "content": txt})
        elif role == "assistant":
            asst_turn: Dict[str, Any] = {"role": "assistant", "content": turn.get("text", "")}
            if "tool_calls" in turn and turn["tool_calls"]:
                asst_turn["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("arguments", {})
                        }
                    }
                    for tc in turn["tool_calls"]
                ]
            messages.append(asst_turn)
            if "tool_results" in turn and "tool_calls" in turn:
                for tc, res in zip(turn["tool_calls"], turn["tool_results"]):
                    messages.append({
                        "role": "tool",
                        "name": tc["name"],
                        "content": json.dumps(res, ensure_ascii=False)
                    })

    # Current user request with casing augmentation
    raw_req = item.get("audio_text", "").strip() or item.get("text", "").strip()
    user_req = augment_user_request(raw_req, rng)
    messages.append({"role": "user", "content": user_req})

    # Target Assistant Turns from ground truth label
    gold_calls = item.get("label", {}).get("tool_calls", [])
    gold_results = item.get("label", {}).get("tool_results", [])

    if not gold_calls:
        # Out-of-Scope sample: Assistant politely refuses without tool calls
        messages.append({
            "role": "assistant",
            "content": "Yêu cầu này nằm ngoài phạm vi hỗ trợ của các ứng dụng hiện có trên thiết bị."
        })
    else:
        # Multi-step ReAct Trajectory
        for i, (tc, res) in enumerate(zip(gold_calls, gold_results)):
            tname = tc["name"]
            targs = tc.get("arguments", {})

            # Assistant emits this specific tool call
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tname,
                            "arguments": targs
                        }
                    }
                ]
            })

            # Environment returns observation
            messages.append({
                "role": "tool",
                "name": tname,
                "content": json.dumps(res, ensure_ascii=False)
            })

            if tname == "sys.ask_missing_info":
                break

        # Final closing response
        if any(c["name"] == "sys.ask_missing_info" for c in gold_calls):
            messages.append({
                "role": "assistant",
                "content": "Đang chờ bạn cung cấp thêm thông tin cần thiết."
            })
        else:
            messages.append({
                "role": "assistant",
                "content": "Đã thực hiện xong yêu cầu của bạn."
            })

    return messages, tools


class LifeMateSFTDataset(Dataset):
    """PyTorch Dataset that masks non-assistant tokens with -100."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        catalog: Dict[str, Any],
        tokenizer: Any,
        max_length: int = 2048,
        seed: int = 42
    ):
        self.samples = samples
        self.catalog = catalog
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

        # Locate delimiter token ids for assistant turns
        # In Qwen: <|im_start|>assistant ... <|im_end|>
        self.asst_start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        messages, tools = build_sample_messages(item, self.catalog, self.rng)

        # Render complete multi-turn conversation into ChatML format with tools
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False
            )
        except Exception:
            # Fallback simple ChatML
            rendered = ""
            for m in messages:
                role = m["role"]
                content = m.get("content", "")
                if "tool_calls" in m:
                    for tc in m["tool_calls"]:
                        fn = tc.get("function", {})
                        content += f"\n<tool_call>\n{json.dumps({'name': fn.get('name'), 'arguments': fn.get('arguments')}, ensure_ascii=False)}\n</tool_call>"
                rendered += f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n"

        encoded = self.tokenizer(
            rendered,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = torch.full_like(input_ids, -100)

        # Exact Assistant Span Unmasking
        # Mask everything except tokens generated between <|im_start|>assistant and <|im_end|>
        ids_list = input_ids.tolist()
        asst_hdr = self.asst_start_tokens
        end_hdr = self.im_end_tokens

        i = 0
        n = len(ids_list)
        while i < n:
            # Check for <|im_start|>assistant
            if ids_list[i:i + len(asst_hdr)] == asst_hdr:
                start_idx = i + len(asst_hdr)
                # Find matching <|im_end|>
                end_idx = n
                j = start_idx
                while j < n:
                    if ids_list[j:j + len(end_hdr)] == end_hdr:
                        end_idx = j + len(end_hdr)
                        break
                    j += 1
                # Unmask assistant tokens
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
                i = end_idx
            else:
                i += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Dynamic padding collator for variable length sequences."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    b_input_ids = []
    b_attention_mask = []
    b_labels = []

    for item in batch:
        cur_len = item["input_ids"].size(0)
        pad_len = max_len - cur_len
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros(pad_len, dtype=torch.long)
            pad_lbl = torch.full((pad_len,), -100, dtype=torch.long)

            b_input_ids.append(torch.cat([item["input_ids"], pad_ids]))
            b_attention_mask.append(torch.cat([item["attention_mask"], pad_mask]))
            b_labels.append(torch.cat([item["labels"], pad_lbl]))
        else:
            b_input_ids.append(item["input_ids"])
            b_attention_mask.append(item["attention_mask"])
            b_labels.append(item["labels"])

    return {
        "input_ids": torch.stack(b_input_ids),
        "attention_mask": torch.stack(b_attention_mask),
        "labels": torch.stack(b_labels)
    }


def main():
    args = parse_args()

    print("=" * 70)
    print(" 🚀 LIFEMATE QWEN 3-4B QLORA FINE-TUNING PIPELINE ")
    print("=" * 70)
    print(f"Base Model:     {args.base_model}")
    print(f"Target Epochs:  {args.epochs}")
    print(f"Batch Size:     {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective Batch: {args.batch_size * args.grad_accum})")
    print(f"Learning Rate:  {args.lr}")
    print(f"LoRA Config:    r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Output LoRA:    {args.output_dir}")
    print(f"Output Merged:  {args.merged_dir}")
    print("=" * 70)

    # 1. Locate Data
    train_file, catalog_file = locate_data_files(args.data_dir)
    print(f"✅ Found Train File:   {train_file}")
    print(f"✅ Found Catalog File: {catalog_file}")

    with open(catalog_file, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    samples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"✅ Loaded {len(samples)} ground-truth samples from train.jsonl.")

    # 2. Load Tokenizer
    print(f"\n[1/4] Loading Tokenizer ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Build Dataset
    print("\n[2/4] Formatting 1,545 multi-turn ReAct trajectories with token masking...")
    dataset = LifeMateSFTDataset(
        samples=samples,
        catalog=catalog,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    print(f"✅ Dataset prepared: {len(dataset)} training examples.")

    # 4. Load Base Model
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    use_4bit = args.load_in_4bit
    if use_4bit:
        try:
            import bitsandbytes
        except ImportError:
            print("[Warning] bitsandbytes is not installed. Falling back to native 16-bit LoRA (fits in 16GB VRAM)...")
            use_4bit = False

    if use_4bit:
        print(f"\n[3/4] Loading Qwen 3-4B in 4-bit NormalFloat (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"\n[3/4] Loading Qwen 3-4B in native {compute_dtype} precision (optimal for 16GB VRAM, no quantization noise)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )

    # 5. Apply LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Training Arguments
    os.makedirs(args.output_dir, exist_ok=True)
    optim_name = "paged_adamw_8bit" if use_4bit else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim=optim_name,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    print("\n[4/4] Starting LoRA Training on GPU...")
    t0 = time.time()
    trainer.train()
    print(f"\n🎉 Training finished in {(time.time() - t0) / 60:.2f} minutes!")

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7. Merge to Standalone Model
    if not args.no_merge:
        print("\n" + "=" * 70)
        print("🚀 AUTOMATIC FAST-MERGE TO STANDALONE MODEL")
        print("=" * 70)
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        os.makedirs(args.merged_dir, exist_ok=True)
        print(f"Loading Base Model {args.base_model} on CPU for clean merge...")
        base_fp16 = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"Attaching LoRA from {args.output_dir}...")
        peft_m = PeftModel.from_pretrained(base_fp16, args.output_dir)
        merged_m = peft_m.merge_and_unload()

        print(f"Saving merged standalone model to {args.merged_dir}...")
        merged_m.save_pretrained(args.merged_dir)
        tokenizer.save_pretrained(args.merged_dir)

        del base_fp16, peft_m, merged_m
        gc.collect()
        print(f"✅ Merged standalone model successfully saved at: {args.merged_dir}")
        print("🎯 Sẵn sàng chạy benchmark 100 câu hoặc nộp bài trực tiếp!")


if __name__ == "__main__":
    main()
