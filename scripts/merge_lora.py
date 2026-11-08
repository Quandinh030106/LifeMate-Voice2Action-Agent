"""Standalone LoRA Merger for LifeMate.
Runs in a separate OS process to prevent Jupyter Python 3.13 C-extension memory corruption.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Bypass PEFT torchao incompatibility
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone LoRA Merger")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model HuggingFace ID or local path.")
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="Path to LoRA adapter directory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save merged standalone model.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 STANDALONE LORA MERGE PROCESS")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Dir:   {args.lora_dir}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)

    out_path = Path(args.output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"✅ Merged model already exists at {out_path}! Skipping merge.")
        return

    print("Loading Base Model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    print(f"Loading LoRA adapter from {args.lora_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_dir)

    print("Merging weights into standalone base model...")
    merged_model = peft_model.merge_and_unload()

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged standalone model to {out_path}...")
    merged_model.save_pretrained(str(out_path))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_path))

    del base_model, peft_model, merged_model
    gc.collect()

    print("\n🎉 Merge completed successfully!")
    print(f"Standalone model ready at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
