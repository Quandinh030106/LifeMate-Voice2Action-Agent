"""LifeMate Inference Script for Zalo AI Challenge 2026.
Entry point for evaluating splits and generating competition submissions.

Usage:
    # Full audio inference on public_test
    python inference.py --split public_test --output submission.jsonl

    # Fast text-only inference on train split (first 20 samples)
    python inference.py --split train --no_audio --limit 20 --output train_sub.jsonl
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Fix console UTF-8 and force immediate unbuffered output (crucial for Kaggle/Jupyter)
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# Ensure every print flushes immediately to Kaggle output cell in real time
print = functools.partial(print, flush=True)

# Fix all seeds for deterministic, reproducible results (BTC requirement)
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable full GPU performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# Set high process priority
try:
    import psutil
    p = psutil.Process()
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass

# Add local package and zaic_tools to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import VietnameseASR
from src.llm_engine import LocalLLMEngine
from src.agent import LifeMateAgent
from src.pipeline import LifeMatePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate Submission Generator")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root folder containing the split folder.")
    parser.add_argument("--split", type=str, default="public_test",
                        help="Dataset split to evaluate ('public_test' or 'train').")
    parser.add_argument("--output", type=str, default="result/submission.jsonl",
                        help="Output path for submission.jsonl.")
    parser.add_argument("--no_audio", action="store_true",
                        help="Skip ASR and use audio_text directly (debugging/training only).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing).")
    parser.add_argument("--asr_engine", type=str, default="auto", choices=["auto", "nvidia", "phowhisper"],
                        help="ASR backend engine (auto, nvidia, or phowhisper).")
    parser.add_argument("--asr_model", type=str, default=None,
                        help="ASR model HuggingFace ID or local path.")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="LLM model HuggingFace ID or local path.")
    parser.add_argument("--max_steps", type=int, default=5,
                        help="Max agent reasoning steps per sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(" 🎙️ LifeMate Agentic AI — Inference Pipeline ")
    print("=" * 60)

    # 1. Determine Paths
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        # Default local development path
        default_path = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test"
        data_root = default_path if default_path.is_dir() else PROJECT_ROOT / "data"

    split_dir = data_root / args.split
    input_jsonl = split_dir / f"{args.split}.jsonl"
    catalog_path = split_dir / f"{args.split}_tools_catalog.json"

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    # Ensure zaic_tools from split directory is importable
    sys.path.insert(0, str(split_dir))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Split:       {args.split}")
    print(f"Data root:   {data_root}")
    print(f"Input file:  {input_jsonl}")
    print(f"Catalog:     {catalog_path}")
    print(f"Output file: {output_path}")
    print(f"Use Audio:   {not args.no_audio}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Stage 1: Audio Transcription Cache & Batch Processing
    audio_transcripts = {}
    transcripts_cache_file = output_path.parent / f"transcripts_{args.split}.json"

    if transcripts_cache_file.is_file():
        try:
            with open(transcripts_cache_file, "r", encoding="utf-8") as f_tr:
                audio_transcripts = json.load(f_tr)
            print(f"Loaded {len(audio_transcripts)} cached audio transcriptions from {transcripts_cache_file}.")
        except Exception:
            audio_transcripts = {}

    if not args.no_audio:
        # Scan all required audio paths
        audio_rel_paths = set()
        with open(input_jsonl, encoding="utf-8") as fin_scan:
            for line in fin_scan:
                if not line.strip():
                    continue
                item = json.loads(line)
                if "audio" in item and item["audio"]:
                    audio_rel_paths.add(item["audio"].replace("\\", "/"))
                for turn in item.get("history", []):
                    if turn.get("role") == "user" and "audio" in turn and not turn.get("audio_text"):
                        audio_rel_paths.add(turn["audio"].replace("\\", "/"))

        missing_paths = [
            rel_p for rel_p in audio_rel_paths
            if rel_p not in audio_transcripts and (split_dir / rel_p).is_file()
        ]

        if missing_paths:
            print(f"Transcribing {len(missing_paths)} audio files using {args.asr_engine}...")
            asr = VietnameseASR(engine=args.asr_engine, model_id=args.asr_model, device=device)
            print(f"ASR active engine: {asr.engine_name}")

            full_paths = [split_dir / p for p in missing_paths]
            if hasattr(asr.asr_model, "transcribe_batch") and full_paths:
                try:
                    results = asr.asr_model.transcribe_batch(full_paths, batch_size=16)
                    for rel_p, text in zip(missing_paths, results):
                        audio_transcripts[rel_p] = text
                except Exception as e:
                    print(f"Batch ASR warning: {e}. Falling back to iterative...")
                    for rel_p in missing_paths:
                        try:
                            audio_transcripts[rel_p] = asr.transcribe(split_dir / rel_p)
                        except Exception:
                            pass
            else:
                for rel_p in missing_paths:
                    try:
                        audio_transcripts[rel_p] = asr.transcribe(split_dir / rel_p)
                    except Exception:
                        pass

            try:
                with open(transcripts_cache_file, "w", encoding="utf-8") as f_tr:
                    json.dump(audio_transcripts, f_tr, ensure_ascii=False, indent=2)
                print(f"Saved {len(audio_transcripts)} transcriptions to cache.")
            except Exception:
                pass

            # CRITICAL: Free ASR model from GPU VRAM completely before loading LLM!
            del asr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("Freed ASR model from GPU VRAM to maximize memory for LLM.")

    # 3. Stage 2: Load LLM with 100% Free VRAM Headroom
    print(f"Loading LLM model: {args.llm_model}...")
    llm = LocalLLMEngine(model_name_or_path=args.llm_model, device=device)

    catalog = json.load(open(catalog_path, encoding="utf-8"))

    # 4. Setup Agent and Pipeline
    agent = LifeMateAgent(
        llm_generate_fn=llm.generate,
        catalog=catalog,
        max_steps=args.max_steps
    )
    pipeline = LifeMatePipeline(
        agent=agent,
        asr_model=None,
        audio_transcripts=audio_transcripts
    )

    # 4. Check for already processed samples (Resume support)
    processed_ids = set()
    if output_path.is_file():
        try:
            with open(output_path, "r", encoding="utf-8") as f_check:
                for l in f_check:
                    if l.strip():
                        processed_ids.add(json.loads(l)["id"])
            print(f"Resuming: Found {len(processed_ids)} already processed samples.")
        except Exception:
            pass

    # Count total samples
    with open(input_jsonl, encoding="utf-8") as fin_count:
        total_in_split = sum(1 for line in fin_count if line.strip())

    # 5. Run Inference Loop
    print(f"\nStarting inference on {total_in_split} samples (remaining: {total_in_split - len(processed_ids)})...")
    total_processed = len(processed_ids)
    total_time = 0.0

    with open(input_jsonl, encoding="utf-8") as fin, open(output_path, "a", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            if args.limit is not None and total_processed >= args.limit:
                break

            item = json.loads(line)
            sample_id = item["id"]

            if sample_id in processed_ids:
                continue

            t0 = time.time()
            tool_calls, state_hash, meta = pipeline.process_sample(
                item=item,
                split_dir=split_dir,
                split_name=args.split,
                data_root=data_root,
                use_audio=not args.no_audio
            )
            elapsed = time.time() - t0
            total_time += elapsed
            total_processed += 1
            processed_ids.add(sample_id)

            # Format submission record exactly as required
            record = {
                "id": sample_id,
                "tool_calls": tool_calls,
                "state_hash": state_hash
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # Log progress with percentage
            pct = (total_processed / total_in_split) * 100
            print(f"[{total_processed:03d}/{total_in_split}] ({pct:5.1f}%) Sample: {sample_id} | Time: {elapsed:.2f}s | Calls: {len(tool_calls)} | Hash: {state_hash[:10]}...")

    avg_time = total_time / total_processed if total_processed > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"Inference Completed!")
    print(f"Total samples:     {total_processed}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Average time/sample: {avg_time:.2f}s (Budget: <= 30.0s)")
    print(f"Output saved to:   {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
