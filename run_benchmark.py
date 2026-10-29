"""Benchmark and Validation Script for LifeMate Agentic AI.
Runs end-to-end pipeline on sample set, evaluates metrics, and verifies latency.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAIN = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "train"
sys.path.insert(0, str(DATA_TRAIN))

from src.asr import VietnameseASR
from src.llm_engine import LocalLLMEngine
from src.agent import LifeMateAgent
from src.pipeline import LifeMatePipeline
from src.evaluator import evaluate_dataset
from zaic_tools import ToolEnv


def run_benchmark(num_samples: int = 10, use_audio: bool = False):
    print("=" * 65)
    print(f" 🚀 RUNNING LIFEMATE BENCHMARK ({num_samples} samples | Audio: {use_audio})")
    print("=" * 65)

    train_file = DATA_TRAIN / "train.jsonl"
    catalog_file = DATA_TRAIN / "train_tools_catalog.json"
    catalog = json.load(open(catalog_file, encoding="utf-8"))

    # Load models
    asr = None
    if use_audio:
        print("Loading PhoWhisper ASR...")
        asr = VietnameseASR("vinai/PhoWhisper-small")

    print("Loading Qwen2.5 LLM...")
    llm = LocalLLMEngine("Qwen/Qwen2.5-1.5B-Instruct")

    agent = LifeMateAgent(llm_generate_fn=llm.generate, catalog=catalog, max_steps=5)
    pipeline = LifeMatePipeline(agent=agent, asr_model=asr)

    predictions = []
    ground_truth = []
    latencies = []

    with open(train_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            item = json.loads(line)

            # Compute gold hash using ToolEnv
            env = ToolEnv("train", item["id"], root=DATA_TRAIN.parent)
            for c in item["label"]["tool_calls"]:
                env.call(c["name"], c["arguments"])
            item["gold_hash"] = env.get_state_hash()
            ground_truth.append(item)

            # Process with our agent pipeline
            t0 = time.time()
            tool_calls, state_hash, meta = pipeline.process_sample(
                item=item,
                split_dir=DATA_TRAIN,
                split_name="train",
                data_root=DATA_TRAIN.parent,
                use_audio=use_audio
            )
            elapsed = time.time() - t0
            latencies.append(elapsed)

            predictions.append({
                "id": item["id"],
                "tool_calls": tool_calls,
                "state_hash": state_hash
            })

            print(f"[{idx+1:02d}/{num_samples:02d}] {item['id']} | Time: {elapsed:.2f}s | Pred Calls: {len(tool_calls)} | Gold Calls: {len(item['label']['tool_calls'])}")

    # Evaluate
    print("\nEvaluating against official metrics...")
    metrics = evaluate_dataset(predictions, ground_truth)

    avg_time = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 65)
    print(" 📊 BENCHMARK RESULTS ")
    print("=" * 65)
    print(f"Total evaluated samples: {metrics['total_samples']}")
    print(f"Official Final Score:   {metrics['final_score']} / 100.0")
    print(f"State Hash Match Rate:  {metrics['hash_match_rate']}%")
    print(f"Perfect Sample Match:   {metrics['perfect_match_rate']}%")
    print(f"Average Latency:        {avg_time:.2f}s per sample (Rule: <= 30.0s)")
    print(f"Max Latency:            {max(latencies):.2f}s")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--use_audio", action="store_true")
    args = parser.parse_args()
    run_benchmark(num_samples=args.num_samples, use_audio=args.use_audio)
