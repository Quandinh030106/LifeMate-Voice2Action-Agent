"""Prepare SFT (Supervised Fine-Tuning) data for Qwen2.5 on LifeMate train.jsonl.
Converts 1,545 multi-turn agentic trajectories into standard ChatML format.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import (
    build_system_prompt,
    extract_available_tools,
)


def format_sample_to_messages(
    sample: Dict[str, Any],
    catalog: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert one sample from train.jsonl into a multi-turn conversation format."""
    # 1. Tools available for this sample
    tools = extract_available_tools(sample, catalog)

    # 2. System prompt
    system_prompt = build_system_prompt(sample)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 3. Conversation history if present
    for turn in sample.get("history", []):
        if "audio_text" in turn:
            messages.append({"role": "user", "content": turn["audio_text"]})
        elif "text" in turn:
            messages.append({"role": "user", "content": turn["text"]})

        if turn.get("role") == "assistant" or "tool_calls" in turn:
            asst_turn: Dict[str, Any] = {"role": "assistant", "content": turn.get("text", "")}
            if "tool_calls" in turn:
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

    # 4. Current user request
    user_request = sample.get("audio_text", "").strip()
    messages.append({"role": "user", "content": user_request})

    # 5. Target Assistant Turn(s) from label
    gold_calls = sample.get("label", {}).get("tool_calls", [])
    gold_results = sample.get("label", {}).get("tool_results", [])

    if not gold_calls:
        # Case 1: Out-of-scope sample -> Assistant produces polite refusal without tool calls
        messages.append({
            "role": "assistant",
            "content": "Yêu cầu này nằm ngoài phạm vi hỗ trợ của các ứng dụng hiện có."
        })
    else:
        # Case 2: Standard or Multi-turn tool execution
        for i, (tc, res) in enumerate(zip(gold_calls, gold_results)):
            tname = tc["name"]
            targs = tc.get("arguments", {})

            # Assistant emits this tool call
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

            # Tool responds with execution observation
            messages.append({
                "role": "tool",
                "name": tname,
                "content": json.dumps(res, ensure_ascii=False)
            })

            if tname == "sys.ask_missing_info":
                # Agent stops to wait for user info
                break

        # Final assistant confirmation closing the loop
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

    return {
        "id": sample["id"],
        "messages": messages,
        "tools": tools
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for LifeMate")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root directory containing train/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/sft",
                        help="Output directory for sft_train.jsonl and sft_val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic split")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        default_p = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test"
        data_root = default_p if default_p.is_dir() else PROJECT_ROOT / "data"

    train_jsonl = data_root / "train" / "train.jsonl"
    catalog_path = data_root / "train" / "train_tools_catalog.json"

    if not train_jsonl.is_file():
        raise FileNotFoundError(f"train.jsonl not found at {train_jsonl}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog not found at {catalog_path}")

    catalog = json.load(open(catalog_path, encoding="utf-8"))

    print(f"Loading samples from {train_jsonl}...")
    samples = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Processing {len(samples)} samples into SFT multi-turn trajectories...")
    processed = []
    for s in samples:
        traj = format_sample_to_messages(s, catalog)
        processed.append(traj)

    # Shuffle deterministically
    random.shuffle(processed)

    val_count = int(len(processed) * args.val_ratio)
    val_set = processed[:val_count]
    train_set = processed[val_count:]

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "sft_train.jsonl"
    val_file = out_dir / "sft_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(" SFT Data Preparation Completed Successfully! ")
    print(f"Total processed: {len(processed)}")
    print(f"Train split:     {len(train_set)} -> {train_file}")
    print(f"Val split:       {len(val_set)} -> {val_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
