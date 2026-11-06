"""Adaptive SFT Data Generator for LifeMate (Zalo AI Challenge 2026).
Constructs high-quality, generalizable Supervised Fine-Tuning trajectories from train.jsonl.

Implements 5 Key Meta-Learning Strategies:
1. Meta-Reasoning Chain-of-Thought (<thought> tag explaining Goal, Missing Slots, and Decision).
2. Dynamic Tool Shuffling (Order-invariant function calling).
3. Distractor Injection (Sampling 2-4 unrelated domain tools to train noise rejection).
4. Out-of-Domain & Refusal Learning (Teaching model when NOT to call tools).
5. Multi-Step Read-Before-Write Dependency Alignment via Skill Knowledge Graph.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import (
    build_system_prompt,
    extract_available_tools,
)
from src.skill_graph import SkillGraph


def locate_data_files(base_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """Locate train.jsonl and train_tools_catalog.json automatically."""
    candidates = []
    if base_dir:
        candidates.append(Path(base_dir))
    candidates.extend([
        PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "train",
        PROJECT_ROOT / "data" / "train",
        Path("/kaggle/input/lifemate-zaic-2026-data/train+public_test/train"),
        Path("/kaggle/input/lifemate-data/train+public_test/train+public_test/train"),
    ])

    for cand in candidates:
        tf = cand / "train.jsonl"
        cf = cand / "train_tools_catalog.json"
        if tf.is_file() and cf.is_file():
            return tf.resolve(), cf.resolve()

    # Broad search
    t_matches = glob.glob("**/train.jsonl", recursive=True)
    c_matches = glob.glob("**/train_tools_catalog.json", recursive=True)
    if t_matches and c_matches:
        return Path(t_matches[0]).resolve(), Path(c_matches[0]).resolve()

    raise FileNotFoundError("Could not locate train.jsonl and train_tools_catalog.json!")


def build_sample_trajectory(
    sample: Dict[str, Any],
    catalog: Dict[str, Any],
    skill_graph: SkillGraph,
    inject_distractors: bool = True,
    shuffle_tools: bool = True,
    seed: int = 42
) -> Optional[Dict[str, Any]]:
    """Build a single adaptive training trajectory with <thought> CoT and tool schemas."""
    # 1. Base tools allowed by installed_apps
    tools = extract_available_tools(sample, catalog)
    installed_names = set(t["function"]["name"] for t in tools)

    # 2. Distractor Injection: Add 2-3 random tools from other apps not installed
    if inject_distractors:
        all_possible_tools = [
            node for name, node in skill_graph.nodes.items()
            if name not in installed_names and not name.startswith("sys.")
        ]
        if all_possible_tools:
            num_distractors = min(3, len(all_possible_tools))
            distractor_nodes = random.sample(all_possible_tools, num_distractors)
            for dnode in distractor_nodes:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": dnode.name,
                        "description": f"[{dnode.kind.upper()}] {dnode.description[:70]}",
                        "parameters": dnode.parameters
                    }
                })

    # 3. Tool Shuffling: Prevent position memorization
    if shuffle_tools:
        random.shuffle(tools)

    # 4. System Prompt
    system_prompt = build_system_prompt(sample)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 5. History turns
    for turn in sample.get("history", []):
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

    # 6. Current User Turn
    user_request = sample.get("audio_text", "").strip() or sample.get("text", "").strip()
    if not user_request:
        return None
    messages.append({"role": "user", "content": user_request})

    # 7. Target Assistant Output(s)
    gold_calls = sample.get("label", {}).get("tool_calls", [])
    gold_results = sample.get("label", {}).get("tool_results", [])

    if not gold_calls:
        # Case 1: Out of scope -> Model reasons that no tool matches and refuses cleanly
        cot = skill_graph.generate_meta_reasoning_cot(
            query=user_request,
            tool_call=None
        )
        messages.append({
            "role": "assistant",
            "content": f"{cot}\n\nYêu cầu này nằm ngoài phạm vi hỗ trợ của các ứng dụng hiện có trên thiết bị."
        })
    else:
        # Case 2: Multi-step or Single-step Tool Execution
        prev_tool_name = None
        for i, tc in enumerate(gold_calls):
            tname = tc["name"]
            targs = tc.get("arguments", {})

            cot = skill_graph.generate_meta_reasoning_cot(
                query=user_request,
                tool_call=tc,
                step=i + 1,
                total_steps=len(gold_calls),
                prev_tool_name=prev_tool_name
            )

            # Assistant turn with thought + function call
            messages.append({
                "role": "assistant",
                "content": cot,
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

            # If there's a corresponding result and subsequent call, append tool observation
            if i < len(gold_results) and (i + 1) < len(gold_calls):
                res = gold_results[i]
                messages.append({
                    "role": "tool",
                    "name": tname,
                    "content": json.dumps(res, ensure_ascii=False)
                })
                prev_tool_name = tname

    return {
        "id": sample["id"],
        "messages": messages,
        "tools": tools
    }


def generate_dataset(
    output_dir: str = "data/adaptive_sft",
    val_ratio: float = 0.1,
    seed: int = 42
):
    random.seed(seed)
    train_file, catalog_file = locate_data_files()
    print("=" * 70)
    print(" 🛠️ GENERATING ADAPTIVE SFT TRAJECTORIES (META-LEARNING)")
    print(f" Train Data:    {train_file}")
    print(f" Tool Catalog:  {catalog_file}")
    print(f" Output Dir:    {output_dir}")
    print("=" * 70)

    # 1. Initialize Skill Knowledge Graph
    with open(catalog_file, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
    skill_graph = SkillGraph(catalog_data)
    print(f"✅ Loaded Skill Knowledge Graph: {len(skill_graph.nodes)} tools, {len(skill_graph.dependencies)} dependencies.")

    # 2. Process all samples
    all_trajectories = []
    skipped = 0
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            traj = build_sample_trajectory(
                sample=sample,
                catalog=catalog_data,
                skill_graph=skill_graph,
                inject_distractors=True,
                shuffle_tools=True,
                seed=seed
            )
            if traj:
                all_trajectories.append(traj)
            else:
                skipped += 1

    print(f"✅ Successfully processed {len(all_trajectories)} trajectories (Skipped: {skipped}).")

    # 3. Train / Validation Split
    random.shuffle(all_trajectories)
    val_count = int(len(all_trajectories) * val_ratio)
    val_set = all_trajectories[:val_count]
    train_set = all_trajectories[val_count:]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_out = out_path / "adaptive_train.jsonl"
    val_out = out_path / "adaptive_val.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"🎉 DATASET READY:")
    print(f" • Train Samples: {len(train_set):>5} -> {train_out}")
    print(f" • Val Samples:   {len(val_set):>5} -> {val_out}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive SFT Data Generator")
    parser.add_argument("--output_dir", type=str, default="data/adaptive_sft",
                        help="Directory to save adaptive_train.jsonl and adaptive_val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of validation split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(output_dir=args.output_dir, val_ratio=args.val_ratio, seed=args.seed)
