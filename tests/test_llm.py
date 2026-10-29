import sys
import io
import json
import torch
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAIN = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "train"
sys.path.insert(0, str(DATA_TRAIN))

from src.llm_engine import LocalLLMEngine
from src.prompt_builder import extract_available_tools, build_messages


def test_qwen_tool_calling():
    print("Testing Qwen LLM for tool calling...")

    # Load 1.5B model on GPU
    llm = LocalLLMEngine(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    catalog = json.load(open(DATA_TRAIN / "train_tools_catalog.json", encoding="utf-8"))
    item = json.loads(open(DATA_TRAIN / "train.jsonl", encoding="utf-8").readline())

    tools = extract_available_tools(item, catalog)
    messages = build_messages(item, catalog, user_request_text=item["audio_text"])

    print(f"Sample: {item['id']}")
    print(f"User request: {item['audio_text']}")
    print(f"Tools passed: {len(tools)}")

    print("\nGenerating model output...")
    res = llm.generate(messages, tools=tools, max_new_tokens=256)

    print("\n--- LLM Output ---")
    print("Content:", res.get("content"))
    print("Parsed Tool Calls:", json.dumps(res.get("tool_calls", []), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_qwen_tool_calling()
