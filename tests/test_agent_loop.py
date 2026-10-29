import sys
import io
import json
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAIN = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "train"
sys.path.insert(0, str(DATA_TRAIN))

from src.agent import LifeMateAgent
from src.evaluator import score_sample


def test_agent_with_oracle():
    """Test the agent loop on train_0001 with an oracle generator simulating multi-step calling."""
    train_file = DATA_TRAIN / "train.jsonl"
    catalog_file = DATA_TRAIN / "train_tools_catalog.json"

    catalog = json.load(open(catalog_file, encoding="utf-8"))
    item = json.loads(open(train_file, encoding="utf-8").readline())

    gold_calls = item["label"]["tool_calls"]
    step_idx = 0

    def oracle_llm(messages, tools):
        nonlocal step_idx
        # Simulate returning 1 tool call at each step
        if step_idx < len(gold_calls):
            call = gold_calls[step_idx]
            step_idx += 1
            return {
                "content": f"Thực hiện bước {step_idx}: {call['name']}",
                "tool_calls": [call]
            }
        # Done
        return {"content": "Hoàn tất các bước.", "tool_calls": []}

    agent = LifeMateAgent(llm_generate_fn=oracle_llm, catalog=catalog, max_steps=10)

    tool_calls, state_hash, meta = agent.run_sample(
        item=item,
        split="train",
        data_root=DATA_TRAIN.parent
    )

    print(f"Sample: {item['id']}")
    print(f"Executed tool calls count: {len(tool_calls)}")
    print(f"Final state hash: {state_hash}")
    print(f"Steps taken: {meta['steps_taken']}")

    # Check score
    score, det = score_sample(
        pred_calls=tool_calls,
        pred_hash=state_hash,
        gold_calls=gold_calls,
        gold_hash=state_hash  # Replayed exactly
    )

    print(f"Score against gold: {score * 100:.1f}/100.0 (Precision={det['precision']:.2f}, Recall={det['recall']:.2f})")
    assert score == 1.0, f"Expected 1.0, got {score}"

    print("\n[SUCCESS] Agent loop test passed with perfect execution!")


if __name__ == "__main__":
    test_agent_with_oracle()
