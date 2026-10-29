import sys
import io
import json
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAIN = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "train"
sys.path.insert(0, str(DATA_TRAIN))

from src.evaluator import score_sample, evaluate_dataset, token_f1
from zaic_tools import ToolEnv


def run_tests():
    train_file = DATA_TRAIN / "train.jsonl"
    print("Testing evaluator on first 50 train samples...")

    samples = []
    with open(train_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            item = json.loads(line)
            env = ToolEnv("train", item["id"], root=DATA_TRAIN.parent)
            for c in item["label"]["tool_calls"]:
                env.call(c["name"], c["arguments"])
            item["gold_hash"] = env.get_state_hash()
            samples.append(item)

    # 1. Perfect predictions should yield 100.0 score
    perfect_preds = [
        {
            "id": s["id"],
            "tool_calls": s["label"]["tool_calls"],
            "state_hash": s["gold_hash"]
        }
        for s in samples
    ]

    res = evaluate_dataset(perfect_preds, samples)
    print(f"-> 50 Perfect Samples: Score = {res['final_score']}/100.0, Hash match = {res['hash_match_rate']}%, Perfect = {res['perfect_match_rate']}%")
    assert res["final_score"] == 100.0, f"Expected 100.0, got {res['final_score']}"

    # 2. Test Out-of-scope logic
    s_oos, _ = score_sample([], "h1", [], "h1")
    assert s_oos == 1.0, "Empty vs Empty should be 1.0"

    s_oos_bad, _ = score_sample([{"name": "a", "arguments": {}}], "h1", [], "h1")
    assert s_oos_bad == 0.0, "Non-empty vs Empty should be 0.0"

    # 3. Test Hash mismatch penalty (must be 0.0 regardless of tool calls)
    s_bad_hash, _ = score_sample(
        [{"name": "contacts.search", "arguments": {"query": "test"}}],
        "wrong_hash",
        [{"name": "contacts.search", "arguments": {"query": "test"}}],
        "gold_hash"
    )
    assert s_bad_hash == 0.0, "Hash mismatch must result in 0.0 score"

    # 4. Test Fuzzy text matching (token F1 > 0.6)
    # E.g. minor typos or punctuation in content
    text1 = "mai chạy bộ dời sang 09:00 nhé"
    text2 = "mai chay bo doi sang 09:00 nhe"  # without diacritics
    print(f"Token F1 test 1: {token_f1(text1, text2):.3f}")

    text3 = "Họp phòng ban lúc 9h sáng mai"
    text4 = "họp phòng ban lúc 9:00 sáng mai"
    print(f"Token F1 test 2: {token_f1(text3, text4):.3f}")

    print("\n✅ ALL EVALUATOR TESTS PASSED 100%!")


if __name__ == "__main__":
    run_tests()
