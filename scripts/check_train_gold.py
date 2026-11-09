import json
import sys
import io
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_file = Path("data/train+public_test/train+public_test/train/train.jsonl")
sys.path.insert(0, str(train_file.parent))
from zaic_tools import ToolEnv

with open(train_file, "r", encoding="utf-8") as f:
    for i in range(3):
        item = json.loads(f.readline())
        sid = item["id"]
        gold_calls = item["label"]["tool_calls"]
        
        # Compute gold state hash by replaying gold calls in ToolEnv
        env = ToolEnv("train", sid, root=train_file.parent.parent)
        for c in gold_calls:
            env.call(c["name"], c.get("arguments", {}))
        gold_hash = env.get_state_hash()
        
        print(f"=== {sid} ===")
        print("Audio text:", item.get("audio_text"))
        print(f"Gold calls ({len(gold_calls)}):")
        for c in gold_calls:
            print("  *", c["name"], c.get("arguments"))
        print("Gold state hash:", gold_hash)
        print()
