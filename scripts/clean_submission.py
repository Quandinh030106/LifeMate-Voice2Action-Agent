import json
import sys
from pathlib import Path

PUB_DIR = Path("LifeMate Agentic AI/data/train+public_test/train+public_test/public_test")
sys.path.insert(0, str(PUB_DIR))
from zaic_tools import ToolEnv, catalog_path, schema_validate

cat = json.load(open(catalog_path("public_test", PUB_DIR.parent), encoding="utf-8"))
sub_file = Path("LifeMate Agentic AI/result/submission.jsonl")

fixed_lines = []
removed_calls = 0

for line in open(sub_file, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    allowed = ToolEnv("public_test", row["id"], PUB_DIR.parent).allowed
    calls = row.get("tool_calls") or []

    val_results = list(schema_validate(calls, cat, allowed))
    valid_calls = []
    for call, res in zip(calls, val_results):
        if res["ok"]:
            valid_calls.append(call)
        else:
            removed_calls += 1
            print(f"Removing invalid call from {row['id']}: {res['name']} - {res['error']}")

    row["tool_calls"] = valid_calls
    fixed_lines.append(row)

out_file = Path("LifeMate Agentic AI/result/submission.jsonl")
with open(out_file, "w", encoding="utf-8") as f:
    for row in fixed_lines:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nDone! Cleaned {removed_calls} invalid tool calls. Saved 100% valid submission to {out_file}!")
