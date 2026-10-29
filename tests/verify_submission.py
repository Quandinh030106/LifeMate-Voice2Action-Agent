import sys
import io
import json
from pathlib import Path

# Fix UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUB_DIR = PROJECT_ROOT / "data" / "train+public_test" / "train+public_test" / "public_test"
sys.path.insert(0, str(PUB_DIR))

from zaic_tools import ToolEnv, catalog_path, schema_validate


def verify_file(sub_path):
    cat = json.load(open(catalog_path("public_test", PUB_DIR.parent), encoding="utf-8"))
    sub_file = Path(sub_path)

    lines = bad = 0
    print(f"Validating: {sub_file}")
    for line in open(sub_file, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        lines += 1
        allowed = ToolEnv("public_test", row["id"], PUB_DIR.parent).allowed
        calls = row.get("tool_calls") or []
        for i, r in enumerate(schema_validate(calls, cat, allowed)):
            if not r["ok"]:
                bad += 1
                print(f"  [ERROR] {row['id']} call {i} {r['name']}: {r['error']}")

    print(f"\n=> Result: {lines} lines checked, {bad} schema errors (100% Valid: {bad == 0})")
    return bad == 0


if __name__ == "__main__":
    sub_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "result" / "submission_test.jsonl")
    verify_file(sub_path)
