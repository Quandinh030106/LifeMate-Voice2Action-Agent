"""LifeMate Submission Cleaner & State Hash Replayer.
Repairs raw submissions by:
1. Deduplicating repeated identical tool calls (resolves the ReAct infinite loop).
2. Performing Entity Resolution on placeholder IDs (acc_1234..., ben_1234...).
3. Filtering invalid / out-of-catalog tools.
4. Replaying valid tool calls in a fresh ToolEnv to compute the exact, clean state_hash.
5. Imputing any missing test samples so submission has exactly 634 lines.
Works seamlessly on both Kaggle and local environments.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure UTF-8 output on all operating systems including Windows cmd/powershell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate Submission Cleaner")
    parser.add_argument("--input", type=str, default="LifeMate Agentic AI/result/submission.jsonl",
                        help="Path to raw submission.jsonl.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save cleaned submission.jsonl (defaults to overwriting --input with backup).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root directory of train+public_test dataset.")
    parser.add_argument("--split", type=str, default="public_test",
                        help="Dataset split name (default: public_test).")
    return parser.parse_args()


def find_data_root(hint: str | None = None) -> Path:
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates.extend([
        Path("/kaggle/input/lifemate-data/train+public_test/train+public_test"),
        Path("LifeMate Agentic AI/data/train+public_test/train+public_test"),
        Path("data/train+public_test/train+public_test"),
    ])
    # Search recursively for public_test.jsonl if not found
    for cand in candidates:
        if (cand / "public_test" / "public_test.jsonl").is_file():
            return cand.resolve()

    found = glob.glob("**/public_test.jsonl", recursive=True)
    if found:
        return Path(found[0]).resolve().parent.parent

    raise FileNotFoundError("Could not locate public_test.jsonl in candidate directories.")


def clean_submission():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        # Fallback for kaggle
        if Path("/kaggle/working/submission.jsonl").is_file():
            input_path = Path("/kaggle/working/submission.jsonl")
        elif Path("result/submission.jsonl").is_file():
            input_path = Path("result/submission.jsonl")
        else:
            raise FileNotFoundError(f"Input submission file '{args.input}' not found!")

    output_path = Path(args.output) if args.output else input_path
    backup_path = input_path.with_name(f"{input_path.stem}_raw.jsonl")

    # Locate dataset
    data_root = find_data_root(args.data_dir)
    split_dir = data_root / args.split
    print(f"Dataset root: {data_root}")
    print(f"Reading raw submission: {input_path}")

    # Add split_dir to sys.path to import zaic_tools
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv, schema_validate, catalog_path

    # Load catalog and all samples in split
    catalog = json.load(open(catalog_path(args.split, data_root), encoding="utf-8"))
    split_file = split_dir / f"{args.split}.jsonl"

    all_samples = {}
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                all_samples[item["id"]] = item

    print(f"Total samples expected in {args.split}: {len(all_samples)}")

    # Read existing rows
    existing_rows = {}
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                existing_rows[row["id"]] = row

    print(f"Found {len(existing_rows)} rows in raw submission.")

    # Create backup
    if input_path == output_path and not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backed up raw submission to {backup_path}")

    cleaned_rows = []
    stats = {
        "dedup_removed": 0,
        "resolved_banking": 0,
        "invalid_removed": 0,
        "missing_imputed": 0,
        "hash_updated": 0,
    }

    for sid, item in all_samples.items():
        raw_row = existing_rows.get(sid)
        env = ToolEnv(args.split, sid, data_root)
        allowed = env.allowed

        if raw_row is None:
            # Impute missing sample with empty calls and clean baseline state hash
            baseline_hash = env.get_state_hash()
            cleaned_rows.append({
                "id": sid,
                "tool_calls": [],
                "state_hash": baseline_hash
            })
            stats["missing_imputed"] += 1
            continue

        raw_calls = raw_row.get("tool_calls") or []

        # Step 1: Remove illegal tools and consecutive duplicates
        clean_calls = []
        prev_sig = None
        for c in raw_calls:
            tname = c.get("name")
            if not tname or tname not in allowed:
                stats["invalid_removed"] += 1
                continue

            targs = c.get("arguments") or {}
            sig = (tname, json.dumps(targs, sort_keys=True, ensure_ascii=False))
            if sig == prev_sig:
                stats["dedup_removed"] += 1
                continue

            clean_calls.append(c)
            prev_sig = sig

        # Step 2: Replay calls into fresh ToolEnv, applying Entity Resolution and Schema Auto-Correction
        fresh_env = ToolEnv(args.split, sid, data_root)
        user_req_text = (item.get("audio_text") or item.get("text") or "").lower()

        for c in clean_calls:
            tname = c["name"]
            targs = dict(c.get("arguments") or {})

            # Parameter key normalizations
            if tname == "sms.send":
                if "phone" in targs and "phone_number" not in targs:
                    targs["phone_number"] = targs.pop("phone")
            elif tname in ["ride_hailing.book", "ride_hailing.get_quotes"]:
                v = targs.get("vehicle")
                if v not in ["bike", "car4", "car7"]:
                    targs["vehicle"] = "car4"
            elif tname == "gold_price.get_price":
                t = targs.get("type")
                if t not in ["sjc", "ring", "world"]:
                    targs["type"] = "sjc"
            elif tname == "bill_payment.pay":
                s = targs.get("source")
                if s not in ["bank", "ewallet"]:
                    targs["source"] = "bank"
            elif tname == "sys.ask_missing_info":
                if "params" not in targs or not isinstance(targs["params"], list):
                    targs["params"] = ["missing_info"]
                if "tool_name" not in targs or not isinstance(targs["tool_name"], str):
                    targs["tool_name"] = "general"

            # Banking transfer entity resolution
            if tname == "banking.transfer":
                from_acc = str(targs.get("from_account_id", ""))
                ben_id = str(targs.get("beneficiary_id", ""))
                if "1234" in from_acc or "1234" in ben_id or "account_" in from_acc or "beneficiary_" in ben_id or not from_acc or not ben_id:
                    # Query get_balance & list_beneficiaries
                    if "banking.get_balance" in allowed:
                        bal_res = fresh_env.call("banking.get_balance", {})
                        if bal_res.get("ok") and bal_res.get("accounts"):
                            targs["from_account_id"] = bal_res["accounts"][0]["id"]
                    if "banking.list_beneficiaries" in allowed:
                        ben_res = fresh_env.call("banking.list_beneficiaries", {})
                        if ben_res.get("ok") and ben_res.get("beneficiaries"):
                            note_ctx = (str(targs.get("note", "")) + " " + user_req_text).lower()
                            matched = None
                            for b in ben_res["beneficiaries"]:
                                if b.get("name", "").lower() in note_ctx:
                                    matched = b["id"]
                                    break
                            targs["beneficiary_id"] = matched if matched else ben_res["beneficiaries"][0]["id"]
                    stats["resolved_banking"] += 1

            # Validate against official schema before executing
            cand_call = {"name": tname, "arguments": targs}
            val_res = list(schema_validate([cand_call], catalog, allowed))[0]
            if not val_res["ok"]:
                stats["invalid_removed"] += 1
                continue

            # Execute in fresh env
            fresh_env.call(tname, targs)

        final_calls = fresh_env.get_tool_calls()
        final_hash = fresh_env.get_state_hash()

        if final_hash != raw_row.get("state_hash"):
            stats["hash_updated"] += 1

        cleaned_rows.append({
            "id": sid,
            "tool_calls": final_calls,
            "state_hash": final_hash
        })

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in cleaned_rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("✨ CLEANING SUMMARY:")
    print(f"  Total records in final output: {len(cleaned_rows)}")
    print(f"  Duplicate calls removed:       {stats['dedup_removed']}")
    print(f"  Banking transfers resolved:    {stats['resolved_banking']}")
    print(f"  Invalid tools removed:         {stats['invalid_removed']}")
    print(f"  Missing samples imputed:       {stats['missing_imputed']}")
    print(f"  State hashes corrected:        {stats['hash_updated']}")
    print(f"  Output saved to:               {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    clean_submission()
