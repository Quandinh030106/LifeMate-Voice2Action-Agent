"""Interactive Benchmark & Validation Script on 100 Train Samples.
Evaluates:
1. ASR Accuracy: Transcribed text vs Ground Truth -> WER & CER per sample.
2. Tool Accuracy: Matched vs Gold calls (Đúng / Thừa / Thiếu, Precision, Recall, F1).
3. State Hash: Replayed Gold Hash vs Predicted Hash (Exact Match).
4. Official Competition Score (0.0 to 100.0).
5. Latency: Elapsed time per sample (Budget: <= 30.0s).
Real-time live logging for every single sample.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import warnings
import logging
warnings.filterwarnings("ignore")
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import VietnameseASR
from src.llm_engine import LocalLLMEngine
from src.agent import LifeMateAgent
from src.pipeline import LifeMatePipeline
from src.evaluator import is_call_match, FREE_TEXT_KEYS


def normalize_for_metric(text: str) -> str:
    """Normalize text for fair metric comparison (lowercase, strip punctuation)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate standard Word Error Rate (WER)."""
    ref_words = normalize_for_metric(reference).split()
    hyp_words = normalize_for_metric(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate standard Character Error Rate (CER)."""
    ref_chars = list(normalize_for_metric(reference).replace(" ", ""))
    hyp_chars = list(normalize_for_metric(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate 100 Train Benchmark")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root folder containing train split.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of samples to benchmark (default: 100).")
    parser.add_argument("--llm_model", type=str, default="/kaggle/working/models/lifemate-qwen3-4b-merged",
                        help="Path or HuggingFace ID of LLM.")
    parser.add_argument("--asr_engine", type=str, default="phowhisper", choices=["phowhisper", "gipformer", "auto"],
                        help="ASR backend engine (default: phowhisper).")
    parser.add_argument("--no_audio", action="store_true",
                        help="Skip audio & ASR (text-only evaluation).")
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Maximum reasoning steps per sample (default: 6).")
    parser.add_argument("--output", type=str, default="/kaggle/working/benchmark_100_results.jsonl",
                        help="Path to save detailed benchmark log.")
    return parser.parse_args()


def find_data_path(hint: Optional[str] = None) -> Tuple[Path, Path, Path]:
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates.extend([
        Path("/kaggle/input/datasets/locvt06/zaic2026-track-lifemate"),
        Path("/kaggle/input/zaic2026-track-lifemate"),
        PROJECT_ROOT / "data" / "train+public_test" / "train+public_test",
        PROJECT_ROOT / "data" / "train+public_test",
        PROJECT_ROOT / "data",
    ])
    for cand in candidates:
        tr_file = cand / "train" / "train.jsonl"
        cat_file = cand / "train" / "train_tools_catalog.json"
        if tr_file.is_file() and cat_file.is_file():
            return cand.resolve(), tr_file.resolve(), cat_file.resolve()

    found_tr = glob.glob("/kaggle/input/**/train.jsonl", recursive=True) + glob.glob("**/train.jsonl", recursive=True)
    if found_tr:
        tr_p = Path(found_tr[0]).resolve()
        cat_p = tr_p.parent / "train_tools_catalog.json"
        return tr_p.parent.parent, tr_p, cat_p

    raise FileNotFoundError("Could not find train.jsonl and train_tools_catalog.json!")


def main():
    args = parse_args()
    data_root, train_file, catalog_file = find_data_path(args.data_dir)
    split_dir = train_file.parent

    # Add tools path to sys.path so ToolEnv is accessible
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    print("=" * 75)
    print(" 🚀 LIFEMATE AGENTIC AI — BENCHMARK 100 TRAIN SAMPLES ")
    print("=" * 75)
    print(f"Data Root:     {data_root}")
    print(f"Train File:    {train_file}")
    print(f"Catalog File:  {catalog_file}")
    print(f"Sample Limit:  {args.limit}")
    print(f"LLM Model:     {args.llm_model}")
    print(f"Use Audio:     {not args.no_audio} (ASR: {args.asr_engine})")
    print("=" * 75)

    with open(catalog_file, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Read items to process
    items_to_process = []
    with open(train_file, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx >= args.limit:
                break
            if line.strip():
                items_to_process.append(json.loads(line))

    # Phase 1: Transcribe audio upfront with ASR and completely free VRAM
    audio_transcripts: Dict[str, str] = {}
    if not args.no_audio:
        import gc
        import torch
        print(f"\n[1/3] Phase 1: Transcribing audio with ASR ({args.asr_engine})...")
        # If Kaggle has GPU 1 available, use GPU 1 for ASR, else GPU 0 or CPU
        asr_device = "cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else ("cuda:0" if torch.cuda.is_available() else "cpu")
        asr = VietnameseASR(engine=args.asr_engine, device=asr_device)

        audio_files = []
        seen_audios = set()
        for it in items_to_process:
            if "audio" in it and it["audio"]:
                rel = it["audio"].replace("\\", "/")
                if rel not in seen_audios:
                    seen_audios.add(rel)
                    audio_files.append((rel, split_dir / rel))
            for turn in it.get("history", []):
                if turn.get("role") == "user" and "audio" in turn and turn["audio"]:
                    rel = turn["audio"].replace("\\", "/")
                    if rel not in seen_audios:
                        seen_audios.add(rel)
                        audio_files.append((rel, split_dir / rel))

        print(f"Transcribing {len(audio_files)} unique audio recordings on {asr_device}...")
        t_asr_start = time.time()
        for rel_p, full_p in audio_files:
            if full_p.is_file():
                try:
                    audio_transcripts[rel_p] = asr.transcribe(full_p)
                except Exception as e:
                    print(f"ASR error on {rel_p}: {e}")
        print(f"Transcribed {len(audio_transcripts)} recordings in {time.time() - t_asr_start:.2f}s.")

        # Completely free ASR from VRAM so LLM has 100% dedicated memory
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ ASR unloaded from VRAM. GPU memory is 100% clean for LLM!\n")

    # Phase 2: Load LLM Planner on GPU
    print(f"[2/3] Phase 2: Loading LLM Planner ({args.llm_model})...")
    llm = LocalLLMEngine(model_name_or_path=args.llm_model)

    # Phase 3: Initialize Agent & Pipeline
    agent = LifeMateAgent(llm_generate_fn=llm.generate, catalog=catalog, max_steps=args.max_steps)
    pipeline = LifeMatePipeline(agent=agent, asr_model=None, audio_transcripts=audio_transcripts)

    # Metrics collectors
    latencies: List[float] = []
    wers: List[float] = []
    cers: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    hash_matches: List[bool] = []
    official_scores: List[float] = []

    out_records = []

    print("\n[3/3] Starting Benchmark Execution with Real-Time Logging...\n")
    print("=" * 75)

    for idx, item in enumerate(items_to_process):
        sid = item["id"]
        gold_calls = item["label"]["tool_calls"]
        gold_text = item.get("audio_text", "")

        # Replay gold calls in ToolEnv to obtain true gold_hash
        env_gold = ToolEnv("train", sid, root=data_root)
        for c in gold_calls:
            env_gold.call(c["name"], c.get("arguments", {}))
        gold_hash = env_gold.get_state_hash()

        t0 = time.time()

        # End-to-end sample processing
        pred_calls, pred_hash, meta = pipeline.process_sample(
            item=item,
            split_dir=split_dir,
            split_name="train",
            data_root=data_root,
            use_audio=not args.no_audio
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        asr_text = meta.get("transcribed_text") or gold_text

        # 1. Compute ASR Metrics
        if not args.no_audio and gold_text:
            sample_wer = calculate_wer(gold_text, asr_text)
            sample_cer = calculate_cer(gold_text, asr_text)
            wers.append(sample_wer)
            cers.append(sample_cer)
        else:
            sample_wer, sample_cer = 0.0, 0.0

        # 2. Compute Tool Call Metrics
        # Greedy bipartite matching
        matched_gold_indices = set()
        matched_count = 0
        for p in pred_calls:
            for j, g in enumerate(gold_calls):
                if j not in matched_gold_indices and is_call_match(p, g, FREE_TEXT_KEYS):
                    matched_gold_indices.add(j)
                    matched_count += 1
                    break

        M = matched_count
        G = len(gold_calls)
        P = len(pred_calls)
        thua = P - M
        thieu = G - M

        if G == 0 and P == 0:
            prec, rec, f1 = 1.0, 1.0, 1.0
        elif M == 0:
            prec = 0.0 if P > 0 else 1.0
            rec = 0.0 if G > 0 else 1.0
            f1 = 0.0
        else:
            prec = M / P if P > 0 else 0.0
            rec = M / G if G > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        # 3. State Hash Match & Score
        is_hash_match = (pred_hash == gold_hash)
        hash_matches.append(is_hash_match)

        score = (f1 * 100.0) if is_hash_match else 0.0
        official_scores.append(score)

        # 4. Live Display on Log
        hash_icon = "MATCH ✅" if is_hash_match else "MISMATCH ❌"
        pred_names = [c["name"] for c in pred_calls]
        gold_names = [c["name"] for c in gold_calls]

        print(f"[{idx+1:03d}/{args.limit:03d}] ({(idx+1)/args.limit*100:5.1f}%) Sample: {sid} | Thời gian: {elapsed:.2f}s")
        if not args.no_audio:
            print(f"🎙️ ASR Text:  \"{asr_text}\"")
            print(f"📝 Gold Text: \"{gold_text}\"")
            print(f"   • ASR Metrics:  WER = {sample_wer*100:5.1f}% | CER = {sample_cer*100:5.1f}%")
        else:
            print(f"📝 User Request: \"{gold_text}\"")

        print(f"🛠️ Tool Calls:  Đúng: {M}/{G} | Thừa: {thua} | Thiếu: {thieu} (P={prec:.2f}, R={rec:.2f}, F1={f1:.2f})")
        print(f"   • Pred: {pred_names}")
        print(f"   • Gold: {gold_names}")
        print(f"🔑 State Hash:  {hash_icon} (Pred: {pred_hash[:10]}... | Gold: {gold_hash[:10]}...)")
        print(f"🏆 Score:       {score:5.1f} / 100.0")
        print("-" * 75)

        out_records.append({
            "id": sid,
            "elapsed": elapsed,
            "gold_text": gold_text,
            "asr_text": asr_text,
            "wer": sample_wer,
            "cer": sample_cer,
            "gold_calls": gold_calls,
            "pred_calls": pred_calls,
            "matched": M,
            "thua": thua,
            "thieu": thieu,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "hash_match": is_hash_match,
            "score": score
        })

    # Save detailed output file
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fout:
        for r in out_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final Summary Report
    total_samples = len(latencies)
    avg_latency = sum(latencies) / total_samples if total_samples else 0.0
    avg_wer = (sum(wers) / len(wers) * 100) if wers else 0.0
    avg_cer = (sum(cers) / len(cers) * 100) if cers else 0.0
    avg_prec = sum(precisions) / total_samples * 100 if total_samples else 0.0
    avg_rec = sum(recalls) / total_samples * 100 if total_samples else 0.0
    avg_f1 = sum(f1s) / total_samples * 100 if total_samples else 0.0
    hash_match_rate = sum(1 for m in hash_matches if m) / total_samples * 100 if total_samples else 0.0
    mean_official_score = sum(official_scores) / total_samples if total_samples else 0.0
    perfect_samples = sum(1 for s in official_scores if s >= 99.9)

    print("\n" + "=" * 75)
    print(" 📊 TỔNG KẾT BENCHMARK TOÀN DIỆN TRÊN TẬP TRAIN ")
    print("=" * 75)
    print(f"Tổng số mẫu đánh giá:      {total_samples} / {args.limit}")
    print(f"1. TỐC ĐỘ THỰC THI:")
    print(f"   • Tổng thời gian:        {sum(latencies):.2f} s")
    print(f"   • Thời gian TB / mẫu:    {avg_latency:.2f} s  (Budget cuộc thi: <= 30.0s) -> {'ĐẠT CHUẨN ✅' if avg_latency <= 30 else 'CẦN TỐI ƯU ⚠️'}")

    if not args.no_audio:
        print(f"2. ĐỘ CHÍNH XÁC NHẬN DẠNG GIỌNG NÓI (ASR):")
        print(f"   • Average WER:           {avg_wer:.2f} %")
        print(f"   • Average CER:           {avg_cer:.2f} %")

    print(f"3. ĐỘ CHÍNH XÁC GỌI TOOL (FUNCTION CALLING):")
    print(f"   • Precision trung bình:  {avg_prec:.2f} %")
    print(f"   • Recall trung bình:     {avg_rec:.2f} %")
    print(f"   • F1-Score trung bình:   {avg_f1:.2f} %")

    print(f"4. TỶ LỆ KHỚP STATE HASH:")
    print(f"   • Trùng khớp hoàn toàn:  {sum(1 for m in hash_matches if m)} / {total_samples}  ({hash_match_rate:.2f} %)")

    print(f"5. ĐIỂM SỐ ZALO AI CHALLENGE CHÍNH THỨC:")
    print(f"   • Điểm trung bình (Score): {mean_official_score:.2f} / 100.0")
    print(f"   • Số mẫu đạt điểm tối đa: {perfect_samples} / {total_samples} ({(perfect_samples/total_samples*100 if total_samples else 0):.1f} %)")
    print("=" * 75)
    print(f"Chi tiết từng mẫu đã lưu tại: {out_p.resolve()}")


if __name__ == "__main__":
    main()
