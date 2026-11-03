"""ASR Benchmark Script for LifeMate (Zalo AI Challenge 2026).
Compares speech-to-text accuracy, character accuracy, and CPU latency across 4 Top Vietnamese ASR Models:
1. VinAI PhoWhisper-large (384M - Seq2Seq Transformer)
2. G-Group Gipformer 1.5 (68M - Zipformer-Transducer ONNX)
3. 5CD-AI VieGem-ASR (359M - Google EmbeddingGemma-2 CTC)
4. Nguyen Vu Le Binh Wav2Vec2-Large (317M - Pretrained on 13k hours, VLSP2020)

Evaluates directly against ground-truth audio_text from train.jsonl.
Engineered to run 100% on CPU (0% GPU VRAM used), preserving all GPU resources for LLM inference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# UTF-8 stdout configuration for all consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Text Distance Metrics ──────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Normalize text for fair metric comparison (lowercase, strip punctuation)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate standard Word Error Rate (WER)."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
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
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
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


# ── Model 1: VinAI PhoWhisper ──────────────────────────────────────────────────
class PhoWhisperRunner:
    """Runner for VinAI PhoWhisper-large (or medium) on CPU."""

    def __init__(self, model_id: str = "vinai/PhoWhisper-large", device: str = "cpu"):
        self.name = f"VinAI {model_id.split('/')[-1]}"
        self.params = "384M" if "large" in model_id else "306M"
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.dtype = torch.float32

        print(f"[{self.name}] Loading model {model_id} on {device} ({self.dtype})...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        print(f"[{self.name}] Loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        import soundfile as sf
        import torch

        speech, sr = sf.read(audio_path, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            speech = scipy.signal.resample(speech, int(len(speech) * 16000 / sr))

        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()


# ── Model 2: G-Group Gipformer 1.5 ─────────────────────────────────────────────
class GipformerRunner:
    """Runner for G-Group Gipformer 1.5 using sherpa-onnx on CPU."""

    def __init__(self, device: str = "cpu"):
        self.name = "Gipformer 1.5 (68M)"
        self.params = "68M"
        try:
            import sherpa_onnx
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("Please install: pip install sherpa-onnx huggingface_hub soundfile")

        print(f"[{self.name}] Downloading / loading checkpoint from HuggingFace...")
        model_dir = snapshot_download(repo_id="g-group-ai-lab/gipformer1.5-68M-rnnt")

        # sherpa-onnx runs CPU natively with multi-threading
        provider = "cpu"

        # Check for INT8 or FP32 onnx files
        encoder_int8 = os.path.join(model_dir, "encoder.int8.onnx")
        decoder_int8 = os.path.join(model_dir, "decoder.int8.onnx")
        joiner_int8 = os.path.join(model_dir, "joiner.int8.onnx")

        encoder = encoder_int8 if os.path.isfile(encoder_int8) else os.path.join(model_dir, "encoder.onnx")
        decoder = decoder_int8 if os.path.isfile(decoder_int8) else os.path.join(model_dir, "decoder.onnx")
        joiner = joiner_int8 if os.path.isfile(joiner_int8) else os.path.join(model_dir, "joiner.onnx")

        # Locate tokens file
        tokens = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(tokens):
            for fname in os.listdir(model_dir):
                if "token" in fname.lower() and fname.endswith(".txt"):
                    tokens = os.path.join(model_dir, fname)
                    break

        if not os.path.isfile(tokens):
            # Fallback: extract tokens from bpe.model if tokens.txt is absent
            bpe_path = os.path.join(model_dir, "bpe.model")
            if os.path.isfile(bpe_path):
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(bpe_path)
                tokens = os.path.join(model_dir, "tokens.txt")
                with open(tokens, "w", encoding="utf-8") as f:
                    for i in range(sp.get_piece_size()):
                        f.write(f"{sp.id_to_piece(i)} {i}\n")

        print(f"[{self.name}] Initializing OfflineRecognizer (provider={provider}, tokens={tokens})...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            provider=provider,
            decoding_method="greedy_search",
        )
        print(f"[{self.name}] Loaded successfully on {provider}.")

    def transcribe(self, audio_path: str) -> str:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        self.recognizer.decode_streams([stream])
        return stream.result.text.strip()


# ── Model 3: 5CD-AI VieGem-ASR ─────────────────────────────────────────────────
class VieGemRunner:
    """Runner for 5CD-AI VieGem-ASR (359M - Google EmbeddingGemma-2 CTC)."""

    def __init__(self, device: str = "cpu"):
        self.name = "5CD-AI VieGem-ASR"
        self.params = "359M"
        self.device = device
        import sys
        import torch
        from huggingface_hub import snapshot_download

        print(f"[{self.name}] Downloading checkpoint from 5CD-AI/VieGem-ASR...")
        model_dir = snapshot_download(repo_id="5CD-AI/VieGem-ASR")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        # Import modules from the repo's asr/ folder
        from asr.model import CTCModel, ModelConfig, greedy_decode
        from asr.text import Units
        from transformers import AutoTokenizer

        ckpt_path = os.path.join(model_dir, "viegem_asr.pt")
        ck = torch.load(ckpt_path, map_location=device)
        cfg = ModelConfig.from_dict(ck["model_config"])
        units = Units(ck["gemma_ids"])

        try:
            self.tok = AutoTokenizer.from_pretrained(cfg.encoder_id)
        except Exception as e:
            raise RuntimeError(
                f"VieGem requires access to tokenizer '{cfg.encoder_id}'. "
                f"Please run 'huggingface-cli login' with an account that accepted Google Gemma terms: {e}"
            )

        self.model = CTCModel(cfg, units.gemma_ids)
        self.model.load_state_dict(ck["model"], strict=False)
        self.model = self.model.to(device).eval()
        self.units = units
        self.greedy_decode = greedy_decode
        print(f"[{self.name}] Loaded successfully on {device}.")

    def transcribe(self, audio_path: str) -> str:
        import soundfile as sf
        import torch
        import scipy.signal

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000

        x = torch.from_numpy(audio)[None].to(self.device)
        with torch.no_grad():
            lens = torch.tensor([x.shape[1]], device=self.device)
            logits, lens_out = self.model(x, lens, streaming=False)
            token_ids = self.greedy_decode(logits, lens_out)[0]
            text = self.units.decode(self.tok, token_ids)
        return text.strip()


# ── Model 4: Nguyen Vu Le Binh Wav2Vec2-Large ──────────────────────────────────
class Wav2Vec2BinhRunner:
    """Runner for nguyenvulebinh/wav2vec2-large-vi-vlsp2020 (317M).

    Pretrained on 13,000 hours of conversational/street Vietnamese speech.
    Non-autoregressive CTC architecture = blazingly fast on CPU.
    """

    def __init__(self, model_id: str = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020", device: str = "cpu"):
        self.name = "Binh-Nguyen Wav2Vec2"
        self.params = "317M"
        self.device = device
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        print(f"[{self.name}] Loading model {model_id} on {device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            from transformers import Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)

        self.model = AutoModelForCTC.from_pretrained(model_id).to(device)
        self.model.eval()
        print(f"[{self.name}] Loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        import soundfile as sf
        import torch
        import scipy.signal

        speech, sr = sf.read(audio_path, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = scipy.signal.resample(speech, int(len(speech) * 16000 / sr))

        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()


# ── Baseline Option: OpenAI Whisper-Turbo ─────────────────────────────────────
class WhisperTurboRunner:
    """Runner for OpenAI Whisper-large-v3-turbo on CPU."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cpu"):
        self.name = "OpenAI Whisper-Turbo"
        self.params = "809M"
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.dtype = torch.float32

        print(f"[{self.name}] Loading model {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        print(f"[{self.name}] Loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        import soundfile as sf
        import torch
        import scipy.signal

        speech, sr = sf.read(audio_path, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = scipy.signal.resample(speech, int(len(speech) * 16000 / sr))

        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                task="transcribe",
                language="vi",
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()


# ── Main Benchmark Engine ──────────────────────────────────────────────────────
def locate_train_data(hint_dir: Optional[str] = None) -> Tuple[Path, Path]:
    candidates = []
    if hint_dir:
        candidates.append(Path(hint_dir))
    candidates.extend([
        Path("/kaggle/input/lifemate-data/train+public_test/train+public_test"),
        Path("LifeMate Agentic AI/data/train+public_test/train+public_test"),
        Path("data/train+public_test/train+public_test"),
    ])
    for cand in candidates:
        train_file = cand / "train" / "train.jsonl"
        if train_file.is_file():
            return train_file.resolve(), (cand / "train").resolve()

    found = glob.glob("**/train.jsonl", recursive=True)
    if found:
        tf = Path(found[0]).resolve()
        return tf, tf.parent

    raise FileNotFoundError("Could not locate train.jsonl dataset!")


def run_benchmark():
    parser = argparse.ArgumentParser(description="LifeMate Pure CPU ASR Arena Benchmark")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of real audio samples to benchmark (default: 5).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to train+public_test directory.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference (default: 'cpu' to preserve 100% GPU for LLM).")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of CPU threads for PyTorch/ONNX (default: 4).")
    parser.add_argument("--include_whisper", action="store_true",
                        help="Also benchmark OpenAI Whisper-Turbo.")
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cpu":
        torch.set_num_threads(args.threads)

    train_file, train_dir = locate_train_data(args.data_dir)
    print("=" * 80)
    print(" 🎙️ LIFEMATE VIETNAMESE ASR ARENA: 4-MODEL CPU BENCHMARK")
    print(f" Dataset train.jsonl: {train_file}")
    print(f" Compute Device:      {device} (Threads: {args.threads})")
    print(f" Samples to test:     {args.num_samples}")
    print("=" * 80)

    # 1. Load benchmark audio samples with ground-truth audio_text
    test_cases = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            audio_rel = item.get("audio")
            gold_text = item.get("audio_text") or item.get("text")
            if audio_rel and gold_text:
                audio_full = train_dir / audio_rel
                if audio_full.is_file():
                    test_cases.append({
                        "id": item["id"],
                        "audio_path": str(audio_full),
                        "gold_text": gold_text.strip()
                    })
                    if len(test_cases) >= args.num_samples:
                        break

    print(f"\n✅ Successfully loaded {len(test_cases)} evaluation samples with Ground-Truth text.\n")

    # 2. Initialize Models to Compare
    runners = []

    # Model 1: VinAI PhoWhisper-large
    try:
        pho = PhoWhisperRunner(model_id="vinai/PhoWhisper-large", device=device)
        runners.append(pho)
    except Exception as e:
        print(f"⚠️ Could not load PhoWhisper-large: {e}")

    # Model 2: G-Group Gipformer 1.5
    try:
        gip = GipformerRunner(device=device)
        runners.append(gip)
    except Exception as e:
        print(f"⚠️ Could not load Gipformer 1.5: {e}")

    # Model 3: 5CD-AI VieGem-ASR
    try:
        viegem = VieGemRunner(device=device)
        runners.append(viegem)
    except Exception as e:
        print(f"⚠️ Could not load VieGem-ASR: {e}")

    # Model 4: Nguyen Vu Le Binh Wav2Vec2-Large
    try:
        binh = Wav2Vec2BinhRunner(device=device)
        runners.append(binh)
    except Exception as e:
        print(f"⚠️ Could not load Wav2Vec2-VLSP2020: {e}")

    # Baseline: OpenAI Whisper Large-v3-Turbo
    if args.include_whisper:
        try:
            whisp = WhisperTurboRunner(device=device)
            runners.append(whisp)
        except Exception as e:
            print(f"⚠️ Could not load Whisper-Turbo: {e}")

    if not runners:
        print("❌ Error: No ASR models could be initialized.")
        return

    # 3. Execute Benchmark & Record Metrics
    stats: Dict[str, Dict[str, Any]] = {
        r.name: {
            "params": r.params,
            "wers": [],
            "cers": [],
            "times": [],
            "transcripts": []
        }
        for r in runners
    }

    print("\n" + "=" * 80)
    print("🏃 RUNNING INFERENCE ON CPU (Preserving 100% GPU VRAM)...")
    print("=" * 80)

    for idx, case in enumerate(test_cases, 1):
        sid = case["id"]
        audio_p = case["audio_path"]
        gold = case["gold_text"]

        print(f"\n────────────────────────────────────────────────────────────────────────────────")
        print(f"📌 [SAMPLE {idx}/{len(test_cases)}] ID: {sid}")
        print(f"🎯 GROUND TRUTH: \"{gold}\"")
        print(f"────────────────────────────────────────────────────────────────────────────────")

        for r in runners:
            t0 = time.time()
            try:
                pred = r.transcribe(audio_p)
                elapsed = time.time() - t0
                wer = calculate_wer(gold, pred)
                cer = calculate_cer(gold, pred)
                acc = max(0.0, 1.0 - wer) * 100

                stats[r.name]["wers"].append(wer)
                stats[r.name]["cers"].append(cer)
                stats[r.name]["times"].append(elapsed)
                stats[r.name]["transcripts"].append(pred)

                print(f"  • {r.name:<26} | {elapsed*1000:6.1f} ms | Word Acc: {acc:5.1f}% | \"{pred}\"")
            except Exception as ex:
                print(f"  • {r.name:<26} | ERROR: {ex}")

    # 4. Final Scoreboard Summary
    print("\n\n" + "=" * 80)
    print("📊 FINAL 4-MODEL CPU SCOREBOARD (Averaged over all test samples):")
    print("=" * 80)
    print(f"{'Model Name':<28} | {'Params':<7} | {'Word Accuracy':<14} | {'Char Accuracy':<14} | {'Avg CPU Latency'}")
    print("-" * 80)

    best_model = None
    best_acc = -1.0

    for name, s in stats.items():
        if s["wers"]:
            avg_wer = sum(s["wers"]) / len(s["wers"])
            avg_cer = sum(s["cers"]) / len(s["cers"])
            avg_time = sum(s["times"]) / len(s["times"])
            word_acc = max(0.0, 1.0 - avg_wer) * 100
            char_acc = max(0.0, 1.0 - avg_cer) * 100

            if word_acc > best_acc:
                best_acc = word_acc
                best_model = name

            print(f"{name:<28} | {s['params']:<7} | {word_acc:12.1f}% | {char_acc:12.1f}% | {avg_time*1000:6.1f} ms / audio")

    print("=" * 80)
    if best_model:
        print(f"🏆 CPU CHAMPION: {best_model} with {best_acc:.1f}% Word Accuracy!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
