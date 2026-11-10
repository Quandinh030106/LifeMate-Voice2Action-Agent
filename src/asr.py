"""ASR (Automatic Speech Recognition) module.
Supports:
1. G-Group Gipformer 1.5 (68M Zipformer-RNNT on CPU via sherpa-onnx) — Ultra-lightweight & blazing fast on CPU
2. VinAI PhoWhisper (vinai/PhoWhisper-large / medium) — Deep contextual Seq2Seq Transformer
3. OpenAI Whisper (openai/whisper-large-v3-turbo) — Multilingual baseline
4. NVIDIA Parakeet FastConformer-CTC — GPU non-autoregressive option
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

# ── Suppress known harmless transformers warnings ──────────────────────────────
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*suppress_tokens.*begin_suppress_tokens.*")
warnings.filterwarnings("ignore", message=".*multilingual Whisper.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
# ──────────────────────────────────────────────────────────────────────────────


def load_audio_file(audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
    """Load audio file, resample to 16kHz mono, and normalize peak volume."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    speech, sr = sf.read(str(audio_path))
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(speech) * target_sr / sr)
        speech = scipy.signal.resample(speech, num_samples)

    speech = speech.astype(np.float32)

    # Normalize peak amplitude to prevent quiet audio from being skipped or hallucinated
    max_val = np.max(np.abs(speech))
    if max_val > 0.001:
        speech = speech / max_val * 0.95

    return speech


class GipformerASR:
    """Wrapper for G-Group Gipformer 1.5 (68M Zipformer-Transducer).
    Runs purely on CPU via sherpa-onnx, using 0% GPU VRAM and completing in ~100-150ms per audio.
    """

    def __init__(
        self,
        model_id: str = "g-group-ai-lab/gipformer1.5-68M-rnnt",
        num_threads: int = 4
    ):
        try:
            import sherpa_onnx
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "Gipformer requires: pip install sherpa-onnx huggingface_hub soundfile"
            ) from e

        print(f"[Gipformer 1.5] Loading checkpoint ({model_id}) on CPU ({num_threads} threads)...")
        model_dir = snapshot_download(repo_id=model_id)

        # Detect ONNX model files (prefer INT8 for CPU speed if present)
        encoder_int8 = os.path.join(model_dir, "encoder.int8.onnx")
        decoder_int8 = os.path.join(model_dir, "decoder.int8.onnx")
        joiner_int8 = os.path.join(model_dir, "joiner.int8.onnx")

        encoder = encoder_int8 if os.path.isfile(encoder_int8) else os.path.join(model_dir, "encoder.onnx")
        decoder = decoder_int8 if os.path.isfile(decoder_int8) else os.path.join(model_dir, "decoder.onnx")
        joiner = joiner_int8 if os.path.isfile(joiner_int8) else os.path.join(model_dir, "joiner.onnx")

        # Locate tokens.txt
        tokens = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(tokens):
            for fname in os.listdir(model_dir):
                if "token" in fname.lower() and fname.endswith(".txt"):
                    tokens = os.path.join(model_dir, fname)
                    break

        if not os.path.isfile(tokens):
            bpe_path = os.path.join(model_dir, "bpe.model")
            if os.path.isfile(bpe_path):
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(bpe_path)
                tokens = os.path.join(model_dir, "tokens.txt")
                with open(tokens, "w", encoding="utf-8") as f:
                    for i in range(sp.get_piece_size()):
                        f.write(f"{sp.id_to_piece(i)} {i}\n")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            provider="cpu",
            decoding_method="greedy_search",
        )
        print("[Gipformer 1.5] Successfully initialized on CPU.")

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> str:
        """Transcribe audio to Vietnamese text on CPU."""
        if isinstance(audio, (str, Path)):
            speech = load_audio_file(audio, target_sr=16000)
        else:
            speech = np.asarray(audio, dtype=np.float32)
            if speech.ndim > 1:
                speech = speech.mean(axis=1)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, speech)
        self.recognizer.decode_streams([stream])
        return stream.result.text.strip()

    def transcribe_batch(self, audio_paths: List[Union[str, Path]], batch_size: int = 16) -> List[str]:
        """Transcribe a batch of audio files."""
        return [self.transcribe(p) for p in audio_paths]

    def to_cpu(self):
        """No-op, already running on CPU."""
        pass


class PhoWhisperASR:
    """Wrapper for PhoWhisper & Whisper sequence-to-sequence ASR models.
    Uses HuggingFace pipeline for optimal decoding, chunking, and high Vietnamese accuracy.
    """

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-large",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        from transformers import pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        device_idx = 0 if self.device.startswith("cuda") else -1

        print(f"Loading Vietnamese PhoWhisper ASR ({model_id}) on {self.device} ({self.dtype})...")
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=self.dtype,
                device=device_idx,
                model_kwargs={"low_cpu_mem_usage": True}
            )
        except Exception as e:
            fallback = "vinai/PhoWhisper-medium"
            print(f"[ASR Warning] Could not load {model_id} ({e}). Falling back to {fallback}...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=fallback,
                torch_dtype=self.dtype,
                device=device_idx,
                model_kwargs={"low_cpu_mem_usage": True}
            )

        self.model = self.pipe.model

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "vi"
    ) -> str:
        """Transcribe audio to Vietnamese text."""
        if isinstance(audio, (str, Path)):
            audio_input = str(Path(audio).resolve())
        else:
            audio_input = {"raw": audio, "sampling_rate": 16000}

        gen_kw = {"language": language, "task": "transcribe"}
        res = self.pipe(audio_input, return_timestamps=False, generate_kwargs=gen_kw)
        text = res["text"] if isinstance(res, dict) else str(res)
        return text.strip()

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 16,
        language: str = "vi"
    ) -> List[str]:
        """Transcribe a batch of audio files using the pipeline."""
        str_paths = [str(Path(p).resolve()) for p in audio_paths]
        gen_kw = {"language": language, "task": "transcribe"}
        results = self.pipe(str_paths, batch_size=batch_size, return_timestamps=False, generate_kwargs=gen_kw)
        return [r["text"].strip() if isinstance(r, dict) else str(r).strip() for r in results]

    def to_cpu(self):
        """Move underlying model to CPU to release GPU memory."""
        if hasattr(self, "pipe") and hasattr(self.pipe, "model"):
            try:
                self.pipe.model.cpu()
            except Exception:
                pass


class VietnameseASR:
    """Unified factory wrapper defaulting to G-Group Gipformer 1.5 on CPU.
    Preserves 100% of GPU VRAM for LLM inference.
    """

    def __init__(
        self,
        engine: str = "gipformer",
        model_id: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        engine: "gipformer" (default, 68M CPU ONNX), "phowhisper" (VinAI), "openai" (Whisper).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine_name = engine
        self._impl = None

        if engine in ("auto", "gipformer"):
            try:
                self._impl = GipformerASR(
                    model_id=model_id or "g-group-ai-lab/gipformer1.5-68M-rnnt"
                )
                self.engine_name = "gipformer"
            except Exception as e:
                print(f"[ASR Warning] Could not initialize Gipformer ({e}). Falling back to PhoWhisper...")
                self._impl = PhoWhisperASR(model_id="vinai/PhoWhisper-large", device=self.device)
                self.engine_name = "phowhisper"

        elif engine in ("phowhisper", "vinai"):
            default_model = model_id if (model_id and "gipformer" not in model_id.lower()) else "vinai/PhoWhisper-large"
            self._impl = PhoWhisperASR(model_id=default_model, device=self.device)
            self.engine_name = "phowhisper"

        elif engine in ("openai", "whisper"):
            default_model = model_id or "openai/whisper-large-v3-turbo"
            self._impl = PhoWhisperASR(model_id=default_model, device=self.device)
            self.engine_name = "openai"

        if self._impl is None:
            self._impl = PhoWhisperASR(model_id="vinai/PhoWhisper-large", device=self.device)
            self.engine_name = "phowhisper"

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> str:
        return self._impl.transcribe(audio)

    def transcribe_batch(self, audio_paths: List[Union[str, Path]], batch_size: int = 16) -> List[str]:
        return self._impl.transcribe_batch(audio_paths, batch_size=batch_size)

    def to_cpu(self):
        """Move underlying model to CPU if supported."""
        if hasattr(self, "_impl") and hasattr(self._impl, "to_cpu"):
            self._impl.to_cpu()
