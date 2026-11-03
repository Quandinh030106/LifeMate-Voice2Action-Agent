"""ASR (Automatic Speech Recognition) module.
Supports:
1. NVIDIA Parakeet FastConformer-CTC (nvidia/parakeet-ctc-0.6b-vi) — state-of-the-art accuracy & ultra-fast inference
2. VinAI PhoWhisper (vinai/PhoWhisper-small, medium, large) & OpenAI Whisper — robust fallback
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

# ── Suppress known harmless transformers warnings ──────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*suppress_tokens.*begin_suppress_tokens.*")
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


class NvidiaParakeetASR:
    """ASR implementation using NVIDIA Parakeet-CTC 0.6B (FastConformer-CTC).
    Extremely fast non-autoregressive transcription fine-tuned on Vietnamese.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b-vi",
        device: Optional[str] = None
    ):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "nemo_toolkit is not installed. Install via: pip install nemo_toolkit['asr']"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NVIDIA Parakeet ASR ({model_name}) on {self.device}...")

        # Load model from NeMo
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name
        )
        self.model.to(self.device)
        self.model.eval()
        print("NVIDIA Parakeet ASR loaded successfully.")

    @torch.inference_mode()
    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribe a single audio file."""
        audio_str = str(Path(audio_path).resolve())
        results = self.model.transcribe([audio_str], verbose=False)
        if isinstance(results, list) and len(results) > 0:
            res = results[0]
            text = res.text if hasattr(res, "text") else str(res)
            return text.strip()
        return ""

    @torch.inference_mode()
    def transcribe_batch(self, audio_paths: List[Union[str, Path]], batch_size: int = 16) -> List[str]:
        """Transcribe a batch of audio files in parallel for maximum GPU throughput."""
        str_paths = [str(Path(p).resolve()) for p in audio_paths]
        results = self.model.transcribe(str_paths, batch_size=batch_size, verbose=False)
        transcriptions = []
        for res in results:
            text = res.text if hasattr(res, "text") else str(res)
            transcriptions.append(text.strip())
        return transcriptions


class PhoWhisperASR:
    """Wrapper for PhoWhisper & Whisper sequence-to-sequence ASR models.
    Uses HuggingFace pipeline for optimal decoding, chunking, and highest Vietnamese accuracy.
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

        res = self.pipe(audio_input, return_timestamps=False)
        text = res["text"] if isinstance(res, dict) else str(res)
        return text.strip()

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 16
    ) -> List[str]:
        """Transcribe a batch of audio files using the pipeline."""
        str_paths = [str(Path(p).resolve()) for p in audio_paths]
        results = self.pipe(str_paths, batch_size=batch_size, return_timestamps=False)
        return [r["text"].strip() if isinstance(r, dict) else str(r).strip() for r in results]

    def to_cpu(self):
        """Move underlying model to CPU to release GPU memory."""
        if hasattr(self, "pipe") and hasattr(self.pipe, "model"):
            try:
                self.pipe.model.cpu()
            except Exception:
                pass


class VietnameseASR:
    """Unified factory wrapper defaulting to state-of-the-art VinAI PhoWhisper."""

    def __init__(
        self,
        engine: str = "phowhisper",
        model_id: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        engine: "phowhisper" (VinAI PhoWhisper, default), "openai" (Whisper), "nvidia" (Parakeet-CTC).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine_name = engine
        self._impl = None

        if engine in ("auto", "phowhisper"):
            default_model = model_id or "vinai/PhoWhisper-large"
            self._impl = PhoWhisperASR(model_id=default_model, device=self.device)
            self.engine_name = "phowhisper"

        elif engine in ("openai", "whisper"):
            default_model = model_id or "openai/whisper-large-v3-turbo"
            self._impl = PhoWhisperASR(model_id=default_model, device=self.device)
            self.engine_name = "openai"

        elif engine == "nvidia":
            nvidia_model = model_id or "nvidia/parakeet-ctc-0.6b-vi"
            self._impl = NvidiaParakeetASR(model_name=nvidia_model, device=self.device)
            self.engine_name = "nvidia"

        if self._impl is None:
            default_model = model_id or "vinai/PhoWhisper-large"
            self._impl = PhoWhisperASR(model_id=default_model, device=self.device)
            self.engine_name = "phowhisper"

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> str:
        return self._impl.transcribe(audio)

    def transcribe_batch(self, audio_paths: List[Union[str, Path]], batch_size: int = 16) -> List[str]:
        return self._impl.transcribe_batch(audio_paths, batch_size=batch_size)

    def to_cpu(self):
        """Move underlying model to CPU to prepare for GPU memory release."""
        if hasattr(self, "_impl") and hasattr(self._impl, "model"):
            try:
                self._impl.model.cpu()
            except Exception:
                pass
