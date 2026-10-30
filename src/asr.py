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
    """Load audio file and resample to 16kHz mono if needed."""
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

    return speech.astype(np.float32)


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
    """Wrapper for PhoWhisper / Whisper sequence-to-sequence ASR models."""

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-small",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        print(f"Loading PhoWhisper/Whisper ASR ({model_id}) on {self.device} ({self.dtype})...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=True,
                clean_up_tokenization_spaces=False,
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(self.device)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                clean_up_tokenization_spaces=False,
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
            self.model.generation_config.forced_decoder_ids = None

        self.model.eval()

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "vi"
    ) -> str:
        """Transcribe audio to Vietnamese text deterministically."""
        if isinstance(audio, (str, Path)):
            speech = load_audio_file(audio)
        else:
            speech = audio

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        predicted_ids = self.model.generate(
            input_features,
            task="transcribe",
            language=language,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcription.strip()

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> List[str]:
        """Transcribe audio files sequentially or in small batches."""
        return [self.transcribe(p) for p in audio_paths]


class VietnameseASR:
    """Unified factory wrapper supporting both NVIDIA Parakeet and PhoWhisper/Whisper."""

    def __init__(
        self,
        engine: str = "auto",
        model_id: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        engine: "nvidia" (Parakeet), "phowhisper" (VinAI), or "auto" (tries nvidia, falls back to phowhisper).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine_name = engine
        self._impl = None

        if engine in ("auto", "nvidia"):
            try:
                nvidia_model = model_id or "nvidia/parakeet-ctc-0.6b-vi"
                self._impl = NvidiaParakeetASR(model_name=nvidia_model, device=self.device)
                self.engine_name = "nvidia"
            except Exception as e:
                if engine == "nvidia":
                    raise e
                print(f"[ASR] NeMo / NVIDIA model not available ({e}). Falling back to PhoWhisper.")

        if self._impl is None:
            whisper_model = model_id or "vinai/PhoWhisper-small"
            self._impl = PhoWhisperASR(model_id=whisper_model, device=self.device)
            self.engine_name = "phowhisper"

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> str:
        return self._impl.transcribe(audio)

    def transcribe_batch(self, audio_paths: List[Union[str, Path]], batch_size: int = 16) -> List[str]:
        return self._impl.transcribe_batch(audio_paths, batch_size=batch_size)
