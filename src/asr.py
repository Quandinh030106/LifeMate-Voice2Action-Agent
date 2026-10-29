"""ASR (Automatic Speech Recognition) module using PhoWhisper / Whisper.
Transcribes spoken Vietnamese requests into normalized text.
"""
from __future__ import annotations

import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Union
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class VietnameseASR:
    """Wrapper for Vietnamese speech-to-text models."""

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-small",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = torch_dtype

        print(f"Loading ASR model {model_id} on {self.device} ({self.dtype})...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True
            ).to(self.device)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
        self.model.eval()

    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
        """Load audio file and resample to 16kHz mono if needed."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        speech, sr = sf.read(str(audio_path))

        # Convert to mono if stereo
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)

        # Simple linear resample if sample rate doesn't match 16kHz
        if sr != target_sr:
            import scipy.signal
            num_samples = int(len(speech) * target_sr / sr)
            speech = scipy.signal.resample(speech, num_samples)

        return speech.astype(np.float32)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "vi"
    ) -> str:
        """Transcribe audio to Vietnamese text deterministically (temperature=0)."""
        if isinstance(audio, (str, Path)):
            speech = self.load_audio(audio)
        else:
            speech = audio

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        # Generate deterministic transcription
        # For PhoWhisper, language='vi' and task='transcribe'
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )

        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256
        )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()
