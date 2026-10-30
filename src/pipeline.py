"""End-to-End Pipeline for LifeMate.
Integrates ASR (Speech-to-Text), Context Assembly, LLM Agent Planning, and Tool Execution.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import LifeMateAgent


class LifeMatePipeline:
    """End-to-end pipeline handling both audio input and text fallback."""

    def __init__(
        self,
        agent: LifeMateAgent,
        asr_model: Optional[Any] = None,
    ):
        self.agent = agent
        self.asr_model = asr_model

    def process_sample(
        self,
        item: Dict[str, Any],
        split_dir: Path,
        split_name: str,
        data_root: Path,
        use_audio: bool = True
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Process a single sample through ASR -> Agent -> ToolEnv."""
        start_time = time.time()
        user_request_text = None

        # 1. Transcribe conversation history user turns if audio present without text
        if use_audio and self.asr_model is not None:
            for turn in item.get("history", []):
                if turn.get("role") == "user" and not turn.get("audio_text") and "audio" in turn:
                    hist_path = split_dir / turn["audio"]
                    if hist_path.is_file():
                        try:
                            turn["audio_text"] = self.asr_model.transcribe(hist_path)
                        except Exception:
                            pass

        # 2. Speech-to-Text (ASR) for final user audio
        if use_audio and self.asr_model is not None and "audio" in item:
            audio_rel_path = item["audio"]
            # Path can be relative to split directory
            audio_full_path = split_dir / audio_rel_path
            if audio_full_path.is_file():
                try:
                    user_request_text = self.asr_model.transcribe(audio_full_path)
                except Exception as e:
                    print(f"ASR error on {audio_full_path}: {e}")
                    user_request_text = item.get("audio_text", "")

        # Fallback to audio_text if ASR not used or failed
        if not user_request_text:
            user_request_text = item.get("audio_text", "")

        # 2. Run ReAct Agent on the transcribed text
        tool_calls, state_hash, meta = self.agent.run_sample(
            item=item,
            split=split_name,
            data_root=data_root,
            user_request_text=user_request_text
        )

        elapsed = time.time() - start_time
        meta["elapsed_time"] = round(elapsed, 2)
        meta["transcribed_text"] = user_request_text

        return tool_calls, state_hash, meta
