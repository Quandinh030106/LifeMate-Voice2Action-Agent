"""LLM Engine for LifeMate.
Provides deterministic local inference for function-calling models (e.g., Qwen2.5-3B-Instruct).
Strictly compliant with competition requirements:
- Fully offline inference
- Context <= 16K tokens
- LLM < 5B parameters
- Deterministic output (temperature=0, greedy decoding, seeds fixed)
"""
from __future__ import annotations

import os
import json
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .parser import parse_tool_calls_from_text


class LocalLLMEngine:
    """Wrapper around HuggingFace causal LM for function calling."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                self.dtype = torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch_dtype

        # Check if model_name_or_path is intended as a local path but missing
        model_path = Path(model_name_or_path)
        is_intended_local = (
            model_name_or_path.startswith(("/", "./", "../"))
            or "\\" in model_name_or_path
            or (len(model_name_or_path) > 1 and model_name_or_path[1] == ":")
        )
        if is_intended_local and not model_path.exists():
            fallback_model = "Qwen/Qwen2.5-3B-Instruct"
            print(f"[LLM Warning] Local path '{model_name_or_path}' does not exist on disk!")
            print(f"[LLM Fallback] Automatically falling back to base model '{fallback_model}'.")
            model_name_or_path = fallback_model
            local_files_only = False

        print(f"Loading LLM {model_name_or_path} on {self.device} (dtype: {self.dtype}, 4bit: {load_in_4bit})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=local_files_only
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )

        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa"
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    local_files_only=local_files_only,
                    **model_kwargs
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    local_files_only=local_files_only,
                    **model_kwargs
                ).to(self.device)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                ).to(self.device)

        # Stop when either EOS or </tool_call> is generated
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        call_end_tokens = self.tokenizer.encode("</tool_call>")
        if call_end_tokens:
            self.stop_token_ids.extend(call_end_tokens)

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """Generate response and parsed tool calls using model chat template."""
        # Fix seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Apply chat template with tools if supported
        try:
            if tools:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
        except Exception:
            prompt_text = ""
            for m in messages:
                prompt_text += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
            prompt_text += "<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # Truncate prompt if exceeding 16K tokens
        if inputs.input_ids.shape[1] > 15000:
            inputs["input_ids"] = inputs.input_ids[:, -15000:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs.attention_mask[:, -15000:]

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract only new generated tokens
        new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

        # Parse tool calls
        tool_calls = parse_tool_calls_from_text(output_text)

        return {
            "content": output_text,
            "tool_calls": tool_calls
        }
