"""Robust Tool Call Parser for LifeMate Agent.
Parses tool calls from LLM textual outputs in various formats:
- Qwen2.5 native <tool_call> tags
- Markdown fenced code blocks (```json ... ```)
- Raw JSON arrays or single tool call objects
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def clean_json_string(s: str) -> str:
    """Clean string for robust json decoding."""
    s = s.strip()
    # Remove leading/trailing backticks if any
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract ordered tool calls from model output."""
    if not text:
        return []

    tool_calls: List[Dict[str, Any]] = []

    # 1. Check for <tool_call> tags (Qwen 2.5 format)
    tag_matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if tag_matches:
        for m in tag_matches:
            cleaned = clean_json_string(m)
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append({
                        "name": data["name"],
                        "arguments": data.get("arguments", {})
                    })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            tool_calls.append({
                                "name": item["name"],
                                "arguments": item.get("arguments", {})
                            })
            except Exception:
                pass

        if tool_calls:
            return tool_calls

    # 2. Check for markdown json block
    json_blocks = re.findall(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    for b in json_blocks:
        try:
            data = json.loads(b)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        tool_calls.append({
                            "name": item["name"],
                            "arguments": item.get("arguments", {})
                        })
            elif isinstance(data, dict) and "name" in data:
                tool_calls.append({
                    "name": data["name"],
                    "arguments": data.get("arguments", {})
                })
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    # 3. Direct JSON array matching [ { "name": ... }, ... ]
    array_match = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group(1))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        tool_calls.append({
                            "name": item["name"],
                            "arguments": item.get("arguments", {})
                        })
                if tool_calls:
                    return tool_calls
        except Exception:
            pass

    # 4. Single JSON object matching { "name": ... , "arguments": ... }
    single_match = re.search(r"(\{\s*\"name\"\s*:\s*\".*?\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\})", text, re.DOTALL)
    if single_match:
        try:
            data = json.loads(single_match.group(1))
            if isinstance(data, dict) and "name" in data:
                tool_calls.append({
                    "name": data["name"],
                    "arguments": data.get("arguments", {})
                })
                return tool_calls
        except Exception:
            pass

    return tool_calls
