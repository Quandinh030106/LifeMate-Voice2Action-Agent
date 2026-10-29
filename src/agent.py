"""Interactive ReAct Agent loop for LifeMate.
Coordinates between LLM planning and zaic_tools execution environment.
Strictly abides by competition rules:
- Only one ToolEnv per sample.
- Tracks all executed calls and retrieves state_hash.
- Handles multi-step read-before-write logic.
- Detects out-of-scope requests and missing info requests.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt_builder import (
    extract_available_tools,
    build_messages,
)


class LifeMateAgent:
    """Agent that manages the reasoning loop for one sample."""

    def __init__(
        self,
        llm_generate_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]],
        catalog: Dict[str, Any],
        max_steps: int = 5,
    ):
        """
        Args:
            llm_generate_fn: A function that takes (messages, tools) and returns:
                {
                    "content": Optional[str],
                    "tool_calls": List[{"name": str, "arguments": dict}]
                }
            catalog: Catalog dictionary of apps and tools.
            max_steps: Maximum reasoning / action steps per sample (to respect <= 30s timeout).
        """
        self.llm_generate_fn = llm_generate_fn
        self.catalog = catalog
        self.max_steps = max_steps

    def run_sample(
        self,
        item: Dict[str, Any],
        split: str,
        data_root: Any,
        user_request_text: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Run the agent loop on a single sample.

        Returns:
            (tool_calls, state_hash, metadata)
        """
        sample_id = item["id"]
        from zaic_tools import ToolEnv
        # Exactly one ToolEnv per sample loaded from local environment
        env = ToolEnv(split, sample_id, root=data_root)

        # 1. Filter tools available for this specific sample
        tools = extract_available_tools(item, self.catalog)

        # 2. Build initial conversation prompt
        messages = build_messages(item, self.catalog, user_request_text=user_request_text)

        step_logs = []

        # 3. Interactive Agent Loop
        for step in range(self.max_steps):
            # Query LLM
            response = self.llm_generate_fn(messages, tools)
            content = response.get("content")
            tool_calls = response.get("tool_calls") or []

            step_logs.append({
                "step": step + 1,
                "content": content,
                "tool_calls": tool_calls
            })

            # If LLM produces NO tool calls, it indicates out-of-scope or completion
            if not tool_calls:
                break

            # Add assistant message to history
            assistant_turn: Dict[str, Any] = {"role": "assistant"}
            if content:
                assistant_turn["content"] = content
            assistant_turn["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                    }
                }
                for tc in tool_calls
            ]
            messages.append(assistant_turn)

            # Execute tool calls via ToolEnv
            stop_loop = False
            for tc in tool_calls:
                tname = tc["name"]
                targs = tc.get("arguments") or {}

                # Execute in zaic_tools
                exec_result = env.call(tname, targs)

                # Append tool observation
                messages.append({
                    "role": "tool",
                    "name": tname,
                    "content": json.dumps(exec_result, ensure_ascii=False)
                })

                # If sys.ask_missing_info is called, agent is waiting for user info -> stop loop
                if tname == "sys.ask_missing_info":
                    stop_loop = True

            if stop_loop:
                break

        # Final submission output according to official requirements
        final_tool_calls = env.get_tool_calls()
        final_state_hash = env.get_state_hash()

        meta = {
            "id": sample_id,
            "steps_taken": len(step_logs),
            "step_logs": step_logs
        }

        return final_tool_calls, final_state_hash, meta
