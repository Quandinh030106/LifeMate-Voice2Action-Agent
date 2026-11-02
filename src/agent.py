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
        allowed_tool_names = set(t["function"]["name"] for t in tools)

        # 2. Build initial conversation prompt
        messages = build_messages(item, self.catalog, user_request_text=user_request_text)

        step_logs = []
        executed_call_signatures = set()

        # Common single-turn write tools that conclude the user request once executed
        SINGLE_WRITE_TOOLS = {
            "ride_hailing.book", "ride_hailing.cancel",
            "banking.transfer", "banking.open_savings",
            "ewallet.transfer", "ewallet.topup_mobile", "ewallet.pay_qr",
            "alarm.create", "alarm.update", "alarm.delete",
            "reminders.create", "reminders.complete", "reminders.delete",
            "sms.send", "phone.call",
            "chat_ott.call", "chat_ott.send_message",
            "smart_home.light_set", "smart_home.fan_set", "smart_home.climate_set",
            "smart_home.switch_set", "smart_home.scene_run",
            "tv.control", "tv.tune",
            "bill_payment.schedule_autopay", "bill_payment.pay",
            "calendar.cancel_event", "calendar.create_event", "calendar.update_event"
        }

        # Informational read-only tools
        PURE_SEARCH_TOOLS = {
            "flights.search", "weather.get_current", "weather.get_forecast",
            "gold_price.get_price", "fuel_price.get_price"
        }

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

            # Add assistant message to history (keep content empty when tool_calls are present to avoid prompt duplication)
            assistant_turn: Dict[str, Any] = {"role": "assistant", "content": ""}
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

                # CRITICAL: Validate against allowed tools to prevent illegal tools from corrupting submission
                if tname not in allowed_tool_names:
                    messages.append({
                        "role": "tool",
                        "name": tname,
                        "content": json.dumps({
                            "error": f"Tool '{tname}' không tồn tại trong danh sách ứng dụng đã cài đặt. Chỉ được gọi các tool: {sorted(list(allowed_tool_names))}"
                        }, ensure_ascii=False)
                    })
                    continue

                # In-Context Entity Resolution Fallback for locations
                for loc_key in ["pickup", "dropoff", "address", "location"]:
                    if loc_key in targs and isinstance(targs[loc_key], str):
                        val_clean = targs[loc_key].strip().lower()
                        if val_clean in ["công ty", "văn phòng", "chỗ làm", "chỗ làm việc", "nơi làm việc", "cty"] and work_addr:
                            targs[loc_key] = work_addr
                        elif val_clean in ["nhà", "nhà riêng", "về nhà", "ở nhà"] and home_addr:
                            targs[loc_key] = home_addr

                # In-Context Entity Resolution for banking.transfer
                if tname == "banking.transfer":
                    from_acc = str(targs.get("from_account_id", ""))
                    ben_id = str(targs.get("beneficiary_id", ""))
                    if "1234" in from_acc or "1234" in ben_id or "account_" in from_acc or "beneficiary_" in ben_id or not from_acc or not ben_id:
                        if "banking.get_balance" in allowed_tool_names and ("banking.get_balance", "{}") not in executed_call_signatures:
                            bal_res = env.call("banking.get_balance", {})
                            executed_call_signatures.add(("banking.get_balance", "{}"))
                            messages.append({"role": "tool", "name": "banking.get_balance", "content": json.dumps(bal_res, ensure_ascii=False)})
                            if bal_res.get("ok") and bal_res.get("accounts"):
                                targs["from_account_id"] = bal_res["accounts"][0]["id"]
                        if "banking.list_beneficiaries" in allowed_tool_names and ("banking.list_beneficiaries", "{}") not in executed_call_signatures:
                            ben_res = env.call("banking.list_beneficiaries", {})
                            executed_call_signatures.add(("banking.list_beneficiaries", "{}"))
                            messages.append({"role": "tool", "name": "banking.list_beneficiaries", "content": json.dumps(ben_res, ensure_ascii=False)})
                            if ben_res.get("ok") and ben_res.get("beneficiaries"):
                                note_context = (str(targs.get("note", "")) + " " + str(user_request_text or "")).lower()
                                matched_ben = None
                                for b in ben_res["beneficiaries"]:
                                    if b.get("name", "").lower() in note_context:
                                        matched_ben = b["id"]
                                        break
                                targs["beneficiary_id"] = matched_ben if matched_ben else ben_res["beneficiaries"][0]["id"]

                # Deduplication check: Break infinite ReAct loop if model emits identical call
                call_sig = (tname, json.dumps(targs, sort_keys=True, ensure_ascii=False))
                if call_sig in executed_call_signatures:
                    stop_loop = True
                    break
                executed_call_signatures.add(call_sig)

                # Schema safety for sys.ask_missing_info
                if tname == "sys.ask_missing_info":
                    if "params" not in targs or not isinstance(targs["params"], list):
                        if "params" in targs and isinstance(targs["params"], str):
                            targs["params"] = [targs["params"]]
                        else:
                            targs["params"] = []
                    if "tool_name" not in targs or not isinstance(targs["tool_name"], str):
                        targs["tool_name"] = "general"

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
                    break

                # Pure search tool early stopping if query is purely informational
                if tname in PURE_SEARCH_TOOLS:
                    req_lower = (user_request_text or "").lower()
                    action_keywords = ["đặt", "book", "mua", "chuyển", "báo", "hẹn", "lịch", "bật", "tắt"]
                    if not any(k in req_lower for k in action_keywords):
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
