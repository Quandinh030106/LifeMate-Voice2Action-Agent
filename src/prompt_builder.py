"""Prompt builder and Schema condensation for LifeMate Agent.
Prepares system prompts, user context, history, and dynamic tool definitions
strictly conforming to the sample's installed_apps.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def clean_parameters_for_schema(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean, standardize, and condense parameters schema for token efficiency."""
    if not params_dict or not isinstance(params_dict, dict):
        return {"type": "object", "properties": {}, "required": []}

    cleaned = {
        "type": "object",
        "properties": {},
        "required": list(params_dict.get("required", []))
    }

    props = params_dict.get("properties", {})
    for k, v in props.items():
        prop_info: Dict[str, Any] = {}
        if "type" in v:
            prop_info["type"] = v["type"]
        if "enum" in v and len(v["enum"]) <= 6:
            prop_info["enum"] = v["enum"]
        if "items" in v and isinstance(v["items"], dict):
            prop_info["items"] = {"type": v["items"].get("type", "string")}

        # Condense description to first sentence, keeping critical alias hints
        raw_desc = v.get("description", "")
        if raw_desc:
            first_sent = raw_desc.split(".")[0].strip()
            if len(first_sent) > 70:
                first_sent = first_sent[:67] + "..."
            prop_info["description"] = first_sent
        cleaned["properties"][k] = prop_info

    return cleaned


def extract_available_tools(
    item: Dict[str, Any],
    catalog: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract tool schemas allowed for this specific sample (installed_apps + sys)."""
    installed = item.get("context", {}).get("installed_apps", {})
    allowed_names = set()
    for app_id, app_info in installed.items():
        for tname in app_info.get("tools", []):
            allowed_names.add(tname)

    # sys.ask_missing_info is always allowed
    allowed_names.add("sys.ask_missing_info")

    catalog_apps = catalog.get("apps", [])
    extracted_tools = []

    for app in catalog_apps:
        for tool in app.get("tools", []):
            tname = tool["name"]
            if tname in allowed_names:
                desc = tool.get("description", "").split(".")[0].strip()
                if len(desc) > 75:
                    desc = desc[:72] + "..."
                kind = tool.get("kind", "")
                full_desc = f"[{kind.upper()}] {desc}" if kind else desc
                clean_params = clean_parameters_for_schema(tool.get("parameters", {}))

                extracted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tname,
                        "description": full_desc,
                        "parameters": clean_params
                    }
                })

    return extracted_tools


def build_system_prompt(item: Dict[str, Any]) -> str:
    """Build clear, structured Vietnamese system prompt incorporating context."""
    ctx = item.get("context", {})
    curr_time = ctx.get("current_time", "Unknown")
    tz = ctx.get("timezone", "Asia/Ho_Chi_Minh")
    loc = ctx.get("location", {})
    profile = ctx.get("user_profile", {})

    profile_lines = []
    if profile.get("name"):
        profile_lines.append(f"- Tên: {profile['name']}")
    if profile.get("home_address"):
        profile_lines.append(f"- Nhà: {profile['home_address']}")
    if profile.get("work_address"):
        profile_lines.append(f"- Công ty: {profile['work_address']}")
    if profile.get("city"):
        profile_lines.append(f"- Thành phố: {profile['city']}")
    profile_str = "\n".join(profile_lines) if profile_lines else "- Chưa có thông tin cụ thể"

    loc_str = f"lat: {loc.get('lat')}, lng: {loc.get('lng')}" if loc else "Không xác định"

    system_prompt = f"""Bạn là LifeMate — Trợ lý AI điều hành tác vụ thông minh cho thiết bị người dùng.
Nhiệm vụ: Lập kế hoạch và gọi công cụ (tool) phù hợp theo ngữ cảnh.

[BỐI CẢNH (CONTEXT)]
- Thời gian hiện tại: {curr_time} ({tz})
- Tọa độ: {loc_str}
- Hồ sơ:
{profile_str}

[QUY TẮC BẮT BUỘC KHI GỌI TOOL]
1. TÍNH TOÁN THỜI GIAN:
   - "hôm nay"/"tối nay": ngày hiện tại (YYYY-MM-DD). "mai"/"sáng mai": ngày hiện tại + 1 ngày.
   - Luôn điền đầy đủ tham số ngày/giờ theo format ISO-8601.
2. NGUYÊN TẮC SUY LUẬN (READ BEFORE WRITE):
   - Có điều kiện (mưa, rảnh...): BẮT BUỘC gọi tool tra cứu (weather.get_forecast, calendar.list_events) trước!
   - Cần ID/địa điểm/thông tin (tên người, thiết bị, chỗ hẹn): BẮT BUỘC gọi contacts.search, calendar.list_events trước!
   - Chỉ gọi 1-2 tool tra cứu cần thiết mỗi bước, chờ kết quả rồi mới gọi bước tiếp theo.
3. ÁNH XẠ THỰC THỂ & ĐỊA ĐIỂM (ENTITY RESOLUTION):
   - "công ty" / "chỗ làm" / "văn phòng" -> BẮT BUỘC điền địa chỉ công ty từ hồ sơ: {profile.get('work_address', 'công ty')}.
   - "nhà" / "về nhà" -> BẮT BUỘC điền địa chỉ nhà từ hồ sơ: {profile.get('home_address', 'nhà')}.
4. NGOÀI PHẠM VI (OUT-OF-SCOPE):
   - Yêu cầu không nằm trong các ứng dụng đã cài đặt: KHÔNG ĐƯỢC gọi bất kỳ tool nào (kết thúc ngay).
5. THIẾU THÔNG TIN:
   - Thiếu tham số bắt buộc không thể tự suy ra: gọi `sys.ask_missing_info` với arguments: {{"tool_name": "<tên_tool>", "params": ["<tham_số>"]}}.
"""
    return system_prompt.strip()


def build_messages(
    item: Dict[str, Any],
    catalog: Dict[str, Any],
    user_request_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build conversational message list including history and the current user request."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(item)}
    ]

    # Process conversation history if present (sliding window: max 4 recent turns)
    history = item.get("history", [])
    if len(history) > 4:
        history = history[-4:]

    for turn in history:
        role = turn.get("role")
        if role == "user":
            user_content = turn.get("audio_text") or turn.get("text") or "[Yêu cầu người dùng trước đó]"
            messages.append({"role": "user", "content": user_content})

        elif role == "assistant":
            asst_msg: Dict[str, Any] = {"role": "assistant"}
            if "text" in turn:
                asst_msg["content"] = turn["text"]
            if "tool_calls" in turn:
                asst_msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("arguments", {})
                        }
                    }
                    for tc in turn["tool_calls"]
                ]
            messages.append(asst_msg)

            # Tool results if present
            if "tool_results" in turn and "tool_calls" in turn:
                for tc, res in zip(turn["tool_calls"], turn["tool_results"]):
                    res_str = json.dumps(res, ensure_ascii=False)
                    if len(res_str) > 800:
                        res_str = res_str[:800] + "... [kết quả dài đã rút gọn]"
                    messages.append({
                        "role": "tool",
                        "name": tc["name"],
                        "content": res_str
                    })

    # Add the current user request
    request = user_request_text or item.get("audio_text", "")
    messages.append({"role": "user", "content": request})

    return messages
