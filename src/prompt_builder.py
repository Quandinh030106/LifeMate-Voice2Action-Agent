"""Prompt builder and Schema condensation for LifeMate Agent.
Prepares system prompts, user context, history, and dynamic tool definitions
strictly conforming to the sample's installed_apps.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def clean_parameters_for_schema(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and standardize parameters schema for standard function calling."""
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
        if "description" in v:
            prop_info["description"] = v["description"]
        if "enum" in v:
            prop_info["enum"] = v["enum"]
        if "minimum" in v:
            prop_info["minimum"] = v["minimum"]
        if "maximum" in v:
            prop_info["maximum"] = v["maximum"]
        if "items" in v:
            prop_info["items"] = v["items"]
        if "format" in v:
            prop_info["description"] = prop_info.get("description", "") + f" (Format: {v['format']})"
        if "x-ref" in v:
            prop_info["description"] = prop_info.get("description", "") + f" (Tham chiếu từ: {v['x-ref']})"
        cleaned["properties"][k] = prop_info

    # Append x-one-of constraint info to description if present
    if "x-one-of" in params_dict:
        groups_str = " HOẶC ".join([" + ".join(g) for g in params_dict["x-one-of"]])
        cleaned["description_note"] = f"Bắt buộc chọn 1 trong các nhóm tham số: {groups_str}"

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
                desc = tool.get("description", "")
                kind = tool.get("kind", "")
                full_desc = f"[{kind.upper()}] {desc}" if kind else desc
                clean_params = clean_parameters_for_schema(tool.get("parameters", {}))
                if "description_note" in clean_params:
                    full_desc += f". Lưu ý: {clean_params.pop('description_note')}"

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
        profile_lines.append(f"- Tên người dùng: {profile['name']}")
    if profile.get("home_address"):
        profile_lines.append(f"- Địa chỉ nhà: {profile['home_address']}")
    if profile.get("work_address"):
        profile_lines.append(f"- Nơi làm việc: {profile['work_address']}")
    if profile.get("city"):
        profile_lines.append(f"- Thành phố: {profile['city']}")
    profile_str = "\n".join(profile_lines) if profile_lines else "- Chưa có thông tin cụ thể"

    loc_str = f"lat: {loc.get('lat')}, lng: {loc.get('lng')}" if loc else "Không xác định"

    system_prompt = f"""Bạn là LifeMate — Trợ lý AI điều hành tác vụ thông minh cho thiết bị người dùng.
Nhiệm vụ của bạn là lắng nghe yêu cầu của người dùng, lập kế hoạch hành động và gọi các công cụ (tools) phù hợp để cập nhật trạng thái ứng dụng.

[BỐI CẢNH HIỆN TẠI (CONTEXT)]
- Thời gian hiện tại: {curr_time} ({tz})
- Tọa độ hiện tại: {loc_str}
- Hồ sơ người dùng:
{profile_str}

[QUY TẮC BẮT BUỘC KHI GỌI TOOL]
1. QUY TẮC TÍNH TOÁN THỜI GIAN & NGÀY THÁNG (BẮT BUỘC):
   - Luôn dựa vào [Thời gian hiện tại] được cung cấp ở trên:
     * "hôm nay" / "tối nay": ngày hiện tại (YYYY-MM-DD).
     * "mai" / "ngày mai" / "sáng mai": ngày hiện tại + 1 ngày (YYYY-MM-DD). Ví dụ hiện tại là 2026-11-28 thì ngày mai là 2026-11-29.
     * "ngày kia": ngày hiện tại + 2 ngày.
   - Khi tool yêu cầu tham số bắt buộc dạng ngày (như `date`: "YYYY-MM-DD") hoặc giờ (như `start`, `time`: ISO-8601), BẮT BUỘC phải quy đổi và truyền đầy đủ, KHÔNG ĐƯỢC bỏ sót!
2. NGUYÊN TẮC SUY LUẬN & KIỂM TRA ĐIỀU KIỆN (READ BEFORE WRITE):
   - ĐỐI VỚI YÊU CẦU CÓ ĐIỀU KIỆN (vd: "Nếu mai mưa...", "Nếu có lịch..."): BẮT BUỘC phải gọi tool tra cứu (ví dụ: weather.get_forecast, calendar.list_events) trước để biết kết quả thực tế, TUYỆT ĐỐI KHÔNG tự bịa ra kết quả hay nhảy cóc thực hiện hành động ghi!
   - ĐỐI VỚI TÊN NGƯỜI / ĐỊA ĐIỂM / THIẾT BỊ CẦN ID (vd: "nhắn Sếp Dũng", "bật đèn phòng khách"): BẮT BUỘC gọi tool tìm kiếm/đọc trước (contacts.search, smart_home.get_status...) để lấy contact_id / device_id chính xác.
   - CHỈ gọi 1 hoặc 2 tool tra cứu cần thiết ở bước hiện tại, chờ nhận kết quả trả về từ hệ thống rồi mới quyết định bước tiếp theo.
3. XỬ LÝ NGOÀI PHẠM VI (OUT-OF-SCOPE):
   - Nếu yêu cầu của người dùng không liên quan hoặc không thể thực hiện bằng bất kỳ tool/app nào đã cài, KHÔNG ĐƯỢC gọi bất kỳ tool nào (kết thúc ngay).
4. XỬ LÝ THIẾU THÔNG TIN:
   - Nếu người dùng yêu cầu một hành động nhưng thiếu tham số bắt buộc mà không thể suy ra từ ngữ cảnh hay công cụ đọc, hãy gọi tool `sys.ask_missing_info` với arguments:
     {{"tool_name": "<tên_tool_thiếu>", "params": ["<tên_tham_số_1>", ...]}}
5. ĐỊNH DẠNG:
   - Phải tuân thủ đúng tên tool và kiểu dữ liệu của tham số (string, int, enum, format ISO-8601).
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

    # Process conversation history if present
    for turn in item.get("history", []):
        # User turn
        if "audio_text" in turn:
            messages.append({"role": "user", "content": turn["audio_text"]})
        elif "text" in turn:
            messages.append({"role": "user", "content": turn["text"]})

        # Assistant turn
        if turn.get("role") == "assistant" or "tool_calls" in turn or "text" in turn:
            asst_msg: Dict[str, Any] = {"role": "assistant"}
            if "text" in turn:
                asst_msg["content"] = turn["text"]
            if "tool_calls" in turn:
                asst_msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                        }
                    }
                    for tc in turn["tool_calls"]
                ]
            messages.append(asst_msg)

            # Tool results if present
            if "tool_results" in turn and "tool_calls" in turn:
                for tc, res in zip(turn["tool_calls"], turn["tool_results"]):
                    messages.append({
                        "role": "tool",
                        "name": tc["name"],
                        "content": json.dumps(res, ensure_ascii=False)
                    })

    # Add the current user request
    request = user_request_text or item.get("audio_text", "")
    messages.append({"role": "user", "content": request})

    return messages
