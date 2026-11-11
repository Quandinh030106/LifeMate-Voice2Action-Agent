"""Argument Normalizer & Post-Processor for LifeMate.
Strictly aligns LLM tool call arguments with official Zalo AI Challenge evaluation standards:
1. ISO-8601 formatting for time/start/end/schedule_time (with +07:00 timezone).
2. Date formatting (YYYY-MM-DD).
3. Entity resolution (resolving 'công ty', 'nhà' into full addresses from user profile).
4. Schema standardization for sys.ask_missing_info (canonical parameter names).
5. Data type casting (integers for amounts/percentages, strings for phone numbers/IDs).
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional


def parse_context_datetime(item: Dict[str, Any]) -> Tuple[datetime.datetime, datetime.date, datetime.date]:
    """Parse current datetime, today, and tomorrow from item context."""
    ctx = item.get("context", {}) or {}
    curr_iso = ctx.get("current_time") or item.get("current_time") or "2026-11-28T07:00:00+07:00"
    try:
        # Python 3.11+ handles fromisoformat with offset +07:00 directly
        curr_dt = datetime.datetime.fromisoformat(curr_iso)
    except Exception:
        # Fallback regex
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", curr_iso)
        if m:
            curr_dt = datetime.datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
                tzinfo=datetime.timezone(datetime.timedelta(hours=7))
            )
        else:
            curr_dt = datetime.datetime(2026, 11, 28, 7, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=7)))

    today = curr_dt.date()
    tomorrow = today + datetime.timedelta(days=1)
    return curr_dt, today, tomorrow


def normalize_datetime_string(
    val: Any,
    today: datetime.date,
    tomorrow: datetime.date,
    user_request_text: str = ""
) -> str:
    """Normalize timestamp string into strict ISO-8601 with +07:00 timezone."""
    if not val:
        return ""
    val_str = str(val).strip()

    # If already full ISO-8601 with timezone (e.g. 2026-11-11T05:30:00+07:00)
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+07:00$", val_str):
        return val_str

    # If full ISO without seconds or without timezone
    # e.g. 2026-11-11T05:30:00 or 2026-11-11 05:30:00
    m_full = re.match(r"^(\d{4}-\d{2}-\d{2})[T\s](\d{2}):(\d{2})(?::(\d{2}))?", val_str)
    if m_full:
        d_part = m_full.group(1)
        hh = m_full.group(2)
        mm = m_full.group(3)
        ss = m_full.group(4) or "00"
        return f"{d_part}T{hh}:{mm}:{ss}+07:00"

    # If time only: e.g. 05:30 or 05:30:00 or 5h30 or 5:30
    m_time = re.match(r"^(\d{1,2})[:h](\d{2})(?::(\d{2}))?", val_str)
    if m_time:
        hh = int(m_time.group(1))
        mm = m_time.group(2)
        ss = m_time.group(3) or "00"
        hh_str = f"{hh:02d}"

        # Determine target date (tomorrow vs today)
        req_lower = user_request_text.lower()
        if any(w in req_lower for w in ["mai", "ngày mai", "sáng mai", "chiều mai", "tối mai", "đêm mai"]):
            target_date = tomorrow
        else:
            target_date = today

        return f"{target_date.strftime('%Y-%m-%d')}T{hh_str}:{mm}:{ss}+07:00"

    return val_str


def normalize_date_string(val: Any) -> str:
    """Normalize date string to YYYY-MM-DD."""
    if not val:
        return ""
    val_str = str(val).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", val_str)
    if m:
        return m.group(1)
    return val_str


def normalize_tool_arguments(
    tool_name: str,
    arguments: Dict[str, Any],
    item: Dict[str, Any],
    user_request_text: str = ""
) -> Dict[str, Any]:
    """Normalize and repair arguments according to official schemas and profiles."""
    if not isinstance(arguments, dict):
        return {}

    args = dict(arguments)
    curr_dt, today, tomorrow = parse_context_datetime(item)

    ctx = item.get("context", {}) or {}
    profile = ctx.get("user_profile") or {}
    work_addr = profile.get("work_address")
    home_addr = profile.get("home_address")

    # 1. Datetime / Timestamp arguments
    for k in ["time", "start", "end", "schedule_time"]:
        if k in args and args[k]:
            args[k] = normalize_datetime_string(args[k], today, tomorrow, user_request_text)

    # 2. Date arguments
    for k in ["date", "date_from", "date_to"]:
        if k in args and args[k]:
            args[k] = normalize_date_string(args[k])

    # 3. Entity Resolution for Ride Hailing
    if tool_name.startswith("ride_hailing."):
        if "pickup" in args and isinstance(args["pickup"], str):
            p_low = args["pickup"].lower().strip()
            if any(w == p_low or p_low.startswith(w) for w in ["công ty", "chỗ làm", "văn phòng", "co quan", "cơ quan"]) and work_addr:
                args["pickup"] = work_addr
            elif any(w == p_low or p_low.startswith(w) for w in ["nhà", "về nhà", "nha"]) and home_addr:
                args["pickup"] = home_addr

        if "dropoff" in args and isinstance(args["dropoff"], str):
            d_low = args["dropoff"].lower().strip()
            if any(w == d_low or d_low.startswith(w) for w in ["công ty", "chỗ làm", "văn phòng", "co quan", "cơ quan"]) and work_addr:
                args["dropoff"] = work_addr
            elif any(w == d_low or d_low.startswith(w) for w in ["nhà", "về nhà", "nha"]) and home_addr:
                args["dropoff"] = home_addr

        if "vehicle" in args and not args["vehicle"]:
            args["vehicle"] = "car4"

    # 4. Canonical schema for sys.ask_missing_info
    if tool_name == "sys.ask_missing_info":
        # Ensure params is list
        params = args.get("params") or []
        if isinstance(params, str):
            params = [params]
        elif not isinstance(params, list):
            params = []

        norm_params = []
        for p in params:
            p_str = str(p).lower().strip()
            # Map canonical missing parameter names
            if p_str in ["time_of_day", "datetime", "thời_gian", "gio", "giờ", "time_str", "timestamp"]:
                norm_params.append("time")
            elif p_str in ["contact", "phone_number", "sdt", "số_điện_thoại", "người_nhận", "recipient"]:
                norm_params.append("contact_id")
            elif p_str in ["so_tien", "số_tiền", "tien", "tiền", "price"]:
                norm_params.append("amount")
            elif p_str in ["noi_dung", "nội_dung", "tin_nhan", "body"]:
                norm_params.append("content")
            else:
                norm_params.append(p_str)

        args["params"] = norm_params if norm_params else ["time"]
        if "tool_name" not in args or not args["tool_name"]:
            args["tool_name"] = "general"

    # 5. Type casting for amounts, phone numbers
    if "amount" in args:
        try:
            args["amount"] = int(float(args["amount"]))
        except (ValueError, TypeError):
            pass

    if "phone" in args:
        args["phone"] = str(args["phone"]).strip()

    # 6. Schema Pruning for alarm.create (removes extra hallucinated keys like 'label' or 'name')
    if tool_name == "alarm.create":
        if "label" in args:
            del args["label"]
        if "name" in args:
            del args["name"]

    return args
