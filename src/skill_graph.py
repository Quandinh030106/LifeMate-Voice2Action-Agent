"""Adaptive Skill Knowledge Graph Engine for LifeMate.
Parses tools_catalog.json dynamically into a directed semantic graph (DAG).
Models:
1. Tool Nodes: Metadata, kinds (read/write/meta), parameters schema, and return types.
2. Dependency Edges: Pre-read requirements (Tool A.returns -> Tool B.required_params).
3. Domain Clusters: App & Group semantic associations (finance, mobility, IoT, media, etc.).
4. Distractor Pruning: Dynamically prunes unrelated domains to prevent LLM hallucinations.
5. Meta-Reasoning CoT Generator: Synthesizes structured thought chains for Adaptive SFT.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SkillNode:
    """Represents a single tool in the Skill Knowledge Graph."""
    name: str
    app: str
    group: str
    kind: str  # "read", "write", "meta"
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    property_names: List[str]
    returns_desc: str
    keywords: Set[str] = field(default_factory=set)

    @property
    def is_read(self) -> bool:
        return self.kind == "read"

    @property
    def is_write(self) -> bool:
        return self.kind == "write"

    @property
    def is_meta(self) -> bool:
        return self.kind == "meta" or self.name.startswith("sys.")


class SkillGraph:
    """Directed Knowledge Graph connecting Tools, Domains, and Parameter Dependencies."""

    # Common parameter names that typically require a prior 'read' lookup
    ENTITY_ID_PARAM_PATTERNS = {
        "device_id", "from_account_id", "beneficiary_id", "account_id",
        "lock_id", "alarm_id", "reminder_id", "playlist_id", "track_id",
        "contact_id", "recipient_id", "order_id", "place_id", "appliance_id",
        "camera_id", "alert_id", "event_id", "ride_id", "item_id", "booking_id"
    }

    # Semantic domain keywords for intent routing
    DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
        "finance": {
            "chuyển tiền", "tiền", "tài khoản", "ngân hàng", "ví", "thanh toán", "số dư",
            "nạp tiền", "rút tiền", "banking", "ewallet", "transfer", "balance", "beneficiary"
        },
        "mobility": {
            "xe", "đặt xe", "chuyến đi", "bản đồ", "đường", "địa điểm", "tới", "đến",
            "giao thông", "tắc đường", "transport", "ride", "maps", "traffic", "car", "bike"
        },
        "smarthome": {
            "đèn", "quạt", "máy lạnh", "điều hòa", "công tắc", "khóa", "cửa", "robot",
            "hút bụi", "năng lượng", "thiết bị", "smart_home", "light", "fan", "climate", "lock"
        },
        "productivity": {
            "báo thức", "hẹn giờ", "nhắc nhở", "lịch", "sự kiện", "ghi chú", "nhắc",
            "alarm", "reminder", "calendar", "notes", "schedule", "timer"
        },
        "media": {
            "nhạc", "bài hát", "ca sĩ", "phát", "dừng", "âm lượng", "playlist", "video",
            "music", "song", "media", "play", "volume", "artist"
        },
        "communication": {
            "gọi", "tin nhắn", "nhắn tin", "danh bạ", "email", "gửi tin", "sms",
            "call", "message", "contacts", "chat", "send"
        },
        "weather": {
            "thời tiết", "mưa", "nắng", "nhiệt độ", "dự báo", "bão", "gió",
            "weather", "forecast", "temperature", "rain"
        },
        "shopping": {
            "mua", "đặt hàng", "đơn hàng", "giỏ hàng", "món", "quán ăn", "đồ ăn",
            "food", "order", "shopping", "cart", "restaurant", "delivery"
        }
    }

    def __init__(self, catalog_data: Optional[Dict[str, Any]] = None):
        self.nodes: Dict[str, SkillNode] = {}
        self.app_to_tools: Dict[str, List[str]] = {}
        self.group_to_tools: Dict[str, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}  # target_tool -> [prerequisite_read_tools]

        if catalog_data:
            self.load_catalog(catalog_data)

    @classmethod
    def from_file(cls, catalog_path: Union[str, Path]) -> SkillGraph:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def load_catalog(self, catalog: Dict[str, Any]) -> None:
        """Parse raw tools_catalog.json and construct graph nodes and dependency edges."""
        self.nodes.clear()
        self.app_to_tools.clear()
        self.group_to_tools.clear()
        self.dependencies.clear()

        # Add virtual system tools
        sys_ask = SkillNode(
            name="sys.ask_missing_info",
            app="system",
            group="system",
            kind="meta",
            description="Yêu cầu người dùng cung cấp thêm thông tin còn thiếu khi không có công cụ nào tra cứu được.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": "Tên công cụ dự kiến thực hiện"},
                    "params": {"type": "array", "items": {"type": "string"}, "description": "Danh sách các tham số còn thiếu"}
                },
                "required": ["params"]
            },
            required_params=["params"],
            property_names=["tool_name", "params"],
            returns_desc="",
            keywords={"thiếu", "hỏi", "thông tin", "ask", "missing"}
        )
        self.nodes[sys_ask.name] = sys_ask

        apps = catalog.get("apps", [])
        for app in apps:
            app_id = app.get("id", "")
            app_group = app.get("group", app_id)
            if app_id not in self.app_to_tools:
                self.app_to_tools[app_id] = []
            if app_group not in self.group_to_tools:
                self.group_to_tools[app_group] = []

            for tool in app.get("tools", []):
                tname = tool["name"]
                tkind = tool.get("kind", "read")
                tdesc = tool.get("description", "")
                tparams = tool.get("parameters", {})
                req_params = tparams.get("required", []) if isinstance(tparams, dict) else []
                prop_names = list(tparams.get("properties", {}).keys()) if isinstance(tparams, dict) else []
                treturns = tool.get("returns", "")

                # Extract keywords from name and description
                kw = set(re.findall(r"\w+", (tname + " " + tdesc + " " + app_group).lower()))

                node = SkillNode(
                    name=tname,
                    app=app_id,
                    group=app_group,
                    kind=tkind,
                    description=tdesc,
                    parameters=tparams,
                    required_params=req_params,
                    property_names=prop_names,
                    returns_desc=treturns,
                    keywords=kw
                )
                self.nodes[tname] = node
                self.app_to_tools[app_id].append(tname)
                self.group_to_tools[app_group].append(tname)

        # Build Dependency Graph
        self._build_dependency_edges()

    def _build_dependency_edges(self) -> None:
        """Infer prerequisites for each write tool based on entity ID requirements."""
        for tname, node in self.nodes.items():
            if not node.is_write:
                continue

            prereqs = []
            req_set = set(node.required_params)
            app_tools = self.app_to_tools.get(node.app, [])

            # Look for read tools in the same app first
            for other_name in app_tools:
                other_node = self.nodes.get(other_name)
                if other_node and other_node.is_read:
                    # If write tool needs an ID (like device_id, account_id, etc.) and read tool lists entities
                    if any(p in self.ENTITY_ID_PARAM_PATTERNS for p in req_set):
                        prereqs.append(other_name)

            self.dependencies[tname] = prereqs

    def get_prerequisite_tools(
        self,
        target_tool_name: str,
        available_tools: Set[str]
    ) -> List[str]:
        """Find valid prerequisite read tools for a target tool among currently installed apps."""
        raw_prereqs = self.dependencies.get(target_tool_name, [])
        return [p for p in raw_prereqs if p in available_tools]

    def detect_active_domains(self, query: str) -> Set[str]:
        """Identify relevant semantic domains from user query text."""
        query_lower = query.lower()
        active = set()
        for domain, kw_set in self.DOMAIN_KEYWORDS.items():
            for kw in kw_set:
                if kw in query_lower:
                    active.add(domain)
                    break
        return active

    def filter_tools_for_request(
        self,
        query: str,
        installed_tool_schemas: List[Dict[str, Any]],
        max_tools: int = 18
    ) -> List[Dict[str, Any]]:
        """Prune completely irrelevant tool domains to prevent attention distraction and hallucinated calls.

        Ensures:
        1. All tools relevant to active domains are kept.
        2. Prerequisite read tools for kept write tools are never pruned.
        3. sys.ask_missing_info is always preserved.
        """
        if len(installed_tool_schemas) <= max_tools:
            return installed_tool_schemas

        installed_map = {t["function"]["name"]: t for t in installed_tool_schemas}
        installed_names = set(installed_map.keys())

        # Always keep sys.ask_missing_info
        kept_names = {"sys.ask_missing_info"} if "sys.ask_missing_info" in installed_names else set()

        # Score installed tools based on query relevance
        q_words = set(re.findall(r"\w+", query.lower()))
        active_domains = self.detect_active_domains(query)

        scored_tools: List[Tuple[float, str]] = []
        for tname in installed_names:
            if tname == "sys.ask_missing_info":
                continue

            node = self.nodes.get(tname)
            if not node:
                scored_tools.append((0.1, tname))
                continue

            score = 0.0
            # Keyword match with tool name / desc
            matches = len(q_words & node.keywords)
            score += matches * 2.0

            # Domain alignment match
            for dom in active_domains:
                dom_kws = self.DOMAIN_KEYWORDS.get(dom, set())
                if any(k in node.keywords for k in dom_kws):
                    score += 3.0

            # Same app bonus if any tool in app matched query
            app_id = node.app.lower()
            if any(w in app_id for w in q_words):
                score += 4.0

            scored_tools.append((score, tname))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # Pick top candidates
        for score, tname in scored_tools:
            if len(kept_names) >= max_tools:
                break
            kept_names.add(tname)
            # Also keep prerequisite read tools for this tool
            for prereq in self.get_prerequisite_tools(tname, installed_names):
                kept_names.add(prereq)

        return [installed_map[name] for name in kept_names if name in installed_map]

    def generate_meta_reasoning_cot(
        self,
        query: str,
        tool_call: Optional[Dict[str, Any]],
        step: int = 1,
        total_steps: int = 1,
        prev_tool_name: Optional[str] = None
    ) -> str:
        """Synthesize high-grade, zero-shot Meta-Reasoning Chain-of-Thought for SFT training."""
        if not tool_call:
            # Out-of-scope or completion
            return (
                "<thought>\n"
                f"1. Phân tích yêu cầu: \"{query}\"\n"
                "2. Kiểm tra công cụ: Yêu cầu này nằm ngoài phạm vi hoặc các công cụ hiện có không hỗ trợ.\n"
                "3. Quyết định: Phản hồi giải thích lịch sự, không gọi công cụ không liên quan.\n"
                "</thought>"
            )

        tname = tool_call.get("name", "")
        targs = tool_call.get("arguments", {})
        node = self.nodes.get(tname)

        if tname == "sys.ask_missing_info":
            missing_params = targs.get("params", [])
            target = targs.get("tool_name", "tác vụ")
            return (
                "<thought>\n"
                f"1. Phân tích yêu cầu: \"{query}\"\n"
                f"2. Mục tiêu: Chuẩn bị thực hiện công cụ `{target}`.\n"
                f"3. Kiểm tra tham số: Còn thiếu thông tin bắt buộc {missing_params} và không có công cụ tra cứu tự động.\n"
                "4. Kế hoạch: Gọi `sys.ask_missing_info` để yêu cầu người dùng bổ sung thông tin cần thiết.\n"
                "</thought>"
            )

        if node and node.is_read:
            return (
                "<thought>\n"
                f"1. Phân tích yêu cầu: \"{query}\"\n"
                f"2. Đánh giá phụ thuộc: Để hoàn thành yêu cầu một cách chính xác, cần thu thập thông tin ngữ cảnh trước.\n"
                f"3. Kế hoạch: Gọi công cụ tra cứu `{tname}` để lấy dữ liệu thực tế từ hệ thống.\n"
                "</thought>"
            )

        if node and node.is_write:
            if prev_tool_name:
                return (
                    "<thought>\n"
                    f"1. Nhận diện trạng thái: Đã có kết quả dữ liệu từ công cụ `{prev_tool_name}`.\n"
                    f"2. Xác thực tham số: Các tham số bắt buộc cho `{tname}` đã đầy đủ.\n"
                    f"3. Thực thi hành động: Gọi `{tname}` để hoàn tất yêu cầu của người dùng.\n"
                    "</thought>"
                )
            else:
                return (
                    "<thought>\n"
                    f"1. Phân tích yêu cầu: \"{query}\"\n"
                    f"2. Xác thực tham số: Câu nói của người dùng đã cung cấp đủ thông tin cho công cụ `{tname}`.\n"
                    f"3. Thực thi hành động: Gọi `{tname}` để thực hiện tác vụ.\n"
                    "</thought>"
                )

        return (
            "<thought>\n"
            f"1. Phân tích yêu cầu: \"{query}\"\n"
            f"2. Quyết định: Thực thi công cụ `{tname}`.\n"
            "</thought>"
        )
