import json
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_cat_path = Path("data/train+public_test/train+public_test/train/train_tools_catalog.json")
test_cat_path = Path("data/train+public_test/train+public_test/public_test/public_test_tools_catalog.json")

with open(train_cat_path, "r", encoding="utf-8") as f:
    train_cat = json.load(f)

apps = train_cat.get("apps", [])
print(f"Tổng số ứng dụng (Apps) trong catalog: {len(apps)}")

total_tools = 0
tools_by_kind = defaultdict(list)
apps_by_domain = defaultdict(list)
all_tools = []

for app in apps:
    app_id = app.get("id", "")
    app_name = app.get("name", "")
    app_group = app.get("group", "other")
    tools = app.get("tools", [])
    total_tools += len(tools)
    
    apps_by_domain[app_group].append((app_id, app_name, len(tools)))
    
    for tool in tools:
        tname = tool["name"]
        tkind = tool.get("kind", "unknown")
        tdesc = tool.get("description", "")
        tparams = tool.get("parameters", {})
        req = tparams.get("required", []) if isinstance(tparams, dict) else []
        props = list(tparams.get("properties", {}).keys()) if isinstance(tparams, dict) else []
        treturns = tool.get("returns", "")
        
        tools_by_kind[tkind].append(tname)
        all_tools.append({
            "name": tname,
            "app": app_id,
            "group": app_group,
            "kind": tkind,
            "description": tdesc,
            "required": req,
            "properties": props,
            "returns": treturns
        })

print(f"Tổng số công cụ (Tools/Skills): {total_tools}")
print(f"Phân loại theo Kind: Read={len(tools_by_kind['read'])}, Write={len(tools_by_kind['write'])}, Meta={len(tools_by_kind['meta'])}, Khác={len(tools_by_kind['unknown'])}")
print("=" * 70)
print("DANH SÁCH CÁC NHÓM ỨNG DỤNG (GROUPS/DOMAINS):")
print("=" * 70)
for grp, app_list in sorted(apps_by_domain.items()):
    total_grp_tools = sum(t_count for _, _, t_count in app_list)
    print(f"\n📂 Nhóm [{grp.upper()}] — {len(app_list)} Apps, {total_grp_tools} Tools:")
    for aid, aname, tcount in app_list:
        app_tools = [t["name"] for t in all_tools if t["app"] == aid]
        read_t = [t["name"] for t in all_tools if t["app"] == aid and t["kind"] == "read"]
        write_t = [t["name"] for t in all_tools if t["app"] == aid and t["kind"] == "write"]
        print(f"   • App '{aid}' ({aname}): {tcount} tools")
        print(f"     - Read:  {read_t}")
        print(f"     - Write: {write_t}")

print("\n" + "=" * 70)
print("KIỂM TRA CÁC RÀNG BUỘC PHỤ THUỘC (DEPENDENCY: READ -> WRITE):")
print("=" * 70)
# Check write tools with required params that match typical read IDs
for t in all_tools:
    if t["kind"] == "write":
        id_params = [p for p in t["required"] if any(k in p for k in ["id", "phone", "date", "account", "event", "contact", "device"])]
        if id_params:
            # find candidate read tools in same app or related
            same_app_reads = [ot["name"] for ot in all_tools if ot["app"] == t["app"] and ot["kind"] == "read"]
            other_reads = [ot["name"] for ot in all_tools if ot["app"] != t["app"] and ot["kind"] == "read" and any(p.replace("_id", "") in ot["name"] for p in id_params)]
            print(f"⚡ Write Tool: {t['name']}")
            print(f"   Yêu cầu tham số: {id_params}")
            print(f"   Tool Read cùng App: {same_app_reads}")
            if other_reads:
                print(f"   Tool Read liên kết ngoài: {other_reads}")
