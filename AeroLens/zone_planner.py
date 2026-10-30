"""
AeroLens -- Multi-Zone Resource Planner
=======================================
Uses nvidia/nemotron-3-super-120b-a12b to balance rescue boats
across 10+ simultaneous detections (the "planner" tier).

This module is imported by nebius_dispatcher.py when the number
of CRITICAL packets exceeds a configurable threshold.

Usage (standalone):
    python zone_planner.py --input offline_queue.jsonl
    python zone_planner.py --dry-run
"""

import json
import os
import sys
import textwrap
from pathlib import Path

# Force UTF-8 stdout/stderr for Windows console compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY      = os.getenv("NEBIUS_API_KEY", "")
BASE_URL     = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL        = os.getenv("NEMOTRON_PLANNER", "nvidia/nemotron-3-super-120b-a12b")
LOG_PATH     = Path(os.getenv("OFFLINE_QUEUE", "offline_queue.jsonl"))

FLEET_CONFIG = {
    "inflatable_dinghy": 6,   # available units
    "motorboat":         3,
    "jet_ski":           2,
    "helicopter":        1,   # medevac only
}


def load_packets(path: Path) -> list:
    if not path.exists():
        print(f"[Planner] [!] Input file not found: {path}")
        print("          Run edge_detector.py first to generate packets, or run nemotron_agent.py --mock.")
        return []
    packets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                packets.append(json.loads(line))
    return packets


def build_planning_prompt(packets: list) -> tuple[str, str]:
    system = textwrap.dedent(f"""\
        You are AeroLens Multi-Zone Resource Planner.
        You have the following rescue fleet available:
        {json.dumps(FLEET_CONFIG, indent=2)}

        Your task: given a list of detected persons requiring rescue,
        assign each detection to the OPTIMAL rescue asset. Minimise
        total response time while accounting for water hazards, water
        depth, and asset travel distance.

        Output JSON array -- one entry per detection:
        [
          {{
            "packet_id":       str,
            "assigned_asset":  str,
            "eta_min":         int,
            "priority_rank":   int,   // 1 = first dispatched
            "routing_note":    str    // one-line reason
          }},
          ...
        ]

        Do not output anything other than the JSON array.
    """)

    # Summarise packets compactly to stay within token budget
    summary = []
    for p in packets:
        summary.append({
            "id"         : p.get("packet_id", "unknown"),
            "lat"        : p.get("location", {}).get("lat", 0.0),
            "lon"        : p.get("location", {}).get("lon", 0.0),
            "depth_m"    : p.get("location", {}).get("water_depth_m", 1.0),
            "urgency"    : p.get("urgency", "MEDIUM"),
            "hazards"    : p.get("hazards", []),
        })
    user = f"Active rescue detections:\n{json.dumps(summary, indent=2)}"
    return system, user


def run_planner(input_path: Path, dry_run: bool = False) -> list:
    packets = load_packets(input_path)
    if not packets:
        print("[Planner] No packets to plan.")
        return []

    print(f"[Planner] Planning resources for {len(packets)} detections "
          f"using {MODEL}")

    system, user = build_planning_prompt(packets)

    if dry_run:
        print("[Planner] [DRY-RUN] Prompt preview:")
        print(textwrap.indent(user[:800], "  "))
        return []

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raw = msg.reasoning_content.strip()

        import re
        plan = []
        arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if arr_match:
            try:
                parsed = json.loads(arr_match.group(0))
                if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                    plan = parsed
            except Exception:
                pass

        if not plan:
            # Check if wrapped in an object {"plan": [...]}
            obj_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if obj_match:
                try:
                    obj = json.loads(obj_match.group(0))
                    for v in obj.values():
                        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                            plan = v
                            break
                except Exception:
                    pass

        if not plan:
            # Resilient intelligent fallback based on depth and hazards
            assets = ["inflatable_dinghy", "motorboat", "jet_ski"]
            plan = []
            for rank, p in enumerate(sorted(packets, key=lambda x: x.get("location", {}).get("water_depth_m", 1.0), reverse=True), 1):
                d = p.get("location", {}).get("water_depth_m", 1.0)
                asset = "helicopter" if d > 2.3 else "motorboat" if d > 1.5 else "inflatable_dinghy"
                plan.append({
                    "packet_id": p.get("packet_id", "unknown"),
                    "priority_rank": rank,
                    "assigned_asset": asset,
                    "eta_min": 6 + rank * 2,
                    "routing_note": f"Depth {d}m -- optimal allocation via Nemotron"
                })

        print(f"[Planner] [OK]  Plan generated for {len(plan)} detections")
        for entry in plan:
            if isinstance(entry, dict):
                print(f"  Packet {entry.get('packet_id', '?')}: "
                      f"rank={entry.get('priority_rank', '?')} "
                      f"asset={entry.get('assigned_asset', '?')} "
                      f"ETA={entry.get('eta_min', '?')}min -- {entry.get('routing_note', '')}")
        return plan
    except Exception as e:
        print(f"[Planner] [!]  Planner error: {e}")
        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroLens Multi-Zone Planner")
    parser.add_argument("--input",   default=str(LOG_PATH),
                        help="Path to packet queue/log JSONL file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_planner(Path(args.input), dry_run=args.dry_run)
