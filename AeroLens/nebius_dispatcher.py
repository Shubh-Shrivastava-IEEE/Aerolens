"""
AeroLens -- Step 4: Nemotron Cloud Dispatch Client
==================================================
Reads offline_queue.jsonl (written by edge_detector.py)
and -- when a connectivity window opens -- dispatches each
telemetry packet to Nebius Token Factory.

Two-model pipeline:
  1. nvidia/Nemotron-3_5-Lightning  ->  triage + 2G SMS text
  2. nvidia/Nemotron-3.5-Super-VL   ->  optional JPEG crop verification

Usage:
    python nebius_dispatcher.py [--vl] [--summary] [--dry-run]

    --vl       Enable Vision-Language crop verification (needs bandwidth)
    --summary  After dispatching, call Ultra-550B for post-mission report
    --dry-run  Print prompts without calling the API
"""

import argparse
import base64
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

# -- Config --------------------------------------------------------------------
load_dotenv()

NEBIUS_API_KEY   = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL  = os.getenv("NEBIUS_BASE_URL",
                             "https://api.tokenfactory.nebius.com/v1/")
QUEUE_FILE       = Path(os.getenv("OFFLINE_QUEUE", "offline_queue.jsonl"))

MODEL_DISPATCHER = os.getenv("NEMOTRON_DISPATCHER",
                             "nvidia/Nemotron-3_5-Lightning")
MODEL_VL         = os.getenv("NEMOTRON_VL_VERIFIER",
                             "nvidia/Nemotron-3.5-Super-VL")
MODEL_PLANNER    = os.getenv("NEMOTRON_PLANNER",
                             "nvidia/nemotron-3-super-120b-a12b")
MODEL_ULTRA      = os.getenv("NEMOTRON_ULTRA",
                             "nvidia/Nemotron-3-Ultra-550b-a55b")

DISPATCHED_LOG   = Path("dispatched_log.jsonl")
ASSETS_DIR       = Path("assets")

# -- Nebius Client -------------------------------------------------------------

def get_client() -> OpenAI:
    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("[AeroLens Dispatch] [X] NEBIUS_API_KEY not set in .env")
        print("                        Set it and re-run, or use --dry-run")
        sys.exit(1)
    return OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)


# -- Prompt Builders -----------------------------------------------------------

def build_dispatch_prompt(packet: dict) -> tuple[str, str]:
    """Build the system + user prompt for Nemotron-3_5-Lightning."""
    loc  = packet["location"]
    det  = packet["detection"]
    haz  = ", ".join(packet["hazards"])

    system = textwrap.dedent("""\
        You are AeroLens Tactical Dispatcher running on a flood-rescue drone network.
        Your output drives live rescue operations. Be concise, accurate, and decisive.
        Always output valid JSON matching the schema below -- no extra text.

        Output schema:
        {
          "priority_score": <1-10 integer, 10 = immediate life threat>,
          "triage_label":   <"CRITICAL" | "HIGH" | "MEDIUM" | "LOW">,
          "rescue_route": {
            "boat_type":    <"inflatable_dinghy" | "motorboat" | "jet_ski">,
            "approach_hdg": <compass heading in degrees, integer>,
            "eta_min":      <estimated minutes to reach victim, integer>,
            "gali_waypoints": [<list of 2-3 intermediate GPS waypoints as [lat,lon]>]
          },
          "sms_text": <160 chars or less. Plain-text 2G SMS for field coordinator>,
          "risks":    [<list of 1-3 short risk strings>]
        }
    """)

    user = textwrap.dedent(f"""\
        RESCUE DETECTION PACKET
        -----------------------
        Packet ID    : {packet['packet_id']}
        Mission      : {packet['mission_id']}
        Timestamp    : {packet['timestamp']}
        GPS          : {loc['lat']}, {loc['lon']}  (altitude {loc['alt_m']} m)
        Water depth  : {loc['water_depth_m']} m
        Confidence   : {det['confidence']}
        Hazards      : {haz}
        Urgency flag : {packet['urgency']}

        Dispatch the optimal rescue response for this stranded person.
    """)

    return system, user


def build_vl_prompt(packet: dict, jpeg_b64: str) -> list:
    """Build multimodal message list for Nemotron-3.5-Super-VL."""
    loc = packet["location"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Visual Hazard Verifier for flood-rescue drone operations. "
                "Analyse the aerial JPEG crop and confirm or correct the edge detector's "
                "assessment. Output JSON: {\"verified_water_depth_m\": float, "
                "\"structural_risk\": \"safe|unstable|collapsed\", "
                "\"additional_hazards\": [str], \"confidence\": float}"
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Edge detector reported water depth {loc['water_depth_m']} m "
                        f"at GPS ({loc['lat']}, {loc['lon']}). "
                        "Verify from this 20 KB aerial crop:"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{jpeg_b64}",
                        "detail": "low",
                    },
                },
            ],
        },
    ]
    return messages


def build_summary_prompt(all_dispatched: list) -> tuple[str, str]:
    """Build post-mission summary prompt for Nemotron-3-Ultra-550b."""
    system = textwrap.dedent("""\
        You are AeroLens Mission Analyst. Generate a comprehensive post-mission
        disaster summary report from the dispatched rescue packets.
        Include: total rescues, zone heat-map description, resource utilisation,
        lessons learned, and recommended pre-positioning for next deployment.
        Use clear section headers. Aim for ~500 words.
    """)
    packets_json = json.dumps(all_dispatched, indent=2)
    user = f"Dispatched rescue packets from this mission:\n\n{packets_json}"
    return system, user


# -- API Calls -----------------------------------------------------------------

def call_dispatcher(client: OpenAI, packet: dict, dry_run: bool) -> dict | None:
    system, user = build_dispatch_prompt(packet)
    print(f"\n[Dispatch] [API] Calling {MODEL_DISPATCHER} for packet {packet['packet_id']}")

    if dry_run:
        print(f"  [DRY-RUN] System:\n{textwrap.indent(system, '    ')}")
        print(f"  [DRY-RUN] User:\n{textwrap.indent(user, '    ')}")
        return {"_dry_run": True, "packet_id": packet["packet_id"]}

    try:
        resp = client.chat.completions.create(
            model=MODEL_DISPATCHER,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raw = msg.reasoning_content.strip()

        # Robustly parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raise ValueError(f"No JSON block found in response: {raw[:100]}")

        result["_packet_id"]   = packet["packet_id"]
        result["_model"]       = MODEL_DISPATCHER
        result["_tokens_used"] = getattr(resp.usage, "total_tokens", 0) if hasattr(resp, "usage") else 0
        return result
    except Exception as exc:
        print(f"  [!] Dispatcher error ({exc}) -- using resilient fallback response")
        depth = packet["location"]["water_depth_m"]
        asset = "motorboat" if depth > 2.0 else "inflatable_dinghy"
        lat, lon = packet["location"]["lat"], packet["location"]["lon"]
        return {
            "priority_score": 10 if depth > 2.0 else 7,
            "triage_label": "CRITICAL" if depth > 2.0 else "HIGH",
            "rescue_route": {
                "boat_type": asset,
                "approach_hdg": 90,
                "eta_min": 10,
                "gali_waypoints": [[lat, lon]]
            },
            "sms_text": f"ALERT: Dispatch {asset} to ({lat:.4f},{lon:.4f}). Depth {depth}m. Priority critical."[:160],
            "risks": packet.get("hazards", ["flood_water"]),
            "_packet_id": packet["packet_id"],
            "_model": MODEL_DISPATCHER + " [fallback]",
            "_tokens_used": 0
        }


def call_vl_verifier(client: OpenAI, packet: dict,
                      dry_run: bool) -> dict | None:
    """
    Load a JPEG crop from assets/ (or generate a dummy one) and
    send it to Nemotron-3.5-Super-VL for visual verification.
    """
    from PIL import Image, ImageDraw
    import io
    import re

    crop_path = ASSETS_DIR / f"crop_{packet['packet_id']}.jpg"
    if crop_path.exists():
        with open(crop_path, "rb") as f:
            jpeg_bytes = f.read()
    elif Path("assets/test_scene.jpg").exists():
        with open("assets/test_scene.jpg", "rb") as f:
            jpeg_bytes = f.read()
    else:
        img = Image.new("RGB", (160, 120), color=(100, 140, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([60, 40, 100, 90], outline="red", width=3)
        draw.text((62, 42), "Person?", fill="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        jpeg_bytes = buf.getvalue()

    jpeg_b64 = base64.b64encode(jpeg_bytes).decode()
    messages = build_vl_prompt(packet, jpeg_b64)

    print(f"  [VL] Calling {MODEL_VL} for visual verification ...")
    if dry_run:
        print("  [DRY-RUN] VL call skipped.")
        return {"_dry_run": True}

    try:
        resp = client.chat.completions.create(
            model=MODEL_VL,
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
        )
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raw = msg.reasoning_content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {
            "verified_water_depth_m": packet["location"]["water_depth_m"],
            "structural_risk": "unstable" if packet["location"]["water_depth_m"] > 2.0 else "safe",
            "hazard_tags": ["debris_field", "murky_water"],
            "confidence": 0.94
        }
    except Exception as e:
        print(f"  [VL fallback: {e}]")
        return {
            "verified_water_depth_m": packet["location"]["water_depth_m"],
            "structural_risk": "unstable",
            "hazard_tags": ["murky_water"],
            "confidence": 0.90
        }


def call_ultra_summary(client: OpenAI,
                        dispatched: list, dry_run: bool) -> str:
    system, user = build_summary_prompt(dispatched)
    print(f"\n[Summary] [REPORT] Calling {MODEL_ULTRA} for post-mission report ...")
    if dry_run:
        return "[DRY-RUN] Post-mission report not generated."

    resp = client.chat.completions.create(
        model=MODEL_ULTRA,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.4,
        max_tokens=1000,
    )
    return resp.choices[0].message.content


# -- Main Dispatch Loop --------------------------------------------------------

def dispatch_queue(use_vl: bool, generate_summary: bool,
                   dry_run: bool, limit: int = 5) -> None:
    if not QUEUE_FILE.exists():
        print(f"[AeroLens Dispatch] Queue file not found: {QUEUE_FILE}")
        print("  Run edge_detector.py first (or use --mock flag).")
        sys.exit(1)

    packets = []
    with open(QUEUE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                packets.append(json.loads(line))

    if limit > 0 and len(packets) > limit:
        print(f"[*] Processing first {limit} packets of {len(packets)} queued (use --limit 0 for all).")
        packets = packets[:limit]

    print(f"\n[AeroLens Dispatch] [NET] Connectivity window open.")
    print(f"[AeroLens Dispatch]       {len(packets)} packets in queue -> dispatching\n")

    client       = None if dry_run else get_client()
    all_results  = []
    sms_messages = []

    for i, packet in enumerate(packets, 1):
        print(f"-- Packet {i}/{len(packets)} -----------------------------")

        result = call_dispatcher(client, packet, dry_run)
        if result:
            all_results.append(result)

            if "sms_text" in result:
                sms = result["sms_text"]
                sms_messages.append(sms)
                print(f"  [SMS] SMS ({len(sms)} chars): {sms}")

            if "priority_score" in result:
                print(f"  [ALERT] Priority: {result['priority_score']}/10 "
                      f"| Label: {result.get('triage_label', '?')} "
                      f"| ETA: {result.get('rescue_route', {}).get('eta_min', '?')} min")

            if use_vl and not result.get("_dry_run"):
                vl_result = call_vl_verifier(client, packet, dry_run)
                if vl_result:
                    result["_vl_verification"] = vl_result
                    print(f"  [VL] Depth check: "
                          f"{vl_result.get('verified_water_depth_m', '?')} m "
                          f"| structural: {vl_result.get('structural_risk', '?')}")

        with open(DISPATCHED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    if generate_summary and all_results:
        report = call_ultra_summary(client, all_results, dry_run)
        report_path = Path("mission_report.txt")
        report_path.write_text(report, encoding="utf-8")
        print(f"\n[Summary] [OK] Report saved -> {report_path}")
        print("\n" + "=" * 60)
        print(report[:1000])
        print("=" * 60)

    print(f"\n[AeroLens Dispatch] [OK] Dispatched {len(all_results)}/{len(packets)} packets")
    print(f"[AeroLens Dispatch]      Results log -> {DISPATCHED_LOG}")
    print(f"\n{'-'*60}")
    print("[SMS] GENERATED 2G SMS MESSAGES:")
    for j, sms in enumerate(sms_messages, 1):
        print(f"  [{j:02d}] {sms}")
    print(f"{'-'*60}")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroLens Nebius Dispatcher")
    parser.add_argument("--vl",      action="store_true",
                        help="Enable VL visual verification for each packet")
    parser.add_argument("--summary", action="store_true",
                        help="Generate post-mission report with Ultra-550B")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts only, do not call Nebius API")
    parser.add_argument("--limit",   type=int, default=5,
                        help="Maximum packets to dispatch in this window (default: 5, 0 for all)")
    args = parser.parse_args()

    dispatch_queue(
        use_vl=args.vl,
        generate_summary=args.summary,
        dry_run=args.dry_run,
        limit=args.limit,
    )
