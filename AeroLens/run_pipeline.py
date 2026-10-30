"""
AeroLens -- Full Pipeline Runner
================================
Runs the complete AeroLens pipeline end-to-end:

  [Edge] Detect -> Queue -> [Cloud] Dispatch -> [Optional] Plan -> Report

Usage:
    python run_pipeline.py --mock                   # full demo, no files needed
    python run_pipeline.py --video assets/flood.mp4 # real drone footage
    python run_pipeline.py --image assets/rooftop/  # image or folder of images
    python run_pipeline.py --mock --vl --summary    # all models active
    python run_pipeline.py --mock --dry-run         # preview only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f">>  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroLens Full Pipeline")
    parser.add_argument("--mock",     action="store_true",
                        help="Synthetic detections (no video needed)")
    parser.add_argument("--video",    default=None,
                        help="Path to drone footage (.mp4, .avi, etc.)")
    parser.add_argument("--image",    default=None,
                        help="Path to single image or folder of images (.jpg, .png)")
    parser.add_argument("--frames",   type=int, default=2,
                        help="Mock: number of simulated frames (default: 2, generating ~4 packets)")
    parser.add_argument("--vl",       action="store_true",
                        help="Enable VL visual verification")
    parser.add_argument("--plan",     action="store_true",
                        help="Run multi-zone planner after dispatch")
    parser.add_argument("--summary",  action="store_true",
                        help="Generate post-mission summary report")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Do not call Nebius API")
    args = parser.parse_args()

    # -- Step 1: Edge Detection ------------------------------------------------
    edge_cmd = ["edge_detector.py"]
    if args.mock:
        edge_cmd += ["--mock", "--frames", str(args.frames)]
    elif args.image:
        edge_cmd += ["--image", args.image]
    elif args.video:
        edge_cmd += ["--video", args.video]
    else:
        # Default fallback
        if Path("assets/flood_drone.mp4").exists():
            edge_cmd += ["--video", "assets/flood_drone.mp4"]
        else:
            print("[*] No custom input provided; defaulting to --mock mode.")
            edge_cmd += ["--mock", "--frames", str(args.frames)]

    rc = run(edge_cmd, "STEP 1 -- Edge Detection & Telemetry Extraction")
    if rc != 0:
        print("[X] Edge detector failed. Aborting.")
        sys.exit(rc)

    # -- Step 2: Cloud Dispatch ------------------------------------------------
    dispatch_cmd = ["nebius_dispatcher.py"]
    if args.vl:
        dispatch_cmd.append("--vl")
    if args.summary:
        dispatch_cmd.append("--summary")
    if args.dry_run:
        dispatch_cmd.append("--dry-run")

    rc = run(dispatch_cmd, "STEP 2 -- Nemotron Cloud Dispatch (Lightning + VL)")
    if rc != 0:
        print("[X] Dispatcher failed.")
        sys.exit(rc)

    # -- Step 3: Multi-Zone Planner (optional) --------------------------------
    if args.plan:
        plan_cmd = ["zone_planner.py", "--input", "offline_queue.jsonl"]
        if args.dry_run:
            plan_cmd.append("--dry-run")
        run(plan_cmd, "STEP 3 -- Multi-Zone Resource Planner (Super-120B)")

    print(f"\n{'='*60}")
    print("AeroLens pipeline complete.")
    print(f"{'='*60}")
    print("Output files:")
    for f in ["offline_queue.jsonl", "dispatched_log.jsonl", "mission_report.txt"]:
        p = Path(f)
        if p.exists():
            print(f"  [OK]  {p}  ({p.stat().st_size} bytes)")
        else:
            print(f"  [--]  {p}  (not generated)")
