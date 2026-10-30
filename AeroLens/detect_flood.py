"""
AeroLens -- Direct Image Flood & Person Detector
=================================================
Provide any photo/aerial drone picture to detect stranded people in floodwaters
using local Edge YOLOv8 + NVIDIA Nemotron-3.5-Super-VL multimodal AI.

Usage:
    python detect_flood.py --image path/to/photo.jpg
    python detect_flood.py --image assets/test_scene.jpg
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

# Force UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from ultralytics import YOLO

load_dotenv()

API_KEY  = os.getenv("NEBIUS_API_KEY", "")
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://integrate.api.nvidia.com/v1")
VL_MODEL = os.getenv("NEMOTRON_VL_VERIFIER", "nvidia/nemotron-3.5-super-vl-120b-a12b")
LIGHTNING_MODEL = os.getenv("NEMOTRON_DISPATCHER", "nvidia/nemotron-3.5-lightning-30b-a3b")
WEIGHTS  = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")


def analyze_image(image_path: str, conf_thresh: float = 0.25):
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"[Error] Image not found: {image_path}")
        sys.exit(1)

    print("=" * 64)
    print(f"  AEROLENS FLOOD PERSON DETECTION")
    print(f"  Target Image : {img_file.resolve()}")
    print("=" * 64)

    # 1. Edge Vision: YOLOv8 local detection
    print("\n[STEP 1] Running Local YOLOv8 Edge Detection...")
    model = YOLO(WEIGHTS)
    frame = cv2.imread(str(img_file))
    if frame is None:
        print(f"[Error] OpenCV could not load image: {image_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    results = model(frame, classes=[0], conf=conf_thresh, verbose=False)
    boxes = results[0].boxes

    person_count = len(boxes)
    print(f"  -> YOLOv8 Persons Detected: {person_count}")

    # Draw boxes on frame and save annotated copy
    annotated = frame.copy()
    detections_info = []
    for i, box in enumerate(boxes):
        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        detections_info.append({"id": i + 1, "bbox": xyxy, "conf": round(conf, 3)})
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        label = f"Person {i+1}: {conf:.2f}"
        cv2.putText(annotated, label, (xyxy[0], max(20, xyxy[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_annotated = img_file.parent / f"detected_{img_file.name}"
    cv2.imwrite(str(out_annotated), annotated)
    print(f"  -> Annotated image saved: {out_annotated.name}")

    # 2. Multimodal Nemotron Super-VL Reasoning
    print(f"\n[STEP 2] Calling NVIDIA Nemotron-3.5-Super-VL ({VL_MODEL})...")
    with open(img_file, "rb") as fh:
        jpeg_b64 = base64.b64encode(fh.read()).decode()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt = (
        "You are AeroLens Autonomous Flood Disaster Vision AI. "
        "Analyze the aerial/ground flood photograph with precision. "
        "Determine if any person or survivor is stranded or present in floodwaters, "
        "assess flood depth, surrounding water current, structural collapse or hazards, "
        "and determine emergency rescue urgency. "
        "Respond with a clear structured analysis and a final JSON block."
    )

    user_query = (
        f"Analyze this image. Local edge detection flagged {person_count} potential person(s). "
        "Answer the following:\n"
        "1. Is someone stranded in or near floodwaters?\n"
        "2. What is their environment (rooftop, tree, submerged road, water)?\n"
        "3. Estimated water depth and hazard risks (fast currents, debris, downed electrical lines)?\n"
        "4. Triage priority level (CRITICAL / HIGH / MEDIUM / SAFE)?\n"
        "5. What rescue craft should be dispatched (motorboat, inflatable dinghy, helicopter)?\n"
        "6. Provide a <= 160-character plain text 2G SMS for field rescue coordinators."
    )

    try:
        resp = client.chat.completions.create(
            model=VL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}", "detail": "high"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", "") or ""

        print("\n" + "=" * 64)
        print("NVIDIA NEMOTRON SUPER-VL ASSESSMENT:")
        print("=" * 64)
        if content:
            print(content)
        elif reasoning:
            # If model used reasoning tokens, display conclusion
            print(reasoning[-1200:])

    except Exception as e:
        print(f"[!] API call note: {e}")
        # Resilient assessment
        print("\n" + "=" * 64)
        print("AEROLENS FLOOD ASSESSMENT SUMMARY:")
        print("=" * 64)
        urgency = "CRITICAL" if person_count > 0 else "SEARCH_GRID"
        asset = "motorboat" if person_count > 0 else "inflatable_dinghy"
        print(f"  Status        : {'SURVIVOR DETECTED IN FLOOD ZONE' if person_count > 0 else 'NO DIRECT PERSON SIGHTED'}")
        print(f"  Persons Count : {person_count}")
        print(f"  Triage Level  : {urgency}")
        print(f"  Recommended   : Dispatch {asset}")
        print(f"  2G SMS Alert  : RESCUE ALERT: Person in flood zone at image coords. Dispatch {asset}. Triage {urgency}.")

    print("=" * 64)
    print(f"\n[Completed] Visual inspection finished.")
    print(f"Annotated result -> {out_annotated.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroLens Flood & Person Image Detector")
    parser.add_argument("--image", required=True, help="Path to input photo (.jpg, .png, etc.)")
    parser.add_argument("--conf",  type=float, default=0.25, help="YOLO person confidence threshold (default: 0.25)")
    args = parser.parse_args()

    analyze_image(args.image, conf_thresh=args.conf)
