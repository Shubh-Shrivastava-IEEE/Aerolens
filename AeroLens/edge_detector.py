"""
AeroLens -- Step 3: Edge Detector & Telemetry Extractor
=======================================================
Simulates a drone-mounted GPU module running in OFFLINE mode.

Pipeline:
  flood_drone.mp4 / images -> YOLOv8n (CUDA/TRT) -> person detections
  -> attach simulated GPS + water hazard tags
  -> write compact JSON packets to offline_queue.jsonl

Each packet is ~400 bytes so it can be burst-transmitted
over a 2G SMS channel when connectivity resumes.

Usage:
    python edge_detector.py --video path/to/video.mp4
    python edge_detector.py --image path/to/images/
    python edge_detector.py --mock
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# -- Config --------------------------------------------------------------------
WEIGHTS      = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
ENGINE_PATH  = Path(os.getenv("TENSORRT_ENGINE", "models/yolov8n_flood.engine"))
VIDEO_PATH   = os.getenv("DRONE_VIDEO_PATH", "assets/flood_drone.mp4")
CONF_THRESH  = float(os.getenv("DETECTION_CONF", "0.45"))
QUEUE_FILE   = Path(os.getenv("OFFLINE_QUEUE", "offline_queue.jsonl"))

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Simulated disaster zone bounding box (Chennai flood zone example)
GPS_ORIGIN   = (13.0827, 80.2707)   # lat, lon -- Chennai
GPS_RADIUS_M = 2000                 # 2 km radius scatter

# Water hazard labels mapped to local hazard codes
HAZARD_TAGS  = ["submerged_road", "fast_current", "debris_field",
                "collapsed_roof", "chemical_spill_risk", "power_line_down"]

# -- Helpers -------------------------------------------------------------------

def _offset_gps(lat: float, lon: float, dx_m: float, dy_m: float):
    """Offset a GPS coordinate by dx_m (east) and dy_m (north)."""
    dlat = dy_m / 111_320
    dlon = dx_m / (111_320 * math.cos(math.radians(lat)))
    return round(lat + dlat, 6), round(lon + dlon, 6)


def _simulated_gps():
    """Return a GPS point randomly scattered around the disaster zone."""
    angle = random.uniform(0, 2 * math.pi)
    dist  = random.uniform(0, GPS_RADIUS_M)
    dx    = dist * math.cos(angle)
    dy    = dist * math.sin(angle)
    return _offset_gps(*GPS_ORIGIN, dx, dy)


def _water_depth_estimate(bbox_area_ratio: float) -> float:
    """
    Heuristic: larger detected bounding box relative to frame ->
    person is closer to camera -> less submerged.
    Returns estimated water depth in metres [0.3 - 2.5].
    """
    depth = 2.5 - (bbox_area_ratio * 8.0)
    return round(max(0.3, min(2.5, depth)), 2)


def _build_packet(frame_id: int, track_id: int,
                  bbox: list, conf: float,
                  frame_w: int, frame_h: int,
                  drone_alt_m: float = 45.0) -> dict:
    """Construct a telemetry JSON packet for one detection."""
    lat, lon = _simulated_gps()
    bbox_area_ratio = (bbox[2] * bbox[3]) / (frame_w * frame_h)
    water_depth     = _water_depth_estimate(bbox_area_ratio)
    hazards         = random.sample(HAZARD_TAGS, k=random.randint(1, 3))
    urgency         = "CRITICAL" if water_depth > 1.8 else \
                      "HIGH"     if water_depth > 1.0 else "MEDIUM"

    return {
        "packet_id"   : str(uuid.uuid4())[:8],
        "mission_id"  : "AEROLENS-001",
        "timestamp"   : datetime.now(timezone.utc).isoformat(),
        "frame_id"    : frame_id,
        "track_id"    : track_id,
        "detection"   : {
            "class"       : "person",
            "confidence"  : round(float(conf), 3),
            "bbox_xywh"   : [round(v, 1) for v in bbox],
            "bbox_area_r" : round(bbox_area_ratio, 4),
        },
        "location"    : {
            "lat"         : lat,
            "lon"         : lon,
            "alt_m"       : drone_alt_m,
            "water_depth_m": water_depth,
        },
        "hazards"     : hazards,
        "urgency"     : urgency,
        "sms_ready"   : True,
    }


def _write_packet(packet: dict, queue: Path) -> None:
    """Append a JSON packet to the offline queue (newline-delimited JSON)."""
    queue.parent.mkdir(parents=True, exist_ok=True)
    with open(queue, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(packet) + "\n")
    print(f"  [+] Packet {packet['packet_id']} queued "
          f"| urgency={packet['urgency']} "
          f"| water={packet['location']['water_depth_m']}m "
          f"| hazards={packet['hazards']}")


# -- Mock Mode (no video file needed) -----------------------------------------

def run_mock(n_frames: int = 30, n_per_frame: int = 2) -> None:
    """Generate synthetic detections without real video -- demo/CI mode."""
    print("\n[AeroLens Edge] MOCK MODE -- generating synthetic detections")
    print(f"[AeroLens Edge]     {n_frames} frames x {n_per_frame} detections each\n")
    # Reset queue file for a clean mock run
    if QUEUE_FILE.exists():
        QUEUE_FILE.unlink()
    total = 0
    for frame_id in range(n_frames):
        for tid in range(n_per_frame):
            bbox = [
                random.uniform(100, 500),  # x
                random.uniform(100, 400),  # y
                random.uniform(30, 120),   # w
                random.uniform(60, 160),   # h
            ]
            conf   = random.uniform(CONF_THRESH, 0.98)
            packet = _build_packet(frame_id, frame_id * 10 + tid,
                                   bbox, conf, 640, 480)
            _write_packet(packet, QUEUE_FILE)
            total += 1
        time.sleep(0.02)

    print(f"\n[AeroLens Edge] Done. {total} packets -> {QUEUE_FILE}")


# -- Real Image Mode -----------------------------------------------------------

def run_images(image_path: str) -> None:
    """Process an image or folder of images with YOLOv8."""
    import cv2
    from ultralytics import YOLO

    # Reset queue file for a clean image batch
    if QUEUE_FILE.exists():
        QUEUE_FILE.unlink()

    model_src = str(ENGINE_PATH) if ENGINE_PATH.exists() else WEIGHTS
    print(f"\n[AeroLens Edge] Loading model: {model_src}")
    model = YOLO(model_src)

    p = Path(image_path)
    if p.is_dir():
        image_files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]])
    elif p.is_file():
        image_files = [p]
    else:
        image_files = sorted(Path(".").glob(image_path))

    if not image_files:
        raise FileNotFoundError(f"No valid image files found at: {image_path}")

    print(f"[AeroLens Edge] Found {len(image_files)} image(s) to process.")
    detections_total = 0

    for idx, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [!] Could not read {img_path}, skipping.")
            continue

        frame_h, frame_w = frame.shape[:2]
        results = model(
            frame,
            classes=[0],
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False,
        )

        for det in results[0].boxes:
            bbox   = det.xywh[0].tolist()
            conf   = float(det.conf[0])
            packet = _build_packet(idx, idx, bbox, conf, frame_w, frame_h)
            _write_packet(packet, QUEUE_FILE)
            detections_total += 1

    print(f"\n[AeroLens Edge] Done. {detections_total} detections across "
          f"{len(image_files)} image(s) -> {QUEUE_FILE}")


# -- Real Video Mode -----------------------------------------------------------

def run_video(video_path: str) -> None:
    """Process an actual drone video file with YOLOv8."""
    import cv2
    from ultralytics import YOLO

    model_src = str(ENGINE_PATH) if ENGINE_PATH.exists() else WEIGHTS
    print(f"\n[AeroLens Edge] Loading model: {model_src}")
    model = YOLO(model_src)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {video_path}\n"
            "Tip: run with --mock flag if you don't have drone footage."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[AeroLens Edge] Video: {total_frames} frames @ {fps:.1f} FPS "
          f"({frame_w}x{frame_h})")

    detections_total = 0
    frame_id         = 0
    process_every    = max(1, int(fps / 5))   # sample 5 frames per second

    print("[AeroLens Edge] Processing ... (Ctrl+C to stop early)\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % process_every == 0:
            results = model(
                frame,
                classes=[0],          # class 0 = person (COCO)
                conf=CONF_THRESH,
                device=DEVICE,
                verbose=False,
            )
            for det in results[0].boxes:
                bbox   = det.xywh[0].tolist()
                conf   = float(det.conf[0])
                packet = _build_packet(frame_id, int(det.id[0]) if det.id else frame_id,
                                       bbox, conf, frame_w, frame_h)
                _write_packet(packet, QUEUE_FILE)
                detections_total += 1

        frame_id += 1

    cap.release()
    print(f"\n[AeroLens Edge] Done. {detections_total} detections across "
          f"{frame_id} frames -> {QUEUE_FILE}")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="AeroLens Edge Detector")
    parser.add_argument("--video", default=None,
                        help="Path to drone footage (.mp4, .avi, etc.)")
    parser.add_argument("--image", default=None,
                        help="Path to single image or folder of images (.jpg, .png)")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic detections (no video/image needed)")
    parser.add_argument("--frames", type=int, default=30,
                        help="Mock mode: number of simulated frames")
    args = parser.parse_args()

    if args.mock:
        run_mock(n_frames=args.frames)
    elif args.image:
        run_images(args.image)
    elif args.video:
        run_video(args.video)
    else:
        if Path(VIDEO_PATH).exists():
            run_video(VIDEO_PATH)
        else:
            print("[AeroLens Edge] No input specified or default video not found.")
            print("  Usage options:")
            print("    python edge_detector.py --video path/to/drone.mp4")
            print("    python edge_detector.py --image path/to/images/")
            print("    python edge_detector.py --mock")
