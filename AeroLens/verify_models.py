"""
AeroLens -- Step 5: Model ID Verifier
======================================
Checks model availability on NVIDIA NIM (integrate.api.nvidia.com)
or Nebius Token Factory, depending on NEBIUS_BASE_URL in .env.

Usage:
    python verify_models.py
"""

import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY  = os.getenv("NEBIUS_API_KEY", "")
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://integrate.api.nvidia.com/v1")

# NVIDIA NIM model IDs (canonical as of 2025)
NVIDIA_NIM_MODELS = [
    "nvidia/nemotron-3-5-lightning-30b-a3b",      # Lightning
    "nvidia/nemotron-3-5-super-vl-120b-a12b",     # Super-VL (multimodal)
    "nvidia/nemotron-3-super-120b-a12b",           # Super planner
    "nvidia/nemotron-3-ultra-550b-a55b",           # Ultra
]

# Nebius Token Factory model IDs
NEBIUS_MODELS = [
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/Nemotron-3.5-Super-VL",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]

is_nim = "integrate.api.nvidia.com" in BASE_URL
EXPECTED_MODELS = NVIDIA_NIM_MODELS if is_nim else NEBIUS_MODELS
endpoint_label  = "NVIDIA NIM" if is_nim else "Nebius Token Factory"

if not API_KEY or API_KEY == "your_nebius_api_key_here":
    print("[X] NEBIUS_API_KEY not set. Add it to your .env file and retry.")
    sys.exit(1)

print(f"Endpoint : {BASE_URL}")
print(f"Provider : {endpoint_label}")
print(f"Key      : {API_KEY[:12]}...{API_KEY[-6:]}\n")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"Fetching model list from {endpoint_label} ...\n")
try:
    models_resp = client.models.list()
    available   = {m.id for m in models_resp.data}

    print(f"{'Model ID':<50} {'Status'}")
    print("-" * 70)
    for model_id in EXPECTED_MODELS:
        status = "[OK]  AVAILABLE" if model_id in available else "[?]   NOT IN LIST"
        print(f"{model_id:<50} {status}")

    # Print all Nemotron models found (catches renamed IDs)
    nemotron_found = sorted(m.id for m in models_resp.data
                            if "nemotron" in m.id.lower())
    if nemotron_found:
        print(f"\nAll Nemotron models on this account ({len(nemotron_found)} found):")
        for mid in nemotron_found:
            marker = "  <-- not in expected list" if mid not in EXPECTED_MODELS else ""
            print(f"  {mid}{marker}")
    else:
        print("\n[!] No Nemotron models returned -- check API key permissions.")

    print(f"\nTotal models available: {len(available)}")

except Exception as exc:
    print(f"[ERROR] Could not connect to {endpoint_label}: {exc}")
    print()
    traceback.print_exc()
    sys.exit(1)
