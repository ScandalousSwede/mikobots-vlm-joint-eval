"""vlm_sim_compare.py — Cross-reference sim strips vs real cam strips via GPT-4o.

Sends (sim highlight, real camera strip) pairs to GPT-4o Vision and asks
it to identify which physical joint is moving in each real strip by
comparing against the simulator ground truth.

Usage:  python vlm_sim_compare.py
Output: calibration_results/joint_map_final.json

Requires: OPENAI_API_KEY in .env
"""

import base64
import json
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

SIM_DIR = os.path.join(SCRIPT_DIR, "calibration_results", "sim_strips")
REAL_DIR = os.path.join(SCRIPT_DIR, "calibration_results")
OUT_PATH = os.path.join(SCRIPT_DIR, "calibration_results", "joint_map_final.json")

MODEL = os.getenv("VLM_MODEL", "gpt-4o")

# Known anchors from joint_id.py visual confirmation
KNOWN_MAP = {
    1: 2,  # firmware J1 → physical joint 2
    3: 1,  # firmware J3 → physical joint 1
    6: 6,  # firmware J6 → physical joint 6
}
UNKNOWN_FW_JOINTS = [2, 4, 5]
REMAINING_PHYSICAL = [3, 4, 5]


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def build_prompt(fw_joint: int) -> str:
    known_str = ", ".join(
        f"firmware J{k} = physical joint {v}" for k, v in sorted(KNOWN_MAP.items())
    )
    remaining_str = ", ".join(str(p) for p in sorted(REMAINING_PHYSICAL))

    return f"""You are analyzing a 6-axis robot arm to map firmware joint numbers to physical joints.

KNOWN MAPPING (confirmed visually):
  {known_str}

UNKNOWN: firmware joints {UNKNOWN_FW_JOINTS} must map to physical joints [{remaining_str}].

IMAGE 1 (left): Simulator rendering of **firmware joint {fw_joint}** moving.
The green-highlighted link is the one driven by firmware joint {fw_joint}.
The left half shows the robot at 0 degrees, the right half shows after the sweep.

IMAGE 2 (right): Real camera strip captured while firmware joint {fw_joint} was commanded to move.
Multiple frames are tiled horizontally showing the motion over time from two camera angles.

TASK:
1. Identify which part of the robot moves in the real camera strip.
2. Compare that motion to the simulator rendering to confirm or refute that the same link moves.
3. Determine which PHYSICAL joint number (1-6) is actually moving in the real strip.

Return ONLY valid JSON:
{{
  "firmware_joint": {fw_joint},
  "physical_joint_observed": <int 1-6>,
  "confidence": <float 0.0 to 1.0>,
  "motion_description": "<brief description of what moves in real strip>",
  "sim_match": <true if real motion matches the sim rendering>,
  "reasoning": "<1-2 sentences>"
}}"""


def analyze_joint(client: OpenAI, fw_joint: int) -> Optional[dict]:
    sim_path = os.path.join(SIM_DIR, f"J{fw_joint}_highlight.png")
    real_path = os.path.join(REAL_DIR, f"J{fw_joint}_strip.jpg")

    if not os.path.isfile(sim_path):
        print(f"  ⚠ Missing sim strip: {sim_path}")
        return None
    if not os.path.isfile(real_path):
        print(f"  ⚠ Missing real strip: {real_path}")
        return None

    sim_b64 = encode_image(sim_path)
    real_b64 = encode_image(real_path)
    sim_mime = get_image_media_type(sim_path)
    real_mime = get_image_media_type(real_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(fw_joint)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{sim_mime};base64,{sim_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{real_mime};base64,{real_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        print(f"  ⚠ J{fw_joint}: non-JSON response: {raw[:200]}")
        return {"firmware_joint": fw_joint, "raw_response": raw, "error": "parse_fail"}
    except Exception as e:
        print(f"  ⚠ J{fw_joint}: API error: {e}")
        return {"firmware_joint": fw_joint, "error": str(e)}


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("=" * 50)
    print("  VLM Sim ↔ Real Cross-Reference")
    print("=" * 50)
    print(f"  Model: {MODEL}")
    print(f"  Known: {KNOWN_MAP}")
    print(f"  Unknown fw joints: {UNKNOWN_FW_JOINTS}")
    print(f"  Remaining physical: {REMAINING_PHYSICAL}")
    print()

    results = {}

    # Analyze ALL 6 joints (known ones serve as validation anchors)
    for fw_j in range(1, 7):
        print(f"Analyzing firmware J{fw_j}...")
        result = analyze_joint(client, fw_j)
        if result:
            results[f"J{fw_j}"] = result
            phys = result.get("physical_joint_observed", "?")
            conf = result.get("confidence", 0)
            match = result.get("sim_match", None)
            print(f"  → physical J{phys}  (confidence={conf:.2f}, sim_match={match})")
        else:
            print(f"  → skipped (missing data)")
        time.sleep(0.5)

    # Build final mapping
    final_map = dict(KNOWN_MAP)
    for fw_j in UNKNOWN_FW_JOINTS:
        key = f"J{fw_j}"
        if key in results and "physical_joint_observed" in results[key]:
            final_map[fw_j] = results[key]["physical_joint_observed"]

    # Validation: check anchors
    anchor_ok = True
    for fw_j, expected_phys in KNOWN_MAP.items():
        key = f"J{fw_j}"
        if key in results:
            observed = results[key].get("physical_joint_observed")
            if observed != expected_phys:
                print(f"\n  ⚠ ANCHOR MISMATCH: fw J{fw_j} expected physical {expected_phys}, got {observed}")
                anchor_ok = False

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "known_anchors": {str(k): v for k, v in KNOWN_MAP.items()},
        "per_joint_results": results,
        "final_firmware_to_physical_map": {str(k): v for k, v in sorted(final_map.items())},
        "anchor_validation_passed": anchor_ok,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("  FINAL FIRMWARE → PHYSICAL MAP")
    print("=" * 50)
    for fw_j in range(1, 7):
        phys = final_map.get(fw_j, "?")
        source = "anchor" if fw_j in KNOWN_MAP else "VLM"
        print(f"  firmware J{fw_j} → physical J{phys}  ({source})")

    print(f"\nAnchor validation: {'PASS ✓' if anchor_ok else 'FAIL ✗'}")
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
