# mikobots-vlm-joint-eval

A multimodal evaluation pipeline that resolves robot joint identity by using GPT-4o as a cross-reference oracle between synthetic simulation and real dual-camera footage. Achieved 6/6 confirmed joint mappings (confidence 0.90–0.95) in a case where conventional diagnostic approaches failed.

## The Problem

The MiKoBots 6-DOF robot arm shipped with a 38-pin ESP32 microcontroller but was physically assembled with a 30-pin variant. This pin offset silently remapped every joint's step/direction GPIO lines — three joints stopped responding entirely, two others responded to wrong commands, and the firmware's joint numbering no longer matched the physical arm's link numbering.

Conventional debugging approaches (multimeter continuity checks, serial command logging, audio-based scanning for motor hum) all failed: they could detect *which* joints were responding, but couldn't identify *which physical link* each firmware joint controlled.

## The Solution: VLM as Evaluation Oracle

The key insight was that ground truth *is* available in simulation. The robot's VTK-based simulator knows exactly which link moves when firmware joint N is commanded. If we generate motion strips from the simulator and compare them against real camera footage, a vision model can make the match.

```
Simulator (ground truth)          Real hardware (unknown)
        │                                  │
sim_capture.py                    dual-camera footage
        │                          (calibration_results/)
        ▼                                  │
sim_strips/J{N}_strip_front.png            │
sim_strips/J{N}_strip_side.png             │
        │                                  │
        └──────────── vlm_sim_compare.py ──┘
                              │
                         GPT-4o prompt:
                    "Left: simulator shows firmware
                     joint N moving. Right: real robot.
                     Does the same link move? Which
                     physical joint is it?"
                              │
                    calibration_results/joint_map_final.json
```

## Results

| Firmware Joint | Physical Joint | Source | Confidence |
|---------------|---------------|--------|-----------|
| J1 | 2 | Anchor | 0.90 |
| J2 | 3 | VLM | 0.90 |
| J3 | 1 | Anchor | 0.95 |
| J4 | 4 | VLM | 0.90 |
| J5 | 5 | VLM | 0.90 |
| J6 | 6 | Anchor | 0.95 |

6/6 mappings confirmed. All `sim_match: true`. Three "anchor" joints (J1, J3, J6) were confirmed by prior mechanical inspection and passed as in-context examples to strengthen the VLM's reasoning on the remaining three.

## Components

### `sim_capture.py` — Headless VTK Strip Generator
- Loads robot geometry from `settings.json` (DH parameters, STL mesh paths, joint limits)
- Instantiates a VTK render window with `SetOffScreenRendering(1)` — no display required
- For each firmware joint 1–6: sweeps from 0° to N° in 8 steps (others fixed at 0°)
- Captures each frame via `vtkWindowToImageFilter` → builds horizontal motion strip
- Outputs `calibration_results/sim_strips/J{n}_strip_front.png` and `_side.png`
- 12 strips total (2 views × 6 joints)

### `vlm_sim_compare.py` — GPT-4o Cross-Reference
- Loads sim strips (ground truth — known joint identity)
- Loads real dual-camera strips from `calibration_results/dual_cam_*/`
- Sends base64-encoded image pairs to GPT-4o with a structured prompt
- Anchor joints passed as in-context examples for better grounding
- Parses structured JSON response: `{firmware_joint, physical_joint, match, confidence, reasoning}`
- Aggregates results → writes `calibration_results/joint_map_final.json`
- Flags any mismatches for manual review

## Setup

```bash
pip install openai vtk Pillow numpy
export OPENAI_API_KEY=<your-key>
```

**Step 1:** Generate simulation strips
```bash
python sim_capture.py --settings /path/to/settings.json --output calibration_results/sim_strips/
```

**Step 2:** Run VLM cross-reference
```bash
python vlm_sim_compare.py \
  --sim-strips calibration_results/sim_strips/ \
  --real-strips calibration_results/dual_cam_20260312_173249/ \
  --output calibration_results/joint_map_final.json
```

**Output:** `joint_map_final.json` with firmware→physical mappings, confidence scores, and per-joint reasoning traces.

## Why This Approach Matters

This pipeline demonstrates a general pattern: **use simulation as synthetic ground truth to evaluate real-world systems where direct measurement is difficult or impossible**. The technique is not specific to robot joint mapping — the same approach applies anywhere you have a simulator that can generate labeled examples for comparison against unlabeled real-world observations.

The VLM's role here is as an evaluation oracle: it's not controlling the robot, it's comparing two visual representations of the same underlying physical reality and extracting structured information. This is a different use case from generative AI — closer to how VLMs are used in evaluation frameworks for robotics and embodied AI.

## Hardware Context

- Robot: MiKoBots 6-DOF arm (community-maintained open-source project)
- Microcontroller: ESP32-WROOM-32 (38-pin variant)
- Cameras: Dual USB webcams (front + side views)
- Simulator: MiKoBots Studio (VTK-based, PyQt5)
- Firmware: FreeRTOS dual-core, Denavit-Hartenberg kinematics, BLE + USB serial
