"""sim_capture.py — Offscreen VTK per-joint motion strip generator.

Renders the MiKo robot in an offscreen VTK window, sweeps each firmware
joint independently, and saves horizontal motion strips for VLM comparison.

Usage:  python sim_capture.py
Output: calibration_results/sim_strips/J{1..6}_strip_{front,side}.png
        calibration_results/sim_strips/J{1..6}_highlight.png
"""

import json
import os
import sys
import copy

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_BASE = os.path.join(
    SCRIPT_DIR, "MiKoBots_Studio", "assets", "MiKoBots_data", "MiKoBots"
)
SETTINGS_PATH = os.path.join(
    ASSETS_BASE, "Robot_library", "MiKo", "settings.json"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "calibration_results", "sim_strips")

# ── DH transform (mirrors kinematics_6_axis.py exactly) ───────────────

def dh_transform(theta_deg: float, alpha_deg: float, d: float, a: float) -> np.ndarray:
    alpha = np.deg2rad(alpha_deg)
    theta = np.deg2rad(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ])


def forward_kinematics(
    joint_angles: list[float],
    dh_params_raw: list[list],
) -> dict[str, np.ndarray]:
    """Return dict {J1..J6, TOOL} of 4×4 matrices (same as Studio FK)."""
    dh = [list(map(float, row)) for row in dh_params_raw]
    names = ["J1", "J2", "J3", "J4", "J5", "J6", "TOOL"]
    pos = {}
    T = np.eye(4)
    for i in range(6):
        p = list(dh[i])
        p[0] += joint_angles[i]
        T = T @ dh_transform(p[0], p[1], p[2], p[3])
        pos[names[i]] = T.copy()
    pos["TOOL"] = T.copy()
    return pos


# ── VTK helpers ────────────────────────────────────────────────────────

def load_stl_actor(filepath: str, color_name: str) -> vtk.vtkActor:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if color_name == "red":
        actor.GetProperty().SetColor(1, 0, 0)
    else:
        actor.GetProperty().SetColor(0.4, 0.4, 0.4)
    return actor


def apply_matrix(actor: vtk.vtkActor, mat4x4: np.ndarray) -> None:
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(mat4x4[r][c]))
    t = vtk.vtkTransform()
    t.SetMatrix(m)
    actor.SetUserTransform(t)


def capture_frame(render_window: vtk.vtkRenderWindow) -> Image.Image:
    """Capture current VTK render to a PIL Image."""
    render_window.Render()
    wti = vtk.vtkWindowToImageFilter()
    wti.SetInput(render_window)
    wti.SetInputBufferTypeToRGB()
    wti.ReadFrontBufferOff()
    wti.Update()
    img_data = wti.GetOutput()
    dims = img_data.GetDimensions()
    arr = vtk_to_numpy(img_data.GetPointData().GetScalars())
    arr = arr.reshape(dims[1], dims[0], 3)
    arr = np.flipud(arr)
    return Image.fromarray(arr)


def add_joint_label(img: Image.Image, text: str) -> Image.Image:
    """Burn a small text label into the top-left corner of an image."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 180, 22], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return img


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(SETTINGS_PATH, "r") as f:
        cfg = json.load(f)

    dh_params = cfg["Settings"]["Set_dh_par"][0]
    model_entries = cfg["3Dfiles"]

    JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
    LINK_LABELS = [f"Link {i}" for i in range(1, 7)]

    # ── set up offscreen VTK ──
    WIDTH, HEIGHT = 640, 480

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)

    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.SetSize(WIDTH, HEIGHT)
    renWin.AddRenderer(ren)

    # load all meshes
    link_actors: dict[str, vtk.vtkActor] = {}
    all_actors: list[vtk.vtkActor] = []

    for entry in model_entries:
        stl_rel = entry[2]  # e.g. "/Robot_library/MiKo/ROBOT_STL/Link1.STL"
        stl_path = os.path.join(ASSETS_BASE, stl_rel.lstrip("/").replace("/", os.sep))
        color = entry[3]
        link_label = entry[5]  # "Link 1", "Link 2", ... or "Base"

        if not os.path.isfile(stl_path):
            print(f"  ⚠ STL not found: {stl_path}")
            continue

        actor = load_stl_actor(stl_path, color)
        ren.AddActor(actor)
        all_actors.append(actor)

        if link_label in LINK_LABELS and link_label not in link_actors:
            link_actors[link_label] = actor

    print(f"Loaded {len(all_actors)} meshes, {len(link_actors)} movable links")

    # ── scene update function ──
    def update_scene(joint_angles: list[float]) -> None:
        fk = forward_kinematics(joint_angles, dh_params)
        for li, jname in enumerate(JOINT_NAMES):
            label = LINK_LABELS[li]
            if label in link_actors:
                apply_matrix(link_actors[label], fk[jname])

    # ── camera views ──
    VIEWS = {
        "front": {"pos": (1500, 0, 500), "up": (0, 0, 1)},
        "side":  {"pos": (0, -1500, 500), "up": (0, 0, 1)},
    }

    # safe sweep angles per firmware joint (degrees)
    SWEEP_DEG = {0: 15, 1: 12, 2: 12, 3: 20, 4: 20, 5: 30}
    N_STEPS = 8

    print(f"\nGenerating motion strips → {OUT_DIR}\n")

    # ── Phase A: multi-frame motion strips (front + side) ──
    for ji in range(6):
        sweep = SWEEP_DEG[ji]
        angles_per_step = np.linspace(0, sweep, N_STEPS)

        for view_name, view_cfg in VIEWS.items():
            cam = ren.GetActiveCamera()
            cam.SetPosition(*view_cfg["pos"])
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(*view_cfg["up"])
            cam.ParallelProjectionOn()
            ren.ResetCamera()

            frames: list[Image.Image] = []
            for step_idx, step_deg in enumerate(angles_per_step):
                joint_angles = [0.0] * 6
                joint_angles[ji] = step_deg
                update_scene(joint_angles)
                img = capture_frame(renWin)
                label_text = f"J{ji+1} @ {step_deg:.1f}deg"
                img = add_joint_label(img, label_text)
                frames.append(img)

            # horizontal motion strip
            strip_w = WIDTH * N_STEPS
            strip = Image.new("RGB", (strip_w, HEIGHT))
            for idx, frame in enumerate(frames):
                strip.paste(frame, (idx * WIDTH, 0))

            fname = f"J{ji+1}_strip_{view_name}.png"
            strip.save(os.path.join(OUT_DIR, fname))
            print(f"  ✓ {fname}")

    # ── Phase B: highlighted before/after (3D view) ──
    for ji in range(6):
        sweep = SWEEP_DEG[ji]
        angles_zero = [0.0] * 6
        angles_swept = [0.0] * 6
        angles_swept[ji] = sweep

        cam = ren.GetActiveCamera()
        cam.SetPosition(2000, -2000, 2000)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOn()
        ren.ResetCamera()

        link_label = LINK_LABELS[ji]
        orig_color = None
        if link_label in link_actors:
            orig_color = link_actors[link_label].GetProperty().GetColor()
            link_actors[link_label].GetProperty().SetColor(0, 1, 0)

        update_scene(angles_zero)
        frame_a = capture_frame(renWin)
        frame_a = add_joint_label(frame_a, f"J{ji+1} BEFORE (0 deg)")

        update_scene(angles_swept)
        frame_b = capture_frame(renWin)
        frame_b = add_joint_label(frame_b, f"J{ji+1} AFTER ({sweep} deg)")

        if orig_color is not None:
            link_actors[link_label].GetProperty().SetColor(*orig_color)

        strip = Image.new("RGB", (WIDTH * 2, HEIGHT))
        strip.paste(frame_a, (0, 0))
        strip.paste(frame_b, (WIDTH, 0))

        fname = f"J{ji+1}_highlight.png"
        strip.save(os.path.join(OUT_DIR, fname))
        print(f"  ✓ {fname} (green = moving link)")

    # ── summary ──
    total = 6 * len(VIEWS) + 6
    print(f"\nDone. {total} strips saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
