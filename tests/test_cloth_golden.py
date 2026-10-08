"""Compare the Blender port with deterministic UE 5.8 modern-evolution trajectories."""
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "waifu_physics" not in sys.modules:
    package = types.ModuleType("waifu_physics")
    package.__path__ = [os.path.join(ROOT, "waifu_physics")]
    sys.modules["waifu_physics"] = package
from waifu_physics.cloth import step_numpy
from waifu_physics.cloth.build import build, Settings
from waifu_physics.cloth.system import System

with open(os.path.join(ROOT, "tests", "golden", "chaos_cloth_ue58.json"), encoding="utf-8-sig") as handle:
    golden = json.load(handle)

failures = []
for fixture in golden["fixtures"]:
    rest = np.asarray(fixture["rest_positions"], np.float32)
    channels = np.zeros((len(rest), 3), np.float32)
    channels[:, 0] = fixture["mask"]
    mesh = build(rest, fixture["triangles"], channels, Settings(max_distance=golden["max_distance"]))
    system = System([mesh], gravity=(0, 0, -980))
    system.reset([rest])
    previous = rest
    errors = []
    for sample in fixture["frames"]:
        targets = np.asarray(sample["targets"], np.float32)
        motion = np.eye(4, dtype=np.float32)
        motion[:3, 3] = targets[0] - previous[0]
        system.step_frame(golden["dt"], targets=[targets], motions=[motion], backend=step_numpy)
        errors.append(np.linalg.norm(system.render_positions(0) - sample["positions"], axis=1))
        previous = targets
    errors = np.asarray(errors)
    rms, maximum = float(np.sqrt(np.mean(errors ** 2))), float(errors.max())
    # Explicit bounds are deliberately strict until first measurement is reviewed.
    ok = rms <= 0.5 and maximum <= 2.0
    print(f"[{'PASS' if ok else 'FAIL'}] {fixture['name']}: RMS {rms:.6f} cm; max {maximum:.6f} cm")
    if not ok:
        failures.append(fixture["name"])
if failures:
    raise AssertionError("Modern Chaos trajectory mismatch: " + ", ".join(failures))
