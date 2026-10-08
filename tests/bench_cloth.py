"""Standalone solver timing; excludes topology build and Blender depsgraph cost.

    python tests/bench_cloth.py [frames=300]
"""
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cloth_build import api
from test_cloth_agreement import shape

build, Settings, System, reference = api()
from waifu_physics.cloth import native

width, height = 40, 38
positions = np.array([[x*2., 0., -y*2.] for y in range(height) for x in range(width)], np.float32)
triangles = []
for y in range(height-1):
    for x in range(width-1):
        a = y*width+x
        triangles.extend([[a, a+1, a+width], [a+1, a+width+1, a+width]])
rgb = np.zeros_like(positions)
rgb[:, 0] = 1
rgb[:width, 0] = 0
t0 = time.perf_counter()
system = System([build(positions, triangles, rgb, Settings(iterations=4, substeps=2))])
build_ms = (time.perf_counter()-t0)*1000
colliders = [shape(0, location=(40, 4, -35), radius=12), shape(5, location=(0, 0, -80))]
frames = int(sys.argv[1]) if len(sys.argv)>1 else 300
print(f"{system.n} particles, {len(system.edges)} edges, {len(system.bending)} bends, "
      f"{len(system.area)} axial, {len(system.tethers)} tethers; build {build_ms:.1f} ms")
for iterations, substeps in ((1, 1), (4, 2)):
    for backend in (reference, native.backend()):
        s = copy.deepcopy(system)
        s.meshes[0].settings.iterations = iterations
        s.meshes[0].settings.substeps = substeps
        for _ in range(10):
            s.step_frame(1/60, colliders=colliders, backend=backend)
        times = []
        for frame in range(frames):
            target = positions.copy()
            target[:, 1] += np.float32(2*np.sin(frame*.04))
            start = time.perf_counter()
            s.step_frame(1/60, targets=[target], colliders=colliders, backend=backend)
            times.append((time.perf_counter()-start)*1000)
        print(f"{backend.NAME}: median {np.median(times):.3f} ms/frame; p95 {np.percentile(times,95):.3f}; "
              f"{frames} frames, {iterations} iterations, {substeps} substeps; finite={np.isfinite(s.x).all()}")
print("native fallback reason:", native.reason() or "none")
