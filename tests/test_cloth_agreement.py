"""C/reference agreement plus analytical collision and frame-state checks."""
import copy
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cloth_build import api


def shape(kind=0, **kwargs):
    values = dict(kind=kind, location=(0, 0, 0), rotation=(0, 0, 0, 1), radius=5.,
                  radius1=3., length=10., extent=(5., 5., 5.), enabled=True)
    values.update(kwargs)
    return SimpleNamespace(**values)


def free_system(positions=None, **kwargs):
    build, Settings, System, ref = api()
    cfg = dict(edge_stiffness=0, bending_stiffness=0, area_stiffness=0, tether_stiffness=0,
               gravity_scale=0, damping=0, anim_drive_damping=0, max_distance=10000)
    cfg.update(kwargs)
    p = positions if positions is not None else [[2, 0, 0], [3, 1, 0], [2, 1, 1]]
    return System([build(p, [[0, 1, 2]], settings=Settings(**cfg))]), ref


class ClothStepTests(unittest.TestCase):
    def test_collapsed_spring_uses_engine_safe_normalize_axis(self):
        s, ref = free_system([[0, 0, 0], [10, 0, 0], [0, 10, 0]], edge_stiffness=1)
        # Isolate one physical constraint in the public flat-array solver state.
        s.edges = np.array([[0, 1]], np.int32)
        s.edge_length = np.array([10], np.float32)
        s.edge_starts = np.array([0, 1], np.int32)
        s._batches["edges"] = [[(0, 1)]]
        s.inv_mass[:] = [1, 1, 0]
        s.x[1] = s.x[0]
        from waifu_physics.cloth import native
        for backend in (ref, native.backend()):
            copy_s = copy.deepcopy(s)
            copy_s.step_frame(1/60, backend=backend)
            np.testing.assert_allclose(copy_s.x[:2], [[5, 0, 0], [-5, 0, 0]], atol=1e-6)

    def test_area_axial_mass_and_engine_multiplier_restore_altitude(self):
        s, ref = free_system([[0, 0, 0], [10, 0, 0], [0, 10, 0]], area_stiffness=1)
        s.x[0] = [-2, -2, 0]
        s.step_frame(1/60, backend=ref)
        expected = [[-2/3, -2/3, 0], [10-2/3, -2/3, 0], [-2/3, 10-2/3, 0]]
        np.testing.assert_allclose(s.x, expected, atol=2e-6)

    def test_iteration_count_rounding_clamping_and_scene_substeps(self):
        s, ref = free_system(iterations=4, substeps=3)
        s.step_frame(1/120, backend=ref)
        self.assertEqual(s.iterations[0], 2)
        self.assertAlmostEqual(float(s.step_dt), 1/360)
        s.step_frame(1., backend=ref)
        self.assertEqual(s.iterations[0], 10)

    def test_rotated_plane_uses_world_normal(self):
        # Local +Z rotates to world +X.
        q = (0, np.sqrt(.5), 0, np.sqrt(.5))
        s, ref = free_system([[0, 0, 0], [0, 2, 0], [0, 0, 2]], friction=0)
        s.step_frame(1/60, colliders=[shape(5, rotation=q)], backend=ref)
        np.testing.assert_allclose(s.x[:, 0], 1, atol=2e-6)

    def test_local_damping_applies_to_velocity_after_external_acceleration(self):
        s, ref = free_system(gravity_scale=1, local_damping=.5)
        initial = s.x.copy()
        s.step_frame(1/60, backend=ref)
        np.testing.assert_allclose(s.x[:, 2]-initial[:, 2], -980.665/7200, atol=2e-6)

    def test_sampled_moving_collider_transfers_tangential_friction(self):
        s, ref = free_system([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        s.step_frame(1/60, colliders=[shape(5)], backend=ref)
        s.x[:, 2] = 0
        s.v.fill(0)
        s.step_frame(1/60, colliders=[shape(5, location=(1, 0, 0))], backend=ref)
        np.testing.assert_allclose(s.x[:, 0], [.8, 2.8, .8], atol=2e-6)

    def test_sphere_projection_and_coulomb_displacement_friction(self):
        s, ref = free_system()
        s.v[:] = [0, 60, 0]
        s.step_frame(1/60, colliders=[shape()], backend=ref)
        self.assertAlmostEqual(float(np.linalg.norm(s.x[0])), 6, delta=.2)
        # A no-friction copy slides farther during normal projection.
        other, _ = free_system(friction=0)
        other.v[:] = [0, 60, 0]
        other.step_frame(1/60, colliders=[shape()], backend=ref)
        self.assertLess(s.x[0, 1], other.x[0, 1])

    def test_primitive_surface_distances_and_thickness(self):
        cases = [(shape(0), [2, 0, 0], [6, 0, 0]),
                 (shape(2), [2, 0, 1], [6, 0, 1]),
                 (shape(3), [2, 0, 0], [5, 0, 0]),
                 (shape(4), [2, 0, 0], [6, 0, 0]),
                 (shape(5), [2, 0, -2], [2, 0, 1])]
        for collider, point, expected in cases:
            with self.subTest(kind=collider.kind):
                s, ref = free_system([point, np.asarray(point)+[0, 1, 0], np.asarray(point)+[0, 0, 1]], friction=0)
                s.step_frame(1/60, colliders=[collider], backend=ref)
                np.testing.assert_allclose(s.x[0], expected, atol=2e-5)

    def test_per_mesh_collider_ranges_do_not_leak(self):
        s, ref = free_system(friction=0)
        from waifu_physics.cloth.system import System
        two = System([s.meshes[0], copy.deepcopy(s.meshes[0])])
        two.step_frame(1/60, colliders=[[shape()], []], backend=ref)
        self.assertAlmostEqual(float(two.x[0, 0]), 6)
        self.assertAlmostEqual(float(two.x[3, 0]), 2)

    def test_motion_scale_and_teleport_keep_velocity(self):
        s, ref = free_system()
        matrix = np.eye(4)
        matrix[0, 3] = 8
        target = s.full_targets[0]+[8, 0, 0]
        s.step_frame(1/60, targets=[target], motions=[matrix], backend=ref)
        self.assertAlmostEqual(float(s.x[0, 0]), 4)
        s.v[:] = [0, 1, 0]
        old = s.x.copy()
        s.step_frame(1/60, targets=[target+[8, 0, 0]], motions=[matrix], teleport=True, backend=ref)
        np.testing.assert_allclose(s.x[:, 0], old[:, 0]+8)
        np.testing.assert_allclose(s.v[:, 1], 1, atol=2e-5)

    def test_snapshot_restore_replays_next_frame(self):
        s, ref = free_system(gravity_scale=1, substeps=2)
        s.step_frame(1/60, backend=ref)
        state = s.snapshot()
        s.step_frame(1/30, backend=ref)
        expected = s.snapshot()
        s.restore(state)
        s.step_frame(1/30, backend=ref)
        np.testing.assert_array_equal(s.x, expected["x"])
        np.testing.assert_array_equal(s.v, expected["v"])
        midpoint = s.interpolate(state, expected, .5)
        np.testing.assert_allclose(midpoint["x"], (state["x"]+expected["x"])*.5, atol=1e-7)

    def test_native_matches_across_constraints_motion_and_colliders(self):
        from test_cloth_build import ROOT
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "waifu_physics", "cloth", "native.py")),
                        "cloth C backend has not been implemented")
        from waifu_physics.cloth import native
        backend = native.backend()
        self.assertEqual(backend.NAME, "c", native.reason())
        build, Settings, System, ref = api()
        p = np.array([[x*4., 0., -y*4.] for y in range(9) for x in range(11)])
        tri = []
        for y in range(8):
            for x in range(10):
                a = y*11+x
                tri.extend([[a, a+1, a+11], [a+1, a+12, a+11]])
        rgb = np.zeros_like(p)
        rgb[:, 0] = 1
        rgb[:11, 0] = 0
        rgb[11:22, 1] = .2
        cfg = Settings(iterations=4, substeps=2, local_damping=.015,
                       edge_stiffness=.95, bending_stiffness=.8, area_stiffness=.9,
                       tether_stiffness=.8, anim_drive_damping=.5)
        a = System([build(p, tri, rgb, cfg)])
        b = copy.deepcopy(a)
        shapes = [shape(0, location=(20, 1, -16), radius=6),
                  shape(2, location=(5, 0, -30), radius=3),
                  shape(3, location=(34, 0, -30), radius=3, radius1=5),
                  shape(4, location=(20, 0, -38), extent=(30, 20, 2)),
                  shape(5, location=(0, 0, -45))]
        max_x = max_v = 0.
        for frame in range(1, 121):
            target = p.copy()
            target[:, 1] += 2*np.sin(frame*.07)
            angle = .002*np.cos(frame*.1)
            motion = np.eye(4)
            motion[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            kwargs = dict(targets=[target], motions=[motion], colliders=shapes)
            a.step_frame(1/60, backend=ref, **kwargs)
            b.step_frame(1/60, backend=backend, **kwargs)
            max_x = max(max_x, float(np.max(np.abs(a.x-b.x))))
            max_v = max(max_v, float(np.max(np.abs(a.v-b.v))))
        print(f"cloth C/reference 120 frames: max position {max_x:.8g} cm; velocity {max_v:.8g} cm/s")
        self.assertLess(max_x, .02)
        self.assertLess(max_v, 1.)
        self.assertTrue(np.isfinite(b.x).all())


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False).result
    print(f"[{'PASS' if result.wasSuccessful() else 'FAIL'}] cloth collision/native checks ({result.testsRun})")
    sys.exit(0 if result.wasSuccessful() else 1)
