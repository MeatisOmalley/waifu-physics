"""Blender-independent cloth topology, binding and analytical physics checks."""
import importlib
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "waifu_physics" not in sys.modules:
    package = types.ModuleType("waifu_physics")
    package.__path__ = [os.path.join(ROOT, "waifu_physics")]
    sys.modules["waifu_physics"] = package


def api():
    if not os.path.isfile(os.path.join(ROOT, "waifu_physics", "cloth", "build.py")):
        raise AssertionError("cloth mesh builder has not been implemented")
    from waifu_physics.cloth.build import build, Settings
    from waifu_physics.cloth.system import System
    from waifu_physics.cloth import step_numpy
    return build, Settings, System, step_numpy


class ClothBuildTests(unittest.TestCase):
    def test_constraints_preserve_triangle_insertion_before_coloring(self):
        build, _, _, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]]
        m = build(p, [[1, 3, 2], [0, 1, 2]])
        # First input edge must remain first even when its index is not smallest.
        np.testing.assert_array_equal(m.edges[0], [1, 3])

    def test_density_mass_anchors_and_constraints(self):
        build, Settings, System, _ = api()
        p = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [100, 100, 0]], float)
        c = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        m = build(p, [[0, 1, 2], [1, 3, 2]], c)
        self.assertEqual(len(m.positions), 4)
        self.assertEqual(len(m.edges), 5)
        self.assertEqual(len(m.bending), 1)
        self.assertEqual(len(m.area), 2)
        self.assertAlmostEqual(float(m.mass.sum()), .35, places=6)
        self.assertEqual(m.inv_mass[0], 0)
        self.assertEqual(len(m.tethers), 3)
        self.assertTrue(np.all(m.max_distance[1:] == 40))
        self.assertEqual(System([m]).n, 4)

    def test_welding_retains_full_render_vertex_mapping(self):
        build, _, System, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [.0004, 0, 0]]
        m = build(p, [[0, 1, 2], [3, 1, 2]])
        self.assertEqual(len(m.positions), 3)
        s = System([m])
        s.x[0, 2] = 3
        out = s.render_positions(0)
        np.testing.assert_array_equal(out[0], out[3])

    def test_game_weld_uses_rounded_cells_and_maximum_seam_channels(self):
        build, _, _, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [.0004, 0, 0]]
        c = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, .7, 0]]
        m = build(p, [[0, 1, 2], [3, 1, 2]], c)
        self.assertGreater(m.inv_mass[0], 0)
        self.assertEqual(m.max_distance[0], 40)
        self.assertAlmostEqual(float(m.anim_weight[0]), .7)
        # Nearby points in different rounded cells are intentionally distinct.
        p[0][0], p[3][0] = .00049, .00051
        m = build(p, [[0, 1, 2], [3, 1, 2]], c)
        self.assertEqual(len(m.positions), 4)

    def test_rider_barycentric_normal_offset_follows_deformed_triangle(self):
        build, _, System, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 3, 2]]
        c = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1]]
        m = build(p, [[0, 1, 2]], c)
        self.assertEqual(len(m.positions), 3)
        s = System([m])
        s.x[:, 2] += 7
        np.testing.assert_allclose(s.render_positions(0)[3], [2, 3, 9], atol=1e-6)

    def test_rider_outside_triangle_retains_plane_projection_offset(self):
        build, _, System, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [12, 0, 2]]
        c = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1]]
        s = System([build(p, [[0, 1, 2]], c)])
        np.testing.assert_allclose(s.render_positions(0)[3], p[3], atol=1e-6)
        # Unbounded barycentrics also preserve the rider's affine extrapolation.
        s.x[1, 0] = 20
        np.testing.assert_allclose(s.render_positions(0)[3], [24, 0, 2], atol=1e-6)

    def test_disconnected_fixed_section_passes_through_animation(self):
        build, _, System, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [30, 0, 0], [40, 0, 0], [30, 10, 0]]
        c = [[1, 0, 0]] * 3 + [[0, 0, 0]] * 3
        m = build(p, [[0, 1, 2], [3, 4, 5]], c)
        self.assertEqual(len(m.positions), 3)
        s = System([m])
        target = np.asarray(p, float) + [0, 0, 10]
        s.reset([target])
        np.testing.assert_allclose(s.render_positions(0)[3:], target[3:])

    def test_degenerate_faces_and_empty_mesh_are_safe(self):
        build, _, System, ref = api()
        m = build([[0, 0, 0], [10, 0, 0], [0, 10, 0]], [[0, 0, 1], [0, 1, 2]])
        self.assertEqual(len(m.triangles), 1)
        s = System([build([], [])])
        s.step_frame(1 / 60, backend=ref)
        self.assertEqual(s.render_positions(0).shape, (0, 3))

    def test_geodesic_tether_keeps_path_length_not_beeline(self):
        build, _, _, _ = api()
        p = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [0, 20, 0], [10, 20, 0]]
        c = [[0, 0, 0]] + [[1, 0, 0]] * 5
        m = build(p, [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]], c)
        i = np.flatnonzero(m.tethers[:, 1] == 5)[0]
        self.assertAlmostEqual(float(m.tether_length[i]), 30., places=5)

    def test_bad_indices_and_nonfinite_positions_rejected(self):
        build, _, _, _ = api()
        with self.assertRaises(ValueError):
            build([[0, 0, 0]], [[0, 1, 2]])
        with self.assertRaises(ValueError):
            build([[float("nan"), 0, 0]], [])

    def test_global_damping_uses_integrated_predictor_and_displacement_velocity(self):
        build, Settings, System, ref = api()
        cfg = Settings(edge_stiffness=0, bending_stiffness=0, area_stiffness=0,
                       tether_stiffness=0, gravity_scale=0, damping=.01, max_distance=1e6)
        s = System([build([[0, 0, 0], [10, 0, 0], [0, 10, 0]], [[0, 1, 2]], settings=cfg)])
        s.v[:] = [10, 0, 0]
        dt = 1 / 60
        length = (np.exp(np.log(.99) * 60 * dt) - 1) / (np.log(.99) * 60)
        s.step_frame(dt, backend=ref)
        self.assertAlmostEqual(float(s.x[0, 0]), 10 * length, delta=2e-6)
        self.assertAlmostEqual(float(s.v[0, 0]), 10 * length / dt, delta=1e-4)

    def test_gravity_anchor_maxdistance_and_reset(self):
        build, Settings, System, ref = api()
        cfg = Settings(edge_stiffness=0, bending_stiffness=0, area_stiffness=0,
                       tether_stiffness=0, damping=0)
        p = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
        m = build(p, [[0, 1, 2]], [[0, 0, 0], [.005, 0, 0], [1, 0, 0]], cfg)
        s = System([m])
        s.step_frame(1 / 60, backend=ref)
        np.testing.assert_array_equal(s.x[0], p[0])
        self.assertAlmostEqual(float(s.x[1, 2]), -.2, places=6)
        self.assertAlmostEqual(float(s.x[2, 2]), -980.665 / 3600, places=5)
        s.reset([p + [0, 0, 20]])
        np.testing.assert_allclose(s.x, p + [0, 0, 20])
        self.assertTrue(np.all(s.v == 0))

    def test_anim_drive_uses_quantized_exponential_fit(self):
        build, Settings, System, ref = api()
        cfg = Settings(edge_stiffness=0, bending_stiffness=0, area_stiffness=0,
                       tether_stiffness=0, damping=0, gravity_scale=0, anim_drive_damping=0)
        p = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
        s = System([build(p, [[0, 1, 2]], [[1, .5, 0]] * 3, cfg)])
        s.x[:, 2] = 10
        k = 1 - (1 - (1000 ** (7 / 15) - 1) / 999) ** 2
        s.step_frame(1 / 60, backend=ref)
        np.testing.assert_allclose(s.x[:, 2], 10 * (1-k), atol=2e-5)


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False).result
    print(f"[{'PASS' if result.wasSuccessful() else 'FAIL'}] cloth topology/analytical checks ({result.testsRun})")
    sys.exit(0 if result.wasSuccessful() else 1)
