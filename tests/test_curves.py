"""Curves along the chain, Follow Selection and Edit Selected Groups."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Vector

addon = fresh_import()
addon.register()
from waifu_physics.solver.curves import LinearCurve
from waifu_physics.data import curves
live = sys.modules["waifu_physics.runtime.live"]
selection = sys.modules["waifu_physics.ui.selection"]
scene = bpy.context.scene
F32 = np.float32


def rich_curve_eval(times, values, t, default=F32(1.0)):
    """FRichCurve::Eval with linear keys and constant extrapolation, transcribed (RichCurve.cpp:1163)."""
    t = F32(t)
    n = len(times)
    if n == 0:
        return default
    if n < 2 or t < times[0]:
        return values[0]
    if t <= times[n - 1]:
        first, last = 1, n - 1
        count = last - first
        while count > 0:
            step = count // 2
            middle = first + step
            if t >= times[middle]:
                first = middle + 1
                count -= step + 1
            else:
                count = step
        k1, k2 = first - 1, first
        diff = F32(times[k2] - times[k1])
        if diff > 0:
            alpha = F32((t - times[k1]) / diff)
            return F32(values[k1] + alpha * F32(values[k2] - values[k1]))
        return values[k1]
    return values[n - 1]


rng = np.random.default_rng(3)
mismatch = 0
for _ in range(50):
    n = int(rng.integers(2, 12))
    times = np.sort(rng.random(n)).astype(F32)
    values = (rng.random(n) * 2).astype(F32)
    curve = LinearCurve(times, values)
    rates = np.concatenate([rng.random(40), [-0.1, 1.1, times[0], times[-1]]]).astype(F32)
    ours = curve.many(rates)
    reference = np.array([rich_curve_eval(times, values, r) for r in rates], dtype=F32)
    mismatch += int((ours != reference).sum())
check("curve evaluation matches FRichCurve::Eval for linear keys, to the bit", mismatch == 0, mismatch)

# --- a Blender Float Curve host, sampled into keys
data = bpy.data.armatures.new("rig")
obj = bpy.data.objects.new("rig", data)
scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
anchor = data.edit_bones.new("anchor")
anchor.head, anchor.tail = (0, 0, 1.9), (0, 0, 2)
for chain in range(2):
    parent, head = anchor, Vector((chain * 0.5, 0, 2))
    for i in range(4):
        bone = data.edit_bones.new(f"c{chain}_{i}")
        bone.head, bone.tail = head, head + Vector((0, 0, -0.2))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
bpy.ops.object.mode_set(mode="OBJECT")
groups = []
for chain in range(2):
    g = obj.waifu_physics.groups.add()
    g.roots.add().name = f"c{chain}_0"
    groups.append(g)
g = groups[0]
g.use_radius_curve = True
node = curves.node(g, "radius", create=False)
host = bpy.data.node_groups.get(curves.HOST)
check("turning a curve on makes its node in the hidden host", node is not None and host is not None
      and host.name.startswith(".") and host.use_fake_user)
points = node.mapping.curves[0].points
points[0].location = (0.0, 1.0)
points[1].location = (1.0, 0.25)
for p in points:
    p.handle_type = "VECTOR"
node.mapping.update()
sampled = curves.curve(g, "radius")
check("the curve samples as drawn: 1 at the root, 0.25 at the tip",
      abs(sampled(0.0) - 1.0) < 1e-6 and abs(sampled(1.0) - 0.25) < 1e-6 and abs(sampled(0.5) - 0.625) < 1e-3,
      (sampled(0.0), sampled(0.5), sampled(1.0)))

scene.waifu_physics.simulate = True
rt = live.runtime(scene)
s = rt.system
rows = np.flatnonzero(s.group == 0)
order = np.argsort(s.length_rate[rows])
radius = s.radius[rows][order]
check("per-point radius follows the curve along the chain", radius[0] > radius[-1] and
      abs(radius[0] - 3.0) < 1e-4 and abs(radius[-1] - 0.75) < 1e-3, radius)
check("... while a group without a curve is uniform", np.ptp(s.radius[s.group == 1]) == 0.0)
scene.waifu_physics.simulate = False

# --- Follow Selection and Edit Selected Groups
bpy.ops.object.mode_set(mode="POSE")
for pb in obj.pose.bones:
    pb.select = False
data.bones.active = data.bones["c1_2"]
selection._follow()
check("clicking a bone shows its group (Follow Selection)", obj.waifu_physics.active_group == 1, obj.waifu_physics.active_group)
data.bones.active = data.bones["c0_1"]
selection._follow()
check("... and another bone's group when it becomes active", obj.waifu_physics.active_group == 0, obj.waifu_physics.active_group)

for name in ("c0_1", "c1_1"):
    obj.pose.bones[name].select = True
found = selection.groups_of_selected(bpy.context)
check("the selected bones' groups are both found", len(found) == 2, len(found))
groups[0].damping = 0.37
check("a change reaches every selected bone's group",
      abs(groups[1].damping - 0.37) < 1e-6, groups[1].damping)

bpy.ops.object.mode_set(mode="OBJECT")
addon.unregister()
check("unregistering stops the Follow Selection timer", not bpy.app.timers.is_registered(selection._follow))
finish()
