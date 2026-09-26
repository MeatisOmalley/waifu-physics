"""Groups down one chain: a group made from part of a chain ends where the selection ends, a group started
partway down another's chain splits it, and the lower group hangs from the upper one's result, as a second
Kawaii node on the chain does."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Vector

addon = fresh_import()
addon.register()
live = sys.modules["waifu_physics.runtime.live"]
scene = bpy.context.scene
BONES = 6


def chain(name):
    """An anchor and a straight horizontal chain c0..c5 from (0, 0, 2) along X."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    anchor = data.edit_bones.new("anchor")
    anchor.head, anchor.tail = (0.0, 0.0, 1.9), (0.0, 0.0, 2.0)
    parent, head = anchor, Vector((0.0, 0.0, 2.0))
    for i in range(BONES):
        bone = data.edit_bones.new(f"c{i}")
        bone.head, bone.tail = head, head + Vector((0.1, 0.0, 0.0))
        bone.parent = parent
        bone.use_connect = i > 0
        parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="POSE")
    return obj


def new_group(obj, selected):
    for pb in obj.pose.bones:
        pb.select = pb.name in selected
    bpy.ops.waifu_physics.group_new()
    return obj.waifu_physics.groups[-1]


def described(obj):
    return [([r.name for r in g.roots], sorted(b.name for b in g.excluded)) for g in obj.waifu_physics.groups]


def reset():
    scene.waifu_physics.simulate = False
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.frame_set(1)


# --- making groups from part of a chain
obj = chain("parts")
new_group(obj, {"c0", "c1", "c2"})
check("a group made from the top half ends where the selection ends", described(obj) == [(["c0"], ["c3"])],
      described(obj))
new_group(obj, {"c3", "c4", "c5"})
check("the bottom half makes a second group, and the first one stays",
      described(obj) == [(["c0"], ["c3"]), (["c3"], [])], described(obj))
reset()

obj = chain("split")
new_group(obj, {"c0"})
check("one selected bone still makes a group of its whole chain", described(obj) == [(["c0"], [])], described(obj))
new_group(obj, {"c3"})
check("a group started partway down another's chain splits it: the first now ends above it",
      described(obj) == [(["c0"], ["c3"]), (["c3"], [])], described(obj))
new_group(obj, {"c0"})
check("a group over the whole chain again takes both halves over",
      described(obj) == [(["c0"], [])], described(obj))
reset()


# --- the lower group hangs from the upper group's result
def run(with_lower):
    obj = chain("run")
    upper = new_group(obj, {"c0", "c1", "c2"})
    upper.stiffness = 0.0
    if with_lower:
        lower = new_group(obj, {"c3", "c4", "c5"})
        lower.stiffness = 1.0                  # held exactly at its pose: its input, carried by the upper half
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    for frame in range(2, 25):
        scene.frame_set(frame)
    bpy.context.view_layer.update()
    rt = live.runtime(scene)
    axes = {f"c{i}": np.array(obj.pose.bones[f"c{i}"].matrix)[:3, 1] for i in range(BONES)}
    axes = {name: axis / np.linalg.norm(axis) for name, axis in axes.items()}
    heads = {f"c{i}": np.array(obj.pose.bones[f"c{i}"].head) for i in range(3)}
    tiers = rt.tiers
    scene.waifu_physics.simulate = False
    reset()
    return axes, heads, tiers


def angle(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


axes, heads, tiers = run(True)
swing = angle(np.array([1.0, 0.0, 0.0]), axes["c2"])
bend = max(angle(axes[f"c{i}"], axes[f"c{i + 1}"]) for i in (2, 3, 4))
check("the upper half swings under gravity", swing > 20.0, swing)
check("the lower half, held stiff, stays straight in line with the upper half's end as it swings "
      "(its pose is carried with it, as a second Kawaii node's input)", bend < 0.5, bend)
check("nested groups step in two tiers", tiers == 2, tiers)
find_tiers = live.Runtime._find_tiers


def one_tier(self):
    """The lower group's root pinned to the input pose, as before tiers."""
    find_tiers(self)
    self.tiers = 1


live.Runtime._find_tiers = one_tier
flat_axes, _heads, _tiers = run(True)
live.Runtime._find_tiers = find_tiers
flat_bend = max(angle(flat_axes[f"c{i}"], flat_axes[f"c{i + 1}"]) for i in (2, 3, 4))
check("... where, pinned to the input pose, it bends away from the swinging half (so it is simulated)",
      flat_bend > 20.0, flat_bend)
alone_axes, alone_heads, alone_tiers = run(False)
drift = max(np.linalg.norm(heads[name] - alone_heads[name]) for name in heads) * 1000
check("the upper half moves exactly as it does alone: the lower group never feeds back", drift < 1e-6, drift)
check("a scene without nested groups steps once, as before", alone_tiers == 1, alone_tiers)

# --- bones moved and deleted in runs, as the chain manager's bone rows do
def select(obj, names):
    for pb in obj.pose.bones:
        pb.select = pb.name in names


obj = chain("move")
new_group(obj, {"c0"})
select(obj, {"c2", "c3"})
bpy.ops.waifu_physics.chains_to_group(index=-1)
check("bones moved from the middle of a chain make a group of just that run; the chain keeps the rest, "
      "above and below", sorted(described(obj)) == [(["c0", "c4"], ["c2"]), (["c2"], ["c4"])], described(obj))
select(obj, {"c4"})
bpy.ops.waifu_physics.chains_remove()
check("deleting a bone removes it and everything below it", sorted(described(obj)) == [(["c0"], ["c2"]), (["c2"], ["c4"])],
      described(obj))
reset()

obj = chain("nested")
new_group(obj, {"c0", "c1", "c2"})
new_group(obj, {"c3", "c4", "c5"})
select(obj, {"c0", "c1", "c2"})
bpy.ops.waifu_physics.chains_remove()
check("deleting the upper group's chain leaves the group hanging below it", described(obj) == [(["c3"], [])],
      described(obj))
reset()


def ladder(name):
    """Two chains a0..a3 and b0..b3 side by side, one group, linked at depths 1 and 2."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    hips = data.edit_bones.new("hips")
    hips.head, hips.tail = (0.0, 0.0, 1.9), (0.0, 0.0, 2.0)
    for side, x in (("a", -0.1), ("b", 0.1)):
        parent, head = hips, Vector((x, 0.0, 2.0))
        for i in range(4):
            bone = data.edit_bones.new(f"{side}{i}")
            bone.head, bone.tail = head, head + Vector((0.0, 0.0, -0.1))
            bone.parent = parent
            parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="POSE")
    group = new_group(obj, {"a0", "b0"})
    for depth in (1, 2):
        link = group.links.add()
        link.bone_a, link.bone_b = f"a{depth}", f"b{depth}"
    return obj


obj = ladder("ladder")
select(obj, {"a2", "a3", "b2", "b3"})
bpy.ops.waifu_physics.chains_to_group(index=-1)
kept, moved = obj.waifu_physics.groups
pairs = lambda g: sorted((l.bone_a, l.bone_b) for l in g.links)
check("a link wholly among the moved bones goes with them; one wholly among those left stays",
      pairs(kept) == [("a1", "b1")] and pairs(moved) == [("a2", "b2")], (pairs(kept), pairs(moved)))
reset()

addon.unregister()
finish()
