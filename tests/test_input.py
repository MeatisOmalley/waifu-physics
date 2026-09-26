"""The solver's input is Blender's evaluated pose with last frame's physics cleared first:
constraints and drivers reach it, and its own output never feeds back."""
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
from waifu_physics.data import links

scene = bpy.context.scene


def armature(name, hair):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    root = data.edit_bones.new("Root")
    root.head, root.tail = (0, 0, 0), (0, 0, 0.1)
    head = data.edit_bones.new("Head")
    head.head, head.tail = (0, 0, 1.5), (0, 0, 1.7)
    head.parent = root
    if hair:
        parent, at = head, Vector((0, -0.1, 1.6))
        for i in range(4):
            bone = data.edit_bones.new(f"hair{i}")
            bone.head, bone.tail = at, at + Vector((0, -0.02, -0.1))
            bone.parent = parent
            parent, at = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def scene_with(roots, walk=True):
    """A VRoid-style pair: `vroid` copies `base` bone for bone; base walks 0.3 m over frames 1-10."""
    scene.waifu_physics.simulate = False
    scene.waifu_physics.use_cache = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.frame_start, scene.frame_end = 1, 30
    base, vroid = armature("base", False), armature("vroid", True)
    for bone in ("Root", "Head"):
        constraint = vroid.pose.bones[bone].constraints.new("COPY_TRANSFORMS")
        constraint.target, constraint.subtarget = base, bone
    if walk:
        pb = base.pose.bones["Root"]
        for frame, y in ((1, 0.0), (10, 0.3), (30, 0.3)):
            pb.location = (0.0, y, 0.0)                  # Root's local Y is world Z: up, then hold
            pb.keyframe_insert("location", frame=frame)
    group = vroid.waifu_physics.groups.add()
    for name in roots:
        group.roots.add().name = name
    return base, vroid, group


def point_head(rt, name):
    return rt.system.frame_pose[rt.point_of[(0, name)]] / rt.cm


# --- a constrained root reaches the solver (Ctrl+A made Root the group's root)
base, vroid, group = scene_with(["Root"])
scene.frame_set(1)
scene.waifu_physics.simulate = True
shown = {}
for f in range(2, 11):
    scene.frame_set(f)
    shown[f] = np.array(base.pose.bones["Root"].head) + np.array(base.matrix_world.translation)
rt = live.runtime(scene)
check("a root driven by a constraint moves the solver's input with it, on time (Fast Evaluation off)",
      np.allclose(point_head(rt, "Root"), shown[10], atol=1e-5), (point_head(rt, "Root"), shown[10]))
scene.waifu_physics.simulate = False
scene.frame_set(1)
scene.waifu_physics.fast_evaluation = True
scene.waifu_physics.simulate = True
for f in range(2, 11):
    scene.frame_set(f)
rt = live.runtime(scene)
check("... and a frame late with Fast Evaluation on: a constraint's result is only known after evaluation",
      np.allclose(point_head(rt, "Root"), shown[9], atol=1e-5), (point_head(rt, "Root"), shown[9]))
scene.waifu_physics.fast_evaluation = False
check("... and the warning names the constrained bones in the group",
      sorted(links.constrained_bones(vroid, group)) == ["Head", "Root"], links.constrained_bones(vroid, group))

# --- the right setup: hair under a constrained head trails the walk
base, vroid, group = scene_with(["hair0"])
check("a group started below the constrained bones has no warning", links.constrained_bones(vroid, group) == [])
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
rest_offset = np.array(vroid.pose.bones["hair3"].tail) - np.array(vroid.pose.bones["hair0"].head)
lag = []
for f in range(2, 12):
    scene.frame_set(f)
    hair0 = np.array(vroid.pose.bones["hair0"].head)
    lag.append((np.array(vroid.pose.bones["hair3"].tail) - hair0)[2] - rest_offset[2])
check("the hair root follows the constrained head: 0.3 m up by frame 10, seen a frame later live",
      np.allclose(point_head(rt, "hair0")[2], 1.6 + 0.3, atol=1e-5), point_head(rt, "hair0"))
check("... and the hair trails it while it rises", min(lag) < -0.005, min(lag))

# --- no feedback: a still, unkeyed chain reads the same input every frame while it swings
base, vroid, group = scene_with(["hair0"], walk=False)
group.gravity = (3.0, 0.0, -1.0)
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
first = rt.system.frame_pose.copy()
for f in range(2, 31):
    scene.frame_set(f)
moved = float(np.abs(np.array(vroid.pose.bones["hair3"].tail) - (np.array(vroid.pose.bones["hair0"].head)
                                                                  + rest_offset)).max())
check("the chain swings", moved > 0.02, moved)
check("... but its input stays the rest pose: last frame's physics never feeds back",
      np.allclose(rt.system.frame_pose, first, atol=1e-6), float(np.abs(rt.system.frame_pose - first).max()))

# --- keyed chain channels are still the input for their bones
base, vroid, group = scene_with(["hair0"], walk=False)
pb = vroid.pose.bones["hair1"]
pb.rotation_mode = "XYZ"
pb.rotation_euler = (0.0, 0.0, 0.0)
pb.keyframe_insert("rotation_euler", frame=1)
pb.rotation_euler = (0.0, 0.0, 1.2)
pb.keyframe_insert("rotation_euler", frame=20)
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
for f in range(2, 21):
    scene.frame_set(f)
hair1_rest = vroid.data.bones["hair1"].matrix_local.to_quaternion()
from mathutils import Quaternion
q = rt.system.frame_pose_rot[rt.point_of[(0, "hair1")]]
input_rot = Quaternion((q[3], q[0], q[1], q[2]))
turn = (hair1_rest.inverted() @ input_rot).to_euler("XYZ")
check("a keyed rotation reaches the solver as its keyed value, not the simulated one",
      abs(turn.z - 1.2) < 1e-4, tuple(turn))

# --- Cache All bakes from a clean input too
base, vroid, group = scene_with(["hair0"], walk=False)
group.gravity = (3.0, 0.0, -1.0)
scene.frame_set(1)
scene.waifu_physics.simulate = True
first = live.runtime(scene).system.frame_pose.copy()
check("Cache All runs", bpy.ops.waifu_physics.cache_all() == {"FINISHED"})
rt = live.runtime(scene)
check("Cache All fills the range", rt.cached_range() == (1, 30), rt.cached_range())
check("... from the rest input at every step", np.allclose(rt.system.frame_pose, first, atol=1e-6),
      float(np.abs(rt.system.frame_pose - first).max()))
scene.frame_set(30)
baked = np.array(vroid.pose.bones["hair3"].tail)
swing = float(np.abs(baked - (np.array(vroid.pose.bones["hair0"].head) + rest_offset)).max())
check("... and plays back swung", swing > 0.02, (swing, live.runtime(scene).cached_range()))

scene.waifu_physics.simulate = False
addon.unregister()
finish()
