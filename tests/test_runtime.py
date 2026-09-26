"""Live simulation on real armatures: input pose, stepping, writing back, resets."""
import math
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


def armature(name, direction=(1.0, 0.0, 0.0), count=3, length=0.2):
    """An anchor bone and a chain of `count` bones from (0, 0, 2) along `direction`."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    anchor = data.edit_bones.new("anchor")
    anchor.head, anchor.tail = (0.0, 0.0, 1.9), (0.0, 0.0, 2.0)
    parent, head = anchor, Vector((0.0, 0.0, 2.0))
    for i in range(count):
        bone = data.edit_bones.new(f"c{i}")
        bone.head = head
        bone.tail = head + Vector(direction) * length
        bone.parent = parent
        bone.use_connect = i > 0
        parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def group(obj, **values):
    g = obj.waifu_physics.groups.add()
    g.roots.add().name = "c0"
    for key, value in values.items():
        setattr(g, key, value)
    return g


def heads(obj):
    return np.array([obj.pose.bones[f"c{i}"].head for i in range(3)] + [obj.pose.bones["c2"].tail])


def play(start, end):
    for frame in range(start, end + 1):
        scene.frame_set(frame)


def reset_scene():
    scene.waifu_physics.simulate = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.frame_start = 1
    scene.frame_set(1)


# --- a chain falls, keeps its lengths, and Blender's bones sit exactly on the solver's points
reset_scene()
rig = armature("fall")
group(rig, damping=0.05, stiffness=0.0)
scene.waifu_physics.simulate = True
start = heads(rig)
play(1, 40)
now = heads(rig)
lengths = np.linalg.norm(np.diff(now, axis=0), axis=1)
check("a horizontal chain falls under gravity", now[-1, 2] < start[-1, 2] - 0.2, (start[-1], now[-1]))
check("... its root stays put", np.allclose(now[0], start[0], atol=1e-6), now[0])
check("... every bone keeps its length", np.allclose(lengths, 0.2, atol=1e-5), lengths)
rt = live.runtime(scene)
s = rt.system
points = s.loc[rt.real] / rt.cm
bones = [rig.pose.bones[rt.rigs[0].names[b]].head for b in rt.bone_of_point]
check("Blender's bone heads sit on the solver's points", np.allclose(points, np.array(bones), atol=1e-5),
      np.abs(points - np.array(bones)).max())
check("the rotations were written as the bones' own mode (quaternion)",
      all(rig.pose.bones[f"c{i}"].rotation_mode == "QUATERNION" for i in range(3)))

# --- turning simulation off puts unkeyed chain bones back at rest
scene.waifu_physics.simulate = False
bpy.context.view_layer.update()
check("turning Simulate off restores the rest pose", np.allclose(heads(rig), start, atol=1e-6), heads(rig))

# --- stiffness 1 holds an unkeyed chain at rest; a keyed chain follows its animation
reset_scene()
rig = armature("held")
group(rig, stiffness=1.0)
scene.waifu_physics.simulate = True
rest = heads(rig)
play(1, 20)
check("stiffness 1 holds an unkeyed chain at its rest against gravity", np.allclose(heads(rig), rest, atol=1e-5),
      heads(rig))

reset_scene()
rig = armature("keyed")
g = group(rig, stiffness=1.0)
pb = rig.pose.bones["c1"]
pb.rotation_mode = "XYZ"
pb.rotation_euler = (0.0, 0.0, 0.0)
pb.keyframe_insert("rotation_euler", frame=1)
pb.rotation_euler = (0.0, 0.0, 1.2)
pb.keyframe_insert("rotation_euler", frame=20)
scene.frame_set(20)
animated = heads(rig)              # the animated pose, before any simulation
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(1, 20)
check("a keyed chain bone is read as input: stiffness 1 follows the animation",
      np.allclose(heads(rig), animated, atol=1e-4), (heads(rig), animated))
check("... and keeps its Euler rotation mode", pb.rotation_mode == "XYZ")

# --- every rotation mode survives, and the result is the same whatever the modes
tails = {}
for modes in (("QUATERNION", "QUATERNION", "QUATERNION"), ("XYZ", "AXIS_ANGLE", "ZXY")):
    reset_scene()
    rig = armature("modes")
    group(rig, damping=0.1)
    for i, mode in enumerate(modes):
        rig.pose.bones[f"c{i}"].rotation_mode = mode
    scene.waifu_physics.simulate = True
    play(1, 25)
    tails[modes] = heads(rig)[-1].copy()
    check(f"rotation modes {modes} are kept", tuple(rig.pose.bones[f"c{i}"].rotation_mode for i in range(3)) == modes)
    scene.waifu_physics.simulate = False
quat, mixed = tails.values()
check("... and the chain moves the same in any mode", np.allclose(quat, mixed, atol=1e-5), (quat, mixed))

# --- keyframed damping takes effect as it plays
runs = {}
for keyed in (False, True):
    reset_scene()
    rig = armature("damped")
    g = group(rig, damping=0.0)
    if keyed:
        g.damping = 0.0
        rig.keyframe_insert('waifu_physics.groups[0].damping', frame=1)
        g.damping = 0.9
        rig.keyframe_insert('waifu_physics.groups[0].damping', frame=15)
    scene.waifu_physics.simulate = True
    play(1, 40)
    runs[keyed] = heads(rig)[-1].copy()
check("keyframed damping changes the motion as it plays", not np.allclose(runs[False], runs[True], atol=1e-3), runs)

# --- world damping 0: moving the armature leaves the chain trailing; a teleport does not fling it
reset_scene()
rig = armature("moving", direction=(0.0, 0.0, -1.0))
group(rig, stiffness=0.0, world_damping_location=0.0, world_damping_rotation=0.0, damping=0.2)
scene.waifu_physics.simulate = True
play(1, 5)
for frame in range(6, 12):
    rig.location.x += 0.05
    scene.frame_set(frame)
trail = heads(rig)[-1, 0]
check("with world damping 0, a chain trails behind its moving armature", trail < -0.01, trail)
before = heads(rig)[-1].copy()
rig.location.x += 20.0
scene.frame_set(12)
after = heads(rig)[-1]
check("a 20 m jump is a teleport: the chain keeps its shape", abs(after[0] - before[0]) < 0.05, (before, after))

# --- jumping back resets to the pose
reset_scene()
rig = armature("jump")
group(rig)
scene.waifu_physics.simulate = True
rest = heads(rig)
play(1, 30)
scene.frame_set(10)
check("jumping backwards starts over from the pose", np.allclose(heads(rig), rest, atol=1e-6), heads(rig))

scene.waifu_physics.simulate = False
# --- under a parent with uneven scale (VRoid hair under a slider-scaled head), the bones land on the simulation
scene.waifu_physics.simulate = False
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)
skew_data = bpy.data.armatures.new("skew")
skew = bpy.data.objects.new("skew", skew_data)
scene.collection.objects.link(skew)
bpy.context.view_layer.objects.active = skew
bpy.ops.object.mode_set(mode="EDIT")
head_bone = skew_data.edit_bones.new("head")
head_bone.head, head_bone.tail = (0, 0, 1.5), (0, 0, 1.7)
parent, at = head_bone, Vector((0.05, 0.0, 1.6))
for i in range(5):
    bone = skew_data.edit_bones.new(f"strand{i}")
    bone.head, bone.tail = at, at + Vector((0.06, 0.02, -0.07))
    bone.parent = parent
    parent, at = bone, bone.tail.copy()
bpy.ops.object.mode_set(mode="OBJECT")
skew.pose.bones["head"].scale = (1.32, 1.28, 1.32)
skew.waifu_physics.groups.add().roots.add().name = "strand0"
scene.frame_set(1)
scene.waifu_physics.simulate = True
for frame in range(2, 21):
    scene.frame_set(frame)
bpy.context.view_layer.update()
rt = live.runtime(scene)
s = rt.system
off = [np.linalg.norm(np.array(skew.pose.bones[s.bone_names[i]].head) - s.loc[i] / rt.cm) * 1000
       for i in range(len(s.loc)) if s.bone_names[i]]
check("under a parent scaled unevenly, every bone's head lands on its simulated point (was 22 cm off at a tip)",
      max(off) < 0.5, max(off))
scene.waifu_physics.simulate = False

# --- the skew guard only runs where a frame really is distorted: Aligned puts the scale on each bone's own
# axes after its frame, so there frames are square and even (float noise aside) and the guard stays off
io = sys.modules["waifu_physics.runtime.io"]
for bone in skew.data.bones:
    if bone.name != "head":
        bone.inherit_scale = "ALIGNED"
guard_calls = []
plain_aimed = io._aimed
io._aimed = lambda *a: (guard_calls.append(len(a[0])), plain_aimed(*a))[1]
scene.frame_set(1)
scene.waifu_physics.simulate = True
for frame in range(2, 21):
    scene.frame_set(frame)
bpy.context.view_layer.update()
rt = live.runtime(scene)
s = rt.system
off = [np.linalg.norm(np.array(skew.pose.bones[s.bone_names[i]].head) - s.loc[i] / rt.cm) * 1000
       for i in range(len(s.loc)) if s.bone_names[i]]
check("under Aligned the heads land on their points too, with the skew guard never running",
      max(off) < 0.5 and not guard_calls, (max(off), sum(guard_calls)))
io._aimed = plain_aimed
scene.waifu_physics.simulate = False
for bone in skew.data.bones:
    bone.inherit_scale = "FULL"
turn = io.unscaled(np.array(skew.pose.bones["head"].matrix)[None, :3, :3])[0]      # a rotation
noise = np.random.default_rng(3).normal(scale=2e-6, size=(3, 3))       # single-precision pose noise
distorted, skewed = io._distortion(np.array([turn * 1.7, turn + noise, turn * (1.001, 1.0, 1.001),
                                             turn @ [[1, 0.2, 0], [0, 1, 0], [0, 0, 1]], turn * (1, 1, -1)]))
check("scaled evenly or off by float noise, a frame is neither; unevenly (even by 0.1%) but square only needs "
      "aiming; skewed or mirrored needs both",
      list(distorted) == [False, False, True, True, True] and list(skewed) == [False, False, False, True, True],
      (list(distorted), list(skewed)))

# --- every Inherit Scale, Inherit Rotation and Local Location: the pose predicted is the pose Blender evaluates
io = sys.modules["waifu_physics.runtime.io"]
MODES = ("FULL", "FIX_SHEAR", "ALIGNED", "AVERAGE", "NONE", "NONE_LEGACY")
strand = [f"strand{i}" for i in range(5)]
rng = np.random.default_rng(7)
skew.pose.bones["head"].rotation_mode = "XYZ"
skew.pose.bones["head"].rotation_euler = (0.3, -0.2, 0.5)
for name in strand:
    pb = skew.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = rng.uniform(-0.6, 0.6, 3)
    pb.location = rng.uniform(-0.03, 0.03, 3)
    pb.scale = rng.uniform(0.8, 1.25, 3)


def predicted_error(settings):
    """settings: (inherit scale, inherit rotation, local location) per strand bone. The largest distance (mm)
    between an axis end or head Blender evaluates and the one the rig predicts from the parent's pose."""
    for name, (mode, rotation, local) in zip(strand, settings):
        bone = skew.data.bones[name]
        bone.inherit_scale, bone.use_inherit_rotation, bone.use_local_location = mode, rotation, local
    bpy.context.view_layer.update()
    rig = io.Rig(skew)
    evaluated = np.array([np.array(pb.matrix) for pb in skew.pose.bones])
    worst = 0.0
    for name in strand:
        i = rig.index[name]
        level = np.array([i])
        basis = np.array(skew.pose.bones[name].matrix_basis)[None]
        made = io._placed(*rig._parent_transforms(level, evaluated[rig.parents[level]]), basis)[0]
        worst = max(worst, np.abs(made - evaluated[i])[:3].max() * 1000)
    return worst


combos = [(mode, rotation, local) for mode in MODES for rotation in (True, False) for local in (True, False)]
errors = {combo: predicted_error([combo] * 5) for combo in combos}
worst = max(errors, key=errors.get)
check("each Inherit Scale, with Inherit Rotation and Local Location on or off, is predicted as Blender evaluates it",
      errors[worst] < 0.01, (worst, errors[worst]))
mixed = [combos[int(k)] for k in rng.integers(0, len(combos), 5)]
check("a chain mixing them too", predicted_error(mixed) < 0.01, mixed)

# --- ... and under each, simulated bones land on their points and point where the solver turned them
def y_axis(name):
    """The bone's Y axis, from the pose matrix that deforms the mesh."""
    axis = np.array(skew.pose.bones[name].matrix)[:3, 1]
    return axis / np.linalg.norm(axis)


written = {}
plain_write = io.Rig.write


def recording_write(self, bones, rotation, location=None, move_location=None):
    """The rotations the last write was given, by bone name."""
    written.update(zip((self.names[i] for i in bones), io.matrices_from_quats(rotation)))
    return plain_write(self, bones, rotation, location, move_location)


io.Rig.write = recording_write
for name in strand:
    pb = skew.pose.bones[name]
    pb.rotation_euler, pb.location, pb.scale = (0, 0, 0), (0, 0, 0), (1, 1, 1)
skew.pose.bones["head"].rotation_euler = (0, 0, 0)
for mode, rotation, local in [("FULL", True, True), ("ALIGNED", True, True), ("FIX_SHEAR", True, True),
                              ("AVERAGE", True, True), ("NONE", True, True), ("NONE_LEGACY", True, True),
                              ("FULL", False, True), ("ALIGNED", True, False)]:
    for name in strand:
        bone = skew.data.bones[name]
        bone.inherit_scale, bone.use_inherit_rotation, bone.use_local_location = mode, rotation, local
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    for frame in range(2, 21):
        scene.frame_set(frame)
    bpy.context.view_layer.update()
    rt = live.runtime(scene)
    s = rt.system
    point = {s.bone_names[i]: s.loc[i] / rt.cm for i in range(len(s.loc)) if s.bone_names[i]}
    off = max(np.linalg.norm(np.array(skew.pose.bones[name].head) - point[name]) * 1000 for name in strand)
    aim = max(np.degrees(np.arctan2(np.linalg.norm(np.cross(y_axis(name), written[name][:, 1])),
                                    np.dot(y_axis(name), written[name][:, 1]))) for name in strand)
    check(f"Inherit Scale {mode}, Inherit Rotation {rotation}, Local Location {local}: heads land on the "
          f"simulation and bones point where it turned them", off < 0.5 and aim < 0.001, (off, aim))
    scene.waifu_physics.simulate = False

scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
before = live.runtime(scene)
skew.data.bones["strand2"].inherit_scale = "FULL"
bpy.context.view_layer.update()
scene.frame_set(3)
check("changing a chain bone's Inherit Scale rebuilds the rig it was read into",
      live.runtime(scene) is not before and live.runtime(scene).rigs[0].inherit_scale[
          live.runtime(scene).rigs[0].index["strand2"]] == io.FULL)
scene.waifu_physics.simulate = False
io.Rig.write = plain_write

# --- a chain's tip point is its last bone's tail, per chain: Blender knows each bone's length (Kawaii's
# DummyBoneLength is only its stand-in), so the last bone swings like the others
reset_scene()
rig = armature("tips", direction=(0.0, 0.0, -1.0))
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
rig.data.edit_bones["c2"].tail = rig.data.edit_bones["c2"].head + Vector((0.0, 0.0, -0.35))   # a longer last bone
second = rig.data.edit_bones.new("d0")
second.head, second.tail = (0.3, 0.0, 2.0), (0.3, 0.0, 1.9)
second.parent = rig.data.edit_bones["anchor"]
bpy.ops.object.mode_set(mode="OBJECT")
g = group(rig)
g.roots.add().name = "d0"
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
s = rt.system
KIND_TIP = sys.modules["waifu_physics.solver.system"].KIND_TIP
tips = np.flatnonzero(s.kind == KIND_TIP)
tails = {name: np.array(rig.data.bones[name].tail_local) for name in ("c2", "d0")}
found = {s.bone_names[s.parent[i]]: s.loc[i] / rt.cm for i in tips}
check("each chain's tip point sits on its own last bone's tail (35 cm on one chain, 10 cm on the other)",
      set(found) == {"c2", "d0"} and all(np.allclose(found[n], tails[n], atol=1e-5) for n in found),
      {n: (found[n], tails[n]) for n in found})
anchor_bone = rig.pose.bones["anchor"]
anchor_bone.rotation_mode = "XYZ"
for frame, turn in ((1, 0.0), (8, 1.2), (20, 1.2)):
    anchor_bone.rotation_euler = (0.0, turn, 0.0)
    anchor_bone.keyframe_insert("rotation_euler", frame=frame)
scene.waifu_physics.simulate = False
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(2, 10)
last = rig.pose.bones["c2"].matrix.to_3x3().col[1].normalized()
before = rig.pose.bones["c1"].matrix.to_3x3().col[1].normalized()
check("... so the last bone swings with the chain instead of keeping its animated rotation",
      last.angle(before) > 0.02 or last.angle(anchor_bone.matrix.to_3x3().col[1]) > 0.02, last.angle(before))
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
rig.data.edit_bones["d0"].tail = (0.3, 0.0, 1.7)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
scene.frame_set(11)
rt = live.runtime(scene)
d0_tip = next(i for i in np.flatnonzero(rt.system.kind == KIND_TIP) if rt.system.bone_names[rt.system.parent[i]] == "d0")
check("lengthening a bone rebuilds the simulation, its tip moving to the new tail",
      rt.system.tip_length[d0_tip] / rt.cm > 0.29, rt.system.tip_length[d0_tip] / rt.cm)
scene.waifu_physics.simulate = False

addon.unregister()
check("unregistering removes the frame handler", live._frame_changed not in bpy.app.handlers.frame_change_post)
finish()
