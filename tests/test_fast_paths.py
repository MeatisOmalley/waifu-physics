"""One evaluation a frame: taken-over keys, the before-frame live solve, saving and the lean bake."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Quaternion, Vector

addon = fresh_import()
addon.register()
live = sys.modules["waifu_physics.runtime.live"]
keys = sys.modules["waifu_physics.runtime.keys"]
io = sys.modules["waifu_physics.runtime.io"]
scene = bpy.context.scene


def build(keyed=True):
    scene.waifu_physics.simulate = False
    scene.waifu_physics.use_cache = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    scene.frame_start, scene.frame_end = 1, 40
    data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    head = data.edit_bones.new("head")
    head.head, head.tail = (0, 0, 1.5), (0, 0, 1.7)
    parent, at = head, Vector((0, -0.1, 1.6))
    for i in range(4):
        bone = data.edit_bones.new(f"h{i}")
        bone.head, bone.tail = at, at + Vector((0, -0.02, -0.1))
        bone.parent = parent
        parent, at = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    for frame, x in ((1, 0.0), (20, 0.4), (40, 0.0)):
        rig.location.x = x
        rig.keyframe_insert("location", index=0, frame=frame)
    if keyed:
        pb = rig.pose.bones["h1"]
        pb.rotation_mode = "XYZ"
        pb.keyframe_insert("rotation_euler", frame=1)
        pb.rotation_euler = (0.0, 0.0, 0.6)
        pb.keyframe_insert("rotation_euler", frame=40)
        pb.rotation_euler = (0.0, 0.0, 0.0)
    group = rig.waifu_physics.groups.add()
    group.roots.add().name = "h0"
    return rig


def chain_curves(rig):
    bag = rig.animation_data.action.layers[0].strips[0].channelbag(rig.animation_data.action_slot)
    return [c for c in bag.fcurves if c.data_path.startswith('pose.bones["h')]


check("Fast Evaluation is off by default: live playback reads its input on time",
      not scene.waifu_physics.fast_evaluation)
scene.waifu_physics.fast_evaluation = True

# --- taken-over keys
rig = build()
curves = chain_curves(rig)
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
check("simulating takes the chain's keys over: their curves are muted", curves and all(c.mute for c in curves))
check("... the object's own animation is left alone",
      not any(c.mute for c in rig.animation_data.action.layers[0].strips[0]
              .channelbag(rig.animation_data.action_slot).fcurves if not c.data_path.startswith("pose")))
check("... and the rig can take the one-evaluation path", rt.fast)

# --- live: playing on solves before the frame; a jump takes the exact path
paths = []
for f in range(2, 21):
    scene.frame_set(f)
    paths.append(rt.stepped_ahead == f)
check("playing on, every frame is solved before Blender evaluates it", all(paths), paths)
turn = rig.pose.bones["h1"].rotation_euler.z
scene.frame_set(35)
check("a jump is not solved ahead (it resets from a freshly evaluated pose)", rt.stepped_ahead is None)
check("the keyed bone still turns toward its keys while simulating", turn > 0.1, turn)

# --- Fast Evaluation off: solved after Blender evaluates the frame, on time. With the scene at the simulation's
# rate (one step a frame, no subframes to sample), live playback is then the cache.
def tips(frames, fast=None):
    """The chain's tip, in world space, each frame: live playback, or the cache (fast None)."""
    settings = scene.waifu_physics
    settings.simulate = False
    scene.frame_set(1)
    if fast is None:
        bpy.ops.waifu_physics.cache_toggle()
    else:
        settings.fast_evaluation = fast
        settings.simulate = True
    found, ahead = [], []
    for f in frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()             # the redraw evaluates a write made after the frame
        found.append(np.array(rig.matrix_world @ rig.pose.bones["h3"].tail))
        ahead.append(live.runtime(scene).stepped_ahead == f)
    if fast is None:
        bpy.ops.waifu_physics.cache_toggle()
    settings.simulate = False
    settings.fast_evaluation = True
    return np.array(found), ahead


scene.render.fps = 60
frames = range(1, 41)
cached, _ = tips(frames)
slow, ahead_slow = tips(frames, fast=False)
fast, ahead_fast = tips(frames, fast=True)
check("only keys move this rig, so even with Fast Evaluation off every frame is solved ahead (one evaluation)",
      all(ahead_slow[1:]) and all(ahead_fast[1:]) and live.runtime(scene).exact_ahead)
check("... and at the simulation's rate, live playback matches the cache",
      float(np.abs(slow - cached).max()) < 1e-6, float(np.abs(slow - cached).max()))
check("... and Fast Evaluation too, here: only keys move the rig, so it reads this frame's input (not a frame late)",
      float(np.abs(fast - cached).max()) < 1e-5, float(np.abs(fast - cached).max()))
scene.render.fps = 24
rig = build()
curves = chain_curves(rig)
scene.frame_set(1)
scene.waifu_physics.simulate = True

# --- Steps per Second changed while playing: the clock follows, the chain swings on (no rebuild, no snap to the
# pose), and it keeps its velocity through the change
def swing(frames, change=None):
    """The tip each frame, and the runtime, playing live with Steps per Second changed at a frame: (frame, rate)."""
    settings = scene.waifu_physics
    settings.simulate = False
    settings.target_framerate = 60
    scene.frame_set(1)
    settings.simulate = True
    found = []
    for f in frames:
        if change is not None and f == change[0]:
            before = live.runtime(scene)
            settings.target_framerate = change[1]
        scene.frame_set(f)
        bpy.context.view_layer.update()
        found.append(np.array(rig.matrix_world @ rig.pose.bones["h3"].tail))
    return np.array(found), (before if change else None)


frames = range(1, 31)
steady, _ = swing(frames)
changed, before = swing(frames, (15, 61))
after = live.runtime(scene)
check("changing Steps per Second while playing keeps the running simulation (no rebuild)",
      after is before and after.system.target_framerate == 61)
frame_motion = float(np.linalg.norm(steady[14] - steady[13]))
check("... so the chain swings on: the same up to the change, and at it within half a frame's motion (a step landing "
      "either side of the frame; the old rebuild snapped it back to the pose)",
      float(np.abs(changed[:14] - steady[:14]).max()) == 0.0
      and float(np.linalg.norm(changed[14] - steady[14])) < 0.5 * frame_motion,
      (float(np.linalg.norm(changed[14] - steady[14])), frame_motion))
doubled, _ = swing(frames, (15, 120))
jumps = np.linalg.norm(np.diff(doubled, axis=0), axis=1)
check("... and doubling it keeps the chain's velocity: no jump at the change",
      jumps[13] < 2.0 * max(jumps[11], jumps[12], 1e-4), (jumps[11], jumps[12], jumps[13]))
# --- the frame range changed while playing: where playback loops, not the physics: the same run swings on
def ranged(frames, changes):
    """The tip each frame, playing live, the frame range changed at frames: {frame: (start, end)}."""
    settings = scene.waifu_physics
    settings.simulate = False
    scene.frame_start, scene.frame_end = 1, 40
    scene.frame_set(1)
    settings.simulate = True
    first, found = live.runtime(scene), []
    for f in frames:
        if f in changes:
            scene.frame_start, scene.frame_end = changes[f]
        scene.frame_set(f)
        bpy.context.view_layer.update()
        found.append(np.array(rig.matrix_world @ rig.pose.bones["h3"].tail))
    return np.array(found), live.runtime(scene) is first


plain, _ = ranged(range(1, 31), {})
moved, same = ranged(range(1, 31), {10: (1, 60), 18: (5, 60)})
check("changing the frame range while playing keeps the running simulation, and changes nothing about it",
      same and float(np.abs(moved - plain).max()) == 0.0, float(np.abs(moved - plain).max()))
scene.frame_start, scene.frame_end = 1, 40
scene.waifu_physics.simulate = False
scene.waifu_physics.target_framerate = 60
scene.frame_set(1)
scene.waifu_physics.simulate = True

# --- Fast Evaluation reads this frame's input where keys alone move it: a keyframed parent swinging the chain,
# a keyframed armature object, and a collider on a keyframed bone the chain hits. Then it is exact.
colliders = sys.modules["waifu_physics.data.colliders"]


def swinger(constrain=False):
    scene.waifu_physics.simulate = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    data = bpy.data.armatures.new("swinger")
    obj = bpy.data.objects.new("swinger", data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    base = data.edit_bones.new("base"); base.head, base.tail = (0, 0, 0), (0, 0, 1.0)
    arm = data.edit_bones.new("arm"); arm.head, arm.tail = (0, 0, 0.5), (0.6, 0, 0.5); arm.parent = base
    parent, at = base, Vector((0, 0, 1.0))
    for i in range(3):
        bone = data.edit_bones.new(f"s{i}"); bone.head, bone.tail = at, at + Vector((0, 0.05, 0.5)); bone.parent = parent
        parent, at = bone, bone.tail.copy()
    target = data.edit_bones.new("target"); target.head, target.tail = (1, 1, 0), (1, 1, 0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    base_bone, arm_bone = obj.pose.bones["base"], obj.pose.bones["arm"]
    for f, turn, lift, x in ((1, 0.0, 0.0, 0.0), (10, 1.6, 0.8, 0.5), (20, -1.4, -0.6, -0.3), (30, 0.4, 1.2, 0.2), (40, 0.0, 0.0, 0.0)):
        base_bone.rotation_mode = "XYZ"
        base_bone.rotation_euler = (turn, 0.0, 0.3 * turn)
        base_bone.keyframe_insert("rotation_euler", frame=f)
        arm_bone.rotation_quaternion = Quaternion((0, 1, 0), lift)
        arm_bone.keyframe_insert("rotation_quaternion", frame=f)
        obj.location.x = x
        obj.keyframe_insert("location", index=0, frame=f)
    if constrain:
        c = base_bone.constraints.new("DAMPED_TRACK")
        c.target, c.subtarget = obj, "target"
    group = obj.waifu_physics.groups.add()
    group.roots.add().name = "s0"
    group.radius = 0.05
    collider = colliders.add(obj, "arm", "Sphere")
    colliders.set_value(collider, "Radius", 0.8)              # overlaps the chain's lower bones as the arm swings
    scene.frame_start, scene.frame_end = 1, 40
    return obj


def played(obj, fast=None):
    """The chain's tips each frame in world space: live playback (fast on or off), or the cache (None)."""
    settings = scene.waifu_physics
    settings.simulate = False
    scene.frame_set(1)
    if fast is None:
        bpy.ops.waifu_physics.cache_toggle()
    else:
        settings.fast_evaluation = fast
        settings.simulate = True
    tips, ahead = [], []
    for f in range(1, 41):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        tips.append([np.array(obj.matrix_world @ obj.pose.bones[f"s{i}"].tail) for i in range(3)])
        ahead.append(live.runtime(scene).stepped_ahead == f)
    rt = live.runtime(scene)
    if fast is None:
        bpy.ops.waifu_physics.cache_toggle()
    settings.simulate = False
    return np.array(tips), ahead, rt


scene.render.fps = 60
obj = swinger()
slow, _, _ = played(obj, fast=False)
fast, ahead, rt = played(obj, fast=True)
cached, _, _ = played(obj)
rig = rt.rigs[0]
check("a keyframed parent, collider bone and armature are read for this frame, not a frame late",
      {rig.names[i] for i in rig.input_keys.bones} == {"base", "arm"} and rig.input_keys.object_ok)
check("... so Fast Evaluation, solving before every frame, plays as the slow path and the cache do",
      all(ahead[1:]) and float(np.abs(fast - slow).max()) < 1e-4 and float(np.abs(fast - cached).max()) < 1e-4,
      (float(np.abs(fast - slow).max()), float(np.abs(fast - cached).max())))
free = swinger()
for collider in colliders.all_of(free):
    collider.waifu_physics_collider.enabled = False
check("... with the chain really hitting the collider on the swinging bone",
      float(np.abs(slow - played(free, fast=False)[0]).max()) > 0.05, float(np.abs(slow - played(free, fast=False)[0]).max()))
obj = swinger(constrain=True)
fast, ahead, rt = played(obj, fast=True)
slow, ahead_slow, rt_slow = played(obj, fast=False)
rig = rt.rigs[0]
check("a constrained parent is left to Blender's last evaluation (and so is what hangs below it)",
      {rig.names[i] for i in rig.input_keys.bones} == set() and all(ahead[1:]),
      [rig.names[i] for i in rig.input_keys.bones])
check("... so solving ahead is not exact: Fast Evaluation off evaluates twice there, on accepts the lag",
      not rt_slow.exact_ahead and not any(ahead_slow) and float(np.abs(fast - slow).max()) > 1e-3,
      float(np.abs(fast - slow).max()))

# --- whatever else moves makes it inexact: an animated scene collider; a constraint added while playing
obj = swinger()
ground = colliders.add_to_scene("Sphere")
ground.location.z = 3.0
ground.keyframe_insert("location", frame=1)
_, ahead_slow, rt_slow = played(obj, fast=False)
check("an animated scene collider is read a frame late ahead, so off evaluates twice", not rt_slow.exact_ahead
      and not any(ahead_slow))
bpy.data.objects.remove(ground)
obj = swinger()
scene.waifu_physics.fast_evaluation = False
scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
check("... and with nothing moving but keys, exact", live.runtime(scene).exact_ahead)
c = obj.pose.bones["base"].constraints.new("DAMPED_TRACK")
c.target, c.subtarget = obj, "target"
bpy.context.view_layer.update()
scene.frame_set(3)
scene.frame_set(4)
check("a constraint added while playing is noticed: the simulation is rebuilt, and no longer solved ahead",
      not live.runtime(scene).exact_ahead and live.runtime(scene).stepped_ahead != 4)
scene.waifu_physics.simulate = False

# --- a cache draws its own frames' points, not the simulation's last state
draw = sys.modules["waifu_physics.ui.draw"]
obj = swinger()
scene.frame_set(1)
bpy.ops.waifu_physics.cache_toggle()
off = []
for f in (7, 19, 31):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    spheres = draw._simulated_spheres(scene)
    heads = [obj.matrix_world @ obj.pose.bones[n].head for n in ("s1", "s2")]
    off.append(max(min((centre - head).length for centre, _r, _c in spheres) for head in heads))
check("with a cache, the collision spheres are drawn where the cached frame put the chain", max(off) < 1e-4, off)
bpy.ops.waifu_physics.cache_toggle()
scene.waifu_physics.simulate = False
ik = obj.pose.bones["s2"].constraints.new("IK")
ik.target, ik.subtarget, ik.chain_count = obj, "target", 0
for c in list(obj.pose.bones["base"].constraints):
    obj.pose.bones["base"].constraints.remove(c)
rig = io.Rig(obj)
rig.set_chain([rig.index["s1"]])
rig.prepare_input([rig.index["arm"]])
check("an IK constraint that reaches up the chain keeps the bones it reaches out",
      {rig.names[i] for i in rig.input_keys.bones} == set(), [rig.names[i] for i in rig.input_keys.bones])
obj.pose.bones["s2"].constraints.remove(ik)
obj = swinger()
scene.frame_set(1)
scene.waifu_physics.fast_evaluation = True
scene.waifu_physics.simulate = True
scene.frame_set(2)
first = live.runtime(scene)
obj.pose.bones["base"].keyframe_insert("location", frame=5)
bpy.context.view_layer.update()
scene.frame_set(3)
check("keys added to a bone the chain hangs from are found again", live.runtime(scene) is not first)
scene.waifu_physics.simulate = False
scene.render.fps = 24
rig = build()
curves = chain_curves(rig)
scene.frame_set(1)
scene.waifu_physics.simulate = True

# --- saving: the file gets the keys unmuted, the session keeps them taken over
path = os.path.join(tempfile.gettempdir(), "waifu_physics_fast_paths.blend")
bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
check("after saving, the keys are taken over again", all(c.mute for c in curves))
scene.waifu_physics.simulate = False
check("stopping the simulation hands the keys back", not any(c.mute for c in curves)
      and keys.MARK not in rig.keys())

# --- a crash while simulating: the next load unmutes what Waifu Physics had muted
rig = build()
scene.frame_set(1)
scene.waifu_physics.simulate = True
curves = chain_curves(rig)
check("the armature remembers what Waifu Physics muted", keys.MARK in rig.keys())
keys.recover(bpy.data.objects)           # what load_post does with a file saved mid-simulation
check("recovery unmutes those curves and forgets them", not any(c.mute for c in curves) and keys.MARK not in rig.keys())
scene.waifu_physics.simulate = False

# --- an NLA-animated chain bone cannot be taken over: the slower, exact path
rig = build(keyed=False)
action = bpy.data.actions.new("nla")
rig.animation_data_create()
track = rig.animation_data.nla_tracks.new()
pb = rig.pose.bones["h2"]
pb.rotation_mode = "XYZ"
pb.keyframe_insert("rotation_euler", frame=1)
nla_action = rig.animation_data.action
rig.animation_data.action = None
track.strips.new("nla", 1, nla_action)
rig.animation_data.action = bpy.data.actions.new("own")
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
check("a chain channel an NLA strip animates keeps the exact path", not rt.fast and rt.needs_post_replay())
scene.waifu_physics.simulate = False

# --- the lean bake: same result, and every object comes back
rig = build()
extra = bpy.data.objects.new("mesh", bpy.data.meshes.new("mesh"))
scene.collection.objects.link(extra)
scene.frame_set(1)
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
needed = live._essentials(scene, rt)
check("the bake evaluates the rig and skips unrelated meshes", "rig" in needed and "mesh" not in needed)
bpy.ops.waifu_physics.cache_all()
check("Cache All fills the range", live.runtime(scene).cached_range() == (1, 40))
check("... and unhides what it hid", not extra.hide_viewport)
lean = {f: live.runtime(scene).cache[f].channels[0]["rotation_euler"].copy() for f in (10, 30)}
with_mesh = live.lean_evaluation
live.lean_evaluation = type("NoLean", (), {"__init__": lambda self, *a: None, "__enter__": lambda self: self,
                                           "__exit__": lambda self, *a: False})
bpy.ops.waifu_physics.cache_all()
live.lean_evaluation = with_mesh
full = {f: live.runtime(scene).cache[f].channels[0]["rotation_euler"] for f in (10, 30)}
check("... with the same result as evaluating everything", all(np.array_equal(lean[f], full[f]) for f in lean))
scene.waifu_physics.simulate = False

addon.unregister()
finish()
