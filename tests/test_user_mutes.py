"""Curves the user mutes on chain bones: the simulation ignores them (their channels at rest, as if unkeyed), leaves
them muted, and follows the user's mutes when they change while it simulates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Quaternion, Vector

addon = fresh_import()
addon.register()
live = sys.modules["waifu_physics.runtime.live"]
scene = bpy.context.scene


def build(keyed=True):
    scene.waifu_physics.simulate = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    anchor = data.edit_bones.new("anchor")
    anchor.head, anchor.tail = (0, 0, 1.6), (0, 0, 2.0)
    parent, head = anchor, Vector((0, 0, 2.0))
    for i in range(4):
        bone = data.edit_bones.new(f"c{i}")
        bone.head, bone.tail = head, head + Vector((0.15, 0, 0))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.waifu_physics.groups.add().roots.add().name = "c0"
    if keyed:
        bone = rig.pose.bones["c1"]
        bone.rotation_quaternion = Quaternion((1, 0, 0), 0.6)
        bone.keyframe_insert("rotation_quaternion", frame=1)
        bone.rotation_quaternion = Quaternion((1, 0, 0), -0.6)
        bone.keyframe_insert("rotation_quaternion", frame=30)
        bone.rotation_quaternion = (1, 0, 0, 0)
    return rig


def curves(rig):
    return list(rig.animation_data.action.layers[0].strips[0].channelbag(rig.animation_data.action_slot).fcurves)


def play(frames):
    tips = []
    for frame in frames:
        scene.frame_set(frame)
        rt = live.runtime(scene)
        tips.append(rt.system.loc[rt.system.parent >= 0][-1].copy())
    return np.array(tips)


def simulate(rig, frames=range(1, 50)):
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    tips = play(frames)
    scene.waifu_physics.simulate = False
    return tips


unkeyed = simulate(build(keyed=False))
followed = simulate(build())

# --- muted before simulating: ignored, and left muted
rig = build()
for curve in curves(rig):
    curve.mute = True
ignored = simulate(rig)
check("a chain curve the user muted is ignored: the chain moves as if it had no keys",
      np.abs(ignored - unkeyed).max() < 1e-6, np.abs(ignored - unkeyed).max())
check("... (keys that are not muted are followed)", np.abs(followed - unkeyed).max() > 1.0)
check("... and it stays muted when simulation stops", all(curve.mute for curve in curves(rig)))

# --- unmuted while simulating: taken over and followed from then on
rig = build()
for curve in curves(rig):
    curve.mute = True
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 10))
for curve in curves(rig):
    curve.mute = False
bpy.context.view_layer.update()
play(range(10, 12))
keys = live.runtime(scene).rigs[0].keys
check("a curve the user unmutes while simulating is taken over (muted by Waifu Physics, and sampled)",
      len(keys.curves) == 4 and keys.muted and all(curve.mute for curve in curves(rig)), len(keys.curves))
scene.waifu_physics.simulate = False
check("... and is left unmuted when simulation stops", not any(curve.mute for curve in curves(rig)))

# --- one of our own muted curves unmuted while simulating: taken over again, never applied over the physics
rig = build()
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 5))
curves(rig)[0].mute = False
bpy.context.view_layer.update()
play(range(5, 7))
keys = live.runtime(scene).rigs[0].keys
check("unmuting a curve Waifu Physics muted to sample only has it taken over again",
      keys.muted and all(curve.mute for curve in curves(rig)) and len(keys.curves) == 4)
scene.waifu_physics.simulate = False
check("... and every curve is unmuted when simulation stops", not any(curve.mute for curve in curves(rig)))


# --- keys added to a second bone while simulating, then the running simulation lost (an undo does that): its
# muted curves must not be mistaken for the user's, or the new keys are ignored for good (the user's report)
keys = sys.modules["waifu_physics.runtime.keys"]
rig = build()
scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
bone = rig.pose.bones["c2"]
bone.rotation_quaternion = Quaternion((0, 0, 1), 0.8)
bone.keyframe_insert("rotation_quaternion", frame=1)
bone.keyframe_insert("rotation_quaternion", frame=30)
bpy.context.view_layer.update()
scene.frame_set(3)
second = [c for c in curves(rig) if 'c2"]' in c.data_path]
check("keys added to another chain bone while simulating are taken over", second and all(c.mute for c in second))
live._undone(scene)                                        # the run lost, its curves still muted
scene.frame_set(4)
rt = live.runtime(scene)
rig_state = rt.rigs[0]
check("a new run takes a lost run's muted curves as its own, not the user's",
      not rig_state.keys.user_muted and rig_state.keyed["rotation"][rig_state.index["c2"]],
      sorted(rig_state.keys.user_muted))
scene.waifu_physics.simulate = False
check("... and Simulate off unmutes them all and forgets them",
      not any(c.mute for c in curves(rig)) and keys.MARK not in rig.keys())

scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
live._undone(scene)                                        # lost again, and then simulation switched off
scene.waifu_physics.simulate = False
check("Simulate off cleans up after a lost run too, with no run left to do it",
      not any(c.mute for c in curves(rig)) and keys.MARK not in rig.keys())

# --- bones added while simulating: the old run cannot use its armature, but still hands its curves back
scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
extra = rig.data.edit_bones.new("extra")
extra.head, extra.tail = (1, 1, 1), (1, 1, 1.2)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
scene.frame_set(3)
scene.waifu_physics.simulate = False
check("bones added while simulating leave no curve muted", not any(c.mute for c in curves(rig))
      and keys.MARK not in rig.keys())

# --- a real undo, where the undo system runs here
try:
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    scene.frame_set(2)
    bpy.ops.ed.undo_push(message="simulating")
    rig.pose.bones["c3"].keyframe_insert("rotation_quaternion", frame=5)
    bpy.ops.ed.undo_push(message="keyed")
    scene.frame_set(3)
    bpy.ops.ed.undo()
    undone = True
except RuntimeError as error:
    undone = False
    print("undo unavailable here:", error)
if undone:
    rig = bpy.data.objects["rig"]
    scene = bpy.context.scene
    scene.frame_set(4)
    scene.waifu_physics.simulate = False
    check("after a real undo, Simulate off leaves no curve muted", not any(c.mute for c in curves(rig)))

# --- a curve the user muted is still theirs
rig = build()
user = next(c for c in curves(rig) if c.array_index == 1)
user.mute = True
scene.frame_set(1)
scene.waifu_physics.simulate = True
scene.frame_set(2)
scene.waifu_physics.simulate = False
check("a curve the user muted stays muted through all this", user.mute)


# --- F-curves are held by path and checked before use: a curve deleted, the action swapped, or the animation
# cleared, each straight before a frame (before the depsgraph handler can rebuild), is followed, never crashes
def playing(rig):
    scene.waifu_physics.simulate = False
    scene.frame_set(1)
    scene.waifu_physics.fast_evaluation = True
    scene.waifu_physics.simulate = True
    scene.frame_set(2)


rig = build()
bone = rig.pose.bones["c2"]
bone.rotation_quaternion = Quaternion((0, 0, 1), 0.8)
bone.keyframe_insert("rotation_quaternion", frame=1)
bone.keyframe_insert("rotation_quaternion", frame=30)
bone.rotation_quaternion = (1, 0, 0, 0)
playing(rig)
bag = rig.animation_data.action.layers[0].strips[0].channelbag(rig.animation_data.action_slot)
gone = next(c for c in bag.fcurves if 'c1"]' in c.data_path and c.array_index == 1)
bag.fcurves.remove(gone)
scene.frame_set(3)
held = live.runtime(scene).rigs[0].keys.curves
check("a chain curve deleted right before a frame is skipped, the rest still sampled (no stale reference used)",
      len(held.resolved()) == len(held) - 1 or live.runtime(scene).rigs[0].keys.total == len(held))
scene.frame_set(4)
check("... and the run is rebuilt without it", not any(p[0].endswith('c1"].rotation_quaternion') and p[1] == 1
                                                        for p in live.runtime(scene).rigs[0].keys.curves.paths))
playing(rig)
copy = rig.animation_data.action.copy()
for layer in copy.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                curve.mute = False
rig.animation_data.action = copy
rig.animation_data.action_slot = copy.slots[0]
scene.frame_set(3)
scene.frame_set(4)
owned = [c for c in curves(rig) if (c.data_path, c.array_index) in live.runtime(scene).rigs[0].keys.curves.paths]
check("the action swapped for a copy (as many curves) is followed: its curves are the ones taken over",
      owned and all(c.mute for c in owned) and rig.animation_data.action == copy)
playing(rig)
rig.animation_data_clear()
scene.frame_set(3)
scene.frame_set(4)
check("the animation cleared right before a frame: no stale animation data used, and the run rebuilt without keys",
      len(live.runtime(scene).rigs[0].keys.curves) == 0 and rig.animation_data is None)
scene.waifu_physics.simulate = False

addon.unregister()
finish()
