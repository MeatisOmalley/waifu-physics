"""Live clock and canonical Cache All at common animation frame rates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Vector

addon = fresh_import()
addon.register()
scene = bpy.context.scene
from waifu_physics.runtime import live


def make_rig(fps, keyed_child=False):
    scene.waifu_physics.simulate = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, fps + 1
    scene.frame_set(1)
    data = bpy.data.armatures.new("rate rig")
    rig = bpy.data.objects.new("rate rig", data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    anchor = data.edit_bones.new("anchor")
    anchor.head, anchor.tail = (0, 0, 1.9), (0, 0, 2)
    child = data.edit_bones.new("child")
    child.head, child.tail = anchor.tail, Vector(anchor.tail) + Vector((0.2, 0, 0))
    child.parent = anchor
    bpy.ops.object.mode_set(mode="OBJECT")
    group = rig.waifu_physics.groups.add()
    group.roots.add().name = "child"
    group.damping = 0.1
    anchor_pose = rig.pose.bones["anchor"]
    anchor_pose.rotation_mode = "XYZ"
    anchor_pose.rotation_euler.z = 0.0
    anchor_pose.keyframe_insert("rotation_euler", frame=1)
    anchor_pose.rotation_euler.z = 0.5
    anchor_pose.keyframe_insert("rotation_euler", frame=fps + 1)
    for layer in rig.animation_data.action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(rig.animation_data.action_slot)
            if bag is not None:
                for fc in bag.fcurves:
                    for key in fc.keyframe_points:
                        key.interpolation = "LINEAR"
    if keyed_child:
        child_pose = rig.pose.bones["child"]
        child_pose.rotation_mode = "XYZ"
        child_pose.rotation_euler.z = 0.0
        child_pose.keyframe_insert("rotation_euler", frame=1)
        child_pose.rotation_euler.z = 0.5
        child_pose.keyframe_insert("rotation_euler", frame=fps + 1)
    return rig


positions = {}
early = {}
for fps in (12, 24, 30, 60, 120):
    rig = make_rig(fps)
    scene.waifu_physics.use_cache = True
    scene.waifu_physics.simulate = True
    bpy.ops.waifu_physics.cache_all()
    rt = live.runtime(scene)
    bpy.context.view_layer.update()
    check(f"{fps} fps cache covers the range", rt.cached_range() == (1, fps + 1), rt.cached_range())
    samples = []
    for frame in (1 + fps // 2, 1 + fps):
        scene.frame_set(frame)
        samples.append(np.array(rig.pose.bones["child"].tail))
    positions[fps] = samples
    if fps == 24:
        bpy.ops.waifu_physics.cache_all()
        repeated = []
        for frame in (1 + fps // 2, 1 + fps):
            scene.frame_set(frame)
            repeated.append(np.array(rig.pose.bones["child"].tail))
        check("rebuilding the same cache reproduces its output",
              all(np.array_equal(a, b) for a, b in zip(samples, repeated)),
              [a - b for a, b in zip(samples, repeated)])
    if fps in (24, 120):
        scene.frame_set(2 if fps == 24 else 6)
        early[fps] = np.array(rig.pose.bones["child"].tail)

reference = positions[60]
for fps, samples in positions.items():
    difference = max(np.linalg.norm(a - b) for a, b in zip(samples, reference))
    check(f"{fps} fps baked motion matches 60 fps at equal times",
          difference < 1e-4, difference)
check("interpolated 120 fps frame matches 24 fps at 1/24 second",
      np.allclose(early[24], early[120], atol=1e-4), early[24] - early[120])

scene.render.fps = 24
scene.frame_set(2)
check("changing scene FPS outdates the old bake (kept until baked again)",
      live.runtime(scene).cache_mode == "canonical" and live.runtime(scene).outdated == "the frame range or rate changed",
      live.runtime(scene).outdated)

rig = make_rig(24, keyed_child=True)
scene.waifu_physics.use_cache = True
scene.waifu_physics.simulate = True
bpy.ops.waifu_physics.cache_all()
rt = live.runtime(scene)
check("a keyed simulated bone is taken over, so its cache replays before evaluation alone",
      not rt.needs_post_replay() and rt.rigs[0].keys.muted)
scene.frame_set(12)
index = rt.rigs[0].index["child"]
expected = rt.cache[12].channels[0]["rotation_euler"].reshape(-1, 3)[index]
actual = np.array(rig.pose.bones["child"].rotation_euler)
check("post-frame replay restores the keyed bone's cached channel",
      np.allclose(actual, expected, atol=1e-5), (actual, expected))

for fps, expected in ((12, 5), (24, 2), (30, 2), (60, 1), (120, 0)):
    rig = make_rig(fps)
    scene.waifu_physics.use_cache = False
    scene.waifu_physics.simulate = True
    rt = live.runtime(scene)
    calls = [0]
    original = rt.system._substep

    def counted(backend):
        calls[0] += 1
        return original(backend)

    rt.system._substep = counted
    scene.frame_set(2)
    check(f"{fps} fps live frame does not lose normal elapsed time",
          calls[0] == expected, calls[0])
    check(f"{fps} fps live keeps the {rt.system.target_framerate} Hz step whatever the frame rate",
          rt.system.target_framerate == scene.waifu_physics.target_framerate)
    if fps == 120:
        shown = [np.array(rig.pose.bones["child"].tail) for _ in [0]]
        scene.frame_set(3)
        shown.append(np.array(rig.pose.bones["child"].tail))
        check("at 120 fps two frames make one 60 Hz step", calls[0] == 1, calls[0])
        scene.frame_set(4)
        shown.append(np.array(rig.pose.bones["child"].tail))
        check("... and the chains move on every frame, shown between steps",
              all(not np.array_equal(a, b) for a, b in zip(shown, shown[1:])), shown)

scene.waifu_physics.simulate = False
addon.unregister()
finish()
