"""Keyframed physics settings in live playback: sampled for the frame on the one-evaluation path, so they take effect
on the frame they are keyed for, as in the cache, not a frame late."""
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
keys = sys.modules["waifu_physics.runtime.keys"]
scene = bpy.context.scene
scene.render.fps = 60                 # one step a frame: live playback and the cache then agree exactly
scene.frame_start, scene.frame_end = 1, 40


def rig_with_keyed_settings(driven=False):
    """A chain on a swinging parent, its group's settings keyed to change sharply: stiffness (a float), gravity
    (one item of a vector), Use Scene Gravity (a boolean), and a Basic force's direction (a vector) and space (an
    enum)."""
    scene.waifu_physics.simulate = False
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    data = bpy.data.armatures.new("keyed")
    obj = bpy.data.objects.new("keyed", data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    base = data.edit_bones.new("base")
    base.head, base.tail = (0, 0, 0), (0, 0, 1.0)
    parent, at = base, Vector((0, 0, 1.0))
    for i in range(3):
        bone = data.edit_bones.new(f"s{i}")
        bone.head, bone.tail, bone.parent = at, at + Vector((0.05, 0, 0.3)), parent
        parent, at = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    swing = obj.pose.bones["base"]
    swing.rotation_mode = "XYZ"
    for f, turn in ((1, 0.0), (12, 1.0), (24, -0.8), (40, 0.0)):
        swing.rotation_euler = (turn, 0.0, 0.0)
        swing.keyframe_insert("rotation_euler", frame=f)
    group = obj.waifu_physics.groups.add()
    group.roots.add().name = "s0"
    group.use_scene_gravity = False
    for f, stiffness, gz, scene_gravity in ((1, 0.0, -1.0, False), (9, 0.0, -1.0, False), (10, 0.6, 1.0, True),
                                            (25, 0.6, 1.0, True), (26, 0.05, -2.0, False)):
        group.stiffness = stiffness
        group.gravity = (0.0, 0.0, gz)
        group.use_scene_gravity = scene_gravity
        group.keyframe_insert("stiffness", frame=f)
        group.keyframe_insert("gravity", index=2, frame=f)
        group.keyframe_insert("use_scene_gravity", frame=f)
    force = group.forces.add()
    force.kind = "BASIC"
    for f, x, space in ((1, 0.0, "WORLD"), (14, 0.0, "WORLD"), (15, 3.0, "COMPONENT"), (30, -3.0, "WORLD")):
        force.direction = (x, 0.0, 0.0)
        force.space = space
        force.keyframe_insert("direction", frame=f)
        force.keyframe_insert("space", frame=f)
    if driven:
        driver = group.driver_add("damping").driver
        driver.expression = "0.2"
    return obj


def played(obj, fast=None):
    """The chain's tips each frame: live playback (Fast Evaluation on or off), or the cache (fast None)."""
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


obj = rig_with_keyed_settings()
cached, _, _ = played(obj)
automatic, ahead_auto, rt = played(obj, fast=False)
fast, ahead_fast, _ = played(obj, fast=True)
check("keyed settings are found: the group's, and its force's", len(rt.rigs[0].setting_keys.curves) == 7
      and not rt.rigs[0].setting_keys.blocked, len(rt.rigs[0].setting_keys.curves))
check("keys alone move this rig and its settings, so even with Fast Evaluation off each frame is solved ahead",
      rt.exact_ahead and all(ahead_auto[1:]) and all(ahead_fast[1:]))
check("... and live playback, sampling the settings for the frame, plays as the cache does",
      float(np.abs(automatic - cached).max()) < 1e-5 and float(np.abs(fast - cached).max()) < 1e-5,
      (float(np.abs(automatic - cached).max()), float(np.abs(fast - cached).max())))

plain_sample = keys.SettingKeys.sample
keys.SettingKeys.sample = lambda self, frame: {}              # as before: the properties' values, a frame late
late, _, _ = played(obj, fast=True)
keys.SettingKeys.sample = plain_sample
check("without sampling, the same run is off (the settings a frame late): the check above can tell",
      float(np.abs(late - cached).max()) > 1e-3, float(np.abs(late - cached).max()))

frame_values = {obj.waifu_physics.groups[0].path_from_id(): {"stiffness": {0: 0.25}, "use_scene_gravity": {0: 0.9},
                                                             "gravity": {2: 7.0}},
                obj.waifu_physics.groups[0].forces[0].path_from_id(): {"space": {0: 1.0}}}
view = keys.Sampled(obj.waifu_physics.groups[0], frame_values)
real = obj.waifu_physics.groups[0]
check("a sampled setting reads as Blender would set it: floats as they are, booleans true only at 1, vector items "
      "alone, enums by number, and unkeyed ones from the property",
      abs(view.stiffness - 0.25) < 1e-7 and view.use_scene_gravity is False and list(view.gravity)[:2] == list(real.gravity)[:2]
      and view.gravity[2] == 7.0 and view.damping == real.damping and view.forces[0].space
      == next(i.identifier for i in real.forces[0].bl_rna.properties["space"].enum_items if i.value == 1),
      (view.stiffness, view.use_scene_gravity, list(view.gravity), view.forces[0].space))

obj = rig_with_keyed_settings(driven=True)
_, ahead_auto, rt = played(obj, fast=False)
check("a driven setting is known only after evaluation: then Fast Evaluation off evaluates first (not ahead)",
      rt.rigs[0].setting_keys.blocked and not rt.exact_ahead and not any(ahead_auto))

addon.unregister()
finish()
