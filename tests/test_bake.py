"""Bake: the cached simulation written as keyframes into a copy of the armature's action."""
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
scene.frame_start, scene.frame_end = 1, 30

data = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
anchor = data.edit_bones.new("anchor")
anchor.head, anchor.tail = (0, 0, 1.9), (0, 0, 2.0)
parent, head = anchor, Vector((0, 0, 2.0))
for i in range(3):
    bone = data.edit_bones.new(f"c{i}")
    bone.head, bone.tail, bone.parent = head, head + Vector((0.2, 0, 0)), parent
    parent, head = bone, bone.tail.copy()
bpy.ops.object.mode_set(mode="OBJECT")
rig.pose.bones["c1"].rotation_mode = "XYZ"              # one Euler bone among the quaternion ones
for frame, x in ((1, 0.0), (30, 0.3)):                  # the armature's own animation, kept by the bake
    rig.pose.bones["anchor"].location = (x, 0, 0)
    rig.pose.bones["anchor"].keyframe_insert("location", frame=frame)
original = rig.animation_data.action
group = rig.waifu_physics.groups.add()
group.roots.add().name = "c0"
group.damping = 0.05


def tips():
    return np.array(rig.pose.bones["c2"].tail)


scene.frame_set(1)
scene.waifu_physics.simulate = True
check("Bake is greyed out until the simulation is cached", not bpy.ops.waifu_physics.bake.poll())
bpy.ops.waifu_physics.cache_toggle()
check("... and available once it is", bpy.ops.waifu_physics.bake.poll())
cached = {}
for frame in range(1, 31):
    scene.frame_set(frame)
    cached[frame] = tips()
check("the cached chain swings", np.linalg.norm(cached[30] - cached[1]) > 0.05)

check("Bake runs", bpy.ops.waifu_physics.bake() == {"FINISHED"})
action = rig.animation_data.action
check("the armature now plays a copy of its action", action is not original
      and action.name == f"{original.name} Baked", action.name)
check("... and the original is kept, with a fake user", original.name in bpy.data.actions and original.use_fake_user)
check("Simulate and the cache are off, so the keys play",
      not scene.waifu_physics.simulate and not scene.waifu_physics.use_cache)
bag = next(bag for layer in action.layers for strip in layer.strips
           for bag in [strip.channelbag(rig.animation_data.action_slot)] if bag is not None)
paths = {(c.data_path, c.array_index) for c in bag.fcurves}
check("each quaternion bone gets its four rotation curves",
      all(('pose.bones["c0"].rotation_quaternion', i) in paths for i in range(4)))
check("... the Euler bone its three Euler curves, not quaternion ones",
      all(('pose.bones["c1"].rotation_euler', i) in paths for i in range(3))
      and ('pose.bones["c1"].rotation_quaternion', 0) not in paths)
check("... and channels the simulation did not move are not keyed", ('pose.bones["c0"].location', 0) not in paths
      and ('pose.bones["c0"].scale', 0) not in paths)
check("the armature's own animation is untouched", ('pose.bones["anchor"].location', 0) in paths)
curve = bag.fcurves.find('pose.bones["c0"].rotation_quaternion', index=0)
check("one linear key per cached frame", len(curve.keyframe_points) == 30
      and all(p.interpolation == "LINEAR" for p in curve.keyframe_points))
worst = 0.0
for frame in range(1, 31):
    scene.frame_set(frame)
    worst = max(worst, float(np.linalg.norm(tips() - cached[frame])))
check("played back without simulating, the keys match the cache", worst < 1e-4, worst)

addon.unregister()
finish()
