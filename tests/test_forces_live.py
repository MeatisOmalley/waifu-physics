"""Forces, wind fields and sync bones in Blender: operators, playback, the cache and saved setups."""
import json
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
from waifu_physics.data import curves, serialize

scene = bpy.context.scene


def build(with_thigh=False):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.waifu_physics.simulate = False
    scene.waifu_physics.use_cache = False
    scene.frame_start, scene.frame_end = 1, 60
    data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    hips = data.edit_bones.new("hips")
    hips.head, hips.tail = (0, 0, 1.0), (0, 0, 1.1)
    thigh = data.edit_bones.new("thigh")
    thigh.head, thigh.tail = (0.1, 0.0, 1.0), (0.1, 0.0, 0.55)
    thigh.parent = hips
    parent, head = hips, Vector((0.0, -0.15, 1.0))
    for i in range(4):
        bone = data.edit_bones.new(f"s{i}")
        bone.head, bone.tail = head, head + Vector((0.0, -0.02, -0.12))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="POSE")
    for pb in rig.pose.bones:
        pb.select = pb.name == "s0"
    bpy.ops.waifu_physics.group_new()
    group = rig.waifu_physics.groups[0]
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig, group


def tip(rig):
    return np.array(rig.pose.bones["s3"].tail)


def play(frames):
    for f in frames:
        scene.frame_set(f)


rig, group = build()
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
rest = tip(rig)
scene.waifu_physics.simulate = False

rig, group = build()
bpy.context.view_layer.objects.active = rig
check("Add Force makes a Basic force", bpy.ops.waifu_physics.force_add(kind="BASIC") == {"FINISHED"}
      and group.forces[0].kind == "BASIC")
group.forces[0].space = "WORLD"
group.forces[0].direction = (0.6, 0.0, 0.0)
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
check("a Basic force pushes the chain in the viewport", tip(rig)[0] - rest[0] > 0.03, tip(rig) - rest)
scene.waifu_physics.simulate = False

# --- a wind force field blows a Wind force
rig, group = build()
bpy.context.view_layer.objects.active = rig
bpy.ops.waifu_physics.force_add(kind="WIND")
bpy.ops.object.effector_add(type="WIND", location=(0.0, 0.0, 1.0), rotation=(-math.pi / 2, 0.0, 0.0))
field = bpy.context.object
field.field.strength = 60.0                    # Unreal's wind speed: centimetres a second here
direction, speed = live.scene_wind(scene, 100.0)
check("a wind field's direction is its local Z and its speed its strength",
      np.allclose(direction, (0.0, 1.0, 0.0), atol=1e-6) and math.isclose(speed, 60.0), (direction, speed))
bpy.context.view_layer.objects.active = rig
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
check("the Wind force blows the chain along the field", tip(rig)[1] - rest[1] > 0.03, tip(rig) - rest)
scene.waifu_physics.simulate = False

# --- Add Wind Field keeps the armature the active object
rig, group = build()
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
check("Add Wind Field adds a field", bpy.ops.waifu_physics.wind_field_add() == {"FINISHED"}
      and any(o.field and o.field.type == "WIND" for o in scene.objects))
check("... and leaves the armature active, in Pose Mode",
      bpy.context.view_layer.objects.active == rig and rig.mode == "POSE")
bpy.ops.object.mode_set(mode="OBJECT")

# --- a sync bone: the thigh swinging forward carries the skirt
def swing(rig):
    pb = rig.pose.bones["thigh"]
    pb.rotation_mode = "XYZ"
    for frame, angle in ((1, 0.0), (20, -0.8), (40, -0.8)):
        pb.rotation_euler = (angle, 0.0, 0.0)
        pb.keyframe_insert("rotation_euler", frame=frame)
    for frame, x in ((1, 0.0), (20, 0.15), (40, 0.15)):
        pb.location = (x, 0.0, 0.0)                   # the thigh's local X is world X
        pb.keyframe_insert("location", frame=frame)


rig, group = build()
group.stiffness = 0.5
swing(rig)
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
unsynced = tip(rig)
scene.waifu_physics.simulate = False
rig, group = build()
group.stiffness = 0.5
swing(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for pb in rig.pose.bones:
    pb.select = pb.name in ("thigh", "s1")
rig.data.bones.active = rig.data.bones["thigh"]
check("Add Sync Bone follows the active bone and targets the selected chain bones",
      bpy.ops.waifu_physics.sync_add() == {"FINISHED"} and group.sync_bones[0].bone == "thigh"
      and [t.bone for t in group.sync_bones[0].targets] == ["s1"])
bpy.ops.object.mode_set(mode="OBJECT")
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
synced = tip(rig)
moved = float(np.abs(synced - unsynced).max())
check("the skirt follows the thigh's sideways movement", synced[0] - unsynced[0] > 0.03, (synced, unsynced))
scene.waifu_physics.simulate = False

# --- a new Procedural Wind force starts as Kawaii's Breeze and blows
rig, group = build()
bpy.context.view_layer.objects.active = rig
bpy.ops.waifu_physics.force_add(kind="PROCEDURAL_WIND")
breeze = group.forces[0]
check("a new Procedural Wind force starts as Kawaii's Breeze, in Blender units",
      math.isclose(breeze.constant, 0.02, rel_tol=1e-6) and math.isclose(breeze.sway_period, 3.0, rel_tol=1e-6)
      and math.isclose(math.degrees(breeze.ripple_delay), 120.0, rel_tol=1e-5))
scene.frame_set(1)
scene.waifu_physics.simulate = True
play(range(1, 41))
check("... and moves the chain out of the box", float(np.abs(tip(rig) - rest).max()) > 0.005, tip(rig) - rest)
scene.waifu_physics.simulate = False
bpy.ops.waifu_physics.wind_preset(preset="STORM")
check("the Storm preset sets Kawaii's Storm values", math.isclose(breeze.constant, 0.15, rel_tol=1e-6))

# --- procedural wind resumes from the cache exactly as an uninterrupted run
def windy():
    rig, group = build()
    bpy.context.view_layer.objects.active = rig
    bpy.ops.waifu_physics.force_add(kind="PROCEDURAL_WIND")
    force = group.forces[0]
    force.direction = (1.0, 0.0, 0.0)
    force.constant, force.sway, force.random = 0.2, 0.3, 0.2
    force.sway_period = 0.7
    return rig, group


rig, group = windy()
scene.frame_set(1)
scene.waifu_physics.simulate = True
bpy.ops.waifu_physics.cache_toggle()
baked = {}
for f in range(1, 31):
    scene.frame_set(f)
    baked[f] = tip(rig)
scene.frame_set(8)
again = {f: (scene.frame_set(f), tip(rig))[1] for f in (25, 12, 30)}
check("procedural wind bakes into the cache and scrubs back exactly",
      live.is_cached(scene) and all(np.array_equal(again[f], baked[f]) for f in again))
check("... and the wind moved it", float(np.abs(baked[30] - rest).max()) > 0.01)
group.forces[0].sway = 0.5
check("changing a force setting keeps the bake, outdated", live.is_cached(scene) and live.runtime(scene).outdated)
bpy.ops.waifu_physics.cache_toggle()
bpy.ops.object.effector_add(type="WIND")
bpy.context.view_layer.update()
bpy.ops.waifu_physics.cache_toggle()
check("baked again", live.is_cached(scene))
field = [o for o in scene.objects if o.field and o.field.type == "WIND"][-1]
field.location.x += 1.0
bpy.context.view_layer.update()
check("moving a wind field keeps the bake, outdated", live.is_cached(scene) and live.runtime(scene).outdated)
scene.waifu_physics.simulate = False
scene.waifu_physics.use_cache = False

# --- saved setups carry forces, curves and sync bones
rig, group = build()
bpy.context.view_layer.objects.active = rig
bpy.ops.waifu_physics.force_add(kind="CURVE")
bpy.ops.waifu_physics.force_add(kind="GRAVITY")
curve_force = group.forces[0]
points = curves.node(curve_force, "force_x").mapping.curves[0].points
points.new(0.5, 0.8)
curves.node(curve_force, "force_x").mapping.update()        # as the curve widget does after an edit
curve_force.amplitude = (2.0, 0.0, 0.0)
group.forces[1].use_rate_curve = True
group.forces[1].apply_bones.add().name = "s2"
sync = group.sync_bones.add()
sync.bone = "thigh"
sync.targets.add().bone = "s1"
sync.use_distance_curve = True
data = serialize.armature_to_dict(rig, scene)
text = json.dumps(data)
saved = data["groups"][0]
check("a setup saves forces with their curves, filters and sync bones",
      [f["settings"]["kind"] for f in saved["forces"]] == ["CURVE", "GRAVITY"]
      and set(saved["forces"][0]["curves"]) == {"force_x", "force_y", "force_z"}
      and len(saved["forces"][0]["curves"]["force_x"]["points"]) == 3
      and saved["forces"][1]["apply_bones"] == ["s2"] and "rate" in saved["forces"][1]["curves"]
      and saved["sync_bones"][0]["settings"]["bone"] == "thigh" and "distance" in saved["sync_bones"][0]["curves"])
other, _ = build()
serialize.armature_from_dict(other, json.loads(text), scene)
again = serialize.armature_to_dict(other, scene)["groups"][0]
def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in set(a) | set(b) for d in differences(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [(path, a, b)]


found = differences({"f": again["forces"], "s": again["sync_bones"]}, {"f": saved["forces"], "s": saved["sync_bones"]})
check("... and loads them back the same", not found, found[:4])

addon.unregister()
finish()
