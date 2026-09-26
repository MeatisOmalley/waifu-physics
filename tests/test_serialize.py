"""Setups as data: a round trip through JSON rebuilds the same setup and the same motion."""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

addon = fresh_import()
addon.register()
from waifu_physics.data import colliders, curves, presets, serialize

scene = bpy.context.scene


def armature(name):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    hips = data.edit_bones.new("hips")
    hips.head, hips.tail = (0, 0, 1.0), (0, 0, 1.1)
    for k in range(4):
        angle = math.pi / 2 * k
        parent, head = hips, Vector((0.2 * math.cos(angle), 0.2 * math.sin(angle), 1.0))
        for i in range(3):
            bone = data.edit_bones.new(f"p{k}_{i}")
            bone.head, bone.tail = head, head + Vector((0.03 * math.cos(angle), 0.03 * math.sin(angle), -0.15))
            bone.parent = parent
            parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def tips(obj):
    return np.array([obj.pose.bones[f"p{k}_2"].tail for k in range(4)])


source = armature("source")
body = armature("body")                 # a second armature whose colliders the skirt also meets
group = source.waifu_physics.groups.add()
group.name = "Skirt"
for k in range(4):
    group.roots.add().name = f"p{k}_0"
group.excluded.add().name = "p3_2"
for k in range(4):
    link = group.links.add()
    link.bone_a, link.bone_b = f"p{k}_1", f"p{(k + 1) % 4}_1"
group.links[1].compliance = "FAT"
group.links[2].exclude_from_subdivision = True
group.damping, group.stiffness, group.radius = 0.23, 0.07, 0.025
group.gravity = (0.5, 0.0, -1.0)
group.bridge_count = 1
group.collider_sets.add().armature = source
group.collider_sets.add().armature = body
group.use_stiffness_curve = True
points = curves.node(group, "stiffness").mapping.curves[0].points
points.new(0.4, 1.6)
points[0].location = (0.0, 0.5)
points[-1].location = (1.0, 1.2)
curves.node(group, "stiffness").mapping.update()

ball = colliders.add(source, "hips", "Tapered Capsule")
colliders.set_value(ball, "Radius", 0.07)
colliders.set_value(ball, "Radius 1", 0.03)
ball.location += Vector((0.02, 0.03, -0.01))
ball.rotation_euler = Euler((0.3, 0.1, -0.2))
ball.scale = (1.0, 1.4, 1.0)
colliders.add(body, "p0_0", "Sphere")
bpy.context.view_layer.update()

data = serialize.armature_to_dict(source, scene)
exported_tips = data["groups"][0]["tips"]
leaves = {name: source.data.bones[name].length for name in exported_tips}
check("a setup lists each chain's tip, its last bone's length (Kawaii's DummyBoneLength, per chain)",
      exported_tips and exported_tips == leaves, exported_tips)
text = json.dumps(data)
check("a setup is plain JSON", isinstance(text, str) and json.loads(text) == data)
check("it names the Kawaii Physics it follows", data["kawaii_commit"].startswith("64cbc77"))
stiffness = data["groups"][0]["curves"]["stiffness"]
sampled = curves.curve(group, "stiffness")
check("a curve carries its control points and the keys the solver used",
      len(stiffness["points"]) == 3 and stiffness["keys"] == sampled.keys() and len(stiffness["keys"]) == 65)

target = armature("target")
bpy.context.view_layer.update()
warnings = serialize.armature_from_dict(target, json.loads(text), scene)
bpy.context.view_layer.update()
check("loading reports nothing missing", warnings == [], warnings)
again = serialize.armature_to_dict(target, scene)
strip = lambda d: {**d, "armature": None, "colliders": [{**c, "name": None} for c in d["colliders"]]}


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in set(a) | set(b) for d in differences(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and ".matrix" in path:
        return [] if math.isclose(a, b, abs_tol=1e-7) else [(path, a, b)]         # float32 decomposition
    return [] if a == b else [(path, a, b)]


check("its own armature's colliders are stored as its own, not by name",
      data["groups"][0]["collider_sets"] == [None, "body"], data["groups"][0]["collider_sets"])
check("the loaded setup saves back to the same data", not differences(strip(again), strip(data)),
      differences(strip(again), strip(data))[:6])
restored = [c for c in target.children if colliders.is_collider(c)][0]
check("a collider comes back on its bone at the same place",
      restored.parent_bone == "hips" and np.allclose(np.array(restored.matrix_world), np.array(ball.matrix_world),
                                                      atol=1e-6))
check("... with its shape", colliders.values(restored) == colliders.values(ball),
      (colliders.values(restored), colliders.values(ball)))
check("a curve comes back point for point", curves.curve(target.waifu_physics.groups[0], "stiffness").keys() == sampled.keys())

# --- the same setup moves the same: posed and animated identically, bit for bit
for obj in (source, target):
    obj.animation_data_create()
    obj.location = (0, 0, 0)
    obj.keyframe_insert("location", frame=1)
    obj.location = (0.4, 0.1, 0)
    obj.keyframe_insert("location", frame=12)
    pb = obj.pose.bones["hips"]
    pb.rotation_mode = "XYZ"
    pb.keyframe_insert("rotation_euler", frame=1)
    pb.rotation_euler = (0.0, 0.0, 0.8)
    pb.keyframe_insert("rotation_euler", frame=12)
scene.frame_set(1)
scene.waifu_physics.simulate = True
motion = {}
for frame in range(1, 25):
    scene.frame_set(frame)
    motion[frame] = (tips(source), tips(target))
scene.waifu_physics.simulate = False
check("the loaded setup simulates exactly as the original",
      all(np.array_equal(a, b) for a, b in motion.values()),
      max(float(np.abs(a - b).max()) for a, b in motion.values()))
check("... and the simulation really moved the chains", float(np.abs(motion[24][0] - motion[1][0]).max()) > 0.01)

bad = dict(data, version=serialize.VERSION + 1)
try:
    serialize.check(bad)
    check("a setup from a newer version is refused", False)
except ValueError as error:
    check("a setup from a newer version is refused", "newer" in str(error))
try:
    serialize.check({"format": "something else"})
    check("anything else is refused", False)
except ValueError:
    check("anything else is refused", True)

# --- presets
hair = source.waifu_physics.groups.add()
hair.use_radius_curve = True
hair.damping = 0.9
presets.apply(hair, "HAIR")
expected = presets.PRESETS["HAIR"][2]
check("a preset sets what it names", all(serialize._plain(getattr(hair, k)) == v or
                                         math.isclose(getattr(hair, k), v, rel_tol=1e-6)
                                         for k, v in expected.items() if not isinstance(v, str)))
check("... and turns curves off", not any(getattr(hair, f"use_{s}_curve") for s in curves.CURVED))
presets.apply(hair, "SKIRT")
check("the skirt preset adds bridge points", hair.bridge_count == 1 and math.isclose(hair.radius, 0.02, rel_tol=1e-6),
      (hair.bridge_count, hair.radius))

# --- the operators: export, import, paste
bpy.context.view_layer.objects.active = source
path = os.path.join(tempfile.gettempdir(), "waifu_physics_setup.json")
check("Export Setup writes a file", bpy.ops.waifu_physics.setup_export(filepath=path) == {"FINISHED"}
      and json.load(open(path, encoding="utf-8"))["groups"][0]["name"] == "Skirt")
fresh = armature("fresh")
bpy.context.view_layer.objects.active = fresh
check("Import Setup loads it", bpy.ops.waifu_physics.setup_import(filepath=path) == {"FINISHED"}
      and [g.name for g in fresh.waifu_physics.groups] == ["Skirt", "Group"])
fresh.waifu_physics.active_group = 1
bpy.ops.object.mode_set(mode="POSE")
for pb in fresh.pose.bones:
    pb.select = False
result = bpy.ops.waifu_physics.group_paste(text=serialize.settings_text(group))
pasted = fresh.waifu_physics.groups[1]
check("Paste Settings copies settings and curves but not chains",
      result == {"FINISHED"} and math.isclose(pasted.damping, 0.23, rel_tol=1e-6) and pasted.use_stiffness_curve
      and curves.curve(pasted, "stiffness").keys() == sampled.keys() and len(pasted.roots) == 0)
try:
    refused = bpy.ops.waifu_physics.group_paste(text="hello") == {"CANCELLED"}
except RuntimeError as error:                   # an operator's error report raises when run from a script
    refused = "no Waifu Physics settings" in str(error)
check("... and refuses anything else", refused and math.isclose(pasted.damping, 0.23, rel_tol=1e-6))
bpy.ops.object.mode_set(mode="OBJECT")

addon.unregister()
finish()
