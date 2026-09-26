"""Collider objects: drawn exactly as they collide, pushing chains, shared across armatures."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

addon = fresh_import()
addon.register()
from waifu_physics.data import colliders
from waifu_physics.solver import uemath as ue
live = sys.modules["waifu_physics.runtime.live"]
scene = bpy.context.scene
CM = 100.0


def armature(name, chain=True):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    anchor = data.edit_bones.new("anchor")
    anchor.head, anchor.tail = (0, 0, 1.6), (0, 0, 2.0)
    if chain:
        parent, head = anchor, Vector((0, 0, 2.0))
        for i in range(4):
            bone = data.edit_bones.new(f"c{i}")
            bone.head, bone.tail = head, head + Vector((0.15, 0, 0))
            bone.parent = parent
            parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def world_vertices(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    points = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    evaluated.to_mesh_clear()
    return points


# --- each shape is drawn exactly where the solver collides, even with an uneven object scale
rig = armature("rig")
col = colliders.add(rig, "anchor", "Sphere")
check("a collider is a wire mesh parented to its bone, never rendered",
      col.parent == rig and col.parent_type == "BONE" and col.parent_bone == "anchor" and col.hide_render
      and col.display_type == "WIRE" and col.waifu_physics_collider.is_collider)
col.scale = (1.5, 0.8, 1.2)
col.rotation_euler = (0.3, -0.2, 0.5)
for shape in colliders.SHAPES:
    colliders.set_value(col, "Shape", shape)
    colliders.set_value(col, "Radius", 0.05)
    colliders.set_value(col, "Radius 1", 0.09)
    colliders.set_value(col, "Length", 0.3)
    colliders.set_value(col, "Extent", (0.04, 0.06, 0.08))
    points = world_vertices(col)
    s = colliders.shape_of(col, rig, CM)
    rot = np.array(s.rotation)[None]
    centre = np.array(s.location) / CM
    local = ue.unrotate_vector(np.repeat(rot, len(points), 0), points - centre)   # the shape's own frame
    if shape in ("Sphere", "Inner Sphere"):
        error = np.abs(np.linalg.norm(local, axis=1) - s.radius / CM).max()
    elif shape == "Capsule":
        half = s.length / CM / 2
        axial = np.clip(local[:, 2], -half, half)
        error = np.abs(np.linalg.norm(local - np.stack([0 * axial, 0 * axial, axial], 1), axis=1) - s.radius / CM).max()
    elif shape == "Tapered Capsule":
        half = s.length / CM / 2
        top = local[:, 2] > 0
        ends = np.where(top, half, -half)
        radius = np.where(top, s.radius, s.radius1) / CM
        error = np.abs(np.linalg.norm(local - np.stack([0 * ends, 0 * ends, ends], 1), axis=1) - radius).max()
    elif shape == "Box":
        error = np.abs(np.abs(local) - np.array(s.extent) / CM).max()
    else:
        on_plane = np.abs(local[:, 2]) < 1e-6
        error = 0.0 if on_plane.sum() >= 4 else 1.0
    check(f"{shape}: drawn on the surface the solver collides with, scaled (1.5, 0.8, 1.2)", error < 1e-5, error)

# --- capsules run along the collider's Y (the bone), which is Kawaii's Z
col.scale = (1, 1, 1)
col.rotation_euler = (0, 0, 0)
col.matrix_world = rig.matrix_world @ rig.pose.bones["anchor"].matrix
colliders.set_value(col, "Shape", "Capsule")
s = colliders.shape_of(col, rig, CM)
axis_z = ue.axis(np.array(s.rotation)[None], 2)[0]
bone_y = np.array(rig.pose.bones["anchor"].matrix.to_3x3().col[1])
check("a capsule lies along its bone: Kawaii's capsule axis is the bone's Y", np.allclose(axis_z, bone_y, atol=1e-6),
      (axis_z, bone_y))

# --- a falling chain rests on a capsule under it
bpy.data.objects.remove(col)
g = rig.waifu_physics.groups.add()
g.roots.add().name = "c0"
g.dummy_bone_length = 0.1
g.radius = 0.02
g.damping = 0.1
floor = colliders.add(rig, "anchor", "Capsule")
floor.matrix_world = Matrix.Translation((0.35, 0.0, 1.8)) @ Matrix.Rotation(math.pi / 2, 4, "Z")
colliders.set_value(floor, "Radius", 0.06)
colliders.set_value(floor, "Length", 0.8)
scene.frame_set(1)
scene.waifu_physics.simulate = True
for frame in range(1, 60):
    scene.frame_set(frame)
rt = live.runtime(scene)
check("the group collides with its armature's collider", len(rt.system.shape_type) == 1, len(rt.system.shape_type))
shape = colliders.shape_of(floor, rig, CM)
start = np.array(shape.location) + ue.axis(np.array(shape.rotation)[None], 2)[0] * shape.length / 2
end = np.array(shape.location) - ue.axis(np.array(shape.rotation)[None], 2)[0] * shape.length / 2
moving = rt.system.loc[rt.system.parent >= 0]
segment = end - start
t = np.clip(((moving - start) @ segment) / (segment @ segment), 0, 1)
gaps = np.linalg.norm(moving - (start + t[:, None] * segment), axis=1) - shape.radius
check("every chain point stays a point radius clear of the capsule", gaps.min() > 2.0 - 1e-3, gaps.min())
check("... after resting on it (the chain fell onto it)", (rt.system.loc[:, 2] < 200.0 - 1.0).any())

# --- disabled colliders are ignored
floor.waifu_physics_collider.enabled = False
scene.frame_set(60)
check("a disabled collider is left out", len(rt.system.shape_type) == 0, len(rt.system.shape_type))
scene.waifu_physics.simulate = False

# --- a group can use another armature's colliders, placed in its own armature's space
body = armature("body", chain=False)
ball = colliders.add(body, "anchor", "Sphere")
body.location = (1.0, 2.0, 0.0)
rig.rotation_euler = (0.0, 0.0, 0.7)
g.collider_sets.add().armature = body
scene.waifu_physics.simulate = True
scene.frame_set(2)
rt = live.runtime(scene)
shapes = rt.system.shape_type
bpy.context.view_layer.update()
expected = (rig.matrix_world.inverted() @ ball.matrix_world).translation * CM
check("another armature's collider joins the group through a collider set", len(shapes) == 1, len(shapes))
check("... in the group's armature space", np.allclose(rt.system.shape_loc[0], expected, atol=1e-4),
      (rt.system.shape_loc[0], tuple(expected)))
g.collider_sets.clear()
g.custom_collider_sets = False
scene.frame_set(3)
check("with Every Collider (the default) a group collides with every armature's colliders, listed or not",
      g.use_all_colliders and len(live.runtime(scene).system.shape_type) == 1, len(live.runtime(scene).system.shape_type))
g.use_all_colliders = False
scene.frame_set(4)
check("... and off, only its armatures' (its own and its parent's, by default)",
      len(live.runtime(scene).system.shape_type) == 0, len(live.runtime(scene).system.shape_type))
g.collider_sets.add().armature = body
g.use_all_colliders = True
scene.waifu_physics.simulate = False

# --- a scene collider, on no armature, is a ground every group stands on
ground = colliders.add_to_scene("Plane", location=(0.0, 0.0, 1.9))
check("a scene collider is on no armature: an upright plane, a ground",
      colliders.is_scene_collider(ground) and colliders.values(ground)["Shape"] == "Plane"
      and ground in colliders.scene_colliders(scene))
scene.frame_set(1)
scene.waifu_physics.simulate = True
for frame in range(1, 60):
    scene.frame_set(frame)
rt = live.runtime(scene)
check("the group collides with the scene's colliders as well as its armatures'", len(rt.system.shape_type) == 2,
      len(rt.system.shape_type))
lowest = rt.system.loc[rt.system.parent >= 0][:, 2].min()
check("the chain falls onto the ground and rests a point radius above it", 190.0 + 2.0 - 1e-3 < lowest < 193.0, lowest)
frame_cache = sys.modules["waifu_physics.runtime.cache"]
before = frame_cache.collider_prints(rt)
ground.location.z = 1.8
check("moving a scene collider is a collider edit (the cache is redone)", frame_cache.collider_prints(rt) != before)
g.use_all_colliders = False
g.use_scene_colliders = False
scene.frame_set(60)
check("a group can leave the scene's colliders out", len(live.runtime(scene).system.shape_type) == 1,
      len(live.runtime(scene).system.shape_type))
g.use_scene_colliders = True
g.use_all_colliders = True
check("the colliders show by default", scene.waifu_physics.show_colliders and ground.visible_get())
scene.waifu_physics.show_colliders = False
check("the eye off hides them, by their collection's eye (their own eyes stay the user's) ...",
      not ground.visible_get() and not ground.hide_get()
      and bpy.context.view_layer.layer_collection.children[colliders.COLLECTION].hide_viewport)
scene.frame_set(61)
check("... and hidden, they still collide", len(live.runtime(scene).system.shape_type) == 2)
extra = colliders.add_to_scene("Sphere")
check("a collider made while they are hidden is hidden like the rest", not extra.visible_get())
ground.hide_set(True)
scene.waifu_physics.show_colliders = True
check("a collider the user hid by its own eye stays hidden when they show", not ground.visible_get()
      and extra.visible_get())
ground.hide_set(False)
scene.waifu_physics.simulate = False
for obj in colliders.scene_colliders(scene, enabled_only=False):
    bpy.data.objects.remove(obj)

# --- keyframed collider sizes: the solver reads the animated value, and the drawing follows
md, ident = colliders.input_path(ball, "Radius")
socket = getattr(md.properties.inputs, ident)
socket.value = 0.05
ball.keyframe_insert(socket.path_from_id("value"), frame=1)
socket.value = 0.2
ball.keyframe_insert(socket.path_from_id("value"), frame=10)
scene.frame_set(10)
radius_at_10 = colliders.shape_of(ball, body, CM).radius
drawn = world_vertices(ball)
drawn_radius = np.linalg.norm(drawn - np.array(ball.matrix_world.translation), axis=1).max()
check("a keyframed collider radius reaches the solver", abs(radius_at_10 - 20.0) < 1e-3, radius_at_10)
check("... and the drawing follows it", abs(drawn_radius - 0.2) < 1e-4, drawn_radius)

# --- colliders on a body with no groups; a garment parented to it collides with them by default
scene.frame_set(1)
peach = armature("peach", chain=False)
check("an armature with no groups can carry colliders", not len(peach.waifu_physics.groups))
hip = colliders.add(peach, "anchor", "Sphere")
garment = armature("garment")
garment.parent = peach
skirt = garment.waifu_physics.groups.add()
skirt.roots.add().name = "c0"
check("a group collides with its own armature's colliders and its parent armature's by default",
      colliders.sources(skirt) == [garment, peach], [a.name for a in colliders.sources(skirt)])
bpy.context.view_layer.objects.active = garment
garment.waifu_physics.active_group = 0
bpy.ops.waifu_physics.collider_set_remove(index=0)
check("removing one of the defaults keeps the other: the list now names what is left",
      colliders.sources(skirt) == [peach], [a.name for a in colliders.sources(skirt)])
bpy.ops.waifu_physics.collider_set_add()
check("adding makes an empty slot to pick an armature in", len(skirt.collider_sets) == 2)
skirt.collider_sets.clear()
check("with the edited list emptied, the group collides with no armature", colliders.sources(skirt) == [])
skirt.custom_collider_sets = False
check("... and unmarked as edited, the defaults are back", colliders.sources(skirt) == [garment, peach])
for other in list(scene.objects):
    other.select_set(False)
hip.select_set(True)
bpy.context.view_layer.objects.active = hip
check("with a collider selected, the panel shows its armature", colliders.armature_of(bpy.context) == peach)
check("... and the list holds that armature's colliders", colliders.all_of(peach) == [hip])
name = hip.name
check("Remove deletes the collider", bpy.ops.waifu_physics.collider_remove(name=name) == {"FINISHED"}
      and name not in bpy.data.objects and colliders.all_of(peach) == [])

# --- colliders from bones, as thick as the skin around them
limb = armature("limb", chain=False)                     # one bone, "anchor": (0, 0, 1.6) up to (0, 0, 2.0)
bpy.ops.object.mode_set(mode="OBJECT")
ring = [(0.1 * math.cos(a), 0.1 * math.sin(a), z) for z in (1.65, 1.8, 1.95) for a in
        np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
skin_mesh = bpy.data.meshes.new("skin")
skin_mesh.from_pydata(ring, [], [])
skin = bpy.data.objects.new("skin", skin_mesh)
scene.collection.objects.link(skin)
skin.modifiers.new("Armature", "ARMATURE").object = limb
skin.vertex_groups.new(name="anchor").add(list(range(len(ring))), 1.0, "REPLACE")
found = colliders.fit(limb, "anchor")
check("a limb of skin fits a capsule along its bone, as thick as the skin",
      found.shape in ("Capsule", "Tapered Capsule") and abs(found.radius - 0.1) < 2e-3
      and abs(found.radius1 - 0.1) < 2e-3, (found.shape, found.radius, found.radius1))
check("... centred on the skin along the bone (from 0.05 to 0.35 of it)", abs(found.center[1] - 0.2) < 1e-3,
      found.center)
check("... nothing is fitted where no skin belongs to the bone", colliders.fit(rig, "c1") is None)
bpy.context.view_layer.objects.active = limb
bpy.ops.object.mode_set(mode="POSE")
for bone in limb.pose.bones:
    bone.select = True
check("Colliders from Bones adds one to each selected bone",
      bpy.ops.waifu_physics.colliders_from_bones() == {"FINISHED"} and len(colliders.all_of(limb)) == 1)
made = colliders.all_of(limb)[0]
check("... fitted to the skin (Auto), named from the bone",
      colliders.values(made)["Shape"] in ("Capsule", "Tapered Capsule") and made.name.startswith("anchor Collider")
      and abs(colliders.values(made)["Radius"] - 0.1) < 2e-3, (made.name, colliders.values(made)))
bpy.ops.waifu_physics.colliders_from_bones(shape="Sphere")
check("... and again, it is regenerated: its collider is replaced, not joined by another",
      len(colliders.all_of(limb)) == 1 and colliders.values(colliders.all_of(limb)[0])["Shape"] == "Sphere")
bpy.ops.object.mode_set(mode="OBJECT")
check("VRoid bone names shorten", colliders.short_name("J_Bip_C_Head") == "Head"
      and colliders.short_name("J_Bip_L_UpperArm") == "L_UpperArm")

# --- Auto picks the shape the skin has: a ball, a limb, a block
from waifu_physics.data import collider_fit
rng = np.random.default_rng(3)
directions = rng.normal(size=(400, 3))
directions /= np.linalg.norm(directions, axis=1)[:, None]
ball = directions * 0.09 + np.array([0.0, 0.1, 0.02])
check("skin shaped like a ball fits a sphere", collider_fit.fit(ball).shape == "Sphere", collider_fit.fit(ball).shape)
check("... of its radius, around its centre", abs(collider_fit.fit(ball, "Sphere").radius - 0.09) < 3e-3
      and np.allclose(collider_fit.fit(ball, "Sphere").center, (0.0, 0.1, 0.02), atol=5e-3))
angles, heights = rng.uniform(0, 2 * np.pi, 600), rng.uniform(0.0, 0.4, 600)
limb_skin = np.column_stack((0.05 * np.cos(angles), heights, 0.05 * np.sin(angles)))
check("skin shaped like a limb fits a capsule", collider_fit.fit(limb_skin).shape == "Capsule",
      collider_fit.fit(limb_skin).shape)
cone_r = 0.03 + 0.05 * heights / 0.4
cone = np.column_stack((cone_r * np.cos(angles), heights, cone_r * np.sin(angles)))
fitted = collider_fit.fit(cone)
check("skin that thickens along the bone fits a tapered capsule, thicker at the +Y end",
      fitted.shape == "Tapered Capsule" and fitted.radius > fitted.radius1, (fitted.shape, fitted.radius, fitted.radius1))
faces = rng.uniform(-1, 1, size=(900, 3))
axis = rng.integers(0, 3, 900)
faces[np.arange(900), axis] = np.sign(faces[np.arange(900), axis])
block = faces * np.array([0.12, 0.05, 0.08])
check("skin shaped like a block fits a box of its half sizes", collider_fit.fit(block).shape == "Box"
      and np.allclose(collider_fit.fit(block).extent, (0.12, 0.05, 0.08), atol=0.01),
      (collider_fit.fit(block).shape, collider_fit.fit(block).extent))
check("a shape asked for is the shape fitted", collider_fit.fit(ball, "Box").shape == "Box")
check("too little skin fits nothing", collider_fit.fit(ball[:5]) is None)

# --- a group's collider list, once edited, stands even empty: its own armature can be removed
own = armature("own_rig")
own_group = own.waifu_physics.groups.add()
own_group.roots.add().name = "c0"
check("by default a group collides with its own armature's colliders", colliders.sources(own_group) == [own])
bpy.context.view_layer.objects.active = own
check("removing its own armature from the list ...",
      bpy.ops.waifu_physics.collider_set_remove(index=0) == {"FINISHED"})
check("... leaves it colliding with no armature (the defaults do not come back)",
      colliders.sources(own_group) == [] and own_group.custom_collider_sets, colliders.sources(own_group))

# --- no collider on a bone a chain simulates: it would chase the chain it pushes
check("the simulated bones are the chains below their roots",
      colliders.simulated_bones(own) == {"c1", "c2", "c3"}, colliders.simulated_bones(own))
bpy.ops.object.mode_set(mode="POSE")
own.data.bones.active = own.data.bones["c2"]
try:
    bpy.ops.waifu_physics.collider_add(shape="Sphere")
    refusal = None
except RuntimeError as error:                   # an ERROR report raises from bpy.ops
    refusal = str(error)
check("Add Collider refuses a bone in a chain, saying why", refusal is not None and "in a chain" in refusal
      and not colliders.all_of(own), refusal)
made = colliders.from_bones(own, ["anchor", "c0", "c1", "c2"], "Capsule")
check("Colliders from Bones skips the chains' bones (a root keeps its animation, so it may carry one)",
      sorted(obj.parent_bone for obj in made) == ["anchor", "c0"], [obj.parent_bone for obj in made])
bpy.ops.object.mode_set(mode="OBJECT")
late = colliders.add(own, "c3", "Sphere")
check("a collider that ends up on a chain bone is flagged", colliders.on_simulated_bone(late)
      and not colliders.on_simulated_bone(made[0]))

# --- the list's pick is the viewport's selection, both ways
settings = scene.waifu_physics
bpy.ops.object.mode_set(mode="OBJECT")
floor_plane = colliders.add_to_scene("Plane")
settings.active_collider = bpy.data.objects.find(floor_plane.name)
check("picking a collider in the list selects it in the viewport",
      bpy.context.view_layer.objects.active == floor_plane and floor_plane.select_get())
bpy.context.view_layer.objects.active = made[0]
check("... and selecting one in the viewport picks it in the list",
      settings.active_collider == bpy.data.objects.find(made[0].name))
for obj in list(bpy.context.view_layer.objects.selected):
    obj.select_set(False)
own.select_set(True)
bpy.context.view_layer.objects.active = own
bpy.ops.object.mode_set(mode="POSE")
settings.active_collider = bpy.data.objects.find(made[0].name)
check("in Pose Mode, picking a bone's collider in the list selects its bone",
      own.data.bones.active.name == made[0].parent_bone and own.pose.bones[made[0].parent_bone].select)
own.data.bones.active = own.data.bones["anchor"]
anchor_collider = next(obj for obj in made if obj.parent_bone == "anchor")
check("... and making a bone active picks its collider",
      settings.active_collider == bpy.data.objects.find(anchor_collider.name))

# --- Shift and Ctrl in the list: a range, and adding or dropping one (the viewport's selection, both ways)
pick = bpy.ops.waifu_physics.collider_pick
chosen = lambda: set(colliders.chosen(scene, bpy.data.objects.get(settings.last_collider)))
listed = colliders.listed(bpy.context)
on_bones = [obj for obj in listed if obj.parent == own]
pick(name=on_bones[0].name)
pick(name=on_bones[1].name, extend=True)
check("in Pose Mode, Ctrl-click adds a collider: its bone joins the selection and is made active",
      chosen() == set(on_bones[:2]) and own.data.bones.active.name == on_bones[1].parent_bone
      and own.pose.bones[on_bones[0].parent_bone].select, sorted(obj.name for obj in chosen()))
pick(name=on_bones[1].name, extend=True)
check("... Ctrl-click on a selected one drops it, and the pick moves to one still selected",
      chosen() == {on_bones[0]} and own.data.bones.active.name == on_bones[0].parent_bone)
bpy.ops.object.mode_set(mode="OBJECT")
second_plane = colliders.add_to_scene("Plane")
listed = colliders.listed(bpy.context)
pick(name=listed[0].name)
pick(name=listed[-1].name, span=True)
everything = {obj for obj in listed if obj.name in bpy.context.view_layer.objects}
check("in Object Mode, Shift-click selects the run of listed colliders from the last one picked",
      chosen() == everything and bpy.context.view_layer.objects.active == listed[-1]
      and settings.last_collider == listed[0].name, (len(chosen()), len(everything)))
pick(name=listed[-1].name, extend=True)
check("... Ctrl-click drops the picked one, and another selected collider is picked",
      listed[-1] not in chosen() and not listed[-1].select_get()
      and colliders.is_collider(bpy.context.view_layer.objects.active))
pick(name=second_plane.name)
check("... a plain click picks one alone", chosen() == {second_plane})
pick(name=floor_plane.name, extend=True)
count = len(colliders.all_of(own))
planes = (floor_plane.name, second_plane.name)
check("the trash on the picked collider deletes every selected one",
      bpy.ops.waifu_physics.collider_remove(name=planes[0]) == {"FINISHED"}
      and not any(name in bpy.data.objects for name in planes))
check("... and leaves the colliders that were not selected", len(colliders.all_of(own)) == count)
bpy.context.view_layer.objects.active = own
bpy.ops.object.mode_set(mode="POSE")

# --- Regenerate replaces a bone's colliders with one fitted; Generate leaves them
before = {obj.name for obj in colliders.all_of(own)}
check("Generate skips bones that have a collider", colliders.from_bones(own, ["anchor"], "Sphere") == [])
redone = colliders.from_bones(own, ["anchor"], "Sphere", replace=True)
on_anchor = [obj for obj in colliders.all_of(own) if obj.parent_bone == "anchor"]
check("Regenerate replaces them with one fitted (a Capsule before, a Sphere now)",
      on_anchor == redone and len(redone) == 1 and colliders.values(redone[0])["Shape"] == "Sphere"
      and len(colliders.all_of(own)) == len(before), [obj.name for obj in on_anchor])
extra = [colliders.add(own, "anchor", "Box") for _ in range(2)]
redone = colliders.from_bones(own, ["anchor"], "Sphere", replace=True)
check("Regenerate erases every collider a bone has, however many, for the one fitted",
      [obj for obj in colliders.all_of(own) if obj.parent_bone == "anchor"] == redone and len(redone) == 1)
bpy.ops.object.mode_set(mode="OBJECT")

# --- showing and hiding before there is any collider: nothing to do, and nothing is made
for obj in [obj for obj in bpy.data.objects if colliders.is_collider(obj)]:
    colliders.remove(obj)
bpy.data.collections.remove(bpy.data.collections[colliders.COLLECTION])
settings.show_colliders = False
settings.show_colliders = True
check("the eye with no colliders makes no collection", colliders.COLLECTION not in bpy.data.collections)

addon.unregister()
finish()
