"""The per-frame cache: replay, resume, uncached frames, invalidation, Cache All, rendering."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

addon = fresh_import()
addon.register()
live = sys.modules["waifu_physics.runtime.live"]
from waifu_physics.data import colliders
scene = bpy.context.scene


def build():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene.waifu_physics.simulate = False
    scene.frame_start, scene.frame_end = 1, 40
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
        bone.head, bone.tail = head, head + Vector((0.2, 0, 0))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    group = rig.waifu_physics.groups.add()
    group.roots.add().name = "c0"
    group.damping = 0.05
    return rig, group


def tip(rig):
    return np.array(rig.pose.bones["c2"].tail)


def play(frames):
    return {f: (scene.frame_set(f), tip(rig))[1] for f in frames}


rig, group = build()
scene.frame_set(1)
scene.waifu_physics.simulate = True
check("Cache bakes the whole frame range", bpy.ops.waifu_physics.cache_toggle() == {"FINISHED"}
      and live.is_cached(scene) and live.runtime(scene).cached_range() == (1, 40), live.runtime(scene).cached_range())
played = play(range(1, 41))
bpy.context.view_layer.update()
check("... and our own writes do not clear it", live.is_cached(scene))
replayed = {f: (scene.frame_set(f), tip(rig))[1] for f in (25, 7, 33, 40)}
check("scrubbing replays the bake exactly",
      all(np.array_equal(replayed[f], played[f]) for f in replayed), {f: replayed[f] - played[f] for f in replayed})
check("clicking Cache again clears the bake and plays live",
      bpy.ops.waifu_physics.cache_toggle() == {"FINISHED"} and not live.is_cached(scene) and not scene.waifu_physics.use_cache)
play(range(1, 11))
check("... simulating each frame as it plays", not live.runtime(scene).cache and live.runtime(scene).last_frame == 10)

rig, group = build()
scene.frame_end = 20
scene.frame_set(1)
scene.waifu_physics.simulate = True
rest = tip(rig)
bpy.ops.waifu_physics.cache_toggle()
scene.frame_set(35)
check("a frame past the baked range shows the unsimulated pose", np.allclose(tip(rig), rest, atol=1e-6),
      (tip(rig), rest))
check("the bake holds frames 1-20", live.runtime(scene).cached_range() == (1, 20), live.runtime(scene).cached_range())
scene.frame_set(10)
baked_10 = tip(rig)
group.damping = 0.3
check("changing a setting keeps the bake, outdated", live.is_cached(scene)
      and live.runtime(scene).outdated == "a setting changed", live.runtime(scene).outdated)
scene.frame_set(9)
scene.frame_set(10)
check("... which still plays as it was baked", np.array_equal(tip(rig), baked_10), (tip(rig), baked_10))
bpy.ops.waifu_physics.cache_all()
check("Recache (Cache All) bakes it again, up to date", live.is_cached(scene) and live.runtime(scene).outdated is None)
pb = rig.pose.bones["anchor"]
pb.keyframe_insert("rotation_quaternion", frame=1)
bpy.context.view_layer.update()
check("inserting a keyframe keeps the bake, outdated", live.is_cached(scene) and live.runtime(scene).outdated,
      live.runtime(scene).outdated)
rig.pose.bones["c1"].keyframe_insert("rotation_quaternion", frame=1)
bpy.context.view_layer.update()
rt = live.runtime(scene)
check("keys on a chain bone rebuild the simulation, and the bake survives it, outdated",
      live.is_cached(scene) and rt.outdated and rt.cached_range() == (1, 20), (rt.outdated, rt.cached_range()))
bag = rig.animation_data.action.layers[0].strips[0].channelbag(rig.animation_data.action_slot)
chain_curves = [c for c in bag.fcurves if c.data_path.startswith('pose.bones["c1"]')]
check("a newly keyed chain bone's curves are taken over (muted), so no post-frame replay is needed",
      chain_curves and all(c.mute for c in chain_curves) and not rt.needs_post_replay())
scene.waifu_physics.simulate = False
check("... and handed back when simulation stops", all(not c.mute for c in chain_curves))
scene.waifu_physics.simulate = True
ball = colliders.add(rig, "anchor", "Sphere")
bpy.ops.waifu_physics.cache_all()
ball.location.x += 0.1
bpy.context.view_layer.update()
check("moving a collider keeps the bake, outdated", live.is_cached(scene)
      and live.runtime(scene).outdated == "the scene changed", live.runtime(scene).outdated)
bpy.ops.waifu_physics.cache_all()
for other in list(bpy.context.view_layer.objects.selected):
    other.select_set(False)
ball.select_set(True)
bpy.context.view_layer.objects.active = ball
check("deleting a collider in the viewport ...", bpy.ops.object.delete() == {"FINISHED"})
bpy.context.view_layer.update()
check("... is noticed: the bake is kept, outdated", live.is_cached(scene) and live.runtime(scene).outdated,
      live.runtime(scene).outdated)
bpy.ops.waifu_physics.cache_all()
check("... and baked again, the chains no longer meet it", len(live.runtime(scene).system.shape_type) == 0,
      len(live.runtime(scene).system.shape_type))
scene.frame_end = 25
scene.frame_set(3)
check("changing the frame range keeps the bake too, outdated", live.is_cached(scene)
      and live.runtime(scene).outdated == "the frame range or rate changed", live.runtime(scene).outdated)
scene.frame_end = 20
bpy.ops.waifu_physics.cache_all()

scene.frame_end = 30
bpy.ops.waifu_physics.cache_all()
check("Cache All fills the frame range", live.runtime(scene).cached_range() == (1, 30),
      live.runtime(scene).cached_range())

# --- a render of a cached frame shows the cached pose
bpy.ops.mesh.primitive_cube_add(size=0.06)
marker = bpy.context.object
marker.parent = rig
marker.parent_type = "BONE"
marker.parent_bone = "c2"
marker.matrix_world = rig.matrix_world @ rig.pose.bones["c2"].matrix
marker.color = (1.0, 0.0, 0.0, 1.0)
camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
camera.location = (0.3, -3.0, 1.6)
camera.rotation_euler = (1.5708, 0.0, 0.0)
scene.camera = camera
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type = "OBJECT"
scene.display.shading.light = "FLAT"
scene.render.resolution_x, scene.render.resolution_y = 320, 240
scene.render.resolution_percentage = 100
scene.world = scene.world or bpy.data.worlds.new("world")
scene.world.color = (0.0, 0.0, 0.0)
bpy.ops.waifu_physics.cache_all()
scene.frame_set(30)
expected = world_to_camera_view(scene, camera, marker.matrix_world.translation)
path = os.path.join(tempfile.gettempdir(), "waifu_physics_cache_render.png")
scene.render.filepath = path
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(path)
pixels = np.array(image.pixels[:]).reshape(image.size[1], image.size[0], 4)
red = (pixels[:, :, 0] > 0.5) & (pixels[:, :, 1] < 0.3)
ys, xs = np.nonzero(red)
found = (xs.mean() / image.size[0], ys.mean() / image.size[1]) if len(xs) else None
check("a render of a cached frame shows the cached pose",
      found is not None and abs(found[0] - expected.x) < 0.03 and abs(found[1] - expected.y) < 0.03,
      (found, tuple(expected)[:2]))

scene.waifu_physics.simulate = False
addon.unregister()
finish()
