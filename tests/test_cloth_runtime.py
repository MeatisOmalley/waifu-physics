"""Live cloth uses evaluated skin, reversible GN output and the existing scene cache."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np

addon = fresh_import()
addon.register()
check("registering adds a cloth setup to mesh objects", hasattr(bpy.types.Object, "waifu_cloth"))
if not hasattr(bpy.types.Object, "waifu_cloth"):
    finish()

from waifu_physics.cloth import convention, display
from waifu_physics.data import serialize
from waifu_physics.runtime import live, cache
from waifu_physics.data import colliders

scene = bpy.context.scene
scene.render.fps = 60
scene.frame_start, scene.frame_end = 1, 30
scene.frame_set(1)
mesh = bpy.data.meshes.new("Flag")
mesh.from_pydata([(x * 0.1, y * 0.1, 1) for y in range(5) for x in range(5)], [],
                 [(y * 5 + x, y * 5 + x + 1, (y + 1) * 5 + x + 1, (y + 1) * 5 + x)
                  for y in range(4) for x in range(4)])
obj = bpy.data.objects.new("Flag", mesh)
scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
original = np.array([v.co[:] for v in mesh.vertices])
old = mesh.color_attributes.new("WS Cloth", "FLOAT_COLOR", "POINT")
colors = np.zeros((25, 4), dtype=np.float32)
colors[:, 3] = 1
colors[5:, 0] = 1
old.data.foreach_set("color_srgb", colors.ravel())
attribute = convention.ensure_attribute(mesh)
check("legacy cloth paint is renamed without losing values", attribute.name == "Waifu Cloth"
      and "WS Cloth" not in mesh.color_attributes and np.allclose(convention.channels(mesh), colors[:, :3], atol=1e-4))
obj.waifu_cloth.enabled = True
obj.waifu_cloth.max_distance = 0.4
bpy.context.view_layer.update()

document = serialize.cloth_to_dict(obj)
obj.waifu_cloth.friction = 0.2
serialize.cloth_from_dict(obj, document)
check("cloth setups serialize and restore", abs(obj.waifu_cloth.friction - 0.8) < 1e-6)

scene.waifu_physics.simulate = True
rt = live.runtime(scene)
check("cloth shares the scene runtime", rt.cloth is not None and len(rt.cloth.meshes) == 1)
check("cloth uses the native C backend", type(rt.cloth.backend).__name__ == "CBackend")
check("a display modifier is added last", obj.modifiers[-1].name == display.MODIFIER)
for frame in range(1, 31):
    scene.frame_set(frame)
bpy.context.view_layer.update()
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
result = np.array([v.co[:] for v in evaluated.vertices])
check("a free cloth edge falls in live playback", result[-1, 2] < original[-1, 2] - 0.001,
      (original[-1], result[-1], live.runtime(scene).cloth.system.x[-1], live.runtime(scene).last_frame))
check("black anchors follow the skin", np.allclose(result[:5], original[:5], atol=1e-6))
check("the original vertices are never changed", np.array_equal(original, np.array([v.co[:] for v in mesh.vertices])))
skinned = evaluated.attributes[display.SKINNED]
values = np.empty(75, dtype=np.float32)
skinned.data.foreach_get("vector", values)
check("skinned input never feeds simulated output back", np.allclose(values.reshape(-1, 3), original))

# The shared cache bakes at solver ticks, replays scrubbing, and restores input outside its range.
scene.waifu_physics.use_cache = True
live.bake_cache(scene)
scene.frame_set(30)
bpy.context.view_layer.update()
at_end = np.array([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
scene.frame_set(10)
scene.frame_set(30)
bpy.context.view_layer.update()
replayed = np.array([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
check("cloth cache replays identically after scrubbing", np.array_equal(at_end, replayed))
check("cloth's own cache replay stays current", live.runtime(scene).outdated is None, live.runtime(scene).outdated)
obj.location.x += 0.01
bpy.context.view_layer.update()
check("editing cloth skin input marks the bake outdated", live.runtime(scene).outdated is not None)
obj.location.x -= 0.01
bpy.context.view_layer.update()
scene.frame_set(40)
bpy.context.view_layer.update()
outside = np.array([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
check("outside the bake range displays the skin", np.allclose(outside, original))
scene.waifu_physics.simulate = False
bpy.context.view_layer.update()
stopped = np.array([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
check("stopping preview restores the skinned mesh", np.allclose(stopped, original))

# Motion is around the component origin, and scale belongs only to skinned input.
obj.location = (3, 2, 0)
obj.scale = (2, 1, 1)
bpy.context.view_layer.update()
scene.waifu_physics.use_cache = False
scene.waifu_physics.simulate = True
rt = live.runtime(scene)
cloth_rt = rt.cloth
_, references = cloth_rt._read(scene)
check("reference motion removes component scale", np.allclose(references[0][:3, :3].T @ references[0][:3, :3], np.eye(3)))
previous_origin = cloth_rt.previous_matrices[0][:3, 3].copy()
captured = {}
advance = cloth_rt.system.step_frame
def capture(dt, **kwargs):
    captured.update(kwargs)
    return advance(dt, **kwargs)
cloth_rt.system.step_frame = capture
cloth_rt.step(scene, 1 / 60)
check("motion retains the previous component pivot", np.array_equal(captured["motions"][0][1], previous_origin))
cloth_rt.system.step_frame = advance
other_scene = bpy.data.scenes.new("Other Scene")
bpy.context.window.scene = other_scene
check("cloth lifetime belongs to its original scene", cloth_rt.alive())
bpy.context.window.scene = scene
bpy.data.scenes.remove(other_scene)
shape = colliders.add_to_scene("Sphere", location=(20, 20, 20))
check("cloth-only cache fingerprints colliders", shape.session_uid in cache.collider_prints(rt))
scene.waifu_physics.simulate = False

# Deleted or edited topology cannot leave dangling RNA references in the frame handlers.
scene.waifu_physics.use_cache = False
scene.waifu_physics.simulate = True
bpy.data.objects.remove(obj, do_unlink=True)
scene.frame_set(2)
check("deleting a running cloth is safe", live.last_error is None, live.last_error)
scene.waifu_physics.simulate = False
addon.unregister()
check("unregister removes cloth RNA and handlers", not hasattr(bpy.types.Object, "waifu_cloth"))
finish()
