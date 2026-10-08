"""Saved/undo output is recoverable; linked objects cannot share cloth state."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import
import bpy
import numpy as np

addon = fresh_import()
addon.register()
from waifu_physics.cloth import convention, display
from waifu_physics.runtime import live

scene = bpy.context.scene
mesh = bpy.data.meshes.new("Cloth")
mesh.from_pydata([(0, 0, 1), (0.1, 0, 1), (0, 0.1, 1)], [], [(0, 1, 2)])
obj = bpy.data.objects.new("Cloth", mesh)
scene.collection.objects.link(obj)
original = np.array([v.co[:] for v in mesh.vertices])
convention.ensure_attribute(mesh).data.foreach_set("color_srgb", np.array([
    [0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float32).ravel())
obj.waifu_cloth.enabled = True
for name in ("edge_stiffness", "bending_stiffness", "area_stiffness", "tether_stiffness"):
    setattr(obj.waifu_cloth, name, 0)
bpy.context.view_layer.update()
scene.frame_set(1)
scene.waifu_physics.simulate = True
for frame in range(2, 5):
    scene.frame_set(frame)
bpy.context.view_layer.update()
check("fixture has simulated display before saving", obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[1].co.z < 0.99)
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "cloth.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    scene, obj = bpy.context.scene, bpy.data.objects["Cloth"]
    bpy.context.view_layer.update()
    check("loading discards saved cloth output", not any(item.value for item in obj.data.attributes[display.ACTIVE].data))
    scene.waifu_physics.simulate = False
    bpy.context.view_layer.update()
    restored = np.array([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
    check("stopping immediately after load shows skin", np.allclose(restored, original))
    display.write(obj, original + (0, 0, 1))
    live._undone(scene)
    check("undo recovery clears restored cloth output", not any(item.value for item in obj.data.attributes[display.ACTIVE].data))

linked = bpy.data.objects.new("Linked", obj.data)
scene.collection.objects.link(linked)
try:
    display.ensure(obj)
    rejected = False
except ValueError as error:
    rejected = "single-user" in str(error)
check("linked duplicates cannot share cloth output", rejected)
check("rejecting linked data preserves source geometry", np.array_equal(original, np.array([v.co[:] for v in obj.data.vertices])))
addon.unregister()
finish()
