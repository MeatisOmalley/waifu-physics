"""An animation render without the cache simulates live, and each frame shows that frame's pose."""
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
scene = bpy.context.scene

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)
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
scene.frame_start, scene.frame_end = 1, 12

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

scene.waifu_physics.use_cache = False
scene.frame_set(1)
scene.waifu_physics.simulate = True
expected = {}
for frame in range(1, 13):
    scene.frame_set(frame)
    expected[frame] = world_to_camera_view(scene, camera, marker.matrix_world.translation)
scene.frame_set(1)
folder = tempfile.mkdtemp()
scene.render.filepath = os.path.join(folder, "f####")
bpy.ops.render.render(animation=True)
for frame in (2, 6, 12):
    image = bpy.data.images.load(os.path.join(folder, f"f{frame:04d}.png"))
    pixels = np.array(image.pixels[:]).reshape(image.size[1], image.size[0], 4)
    ys, xs = np.nonzero((pixels[:, :, 0] > 0.5) & (pixels[:, :, 1] < 0.3))
    found = (xs.mean() / image.size[0], ys.mean() / image.size[1]) if len(xs) else None
    check(f"live animation render frame {frame} shows that frame's pose, not the one before",
          found is not None and abs(found[0] - expected[frame].x) < 0.01 and abs(found[1] - expected[frame].y) < 0.01,
          (found, tuple(expected[frame])[:2]))

scene.waifu_physics.simulate = False
addon.unregister()
finish()
