"""Add/Regenerate accept collider object selections as well as pose bones."""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

addon = fresh_import()
addon.register()
from waifu_physics.data import colliders


def rig(name):
    obj = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for name in ('hips', 'leg'):
        b = obj.data.edit_bones.new(name)
        b.head, b.tail = (0, 0, 1), (0, 0, 2)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def select(objects, active=None):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active or (objects[-1] if objects else None)


a, b = rig('A'), rig('B')
first = colliders.add(a, 'hips', 'Sphere')
second = colliders.add(b, 'hips', 'Sphere')
unselected = colliders.add(a, 'leg', 'Box')
select([first])
check('Add and Regenerate enabled with a collider active in Object Mode',
      bpy.ops.waifu_physics.collider_add.poll() and bpy.ops.waifu_physics.colliders_from_bones.poll())
check('Add uses selected collider parent bone', bpy.ops.waifu_physics.collider_add(shape='Capsule') == {'FINISHED'}
      and len([c for c in colliders.all_of(a) if c.parent_bone == 'hips']) == 2)
duplicate = next(c for c in colliders.all_of(a) if c not in (first, unselected))
select([first, duplicate, second])
targets = colliders.generation_bones(bpy.context)
check('multiple colliders deduplicate by armature and bone',
      {(pb.id_data.name, pb.name) for pb in targets} == {('A', 'hips'), ('B', 'hips')} and len(targets) == 2)
check('Add prefers active collider owner', colliders.generation_bones(bpy.context, active_only=True)[0].id_data == b)
result = bpy.ops.waifu_physics.colliders_from_bones(shape='Tapered Capsule')
made = [c for arm in (a, b) for c in colliders.all_of(arm) if c.parent_bone == 'hips']
check('Regenerate replaces selected bone colliders across armatures', result == {'FINISHED'} and len(made) == 2
      and all(colliders.values(c)['Shape'] == 'Tapered Capsule' for c in made))
check('unselected bone collider unchanged', unselected in colliders.all_of(a)
      and colliders.values(unselected)['Shape'] == 'Box')
check('replacements remain selected and active', set(bpy.context.selected_objects) == set(made)
      and bpy.context.object in made and bpy.ops.waifu_physics.colliders_from_bones.poll())
bpy.ops.waifu_physics.colliders_from_bones(shape='Sphere')
check('regeneration can be repeated directly', all(colliders.values(c)['Shape'] == 'Sphere'
      for c in bpy.context.selected_objects) and len(bpy.context.selected_objects) == 2)

scene_col = colliders.add_to_scene('Plane')
select([scene_col])
check('scene-only collider selection does not enable bone fitting', not bpy.ops.waifu_physics.collider_add.poll()
      and not bpy.ops.waifu_physics.colliders_from_bones.poll())
select([a])
check('stored collider cannot enable commands without actual collider/bone selection',
      not bpy.ops.waifu_physics.collider_add.poll() and not bpy.ops.waifu_physics.colliders_from_bones.poll())
bpy.ops.object.mode_set(mode='POSE')
for bone in a.pose.bones:
    bone.select = bone.name == 'leg'
a.data.bones.active = a.data.bones['leg']
check('pose bone selection still enables both commands', bpy.ops.waifu_physics.collider_add.poll()
      and bpy.ops.waifu_physics.colliders_from_bones.poll())
check('pose selection regenerates the selected bone only', bpy.ops.waifu_physics.colliders_from_bones(shape='Capsule') == {'FINISHED'}
      and colliders.values(next(c for c in colliders.all_of(a) if c.parent_bone == 'leg'))['Shape'] == 'Capsule')
bpy.ops.object.mode_set(mode='OBJECT')
finish()
