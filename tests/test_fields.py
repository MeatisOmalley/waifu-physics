"""Blender Force Fields: the port of Blender's effectors against Blender's own particles, the chains pushed by a
field only when the group asks, old setups' simple force and scene wind converted, and the force type header."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Euler, Vector

addon = fresh_import()
addon.register()
from waifu_physics.data import legacy, serialize
from waifu_physics.runtime import fields as scene_fields
from waifu_physics.solver import fields as field_math

live = sys.modules["waifu_physics.runtime.live"]
scene = bpy.context.scene


def clear():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)


# --- the port gives what Blender gives a particle at the same place and speed
scene.render.fps, scene.render.fps_base = 25, 1.0


def blender_says(kind, point, v0, rotation, **settings):
    """One particle, no gravity, one Euler step: the field's acceleration as Blender applies it."""
    clear()
    mesh = bpy.data.meshes.new("m")
    mesh.from_pydata([(0, 0, 0)], [], [])
    emitter = bpy.data.objects.new("e", mesh)
    scene.collection.objects.link(emitter)
    emitter.location = point
    ps = emitter.modifiers.new("p", "PARTICLE_SYSTEM").particle_system.settings
    ps.count, ps.frame_start, ps.frame_end, ps.lifetime = 1, 1, 1, 100
    ps.emit_from, ps.use_emit_random, ps.normal_factor, ps.object_align_factor = "VERT", False, 0.0, v0
    ps.physics_type, ps.integrator, ps.timestep, ps.subframes = "NEWTON", "EULER", 1.0 / 25.0, 0
    ps.use_adaptive_subframes, ps.mass, ps.effector_weights.gravity = False, 1.0, 0.0
    bpy.ops.object.effector_add(type=kind, location=(0.2, -0.1, 0.3), rotation=rotation)
    field = bpy.context.object
    field.field.shape = "POINT"
    for name, value in settings.items():
        setattr(field.field, name, value)
    states = []
    for frame in (1, 2):
        scene.frame_set(frame)
        p = emitter.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0].particles[0]
        states.append((np.array(p.location), np.array(p.velocity)))
    measured = (states[1][1] - states[0][1]) * 25.0
    ported = field_math.accelerations([scene_fields.spec_of(field)], [states[0][0]], [states[0][1]], 2)[0]
    return measured, ported


cases = {
    "Wind, plane shape, turned, with flow": ("WIND", dict(strength=5.0, shape="PLANE", flow=0.7)),
    "Force, tube falloff with a radial limit": ("FORCE", dict(strength=5.0, falloff_type="TUBE", falloff_power=1.0,
                                                               radial_falloff=2.0, use_radial_max=True,
                                                               radial_max=6.0)),
    "Vortex, plane shape": ("VORTEX", dict(strength=5.0, shape="PLANE")),
    "Harmonic with a rest length and damping": ("HARMONIC", dict(strength=5.0, harmonic_damping=0.3,
                                                                 rest_length=1.0)),
    "Magnetic": ("MAGNET", dict(strength=5.0)),
    "Turbulence": ("TURBULENCE", dict(strength=5.0, size=1.3)),
    "Drag": ("DRAG", dict(linear_drag=1.0, quadratic_drag=0.5)),
    "Wind with noise (Blender's random numbers)": ("WIND", dict(strength=5.0, noise=2.0, seed=7)),
    "Force, cone falloff, positive Z only": ("FORCE", dict(strength=5.0, falloff_type="CONE", falloff_power=2.0,
                                                           z_direction="POSITIVE")),
}
for label, (kind, settings) in cases.items():
    measured, ported = blender_says(kind, (1.0, 2.0, 3.0), (0.3, -0.2, 0.5), (0.5, 0.2, 0.1), **settings)
    error = float(np.linalg.norm(measured - ported)) / max(float(np.linalg.norm(measured)), 1e-6)
    check(f"{label}: the port matches Blender", error < 1e-5 and np.linalg.norm(measured) > 1e-4,
          (measured, ported))
scene.frame_set(1)
clear()

# --- the chains feel the fields only when the group asks
scene.frame_start, scene.frame_end = 1, 30
data = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
parent, head = None, Vector((0, 0, 1.0))
for i in range(4):
    bone = data.edit_bones.new(f"b{i}")
    bone.head, bone.tail, bone.parent = head, head + Vector((0, 0, -0.2)), parent
    parent, head = bone, bone.tail.copy()
bpy.ops.object.mode_set(mode="OBJECT")
group = rig.waifu_physics.groups.add()
group.roots.add().name = "b1"
group.gravity_scale = 0.0
bpy.ops.object.effector_add(type="WIND", location=(-2, 0, 0.5), rotation=(0.0, math.pi / 2, 0.0))   # blows +X
wind = bpy.context.object
wind.field.strength, wind.field.shape = 4.0, "PLANE"
bpy.context.view_layer.objects.active = rig


def tip_x_after(frames=15):
    scene.waifu_physics.simulate = False
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    for frame in range(2, 2 + frames):
        scene.frame_set(frame)
    x = rig.pose.bones["b3"].tail.x
    scene.waifu_physics.simulate = False
    return x


still = tip_x_after()
check("with Blender Force Fields off, a Wind field does nothing", abs(still) < 1e-4, still)
group.use_force_fields = True
blown = tip_x_after()
check("with it on, the chains are blown along the wind", blown > 0.02, blown)
group.force_field_strength = 0.0
check("Strength 0 turns every field off for the group", abs(tip_x_after()) < 1e-4)
group.force_field_strength = 1.0
empty = bpy.data.collections.new("Only these")
scene.collection.children.link(empty)
group.force_field_collection = empty
check("a collection with no fields in it: nothing blows", abs(tip_x_after()) < 1e-4)
empty.objects.link(wind)
check("... and with the field in it, it blows again", tip_x_after() > 0.02)
group.force_field_collection = None
bpy.ops.object.effector_add(type="TEXTURE", location=(0, 0, 0))
unfelt = bpy.context.object
check("types the chains cannot feel are listed but give no spec",
      unfelt in scene_fields.field_objects(scene) and scene_fields.spec_of(unfelt) is None)
bpy.context.view_layer.objects.active = rig

# --- Cache All disables what the simulation does not read while it bakes; every field is read, not only wind
wind.field.strength = 0.0
bpy.ops.object.effector_add(type="FORCE", location=(-1.0, 0.0, 0.5))    # pushes the chain along +X
force = bpy.context.object
force.field.strength = 4.0
bpy.context.view_layer.objects.active = rig
played = tip_x_after(29)
scene.frame_set(1)
bpy.ops.waifu_physics.cache_toggle()
scene.frame_set(30)
baked = rig.pose.bones["b3"].tail.x
check("a Force field pushes the chain in a bake as it does in live playback",
      played > 0.02 and 0.8 < baked / played < 1.25, (played, baked))
check("... and the bake leaves every object as it found it",
      not any(obj.hide_viewport for obj in scene.objects))
bpy.ops.waifu_physics.cache_toggle()
scene.waifu_physics.simulate = False
bpy.data.objects.remove(force)
wind.field.strength = 4.0
bpy.context.view_layer.objects.active = rig

# --- old setups: the simple force becomes a Push, scene wind and Kawaii's Wind force turn on force fields
old = rig.waifu_physics.groups.add()
old.simple_external_force = (0.0, 30.0, 0.0)
old.world_space_simple_external_force = False
old.enable_wind, old.wind_scale = True, 0.6
kawaii_wind = old.forces.add()
kawaii_wind.kind = "WIND"
kept = old.forces.add()
kept.kind, kept.name = "GRAVITY", "Gravity"
check("an old group converts", legacy.upgrade_group(old))
push = next((f for f in old.forces if f.kind == "BASIC"), None)
check("the simple force is now a Push force with the same push and space",
      push is not None and tuple(push.direction) == (0.0, 30.0, 0.0) and push.space == "COMPONENT")
check("... and the old field is cleared", tuple(old.simple_external_force) == (0.0, 0.0, 0.0))
check("scene wind became Blender Force Fields at its scale",
      old.use_force_fields and not old.enable_wind and abs(old.force_field_strength - 0.6) < 1e-6)
check("Kawaii's field-reading Wind force is gone, other forces stay",
      [f.kind for f in old.forces] == ["GRAVITY", "BASIC"], [f.kind for f in old.forces])
check("converting again changes nothing", not legacy.upgrade_group(old))
saved = serialize.group_to_dict(old)
saved["settings"].update(simple_external_force=[1.0, 0.0, 0.0], enable_wind=True)
fresh = rig.waifu_physics.groups.add()
serialize.group_from_dict(fresh, saved)
check("a setup saved with a simple force converts as it loads",
      sum(f.kind == "BASIC" for f in fresh.forces) == 2 and fresh.use_force_fields and not fresh.enable_wind)

# --- the type header
force = group.forces.add()
force.kind, force.name = "BASIC", "Push"
check("a Push force shows as Push", force.category == "PUSH")
force.category = "WIND"
check("choosing Wind makes it Procedural Wind, and its default name follows",
      force.kind == "PROCEDURAL_WIND" and force.name == "Wind")
force.name = "My Gust"
force.category = "CURVE"
check("a name the user chose stays", force.kind == "CURVE" and force.name == "My Gust")
check("the header's type is not saved as a setting", "category" not in serialize.settings_to_dict(force))
check("the Add menu is registered", hasattr(bpy.types, "WAIFU_PHYSICS_MT_force_add"))

addon.unregister()
finish()
