"""Mesh cloth setup. Lengths are Blender units; the solver receives centimetres."""
import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import PropertyGroup

from .props import WaifuPhysicsColliderSet, WaifuPhysicsBoneName

VALUE_NAMES = ("enabled", "max_distance", "density", "min_mass", "gravity_scale", "damping", "local_damping",
               "linear_velocity_scale", "angular_velocity_scale", "iterations", "max_iterations", "substeps",
               "edge_stiffness", "bending_stiffness", "area_stiffness", "tether_stiffness", "tether_scale",
               "anim_drive_stiffness", "anim_drive_damping", "thickness", "friction", "teleport_distance",
               "use_all_colliders", "use_scene_colliders", "custom_collider_sets")


def changed(self, context):
    from ..runtime import live
    live.mark_dirty()
    live.invalidate(reason="cloth settings changed")
    obj = self.id_data
    if obj.type == "MESH" and not self.enabled:
        from ..cloth import display
        display.clear(obj)


def unit(name, default, **kwargs):
    return FloatProperty(name=name, default=default, min=0, max=1, update=changed, **kwargs)


class WaifuClothSetup(PropertyGroup):
    show_settings: BoolProperty(name="Cloth Settings", default=False)
    enabled: BoolProperty(name="Simulate as cloth", default=False, update=changed)
    max_distance: FloatProperty(name="Max Distance", default=0.4, min=0, subtype="DISTANCE", update=changed)
    density: FloatProperty(name="Density (kg/m²)", default=0.35, min=0.000001, update=changed)
    min_mass: FloatProperty(name="Min Particle Mass (kg)", default=0.0001, min=0.000001, precision=6, update=changed)
    gravity_scale: FloatProperty(name="Gravity Scale", default=1, min=0, update=changed)
    damping: unit("Damping", 0.01)
    local_damping: unit("Local Damping", 0)
    linear_velocity_scale: unit("Linear Velocity Scale", 0.75)
    angular_velocity_scale: unit("Angular Velocity Scale", 0.75)
    edge_stiffness: unit("Edge Stiffness", 1)
    bending_stiffness: unit("Bending Stiffness", 1)
    area_stiffness: unit("Area Stiffness", 1)
    tether_stiffness: unit("Tether Stiffness", 1)
    tether_scale: FloatProperty(name="Tether Scale", default=1, min=0.01, update=changed)
    anim_drive_stiffness: unit("Shape Hold", 1)
    anim_drive_damping: unit("Shape Hold Damping", 1)
    thickness: FloatProperty(name="Collision Thickness", default=0.01, min=0, subtype="DISTANCE", update=changed)
    friction: unit("Friction", 0.8)
    iterations: IntProperty(name="Iterations", default=1, min=1, max=100, update=changed)
    max_iterations: IntProperty(name="Max Iterations", default=10, min=1, max=100, update=changed)
    substeps: IntProperty(name="Substeps", default=1, min=1, max=32, update=changed,
                          description="The highest cloth substep setting applies to the shared scene solver")
    teleport_distance: FloatProperty(name="Teleport Distance", default=1, min=0, subtype="DISTANCE", update=changed)
    use_all_colliders: BoolProperty(name="Every Collider", default=True, update=changed)
    use_scene_colliders: BoolProperty(name="Scene Colliders", default=True, update=changed)
    custom_collider_sets: BoolProperty(default=False, update=changed)
    collider_sets: CollectionProperty(type=WaifuPhysicsColliderSet)
    override_armature: PointerProperty(type=bpy.types.Object, options={"HIDDEN"})
    override_groups: CollectionProperty(type=WaifuPhysicsBoneName, options={"HIDDEN"})


def register():
    bpy.utils.register_class(WaifuClothSetup)
    bpy.types.Object.waifu_cloth = PointerProperty(type=WaifuClothSetup)


def unregister():
    del bpy.types.Object.waifu_cloth
    bpy.utils.unregister_class(WaifuClothSetup)
