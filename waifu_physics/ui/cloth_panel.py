"""Cloth authoring UI, also drawn by Workshop for its selected garment."""
import bpy
from bpy.props import IntProperty

from ..cloth import convention, native
from ..runtime import live


class WAIFU_CLOTH_OT_paint(bpy.types.Operator):
    bl_idname = "waifu_cloth.paint"
    bl_label = "Vertex Paint"
    bl_description = "Paint cloth: red simulates, green holds shape, blue rides; black anchors"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH" and context.object.waifu_cloth.enabled

    def execute(self, context):
        obj = context.object
        if obj.mode == "VERTEX_PAINT":
            bpy.ops.object.mode_set(mode="OBJECT")
            live.mark_dirty(context.scene)
            return {"FINISHED"}
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        convention.ensure_attribute(obj.data)
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        paint = context.tool_settings.vertex_paint
        unified = (paint.unified_paint_settings if hasattr(paint, "unified_paint_settings")
                   else context.tool_settings.unified_paint_settings)
        unified.use_unified_color = True
        unified.color = (1, 0, 0)
        if paint.brush is not None:
            paint.brush.color = (1, 0, 0)
        return {"FINISHED"}


class WAIFU_CLOTH_OT_collider_set(bpy.types.Operator):
    bl_idname = "waifu_cloth.collider_set"
    bl_label = "Edit Cloth Collider Sets"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def execute(self, context):
        props = context.object.waifu_cloth
        props.custom_collider_sets = True
        if self.index < 0:
            props.collider_sets.add()
        elif self.index < len(props.collider_sets):
            props.collider_sets.remove(self.index)
        live.mark_dirty(context.scene)
        return {"FINISHED"}


def draw_settings(layout, context, obj):
    props = obj.waifu_cloth
    header = layout.row()
    header.prop(props, "show_settings", text="Cloth Settings", emboss=False,
                icon="TRIA_DOWN" if props.show_settings else "TRIA_RIGHT")
    if not props.show_settings:
        return
    column = layout.column(align=True)
    column.use_property_split = True
    for name in ("max_distance", "density", "damping", "local_damping", "edge_stiffness", "bending_stiffness",
                 "area_stiffness", "tether_stiffness", "tether_scale", "anim_drive_stiffness", "anim_drive_damping",
                 "thickness", "friction", "gravity_scale", "linear_velocity_scale", "angular_velocity_scale",
                 "iterations", "max_iterations", "substeps", "teleport_distance"):
        column.prop(props, name)
    column.prop(props, "use_all_colliders")
    if not props.use_all_colliders:
        column.prop(props, "use_scene_colliders")
        for index, entry in enumerate(props.collider_sets):
            row = layout.row(align=True)
            row.prop(entry, "armature", text="")
            row.operator("waifu_cloth.collider_set", text="", icon="X").index = index
        layout.operator("waifu_cloth.collider_set", text="Add Collider Set", icon="ADD")
    if native.backend() is native.step_numpy:
        layout.label(text="Cloth uses numpy: " + str(native.reason()), icon="INFO")


class WAIFU_CLOTH_PT_cloth(bpy.types.Panel):
    bl_idname = "WAIFU_CLOTH_PT_cloth"
    bl_label = "Cloth"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Waifu Physics"
    bl_parent_id = "WAIFU_PHYSICS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        obj, layout = context.object, self.layout
        row = layout.row(align=True)
        row.prop(obj.waifu_cloth, "enabled", text="Simulate")
        buttons = row.row(align=True)
        buttons.enabled = obj.waifu_cloth.enabled
        buttons.operator("waifu_cloth.paint", text="Stop Painting" if obj.mode == "VERTEX_PAINT" else "Vertex Paint",
                         icon="VPAINT_HLT")
        if obj.waifu_cloth.enabled:
            draw_settings(layout, context, obj)


CLASSES = (WAIFU_CLOTH_OT_paint, WAIFU_CLOTH_OT_collider_set, WAIFU_CLOTH_PT_cloth)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
