"""The Waifu Physics sidebar tab in the 3D viewport."""
import bpy

from ..data import colliders, curves
from ..data import links as chain_links
from ..data.props import FORCE_CHANNELS, FORCE_KINDS
from ..solver import native


def _tree_armatures(context):
    """The armatures the Groups box lists. Selected Only: every selected armature, groups or not.
    Otherwise: every armature with a group, plus the active one, so it can be given its first."""
    active = context.object if context.object is not None and context.object.type == "ARMATURE" else None
    if context.scene.waifu_physics.selected_only:
        found = [o for o in context.selected_objects if o.type == "ARMATURE"]
        if active is not None and active in found:          # the active one first
            found.remove(active)
            found.insert(0, active)
        return found
    found = [o for o in context.scene.objects if o.type == "ARMATURE" and len(o.waifu_physics.groups)]
    if active is not None and active not in found:
        found.insert(0, active)
    return found


def _plural(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


class WAIFU_PHYSICS_UL_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "name", text="", emboss=False, icon="BONE_DATA")
        count = row.row()
        count.alignment = "RIGHT"
        count.enabled = False
        count.label(text=_plural(len(item.roots), "chain"))


class WAIFU_PHYSICS_PT_main(bpy.types.Panel):
    """Simulate, the cache, the Groups box, then (under a divider) the Physics and Colliders tabs and their
    pages. The group settings subpanels follow on the Physics tab."""
    bl_idname = "WAIFU_PHYSICS_PT_main"
    bl_label = "Waifu Physics"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Waifu Physics"

    def draw(self, context):
        from ..runtime import live
        layout = self.layout
        scene, settings = context.scene, context.scene.waifu_physics
        from . import manager
        # With no group in the scene there is nothing to simulate: greyed, but a simulation or cache left on
        # (say, every group deleted) can still be switched off.
        grouped = any(obj.type == "ARMATURE" and len(obj.waifu_physics.groups) for obj in scene.objects)
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.enabled = grouped or settings.simulate
        row.prop(settings, "simulate", toggle=True, icon="PHYSICS")
        row.operator("waifu_physics.reset", text="", icon="FILE_REFRESH")
        current = live._runtimes.get(scene.as_pointer())
        span = current.cached_range() if live.is_cached(scene) else None
        row = layout.row(align=True)
        cache = row.row(align=True)
        cache.enabled = grouped or span is not None
        cache.operator("waifu_physics.cache_toggle", text=f"Cached  {span[0]}-{span[1]}" if span else "Cache",
                       icon="DISK_DRIVE", depress=span is not None)
        row.operator("waifu_physics.bake", icon="KEYFRAME")          # greyed until cached (its poll)
        if span is not None and current.outdated:                   # kept, not thrown away: say so
            warning = layout.column(align=True)
            note = warning.box().row()
            note.alert = True
            note.label(text=f"Outdated because {current.outdated}.", icon="ERROR")
            warning.operator("waifu_physics.cache_all", text="Recache", icon="FILE_REFRESH")
        if native.backend() is native.step_numpy:
            layout.label(text=f"Using the slower numpy step: {native.reason()}", icon="INFO")
        showing = manager.is_open(context.area)
        row = layout.row()
        row.enabled = grouped or showing          # an open manager can always be closed
        row.operator("waifu_physics.chain_manager", icon="OUTLINER", depress=showing,
                     text="Hide Chain Manager" if showing else "Chain Manager")
        if showing and not context.space_data.show_gizmo:
            note = layout.row()
            note.alert = True
            note.label(text="Turn on Gizmos in the viewport header to see it.", icon="ERROR")
        _draw_groups(layout, context)
        _draw_group_tools(layout, context, span)


def _outside_buttons(layout):
    """A box, and a column to its right, outside it, for its + and - (+ above -), as Blender's lists have."""
    row = layout.row()
    return row.box(), row.column(align=True)


def _draw_groups(layout, context):
    """The Groups box: a folder per armature (as _tree_armatures lists them) holding its groups."""
    settings = context.scene.waifu_physics
    obj = context.object
    armatures = _tree_armatures(context)
    box, side = _outside_buttons(layout)
    side.enabled = obj is not None and obj.type == "ARMATURE"
    side.operator("waifu_physics.group_new", text="", icon="ADD")
    side.operator("waifu_physics.group_remove", text="", icon="REMOVE")
    header = box.row(align=True)
    header.label(text="Groups")
    header.prop(settings, "selected_only", text="", icon="RESTRICT_SELECT_OFF")
    if not armatures:
        box.label(text="Select an armature to see its groups." if settings.selected_only
              else "No armature has a group yet.",
                  icon="INFO")
        return
    for number, rig in enumerate(armatures):
        if number:
            box.separator(factor=0.8)            # a little space between armatures
        row = box.row(align=True)
        row.prop(rig.waifu_physics, "expanded", text="", emboss=False,
                 icon="DOWNARROW_HLT" if rig.waifu_physics.expanded else "RIGHTARROW")
        left = row.row(align=True)
        left.alignment = "LEFT"
        name = left.operator("waifu_physics.armature_activate", text=rig.name, icon="ARMATURE_DATA",
                             emboss=rig == obj, depress=rig == obj)
        name.armature = rig.name
        if not rig.waifu_physics.expanded:
            continue
        if not len(rig.waifu_physics.groups):
            if rig == obj:                       # one hint, for the armature being edited
                hint = box.row()
                hint.enabled = False
                hint.label(text="Select bones and click +.")
            continue
        lists = box.row()
        lists.active = rig == obj                # other armatures' lists are dimmed: not being edited
        lists.template_list("WAIFU_PHYSICS_UL_groups", rig.name, rig.waifu_physics, "groups", rig.waifu_physics,
                            "active_group", rows=len(rig.waifu_physics.groups),
                            maxrows=len(rig.waifu_physics.groups))
    if obj is not None and obj.type == "ARMATURE":
        row = layout.row(align=True)
        row.operator("waifu_physics.group_add", icon="PLUS")
        row.operator("waifu_physics.exclude", icon="X")


def _draw_group_tools(layout, context, span):
    """Under the Groups box: the active armature's warnings and the group's preset. Its settings follow in the
    subpanels."""
    obj = context.object
    if obj is None or obj.type != "ARMATURE" or not len(obj.waifu_physics.groups):
        return
    from ..data import bone_refs
    lost = bone_refs.missing(obj)
    if lost:
        box = layout.box().column(align=True)
        box.alert = True
        row = box.split(factor=0.65)
        row.label(text=f"{len(lost)} missing bone{'' if len(lost) == 1 else 's'}", icon="ERROR")
        row.operator("waifu_physics.bones_clean_up")
        box.label(text=", ".join(lost[:3]) + (" ..." if len(lost) > 3 else ""))
    group = obj.waifu_physics.groups[min(obj.waifu_physics.active_group, len(obj.waifu_physics.groups) - 1)]
    constrained = chain_links.constrained_bones(obj, group)
    if constrained:
        box = layout.box().column(align=True)
        box.alert = True
        box.label(text=f"{_plural(len(constrained), 'bone')} with constraints: physics can't move them.",
                  icon="ERROR")
        box.label(text="Start the group below them: " + ", ".join(constrained[:3])
                       + (" ..." if len(constrained) > 3 else ""))
    row = layout.row(align=True)
    row.enabled = span is None                  # a cache plays what was simulated: settings wait for it to clear
    from ..data import presets
    row.menu("WAIFU_PHYSICS_MT_presets", text=presets.matching(group) or "Preset", icon="PRESET")
    row.operator("waifu_physics.preset_save", text="", icon="ADD")
    current = presets.matching(group)
    trash = row.row(align=True)
    trash.enabled = bool(current) and current in presets.user_presets()     # built-in presets stay
    trash.operator("waifu_physics.preset_delete", text="", icon="TRASH").name = current or ""
    row.operator("waifu_physics.group_copy", text="", icon="COPYDOWN")
    row.operator("waifu_physics.group_paste", text="", icon="PASTEDOWN")


class _GroupPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Waifu Physics"
    bl_parent_id = "WAIFU_PHYSICS_PT_main"

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "ARMATURE" and len(obj.waifu_physics.groups) > 0

    @staticmethod
    def group(context):
        found = context.object.waifu_physics
        return found.groups[min(found.active_group, len(found.groups) - 1)]

    def draw(self, context):
        """Every settings panel: greyed while a cache plays, since changing settings could not change what it
        shows. The panels draw their own contents in draw_settings."""
        from ..runtime import live
        cached = live.is_cached(context.scene)
        if cached and self.bl_idname == "WAIFU_PHYSICS_PT_settings":
            self.layout.label(text="Cached. Clear the cache to edit.", icon="LOCKED")
        self.layout.enabled = not cached
        self.draw_settings(context)


# Settings shown the intuitive way round (display only: Kawaii's values are stored and exported).
_SHOWN = {"stiffness": "stiffness_level", "damping": "damping_level", "world_damping_location": "world_location_inertia",
          "world_damping_rotation": "world_rotation_inertia"}
_SHORT_LABELS = {"world_damping_location": "Moving", "world_damping_rotation": "Rotating"}


class WAIFU_PHYSICS_PT_settings(_GroupPanel, bpy.types.Panel):
    bl_idname = "WAIFU_PHYSICS_PT_settings"
    bl_label = "Physics"

    def draw_settings(self, context):
        from .selection import groups_of_selected
        group = self.group(context)
        layout = self.layout
        shared = groups_of_selected(context)
        if len(shared) > 1:
            note = layout.row()
            note.enabled = False
            note.label(text=f"Changes apply to all {len(shared)} selected groups.", icon="INFO")
        column = layout.column(align=True)
        for name in curves.CURVED:
            if name == "radius":                         # a collision setting: on the Colliders tab
                continue
            if name == "world_damping_location":
                heading = column.split(factor=0.5)
                heading.label(text="Inertia")
            elif name == "limit_angle":                  # the same gap after the Inertia pair, one row
                column.split(factor=0.5).label(text="")
            _setting_row(column, group, name)
        layout.separator()
        column = layout.column()
        column.use_property_split = True
        column.use_property_decorate = False
        column.prop(group, "use_scene_gravity")
        if group.use_scene_gravity:
            column.prop(group, "gravity_scale")
        else:
            column.prop(group, "gravity")
            column.prop(group, "use_world_space_gravity")


def _setting_row(column, group, name):
    """One curved setting: its label, its value (shown the intuitive way round), its curve switch, and the curve
    when switched on."""
    shown = _SHOWN.get(name, name)
    split = column.split(factor=0.5, align=True)
    label = split.row()
    label.alignment = "RIGHT"
    label.label(text=_SHORT_LABELS.get(name, group.bl_rna.properties[shown].name))
    row = split.row(align=True)
    row.prop(group, shown, text="")
    row.prop(group, f"use_{name}_curve", text="", icon="FCURVE")
    if getattr(group, f"use_{name}_curve"):
        node = curves.node(group, name, create=False)
        if node is not None:
            box = column.box()
            box.label(text=("Scales Kawaii's value along the chain, root to tip" if name in _SHOWN
                            else "Along the chain, root to tip"), icon="IPO_LINEAR")
            box.template_curve_mapping(node, "mapping")


class WAIFU_PHYSICS_PT_advanced(bpy.types.Panel):
    """Always shown: the solver's settings are the scene's, for every armature. The active group's advanced
    settings follow when there is one, greyed while a cache plays (as _GroupPanel greys the others)."""
    bl_idname = "WAIFU_PHYSICS_PT_advanced"
    bl_label = "Advanced"
    bl_options = {"DEFAULT_CLOSED"}
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Waifu Physics"
    bl_parent_id = "WAIFU_PHYSICS_PT_main"

    def draw(self, context):
        from ..runtime import live
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.waifu_physics
        layout.label(text="Solver")
        layout.prop(settings, "target_framerate")
        live_playback = layout.column(heading="Live Playback")
        live_playback.prop(settings, "fixed_substepping")
        live_playback.prop(settings, "fast_evaluation")
        if _GroupPanel.poll(context):
            group_settings = layout.column()
            group_settings.enabled = not live.is_cached(context.scene)
            group_settings.separator()
            self.draw_group(group_settings, _GroupPanel.group(context))

    @staticmethod
    def draw_group(layout, group):
        layout.label(text="Chain Shape")
        layout.prop(group, "dummy_bone_length")
        layout.prop(group, "bone_subdivision_count")
        sub = layout.column()
        sub.active = group.bone_subdivision_count > 0
        sub.prop(group, "bone_subdivision_collision_only")
        sub.prop(group, "bone_subdivision_densify_by_radius")
        layout.prop(group, "planar_constraint")
        if len(group.excluded):
            listed = layout.column(align=True)
            listed.label(text="Excluded Bones")
            for bone in group.excluded:
                listed.label(text=bone.name, icon="X")
        layout.separator()
        layout.label(text="Group")
        layout.prop(group, "legacy_gravity")
        layout.prop(group, "teleport_distance")
        layout.prop(group, "teleport_rotation")
        layout.prop(group, "warm_up_frames")
        layout.separator()
        layout.label(text="Setup File")
        files = layout.row(align=True)
        files.use_property_split = False
        files.operator("waifu_physics.setup_export", text="Export Setup", icon="EXPORT")
        files.operator("waifu_physics.setup_import", text="Import Setup", icon="IMPORT")
        _caption(layout, "Save or load this armature's groups", "and colliders as a file.")


class WAIFU_PHYSICS_UL_links(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.label(text=f"{item.bone_a}  –  {item.bone_b}", icon="CONSTRAINT_BONE")
        row.prop(item, "compliance", text="")
        row.operator("waifu_physics.link_remove", text="", icon="X", emboss=False).index = index


def _eye(layout, what, shown):
    """A panel header's show/hide button: raised, with a border, and never lit, its eye open or shut."""
    layout.operator("waifu_physics.toggle_shown", text="", icon="HIDE_OFF" if shown else "HIDE_ON").what = what


class WAIFU_PHYSICS_PT_links(_GroupPanel, bpy.types.Panel):
    bl_idname = "WAIFU_PHYSICS_PT_links"
    bl_label = "Links"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header_preset(self, context):
        _eye(self.layout, "LINKS", context.scene.waifu_physics.show_links)

    def draw_settings(self, context):
        group = self.group(context)
        layout = self.layout
        row = layout.column(align=True)
        row.scale_y = 1.2
        row.operator("waifu_physics.link_bones", text="Link Selected Bones", icon="LINKED")
        row.operator("waifu_physics.link_chains", text="Link Whole Chains", icon="OUTLINER_DATA_ARMATURE")
        _caption(layout, "Rings are left open. To close one,", "link its two end chains.")
        layout.template_list("WAIFU_PHYSICS_UL_links", "", group, "links", group, "active_link", rows=3)
        row = layout.row(align=True)
        row.operator("waifu_physics.links_clear", icon="TRASH")
        layout.use_property_split = True
        layout.prop(group, "compliance")
        layout.prop(group, "auto_child_dummy_links")
        layout.prop(group, "bridge_count")
        sub = layout.column()
        sub.active = group.bridge_count > 0
        sub.prop(group, "bridge_feedback")
        # Rarely touched (Kawaii's 1 and 1 suit most): which wins where links and colliders disagree.
        header, body = layout.panel("waifu_physics_link_iterations", default_closed=True)
        header.label(text="Iterations")
        if body is not None:
            body.prop(group, "iterations_before_collision", text="Before Collision")
            body.prop(group, "iterations_after_collision", text="After Collision")
            _caption(body, "More before holds the spacing firmer.", "Fewer after lets colliders win.")


def _dim(layout, text, icon="NONE"):
    """Secondary text: a hint, or what a row says after its name."""
    row = layout.row()
    row.enabled = False
    row.label(text=text, icon=icon)


def _folder_row(layout, owner, prop, text, icon, count, armature=None, active=False):
    """A folder's row: its arrow, its name (an armature's activates it), and how many it holds, dimmed."""
    row = layout.row(align=True)
    shown = getattr(owner, prop)
    row.prop(owner, prop, text="", emboss=False, icon="DOWNARROW_HLT" if shown else "RIGHTARROW")
    name = row.row(align=True)
    name.alignment = "LEFT"
    if armature is not None:
        name.operator("waifu_physics.armature_activate", text=text, icon=icon, emboss=active,
                      depress=active).armature = armature.name
    else:
        name.label(text=text, icon=icon)
    tally = row.row()
    tally.alignment = "RIGHT"
    tally.enabled = False
    tally.label(text=str(count) if count else "")
    return shown


def _collider_row(layout, obj, picked, chosen, indent=True):
    """A collider's row, in its folder: on or off, its shape and name (picks it; lit when picked, lit dimmer when
    only selected, as the outliner shades them), and the bone it is on, dimmed (red if a chain simulates it)."""
    found = colliders.values(obj) or {}
    row = layout.row(align=True)
    if indent:
        row.separator(factor=1.6)                # inside its folder
    row.prop(obj.waifu_physics_collider, "enabled", text="")
    name = row.row(align=True)
    name.alignment = "LEFT"                      # as the armatures' names: lit like them when picked
    lit = obj == picked or obj in chosen
    name.emboss = "NORMAL" if lit else "NONE"
    name.active = obj == picked or not lit
    name.operator("waifu_physics.collider_pick", text=obj.name, depress=lit,
                  icon=colliders.SHAPE_ICONS.get(found.get("Shape"), "MESH_UVSPHERE")).name = obj.name
    where = row.row()
    where.alignment = "RIGHT"
    if colliders.on_simulated_bone(obj):
        where.alert = True
        where.label(text=colliders.short_name(obj.parent_bone), icon="ERROR")
    elif obj.parent is not None:
        where.enabled = False
        where.label(text=colliders.short_name(obj.parent_bone) if obj.parent_type == "BONE" else obj.parent.name)


SCENE_SHAPES = (("Plane", "Ground"), ("Sphere", "Sphere"), ("Capsule", "Capsule"), ("Box", "Box"))


class WAIFU_PHYSICS_PT_colliders(bpy.types.Panel):
    """Colliders, under Physics. Shown whatever is active (a body with no groups can carry colliders, and the
    scene's need no armature); the eye in the header shows or hides them and the chains' collision spheres."""
    bl_idname = "WAIFU_PHYSICS_PT_colliders"
    bl_label = "Colliders"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Waifu Physics"
    bl_parent_id = "WAIFU_PHYSICS_PT_main"

    def draw_header_preset(self, context):
        settings = context.scene.waifu_physics
        _eye(self.layout, "COLLIDERS", settings.show_colliders)

    def draw(self, context):
        _draw_colliders(self.layout, context)


def _draw_colliders(layout, context):
    """The Colliders panel, in sections that keep their places whatever is picked: Armature Colliders (with a
    group's armature active, first its chains' collision radius; then the Collider Type and its buttons, for
    the active bone or Generate for the selected ones; then a folder per armature with colliders), Scene
    Colliders (a button per shape, then the scene's colliders), Collides Against (with a group's armature
    active: what its chains collide with), and last the picked collider's settings, with its Remove. Both
    collider sections put their buttons above their colliders. Picking a collider here is selecting it in the
    viewport, and the other way round."""
    from ..runtime import live
    settings = context.scene.waifu_physics
    layout = layout.column()
    layout.enabled = not live.is_cached(context.scene)
    index = settings.active_collider
    picked = bpy.data.objects[index] if 0 <= index < len(bpy.data.objects) else None
    picked = picked if colliders.is_collider(picked) else None
    chosen = colliders.chosen(context.scene, bpy.data.objects.get(settings.last_collider))
    obj = context.object
    groups = obj.waifu_physics.groups if obj is not None and obj.type == "ARMATURE" else ()
    group = groups[min(obj.waifu_physics.active_group, len(groups) - 1)] if len(groups) else None

    header, body = layout.panel("waifu_physics_colliders", default_closed=False)
    header.label(text="Armature Colliders", icon="ARMATURE_DATA")
    if body is not None:
        if group is not None:
            _chain_radius(body, context, group)
            body.separator()
        _add_to_bones(body, context, settings)
        armatures = colliders.listed_armatures(context)
        tree = body.box().column(align=True)
        active = colliders.armature_of(context)
        for number, rig in enumerate(armatures):
            if number:
                tree.separator(factor=0.6)
            found = colliders.all_of(rig)
            if _folder_row(tree, rig.waifu_physics, "colliders_expanded", rig.name, "ARMATURE_DATA", len(found),
                           rig, rig == active):
                for collider in found:
                    _collider_row(tree, collider, picked, chosen)
                if not found:
                    hint = tree.row()
                    hint.separator(factor=1.6)
                    _dim(hint, "None yet. Generate them above.")
        if not armatures:
            _dim(tree, "Select an armature to see its colliders.")

    header, body = layout.panel("waifu_physics_scene_colliders", default_closed=False)
    header.label(text="Scene Colliders", icon="SCENE_DATA")
    if body is not None:
        grid = body.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True)
        grid.scale_y = 1.3
        for shape, label in SCENE_SHAPES:
            grid.operator("waifu_physics.scene_collider_add", text=label,
                          icon=colliders.SHAPE_ICONS[shape]).shape = shape
        _caption(body, "They spawn at the 3D cursor.")
        found = colliders.scene_colliders(context.scene, enabled_only=False)
        if found:
            listed = body.box().column(align=True)
            for collider in found:
                _collider_row(listed, collider, picked, chosen, indent=False)

    if group is not None:
        header, body = layout.panel("waifu_physics_collides_against", default_closed=False)
        header.label(text="Collides Against", icon="PHYSICS")
        if body is not None:
            _collides_against(body, obj, group)

    if picked is not None:
        header, body = layout.panel("waifu_physics_collider", default_closed=False)
        header.label(text=picked.name, icon=colliders.SHAPE_ICONS.get(
            (colliders.values(picked) or {}).get("Shape"), "MESH_UVSPHERE"))
        trash = header.row()
        trash.alignment = "RIGHT"
        if len(chosen) > 1:                      # the trash deletes them all: say how many
            _dim(trash, f"{len(chosen)} selected")
        trash.operator("waifu_physics.collider_remove", text="", icon="TRASH", emboss=False).name = picked.name
        if body is not None:
            _collider_settings(body, picked)


def _add_to_bones(layout, context, settings):
    """Making armature colliders: the Collider Type, then a button for the active bone and Generate (or
    Regenerate) for the selected ones, with how they are fitted."""
    split = layout.split(factor=0.5, align=True)       # lined up with Chain Collision Radius above
    label = split.row()
    label.alignment = "RIGHT"
    label.label(text="Collider Type")
    split.prop(settings, "collider_shape", text="")
    bones = (context.selected_pose_bones or ()) if context.mode == "POSE" else ()
    again = any(colliders.has_collider(bone.id_data, bone.name) for bone in bones)
    # Bones a group simulates take no collider (it would chase the chain it pushes): with one selected, the
    # buttons wait.
    grouped = [bone for bone in bones if bone.name in colliders.simulated_bones(bone.id_data)]
    row = layout.row()
    row.scale_y = 1.3
    row.enabled = not grouped
    row.operator("waifu_physics.collider_add", text="Active Bone", icon="ADD").shape = settings.collider_shape
    generate = row.row()
    generate.active_default = True
    generate.operator("waifu_physics.colliders_from_bones", text="Regenerate" if again else "Generate",
                      icon="FILE_REFRESH" if again else "MOD_PHYSICS").shape = settings.collider_shape
    if grouped:
        _caption(layout, "Bones in a group can't have colliders.", "Select body bones instead.")
    elif bones:
        _caption(layout, "Each selected bone gets one collider,", "fitted to the skin weighted to it.")
    else:
        _caption(layout, "Select bones in Pose Mode. Each one gets", "a collider fitted to the skin weighted to it.")


def _chain_radius(layout, context, group):
    """How big the active group's chain points are when they collide: the radius, with its curve (the spheres
    the viewport draws while colliders show)."""
    from .selection import groups_of_selected
    _setting_row(layout.column(align=True), group, "radius")
    shared = groups_of_selected(context)
    if len(shared) > 1:
        _dim(layout, f"Changes apply to all {len(shared)} selected groups.", "INFO")


def _collides_against(layout, armature, group):
    """What the active group's chains collide with: every collider, or the colliders of chosen armatures
    (its own and the one it hangs from, until the list is edited) and, if wanted, the scene's."""
    layout.prop(group, "use_all_colliders")
    if not group.use_all_colliders:
        layout.label(text="Collides with colliders on these armatures:")
        column = layout.column(align=True)
        if group.custom_collider_sets or len(group.collider_sets):
            for index, item in enumerate(group.collider_sets):
                row = column.row(align=True)
                row.prop(item, "armature", text="")
                row.operator("waifu_physics.collider_set_remove", text="", icon="X").index = index
            if not len(group.collider_sets):
                _dim(column, "No armatures yet. Add one below.", "BLANK1")
        else:                                               # the defaults, until the list is edited
            for index, source in enumerate(colliders.default_sources(armature)):
                row = column.row(align=True)
                row.label(text="Its own armature" if source == armature else "The armature it hangs from",
                          icon="ARMATURE_DATA")
                row.operator("waifu_physics.collider_set_remove", text="", icon="X").index = index
        layout.operator("waifu_physics.collider_set_add", text="Add Armature", icon="ADD")
        layout.prop(group, "use_scene_colliders")


def _caption(layout, *lines):
    """Dimmed help under a section's controls, a line at a time (labels do not wrap)."""
    column = layout.column(align=True)
    column.enabled = False
    column.scale_y = 0.8
    for line in lines:
        column.label(text=line)


def _collider_settings(layout, obj):
    """The picked collider: its shape and sizes, where it hangs, and a warning if its bone is simulated."""
    found = colliders.values(obj)
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    if found is not None:
        md, ident = colliders.input_path(obj, "Shape")
        col.prop(getattr(md.properties.inputs, ident), "value", text="Shape")
        names = {"Box": ("Extent",), "Plane": ("Radius",), "Capsule": ("Radius", "Length"),
                 "Tapered Capsule": ("Radius", "Radius 1", "Length")}
        for name in names.get(found["Shape"], ("Radius",)):
            md, ident = colliders.input_path(obj, name)
            col.prop(getattr(md.properties.inputs, ident), "value", text=name)
    if obj.parent is not None and obj.parent_type == "BONE" and obj.parent_bone:
        _dim(layout, f"Attached to the bone {obj.parent_bone}.", "BONE_DATA")
    elif obj.parent is not None:
        _dim(layout, f"Attached to {obj.parent.name}.", "OBJECT_DATA")
    else:
        _dim(layout, "A scene collider, on no armature.", "SCENE_DATA")
    if colliders.on_simulated_bone(obj):
        warning = layout.column(align=True)
        warning.alert = True
        warning.label(text="Its bone is in a chain, so it chases", icon="ERROR")
        warning.label(text="the chain it pushes. Move it higher up.")


def _note(layout, text, icon="INFO"):
    _dim(layout, text, icon)


def _curve_box(layout, owner, setting, label):
    node = curves.node(owner, setting, create=False)
    if node is not None:
        box = layout.box()
        box.label(text=label, icon="IPO_LINEAR")
        box.template_curve_mapping(node, "mapping")


def _filter_row(layout, force, collection, label, target):
    row = layout.row(align=True)
    names = [item.name for item in getattr(force, collection)]
    if names:
        shown = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
    else:
        shown = "every bone" if collection == "apply_bones" else "none"
    row.label(text=f"{label}: {shown}")
    op = row.operator("waifu_physics.force_filter", text="", icon="RESTRICT_SELECT_OFF")
    op.target, op.clear = target, False
    op = row.operator("waifu_physics.force_filter", text="", icon="X")
    op.target, op.clear = target, True


class WAIFU_PHYSICS_UL_forces(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        kind_icon = next(entry[3] for entry in FORCE_KINDS if entry[0] == item.kind)
        row.prop(item, "name", text="", emboss=False, icon=kind_icon)


def _pair_row(layout, label, owner, *names):
    """Values sharing one label, laid out as Blender's property split lays out one: the label right-aligned in
    the same 40% (so it lines up with the rows around it and shortens as the sidebar narrows)."""
    split = layout.split(factor=0.4, align=True)
    left = split.row()
    left.alignment = "RIGHT"
    left.label(text=label)
    row = split.row(align=True)
    for name in names:
        row.prop(owner, name, text="")


class WAIFU_PHYSICS_PT_forces(_GroupPanel, bpy.types.Panel):
    bl_idname = "WAIFU_PHYSICS_PT_forces"
    bl_label = "Forces and Wind"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_settings(self, context):
        from ..runtime import fields as scene_fields
        group = self.group(context)
        layout = self.layout

        # Blender's force fields, felt as Blender computes them
        layout.prop(group, "use_force_fields")
        if group.use_force_fields:
            col = layout.column()
            col.use_property_split = True
            col.use_property_decorate = False
            col.prop(group, "force_field_collection")
            col.prop(group, "force_field_strength")
            found = scene_fields.field_objects(context.scene, group.force_field_collection)
            felt = [obj for obj in found if scene_fields.spec_of(obj) is not None]
            if not found:
                row = col.row()
                row.label(text="The scene has no force fields yet.", icon="INFO")
                row.operator("waifu_physics.wind_field_add", text="", icon="FORCE_WIND")
            elif len(felt) < len(found):
                note = col.row()
                note.enabled = False
                note.label(text=f"{len(found) - len(felt)} of {len(found)} fields are types the chains can't feel",
                           icon="INFO")
        layout.separator()

        # The group's own forces
        row = layout.row()
        row.template_list("WAIFU_PHYSICS_UL_forces", "", group, "forces", group, "active_force", rows=3)
        column = row.column(align=True)
        column.menu("WAIFU_PHYSICS_MT_force_add", text="", icon="ADD")
        column.operator("waifu_physics.force_remove", text="", icon="REMOVE")
        if not len(group.forces):
            return
        force = group.forces[min(group.active_force, len(group.forces) - 1)]
        box = layout.box()
        header = box.row()
        header.scale_y = 1.2
        header.prop(force, "category", text="")                # what the force is, as its header
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        kind = force.kind
        if kind == "BASIC":
            col.prop(force, "space")
            col.prop(force, "direction", text="Push")
            col.prop(force, "interval", text="Pulse Every")
        elif kind == "GRAVITY":
            col.label(text="Direction")                          # a heading, as the Physics panel's Inertia
            col.prop(force, "override_direction", text="Custom")
            sub = col.column(align=True)
            sub.active = force.override_direction
            for index, axis in enumerate("XYZ"):
                sub.prop(force, "direction", index=index, text=axis)
        elif kind == "CURVE":
            col.prop(force, "space")
            col.prop(force, "amplitude")
            col.prop(force, "duration")
            col.prop(force, "time_scale")
            col.prop(force, "evaluate")
            if force.evaluate != "SINGLE":
                col.prop(force, "substeps")
            for axis, channel in zip("XYZ", FORCE_CHANNELS):
                _curve_box(box, force, channel, f"{axis} over time, -1 to 1")
        elif kind == "PROCEDURAL_WIND":
            col.operator_menu_enum("waifu_physics.wind_preset", "preset", text="Preset", icon="PRESET")
            col.prop(force, "space")
            col.prop(force, "direction")
            winds = col.column(align=True)
            winds.prop(force, "constant", text="Steady")
            for label, amount, period in (("Sway", "sway", "sway_period"), ("Ripples", "ripple", "ripple_period"),
                                          ("Flutter", "random", "random_period")):
                _pair_row(winds, label, force, amount, period)
            _pair_row(winds, "Gusts", force, "cycle_min", "cycle_max")
            _pair_row(winds, "Gust Period", force, "cycle_period")
            col.prop(force, "show_advanced")
            if force.show_advanced:
                for name in ("noise_angle", "noise_period", "time_scale", "sway_phase", "ripple_phase",
                             "ripple_delay", "cycle_phase", "seed"):
                    col.prop(force, name)
        if kind != "PROCEDURAL_WIND":
            _pair_row(col, "Strength" if kind == "GRAVITY" else "Random Scale", force, "random_min", "random_max")
        box.prop(force, "use_rate_curve", icon="FCURVE")
        if force.use_rate_curve:
            _curve_box(box, force, "rate", "Along the chain, root to tip")
        _filter_row(box, force, "apply_bones", "Only", "APPLY")
        _filter_row(box, force, "ignore_bones", "Ignore", "IGNORE")


class WAIFU_PHYSICS_UL_sync(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="CON_TRACKTO")
        row.label(text=f"{len(item.targets)} target{'s' if len(item.targets) != 1 else ''}")


class WAIFU_PHYSICS_UL_sync_targets(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.label(text=item.bone, icon="BONE_DATA")
        row.prop(item, "include_children", text="", icon="OUTLINER_OB_ARMATURE")
        row.prop(item, "use_rate_curve", text="", icon="FCURVE")


class WAIFU_PHYSICS_PT_sync(_GroupPanel, bpy.types.Panel):
    bl_idname = "WAIFU_PHYSICS_PT_sync"
    bl_label = "Sync Bones"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_settings(self, context):
        group = self.group(context)
        layout = self.layout
        layout.label(text="Chains follow a bone, as a skirt follows a thigh.", icon="INFO")
        row = layout.row()
        row.template_list("WAIFU_PHYSICS_UL_sync", "", group, "sync_bones", group, "active_sync", rows=2)
        column = row.column(align=True)
        column.operator("waifu_physics.sync_add", text="", icon="ADD")
        column.operator("waifu_physics.sync_remove", text="", icon="REMOVE")
        if not len(group.sync_bones):
            return
        sync = group.sync_bones[min(group.active_sync, len(group.sync_bones) - 1)]
        box = layout.box()
        box.prop_search(sync, "bone", context.object.data, "bones", icon="BONE_DATA")
        box.label(text="Targets (bones of this group):")
        row = box.row()
        row.template_list("WAIFU_PHYSICS_UL_sync_targets", "", sync, "targets", sync, "active_target", rows=2)
        column = row.column(align=True)
        column.operator("waifu_physics.sync_target_add", text="", icon="ADD")
        column.operator("waifu_physics.sync_target_remove", text="", icon="REMOVE")
        if len(sync.targets):
            target = sync.targets[min(sync.active_target, len(sync.targets) - 1)]
            if target.use_rate_curve:
                _curve_box(box, target, "rate", f"{target.bone} to its tip")
        col = box.column()
        col.use_property_split = True
        col.prop(sync, "global_scale")
        row = col.row(align=True)
        row.prop(sync, "direction_x", text="X")
        row.prop(sync, "direction_y", text="Y")
        row.prop(sync, "direction_z", text="Z")
        col.prop(sync, "use_distance_curve")
        if sync.use_distance_curve:
            col.prop(sync, "distance")
            _curve_box(box, sync, "distance", "By the source's movement, 0 to Distance")
        col.prop(sync, "attenuation")
        sub = col.column()
        sub.active = sync.attenuation
        sub.prop(sync, "inner_radius")
        sub.prop(sync, "outer_radius")
        sub.prop(sync, "max_attenuation")


CLASSES = (WAIFU_PHYSICS_UL_groups, WAIFU_PHYSICS_UL_links, WAIFU_PHYSICS_UL_forces, WAIFU_PHYSICS_UL_sync, WAIFU_PHYSICS_UL_sync_targets, WAIFU_PHYSICS_PT_main,
           WAIFU_PHYSICS_PT_settings, WAIFU_PHYSICS_PT_colliders, WAIFU_PHYSICS_PT_links, WAIFU_PHYSICS_PT_forces, WAIFU_PHYSICS_PT_sync,
           WAIFU_PHYSICS_PT_advanced)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
