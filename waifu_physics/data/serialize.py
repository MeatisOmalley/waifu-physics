"""Waifu Physics setups as plain data: versioned dicts to save, share, paste and export.

An armature's setup is its groups (settings, chains, links, collider sets,
curves) and its colliders; the scene's stepping settings ride along, since a
setup was tuned at them. Everything is JSON-safe. Lengths are Blender units in
the armature's space, angles radians, as in the properties.

Curves carry both their control points, to edit again in Blender, and the
dense linear keys the solver simulated with (curves.py), which is what an
exporter hands a game's FRichCurve.

A collider's transform is stored relative to its bone's head frame, which
does not depend on the pose, and restored as a bone-parented object.
"""
import bpy
from mathutils import Matrix

from . import colliders, curves

FORMAT = "waifu_physics"
LEGACY_FORMATS = {"swish_physics"}      # setups saved before the add-on was renamed
VERSION = 1
KAWAII_COMMIT = "64cbc77ad4d75f6eb8c8f5673b4b4452f838ec21"
SCENE_SETTINGS = ("target_framerate", "fixed_substepping")
_NOT_SETTINGS = {"rna_type", "name", "curve_key", "active_link", "active_force", "active_sync", "active_target",
                 "show_advanced", "stiffness_level", "damping_level", "world_location_inertia", "world_rotation_inertia",
                 "show_chains", "preset_name", "category"}


def _plain(value):
    return list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value


def settings_names(group):
    """Every value property of a group: Kawaii's settings plus Waifu Physics' own switches."""
    return [p.identifier for p in group.bl_rna.properties
            if p.type in ("BOOLEAN", "INT", "FLOAT", "ENUM", "STRING") and p.identifier not in _NOT_SETTINGS]


def settings_to_dict(group):
    return {name: _plain(getattr(group, name)) for name in settings_names(group)}


def settings_from_dict(group, data):
    """Apply what the dict holds; names this version does not know are ignored."""
    known = set(settings_names(group))
    for name, value in data.items():
        if name in known and _plain(getattr(group, name)) != value:
            setattr(group, name, tuple(value) if isinstance(value, list) else value)


def curve_to_dict(group, setting, required=False):
    found = curves.node(group, setting, create=False)
    if found is None:
        return None
    points = found.mapping.curves[0].points
    sampled = curves.curve(group, setting, required)
    return {"points": [[p.location[0], p.location[1], p.handle_type] for p in points],
            "keys": sampled.keys() if sampled is not None else None}


def curve_from_dict(group, setting, data):
    mapping = curves.node(group, setting).mapping
    curve = mapping.curves[0]
    points = data["points"]
    while len(curve.points) > max(2, len(points)):
        curve.points.remove(curve.points[-1])
    while len(curve.points) < len(points):
        curve.points.new(0.0, 1.0)
    # Points keep themselves sorted by x; set them in order, left to right.
    for point, (x, y, handle) in zip(curve.points, sorted(points)):
        point.location = (x, y)
        point.handle_type = handle
    mapping.update()


FORCE_CURVES = ("rate", "force_x", "force_y", "force_z")


def _item_to_dict(item, curve_names, collections=()):
    """A force, sync bone or sync target: its values, its name lists and the curves it uses."""
    data = {"name": getattr(item, "name", ""), "settings": settings_to_dict(item)}
    for name in collections:
        data[name] = [entry.name for entry in getattr(item, name)]
    used = {}
    for setting in curve_names:
        required = not hasattr(item, f"use_{setting}_curve")
        if required or getattr(item, f"use_{setting}_curve"):
            found = curve_to_dict(item, setting, required)
            if found is not None:
                used[setting] = found
    data["curves"] = used
    return data


def _item_from_dict(item, data, collections=()):
    if "name" in data and hasattr(item, "name") and data["name"]:
        item.name = data["name"]
    settings_from_dict(item, data.get("settings", {}))
    for name in collections:
        for entry in data.get(name, ()):
            getattr(item, name).add().name = entry
    for setting, curve in data.get("curves", {}).items():
        curve_from_dict(item, setting, curve)


def forces_to_list(group):
    return [_item_to_dict(force, FORCE_CURVES if force.kind == "CURVE" else ("rate",),
                          ("apply_bones", "ignore_bones")) for force in group.forces]


def sync_to_list(group):
    out = []
    for sync in group.sync_bones:
        data = _item_to_dict(sync, ("distance",))
        data["targets"] = [_item_to_dict(target, ("rate",)) for target in sync.targets]
        out.append(data)
    return out


def group_to_dict(group):
    used = {setting: curve_to_dict(group, setting) for setting in curves.CURVED
            if getattr(group, f"use_{setting}_curve")}
    return {
        "name": group.name,
        "settings": settings_to_dict(group),
        "roots": [item.name for item in group.roots],
        "excluded": [item.name for item in group.excluded],
        "links": [{"bone_a": link.bone_a, "bone_b": link.bone_b, "compliance": link.compliance,
                   "exclude_from_subdivision": link.exclude_from_subdivision} for link in group.links],
        # The group's own armature is null, so a setup moved to another armature keeps its own colliders.
        "collider_sets": [None if item.armature == group.id_data else item.armature.name
                          for item in group.collider_sets if item.armature is not None],
        "curves": {setting: data for setting, data in used.items() if data is not None},
        "forces": forces_to_list(group),
        "sync_bones": sync_to_list(group),
        # Where each chain's tip point sits past its last bone: that bone's length. Blender knows it; Kawaii
        # (DummyBoneLength) must be told, per chain. Written for exporters, never read back.
        "tips": chain_tips(group),
    }


def chain_tips(group):
    """{last bone of each chain (every branch's end): its rest length, in Blender units}."""
    from .links import chain_subtree
    obj = group.id_data
    excluded = [item.name for item in group.excluded]
    found = {}
    for root in group.roots:
        names = chain_subtree(obj, root.name, excluded)
        for name in names:
            bone = obj.data.bones[name]
            if not any(child.name in names for child in bone.children):
                found[name] = bone.length
    return found


def group_from_dict(group, data):
    """Fill an empty group. Returns the collider set armatures it names that this file lacks."""
    group.name = data.get("name", group.name)
    settings_from_dict(group, data.get("settings", {}))
    if "use_all_colliders" in data.get("settings", {}):    # stored even when it is the default, so the
        group.use_all_colliders = data["settings"]["use_all_colliders"]   # upgrade below leaves it
    for name in data.get("roots", ()):
        group.roots.add().name = name
    for name in data.get("excluded", ()):
        group.excluded.add().name = name
    for found in data.get("links", ()):
        link = group.links.add()
        for key, value in found.items():
            setattr(link, key, value)
    missing = []
    for name in data.get("collider_sets", ()):
        armature = group.id_data if name is None else bpy.data.objects.get(name)
        if armature is None or armature.type != "ARMATURE":
            missing.append(name)
            continue
        group.collider_sets.add().armature = armature
    for setting, curve in data.get("curves", {}).items():
        if setting in curves.CURVED:
            curve_from_dict(group, setting, curve)
    for found in data.get("forces", ()):
        _item_from_dict(group.forces.add(), found, ("apply_bones", "ignore_bones"))
    for found in data.get("sync_bones", ()):
        sync = group.sync_bones.add()
        _item_from_dict(sync, found)
        for target in found.get("targets", ()):
            _item_from_dict(sync.targets.add(), target)
    from . import legacy
    legacy.upgrade_group(group)                # setups saved with the simple force or scene wind
    return missing


def settings_text(group):
    """A group's settings and curves, without its chains, as JSON for the clipboard."""
    import json
    found = group_to_dict(group)
    return json.dumps({"format": FORMAT, "version": VERSION, "settings": found["settings"], "curves": found["curves"]})


def paste(group, text):
    """Apply settings_text() to a group; raises ValueError for anything else."""
    import json
    data = json.loads(text)
    check(data)
    group_from_dict(group, {"settings": data.get("settings", {}), "curves": data.get("curves", {})})


def _bone_frame(obj):
    """The collider's transform in its bone's head frame (BONE parenting hangs it off the tail)."""
    length = obj.parent.data.bones[obj.parent_bone].length
    return Matrix.Translation((0.0, length, 0.0)) @ obj.matrix_parent_inverse @ obj.matrix_basis


def collider_to_dict(obj):
    return {"name": obj.name, "bone": obj.parent_bone, "enabled": obj.waifu_physics_collider.enabled,
            "shape": {name: _plain(value) for name, value in colliders.values(obj).items()},
            "matrix": [list(row) for row in _bone_frame(obj)]}


def collider_from_dict(armature, data):
    obj = colliders.add(armature, data["bone"], data["shape"].get("Shape", "Sphere"))
    obj.name = data.get("name", obj.name)
    obj.waifu_physics_collider.enabled = data.get("enabled", True)
    for name, value in data["shape"].items():
        if name != "Shape":
            colliders.set_value(obj, name, tuple(value) if isinstance(value, list) else value)
    length = armature.data.bones[data["bone"]].length
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Translation((0.0, -length, 0.0)) @ Matrix(data["matrix"])
    return obj


def armature_to_dict(obj, scene=None, include_colliders=True):
    data = {"format": FORMAT, "version": VERSION, "kawaii_commit": KAWAII_COMMIT,
            "armature": obj.name, "groups": [group_to_dict(group) for group in obj.waifu_physics.groups]}
    if scene is not None:
        data["scene"] = {name: getattr(scene.waifu_physics, name) for name in SCENE_SETTINGS}
    if include_colliders:
        data["colliders"] = [collider_to_dict(child) for child in obj.children if colliders.is_collider(child)
                             and child.parent_type == "BONE" and child.parent_bone in obj.data.bones]
    return data


def check(data):
    if not isinstance(data, dict) or data.get("format") not in {FORMAT, *LEGACY_FORMATS}:
        raise ValueError("not a Waifu Physics setup")
    if data.get("version", 0) > VERSION:
        raise ValueError(f"made by a newer Waifu Physics (setup version {data['version']}, this reads {VERSION})")


def armature_from_dict(obj, data, scene=None, replace=True, include_colliders=True):
    """Load a setup onto an armature. Returns warnings: bones, armatures the setup names but this file lacks."""
    check(data)
    from ..runtime import live
    warnings = []
    if replace:
        for group in obj.waifu_physics.groups:
            curves.remove_owned(group)
        obj.waifu_physics.groups.clear()
        if include_colliders and "colliders" in data:
            for child in [c for c in obj.children if colliders.is_collider(c)]:
                mesh = child.data
                bpy.data.objects.remove(child)
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
    bones = obj.data.bones
    for found in data.get("groups", ()):
        group = obj.waifu_physics.groups.add()
        for name in group_from_dict(group, found):
            warnings.append(f"collider set armature '{name}' not found")
        for name in [item.name for item in group.roots] + [item.name for item in group.excluded]:
            if name not in bones:
                warnings.append(f"group '{group.name}': bone '{name}' not found")
    if include_colliders:
        for found in data.get("colliders", ()):
            if found["bone"] not in bones:
                warnings.append(f"collider '{found['name']}': bone '{found['bone']}' not found")
                continue
            collider_from_dict(obj, found)
    if scene is not None:
        for name, value in data.get("scene", {}).items():
            if name in SCENE_SETTINGS:
                setattr(scene.waifu_physics, name, value)
    obj.waifu_physics.active_group = 0
    live.mark_dirty()
    return warnings
