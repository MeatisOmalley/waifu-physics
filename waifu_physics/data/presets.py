"""Group presets: starting points for the chains most rigs have.

A preset is a partial settings dict (serialize.settings_to_dict): what it
names is set, the rest of the group is left alone, and every curve is turned
off. Lengths are Blender units for a rig about 1.6 units tall.
"""
import json
import math
import os

from . import curves, serialize

_NO_CURVES = {f"use_{setting}_curve": False for setting in curves.CURVED}

PRESETS = {
    "HAIR": ("Hair", "Light strands: loose, quick to settle, riding along with the head", {
        "damping": 0.1, "stiffness": 0.05, "world_damping_location": 0.8, "world_damping_rotation": 0.8,
        "radius": 0.01, "limit_angle": 0.0,
        "bone_subdivision_count": 0, "bridge_count": 0, "planar_constraint": "NONE",
    }),
    "SKIRT": ("Skirt", "Panels hanging from the hips: heavier, held in shape, colliding as one surface", {
        "damping": 0.2, "stiffness": 0.1, "world_damping_location": 0.6, "world_damping_rotation": 0.6,
        "radius": 0.02, "limit_angle": 0.0,
        "compliance": "LEATHER", "auto_child_dummy_links": True, "bridge_count": 1, "bridge_feedback": 1.0,
        "bone_subdivision_count": 0, "planar_constraint": "NONE",
    }),
}
ITEMS = [(key, label, description) for key, (label, description, _values) in PRESETS.items()]


def apply(group, key):
    serialize.settings_from_dict(group, {**_NO_CURVES, **PRESETS[key][2]})
    group.preset_name = PRESETS[key][0]


# --------------------------------------------------------------------------- the user's own presets
# Saved as the group's settings and curves (what Copy Settings copies), one JSON file each, in the extension's own
# user folder, so every file can use them and Blender removes them with the extension. The group's on/off switch and
# what it collides with are not settings a preset should set.
FOLDER = "presets/waifu_physics/groups"         # loaded from a checkout rather than installed: Blender's presets
_NOT_PRESET = {"enabled", "use_all_colliders", "use_scene_colliders", "custom_collider_sets"}


def folder(create=False):
    import bpy
    package = __package__.rsplit(".", 1)[0]    # the extension's module: bl_ext.<repository>.waifu_physics
    try:
        return bpy.utils.extension_path_user(package, path="presets", create=create)
    except ValueError:                          # not installed as an extension (a development checkout)
        return bpy.utils.user_resource("SCRIPTS", path=FOLDER, create=create)


def user_presets():
    """{name: path} of the user's presets, by name."""
    found = {}
    path = folder()
    if path and os.path.isdir(path):
        for entry in sorted(os.listdir(path), key=str.lower):
            if entry.lower().endswith(".json"):
                found[entry[:-5]] = os.path.join(path, entry)
    return found


def _file_name(name):
    return "".join("_" if c in '<>:"/\\|?*' else c for c in name).strip(" .") or "Preset"


def save_user(group, name):
    """Save the group's settings as a preset of this name, replacing one with the same name."""
    data = json.loads(serialize.settings_text(group))
    for key in _NOT_PRESET:
        data["settings"].pop(key, None)
    name = _file_name(name)
    with open(os.path.join(folder(create=True), name + ".json"), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    group.preset_name = name
    return name


def delete_user(name):
    path = user_presets().get(name)
    if path is not None:
        os.remove(path)
        return True
    return False


_cache = {}                # path -> (modified time, data): the panel asks which preset matches on every redraw


def _read(path):
    stamp = os.path.getmtime(path)
    found = _cache.get(path)
    if found is None or found[0] != stamp:
        with open(path, encoding="utf-8") as handle:
            found = _cache[path] = (stamp, json.load(handle))
    return json.loads(json.dumps(found[1]))            # a copy: callers edit it


def apply_user(group, name):
    data = _read(user_presets()[name])
    for key in _NOT_PRESET:
        data["settings"].pop(key, None)
    serialize.paste(group, json.dumps(data))
    group.preset_name = name


# --------------------------------------------------------------------------- which preset a group still matches

def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _matches(group, current, settings, saved_curves=None):
    if not all(name in current and _same(current[name], value) for name, value in settings.items()):
        return False
    for setting, data in (saved_curves or {}).items():
        now = serialize.curve_to_dict(group, setting)
        if now is None or not _same(now["points"][:], data["points"]):
            return False
    return True


def matching(group):
    """The name of the preset the group's settings still match, or None once one of them has changed.
    The preset applied last is tried first; the others after, for a group set to one by hand."""
    current = serialize.settings_to_dict(group)                 # read once, compared with each preset
    candidates = [(label, lambda values=values: _matches(group, current, {**_NO_CURVES, **values}))
                  for label, _description, values in PRESETS.values()]
    for name, path in user_presets().items():
        def user(path=path):
            try:
                data = _read(path)
            except (OSError, ValueError):
                return False
            settings = {k: v for k, v in data.get("settings", {}).items() if k not in _NOT_PRESET}
            return _matches(group, current, settings, data.get("curves"))
        candidates.append((name, user))
    candidates.sort(key=lambda item: item[0] != group.preset_name)
    for name, test in candidates:
        if test():
            return name
    return None


# Kawaii's own Procedural Wind presets (KawaiiPhysicsWindPresetDataAsset.cpp, GetDefaultPresets), in its
# units: forces in centimetres a second, periods in seconds, angles in degrees. Its forces act every step
# and build up like velocity, so small numbers push harder than they look.
WIND_PRESETS = {
    "BREEZE": ("Breeze", dict(constant=2.0, sway=1.0, sway_period=3.0, ripple=1.0, ripple_period=1.2,
                              ripple_delay=120.0, cycle_min=0.6, cycle_max=1.0, cycle_period=20.0,
                              random=0.5, random_period=0.8, noise_angle=5.0)),
    "STRONG": ("Strong", dict(constant=8.0, sway=4.0, sway_period=1.6, ripple=3.0, ripple_period=0.5,
                              ripple_delay=120.0, cycle_min=0.7, cycle_max=1.3, cycle_period=12.0,
                              random=2.0, random_period=0.5, noise_angle=10.0)),
    "STORM": ("Storm", dict(constant=15.0, sway=10.0, sway_period=0.9, ripple=8.0, ripple_period=0.3,
                            ripple_delay=180.0, cycle_min=0.5, cycle_max=1.6, cycle_period=7.0,
                            random=6.0, random_period=0.3, noise_angle=20.0)),
}
WIND_ITEMS = [(key, label, f"Kawaii's {label} wind preset") for key, (label, _values) in WIND_PRESETS.items()]
_WIND_FORCES = ("constant", "sway", "ripple", "random")
_WIND_ANGLES = ("ripple_delay", "noise_angle")


def apply_wind(force, key, cm_per_unit):
    """Set a Procedural Wind force to one of Kawaii's presets, converted to Blender units."""
    for name, value in WIND_PRESETS[key][1].items():
        if name in _WIND_FORCES:
            value = value / cm_per_unit
        elif name in _WIND_ANGLES:
            value = math.radians(value)
        setattr(force, name, value)
