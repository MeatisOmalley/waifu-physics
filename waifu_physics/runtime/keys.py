"""Waifu Physics takes over the chain bones' keyframes while it simulates.

Blender applies a bone's keyframes every time it evaluates a frame, after any
value Waifu Physics wrote before the evaluation. So while it simulates, Waifu Physics mutes
the F-curves of the chain bones' channels in the armature's action and samples
them itself (FCurve.evaluate, as Instant Physics does): Blender then keeps
what Waifu Physics writes before a frame -- a cached frame, or a live solve -- and
evaluates each frame once instead of twice.

Only curves in the armature's own action are taken over. A chain channel that
an NLA strip or a driver animates cannot be, and a rig with one keeps the
slower path (write after evaluation, evaluate again). An action blended at
less than full influence, or not replacing, is left alone the same way.

Muted curves are the user's data: they are unmuted whenever Waifu Physics stops
simulating, before every save (and muted again after), and on load if a
previous session did not get to (the armature remembers what Waifu Physics muted).
"""
import json

import bpy
import numpy as np

MARK = "waifu_physics_muted"       # object ID property: the curves Waifu Physics muted, to recover after a crash
LEGACY_MARK = "swish_muted"         # the same, from before the add-on was renamed
CHANNELS = {"location": 3, "rotation_quaternion": 4, "rotation_euler": 3, "rotation_axis_angle": 4, "scale": 3}


def _bone_path(path):
    """('bone name', 'channel') for a pose bone transform path, else None."""
    if not path.startswith('pose.bones["'):
        return None
    end = path.find('"]', 12)
    if end < 0:
        return None
    channel = path[end + 3:]
    return (path[12:end], channel) if channel in CHANNELS else None


def _action_curves(data):
    """F-curves of the armature's own action for its slot, every layer and strip."""
    action, slot = data.action, data.action_slot
    if action is None or slot is None:
        return []
    found = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot)
            if bag is not None:
                found += list(bag.fcurves)
    return found


def _layout(obj):
    """What the object's action curves were found in: the action, its slot, and each channel bag's curve count.
    If this changes, references to its F-curves may no longer be good."""
    data = obj.animation_data
    if data is None or data.action is None or data.action_slot is None:
        return None
    slot = data.action_slot
    return (data.action.as_pointer(), slot.handle,
            tuple(len(bag.fcurves) for layer in data.action.layers for strip in layer.strips
                  for bag in [strip.channelbag(slot)] if bag is not None))


class HeldCurves:
    """F-curves of an object's own action, held by what finds them again: (data path, array index). An F-curve is
    no ID, so no pointer property can hold one, and a Python reference to one outlives it (a deleted channel,
    another action; an undo drops the whole run). Before each use the action's layout (_layout) is compared with
    the one the curves were found in; on any change they are found again by path, and any gone are skipped
    (the depsgraph handler notices the change and rebuilds). Paths are read from here, never from a curve."""

    def __init__(self, obj, entries):
        """entries: (fcurve, data path, array index, payload); the payload is handed back with each curve."""
        self.obj = obj
        self.entries = [(path, index, payload) for _curve, path, index, payload in entries]
        self.paths = {(path, index) for path, index, _payload in self.entries}
        self._refs = [curve for curve, *_ in entries]
        self._layout = self.built = _layout(obj)

    def stale(self):
        """Has the action's layout changed since the curves were found: curves added or removed, another action
        or slot? Then what was worked out from them (which to take over, sample, or leave) must be again."""
        return _layout(self.obj) != self.built

    def __len__(self):
        return len(self.entries)

    def resolved(self):
        """[(fcurve, payload)] for the curves as they are now."""
        now = _layout(self.obj)
        if now != self._layout:
            data = self.obj.animation_data
            found = {(c.data_path, c.array_index): c for c in _action_curves(data)} if data is not None else {}
            self._refs = [found.get((path, index)) for path, index, _payload in self.entries]
            self._layout = now
        return [(curve, payload) for curve, (_p, _i, payload) in zip(self._refs, self.entries) if curve is not None]


class ChainKeys:
    """The chain bones' keyframe curves of one rig, which Waifu Physics samples while it owns them."""

    def __init__(self, rig):
        obj = rig.obj
        self.obj = obj
        self.curves = HeldCurves(obj, [])    # the chain curves taken over; payload (bone index, channel, index)
        self.ownable = True
        self.muted = False
        self.total = 0                     # chain curves in the action, muted or not: notices new keys
        self.user_muted = set()            # chain curves the user muted: left alone, and their channels at rest
        # Curves a previous run muted and never handed back (its references went stale: an undo, a changed
        # armature) would look muted by the user, and never be unmuted again: hand them back first.
        release_marked(obj)
        data = obj.animation_data
        if data is None:
            return
        chain = {rig.names[i] for i in np.flatnonzero(rig.chain)}
        owned = []
        for curve in _action_curves(data):
            found = _bone_path(curve.data_path)
            if found and found[0] in chain:
                self.total += 1
                if curve.mute:
                    self.user_muted.add((curve.data_path, curve.array_index))
                else:
                    owned.append((curve, curve.data_path, curve.array_index,
                                  (rig.index[found[0]], found[1], curve.array_index)))
        self.curves = HeldCurves(obj, owned)
        # Animation Waifu Physics cannot take over: NLA strips and drivers on chain channels, or an
        # action that does not simply replace.
        foreign = []
        for track in data.nla_tracks:
            if track.mute:
                continue
            for strip in track.strips:
                if strip.mute or strip.action is None:
                    continue
                slot = getattr(strip, "action_slot", None)
                for layer in strip.action.layers:
                    for s in layer.strips:
                        bag = s.channelbag(slot) if slot is not None else None
                        foreign += [c.data_path for c in (bag.fcurves if bag else ())]
        foreign += [d.data_path for d in data.drivers]
        if any(_bone_path(p) and _bone_path(p)[0] in chain for p in foreign):
            self.ownable = False
        if len(self.curves) and (data.action_influence < 1.0 or data.action_blend_type != "REPLACE"):
            self.ownable = False

    def count(self, rig):
        """Chain curves in the action now, muted or not."""
        data = self.obj.animation_data
        if data is None:
            return 0
        chain = {rig.names[i] for i in np.flatnonzero(rig.chain)}
        return sum(1 for c in _action_curves(data) if _bone_path(c.data_path)
                   and _bone_path(c.data_path)[0] in chain)

    def mutes_changed(self, rig):
        """Has a chain curve's mute been flipped since the curves were taken over? A curve the user unmuted
        must be taken over, one they muted let go; one of ours unmuted would be applied over the physics.
        Any of them: take the curves over again (Waifu Physics' own mute cannot show the user's)."""
        data = self.obj.animation_data
        if data is None:
            return bool(self.user_muted)
        chain = {rig.names[i] for i in np.flatnonzero(rig.chain)}
        owned = self.curves.paths if self.muted else set()
        user_muted, owned_unmuted = set(), False
        for curve in _action_curves(data):
            found = _bone_path(curve.data_path)
            if not found or found[0] not in chain:
                continue
            key = (curve.data_path, curve.array_index)
            if key in owned:
                owned_unmuted = owned_unmuted or not curve.mute
            elif curve.mute:
                user_muted.add(key)
        return owned_unmuted or user_muted != self.user_muted

    def mute(self):
        """Take the chain curves over. Only curves not already muted are touched."""
        if not self.ownable or not len(self.curves) or self.muted:
            return
        # The record comes first, and holds every curve muted: it, not the references kept here, is what hands
        # the curves back (release_marked), so a stale reference can never leave one muted.
        marked = set(_marked(self.obj)) | self.curves.paths
        self.obj[MARK] = json.dumps(sorted(marked))
        for curve, _payload in self.curves.resolved():
            curve.mute = True
        self.muted = True

    def unmute(self):
        """Hand the curves back: every one the armature's record names, found again by path."""
        if not self.muted:
            return
        try:
            release_marked(self.obj)
        except ReferenceError:              # the armature is gone, and its curves with it
            pass
        self.muted = False

    def sample(self, frame, buffers):
        """Write the sampled keyframe values at frame into per-bone channel buffers."""
        for curve, (bone, channel, index) in self.curves.resolved():
            buffers[channel][bone * CHANNELS[channel] + index] = curve.evaluate(frame)


OBJECT_CHANNELS = {"location": 3, "rotation_quaternion": 4, "rotation_euler": 3, "rotation_axis_angle": 4, "scale": 3}
OBJECT_BLOCKERS = ("delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")


def _foreign_paths(data):
    """Data paths something other than the action animates: unmuted NLA strips and drivers."""
    found = []
    for track in data.nla_tracks:
        if track.mute:
            continue
        for strip in track.strips:
            if strip.mute or strip.action is None:
                continue
            slot = getattr(strip, "action_slot", None)
            for layer in strip.action.layers:
                for s in layer.strips:
                    bag = s.channelbag(slot) if slot is not None else None
                    found += [c.data_path for c in (bag.fcurves if bag else ())]
    return found + [d.data_path for d in data.drivers]


class InputKeys:
    """The keys of what the chains hang from, sampled for the frame without being taken over, so the
    one-evaluation path reads this frame's input before Blender evaluates it (Blender still applies these
    curves itself: the bones and the armature must show). Covers the bones above the chains and those
    colliders hang from, and the armature object's transform, where nothing but the armature's own action
    moves them: a bone with a constraint, a driver or NLA animation, one an IK constraint can reach, and any
    bone below those keep Blender's last evaluation (a frame late), as does an object with a parent,
    constraints, drivers or NLA on its transform, or delta transforms."""

    def __init__(self, rig, wanted):
        obj = rig.obj
        blocked = set()
        for i, pb in enumerate(obj.pose.bones):
            for c in pb.constraints:
                if not c.enabled or c.influence <= 0:
                    continue
                blocked.add(i)
                if c.type in ("IK", "SPLINE_IK"):      # an IK chain moves the bones above its owner too
                    reach, bone = getattr(c, "chain_count", 0), pb.parent
                    depth = 1
                    while bone is not None and (reach == 0 or depth < reach):
                        blocked.add(rig.index[bone.name])
                        bone, depth = bone.parent, depth + 1
        data = obj.animation_data
        foreign = _foreign_paths(data) if data is not None else []
        blocked |= {rig.index[found[0]] for found in map(_bone_path, foreign) if found and found[0] in rig.index}
        whole = data is None or (data.action_influence >= 1.0 and data.action_blend_type == "REPLACE")
        own = _action_curves(data) if data is not None else []
        if not whole:
            blocked |= {rig.index[found[0]] for found in map(_bone_path, (c.data_path for c in own))
                        if found and found[0] in rig.index}
        ok = np.zeros(rig.count, dtype=bool)
        for i in range(rig.count):                     # parents first: a bone under a blocked one is out too
            p = rig.parents[i]
            ok[i] = i not in blocked and (p < 0 or ok[p])
        wanted = {int(i) for i in wanted if not rig.chain[i]}
        self.bones = np.array(sorted(i for i in wanted if ok[i]), dtype=int)
        self.complete = len(self.bones) == len(wanted)        # every bone asked for can be sampled
        chosen = {rig.names[i] for i in self.bones}
        sampled = []
        for curve in own:
            found = _bone_path(curve.data_path)
            if found and found[0] in chosen and not curve.mute:
                sampled.append((curve, curve.data_path, curve.array_index,
                                (rig.index[found[0]], found[1], curve.array_index)))
        self.curves = HeldCurves(obj, sampled)
        self.object_ok = (whole and obj.parent is None
                          and not any(c.enabled and c.influence > 0 for c in obj.constraints)
                          and not any(p in OBJECT_CHANNELS or p in OBJECT_BLOCKERS for p in foreign)
                          and not any(p.startswith("delta_") for p in (c.data_path for c in own))
                          and tuple(obj.delta_location) == (0, 0, 0) and tuple(obj.delta_scale) == (1, 1, 1)
                          and tuple(obj.delta_rotation_euler) == (0, 0, 0)
                          and tuple(obj.delta_rotation_quaternion) == (1, 0, 0, 0))
        self.object_curves = HeldCurves(obj, [(c, c.data_path, c.array_index, (c.data_path, c.array_index))
                                              for c in own if self.object_ok and c.data_path in OBJECT_CHANNELS
                                              and not c.mute])
        self.rig = rig
        self.total = self._count()

    def _count(self):
        """What the sampling was worked out from: the action and its curves (with their mute flags), and what
        decides which bones and whether the object can be sampled -- constraints, drivers, NLA, parent."""
        obj = self.rig.obj
        try:
            data = obj.animation_data
            curves = _action_curves(data) if data is not None else []
            return (data.action if data is not None else None, len(curves), tuple(c.mute for c in curves),
                    tuple((c.type, c.enabled, c.influence > 0) for pb in obj.pose.bones for c in pb.constraints),
                    tuple((c.type, c.enabled, c.influence > 0) for c in obj.constraints), obj.parent,
                    len(data.drivers) if data is not None else 0,
                    tuple((t.mute, len(t.strips)) for t in data.nla_tracks) if data is not None else ())
        except ReferenceError:
            return None

    def changed(self):
        """Curves added, removed or muted, another action, or constraints, drivers, NLA or the parent changed:
        what is sampled must be worked out again."""
        return self._count() != self.total

    def sample(self, frame, buffers):
        """This frame's keyed values into the bones' channel buffers."""
        for curve, (bone, channel, index) in self.curves.resolved():
            buffers[channel][bone * CHANNELS[channel] + index] = curve.evaluate(frame)

    def world(self, obj, frame):
        """The armature object's world matrix at frame, or None when it cannot be known before evaluation."""
        if not self.object_ok:
            return None
        from mathutils import Euler, Matrix, Quaternion, Vector
        values = {path: list(getattr(obj, path)) for path in OBJECT_CHANNELS}
        for curve, (path, index) in self.object_curves.resolved():
            values[path][index] = curve.evaluate(frame)
        mode = obj.rotation_mode
        if mode == "QUATERNION":
            rotation = Quaternion(values["rotation_quaternion"]).normalized()
        elif mode == "AXIS_ANGLE":
            angle, *axis = values["rotation_axis_angle"]
            rotation = Quaternion(Vector(axis), angle)
        else:
            rotation = Euler(values["rotation_euler"], mode)
        return Matrix.LocRotScale(Vector(values["location"]), rotation, Vector(values["scale"]))


def _marked(obj):
    """(data path, index) of the curves Waifu Physics muted on this object, from its record."""
    found = []
    for mark in (MARK, LEGACY_MARK):
        if mark in obj.keys():
            try:
                found += [tuple(entry) for entry in json.loads(obj[mark])]
            except (TypeError, ValueError):
                pass
    return found


def release_marked(obj):
    """Unmute the curves the object's record says Waifu Physics muted, found by path in its action, and forget
    them. Safe whatever Python references went stale, since it looks the curves up now."""
    wanted = set(_marked(obj))
    data = obj.animation_data
    if wanted and data is not None:
        for curve in _action_curves(data):
            if (curve.data_path, curve.array_index) in wanted:
                curve.mute = False
    for mark in (MARK, LEGACY_MARK):
        if mark in obj.keys():
            del obj[mark]


SETTINGS_ROOT = "waifu_physics."


class SettingKeys:
    """Keyframed physics settings of an armature's groups (and of their forces, sync bones and targets), sampled for
    the frame. Before Blender evaluates a frame the properties still hold last frame's values, so the
    one-evaluation path reads them through these (Sampled). Held by path (HeldCurves). Settings a driver or an NLA
    strip animates, or an action blended at less than full, cannot be known before evaluation: `blocked` says so,
    and live playback then evaluates first unless Fast Evaluation accepts the lag."""

    def __init__(self, obj):
        data = obj.animation_data
        entries = []
        self.blocked = False
        if data is not None:
            for curve in _action_curves(data):
                path = curve.data_path
                if not path.startswith(SETTINGS_ROOT) or curve.mute or "." not in path:
                    continue
                owner, attribute = path.rsplit(".", 1)
                try:                                  # the owner as read at run time names itself (path_from_id)
                    owner = obj.path_resolve(owner).path_from_id()
                except (ValueError, AttributeError):
                    continue
                entries.append((curve, path, curve.array_index, (owner, attribute, curve.array_index)))
            foreign = _foreign_paths(data)
            whole = data.action_influence >= 1.0 and data.action_blend_type == "REPLACE"
            self.blocked = any(p.startswith(SETTINGS_ROOT) for p in foreign) or bool(entries and not whole)
        self.curves = HeldCurves(obj, entries)

    def sample(self, frame):
        """{owner path: {attribute: {array index: value}}}: this frame's keyed values, as the curves give them."""
        found = {}
        for curve, (owner, attribute, index) in self.curves.resolved():
            found.setdefault(owner, {}).setdefault(attribute, {})[index] = curve.evaluate(frame)
        return found


def _as_blender_sets(struct, name, current, keyed):
    """A keyed value as Blender's animation system would have set it (anim_sys.cc animsys_write_rna_setting):
    booleans true above 1 - epsilon, integers truncated and clamped, floats in single precision and clamped,
    enums by their number. Arrays keep their unkeyed items."""
    prop = struct.bl_rna.properties[name]

    def one(value, was):
        if prop.type == "BOOLEAN":
            return bool(value > 1.0 - 1.1920929e-07)
        if prop.type == "INT":
            return min(max(int(value), prop.hard_min), prop.hard_max)
        if prop.type == "FLOAT":
            return float(min(max(np.float32(value), np.float32(prop.hard_min)), np.float32(prop.hard_max)))
        if prop.type == "ENUM":
            return next((item.identifier for item in prop.enum_items if item.value == int(value)), was)
        return was

    if getattr(prop, "is_array", False) and prop.type != "ENUM":
        out = list(current)
        for index, value in keyed.items():
            if 0 <= index < len(out):
                out[index] = one(value, out[index])
        return out
    return one(keyed.get(0, keyed.get(-1)), current) if keyed else current


class Sampled:
    """A settings struct read with this frame's sampled keys (SettingKeys.sample) over its own values: a keyed
    property gives the frame's value, anything else the property's own. Collections of settings structs (forces,
    sync bones, their targets) are read the same way. Writes go to the struct itself."""
    __slots__ = ("_real", "_frame", "_keyed")

    def __init__(self, real, frame_values):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_frame", frame_values)
        object.__setattr__(self, "_keyed", frame_values.get(real.path_from_id(), {}))

    def __getattr__(self, name):
        real = self._real
        value = getattr(real, name)
        keyed = self._keyed.get(name)
        if keyed is not None:
            return _as_blender_sets(real, name, value, keyed)
        if self._frame and isinstance(value, bpy.types.bpy_prop_collection):
            return [Sampled(item, self._frame) if isinstance(item, bpy.types.PropertyGroup) else item
                    for item in value]
        return value

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def recover(objects):
    """Unmute curves a previous session left muted (it ended while simulating)."""
    for obj in objects:
        release_marked(obj)