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


class ChainKeys:
    """The chain bones' keyframe curves of one rig, which Waifu Physics samples while it owns them."""

    def __init__(self, rig):
        obj = rig.obj
        self.obj = obj
        self.curves = []                   # (fcurve, bone index, channel, array index)
        self.ownable = True
        self.muted = False
        self.total = 0                     # chain curves in the action, muted or not: notices new keys
        self.user_muted = set()            # chain curves the user muted: left alone, and their channels at rest
        data = obj.animation_data
        if data is None:
            return
        chain = {rig.names[i] for i in np.flatnonzero(rig.chain)}
        for curve in _action_curves(data):
            found = _bone_path(curve.data_path)
            if found and found[0] in chain:
                self.total += 1
                if curve.mute:
                    self.user_muted.add((curve.data_path, curve.array_index))
                else:
                    self.curves.append((curve, rig.index[found[0]], found[1], curve.array_index))
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
        if self.curves and (data.action_influence < 1.0 or data.action_blend_type != "REPLACE"):
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
        owned = {(c.data_path, c.array_index) for c, *_ in self.curves} if self.muted else set()
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
        if not self.ownable or not self.curves or self.muted:
            return
        for curve, *_ in self.curves:
            curve.mute = True
        self.obj[MARK] = json.dumps([(c.data_path, c.array_index) for c, *_ in self.curves])
        self.muted = True

    def unmute(self):
        if not self.muted:
            return
        for curve, *_ in self.curves:
            try:
                curve.mute = False
            except ReferenceError:
                pass
        try:
            if MARK in self.obj:
                del self.obj[MARK]
        except ReferenceError:
            pass
        self.muted = False

    def sample(self, frame, buffers):
        """Write the sampled keyframe values at frame into per-bone channel buffers."""
        for curve, bone, channel, index in self.curves:
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
        self.data = obj.animation_data
        self.action = self.data.action if self.data is not None else None
        self.curves = []                   # (fcurve, bone index, channel, array index)
        self.object_curves = []            # (fcurve, channel, array index)
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
        data = self.data
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
        for curve in own:
            found = _bone_path(curve.data_path)
            if found and found[0] in chosen and not curve.mute:
                self.curves.append((curve, rig.index[found[0]], found[1], curve.array_index))
        self.object_ok = (whole and obj.parent is None
                          and not any(c.enabled and c.influence > 0 for c in obj.constraints)
                          and not any(p in OBJECT_CHANNELS or p in OBJECT_BLOCKERS for p in foreign)
                          and not any(p.startswith("delta_") for p in (c.data_path for c in own))
                          and tuple(obj.delta_location) == (0, 0, 0) and tuple(obj.delta_scale) == (1, 1, 1)
                          and tuple(obj.delta_rotation_euler) == (0, 0, 0)
                          and tuple(obj.delta_rotation_quaternion) == (1, 0, 0, 0))
        if self.object_ok:
            self.object_curves = [(c, c.data_path, c.array_index) for c in own
                                  if c.data_path in OBJECT_CHANNELS and not c.mute]
        self.rig = rig
        self.total = self._count()

    def _count(self):
        """What the sampling was worked out from: the action and its curves (with their mute flags), and what
        decides which bones and whether the object can be sampled -- constraints, drivers, NLA, parent."""
        data, obj = self.data, self.rig.obj
        try:
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
        for curve, bone, channel, index in self.curves:
            buffers[channel][bone * CHANNELS[channel] + index] = curve.evaluate(frame)

    def world(self, obj, frame):
        """The armature object's world matrix at frame, or None when it cannot be known before evaluation."""
        if not self.object_ok:
            return None
        from mathutils import Euler, Matrix, Quaternion, Vector
        values = {path: list(getattr(obj, path)) for path in OBJECT_CHANNELS}
        for curve, path, index in self.object_curves:
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


def recover(objects):
    """Unmute curves a previous session left muted (it ended while simulating)."""
    for obj in objects:
        for mark in (MARK, LEGACY_MARK):
            if mark not in obj.keys():
                continue
            try:
                wanted = {tuple(entry) for entry in json.loads(obj[mark])}
            except (TypeError, ValueError):
                wanted = set()
            data = obj.animation_data
            if data is not None:
                for curve in _action_curves(data):
                    if (curve.data_path, curve.array_index) in wanted:
                        curve.mute = False
            del obj[mark]
