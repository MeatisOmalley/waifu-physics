"""Live simulation and deterministic fixed-clock Cache All baking.

One System holds every group of every armature. After Blender evaluates a
frame's animation (frame_change_post), the runtime reads each armature's
input pose, steps the solver by the frames' time, and writes the chains'
rotations back. It starts over -- points at the pose -- on the scene's first
frame, on a jump of more than a few frames or backwards, and whenever a
group's structure changes.
"""
import math

import bpy
import numpy as np
from bpy.app.handlers import persistent

from ..data import bone_refs
from ..data import colliders as collider_objects
from ..data import curves as group_curves
from ..data.props import FORCE_CHANNELS
from ..solver import forces as frame_forces
from ..solver import native
from ..solver import uemath as ue
from ..solver.build import Skeleton, GroupSpec, build, ComponentMotion
from ..solver.curves import LinearCurve
from ..solver.system import Group, COMPLIANCE_TYPES, PLANAR_NONE, PLANAR_X, PLANAR_Y, PLANAR_Z
from . import cache as frame_cache
from . import io
from . import keys as chain_keys

F32 = np.float32
PLANAR = {"NONE": PLANAR_NONE, "X": PLANAR_X, "Y": PLANAR_Y, "Z": PLANAR_Z}
MAX_FRAME_STEP = 4            # frames forward that still count as playing on; more is a jump

_runtimes = {}                # scene pointer -> Runtime
_dirty = set()                # scene pointers whose groups changed shape
_writing = False              # our own writes are in progress (the cache ignores them)


_building_cache = False


def _outdate(current, reason):
    """What the frames were made from changed. A baked cache (Cache, Cache All) is kept, playing as it was
    baked, and says it is outdated until it is baked again: an edit, even a mistaken one, never throws away a
    bake. Frames kept while playing live are dropped: playing on makes them again."""
    if current.cache_mode == "canonical" and current.cache:
        if current.outdated is None:
            current.outdated = reason
    else:
        current.cache.clear()
        current.cache_mode = "auto"


def invalidate(scene=None, reason="the setup changed"):
    """Something that changes the result changed: a bake is outdated, live frames are dropped."""
    if _building_cache:
        return
    for key, current in _runtimes.items():
        if scene is None or key == scene.as_pointer():
            _outdate(current, reason)


def clear_cache(scene):
    """Drop the scene's cached frames, baked or not (the Cache button turned off)."""
    current = _runtimes.get(scene.as_pointer())
    if current is not None:
        current.cache.clear()
        current.cache_mode = "auto"
        current.cache_rigs = None
        current.outdated = None


def is_cached(scene):
    """The scene plays a baked cache (Cache on and the bake still valid); otherwise it plays live."""
    current = _runtimes.get(scene.as_pointer())
    return bool(scene.waifu_physics.simulate and scene.waifu_physics.use_cache and current is not None
                and current.cache_mode == "canonical" and current.cache)


def mark_dirty(scene=None):
    """A group's structure changed: rebuild before the next step."""
    if scene is None:
        _dirty.update(_runtimes.keys())
        _dirty.add("all")
    else:
        _dirty.add(scene.as_pointer())


def armatures(scene):
    """Armature objects in the scene with at least one enabled group."""
    return [obj for obj in scene.objects if obj.type == "ARMATURE"
            and any(group.enabled and len(group.roots) for group in obj.waifu_physics.groups)]


class Runtime:
    """The simulation of one scene."""

    def __init__(self, scene, target_framerate=None, settle=False):
        """settle: clear the chain channels and re-evaluate before reading the input pose. Needed
        outside frame changes, where frame_change_pre has not cleared them."""
        self.scene_pointer = scene.as_pointer()
        self.cm = io.cm_per_unit(scene)
        self.rigs = [io.Rig(obj) for obj in armatures(scene)]
        self.group_props = []          # (rig index, group index in obj.waifu_physics.groups)
        names, parents, ref_length, pose, rotation = [], [], [], [], []
        offsets = []
        for rig in self.rigs:
            groups = [props for props in rig.obj.waifu_physics.groups if props.enabled and len(props.roots)]
            # Each group's chains cut off at its own excluded bones: a group can start where another ends.
            rig.set_chain(sorted({bone for props in groups for bone in rig.subtree(
                [root.name for root in props.roots], [bone.name for bone in props.excluded])}))
        self.stepped_ahead = None
        self.action_prints = {}
        if settle:
            self.settle(scene)
        for r, rig in enumerate(self.rigs):
            offsets.append(len(names))
            input_pose = rig.read()
            names += [f"{r}|{name}" for name in rig.names]
            parents += [p + offsets[-1] if p >= 0 else -1 for p in rig.parents]
            ref_length += list(rig.ref_lengths() * self.cm)
            pose.append(input_pose[:, :3, 3] * self.cm)
            rotation.append(io.quats_from_matrices(io.unscaled(input_pose[:, :3, :3])))
        self.offsets = offsets
        specs = []
        for r, rig in enumerate(self.rigs):
            for g, props in enumerate(rig.obj.waifu_physics.groups):
                if not props.enabled or not len(props.roots):
                    continue
                self.group_props.append((r, g))
                specs.append(self._spec(r, props))
        scene_settings = scene.waifu_physics
        self.system = build(Skeleton(names, parents, ref_length,
                                     np.concatenate(pose) if pose else np.zeros((0, 3)),
                                     np.concatenate(rotation) if rotation else np.zeros((0, 4))),
                            specs, target_framerate=target_framerate or scene_settings.target_framerate,
                            fixed_substepping=scene_settings.fixed_substepping)
        s = self.system
        # Stiffness and damping act per step, so the step rate stays the scene's simulation rate
        # (Kawaii's 60 Hz by default) whatever the frame rate. Above it a frame can fall between
        # steps; live playback then shows the chains between the last two steps.
        self.scene_clock = target_framerate is None
        self.interpolate = (self.scene_clock
                            and scene.render.fps / scene.render.fps_base > s.target_framerate)
        self.real = np.flatnonzero(s.bone >= 0)
        combined = s.bone[self.real]
        self.rig_of_point = np.searchsorted(np.array(offsets + [len(names)]), combined, side="right") - 1
        self.bone_of_point = combined - np.array(offsets)[self.rig_of_point] if len(offsets) else combined
        for r, rig in enumerate(self.rigs):
            rig.set_chain(self.bone_of_point[self.rig_of_point == r])
            rig.keys.mute()                # Waifu Physics samples the chains' keys itself while it simulates
        # One evaluation a frame needs every chain's keys to be Waifu Physics' (keys.py).
        self.fast = all(rig.keys.ownable for rig in self.rigs)
        # ... and reads this frame's input where it can be known before evaluation: the bones the chains
        # hang from and those the groups' colliders hang from (Rig.prepare_input).
        on_bones = {rig.uid: set() for rig in self.rigs}
        for g in range(len(self.group_props)):
            for obj in collider_objects.group_colliders(self._group(g)[1], scene):
                if obj.parent_type == "BONE" and obj.parent is not None and obj.parent.session_uid in on_bones:
                    rig = next(x for x in self.rigs if x.uid == obj.parent.session_uid)
                    if obj.parent_bone in rig.index:
                        on_bones[rig.uid].add(rig.index[obj.parent_bone])
        for rig in self.rigs:
            rig.prepare_input(on_bones[rig.uid])
        # Force filters and sync targets name bones; the points of each group by bone name.
        for i in self.real:
            s.bone_names[i] = names[s.bone[i]].split("|", 1)[1]
        self.point_of = {(int(s.group[i]), s.bone_names[i]): int(i) for i in self.real}
        self._find_tiers()
        self.motions = [ComponentMotion() for _ in self.group_props]
        self.last_frame = None
        self.backend = native.backend()
        self.cache = {}                   # frame -> cache.Snapshot
        self.cache_mode = "auto"
        self.cache_key = frame_cache.key(scene)
        self.cache_rigs = None            # a bake's armatures: (session_uid, bone count, chain rows), per snapshot
        self.outdated = None              # why a bake no longer matches the scene, or None
        self.own_update = False           # the next depsgraph update is our own write
        self.collider_prints = {}

    def _find_tiers(self):
        """Groups that start partway down another group's chain, as a second Kawaii node on the chain does.
        Each such root's anchor is the other group's point it hangs from. A group's tier is one more than the
        highest tier it hangs from, so tier 0 hangs from nothing simulated. Kawaii runs its nodes in order, each
        starting from the pose the one before it made; step_seconds does the same, tier by tier."""
        s = self.system
        n = len(s.parent)
        point = {(int(self.rig_of_point[k]), int(self.bone_of_point[k])): int(i) for k, i in enumerate(self.real)}
        anchor_of_root = np.full(n, -1)
        for k, i in enumerate(self.real):
            if s.parent[i] >= 0:
                continue
            r = int(self.rig_of_point[k])
            above = int(self.rigs[r].parents[self.bone_of_point[k]])
            a = point.get((r, above), -1) if above >= 0 else -1
            if a >= 0 and s.group[a] != s.group[i]:
                anchor_of_root[i] = a
        root = np.arange(n)
        while True:                                       # each point's root, up its parents
            up = s.parent[root] >= 0
            if not up.any():
                break
            root[up] = s.parent[root[up]]
        self.anchor = anchor_of_root[root]
        groups = len(self.group_props)
        hangs_from = [set() for _ in range(groups)]
        for i in np.flatnonzero(anchor_of_root >= 0):
            hangs_from[int(s.group[i])].add(int(s.group[anchor_of_root[i]]))
        tier = np.zeros(groups, dtype=int)
        for _ in range(groups):                           # at most one tier per group; a cycle stops there
            tier = np.array([max((tier[h] + 1 for h in hangs_from[g]), default=0) for g in range(groups)],
                            dtype=int).clip(max=groups)
        self.tiers = int(tier.max()) + 1 if groups else 1
        carried = np.intersect1d(self.real, np.flatnonzero(self.anchor >= 0))
        self.tier_rows = [carried[tier[s.group[carried]] == t] for t in range(self.tiers)]

    def _step_tiers(self, seconds):
        """One frame's steps in Kawaii's node order. With groups hanging from others, the frame is stepped
        once per tier from the same start. After each, the next tier's input pose is carried with the
        points it hangs from: each keeps its place relative to its anchor, as a bone below an excluded one
        keeps its local transform in Kawaii's output. The upper tiers never read the lower ones, so their
        result is the same every time; the last step is the one kept."""
        s = self.system
        if self.tiers < 2:
            s.step_frame(F32(seconds), self.backend)
            return
        start = frame_cache.Snapshot(self)
        pose, rotation = s.frame_pose.copy(), s.frame_pose_rot.copy()
        carried_pose, carried_rotation = pose.copy(), rotation.copy()
        inverse = np.array([-1.0, -1.0, -1.0, 1.0])
        for t in range(1, self.tiers):
            s.step_frame(F32(seconds), self.backend)
            rows = self.tier_rows[t]
            if len(rows):
                anchors = self.anchor[rows]
                turned, _turned = s.results()
                turn = ue.quat_multiply(turned[anchors], rotation[anchors] * inverse)
                carried_pose[rows] = s.loc[anchors] + ue.rotate_vector(turn, pose[rows] - pose[anchors])
                carried_rotation[rows] = ue.quat_multiply(turn, rotation[rows])
            start.restore_state(self)
            s.frame_pose[:], s.frame_pose_rot[:] = carried_pose, carried_rotation
        s.step_frame(F32(seconds), self.backend)

    def _spec(self, r, props):
        prefix = f"{r}|"
        cm = self.cm
        group = Group(
            settings=self._settings(props), curves=group_curves.curves(props),
            dummy_bone_length=props.dummy_bone_length * cm,
            bone_subdivision_count=props.bone_subdivision_count,
            bone_subdivision_collision_only=props.bone_subdivision_collision_only,
            bone_subdivision_densify_by_radius=props.bone_subdivision_densify_by_radius,
            planar_constraint=PLANAR[props.planar_constraint], legacy_gravity=props.legacy_gravity,
            compliance_type=COMPLIANCE_TYPES.index(props.compliance),
            iterations_before_collision=props.iterations_before_collision,
            iterations_after_collision=props.iterations_after_collision,
            auto_add_child_dummy_constraint=props.auto_child_dummy_links,
            constraint_subdivision_count=props.bridge_count,
            constraint_subdivision_feedback_scale=props.bridge_feedback)
        links = [(prefix + link.bone_a, prefix + link.bone_b,
                  -1 if link.compliance == "GROUP" else COMPLIANCE_TYPES.index(link.compliance),
                  link.exclude_from_subdivision) for link in props.links]
        return GroupSpec(group, roots=[(prefix + root.name, None) for root in props.roots],
                         exclude=[prefix + bone.name for bone in props.excluded], links=links)

    def _settings(self, props):
        """Kawaii's units: radius in centimetres, limit angle in degrees."""
        return dict(damping=props.damping, stiffness=props.stiffness,
                    world_damping_location=props.world_damping_location,
                    world_damping_rotation=props.world_damping_rotation,
                    radius=props.radius * self.cm, limit_angle=math.degrees(props.limit_angle))

    def _frame_forces(self, g, rig, props, scene):
        """This frame's external forces, wind settings and sync bones of a group, in Kawaii's units."""
        grp, cm = self.system.groups[g], self.cm
        grp.forces = [self._force(f) for f in props.forces]
        if props.use_force_fields:
            vectors = self._field_accelerations(g, rig, props, scene)
            if vectors is not None:
                grp.forces.append(frame_forces.ForceSpec(frame_forces.FIELDS, params=dict(vectors=vectors)))
        grp.sync_bones = [self._sync(g, rig, sync) for sync in props.sync_bones]
        grp.simple_external_force = tuple(np.array(props.simple_external_force) * cm)
        grp.world_space_simple_external_force = props.world_space_simple_external_force
        grp.enable_wind = props.enable_wind
        grp.wind_scale = props.wind_scale
        grp.wind_direction_noise_angle = math.degrees(props.wind_direction_noise_angle)

    def _field_accelerations(self, g, rig, props, scene):
        """The scene's force fields on the group's points, as Blender computes them for a particle there with
        the point's velocity (solver/fields.py), in simulation space: centimetres a second squared."""
        from . import fields as scene_fields
        from ..solver import fields as field_math
        specs = scene_fields.specs(scene, props.force_field_collection, props.force_field_strength)
        if not specs:
            return None
        s = self.system
        rows = np.flatnonzero(s.group == g)
        world = np.array(rig.world)
        turn = world[:3, :3]
        positions = (s.loc[rows] / self.cm) @ turn.T + world[:3, 3]
        dt = max(float(s.dt_old), 1.0e-6)
        velocities = ((s.loc[rows] - s.prev[rows]) / (dt * self.cm)) @ turn.T
        accelerations = field_math.accelerations(specs, positions, velocities, s.frame_number)
        return (accelerations @ np.linalg.inv(turn).T) * self.cm

    def _force(self, f):
        cm, kind = self.cm, f.kind
        scale = (f.random_min, f.random_max)
        if kind == "BASIC":
            params = dict(direction=tuple(np.array(f.direction) * cm), interval=f.interval)
        elif kind == "GRAVITY":
            params = dict(override_direction=f.override_direction, direction=tuple(f.direction))
            scale = (f.random_min * cm, f.random_max * cm)          # the pull's strength is an acceleration
        elif kind == "CURVE":
            channels = []
            for axis, channel in enumerate(FORCE_CHANNELS):
                found = group_curves.curve(f, channel, required=True)
                channels.append(None if found is None else LinearCurve(
                    found.times * F32(f.duration), found.values * F32(f.amplitude[axis] * cm)))
            params = dict(curves=tuple(channels), max_time=f.duration, time_scale=f.time_scale,
                          evaluate=f.evaluate, substeps=f.substeps)
        elif kind == "WIND":
            params = dict(noise_angle=math.degrees(f.noise_angle))
        else:
            params = dict(direction=tuple(f.direction), noise_angle=math.degrees(f.noise_angle),
                          noise_period=f.noise_period, time_scale=f.time_scale, constant=f.constant * cm,
                          sway=f.sway * cm, sway_period=f.sway_period, sway_phase=math.degrees(f.sway_phase),
                          ripple=f.ripple * cm, ripple_period=f.ripple_period,
                          ripple_phase=math.degrees(f.ripple_phase), ripple_delay=math.degrees(f.ripple_delay),
                          cycle_min=f.cycle_min, cycle_max=f.cycle_max, cycle_period=f.cycle_period,
                          cycle_phase=math.degrees(f.cycle_phase), random=f.random * cm,
                          random_period=f.random_period, seed=f.seed)
        return frame_forces.ForceSpec(
            kind, enabled=f.enabled, space="WORLD" if kind in ("GRAVITY", "WIND") else f.space,
            apply_bones=frozenset(b.name for b in f.apply_bones),
            ignore_bones=frozenset(b.name for b in f.ignore_bones),
            random_range=scale, rate_curve=group_curves.curve(f, "rate"), params=params)

    def _sync(self, g, rig, sync):
        cm = self.cm
        index = rig.index.get(sync.bone, -1)
        location = rest = None
        if index >= 0:
            location = rig.pose[index][:3, 3] * cm
            rest = rig.rest[index][:3, 3] * cm
        distance = group_curves.curve(sync, "distance")
        if distance is not None:
            distance = LinearCurve(distance.times * F32(sync.distance * cm), distance.values)
        targets = [frame_forces.SyncTargetSpec(root=self.point_of.get((g, t.bone), -1),
                                               include_children=t.include_children,
                                               rate_curve=group_curves.curve(t, "rate")) for t in sync.targets]
        return frame_forces.SyncSpec(
            location=location, rest_location=rest, targets=targets, global_scale=tuple(sync.global_scale),
            distance_curve=distance, directions=(sync.direction_x, sync.direction_y, sync.direction_z),
            attenuation=sync.attenuation, inner_radius=sync.inner_radius * cm, outer_radius=sync.outer_radius * cm,
            max_attenuation=sync.max_attenuation)

    def _group(self, g):
        r, index = self.group_props[g]
        return self.rigs[r], self.rigs[r].obj.waifu_physics.groups[index]

    # ------------------------------------------------------------------ frames
    def _read(self, scene, sample_number=None, ahead=False):
        """This frame's input pose and component movement into the system. ahead: before Blender
        evaluates the frame, from its last evaluation (Rig.read_ahead)."""
        s = self.system
        frame = frame_of(scene)
        poses = [rig.read_ahead(frame) if ahead else rig.read() for rig in self.rigs]
        pose = np.zeros((len(self.real), 4, 4))
        for r, rig_pose in enumerate(poses):
            rows = self.rig_of_point == r
            pose[rows] = rig_pose[self.bone_of_point[rows]]
        s.frame_pose[self.real] = pose[:, :3, 3] * self.cm
        s.frame_pose_rot[self.real] = io.quats_from_matrices(io.unscaled(pose[:, :3, :3]))
        s.frame_pose_scale[self.real] = np.linalg.norm(pose[:, :3, :3], axis=1)
        s.frame_number = scene.frame_current if sample_number is None else sample_number
        wind = scene_wind(scene, self.cm)
        for g in range(len(self.group_props)):
            rig, props = self._group(g)
            s.groups[g].settings = self._settings(props)
            s.groups[g].curves = group_curves.curves(props)
            self._frame_forces(g, rig, props, scene)
            s.wind[g] = wind
            world = rig.world
            s.world_to_sim[g] = np.linalg.inv(np.array(world.to_3x3()))
            if props.use_scene_gravity:          # the scene's gravity, in world space, scaled
                gravity = np.array(scene.gravity, dtype=float) * props.gravity_scale
                world_space = True
            else:
                gravity = np.array(props.gravity, dtype=float)
                world_space = props.use_world_space_gravity
            if world_space:
                gravity = np.linalg.inv(np.array(world.to_3x3())) @ gravity
            s.gravity[g] = gravity * self.cm
            location, rotation, scale = world.decompose()
            motion = self.motions[g]
            motion.teleport_distance = F32(props.teleport_distance * self.cm)
            motion.teleport_rotation = F32(math.degrees(props.teleport_rotation))
            move, move_rot, teleport = motion.update(np.array(location) * self.cm,
                                                     (rotation.x, rotation.y, rotation.z, rotation.w), tuple(scale))
            s.frame_move[g] = move
            s.frame_move_rot[g] = move_rot
            s.teleport[g] = teleport
        s.set_shapes([self._shapes(g, scene) for g in range(len(self.group_props))])
        s.resolve_settings()

    def _shapes(self, g, scene):
        """This frame's colliders for a group (colliders.group_colliders), in its armature's space (Kawaii's
        Update*Limits, once a frame). Read ahead of evaluation, a collider on a bone whose pose was rebuilt for
        this frame (Rig.read_ahead) moves with it; any other is where Blender last evaluated it."""
        rig, props = self._group(g)
        to_armature = np.linalg.inv(np.array(rig.world))
        shapes = []
        for obj in collider_objects.group_colliders(props, scene):
            matrix = to_armature @ self._collider_world(obj)
            shape = collider_objects.shape_of(obj, rig.obj, self.cm, matrix=matrix)
            if shape is not None:
                shapes.append(shape)
        return shapes

    def _collider_world(self, obj):
        """A collider's world matrix this frame: carried by its bone from where Blender last evaluated both,
        when that bone was rebuilt for this frame; otherwise the collider's last evaluated matrix."""
        last = np.array(obj.matrix_world)
        parent = obj.parent
        if obj.parent_type != "BONE" or parent is None:
            return last
        rig = next((r for r in self.rigs if r.uid == parent.session_uid), None)
        keys = rig.input_keys if rig is not None else None
        bone = rig.index.get(obj.parent_bone) if rig is not None else None
        if keys is None or bone is None or bone not in keys.bones or getattr(rig, "evaluated", None) is None:
            return last
        world_now, world_then = np.array(rig.world), np.array(rig.evaluated_world)
        carried = rig.pose[bone] @ np.linalg.inv(rig.evaluated[bone])
        return world_now @ carried @ np.linalg.inv(world_then) @ last

    def scene(self):
        """The scene this runtime simulates, or None once it is gone."""
        return next((scene for scene in bpy.data.scenes if scene.as_pointer() == self.scene_pointer), None)

    def sync_clock(self, scene):
        """Steps per Second or Fixed Steps changed: the solver's clock follows where it is, so the chains swing
        on (a rebuild would put them back at the pose). Their velocities are kept: the solver takes velocity as
        the last step's displacement over the previous step's time, so the displacement is rescaled to the new
        step time. A bake keeps playing, outdated; live frames are dropped. Returns whether it changed."""
        key = frame_cache.key(scene)
        if not self.scene_clock or key == self.cache_key or key[:4] != self.cache_key[:4]:
            return False                          # unchanged, a bake's own clock, or the frame range/rate moved
        s, settings = self.system, scene.waifu_physics
        before = F32(s.dt_old)
        s.target_framerate = int(settings.target_framerate)
        s.fixed_substepping = bool(settings.fixed_substepping)
        after = F32(1.0) / F32(s.target_framerate) if s.fixed_substepping else before
        if before > 0 and after != before:
            s.prev[:] = s.loc - (s.loc - s.prev) * (after / before)
            s.dt_old = after
        fps = scene.render.fps / scene.render.fps_base
        self.interpolate = fps > s.target_framerate
        _outdate(self, "the simulation rate changed")
        self.cache_key = key
        return True

    def reset(self, scene):
        """Points back at the pose; warm-up steps if a group asks for them."""
        self._read(scene)
        s = self.system
        s.begin(s.frame_pose.copy(), s.frame_pose_rot.copy())
        for motion in self.motions:
            motion.previous = None
        self._read(scene)                     # movement relative to this frame: none
        warm = [self._group(g)[1].warm_up_frames for g in range(len(self.group_props))]
        if max(warm, default=0) > 0:
            self._warm_up(warm)
        self._write()

    def _warm_up(self, warm):
        """WarmUp (Simulation.cpp:1364): steps of the fixed step before the first frame, per group."""
        s = self.system
        saved = {}
        fixed_dt = F32(1.0) / F32(s.target_framerate)
        for k in range(max(warm)):
            for g, count in enumerate(warm):
                if count == k:
                    rows = s.group == g
                    saved[g] = (s.loc[rows].copy(), s.prev[rows].copy())
            accumulator = s.accumulator
            s.accumulator = F32(0.0)
            s.step_frame(fixed_dt, self.backend, prepare=(k == 0), call=k)
            s.accumulator = accumulator
        for g, (loc, prev) in saved.items():
            rows = s.group == g
            s.loc[rows], s.prev[rows] = loc, prev

    def step_seconds(self, scene, seconds, sample_number=None, max_substeps=None, ahead=False):
        self._read(scene, sample_number, ahead)
        s = self.system
        old_cap = s.max_substeps
        if max_substeps is not None:
            s.max_substeps = max_substeps
        try:
            self._step_tiers(seconds)
        finally:
            s.max_substeps = old_cap
        for g in range(len(self.group_props)):
            rig, _props = self._group(g)
            location, rotation, scale = rig.world.decompose()
            self.motions[g].consume(s.consume_fraction, np.array(location) * self.cm,
                                    (rotation.x, rotation.y, rotation.z, rotation.w), tuple(scale))
        self._write()

    def step(self, scene, frames, ahead=False):
        """ahead: solve in frame_change_pre, before Blender evaluates the frame, so it evaluates the
        frame once with the result. The body's input is then a frame late; the chains' keys are
        sampled for this frame."""
        if ahead:
            self.restore(frame_of(scene))
        seconds = frames * scene.render.fps_base / scene.render.fps
        # Every step the frame's time holds: Kawaii caps the steps a frame may take (dropping the rest) against
        # game hitches, but a Blender frame's time is exact, and dropping any would slow the chains.
        needed = math.ceil((float(self.system.accumulator) + seconds) * self.system.target_framerate)
        self.step_seconds(scene, seconds, max_substeps=max(needed, 1), ahead=ahead)

    def _write(self):
        """Rotations back onto every chain bone, and heads for bones Kawaii places directly."""
        global _writing
        s = self.system
        simulated = s.loc
        if self.interpolate and s.step_start_loc is not None:
            # Between steps: blend from the last step's start by the time carried past it.
            alpha = min(1.0, float(s.accumulator) * s.target_framerate)
            s.loc = s.step_start_loc + (simulated - s.step_start_loc) * alpha
        try:
            rotation, _turned = s.results()
            heads = s.loc
        finally:
            s.loc = simulated
        # Every bone below a group's root takes its simulated head (ApplySimulateResult sets each
        # such bone's location). Where a parent has one child its rotation already puts the head
        # there; sync bones and branching chains need the location itself.
        placed = s.parent[self.real] >= 0
        _writing = True
        self.own_update = True
        try:
            for r, rig in enumerate(self.rigs):
                rows = self.rig_of_point == r
                rig.write(self.bone_of_point[rows], rotation[self.real[rows]],
                          heads[self.real[rows]] / self.cm, placed[rows])
        finally:
            _writing = False

    def alive(self):
        """Every armature it was built for is still there, with the same bones."""
        return all(rig.alive() for rig in self.rigs)

    def restore(self, frame=None):
        for rig in self.rigs:
            if rig.alive():
                rig.restore(frame)

    def release(self):
        """Stop simulating: the chains' keys back to Blender, the chains back to their input. A deleted
        armature is skipped; its keys went with it."""
        for rig in self.rigs:
            if not rig.alive():
                continue
            try:
                rig.keys.unmute()
                rig.restore()
            except ReferenceError:
                pass

    def settle(self, scene):
        """Clear the chains' physics and re-evaluate now, so read() sees a clean input pose."""
        self.restore(frame_of(scene))
        for layer in scene.view_layers:
            layer.update()

    def store(self, frame):
        if not self.cache:
            self.collider_prints = frame_cache.collider_prints(self)
        self.cache[frame] = frame_cache.Snapshot(self)

    def show_unsimulated(self, frame=None):
        """A frame the cache has not reached: the chains at their input, as cloth shows them."""
        self.own_update = True
        for rig in self.rigs:
            rig.read()
            rig.restore(frame)

    def cached_range(self):
        return (min(self.cache), max(self.cache)) if self.cache else None

    def needs_post_replay(self):
        """Keyed chain channels Blender still applies (not taken over) overwrite a replay made
        before evaluation, so it must be made again after."""
        return any(np.any(rig.chain & rig.keyed[kind]) and not rig.keys.muted
                   for rig in self.rigs for kind in ("location", "rotation", "scale"))

    def keys_changed(self):
        """Keys added to or removed from a chain, or a chain curve muted or unmuted: the curves Waifu Physics
        owns must be found again."""
        return any((rig.keys is not None and (rig.keys.count(rig) != rig.keys.total or rig.keys.mutes_changed(rig)))
                   or (rig.input_keys is not None and rig.input_keys.changed()) for rig in self.rigs)


def frame_of(scene):
    return scene.frame_current + scene.frame_subframe


def _essentials(scene, rt):
    """The objects the simulation's input depends on: the armatures, colliders and force fields of every
    type (the solver reads them all, and a disabled field is not visible, so it would be skipped), with
    what they are parented to and their constraint and driver targets, followed through."""
    needed = set()
    stack = [rig.obj for rig in rt.rigs] + [
        obj for obj in scene.objects
        if obj.waifu_physics_collider.is_collider or (obj.field is not None and obj.field.type != "NONE")]
    while stack:
        obj = stack.pop()
        if obj is None or obj.name in needed:
            continue
        needed.add(obj.name)
        stack.append(obj.parent)
        owners = [obj] + (list(obj.pose.bones) if obj.pose is not None else [])
        for owner in owners:
            for constraint in owner.constraints:
                stack.append(getattr(constraint, "target", None))
                stack += [t.target for t in getattr(constraint, "targets", ())]
        data = obj.animation_data
        if data is not None:
            for driver in data.drivers:
                for variable in driver.driver.variables:
                    stack += [t.id for t in variable.targets if isinstance(t.id, bpy.types.Object)]
    return needed


class lean_evaluation:
    """While baking, Blender evaluates only what the simulation reads: everything else is disabled
    in the viewport (hide_viewport), so skinning, modifiers and geometry nodes are skipped."""

    def __init__(self, scene, rt):
        needed = _essentials(scene, rt)
        self.hidden = [obj for obj in scene.objects if obj.name not in needed and not obj.hide_viewport]

    def __enter__(self):
        for obj in self.hidden:
            obj.hide_viewport = True
        return self

    def __exit__(self, *exc):
        for obj in self.hidden:
            try:
                obj.hide_viewport = False
            except ReferenceError:
                pass
        return False


def scene_wind(scene, cm):
    """The scene's wind, standing in for Unreal's wind sources: every visible Wind force field
    blows along its local Z at its Strength (Unreal's wind Speed; not converted, since Kawaii
    uses it as is). (world direction, speed), or None when nothing blows."""
    total = np.zeros(3)
    found = False
    for obj in scene.objects:
        field = obj.field
        if field is None or field.type != "WIND" or not field.strength:
            continue
        try:
            if not obj.visible_get():
                continue
        except RuntimeError:
            pass
        axis = np.array(obj.matrix_world.to_3x3().col[2])
        length = np.linalg.norm(axis)
        if length > 0:
            total += axis / length * field.strength
            found = True
    speed = float(np.linalg.norm(total))
    if not found or speed == 0.0:
        return None
    return total / speed, speed


def runtime(scene, rebuild=False):
    key = scene.as_pointer()
    current = _runtimes.get(key)
    stale = current is not None and not current.alive()
    if current is not None and not (rebuild or stale or key in _dirty or "all" in _dirty):
        current.sync_clock(scene)
    changed_clock = current is not None and current.cache_key != frame_cache.key(scene)
    if current is None or rebuild or changed_clock or stale or key in _dirty or "all" in _dirty:
        previous = current
        if previous is not None:
            previous.release()
        for obj in armatures(scene):
            bone_refs.repair(obj)          # renamed bones followed before the groups are read
        _dirty.discard(key)
        _dirty.discard("all")
        current = _runtimes[key] = Runtime(scene)
        if previous is not None and previous.cache_mode == "canonical" and previous.cache:
            # A bake outlives the simulation it was made with: it plays on, outdated.
            current.cache, current.cache_mode = previous.cache, "canonical"
            current.cache_rigs, current.collider_prints = previous.cache_rigs, previous.collider_prints
            current.outdated = previous.outdated or (
                "the frame range or rate changed" if changed_clock else "the chains changed")
    return current


def _drop_if_stale(scene):
    """A runtime whose armature was deleted or re-boned is dropped before anything touches it; the next
    frame builds a new one. Returns the runtime still usable, or None."""
    key = scene.as_pointer()
    current = _runtimes.get(key)
    if current is not None and not current.alive():
        _runtimes.pop(key).release()
        _dirty.add(key)
        return None
    return current


def bake_cache(scene, progress=None):
    """Bake on a scene-FPS-independent clock, sampling the input pose each tick.

    Only the requested output frames are retained. The transient solver ticks
    between them are evaluated once; later scrubbing only replays channels.
    """
    global _building_cache
    if _building_cache:
        raise RuntimeError("A Waifu Physics cache build is already running")
    key = scene.as_pointer()
    original = scene.frame_current
    rate = scene.waifu_physics.target_framerate
    fps = scene.render.fps / scene.render.fps_base
    if rate < 1 or fps <= 0:
        raise ValueError("Invalid simulation or scene frame rate")
    previous = _runtimes.get(key)
    if previous is not None:
        previous.release()
    _building_cache = True
    try:
        rt = Runtime(scene, target_framerate=rate, settle=True)
        _runtimes[key] = rt
        _dirty.discard(key)
        _dirty.discard("all")
        start, end = scene.frame_start, scene.frame_end
        lean = lean_evaluation(scene, rt).__enter__()
        scene.frame_set(start)
        if not scene.waifu_physics.simulate:
            raise RuntimeError("Simulate must stay enabled while building the cache")
        rt.reset(scene)
        earlier = frame_cache.Snapshot(rt)
        rt.cache[start] = earlier
        next_frame = start + 1
        end_tick = math.ceil((end - start) * rate / fps - 1e-10)
        for tick in range(1, end_tick + 1):
            position = start + tick * fps / rate
            whole = math.floor(position + 1e-10)
            scene.frame_set(whole, subframe=position - whole)
            if not scene.waifu_physics.simulate:
                raise RuntimeError("Simulate must stay enabled while building the cache")
            if key in _dirty or "all" in _dirty:
                raise RuntimeError("A group's structure changed while building the cache")
            rt.step_seconds(scene, 1.0 / rate, sample_number=tick, max_substeps=1)
            later = frame_cache.Snapshot(rt)
            while next_frame <= end and (next_frame - start) / fps <= tick / rate + 1e-10:
                fraction = ((next_frame - start) / fps - (tick - 1) / rate) * rate
                rt.cache[next_frame] = frame_cache.Snapshot.between(
                    earlier, later, min(1.0, max(0.0, fraction)), rt)
                if progress is not None:
                    progress(next_frame)
                next_frame += 1
            earlier = later
        if next_frame <= end:
            raise RuntimeError(f"Cache build stopped before frame {next_frame}")
        rt.collider_prints = frame_cache.collider_prints(rt)
        rt.cache_key = frame_cache.key(scene)
        rt.cache_mode = "canonical"
        rt.cache_rigs = [(rig.uid, len(rig.obj.pose.bones), np.flatnonzero(rig.chain)) for rig in rt.rigs]
        rt.outdated = None
        rt.last_frame = None
    except BaseException:
        current = _runtimes.get(key)
        if current is not None:
            current.cache.clear()
            current.cache_mode = "auto"
        raise
    finally:
        if "lean" in locals():
            lean.__exit__(None, None, None)
        _building_cache = False
        scene.frame_set(original)


def set_simulating(scene, on):
    if _building_cache:
        return
    if on:
        previous = _runtimes.pop(scene.as_pointer(), None)
        if previous is not None:
            previous.release()
        rt = _runtimes[scene.as_pointer()] = Runtime(scene, settle=True)
        _dirty.discard(scene.as_pointer())
        _dirty.discard("all")
        rt.reset(scene)
        rt.last_frame = scene.frame_current
    else:
        current = _runtimes.pop(scene.as_pointer(), None)
        if current is not None:
            current.release()


@persistent
def _frame_changing(scene, depsgraph=None):
    """Before Blender evaluates a frame: write a cached frame, so renders see it; otherwise clear
    last frame's physics from the chains, so the evaluated pose is a clean input."""
    settings = scene.waifu_physics
    if not settings.simulate:
        return
    key = scene.as_pointer()
    current = _drop_if_stale(scene)
    if current is None:
        return
    current.stepped_ahead = None
    dirty = key in _dirty or "all" in _dirty
    if not dirty and not _building_cache:
        current.sync_clock(scene)
    frame = scene.frame_current
    cached = settings.use_cache and current.cache_mode == "canonical"
    if not _building_cache and cached:                    # a bake plays even when outdated
        snapshot = current.cache.get(frame)
        if snapshot is not None:
            snapshot.replay(current)
            return
    # Playing on: solve now, so Blender evaluates the frame once (the body's input a frame late). With Fast
    # Evaluation off, or keys it cannot take over, it solves after evaluation instead and Blender evaluates twice.
    if not _building_cache and not dirty and current.fast and settings.fast_evaluation \
            and current.cache_mode != "canonical" \
            and current.cache_key == frame_cache.key(scene) and frame != scene.frame_start:
        last = current.last_frame
        frames = None if last is None else frame - last
        if frames is not None and 1 <= frames <= MAX_FRAME_STEP:
            current.step(scene, frames, ahead=True)
            current.stepped_ahead = frame
            return
    current.restore(frame_of(scene))


@persistent
def _frame_changed(scene, depsgraph=None):
    if _building_cache:
        return
    settings = scene.waifu_physics
    if not settings.simulate:
        return
    rt = runtime(scene)
    frame = scene.frame_current
    if rt.stepped_ahead == frame:
        rt.last_frame = frame
        return
    frames = None if rt.last_frame is None else frame - rt.last_frame
    playing_on = frames is not None and 1 <= frames <= MAX_FRAME_STEP
    if not (settings.use_cache and rt.cache_mode == "canonical"):          # live
        if not playing_on or frame == scene.frame_start:
            rt.reset(scene)
        else:
            rt.step(scene, frames)
        rt.last_frame = frame
        return
    snapshot = rt.cache.get(frame)
    if snapshot is not None:
        if rt.cache_mode != "canonical":
            snapshot.restore_state(rt)
        # The pre-frame write survives evaluation when the chain channels are
        # unkeyed. Only keyed channels need a second, post-evaluation replay.
        if rt.needs_post_replay():
            snapshot.replay(rt)
        rt.last_frame = frame
    elif rt.cache_mode == "canonical":
        rt.show_unsimulated(frame)
        rt.last_frame = None
    elif frame == scene.frame_start:
        rt.reset(scene)
        rt.store(frame)
        rt.last_frame = frame
    elif playing_on and rt.last_frame in rt.cache:
        rt.cache[rt.last_frame].restore_state(rt)
        rt.step(scene, frames)
        rt.store(frame)
        rt.last_frame = frame
    else:
        rt.show_unsimulated(frame)
        rt.last_frame = None


def _armature_changes(scene, depsgraph):
    """Bones renamed, added or removed, and armatures with groups added or deleted: renamed bones are
    followed (bone_refs), and the runtime is rebuilt when what it was built from changed."""
    if depsgraph.id_type_updated("ARMATURE"):           # bones renamed, added or removed: follow renames
        edited = {update.id.original.session_uid for update in depsgraph.updates
                  if isinstance(update.id, bpy.types.Armature)}
        for obj in scene.objects:
            if (obj.type == "ARMATURE" and obj.data is not None and obj.data.session_uid in edited
                    and obj.mode != "EDIT" and len(obj.waifu_physics.groups) and bone_refs.repair(obj)):
                mark_dirty(scene)
    current = _runtimes.get(scene.as_pointer())
    if current is None:
        return
    if not all(rig.same_bones() for rig in current.rigs):
        mark_dirty(scene)
    elif depsgraph.id_type_updated("ARMATURE") and not all(rig.same_inheritance() for rig in current.rigs):
        mark_dirty(scene)                      # a bone's Inherit Scale, Inherit Rotation or Local Location
    elif depsgraph.id_type_updated("COLLECTION") or depsgraph.id_type_updated("SCENE") \
            or depsgraph.id_type_updated("OBJECT"):
        if [obj.session_uid for obj in armatures(scene)] != [rig.uid for rig in current.rigs]:
            mark_dirty(scene)                  # an armature with groups added, deleted or unlinked


@persistent
def _depsgraph_updated(scene, depsgraph):
    if _building_cache:
        return
    _armature_changes(scene, depsgraph)
    current = _drop_if_stale(scene)
    if current is None:
        return
    if frame_cache.relevant_update(current, depsgraph):
        _outdate(current, "the scene changed")
        for rig in current.rigs:
            rig.refresh_keyed()
        if current.keys_changed():
            mark_dirty(scene)
    elif depsgraph.id_type_updated("ACTION") and current.keys_changed():
        # One of our muted curves unmuted is no change to what the keys say (relevant_update skips our
        # mutes), but Blender would now apply it over the physics: take the curves over again.
        _outdate(current, "the keys changed")
        mark_dirty(scene)


@persistent
def _file_loaded(_file):
    _runtimes.clear()
    _dirty.clear()


@persistent
def _file_ready(_file):
    """Curves a previous session left muted (it ended while simulating) are unmuted; bones renamed while the
    add-on was off are followed."""
    chain_keys.recover(bpy.data.objects)
    for obj in bpy.data.objects:
        if obj.library is None and obj.type == "ARMATURE" and len(obj.waifu_physics.groups):
            bone_refs.repair(obj)


@persistent
def _saving(_file):
    """Files are saved with the user's keys unmuted."""
    for current in _runtimes.values():
        for rig in current.rigs:
            if rig.alive():
                rig.keys.unmute()


@persistent
def _saved(_file):
    for current in _runtimes.values():
        for rig in current.rigs:
            if rig.alive():
                rig.keys.mute()


def register():
    if _frame_changed not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_changed)
    if _frame_changing not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_frame_changing)
    if _depsgraph_updated not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_updated)
    if _file_loaded not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_file_loaded)
    for handler, name in ((_file_ready, "load_post"), (_saving, "save_pre"), (_saved, "save_post")):
        if handler not in getattr(bpy.app.handlers, name):
            getattr(bpy.app.handlers, name).append(handler)


def unregister():
    for current in _runtimes.values():
        current.release()
    _runtimes.clear()
    _dirty.clear()
    while _frame_changed in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_changed)
    while _frame_changing in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_frame_changing)
    while _depsgraph_updated in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_updated)
    while _file_loaded in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_file_loaded)
    for handler, name in ((_file_ready, "load_post"), (_saving, "save_pre"), (_saved, "save_post")):
        while handler in getattr(bpy.app.handlers, name):
            getattr(bpy.app.handlers, name).remove(handler)
