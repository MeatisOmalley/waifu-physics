"""Every simulated point in a scene, as flat arrays, and Kawaii Physics' frame loop.

A *group* is one Kawaii node: its root bones, settings, curves, links and
colliders. All groups of all armatures share one set of arrays so that a step
is a handful of array operations (numpy) or one call (C). Within a group,
points keep Kawaii's ModifyBones order; the arrays as a whole are then sorted
by depth in the chain, parents first, which leaves every result unchanged
(each point depends only on its parent, links and collisions are ordered by
their own lists) and lets numpy work one depth at a time.

Porting notes cite Kawaii at commit 64cbc77 as file:line:
  Simulation.cpp = AnimNode_KawaiiPhysicsSimulation.cpp
  Collision.cpp  = AnimNode_KawaiiPhysicsCollision.cpp
  ModifyBones.cpp = AnimNode_KawaiiPhysicsModifyBones.cpp
"""
from dataclasses import dataclass, field

import numpy as np

from . import forces as frame_forces
from . import uemath as ue

F32, F64, I32, I8 = np.float32, np.float64, np.int32, np.int8

KIND_BONE, KIND_TIP, KIND_INTER, KIND_BRIDGE = 0, 1, 2, 3
SPHERE_OUTER, SPHERE_INNER, CAPSULE, TAPERED, BOX, PLANE = 0, 1, 2, 3, 4, 5
PLANAR_NONE, PLANAR_X, PLANAR_Y, PLANAR_Z = 0, 1, 2, 3
# Kawaii tests spheres, then capsules, tapered capsules, boxes and planes (Simulation.cpp SimulateOnce).
TYPE_RANK = {SPHERE_OUTER: 0, SPHERE_INNER: 0, CAPSULE: 1, TAPERED: 2, BOX: 3, PLANE: 4}

# XPBD compliance by material (Collision.cpp:1413), as the float literals Kawaii uses.
COMPLIANCE_TYPES = ("CONCRETE", "WOOD", "LEATHER", "TENDON", "RUBBER", "MUSCLE", "FAT")
COMPLIANCE = np.array([0.00000000004, 0.00000000016, 0.000000001, 0.000000002, 0.0000001, 0.00002, 0.0001],
                      dtype=F32)

SETTINGS = ("damping", "stiffness", "world_damping_location", "world_damping_rotation", "radius", "limit_angle")
# FKawaiiPhysicsSettings defaults (KawaiiPhysicsTypes.h); radius in centimetres, limit angle in degrees.
DEFAULT_SETTINGS = dict(damping=0.1, stiffness=0.05, world_damping_location=0.8, world_damping_rotation=0.8,
                        radius=3.0, limit_angle=0.0)


@dataclass
class Group:
    """One Kawaii node's configuration."""
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    # Curves over the length rate from root (0) to tip (1), multiplying each
    # setting; None is an empty curve (1.0 everywhere), as in Kawaii.
    curves: dict = field(default_factory=dict)
    dummy_bone_length: float = 0.0
    bone_subdivision_count: int = 0
    bone_subdivision_collision_only: bool = True
    bone_subdivision_densify_by_radius: bool = False
    planar_constraint: int = PLANAR_NONE
    legacy_gravity: bool = False
    compliance_type: int = COMPLIANCE_TYPES.index("LEATHER")
    iterations_before_collision: int = 1
    iterations_after_collision: int = 1
    auto_add_child_dummy_constraint: bool = True
    constraint_subdivision_count: int = 0
    constraint_subdivision_feedback_scale: float = 1.0
    # Read every frame (Kawaii units: centimetres, seconds, degrees).
    forces: list = field(default_factory=list)            # forces.ForceSpec, in ExternalForces order
    sync_bones: list = field(default_factory=list)        # forces.SyncSpec, in SyncBones order
    simple_external_force: tuple = (0.0, 0.0, 0.0)
    world_space_simple_external_force: bool = True
    enable_wind: bool = False
    wind_scale: float = 1.0
    wind_direction_noise_angle: float = 0.0


@dataclass
class Shape:
    """A collision limit in simulation space (Kawaii's F*Limit after Update*Limits)."""
    kind: int
    location: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)       # Unreal order: x, y, z, w
    radius: float = 5.0                           # sphere, capsule; tapered capsule's +Z end
    radius1: float = 5.0                          # tapered capsule's -Z end
    length: float = 10.0                          # capsule, tapered capsule
    extent: tuple = (5.0, 5.0, 5.0)               # box half size
    enabled: bool = True


class System:
    """The points of every group, and everything a step reads and writes."""

    def __init__(self, groups, points, links, target_framerate=60, max_substeps=4, fixed_substepping=True):
        """groups: list of Group. points: dict of per-point lists in Kawaii order within each group
        (parent, group, kind, real_parent, real_child, alpha, location, pose, pose_rotation,
        length_rate, and optionally bone -- the caller's bone index, -1 for dummies).
        links: dict of per-link lists (a, b, length, compliance_type or -1 for the group's)."""
        self.groups = list(groups)
        self.target_framerate = int(target_framerate)
        self.max_substeps = max(1, int(max_substeps))
        self.fixed_substepping = bool(fixed_substepping)
        n = len(points["parent"])
        depth = np.zeros(n, dtype=I32)
        parent = np.asarray(points["parent"], dtype=I32)
        for i in range(n):
            if parent[i] >= 0:
                depth[i] = depth[parent[i]] + 1
        order = np.argsort(depth, kind="stable")
        new_index = np.empty(n, dtype=I32)
        new_index[order] = np.arange(n, dtype=I32)
        def remap(idx):
            idx = np.asarray(idx, dtype=I32)
            return np.where(idx >= 0, new_index[np.maximum(idx, 0)], -1).astype(I32)

        self.n = n
        self.order = order                       # sorted position -> caller's point index
        self.depth = depth[order]
        self.parent = remap(parent[order])
        self.group = np.asarray(points["group"], dtype=I32)[order]
        self.kind = np.asarray(points["kind"], dtype=I8)[order]
        self.real_parent = remap(np.asarray(points["real_parent"])[order])
        self.real_child = remap(np.asarray(points["real_child"])[order])
        self.alpha = np.asarray(points["alpha"], dtype=F32)[order]
        self.length_rate = np.asarray(points["length_rate"], dtype=F32)[order]
        self.bone_length = np.asarray(points.get("bone_length", [0.0] * n), dtype=F32)[order]
        self.length_from_root = np.asarray(points.get("length_from_root", [0.0] * n), dtype=F32)[order]
        self.bone = np.asarray(points.get("bone", [-1] * n), dtype=I32)[order]
        # Each tip's distance past its real bone; without one, the group's DummyBoneLength, as in Kawaii.
        tips = points.get("tip_length") or [groups[g].dummy_bone_length for g in points["group"]]
        self.tip_length = np.asarray(tips, dtype=F32)[order]
        self.loc = np.ascontiguousarray(np.asarray(points["location"], dtype=F64).reshape(n, 3)[order])
        self.prev = self.loc.copy()
        self.pose = np.ascontiguousarray(np.asarray(points["pose"], dtype=F64).reshape(n, 3)[order])
        self.pose_rot = np.ascontiguousarray(np.asarray(points["pose_rotation"], dtype=F64).reshape(n, 4)[order])
        # The pose of the frame being simulated and of the previous frame, for substep interpolation.
        self.frame_pose, self.frame_pose_rot = self.pose.copy(), self.pose_rot.copy()
        self.prev_pose, self.prev_pose_rot = self.pose.copy(), self.pose_rot.copy()
        self.pose_initialized = False
        self.step_start_loc = None
        self.frame_pose_scale = np.ones((n, 3))
        # Children in Kawaii's order (ChildIndices), and the bone whose transform a point's
        # bone-space forces use (ResolveExternalForceBoneTransform).
        self.child_lists = [[] for _ in range(n)]
        for caller in range(n):
            i = int(new_index[caller])
            if self.parent[i] >= 0:
                self.child_lists[int(self.parent[i])].append(i)
        dummy = self.kind != KIND_BONE
        self.tm_point = np.where(dummy, np.where(self.real_parent >= 0, self.real_parent, self.parent),
                                 np.arange(n)).astype(I32)
        self.tm_point = np.where(self.tm_point >= 0, self.tm_point, np.arange(n)).astype(I32)
        self.bone_names = [""] * n                  # a point's bone name, "" for dummies (force filters)

        for name in SETTINGS:
            setattr(self, name, np.zeros(n, dtype=F32))
        self.pull = np.zeros(n, dtype=F32)
        self.bridge_base = np.zeros((n, 3))
        self.push_sum = np.zeros((n, 3))
        self.push_weight = np.zeros(n, dtype=F32)

        g = len(self.groups)
        self.gravity = np.zeros((g, 3))
        self.move = np.zeros((g, 3))
        self.move_rot = np.tile([0.0, 0.0, 0.0, 1.0], (g, 1))
        self.frame_move = np.zeros((g, 3))
        self.frame_move_rot = np.tile([0.0, 0.0, 0.0, 1.0], (g, 1))
        self.teleport = np.zeros(g, dtype=I8)
        self.world_to_sim = np.tile(np.eye(3), (g, 1, 1))
        self.wind = [None] * g                      # (world direction, speed) of the scene's wind, per group
        self.frame_number = 0
        self.skip_known = False                     # Kawaii's bSkipSimulate is set by the first simulated frame
        self._forces = [[] for _ in range(g)]
        self._wind_noise = (None, None)
        self.wind_vel = np.zeros((n, 3))
        self.simple_force = np.zeros((g, 3))
        self.simple_on = np.zeros(g, dtype=I8)
        self._set_force_slots(0, 0)
        self.legacy_gravity = np.array([grp.legacy_gravity for grp in self.groups], dtype=I8)
        self.planar_axis = np.array([grp.planar_constraint for grp in self.groups], dtype=I8)
        self.collision_only = np.array([grp.bone_subdivision_collision_only for grp in self.groups], dtype=I8)
        self.iterations_before = np.array([grp.iterations_before_collision for grp in self.groups], dtype=I32)
        self.iterations_after = np.array([grp.iterations_after_collision for grp in self.groups], dtype=I32)
        self.feedback_scale = np.array(
            [grp.constraint_subdivision_feedback_scale if grp.constraint_subdivision_count > 0 else 0.0
             for grp in self.groups], dtype=F32)

        a = remap(links.get("a", []))
        b = remap(links.get("b", []))
        link_group = self.group[a] if len(a) else np.zeros(0, dtype=I32)
        ctype = np.asarray(links.get("compliance_type", [-1] * len(a)), dtype=I32)
        ctype = np.where(ctype >= 0, ctype, [self.groups[k].compliance_type for k in link_group]) \
            if len(a) else ctype
        self._set_links(a, b, np.asarray(links.get("length", []), dtype=F32), COMPLIANCE[ctype] if len(a) else
                        np.zeros(0, dtype=F32), link_group)

        self.step_dt = F32(1.0) / F32(self.target_framerate)
        self.dt_old = F32(0.0)
        self.exponent = F32(1.0)
        self.accumulator = F32(0.0)
        self.consume_fraction = F32(1.0)
        self._prepare_numpy()
        self.set_shapes([[] for _ in self.groups])

    # ------------------------------------------------------------------ links
    def _set_links(self, a, b, length, compliance, link_group):
        """Links in batches that share no point; the order within the list is the solve order.
        A link without a positive length is skipped, as FModifyBoneConstraint::IsValid does."""
        keep = np.flatnonzero(length > 0)
        a, b, length, compliance, link_group = a[keep], b[keep], length[keep], compliance[keep], link_group[keep]
        batches, left = [], list(range(len(a)))
        while left:
            used, batch, rest = set(), [], []
            for k in left:
                if a[k] in used or b[k] in used:
                    rest.append(k)
                else:
                    batch.append(k)
                    used.update((int(a[k]), int(b[k])))
            batches.append(batch)
            left = rest
        flat = [k for batch in batches for k in batch]
        self.link_a = np.ascontiguousarray(a[flat], dtype=I32)
        self.link_b = np.ascontiguousarray(b[flat], dtype=I32)
        self.link_length = np.ascontiguousarray(length[flat], dtype=F32)
        self.link_compliance = np.ascontiguousarray(compliance[flat], dtype=F32)
        self.link_group = np.ascontiguousarray(link_group[flat], dtype=I32)
        self.lambda_ = np.zeros(len(flat), dtype=F32)
        self.link_batches, start = [], 0
        for batch in batches:
            self.link_batches.append(slice(start, start + len(batch)))
            start += len(batch)

    # ----------------------------------------------------------------- shapes
    def set_shapes(self, shapes_by_group):
        """This frame's collision limits per group, in Kawaii's order by type.

        Their runtime caches (KawaiiPhysicsCollisionLimits.h UpdateRuntimeCache)
        are computed here; Kawaii recomputes them every substep from the same values.
        """
        ordered = []
        start, count = [], []
        for group_shapes in shapes_by_group:
            start.append(len(ordered))
            # Shapes Kawaii would skip (disabled, or degenerate: Collision.cpp:1087,1167,1194).
            usable = [s for s in group_shapes if s.enabled and not (
                (s.kind in (SPHERE_OUTER, SPHERE_INNER) and F32(s.radius) <= 0)
                or (s.kind == CAPSULE and (F32(s.radius) <= 0 or F32(s.length) <= 0))
                or (s.kind == TAPERED and F32(s.radius) <= 0 and F32(s.radius1) <= 0))]
            ordered += sorted(usable, key=lambda s: TYPE_RANK[s.kind])
            count.append(len(ordered) - start[-1])
        m = len(ordered)
        self.shape_start = np.array(start, dtype=I32)
        self.shape_count = np.array(count, dtype=I32)
        self.shape_type = np.array([s.kind for s in ordered], dtype=I8)
        loc = np.array([s.location for s in ordered], dtype=F64).reshape(m, 3)
        rot = np.array([s.rotation for s in ordered], dtype=F64).reshape(m, 4)
        self.shape_loc = np.ascontiguousarray(loc)
        self.shape_rot = np.ascontiguousarray(rot)
        self.shape_radius0 = np.array([s.radius for s in ordered], dtype=F32)
        self.shape_radius1 = np.array([s.radius1 for s in ordered], dtype=F32)
        self.shape_length = np.array([s.length for s in ordered], dtype=F32)
        self.shape_extent = np.ascontiguousarray(np.array([s.extent for s in ordered], dtype=F64).reshape(m, 3))
        axis_x = ue.axis(rot, 0) if m else np.zeros((0, 3))
        axis_z = ue.axis(rot, 2) if m else np.zeros((0, 3))
        half = (axis_z * self.shape_length.astype(F64)[:, None]) * F64(F32(0.5))
        self.shape_start_point = np.ascontiguousarray(loc + half)
        # Capsule: Location + AxisZ * Length * -0.5f; tapered: Location - AxisZ * Length * 0.5f.
        neg_half = (axis_z * self.shape_length.astype(F64)[:, None]) * F64(F32(-0.5))
        end = np.where((self.shape_type == TAPERED)[:, None], loc - half, loc + neg_half)
        self.shape_end_point = np.ascontiguousarray(end)
        segment = end - (loc + half)
        self.shape_segment = np.ascontiguousarray(segment)
        self.shape_segment_sq = np.ascontiguousarray(ue.size_squared(segment)) if m else np.zeros(0)
        self.shape_fallback_dir = np.ascontiguousarray(axis_x)
        self.shape_normal = np.ascontiguousarray(axis_z)
        self.shape_plane_w = np.ascontiguousarray(ue.dot(loc, axis_z)) if m else np.zeros(0)
        # Tapered capsule sphere fallback (UsesSphereFallback / GetFallbackSphere*).
        r0 = np.maximum(self.shape_radius0, F32(0.0))
        r1 = np.maximum(self.shape_radius1, F32(0.0))
        eff_length = np.maximum(self.shape_length, F32(0.0))
        self.shape_fallback = (eff_length <= np.abs(r0 - r1) + ue.KINDA_SMALL).astype(I8)
        self.shape_fallback_radius = np.maximum(r0, r1)
        direction = np.where(r0 >= r1, F32(1.0), F32(-1.0))
        self.shape_fallback_center = np.ascontiguousarray(
            loc + (axis_z * (eff_length * F32(0.5)).astype(F64)[:, None]) * direction.astype(F64)[:, None])
        self._prepare_slots()

    # ---------------------------------------------------------------- settings
    def resolve_settings(self):
        """Each point's settings: the group's value times its curve at the point's length rate
        (UpdatePhysicsSettingsOfModifyBones, Simulation.cpp:203; multipliers are always 1)."""
        for index, grp in enumerate(self.groups):
            rows = self.group == index
            rate = self.length_rate[rows]

            def curve(name):
                fn = grp.curves.get(name)
                if fn is None:
                    return np.ones(rows.sum(), dtype=F32)
                if hasattr(fn, "many"):
                    return fn.many(rate)
                return np.asarray([fn(float(r)) for r in rate], dtype=F32)

            base = {name: F32(grp.settings[name]) for name in SETTINGS}
            clamp01 = lambda v: np.clip(v, F32(0.0), F32(1.0)).astype(F32)
            self.damping[rows] = clamp01(base["damping"] * curve("damping"))
            self.world_damping_location[rows] = clamp01(base["world_damping_location"] * curve("world_damping_location"))
            self.world_damping_rotation[rows] = clamp01(base["world_damping_rotation"] * curve("world_damping_rotation"))
            self.stiffness[rows] = clamp01(base["stiffness"] * curve("stiffness"))
            self.radius[rows] = np.maximum(base["radius"] * curve("radius"), F32(0.0))
            limit = np.maximum(base["limit_angle"] * curve("limit_angle"), F32(0.0))
            self.limit_angle[rows] = np.where(limit > 0, np.maximum(limit, ue.KINDA_SMALL), F32(0.0)).astype(F32)

    # ------------------------------------------------------- numpy index sets
    def _prepare_numpy(self):
        parent_ok = self.parent >= 0
        self.skip = (~parent_ok) & (self.kind != KIND_BRIDGE)
        self.root_rows = np.flatnonzero(self.skip)
        collision_only = self.collision_only[self.group].astype(bool)
        simulated = parent_ok & (self.kind != KIND_BRIDGE) & ~((self.kind == KIND_INTER) & collision_only)
        self.sim_rows = np.flatnonzero(simulated)
        self.sim_levels = [rows for rows in (self.sim_rows[self.depth[self.sim_rows] == d]
                                             for d in range(1, int(self.depth.max(initial=0)) + 1)) if len(rows)]
        self.inter_rows = np.flatnonzero((self.kind == KIND_INTER) & collision_only)
        self.bridge_rows = np.flatnonzero(self.kind == KIND_BRIDGE)
        self.collide_rows = np.flatnonzero(~self.skip)
        limited = parent_ok & (self.kind != KIND_BRIDGE)
        self.limit_levels = [rows for rows in (np.flatnonzero(limited & (self.depth == d))
                                               for d in range(1, int(self.depth.max(initial=0)) + 1)) if len(rows)]
        self.bridge_feedback_rows = self.bridge_rows[self.feedback_scale[self.group[self.bridge_rows]] > 0]

    def _prepare_slots(self):
        """Collision passes for numpy: the k-th shape of one type in every group at once.

        Kawaii tests each point against spheres, then capsules, tapered capsules,
        boxes and planes, each in list order; a slot is one such position."""
        self.slots = []
        rows = getattr(self, "collide_rows", None)
        if rows is None or not len(rows):
            return
        groups = self.group[rows]
        type_order = ((SPHERE_OUTER, SPHERE_INNER), (CAPSULE,), (TAPERED,), (BOX,), (PLANE,))
        for kinds in type_order:
            per_group = []
            for g in range(len(self.groups)):
                idx = [self.shape_start[g] + k for k in range(self.shape_count[g])
                       if self.shape_type[self.shape_start[g] + k] in kinds]
                per_group.append(idx)
            depth = max((len(ix) for ix in per_group), default=0)
            for k in range(depth):
                shape_of_group = np.array([ix[k] if k < len(ix) else -1 for ix in per_group], dtype=I32)
                shape = shape_of_group[groups]
                mask = shape >= 0
                if mask.any():
                    self.slots.append((rows[mask], shape[mask]))

    # ------------------------------------------------------------------ frames
    def update_dummy_poses(self, subdivided_tips_only=False):
        """Tip and inter-bone dummy poses from the real bones' (UpdateModifyBonesPoseTransform,
        ModifyBones.cpp:520). Call after setting the real bones' frame pose. After sync
        bones only subdivided tips are updated (UpdateSubdivisionDummyPoseAfterSyncBones)."""
        pose, rot = self.frame_pose, self.frame_pose_rot
        tips = self.kind == KIND_TIP
        if subdivided_tips_only:
            tips &= self.real_parent >= 0
        tips = np.flatnonzero(tips)
        if len(tips):
            ancestor = np.where(self.real_parent[tips] >= 0, self.real_parent[tips], self.parent[tips])
            forward = ue.axis(rot[ancestor], self.forward_axis)
            pose[tips] = pose[ancestor] + forward * self.tip_length[tips].astype(F64)[:, None]
            rot[tips] = rot[ancestor]
            self.frame_pose_scale[tips] = self.frame_pose_scale[ancestor]
        inter = np.flatnonzero(self.kind == KIND_INTER)
        if len(inter):
            rp, rc = self.real_parent[inter], self.real_child[inter]
            pose[inter] = ue.lerp(pose[rp], pose[rc], self.alpha[inter])
            for i, p, c, a in zip(inter, rp, rc, self.alpha[inter]):
                rot[i] = ue.slerp(rot[p][None], rot[c][None], a)[0]
            self.frame_pose_scale[inter] = ue.lerp(self.frame_pose_scale[rp], self.frame_pose_scale[rc],
                                                   self.alpha[inter])

    def prepare_frame(self):
        """This frame's pose targets, once, before warm-up and simulation: dummy poses, then
        sync bones (EvaluateSkeletalControl: UpdateModifyBonesPoseTransform, ApplySyncBones)."""
        self.update_dummy_poses()
        synced = False
        for grp in self.groups:
            for spec in grp.sync_bones:
                frame_forces.apply_sync(self, spec)
                synced = True
        if synced:
            self.update_dummy_poses(subdivided_tips_only=True)

    # ------------------------------------------------------------------ forces
    def _set_force_slots(self, velocity_slots, position_slots):
        n = self.n
        self.vforce = np.zeros((velocity_slots, n, 3))
        self.vmask = np.zeros((velocity_slots, n), dtype=I8)
        self.pforce = np.zeros((position_slots, n, 3))
        self.pmask = np.zeros((position_slots, n), dtype=I8)

    def _force_state(self, g):
        """The group's force objects, remade where a force was added, removed or changed kind."""
        specs = self.groups[g].forces
        current = self._forces[g]
        kept = []
        for k, spec in enumerate(specs):
            found = current[k] if k < len(current) and current[k].spec.kind == spec.kind else None
            if found is None:
                found = frame_forces.make(spec)
            found.spec = spec
            kept.append(found)
        self._forces[g] = kept
        return kept

    def force_states(self):
        """Every force's clock, for the cache."""
        return [[force.state() for force in group] for group in self._forces]

    def set_force_states(self, states):
        for g, group_states in enumerate(states):
            for force, state in zip(self._force_state(g), group_states):
                force.set_state(state)

    def pre_apply(self, frame_dt, call):
        """Every group's external forces and wind for this SimulateModifyBones (Simulation.cpp:645)."""
        rng = frame_forces.RandomStream(frame_forces.stable_hash(0x53574953, self.frame_number, call))
        if self._wind_noise[0] != self.frame_number:
            wind_rng = frame_forces.RandomStream(frame_forces.stable_hash(0x57494E44, self.frame_number, 0))
            self._wind_noise = (self.frame_number, [
                frame_forces.scene_wind_noise(wind_rng, grp.wind_direction_noise_angle) for grp in self.groups])
        velocity, position = [], []
        self.wind_vel[:] = 0.0
        for g, grp in enumerate(self.groups):
            simple = np.asarray(grp.simple_external_force, dtype=F64)
            on = not frame_forces._nearly_zero(simple)
            if on and grp.world_space_simple_external_force:
                simple = self.world_to_sim[g] @ simple
            self.simple_force[g] = simple if on else 0.0
            self.simple_on[g] = on
            forces = [f for f in self._force_state(g) if f.spec.enabled]
            if not forces and not grp.enable_wind:
                continue
            rows = np.flatnonzero(self.group == g)
            ctx = frame_forces.FrameContext(self, g, rows, rng, frame_dt, self.wind[g])
            if grp.enable_wind:
                gust, rotation = self._wind_noise[1][g]
                settings = dict(wind_scale=grp.wind_scale, wind_noise_angle=grp.wind_direction_noise_angle)
                self.wind_vel[rows] = frame_forces.scene_wind_velocity(ctx, settings, gust, rotation,
                                                                       self.target_framerate)
            v_slot = p_slot = 0
            for force in forces:
                kind, vectors, mask = force.pre_apply(ctx)
                if kind == "velocity":
                    velocity.append((v_slot, rows, vectors, mask))
                    v_slot += 1
                else:
                    position.append((p_slot, rows, vectors, mask))
                    p_slot += 1
        v_count = 1 + max((slot for slot, *_ in velocity), default=-1)
        p_count = 1 + max((slot for slot, *_ in position), default=-1)
        if self.vforce.shape[0] != v_count or self.pforce.shape[0] != p_count:
            self._set_force_slots(v_count, p_count)
        else:
            self.vmask[:] = 0
            self.pmask[:] = 0
        for slot, rows, vectors, mask in velocity:
            self.vforce[slot, rows] = vectors
            self.vmask[slot, rows] = mask
        for slot, rows, vectors, mask in position:
            self.pforce[slot, rows] = vectors
            self.pmask[slot, rows] = mask

    forward_axis = 1   # Blender bones point along +Y (Kawaii's BoneForwardAxis Y_Positive)

    def begin(self, pose, pose_rotation):
        """First frame: points start at the pose (AddModifyBone, ModifyBones.cpp:142)."""
        self.frame_pose[:] = pose
        self.frame_pose_rot[:] = pose_rotation
        self.update_dummy_poses()
        self.loc[:] = self.frame_pose
        self.prev[:] = self.frame_pose
        self.pose[:] = self.frame_pose
        self.pose_rot[:] = self.frame_pose_rot
        self.pose_initialized = False
        self.accumulator = F32(0.0)
        self.skip_known = False
        self._forces = [[] for _ in self.groups]
        self.step_start_loc = None

    def step_frame(self, frame_dt, backend, prepare=True, call=0):
        """SimulateModifyBones (Simulation.cpp:597): fixed substeps or one legacy step.

        Before calling: frame_pose / frame_pose_rot hold this frame's real-bone pose
        (dummies are derived here), frame_move / frame_move_rot this frame's component
        movement per group, teleport per group, gravity per group in simulation space.
        prepare=False reuses the pose targets already prepared for this frame (warm-up
        steps after the first); call numbers the steps of one frame for its random draws."""
        frame_dt = F32(frame_dt)
        if frame_dt <= 0:
            return
        if prepare:
            self.prepare_frame()
        self.pre_apply(frame_dt, call)
        if not self.pose_initialized:
            self.prev_pose[:] = self.frame_pose
            self.prev_pose_rot[:] = self.frame_pose_rot
        self.pose_initialized = True
        if not self.fixed_substepping:
            if self.dt_old <= 0:
                self.dt_old = F32(1.0) / F32(self.target_framerate)
            self.pose[:] = self.frame_pose
            self.pose_rot[:] = self.frame_pose_rot
            self.move[:] = self.frame_move
            self.move_rot[:] = self.frame_move_rot
            self.step_dt = frame_dt
            self.exponent = F32(self.target_framerate) * frame_dt
            self._substep(backend)
            self.dt_old = frame_dt
            self.consume_fraction = F32(1.0)
        else:
            fixed_dt = F32(1.0) / F32(self.target_framerate)
            raw = max(self.accumulator + frame_dt, ue.KINDA_SMALL)
            self.accumulator = min(raw, F32(self.max_substeps) * fixed_dt)
            dropped = raw - self.accumulator
            steps = int(np.floor(self.accumulator / fixed_dt))
            self.accumulator = self.accumulator - F32(steps) * fixed_dt
            move_frac = fixed_dt / raw
            self.consume_fraction = F32(min(max((F32(steps) * fixed_dt + dropped) / raw, F32(0.0)), F32(1.0)))
            self.step_dt = fixed_dt
            self.dt_old = fixed_dt
            self.exponent = F32(self.target_framerate) * fixed_dt
            move = self.frame_move * F64(move_frac)
            identity = np.tile([0.0, 0.0, 0.0, 1.0], (len(self.groups), 1))
            move_rot = ue.slerp(identity, self.frame_move_rot, move_frac)
            for k in range(steps):
                alpha = F32(k + 1) / F32(steps)
                self.pose[:] = ue.lerp(self.prev_pose, self.frame_pose, alpha)
                self.pose_rot[:] = ue.slerp(self.prev_pose_rot, self.frame_pose_rot, alpha)
                self.move[:] = move
                self.move_rot[:] = move_rot
                self._substep(backend)
            self.pose[:] = self.frame_pose
            self.pose_rot[:] = self.frame_pose_rot
        self.prev_pose[:] = self.frame_pose
        self.prev_pose_rot[:] = self.frame_pose_rot
        self.skip_known = True

    def _substep(self, backend):
        self.step_start_loc = self.loc.copy()        # for showing the chains between steps
        self.pull[:] = backend.pull(self.stiffness, self.exponent)
        backend.simulate_once(self)

    def unsorted(self, values):
        """Per-point values back in the caller's point order."""
        out = np.empty_like(values)
        out[self.order] = values
        return out

    # ----------------------------------------------------------------- results
    def results(self):
        """ApplySimulateResult (Simulation.cpp:1396): per point, the rotation its bone takes,
        and whether it takes one. A bone with a single child turns so that its
        pose direction to the child points at the child's simulated position."""
        rotation = self.pose_rot.copy()
        turned = np.zeros(self.n, dtype=bool)
        child_count = np.bincount(self.parent[self.parent >= 0], minlength=self.n)
        children = np.flatnonzero(self.parent >= 0)
        single = children[(child_count[self.parent[children]] <= 1) & (self.kind[self.parent[children]] == KIND_BONE)
                          & (self.kind[children] != KIND_BRIDGE)]
        if len(single):
            parents = self.parent[single]
            pose_vector = self.pose[single] - self.pose[parents]
            sim_vector = self.loc[single] - self.loc[parents]
            same = np.all(ue.safe_normal(pose_vector) == ue.safe_normal(sim_vector), axis=1)
            keep = ~same
            parents, pose_vector, sim_vector = parents[keep], pose_vector[keep], sim_vector[keep]
            rotation[parents] = ue.quat_multiply(ue.find_between(pose_vector, sim_vector), self.pose_rot[parents])
            turned[parents] = True
        return rotation, turned
