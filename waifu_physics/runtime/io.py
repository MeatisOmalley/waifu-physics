"""Armatures in and out of the solver: pose input, rotations back, in bulk.

The solver works in Kawaii's units, centimetres, in the armature's own space
(Kawaii's component space). Input is the animated pose: every bone as Blender
evaluated it, except the simulated chain bones, whose pose is rebuilt from
their keyed channels (or their rest, where a channel is not keyed) -- their
evaluated matrices still hold the previous frame's physics. Output is each
chain bone's local rotation, written in the bone's own rotation mode.
"""
import numpy as np
from mathutils import Euler, Matrix, Quaternion

from . import keys as chain_keys

EULER_ORDERS = {1: "XYZ", 2: "XZY", 3: "YXZ", 4: "YZX", 5: "ZXY", 6: "ZYX"}   # rotation_mode as foreach_get reads it
QUATERNION, AXIS_ANGLE = 0, -1
INHERIT_SCALE = {"FULL": 0, "FIX_SHEAR": 1, "ALIGNED": 2, "AVERAGE": 3, "NONE": 4, "NONE_LEGACY": 5}
FULL, FIX_SHEAR, ALIGNED, AVERAGE, NONE, NONE_LEGACY = range(6)
FLT_EPSILON = 1.1920929e-07
# How far a bone's frame may be from a rotation and one scale before the write corrects for it (_distortion).
# Measured: Blender's single-precision pose matrices alone leave frames up to 6e-6 off, while a slider scale
# uneven by just 0.1% puts them 7e-4 off. Below this the error is under 1e-4 radians a bone: micrometres.
DISTORTION = 1e-4


def cm_per_unit(scene):
    """Centimetres in one Blender unit."""
    return 100.0 * scene.unit_settings.scale_length


def animated_channels(obj, owned=frozenset()):
    """{bone name: {"location", "rotation", "scale"}} that animation drives: the action's
    slot, unmuted NLA strips and drivers. A muted curve drives nothing, unless it is one of the
    curves Waifu Physics muted to sample itself (owned: (data path, index)): a curve the user muted
    leaves its channel at rest, as if it had no keys."""
    found = {}
    data = obj.animation_data
    if data is None:
        return found

    def note(path):
        if not path.startswith('pose.bones["'):
            return
        end = path.find('"]', 12)
        if end < 0:
            return
        name, prop = path[12:end], path[end + 3:]
        kind = "rotation" if prop.startswith("rotation_") else prop
        if kind in ("location", "rotation", "scale"):
            found.setdefault(name, set()).add(kind)

    def scan(action, slot):
        if action is None or slot is None:
            return
        for layer in getattr(action, "layers", ()):
            for strip in layer.strips:
                bag = strip.channelbag(slot)
                if bag is not None:
                    for curve in bag.fcurves:
                        if not curve.mute or (curve.data_path, curve.array_index) in owned:
                            note(curve.data_path)

    scan(data.action, data.action_slot)
    for track in data.nla_tracks:
        if not track.mute:
            for strip in track.strips:
                if not strip.mute:
                    scan(strip.action, getattr(strip, "action_slot", None))
    for driver in data.drivers:
        note(driver.data_path)
    return found


def _nearest_rotations(m):
    """The rotation nearest each 3x3 matrix (its polar factor)."""
    u, _singular, vt = np.linalg.svd(m)
    flip = np.linalg.det(u @ vt) < 0
    u[flip, :, -1] *= -1
    return u @ vt


def _aimed(local, frame, direction):
    """Local rotations turned (the shortest way) so the bone's Y axis, through its frame, points along
    direction (armature space). Where the frame has no skew this changes nothing."""
    want = np.linalg.solve(frame, direction[:, :, None])[:, :, 0]
    want /= np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-12)
    have = local[:, :, 1]
    hx, hy, hz = have[:, 0], have[:, 1], have[:, 2]
    wx, wy, wz = want[:, 0], want[:, 1], want[:, 2]
    ax, ay, az = hy * wz - hz * wy, hz * wx - hx * wz, hx * wy - hy * wx       # have x want
    cos = hx * wx + hy * wy + hz * wz
    k = np.zeros((len(local), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -az, ay, -ax
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = az, -ay, ax
    # Rodrigues' turn, I + K + K^2 / (1 + cos); opposite directions have no shortest turn: left as they are.
    ok = cos > -0.999999
    turn = k @ k / np.where(ok, 1.0 + cos, 1.0)[:, None, None] + k
    turn[:, 0, 0] += 1.0
    turn[:, 1, 1] += 1.0
    turn[:, 2, 2] += 1.0
    turn[~ok] = np.eye(3)
    return turn @ local


def _distortion(frame):
    """Which frames are not a rotation and one scale (scaled unevenly, skewed or mirrored), and of those, which
    are skewed or mirrored. Distorted, a local rotation needs _aimed; skewed or mirrored, it is no rotation at
    all through the frame and needs _nearest_rotations first. Scaled unevenly but square (Inherit Scale Aligned)
    it is a rotation already. Everywhere else both change nothing, so they are skipped."""
    gram = np.einsum("nji,njk->nik", frame, frame)
    lengths_sq = np.einsum("nii->ni", gram)
    size = lengths_sq.sum(axis=1) / 3.0
    mirrored = np.linalg.det(frame) < 0
    even = gram / np.where(size > 0.0, size, 1.0)[:, None, None]
    distorted = (np.abs(even - np.eye(3)).max(axis=(1, 2)) > DISTORTION) | mirrored
    lengths = np.sqrt(np.where(lengths_sq > 0.0, lengths_sq, 1.0))
    square = gram / (lengths[:, :, None] * lengths[:, None, :])
    skewed = (np.abs(square - np.eye(3)).max(axis=(1, 2)) > DISTORTION) | mirrored
    return distorted, skewed & distorted


def _normalized(v):
    """Blender's normalize_v3_v3: unit vectors and their lengths; zero for (near) zero vectors."""
    length = np.linalg.norm(v, axis=1)
    ok = length > 1e-35
    return np.where(ok[:, None], v / np.where(ok, length, 1.0)[:, None], 0.0), np.where(ok, length, 0.0)


def _orthogonalized(m, normalize):
    """Blender's orthogonalize_m4_stable(m, 1, normalize) on 3x3 matrices (axes as columns): X and Z made
    square to Y, then to each other by turning both by the same angle. Unnormalized, they keep their area."""
    y, x, z = m[:, :, 1].copy(), m[:, :, 0].copy(), m[:, :, 2].copy()
    len_sq = np.einsum("ni,ni->n", y, y)
    ok = len_sq > 0.0
    safe = np.where(ok, len_sq, 1.0)
    x -= y * np.where(ok, np.einsum("ni,ni->n", x, y) / safe, 0.0)[:, None]
    z -= y * np.where(ok, np.einsum("ni,ni->n", z, y) / safe, 0.0)[:, None]
    if normalize:
        y = np.where(ok[:, None], y / np.sqrt(safe)[:, None], y)
    norm_x, length_x = _normalized(x)
    norm_z, length_z = _normalized(z)
    cos = np.einsum("ni,ni->n", norm_x, norm_z)
    fix = (np.abs(cos) > 1e-4) & (np.abs(cos) < 1.0 - FLT_EPSILON)
    if fix.any():
        c, nx, nz = cos[fix], norm_x[fix], norm_z[fix]
        angle = np.arccos(np.clip(c, -1.0, 1.0))
        target = angle + (np.pi / 2 - angle) / 2
        nx = nx - nz * c[:, None]
        nx = nx * (np.sin(target) / np.linalg.norm(nx, axis=1))[:, None] + nz * np.cos(target)[:, None]
        nz, _length = _normalized(np.cross(np.cross(nx, nz), nx))
        norm_x[fix], norm_z[fix] = nx, nz
        if not normalize:
            area = np.sqrt(np.sin(angle))
            x[fix] = nx * (length_x[fix] * area)[:, None]
            z[fix] = nz * (length_z[fix] * area)[:, None]
    if normalize:
        x, z = norm_x, norm_z
    out = np.empty_like(m)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = x, y, z
    return out


def _size_fix_shear(m):
    """Blender's mat4_to_size_fix_shear on 3x3 matrices: the axes' lengths, evened out to the true volume."""
    size = np.linalg.norm(m, axis=1)
    volume = size.prod(axis=1)
    ok = volume != 0.0
    factor = np.cbrt(np.abs(np.linalg.det(m) / np.where(ok, volume, 1.0)))
    return size * np.where(ok, factor, 1.0)[:, None]


def _placed(rotscale, loc, post, basis):
    """Blender's BKE_bone_parent_transform_apply: bones' pose matrices from their parent transforms and
    basis matrices."""
    out = rotscale @ basis
    out[:, :3, 3] = np.einsum("nij,nj->ni", loc[:, :3, :3], basis[:, :3, 3]) + loc[:, :3, 3]
    if post is not None:
        out[:, :3, :3] *= post[:, None, :]
    return out


def unscaled(m):
    """The rotation of matrices that may carry scale: columns normalised."""
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def quats_from_matrices(m):
    """Rotation matrices to quaternions in Unreal's (x, y, z, w) order (Shepperd's method)."""
    q = np.empty((len(m), 4))
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    trace = m00 + m11 + m22
    w_big = trace > 0.0                                   # Shepperd's four cases, in the same order
    x_big = ~w_big & (m00 > m11) & (m00 > m22)
    y_big = ~w_big & ~x_big & (m11 > m22)
    z_big = ~(w_big | x_big | y_big)
    k = w_big
    s = np.sqrt(trace[k] + 1.0) * 2.0
    q[k] = np.stack([(m21[k] - m12[k]) / s, (m02[k] - m20[k]) / s, (m10[k] - m01[k]) / s, 0.25 * s], axis=1)
    k = x_big
    s = np.sqrt(1.0 + m00[k] - m11[k] - m22[k]) * 2.0
    q[k] = np.stack([0.25 * s, (m01[k] + m10[k]) / s, (m02[k] + m20[k]) / s, (m21[k] - m12[k]) / s], axis=1)
    k = y_big
    s = np.sqrt(1.0 + m11[k] - m00[k] - m22[k]) * 2.0
    q[k] = np.stack([(m01[k] + m10[k]) / s, 0.25 * s, (m12[k] + m21[k]) / s, (m02[k] - m20[k]) / s], axis=1)
    k = z_big
    s = np.sqrt(1.0 + m22[k] - m00[k] - m11[k]) * 2.0
    q[k] = np.stack([(m02[k] + m20[k]) / s, (m12[k] + m21[k]) / s, 0.25 * s, (m10[k] - m01[k]) / s], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def matrices_from_quats(q):
    """(x, y, z, w) quaternions to rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


class Rig:
    """One armature's bones, read and written in bulk."""

    def __init__(self, obj):
        self.obj = obj
        self.uid, self.data_uid = obj.session_uid, obj.data.session_uid
        bones = obj.pose.bones
        self.count = len(bones)
        self.names = [pb.name for pb in bones]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.parents = np.array([self.index[pb.parent.name] if pb.parent else -1 for pb in bones])
        rest = np.array([np.array(pb.bone.matrix_local) for pb in bones]).reshape(self.count, 4, 4)
        self.rest = rest
        parent_rest = np.where((self.parents >= 0)[:, None, None], rest[np.maximum(self.parents, 0)], np.eye(4))
        self.parent_rest = parent_rest
        self.rest_rel = np.linalg.inv(parent_rest) @ rest              # rest offset from the parent
        self.inheritance = self._inheritance()
        flags = np.array(self.inheritance, dtype=int).reshape(-1, 3)
        self.inherit_scale, self.hinge, self.no_local_location = flags[:, 0], flags[:, 1] == 1, flags[:, 2] == 1
        # Bones that take all of their parent's pose (Blender's default): parent pose @ rest offset @ basis.
        self.plain = (self.inherit_scale == FULL) & ~self.hinge & ~self.no_local_location
        self._matrices = np.empty(self.count * 16, dtype=np.float32)
        self._modes = np.empty(self.count, dtype=np.int32)
        self._buffers = {path: np.empty(self.count * size, dtype=np.float32) for path, size in
                         (("location", 3), ("rotation_quaternion", 4), ("rotation_euler", 3),
                          ("rotation_axis_angle", 4), ("scale", 3))}
        self.chain = np.zeros(self.count, dtype=bool)
        self.chain_levels = []
        self.keyed = {}
        self.keys = None
        self.constrained = np.zeros(self.count, dtype=bool)
        self.input_keys = None                 # chain_keys.InputKeys: what the chains hang from, sampled ahead
        self.ahead_levels = []                 # sampled bones and chain bones by depth, parents first
        self.world = obj.matrix_world.copy()   # the armature's world matrix this frame, as far as it is known
        self.evaluated = None                  # the pose and world matrix as Blender last evaluated them
        self.evaluated_world = self.world

    def alive(self):
        """The armature is still there with the bones this rig was built for. Its bones are read and written
        by index, so a deleted object, a swapped armature or bones added or removed make the rig unusable
        (touching a removed object's pose can crash Blender)."""
        try:
            obj = self.obj
            return (obj.session_uid == self.uid and obj.type == "ARMATURE" and obj.data is not None
                    and obj.data.session_uid == self.data_uid and obj.pose is not None
                    and len(obj.pose.bones) == self.count)
        except ReferenceError:
            return False

    def _inheritance(self):
        """Per bone: its Inherit Scale, and whether Inherit Rotation and Local Location are off."""
        return [(INHERIT_SCALE[pb.bone.inherit_scale], not pb.bone.use_inherit_rotation,
                 not pb.bone.use_local_location) for pb in self.obj.pose.bones]

    def same_inheritance(self):
        """The bones still take their parents' pose the way this rig was built for."""
        return self.alive() and self._inheritance() == self.inheritance

    def same_bones(self):
        """Bones by the same names in the same order: renames too mean rebuilding."""
        return self.alive() and [pb.name for pb in self.obj.pose.bones] == self.names

    # ------------------------------------------------------------------ setup
    def set_chain(self, bone_indices):
        """The bones the solver moves; their pose is rebuilt from keyed channels, parents first."""
        self.chain[:] = False
        self.chain[list(bone_indices)] = True
        depth = np.zeros(self.count, dtype=int)
        for i in range(self.count):
            p = self.parents[i]
            depth[i] = depth[p] + 1 if p >= 0 else 0
        rows = np.flatnonzero(self.chain)
        self.chain_levels = [rows[depth[rows] == d] for d in sorted(set(depth[rows]))]
        self.refresh_keyed()
        if self.keys is not None:
            self.keys.unmute()
        self.keys = chain_keys.ChainKeys(self)
        bones = self.obj.pose.bones
        self.constrained = np.array([any(c.enabled and c.influence > 0 for c in pb.constraints) for pb in bones])

    def prepare_input(self, extra=()):
        """What the one-evaluation path reads for this frame before Blender evaluates it (read_ahead): the
        bones above the chains, and extra ones (bones colliders hang from) with theirs, where their keys can
        be sampled (chain_keys.InputKeys); the rest stays Blender's last evaluation."""
        wanted = set()
        for i in list(np.flatnonzero(self.chain)) + [int(b) for b in extra]:
            p = self.parents[i] if self.chain[i] else i
            while p >= 0:
                if not self.chain[p]:
                    wanted.add(int(p))
                p = self.parents[p]
        self.input_keys = chain_keys.InputKeys(self, wanted)
        rows = np.union1d(self.input_keys.bones, np.flatnonzero(self.chain)).astype(int)
        depth = np.zeros(self.count, dtype=int)
        for i in range(self.count):
            p = self.parents[i]
            depth[i] = depth[p] + 1 if p >= 0 else 0
        self.ahead_levels = [rows[depth[rows] == d] for d in sorted(set(depth[rows]))]

    def subtree(self, roots, excluded):
        """Bones under the roots, as Kawaii collects them: an excluded bone cuts off its subtree."""
        excluded = set(excluded)
        found, stack = [], [self.index[r] for r in roots if r in self.index and r not in excluded]
        children = {}
        for i, p in enumerate(self.parents):
            children.setdefault(int(p), []).append(i)
        while stack:
            i = stack.pop()
            found.append(i)
            stack += [c for c in children.get(i, []) if self.names[c] not in excluded]
        return sorted(set(found))

    def refresh_keyed(self):
        keys = self.keys
        owned = {(c.data_path, c.array_index) for c, *_ in keys.curves} if keys is not None and keys.muted else ()
        channels = animated_channels(self.obj, frozenset(owned))
        self.keyed = {kind: np.array([kind in channels.get(name, ()) for name in self.names])
                      for kind in ("location", "rotation", "scale")}

    def ref_lengths(self):
        """Each bone's rest offset from its parent (Kawaii's BoneLength), in Blender units."""
        return np.linalg.norm(self.rest_rel[:, :3, 3], axis=1)

    # ------------------------------------------------------------------ input
    def read(self):
        """The input pose, Blender's evaluated matrices in armature space, and the chain bones'
        local basis matrices from their keyed channels.

        The evaluated pose is the input only because restore() cleared last frame's physics
        from the chain channels before Blender evaluated it (frame_change_pre), so constraints,
        drivers and IK on any bone reach the solver."""
        bones = self.obj.pose.bones
        bones.foreach_get("matrix", self._matrices)
        evaluated = self._matrices.reshape(self.count, 4, 4).transpose(0, 2, 1).astype(np.float64)
        bones.foreach_get("rotation_mode", self._modes)
        for path, buffer in self._buffers.items():
            bones.foreach_get(path, buffer)
        self.basis = self._basis()
        self.pose = evaluated
        self.evaluated = evaluated
        self.world = self.evaluated_world = self.obj.matrix_world.copy()
        return evaluated

    def read_ahead(self, frame=None):
        """The input before Blender evaluates the frame (live's one-evaluation path). The chain is rebuilt from
        its channels, which restore(frame) has just set to this frame's keys or rest; the bones it hangs from
        (and colliders' bones) are rebuilt from their keys sampled for the frame, and the armature's world
        matrix too, where only its action moves them (prepare_input). Everything else -- constrained or driven
        bones and whatever hangs below them -- is as Blender last evaluated it: a frame late."""
        bones = self.obj.pose.bones
        bones.foreach_get("matrix", self._matrices)
        evaluated = self._matrices.reshape(self.count, 4, 4).transpose(0, 2, 1).astype(np.float64)
        bones.foreach_get("rotation_mode", self._modes)
        for path, buffer in self._buffers.items():
            bones.foreach_get(path, buffer)
        basis = self._basis()
        keys = self.input_keys
        if keys is not None and frame is not None and len(keys.bones):
            keys.sample(frame, self._buffers)
            basis[keys.bones] = self._basis_of(keys.bones)
        pose = evaluated.copy()
        levels = self.ahead_levels if keys is not None and frame is not None else self.chain_levels
        for level in levels:
            parent = self.parents[level]
            parent_pose = np.where((parent >= 0)[:, None, None], pose[np.maximum(parent, 0)], np.eye(4))
            rebuilt = _placed(*self._parent_transforms(level, parent_pose), basis[level])
            keep = self.constrained[level] & self.chain[level]            # a constrained chain bone: Blender's
            pose[level] = np.where(keep[:, None, None], evaluated[level], rebuilt)
        world = keys.world(self.obj, frame) if keys is not None and frame is not None else None
        self.evaluated = evaluated
        self.evaluated_world = self.obj.matrix_world.copy()
        self.world = world if world is not None else self.evaluated_world
        self.basis = basis
        self.pose = pose
        return pose

    def _basis_of(self, rows):
        """Local basis matrices from the bones' channels as they are in the buffers, every channel: Blender's
        BKE_pchan_to_mat4 (location, then rotation -- a quaternion normalised -- then scale)."""
        out = np.tile(np.eye(4), (len(rows), 1, 1))
        location = self._buffers["location"].reshape(-1, 3)
        scale = self._buffers["scale"].reshape(-1, 3)
        for k, i in enumerate(rows):
            out[k, :3, :3] = np.array(self._rotation_of(i).to_matrix()) * scale[i]
            out[k, :3, 3] = location[i]
        return out

    def _parent_transforms(self, level, parent_pose):
        """Blender's BKE_bone_parent_transform_calc_from_matrices for a level of bones: the matrix their basis
        rotation and scale go through, the one their location goes through, and the scale put on their own
        axes after (Aligned; None when no bone has it). parent_pose: the parents' pose (identity for roots).
        The same rule Blender evaluates bones by, for every Inherit Scale, Inherit Rotation and Local Location."""
        offs = self.rest_rel[level]
        rotscale = parent_pose @ offs
        if self.plain[level].all():
            return rotscale, rotscale, None
        mode, hinge, no_local = self.inherit_scale[level], self.hinge[level], self.no_local_location[level]
        parented = self.parents[level] >= 0
        loc = rotscale.copy()
        post = np.ones((len(level), 3))
        partial = parented & (hinge | (mode != FULL))
        if partial.any():
            pose3 = parent_pose[:, :3, :3]
            tmat = np.where(hinge[:, None, None], self.parent_rest[level], parent_pose)
            t3 = tmat[:, :3, :3].copy()
            turned = partial & ~hinge
            pick = turned & ((mode == NONE) | (mode == AVERAGE))
            t3[pick] = _orthogonalized(t3[pick], True)
            pick = turned & (mode == ALIGNED)
            if pick.any():
                square = _orthogonalized(t3[pick], False)
                size = np.linalg.norm(square, axis=1)
                t3[pick] = square / np.where(size != 0.0, size, 1.0)[:, None, :]
                post[pick] = size
            pick = turned & (mode == NONE_LEGACY)
            size = np.linalg.norm(t3[pick], axis=1)
            t3[pick] /= np.where(size != 0.0, size, 1.0)[:, None, :]
            pick = partial & hinge & (mode == FULL)
            t3[pick] *= np.linalg.norm(pose3[pick], axis=1)[:, None, :]
            pick = partial & hinge & (mode == FIX_SHEAR)
            t3[pick] *= _size_fix_shear(pose3[pick])[:, None, :]
            pick = partial & hinge & (mode == ALIGNED)
            post[pick] = _size_fix_shear(pose3[pick])
            pick = partial & (mode == AVERAGE)
            t3[pick] *= np.cbrt(np.abs(np.linalg.det(pose3[pick])))[:, None, None]
            tmat[:, :3, :3] = t3
            made = tmat @ offs
            pick = partial & (mode == FIX_SHEAR)
            made[pick, :3, :3] = _orthogonalized(made[pick, :3, :3], False)
            rotscale[partial] = made[partial]
        pick = no_local & parented                      # location along the parent's pose axes, not the bone's
        loc[pick, :3, :3] = parent_pose[pick, :3, :3]
        pick = no_local & ~parented                     # a root's: along the armature's axes
        loc[pick, :3, :3] = np.eye(3)
        return rotscale, loc, (post if (post != 1.0).any() else None)

    def _basis(self):
        """The chain bones' local basis matrices from their keyed channels (rest where unkeyed)."""
        basis = np.tile(np.eye(4), (self.count, 1, 1))
        rows = np.flatnonzero(self.chain)
        keyed_loc = rows[self.keyed["location"][rows]]
        keyed_rot = rows[self.keyed["rotation"][rows]]
        keyed_scale = rows[self.keyed["scale"][rows]]
        location = self._buffers["location"].reshape(-1, 3)
        scale = self._buffers["scale"].reshape(-1, 3)
        for i in keyed_rot:
            basis[i, :3, :3] = np.array(self._rotation_of(i).to_matrix())
        for i in keyed_scale:
            basis[i, :3, :3] = basis[i, :3, :3] * scale[i]
        basis[keyed_loc, :3, 3] = location[keyed_loc]
        return basis

    def _rotation_of(self, i):
        mode = int(self._modes[i])
        if mode == QUATERNION:
            return Quaternion(self._buffers["rotation_quaternion"][4 * i:4 * i + 4]).normalized()
        if mode == AXIS_ANGLE:
            angle, x, y, z = self._buffers["rotation_axis_angle"][4 * i:4 * i + 4]
            return Quaternion((x, y, z), angle)
        return Euler(self._buffers["rotation_euler"][3 * i:3 * i + 3], EULER_ORDERS[mode]).to_quaternion()

    # ----------------------------------------------------------------- output
    def write(self, bones, rotation, location=None, move_location=None):
        """Give chain bones their armature-space rotations (x y z w).

        bones: the rig's bone indices; rotation: their rotations; location and
        move_location: armature-space head positions for the bones Kawaii
        places directly (every bone below a group's root), and which those are. Everything is converted to local basis channels, parents
        first, through each bone's parent transform as Blender evaluates it (_parent_transforms), and written
        in one call per channel array."""
        target_rot = np.zeros((self.count, 3, 3))
        target_rot[bones] = matrices_from_quats(rotation)
        placed = np.zeros(self.count, dtype=bool)
        head = np.zeros((self.count, 3))
        if location is not None and move_location is not None:
            placed[bones[move_location]] = True
            head[bones[move_location]] = location[move_location]
        out = self.pose.copy()
        local_rot = np.zeros((self.count, 3, 3))
        local_loc = np.zeros((self.count, 3))
        for level in self.chain_levels:
            parent = self.parents[level]
            parent_out = np.where((parent >= 0)[:, None, None], out[np.maximum(parent, 0)], np.eye(4))
            rotscale, loc, post = self._parent_transforms(level, parent_out)
            frame = rotscale[:, :3, :3]                                 # where the bone's basis starts
            frame_rot = unscaled(frame)
            local = np.einsum("nji,njk->nik", frame_rot, target_rot[level])
            # Under a parent with uneven scale the frame is stretched (and, inherited in full, skewed), so the
            # rotation above does not aim the bone where the simulation did, or is no rotation at all. Blender
            # makes a rotation of whatever the channel holds: predicting the pose from anything else puts every
            # child's frame, and the head placed through it, off (measured: 22 cm at the tip of a VRoid hair
            # strand under a 1.32 x 1.28 head scale). So: the nearest rotation, where the frame is skewed or
            # mirrored, then turned so the bone points where the simulation aimed it, through the frame.
            distorted, skewed = _distortion(frame)
            if distorted.any():
                fixed = local[distorted]
                bent = skewed[distorted]
                if bent.any():
                    fixed[bent] = _nearest_rotations(fixed[bent])
                local[distorted] = _aimed(fixed, frame[distorted], target_rot[level][distorted][:, :, 1])
            basis = self.basis[level].copy()
            scale = np.linalg.norm(basis[:, :3, :3], axis=1)
            basis[:, :3, :3] = local * scale[:, None, :]
            moved = placed[level]
            if moved.any():
                inverse = np.linalg.inv(loc[moved])
                point = np.concatenate([head[level[moved]], np.ones((moved.sum(), 1))], axis=1)
                basis[moved, :3, 3] = np.einsum("nij,nj->ni", inverse, point)[:, :3]
            out[level] = _placed(rotscale, loc, post, basis)
            local_rot[level] = local
            local_loc[level] = basis[:, :3, 3]
        rows = np.flatnonzero(self.chain)
        moved = np.flatnonzero(placed)
        self._write_channels(rows, local_rot[rows], moved, local_loc[moved])
        return out

    def _write_channels(self, rows, local_rot, moved, moved_loc):
        """One foreach_set per changed channel array, each bone in its own rotation mode. rows: the chain bones,
        local_rot their local rotations; moved: the bones placed by location, moved_loc their local locations."""
        bones = self.obj.pose.bones
        changed = set()
        modes = self._modes[rows]
        quat = self._buffers["rotation_quaternion"].reshape(-1, 4)
        euler = self._buffers["rotation_euler"].reshape(-1, 3)
        axis_angle = self._buffers["rotation_axis_angle"].reshape(-1, 4)
        for k, i in enumerate(rows):
            q = Matrix(local_rot[k].tolist()).to_quaternion()
            mode = int(modes[k])
            if mode == QUATERNION:
                quat[i] = q
                changed.add("rotation_quaternion")
            elif mode == AXIS_ANGLE:
                axis, angle = q.to_axis_angle()
                axis_angle[i] = (angle, axis.x, axis.y, axis.z)
                changed.add("rotation_axis_angle")
            elif mode in EULER_ORDERS:
                # Compatible with the current value, so angles do not flip between frames.
                euler[i] = q.to_euler(EULER_ORDERS[mode], Euler(euler[i].tolist(), EULER_ORDERS[mode]))
                changed.add("rotation_euler")
        if len(moved):
            self._buffers["location"].reshape(-1, 3)[moved] = moved_loc
            changed.add("location")
        for path in changed:
            bones.foreach_set(path, self._buffers[path])
        if changed:
            self.obj.update_tag(refresh={"DATA"})

    def restore(self, frame=None):
        """Put the chain bones back to their input channels: keyed values, or rest. Before a frame
        is evaluated this clears last frame's physics, so the evaluated pose is a clean input.
        Keys Waifu Physics has taken over (muted) are sampled at frame; with no frame they are left."""
        bones = self.obj.pose.bones
        rows = np.flatnonzero(self.chain)
        paths = (("location", 3, 0.0), ("rotation_quaternion", 4, None), ("rotation_euler", 3, 0.0),
                 ("rotation_axis_angle", 4, None), ("scale", 3, 1.0))
        for path, size, rest_value in paths:
            kind = "rotation" if path.startswith("rotation") else path
            buffer = self._buffers[path]
            bones.foreach_get(path, buffer)
            values = buffer.reshape(-1, size)
            reset = rows[~self.keyed[kind][rows]]
            if path == "rotation_quaternion":
                values[reset] = (1.0, 0.0, 0.0, 0.0)
            elif path == "rotation_axis_angle":
                values[reset] = (0.0, 0.0, 1.0, 0.0)
            else:
                values[reset] = rest_value
        if frame is not None and self.keys is not None and self.keys.muted:
            self.keys.sample(frame, self._buffers)
        for path, _size, _rest in paths:
            bones.foreach_set(path, self._buffers[path])
        self.obj.update_tag(refresh={"DATA"})
