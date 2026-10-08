"""Scene-wide cloth arrays and frame sampling in world centimetres."""
import math

import numpy as np

F32 = np.float32


def stiffness(value, dt, iterations):
    """Chaos PBD exponential parameter fit, with exact endpoints."""
    v = np.clip(np.asarray(value, np.float32), 0, 1)
    fit = (np.exp(F32(math.log(1000)) * v) - F32(1)) / F32(999)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = F32(1) - np.exp(np.log(F32(1)-fit) * F32(F32(dt)*F32(120)/F32(iterations)))
    return np.where(v <= 1e-8, 0, np.where(v >= 1, 1, out)).astype(np.float32)


def _rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, float)
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _axis_angle(rotation):
    angle = math.acos(float(np.clip((np.trace(rotation)-1)*.5, -1, 1)))
    if angle < 1e-8:
        return np.zeros(3), 0.
    if abs(math.pi-angle) < 1e-5:
        values, vectors = np.linalg.eigh(rotation)
        axis = vectors[:, np.argmax(values)]
    else:
        axis = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0],
                         rotation[1, 0]-rotation[0, 1]]) / (2*math.sin(angle))
    return axis, angle


def _axis_rotation(axis, angle):
    if angle == 0:
        return np.eye(3)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


class System:
    """One backend call per substep for all cloths.

    targets: list of full original-vertex world cm arrays. motions: list of
    previous-to-current rigid 4x4 world transforms; an optional (matrix, old
    reference origin) tuple preserves rotations about a moving reference bone.
    colliders: shared list of Kawaii-compatible Shape objects. Optional
    velocity/angular_velocity attributes use cm/s and radians/s.
    """
    def __init__(self, meshes, gravity=(0., 0., -980.665)):
        self.meshes = list(meshes)
        self.gravity = np.asarray(gravity, np.float32)
        self.starts = np.asarray(np.r_[0, np.cumsum([len(m.positions) for m in meshes])], np.int32)
        self.n = int(self.starts[-1])
        self.ng = len(meshes)
        def cat(name, shape=None):
            arrays = [getattr(m, name) for m in meshes]
            return np.ascontiguousarray(np.concatenate(arrays) if arrays else np.empty(shape or (0,), np.float32))
        self.x = cat("positions", (0, 3))
        self.p = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.target = self.x.copy()
        self.anim_vel = np.zeros_like(self.x)
        self.inv_mass = cat("inv_mass")
        self.mass = cat("mass")
        self.max_distance = cat("max_distance")
        self.anim_stiff = np.zeros(self.n, np.float32)
        self.anim_damp = np.zeros(self.n, np.float32)
        self.acceleration = np.zeros_like(self.x)
        self.local_dv = np.zeros_like(self.x)
        self.iterations = np.ones(self.ng, np.int32)
        # integrated damping, edge k, bending k, area k, tether k, scale, thickness, friction
        self.params = np.zeros((self.ng, 8), np.float32)
        self.full_targets = [m.original_positions.copy() for m in meshes]
        self.shapes = np.empty((0, 25), np.float32)
        self.shape_starts = np.zeros(self.ng+1, np.int32)
        self.step_dt = F32(0)
        self._batches = {}
        for name, width in (("edges", 2), ("bending", 2), ("area", 3), ("tethers", 2)):
            rows, lengths, barys, starts, batches = [], [], [], [0], []
            singular = {"edges": "edge", "tethers": "tether"}.get(name, name)
            for g, m in enumerate(meshes):
                a = getattr(m, name)
                rows.extend(a + self.starts[g])
                lengths.extend(getattr(m, singular+"_length"))
                colors = getattr(m, singular+"_colors")
                batches.append([(starts[-1]+int(a), starts[-1]+int(b)) for a, b in zip(colors[:-1], colors[1:])])
                starts.append(starts[-1]+len(a))
                if name == "area":
                    barys.extend(m.area_bary)
            setattr(self, name, np.ascontiguousarray(np.asarray(rows, np.int32).reshape(-1, width)))
            setattr(self, singular+"_length", np.asarray(lengths, np.float32))
            setattr(self, singular+"_starts", np.asarray(starts, np.int32))
            self._batches[name] = batches
            if name == "area":
                self.area_bary = np.asarray(barys, np.float32)

    def _targets(self, targets):
        if targets is None:
            return [a.copy() for a in self.full_targets]
        if len(targets) != self.ng:
            raise ValueError("one target vertex array is required per cloth mesh")
        out = []
        for m, values in zip(self.meshes, targets):
            a = np.ascontiguousarray(values, np.float32)
            if a.shape != m.original_positions.shape or not np.all(np.isfinite(a)):
                raise ValueError("cloth targets must match original vertices and be finite")
            out.append(a)
        return out

    def reset(self, targets=None):
        self.full_targets = self._targets(targets)
        for g, m in enumerate(self.meshes):
            sl = slice(self.starts[g], self.starts[g+1])
            self.x[sl] = self.full_targets[g][m.representatives]
        self.p[:] = self.target[:] = self.x
        self.v.fill(0)
        self.anim_vel.fill(0)

    def set_colliders(self, colliders, dt=None):
        colliders = list(colliders or ())
        nested = bool(colliders) and not hasattr(colliders[0], "kind")
        groups = colliders if nested else [colliders]*self.ng
        if len(groups) != self.ng:
            raise ValueError("one collider list is required per cloth mesh")
        rows, starts, explicit_velocity, explicit_angular = [], [0], [], []
        for group in groups:
            for shape in group:
                if not getattr(shape, "enabled", True) or shape.kind not in (0, 2, 3, 4, 5):
                    continue
                row = np.zeros(25, np.float32)
                row[0], row[1:4] = shape.kind, shape.location
                row[4:13] = _rotation(shape.rotation).ravel()
                row[13:16] = max(0, shape.radius), max(0, shape.radius1), max(0, shape.length)*.5
                row[16:19] = np.maximum(shape.extent, 0)
                row[19:22] = getattr(shape, "velocity", (0, 0, 0))
                row[22:25] = getattr(shape, "angular_velocity", (0, 0, 0))
                if not np.all(np.isfinite(row)):
                    raise ValueError("cloth colliders must be finite")
                rows.append(row)
                explicit_velocity.append(hasattr(shape, "velocity"))
                explicit_angular.append(hasattr(shape, "angular_velocity"))
            starts.append(len(rows))
        new_shapes = np.ascontiguousarray(np.asarray(rows, np.float32).reshape(-1, 25))
        if (dt is not None and dt > 0 and new_shapes.shape == self.shapes.shape
                and np.array_equal(new_shapes[:, 0], self.shapes[:, 0])
                and np.array_equal(starts, self.shape_starts)):
            for i, (new, old) in enumerate(zip(new_shapes, self.shapes)):
                if not explicit_velocity[i]:
                    new[19:22] = (new[1:4]-old[1:4])/F32(dt)
                if not explicit_angular[i]:
                    rotation = new[4:13].reshape(3, 3)@old[4:13].reshape(3, 3).T
                    axis, angle = _axis_angle(rotation)
                    new[22:25] = axis*(angle/dt)
        self.shapes = new_shapes
        self.shape_starts = np.asarray(starts, np.int32)

    def _local_damping(self, g, cfg):
        sl = slice(self.starts[g], self.starts[g+1])
        dynamic = self.inv_mass[sl] > 0
        if cfg.local_damping <= 0 or not np.any(dynamic):
            self.local_dv[sl] = 0
            return
        x, v, mass = self.x[sl][dynamic], self.v[sl][dynamic], self.mass[sl][dynamic]
        cm = np.sum(x*mass[:, None], axis=0)/np.sum(mass)
        cv = np.sum(v*mass[:, None], axis=0)/np.sum(mass)
        r = x-cm
        angular = np.sum(np.cross(r, mass[:, None]*v), axis=0)
        inertia = np.eye(3, dtype=np.float32)*np.sum(mass*np.sum(r*r, axis=1)) - np.einsum("n,ni,nj->ij", mass, r, r)
        det = np.linalg.det(inertia)
        omega = np.linalg.solve(inertia, angular) if np.isfinite(det) and det >= 1e-8 else np.zeros(3, np.float32)
        accelerated_v = self.v[sl]+self.acceleration[sl]*self.step_dt
        self.local_dv[sl] = F32(cfg.local_damping)*(cv-accelerated_v+np.cross(self.x[sl]-cm, omega))

    def step_frame(self, dt, targets=None, colliders=None, motions=None, backend=None,
                   gravity=None, teleport=False, reset=False):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("cloth dt must be finite and nonnegative")
        if gravity is not None:
            value = np.asarray(gravity, np.float32)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("cloth gravity must be a finite 3-vector")
            self.gravity[:] = value
        new_targets = self._targets(targets)
        if reset:
            self.reset(new_targets)
            return
        if dt == 0:
            return
        if backend is None:
            from .native import backend as default_backend
            backend = default_backend()
        old_shapes = self.shapes.copy()
        old_shape_starts = self.shape_starts.copy()
        if colliders is not None:
            self.set_colliders(colliders, dt)
        frame_shapes = self.shapes.copy()
        shape_rotations = []
        if (old_shapes.shape == frame_shapes.shape and np.array_equal(old_shapes[:, 0], frame_shapes[:, 0])
                and np.array_equal(old_shape_starts, self.shape_starts)):
            for new, old in zip(frame_shapes, old_shapes):
                shape_rotations.append(_axis_angle(new[4:13].reshape(3, 3)@old[4:13].reshape(3, 3).T))
        old_target = self.target.copy()
        frame_target = self.target.copy()
        angular_fields = [None]*self.ng
        if motions is not None and len(motions) != self.ng:
            raise ValueError("one motion transform is required per cloth mesh")
        for g, m in enumerate(self.meshes):
            cfg = m.settings
            sl = slice(self.starts[g], self.starts[g+1])
            if motions is not None and motions[g] is not None:
                motion = motions[g]
                origin = np.zeros(3)
                if isinstance(motion, tuple) and len(motion) == 2:
                    motion, origin = motion
                    origin = np.asarray(origin, float)
                matrix = np.asarray(motion, float)
                if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                    raise ValueError("cloth motion must be a finite rigid 4x4 transform")
                axis, angle = _axis_angle(matrix[:3, :3])
                rotation = _axis_rotation(axis, angle*(1 if teleport else 1-cfg.angular_velocity_scale))
                full_delta = matrix[:3, :3]@origin+matrix[:3, 3]-origin
                translation = full_delta*(1 if teleport else 1-cfg.linear_velocity_scale)
                self.x[sl] = (self.x[sl]-origin)@rotation.T+origin+translation
                self.v[sl] = self.v[sl]@rotation.T
                old_target[sl] = (old_target[sl]-origin)@rotation.T+origin+translation
                if not teleport:
                    omega = axis*(angle/dt)
                    reference = matrix[:3, :3]@origin+matrix[:3, 3]
                    angular_fields[g] = omega, reference
            frame_target[sl] = new_targets[g][m.representatives]
            self.iterations[g] = max(1, min(int(cfg.max_iterations), int(math.floor(60*dt*cfg.iterations+.5))))
        substeps = max((max(1, int(m.settings.substeps)) for m in self.meshes), default=1)
        self.step_dt = F32(dt/substeps)
        for g, m in enumerate(self.meshes):
            cfg, iterations = m.settings, int(self.iterations[g])
            sl = slice(self.starts[g], self.starts[g+1])
            damping = float(np.clip(cfg.damping, 0, 1))
            if damping > 1-1e-4:
                integrated = 0
            elif damping > 1e-8:
                rate = F32(np.log(F32(1)-F32(damping))*F32(60))
                integrated = F32((np.exp(F32(rate*self.step_dt))-F32(1))/rate)
            else:
                integrated = self.step_dt
            self.params[g] = [integrated, stiffness(cfg.edge_stiffness, self.step_dt, iterations),
                              stiffness(cfg.bending_stiffness, self.step_dt, iterations),
                              stiffness(cfg.area_stiffness, self.step_dt, iterations),
                              stiffness(cfg.tether_stiffness, self.step_dt, 1), cfg.tether_scale,
                              cfg.thickness, cfg.friction]
            weight = np.floor(np.clip(m.anim_weight, 0, 1)*15).astype(np.float32)/F32(15)
            self.anim_stiff[sl] = stiffness(weight*F32(cfg.anim_drive_stiffness), self.step_dt, iterations)
            self.anim_damp[sl] = stiffness(weight*F32(cfg.anim_drive_damping), self.step_dt, iterations)
        self.anim_vel[:] = (frame_target-old_target)/F32(dt)
        for step in range(substeps):
            alpha = F32((step+1)/substeps)
            self.target[:] = old_target+(frame_target-old_target)*alpha
            if shape_rotations:
                self.shapes[:, 1:4] = old_shapes[:, 1:4]+(frame_shapes[:, 1:4]-old_shapes[:, 1:4])*alpha
                self.shapes[:, 13:19] = old_shapes[:, 13:19]+(frame_shapes[:, 13:19]-old_shapes[:, 13:19])*alpha
                for i, (axis, angle) in enumerate(shape_rotations):
                    self.shapes[i, 4:13] = (_axis_rotation(axis, angle*alpha)@old_shapes[i, 4:13].reshape(3, 3)).ravel()
            for g, m in enumerate(self.meshes):
                sl = slice(self.starts[g], self.starts[g+1])
                self.acceleration[sl] = self.gravity*F32(m.settings.gravity_scale)
                if angular_fields[g] is not None:
                    omega, reference = angular_fields[g]
                    self.acceleration[sl] -= np.cross(omega, np.cross(omega, self.x[sl]-reference))
                self._local_damping(g, m.settings)
            backend.simulate_once(self)
        self.shapes[:] = frame_shapes
        self.full_targets = [a.copy() for a in new_targets]

    def render_positions(self, mesh_index):
        m = self.meshes[mesh_index]
        x = self.x[self.starts[mesh_index]:self.starts[mesh_index+1]]
        out = self.full_targets[mesh_index].copy()
        valid = m.original_to_sim >= 0
        out[valid] = x[m.original_to_sim[valid]]
        if len(m.ride_vertices):
            tri = x[m.ride_triangles]
            normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            length = np.linalg.norm(normals, axis=1)
            normal = normals/np.maximum(length[:, None], 1e-8)
            out[m.ride_vertices] = np.sum(tri*m.ride_bary[:, :, None], axis=1)+normal*m.ride_offset[:, None]
        return out

    def snapshot(self):
        return {"x": self.x.copy(), "v": self.v.copy(), "target": self.target.copy(),
                "full_targets": [a.copy() for a in self.full_targets], "gravity": self.gravity.copy(),
                "shapes": self.shapes.copy(), "shape_starts": self.shape_starts.copy()}

    def restore(self, state):
        for name in ("x", "v", "target", "gravity"):
            getattr(self, name)[:] = state[name]
        self.p[:] = self.x
        self.full_targets = [a.copy() for a in state["full_targets"]]
        self.shapes = state["shapes"].copy()
        self.shape_starts = state["shape_starts"].copy()

    @staticmethod
    def interpolate(a, b, alpha):
        t = F32(np.clip(alpha, 0, 1))
        out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in b.items()}
        for key in ("x", "v", "target"):
            out[key] = a[key]+(b[key]-a[key])*t
        out["full_targets"] = [x+(y-x)*t for x, y in zip(a["full_targets"], b["full_targets"])]
        return out
