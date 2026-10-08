"""Float32 reference for the default FEvolution PBD path; see cloth-reference.md."""
import numpy as np

NAME = "numpy"
F32 = np.float32


def _norm(v):
    return np.sqrt(np.sum(v*v, axis=-1))


def _springs(s, g, name, stiffness):
    if stiffness == 0:
        return
    singular = {"edges": "edge", "tethers": "tether"}.get(name, name)
    rest = getattr(s, singular+"_length")
    constraints = getattr(s, name)
    for start, end in s._batches[name][g]:
        a, b = constraints[start:end].T
        diff = s.p[a]-s.p[b]
        length = _norm(diff)
        w = s.inv_mass[b]+s.inv_mass[a]
        small = np.sum(diff*diff, axis=1) < F32(1e-4)
        direction = diff/np.maximum(length[:, None], F32(1e-8))
        direction[small] = [1, 0, 0]
        length[small] = 0
        delta = (length-rest[start:end])[:, None]*direction
        delta = stiffness*delta/np.maximum(w[:, None], F32(1e-8))
        delta[w == 0] = 0
        s.p[a] -= s.inv_mass[a, None]*delta
        s.p[b] += s.inv_mass[b, None]*delta


def _area(s, g, stiffness):
    if stiffness == 0:
        return
    for start, end in s._batches["area"][g]:
        a, b, c = s.area[start:end].T
        bary = s.area_bary[start:end]
        diff = s.p[a]-((s.p[b]-s.p[c])*bary[:, None]+s.p[c])
        length = _norm(diff)
        w = (s.inv_mass[c]*(F32(1)-bary)+s.inv_mass[b]*bary)+s.inv_mass[a]
        valid = (length > 1e-8) & (w > 0)
        # The axial rule's extra multiplier is intentional (not textbook PBD).
        multiplier = F32(2)/(np.maximum(bary, F32(1)-bary)+F32(1))
        direction = diff/np.maximum(length[:, None], F32(1e-8))
        delta = (length-s.area_length[start:end])[:, None]*direction
        delta = stiffness*delta/np.maximum(w[:, None], F32(1e-8))
        delta[~valid] = 0
        s.p[a] -= (multiplier*s.inv_mass[a])[:, None]*delta
        s.p[b] += (multiplier*s.inv_mass[b]*bary)[:, None]*delta
        s.p[c] += (multiplier*s.inv_mass[c]*(F32(1)-bary))[:, None]*delta


def _collide(s, g):
    start, end = s.starts[g:g+2]
    ids = np.arange(start, end)[s.inv_mass[start:end] > 0]
    thick, friction = s.params[g, 6:8]
    for shape in s.shapes[s.shape_starts[g]:s.shape_starts[g+1]]:
        kind = int(shape[0])
        rotation = shape[4:13].reshape(3, 3)
        local = (s.p[ids]-shape[1:4])@rotation
        normal = np.zeros_like(local)
        if kind in (0, 2, 3):
            diff = local.copy()
            radius = np.full(len(ids), shape[13], np.float32)
            if kind in (2, 3):
                z = np.clip(local[:, 2], -shape[15], shape[15])
                diff[:, 2] -= z
                if kind == 3:
                    if shape[15] > 1e-8:
                        radius = shape[14]+(shape[13]-shape[14])*(z+shape[15])/(F32(2)*shape[15])
                    else:
                        radius.fill(max(shape[13], shape[14]))
            length = _norm(diff)
            nonzero = length > 1e-8
            normal[nonzero] = diff[nonzero]/length[nonzero, None]
            normal[~nonzero, 0] = 1  # deterministic undefined-normal convention
            phi = length-radius
        elif kind == 4:
            q = np.abs(local)-shape[16:19]
            outside = np.maximum(q, 0)
            length = _norm(outside)
            mask = length > 1e-8
            normal[mask] = outside[mask]*np.where(local[mask] < 0, -1, 1)/length[mask, None]
            axis = np.argmax(q, axis=1)
            rows = np.arange(len(ids))[~mask]
            normal[rows, axis[~mask]] = np.where(local[rows, axis[~mask]] < 0, -1, 1)
            phi = length+np.minimum(np.max(q, axis=1), 0)
        else:
            normal[:, 2] = 1
            phi = local[:, 2]
        penetration = thick-phi
        hit = penetration > 0
        rows = ids[hit]
        n = normal[hit]@rotation.T
        s.p[rows] += penetration[hit, None]*n
        if friction > 1e-4:
            point_velocity = shape[19:22]+np.cross(shape[22:25], s.p[rows]-shape[1:4])
            displacement = (s.p[rows]-s.x[rows])-point_velocity*s.step_dt
            tangent = displacement-n*np.sum(displacement*n, axis=1)[:, None]
            length = _norm(tangent)
            ratio = np.minimum(penetration[hit]*friction, length)/np.maximum(length, F32(1e-8))
            s.p[rows] -= ratio[:, None]*tangent


def simulate_once(s):
    dynamic = s.inv_mass > 0
    s.v[dynamic] += s.acceleration[dynamic]*s.step_dt
    s.v[dynamic] += s.local_dv[dynamic]
    s.p[:] = s.target
    for g in range(s.ng):
        start, end = s.starts[g:g+2]
        ids = np.arange(start, end)[dynamic[start:end]]
        s.p[ids] = s.x[ids]+s.v[ids]*s.params[g, 0]
        # Long range attachment applies once after the initial guess.
        for first, last in s._batches["tethers"][g]:
            a, b = s.tethers[first:last].T
            diff = s.p[a]-s.p[b]
            length = _norm(diff)
            offset = np.maximum(length-s.tether_length[first:last]*s.params[g, 5], F32(0))
            delta = diff*(s.params[g, 4]*offset/np.maximum(length, F32(1e-8)))[:, None]
            s.p[b] += delta
        for _ in range(int(s.iterations[g])):
            _springs(s, g, "edges", s.params[g, 1])
            _springs(s, g, "bending", s.params[g, 2])
            _area(s, g, s.params[g, 3])
            diff = s.p[ids]-s.target[ids]
            length = _norm(diff)
            hit = length > s.max_distance[ids]
            s.p[ids[hit]] = s.target[ids[hit]]+diff[hit]*(s.max_distance[ids[hit]]/length[hit])[:, None]
            s.p[ids] -= s.anim_stiff[ids, None]*(s.p[ids]-s.target[ids])
            s.p[ids] -= s.anim_damp[ids, None]*((s.p[ids]-s.x[ids])-s.anim_vel[ids]*s.step_dt)
            _collide(s, g)
    s.v[:] = (s.p-s.x)/s.step_dt
    s.x[:] = s.p
