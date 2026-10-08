"""Build cloth topology once; no Blender or engine dependencies.

Numerical behavior is derived from UE 5.8 Chaos; see docs/cloth-reference.md.
Mesh welding and RGB authoring are Workshop conventions, not Chaos features.
"""
from dataclasses import dataclass
import heapq
import itertools

import numpy as np


@dataclass
class Settings:
    max_distance: float = 40.0
    density: float = .35
    min_mass: float = .0001
    gravity_scale: float = 1.0
    damping: float = .01
    local_damping: float = 0.0
    linear_velocity_scale: float = .75
    angular_velocity_scale: float = .75
    iterations: int = 1
    max_iterations: int = 10
    substeps: int = 1
    edge_stiffness: float = 1.0
    bending_stiffness: float = 1.0
    area_stiffness: float = 1.0
    tether_stiffness: float = 1.0
    tether_scale: float = 1.0
    anim_drive_stiffness: float = 1.0
    anim_drive_damping: float = 1.0
    thickness: float = 1.0
    friction: float = .8


@dataclass
class Mesh:
    settings: Settings
    original_positions: np.ndarray
    original_to_sim: np.ndarray
    representatives: np.ndarray
    positions: np.ndarray
    triangles: np.ndarray
    mass: np.ndarray
    inv_mass: np.ndarray
    max_distance: np.ndarray
    anim_weight: np.ndarray
    edges: np.ndarray
    edge_length: np.ndarray
    edge_colors: np.ndarray
    bending: np.ndarray
    bending_length: np.ndarray
    bending_colors: np.ndarray
    area: np.ndarray
    area_bary: np.ndarray
    area_length: np.ndarray
    area_colors: np.ndarray
    tethers: np.ndarray
    tether_length: np.ndarray
    tether_colors: np.ndarray
    ride_vertices: np.ndarray
    ride_triangles: np.ndarray
    ride_bary: np.ndarray
    ride_offset: np.ndarray


def _array(values, width, dtype=np.int32):
    return np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1, width))


def _color(rows, inv_mass):
    """Greedy dynamic-node coloring, deterministic input order and contiguous batches."""
    colors, occupied = [], []
    for row in rows:
        dynamic = {int(i) for i in row if inv_mass[i] > 0}
        color = next((i for i, used in enumerate(occupied) if used.isdisjoint(dynamic)), len(occupied))
        if color == len(occupied):
            occupied.append(set())
        occupied[color].update(dynamic)
        colors.append(color)
    order = np.argsort(colors, kind="stable")
    counts = np.bincount(colors, minlength=len(occupied)) if colors else np.zeros(0, int)
    return order, np.asarray(np.r_[0, np.cumsum(counts)], np.int32)


def _closest_bary(point, tri, clamp=True):
    """Closest-triangle selection or unbounded plane-projection binding."""
    a, b, c = tri
    ab, ac = b-a, c-a
    gram = np.array([[ab@ab, ab@ac], [ab@ac, ac@ac]])
    uv = np.linalg.solve(gram, [ab@(point-a), ac@(point-a)])
    bary = np.array([1-uv.sum(), *uv])
    if not clamp:
        return bary
    candidates = [bary] if np.all(bary >= 0) else []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        d = tri[j]-tri[i]
        t = np.clip((point-tri[i])@d / (d@d), 0, 1)
        q = np.zeros(3)
        q[i], q[j] = 1-t, t
        candidates.append(q)
    return min(candidates, key=lambda q: np.sum((q@tri-point)**2))


def _tethers(positions, neighbors, pinned):
    """Closest boundary seed per anchor island, at most four islands per point.

    Paths never progress through a different kinematic node. A multi-source
    Dijkstra per island is equivalent to taking the minimum per seed distance.
    """
    unseen = set(np.flatnonzero(pinned))
    islands = []
    while unseen:
        queue, island = [min(unseen)], []
        unseen.remove(queue[0])
        while queue:
            i = queue.pop()
            island.append(i)
            for j in sorted(neighbors[i]):
                if j in unseen:
                    unseen.remove(j)
                    queue.append(j)
        islands.append(island)
    candidates = [[] for _ in positions]
    for island in islands:
        distance = np.full(len(positions), np.inf)
        seed_for = np.full(len(positions), -1, np.int32)
        heap = []
        for seed in sorted(island):
            if any(not pinned[j] for j in neighbors[seed]):
                distance[seed], seed_for[seed] = 0, seed
                heapq.heappush(heap, (0., seed, seed))
        while heap:
            d, seed, i = heapq.heappop(heap)
            if d != distance[i] or seed != seed_for[i]:
                continue
            for j in sorted(neighbors[i]):
                if pinned[j]:
                    continue
                new_d = float(np.float32(d + np.linalg.norm(positions[j]-positions[i])))
                if new_d < distance[j] or (new_d == distance[j] and seed < seed_for[j]):
                    distance[j], seed_for[j] = new_d, seed
                    heapq.heappush(heap, (new_d, seed, j))
        for i in np.flatnonzero(~pinned & np.isfinite(distance)):
            candidates[i].append((distance[i], int(seed_for[i])))
    rows, lengths = [], []
    for i, choices in enumerate(candidates):
        for length, seed in sorted(choices)[:4]:
            rows.append((seed, i))
            lengths.append(length)
    return _array(rows, 2), np.asarray(lengths, np.float32)


def build(positions, triangles, channels=None, settings=None):
    """World centimetres, triangle vertex indices, and optional RGB weights.

    Red is max distance / settings.max_distance; green drives animation;
    blue >= .5 rides the nearest participating simulation triangle.
    Non-riding vertices weld by rounded .001 cm (0.01 mm) cells. For seams,
    maximum red and green match the game's imported cloth construction.
    Fixed-only connected components remain animated pass-through geometry.
    """
    cfg = settings or Settings()
    if isinstance(cfg, dict):
        cfg = Settings(**cfg)
    p = _array(positions, 3, np.float32)
    t = _array(triangles, 3)
    c = np.tile([1., 0., 0.], (len(p), 1)) if channels is None else np.asarray(channels, float)
    if c.shape != p.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(c)):
        raise ValueError("cloth positions and RGB must be finite Nx3 arrays")
    if len(t) and (t.min() < 0 or t.max() >= len(p)):
        raise ValueError("cloth triangle index outside vertex array")
    if any(not np.isfinite(v) for v in vars(cfg).values()) or cfg.density < 0 or cfg.min_mass <= 0 or cfg.max_distance < 0:
        raise ValueError("cloth settings must be finite with nonnegative density/distance and positive mass")
    c = np.clip(c, 0, 1)
    ride = c[:, 2] >= .5
    welded, wc, representatives = [], [], []
    mapping = np.full(len(p), -1, np.int32)
    buckets = {}
    for i in np.flatnonzero(~ride):
        # FMath::RoundToInt is floor(x+.5), including negative half ties.
        cell = tuple(np.floor(p[i]*np.float32(1000)+np.float32(.5)).astype(np.int64))
        found = buckets.get(cell)
        if found is None:
            found = len(welded)
            welded.append(p[i])
            wc.append(c[i].copy())
            representatives.append(i)
            buckets[cell] = found
        else:
            wc[found][0] = max(wc[found][0], c[i, 0])
            wc[found][1] = max(wc[found][1], c[i, 1])
        mapping[i] = found
    wp = _array(welded, 3, np.float32)
    wc = _array(wc, 3, np.float32)
    faces, seen = [], set()
    for face in t:
        row = mapping[face]
        key = tuple(sorted(row))
        if row.min() < 0 or len(set(row)) != 3 or key in seen:
            continue
        if np.linalg.norm(np.cross(wp[row[1]]-wp[row[0]], wp[row[2]]-wp[row[0]])) <= 1e-8:
            continue
        faces.append(row)
        seen.add(key)
    neighbors = [set() for _ in wp]
    for row in faces:
        for a, b in itertools.combinations(row, 2):
            neighbors[a].add(int(b))
            neighbors[b].add(int(a))
    unseen, active = set(range(len(wp))), set()
    while unseen:
        queue, component = [min(unseen)], []
        unseen.remove(queue[0])
        while queue:
            i = queue.pop()
            component.append(i)
            for j in neighbors[i]:
                if j in unseen:
                    unseen.remove(j)
                    queue.append(j)
        if any(wc[i, 0] * cfg.max_distance >= .1 and neighbors[i] for i in component):
            active.update(component)
    keep = sorted(active)
    remap = np.full(len(wp), -1, np.int32)
    remap[keep] = np.arange(len(keep))
    mapping[mapping >= 0] = remap[mapping[mapping >= 0]]
    wp, wc = wp[keep], wc[keep]
    reps = np.asarray(representatives, np.int32)[keep]
    faces = _array([remap[row] for row in faces if row[0] in active], 3)
    neighbors = [set() for _ in wp]
    edge_opposite = {}
    edge_ordered = {}
    mass = np.zeros(len(wp), np.float32)
    for row in faces:
        area = np.linalg.norm(np.cross(wp[row[1]]-wp[row[0]], wp[row[2]]-wp[row[0]])) * .5
        mass[row] += np.float32(area * cfg.density / 30000.)
        for a, b in itertools.combinations(row, 2):
            edge_ordered.setdefault(tuple(sorted((int(a), int(b)))), None)
        for k in range(3):
            a, b, opposite = int(row[k]), int(row[(k+1)%3]), int(row[(k+2)%3])
            edge_opposite.setdefault(tuple(sorted((a, b))), []).append(opposite)
            neighbors[a].add(b)
            neighbors[b].add(a)
    mass = np.maximum(mass, np.float32(cfg.min_mass))
    max_distance = np.ascontiguousarray(wc[:, 0] * np.float32(cfg.max_distance))
    inv_mass = np.where(max_distance < .1, 0., 1./mass).astype(np.float32)
    edges = _array(list(edge_ordered), 2)
    edges = edges[np.any(inv_mass[edges] > 0, axis=1)]
    bending = _array(list(dict.fromkeys(tuple(sorted(pair)) for opp in edge_opposite.values()
                                       for pair in itertools.combinations(opp, 2) if pair[0] != pair[1])), 2)
    bending = bending[np.any(inv_mass[bending] > 0, axis=1)]
    area_rows, barys, area_lengths = [], [], []
    for row in dict.fromkeys(tuple(sorted(row)) for row in faces):
        if not np.any(inv_mass[list(row)]):
            continue
        choices = []
        for k in range(3):
            a, b, d = row[k], row[(k+1)%3], row[(k+2)%3]
            e = wp[d]-wp[b]
            bary = float(np.clip(e@(wp[d]-wp[a])/(e@e), 0, 1))
            choices.append(((a, b, d), bary))
        # Strict comparisons reproduce the engine's tie preference (first).
        dist = [abs(b-.5) for _, b in choices]
        pick = 2 if dist[2] < dist[1] and dist[2] < dist[0] else 1 if dist[1] < dist[0] and dist[1] < dist[2] else 0
        row, bary = choices[pick]
        area_rows.append(row)
        barys.append(bary)
        area_lengths.append(np.linalg.norm(wp[row[0]]-((wp[row[1]]-wp[row[2]])*bary+wp[row[2]])))
    area_rows = _array(area_rows, 3)
    tethers, tether_length = _tethers(wp, neighbors, inv_mass == 0)
    edge_order, edge_colors = _color(edges, inv_mass)
    bend_order, bend_colors = _color(bending, inv_mass)
    area_order, area_colors = _color(area_rows, inv_mass)
    tether_order, tether_colors = _color(tethers, inv_mass)
    edge_length = np.linalg.norm(wp[edges[:, 0]]-wp[edges[:, 1]], axis=1).astype(np.float32)
    bending_length = np.linalg.norm(wp[bending[:, 0]]-wp[bending[:, 1]], axis=1).astype(np.float32)
    riders, ride_tri, ride_bary, ride_offset = [], [], [], []
    for i in np.flatnonzero(ride):
        best = None
        for row in faces:
            tri = wp[row].astype(float)
            bary = _closest_bary(p[i], tri)
            delta = p[i]-bary@tri
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            normal /= np.linalg.norm(normal)
            value = (float(delta@delta), row, bary, float(delta@normal))
            if best is None or value[0] < best[0]:
                best = value
        if best is not None:
            riders.append(i)
            ride_tri.append(best[1])
            # Selection uses distance to the finite triangle, but binding uses
            # its entire plane so boundary riders retain tangential position.
            ride_bary.append(_closest_bary(p[i], wp[best[1]].astype(float), clamp=False))
            ride_offset.append(best[3])
    return Mesh(cfg, p.copy(), mapping, reps, wp, faces, mass, inv_mass, max_distance,
                np.ascontiguousarray(wc[:, 1]), edges[edge_order], edge_length[edge_order], edge_colors,
                bending[bend_order], bending_length[bend_order], bend_colors,
                area_rows[area_order], np.asarray(barys, np.float32)[area_order],
                np.asarray(area_lengths, np.float32)[area_order], area_colors,
                tethers[tether_order], tether_length[tether_order], tether_colors,
                np.asarray(riders, np.int32), _array(ride_tri, 3), _array(ride_bary, 3, np.float32),
                np.asarray(ride_offset, np.float32))
