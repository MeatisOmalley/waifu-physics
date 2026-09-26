"""Link Chains: join neighbouring chains of a group, bone by bone at each depth.

Chains are put side by side (side_by_side), so a skirt's panels link to the
panels beside them whatever order they were selected in, and never closed into a
ring: the two end chains are left for the user to link, which closes it. Each
link joins the bones at the same depth below the roots -- the roots themselves
follow the animation --
and the group's Link Tips setting adds links between their tip and
subdivision points when the chains are built (Kawaii's automatic dummy links).
"""
import math

import numpy as np


def chain_subtree(obj, root, excluded=()):
    """Every bone of the chain under root, root first: what Waifu Physics simulates for it."""
    excluded = set(excluded)
    found, stack = [], [obj.pose.bones.get(root)]
    while stack:
        bone = stack.pop()
        if bone is None or bone.name in excluded:
            continue
        found.append(bone.name)
        stack.extend(reversed(bone.children))
    return found


def constrained_bones(obj, group):
    """Bones in a group's chains with an active constraint: Blender applies it after the
    simulation's output, so the simulation cannot move them."""
    excluded = {bone.name for bone in group.excluded}
    found, stack = [], [obj.pose.bones.get(root.name) for root in group.roots]
    while stack:
        bone = stack.pop()
        if bone is None or bone.name in excluded:
            continue
        if any(c.enabled and c.influence > 0 for c in bone.constraints):
            found.append(bone.name)
        stack.extend(bone.children)
    return found


def chain_bones(obj, root, excluded=()):
    """The bones of a chain from its root down, following the first child at each step."""
    bones, bone = [], obj.pose.bones.get(root)
    while bone is not None and bone.name not in excluded:
        bones.append(bone.name)
        children = [c for c in bone.children if c.name not in excluded]
        bone = children[0] if children else None
    return bones


def ordered(obj, roots):
    """Roots by angle around their centre, looking along the direction the chains hang."""
    heads = {r: np.array(obj.pose.bones[r].bone.head_local) for r in roots}
    downs = []
    for r in roots:
        chain = chain_bones(obj, r)
        tail = np.array(obj.pose.bones[chain[-1]].bone.tail_local)
        downs.append(tail - heads[r])
    axis = np.sum(downs, axis=0)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([0.0, 0.0, -1.0])
    axis /= np.linalg.norm(axis)
    centre = np.mean(list(heads.values()), axis=0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = {r: math.atan2((heads[r] - centre) @ v, (heads[r] - centre) @ u) for r in roots}
    return sorted(roots, key=lambda r: angle[r])


def in_a_row(obj, roots):
    """Roots in order along the line they spread out on most (a cape's chains, left to right). An angle around
    their centre (ordered) cannot order chains in a straight row: they all sit at one of two angles."""
    heads = np.array([obj.pose.bones[r].bone.head_local for r in roots], dtype=float)
    spread = heads - heads.mean(axis=0)
    direction = np.linalg.svd(spread, full_matrices=False)[2][0] if len(roots) > 1 else np.zeros(3)
    return [roots[i] for i in np.argsort(spread @ direction, kind="stable")]


def side_by_side(obj, names):
    """Bones (or chain roots) in the order they sit side by side, end to end and never closed. Two orders are
    tried: round their centre, opened at the widest gap (a skirt, or a cape wrapped round the back), and along
    the line they spread out on most (a flat cape). The shorter path between neighbours wins: the wrong order
    zig-zags from one side to the other, so it is always the longer."""
    if len(names) < 3:
        return list(names)
    heads = {name: np.array(obj.pose.bones[name].bone.head_local) for name in names}
    ring = ordered(obj, names)
    gaps = [np.linalg.norm(heads[a] - heads[b]) for a, b in zip(ring, ring[1:] + ring[:1])]
    widest = int(np.argmax(gaps))
    arc = ring[widest + 1:] + ring[:widest + 1]

    def length(order):
        return sum(np.linalg.norm(heads[a] - heads[b]) for a, b in zip(order, order[1:]))
    return min((arc, in_a_row(obj, names)), key=length)


def neighbours(obj, names):
    """[(name, name)] of bones side by side (side_by_side): the last is not linked back to the first."""
    order = side_by_side(obj, names)
    return list(zip(order, order[1:]))


def _root_and_depth(obj, group, name):
    """(root, bones below it) of the group's chain a bone is simulated in, or None: a bone under an excluded
    one (or under another group's root, which excludes it) is not."""
    roots = {root.name for root in group.roots}
    excluded = {bone.name for bone in group.excluded}
    bone, depth = obj.pose.bones.get(name), 0
    while bone is not None:
        if bone.name in excluded:
            return None
        if bone.name in roots:
            return bone.name, depth
        bone, depth = bone.parent, depth + 1
    return None


def _in_line(obj, a, b):
    """Does one bone hang from the other, however far down?"""
    def above(upper, lower):
        bone = obj.pose.bones[lower].parent
        while bone is not None:
            if bone.name == upper:
                return True
            bone = bone.parent
        return False
    return above(a, b) or above(b, a)


def bone_pairs(obj, group, names):
    """Link Selected Bones: (the new links, None), or ([], why there are none). Only bones the group simulates
    below its roots count: a root follows the animation, so a link to it holds nothing (Kawaii moves both ends,
    and the pose puts the root back). Two bones make one link unless one hangs from the other: their chain
    already keeps that distance, and a link along it only stiffens the chain. More are linked side by side
    (side_by_side) at each depth below the roots, as Link Whole Chains does, so bones of one chain never are.
    Kawaii checks none of this (any two of a node's bones make a constraint); these are the links that do
    what links are for, holding neighbouring chains apart."""
    depth = {}
    for name in names:
        found = _root_and_depth(obj, group, name)
        if found is not None and found[1] > 0:
            depth[name] = found[1]
    if len(depth) < 2:
        return [], "Select two bones in different chains of the active group, below their roots"
    if len(depth) == 2:
        a, b = depth
        if _in_line(obj, a, b):
            return [], "One bone hangs from the other: its chain already keeps them apart"
        found = [(a, b)]
    else:
        rows = {}
        for name, level in depth.items():
            rows.setdefault(level, []).append(name)
        found = [pair for row in rows.values() if len(row) > 1 for pair in neighbours(obj, row)]
        if not found:
            return [], "Select two bones, or bones at the same depth in different chains"
    existing = {frozenset((link.bone_a, link.bone_b)) for link in group.links}
    found = [pair for pair in found if frozenset(pair) not in existing]
    if not found:
        return [], "The selected bones are already linked"
    return found, None


def pairs(obj, roots, excluded=()):
    """[(bone, bone)] linking each chain to its neighbour, at every depth below the roots (the roots do not
    move): a ladder's rungs between neighbouring chains, in side_by_side order, the two end chains left
    unlinked. Returns (the pairs, the roots in order)."""
    order = side_by_side(obj, roots)
    chains = [chain_bones(obj, r, excluded) for r in order]
    neighbours = list(zip(chains, chains[1:]))
    found = []
    for a, b in neighbours:
        for depth in range(1, min(len(a), len(b))):
            found.append((a[depth], b[depth]))
    return found, order
