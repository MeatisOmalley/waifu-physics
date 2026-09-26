"""Building a System from a skeleton, and tracking the component's movement.

build() ports Kawaii's node initialisation: InitModifyBones and AddModifyBone
(tip dummies, inter-bone subdivision, bone lengths and length rates,
ModifyBones.cpp:79-465) and InitBoneConstraints with its automatic dummy links
and bridge dummies (Collision.cpp:1474-1650). ComponentMotion ports
UpdateSkelCompMove and the carry-over of the component transform between frames
(ModifyBones.cpp:578, AnimNode_KawaiiPhysics.cpp:915).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import uemath as ue
from .system import System, Group, KIND_BONE, KIND_TIP, KIND_INTER, KIND_BRIDGE

F32, F64 = np.float32, np.float64


@dataclass
class Skeleton:
    """The bones a group may use, in the armature's order (every parent before its children).

    ref_length: each bone's rest offset from its parent (the reference pose's local
    translation), whose size is Kawaii's BoneLength. pose / rotation: the current
    pose in simulation space, which is where points start. tip_length: how far past each bone a
    chain ending there puts its tip point, where the skeleton knows (a Blender bone's own length);
    None uses each group's dummy_bone_length, as Kawaii does, whose bones are joints with no length."""
    names: list
    parents: list
    ref_length: list
    pose: np.ndarray                 # (bones, 3), centimetres
    rotation: np.ndarray             # (bones, 4), x y z w
    tip_length: list = None          # (bones,), centimetres

    def children(self, index):
        """CollectChildBones: every bone whose parent this is, in skeleton order."""
        return [i for i in range(index + 1, len(self.names)) if self.parents[i] == index]


@dataclass
class GroupSpec:
    """A group to build: its settings, root bones and links, by bone name."""
    group: Group
    roots: list                                         # [(root name, excluded names or None to use `exclude`)]
    exclude: list = field(default_factory=list)
    links: list = field(default_factory=list)           # [(bone1, bone2, compliance type or -1, exclude_from_subdivision)]


class _Node:
    """One Kawaii ModifyBone while building."""
    __slots__ = ("kind", "bone", "parent", "children", "loc", "rot", "real_parent", "real_child", "alpha",
                 "bone_length", "length_from_root", "length_rate", "group", "tip_length")

    def __init__(self, kind, bone, loc, rot, group):
        self.kind, self.bone, self.loc, self.rot, self.group = kind, bone, np.asarray(loc, F64), \
            np.asarray(rot, F64), group
        self.parent, self.children = -1, []
        self.real_parent = self.real_child = -1
        self.alpha = F32(0.0)
        self.bone_length = self.length_from_root = self.length_rate = F32(0.0)
        self.tip_length = F32(0.0)                    # a tip's distance past its real bone


def _size(v):
    """TVector::Size of one vector, as a float where Kawaii stores it in one."""
    return F32(ue.size(np.asarray(v, F64)[None])[0])


def _min_radius_curve_scale(grp):
    """Densify-by-radius uses the radius curve's smallest value over [0, 1] (ModifyBones.cpp:318)."""
    fn = grp.curves.get("radius")
    if fn is None:
        return 1.0
    return min(fn(i / 100.0) for i in range(101))


def _inter_dummies(nodes, grp, index, parent_node, child_loc, child_rot, distance, group):
    """InsertInterBoneDummyBonesCore (ModifyBones.cpp:297): dummies from parent to child."""
    count = min(max(grp.bone_subdivision_count, 0), 10) if distance > ue.KINDA_SMALL else 0
    if grp.bone_subdivision_count <= 0:
        return parent_node, []
    if grp.bone_subdivision_densify_by_radius and distance > ue.KINDA_SMALL:
        average = F32(grp.settings["radius"]) * F32(max(_min_radius_curve_scale(grp), 0.0))
        if not (distance <= ue.KINDA_SMALL or average <= ue.KINDA_SMALL):
            coverage = max(math.ceil(distance / (F32(2.0) * average)) - 1, 0)
            count = max(count, coverage)
    count = min(count, 50)
    parent = nodes[parent_node]
    effective, inserted = parent_node, []
    for j in range(count):
        alpha = F32(j + 1) / F32(count + 1)
        node = _Node(KIND_INTER, -1, ue.lerp(parent.loc[None], child_loc[None], alpha)[0],
                     ue.slerp(parent.rot[None], np.asarray(child_rot, F64)[None], alpha)[0], group)
        node.alpha = alpha
        node.real_parent = parent_node
        node.bone_length = F32(distance / F32(count + 1))
        nodes.append(node)
        new = len(nodes) - 1
        nodes[effective].children.append(new)
        node.parent = effective
        effective = new
        inserted.append(new)
    return effective, inserted


def _add_bone(nodes, skeleton, spec, bone, excluded, group):
    """AddModifyBone (ModifyBones.cpp:142)."""
    if skeleton.names[bone] in excluded:
        return -1
    grp = spec.group
    node = _Node(KIND_BONE, bone, skeleton.pose[bone], skeleton.rotation[bone], group)
    nodes.append(node)
    index = len(nodes) - 1
    added = False
    for child in skeleton.children(bone):
        distance = _size(skeleton.pose[child] - nodes[index].loc)
        effective, inserted = _inter_dummies(nodes, grp, index, index, skeleton.pose[child],
                                             skeleton.rotation[child], distance, group)
        child_index = _add_bone(nodes, skeleton, spec, child, excluded, group)
        if child_index >= 0:
            nodes[effective].children.append(child_index)
            nodes[child_index].parent = effective
            added = True
            for d in inserted:
                nodes[d].real_child = child_index
        elif inserted:
            for d in reversed(inserted):
                if d == len(nodes) - 1:
                    nodes.pop()
            nodes[index].children = [c for c in nodes[index].children if c not in inserted]
    length = F32(grp.dummy_bone_length if skeleton.tip_length is None else skeleton.tip_length[bone])
    if not added and length > 0:
        forward = ue.axis(node.rot[None], System.forward_axis)[0]
        tip = node.loc + forward * F64(length)
        effective, inserted = _inter_dummies(nodes, grp, index, index, tip, node.rot, length, group)
        dummy = _Node(KIND_TIP, -1, tip, node.rot, group)
        dummy.tip_length = length
        if inserted:
            dummy.real_parent = index
            dummy.bone_length = F32(length / F32(len(inserted) + 1))
        nodes.append(dummy)
        dummy_index = len(nodes) - 1
        nodes[effective].children.append(dummy_index)
        dummy.parent = effective
        for d in inserted:
            nodes[d].real_child = dummy_index
    return index


def _lengths(nodes, skeleton, grp, root):
    """CalcBoneLength (ModifyBones.cpp:427) and the length rate from the root."""
    total = [F32(0.0)]

    def walk(i):
        node = nodes[i]
        if node.parent < 0:
            node.length_from_root = F32(0.0)
            node.bone_length = F32(0.0)
        else:
            if node.kind == KIND_BONE:
                node.bone_length = F32(skeleton.ref_length[node.bone])
            elif node.kind == KIND_TIP and nodes[node.parent].kind != KIND_INTER:
                node.bone_length = F32(node.tip_length)
            node.length_from_root = F32(nodes[node.parent].length_from_root + node.bone_length)
            total[0] = max(total[0], node.length_from_root)
        for child in node.children:
            walk(child)

    walk(root)
    return total[0]


def build(skeleton, specs, target_framerate=60, max_substeps=4, fixed_substepping=True):
    """A System for these groups, points starting at the skeleton's pose."""
    groups = [spec.group for spec in specs]
    all_nodes, links = [], []
    for g, spec in enumerate(specs):
        nodes = []
        for root_name, root_exclude in spec.roots:
            if root_name not in skeleton.names:
                continue
            excluded = set(root_exclude if root_exclude is not None else spec.exclude)
            first = len(nodes)
            root = _add_bone(nodes, skeleton, spec, skeleton.names.index(root_name), excluded, g)
            if root < 0:
                continue
            total = _lengths(nodes, skeleton, spec.group, root)
            for node in nodes[first:]:
                if node.length_from_root > 0:
                    node.length_rate = F32(node.length_from_root / total)
        group_links, bridges = _constraints(nodes, skeleton, spec)
        offset = len(all_nodes)
        for node in nodes + bridges:
            for attr in ("parent", "real_parent", "real_child"):
                value = getattr(node, attr)
                if value >= 0:
                    setattr(node, attr, value + offset)
        all_nodes += nodes + bridges
        links += [(a + offset, b + offset, length, ctype) for a, b, length, ctype in group_links]
    points = dict(
        parent=[n.parent for n in all_nodes], group=[n.group for n in all_nodes],
        kind=[n.kind for n in all_nodes], real_parent=[n.real_parent for n in all_nodes],
        real_child=[n.real_child for n in all_nodes], alpha=[n.alpha for n in all_nodes],
        location=[n.loc for n in all_nodes], pose=[n.loc for n in all_nodes],
        pose_rotation=[n.rot for n in all_nodes], length_rate=[n.length_rate for n in all_nodes],
        bone=[n.bone for n in all_nodes], bone_length=[n.bone_length for n in all_nodes],
        length_from_root=[n.length_from_root for n in all_nodes], tip_length=[n.tip_length for n in all_nodes])
    link_arrays = dict(a=[l[0] for l in links], b=[l[1] for l in links], length=[l[2] for l in links],
                       compliance_type=[l[3] for l in links])
    system = System(groups, points, link_arrays, target_framerate, max_substeps, fixed_substepping)
    system.resolve_settings()
    return system


def _constraints(nodes, skeleton, spec):
    """InitBoneConstraints and InsertBridgeDummiesForConstraints: links and bridge dummies."""
    grp = spec.group
    by_bone = {}
    for i, node in enumerate(nodes):
        if node.kind == KIND_BONE and node.bone not in by_bone:
            by_bone[node.bone] = i

    def distance(a, b):
        return _size(nodes[a].loc - nodes[b].loc)

    def tip_child(i):
        return next((c for c in nodes[i].children if nodes[c].kind == KIND_TIP), -1)

    def inter_chain(i):
        chain = []
        for c in nodes[i].children:
            if nodes[c].kind == KIND_INTER:
                k = c
                while k >= 0 and nodes[k].kind == KIND_INTER:
                    chain.append(k)
                    k = next((cc for cc in nodes[k].children if nodes[cc].kind == KIND_INTER), -1)
                if chain:
                    tip = next((cc for cc in nodes[chain[-1]].children if nodes[cc].kind == KIND_TIP), -1)
                    if tip >= 0:
                        chain.append(tip)
                break
        return chain

    merged, dummy_links = [], []
    for first, second, ctype, no_subdivision in spec.links:
        if first not in skeleton.names or second not in skeleton.names:
            continue
        a = by_bone.get(skeleton.names.index(first), -1)
        b = by_bone.get(skeleton.names.index(second), -1)
        if a < 0 or b < 0:
            continue
        merged.append((a, b, distance(a, b), ctype, no_subdivision))
        if grp.auto_add_child_dummy_constraint:
            ta, tb = tip_child(a), tip_child(b)
            if ta >= 0 and tb >= 0:
                dummy_links.append((ta, tb, distance(ta, tb), -1, no_subdivision))
            chain_a, chain_b = inter_chain(a), inter_chain(b)
            for da, db in zip(chain_a, chain_b):
                dummy_links.append((da, db, distance(da, db), -1, no_subdivision))
    merged += dummy_links

    bridges = []
    count = grp.constraint_subdivision_count
    if count > 0:
        radius_curve = grp.curves.get("radius") or (lambda _rate: 1.0)
        base_radius = F32(grp.settings["radius"])
        for a, b, _length, _ctype, no_subdivision in merged:
            if no_subdivision:
                continue
            p1, p2 = nodes[a].loc, nodes[b].loc
            dist = _size(p2 - p1)
            lr1, lr2 = nodes[a].length_rate, nodes[b].length_rate
            r1 = base_radius * F32(max(radius_curve(float(lr1)), 0.0))
            r2 = base_radius * F32(max(radius_curve(float(lr2)), 0.0))
            if dist <= max(r1 + r2, ue.KINDA_SMALL):
                continue
            for k in range(count):
                alpha = F32(k + 1) / F32(count + 1)
                bridge = _Node(KIND_BRIDGE, -1, ue.lerp(p1[None], p2[None], alpha)[0],
                               ue.slerp(nodes[a].rot[None], nodes[b].rot[None], alpha)[0], nodes[a].group)
                bridge.real_parent, bridge.real_child, bridge.alpha = a, b, alpha
                bridge.bone_length = F32(dist / F32(count + 1))
                bridge.length_rate = F32(F32(0.5) * (lr1 + lr2))
                bridges.append(bridge)
    # Bridge dummies follow the node's bones, as ModifyBones.Add appends them.
    return [(a, b, length, ctype) for a, b, length, ctype, _ns in merged], bridges


class ComponentMotion:
    """The component's movement between frames, as Kawaii feeds it to the step.

    UpdateSkelCompMove (ModifyBones.cpp:578): the previous component transform
    seen from the current one, and a teleport when it moved or turned too far.
    After the frame, the previous transform advances by the fraction of time the
    substeps consumed (AnimNode_KawaiiPhysics.cpp:915)."""

    def __init__(self, teleport_distance=300.0, teleport_rotation=10.0, move_scale=(1.0, 1.0, 1.0)):
        self.teleport_distance = F32(teleport_distance)
        self.teleport_rotation = F32(teleport_rotation)
        self.move_scale = np.asarray(move_scale, F64)
        self.previous = None                 # (location, rotation x y z w, scale)

    def update(self, location, rotation, scale):
        """(move vector, move rotation, teleport) for this frame."""
        location, rotation, scale = (np.asarray(v, F64) for v in (location, rotation, scale))
        if self.previous is None:
            self.previous = (location.copy(), rotation.copy(), scale.copy())
        prev_location, prev_rotation, _prev_scale = self.previous
        reciprocal = np.where(np.abs(scale) <= F64(ue.SMALL), 0.0, 1.0 / np.where(scale == 0, 1.0, scale))
        move = ue.unrotate_vector(rotation[None], (prev_location - location)[None])[0] * reciprocal
        move = move * self.move_scale
        inverse = np.array([-rotation[0], -rotation[1], -rotation[2], rotation[3]])
        move_rotation = ue.quat_multiply(inverse[None], prev_rotation[None])[0]
        teleport = False
        if self.teleport_distance > 0 and (move @ move) > F64(self.teleport_distance) * F64(self.teleport_distance):
            teleport = True
        angle = 2.0 * math.acos(max(-1.0, min(1.0, move_rotation[3])))
        if self.teleport_rotation > 0 and F32(math.degrees(angle)) > self.teleport_rotation:
            teleport = True
        return move, move_rotation, teleport

    def consume(self, fraction, location, rotation, scale):
        """Advance the previous transform by the consumed fraction of the frame."""
        location, rotation, scale = (np.asarray(v, F64) for v in (location, rotation, scale))
        fraction = F32(min(max(fraction, F32(0.0)), F32(1.0)))
        if fraction >= F32(1.0) - ue.KINDA_SMALL or self.previous is None:
            self.previous = (location.copy(), rotation.copy(), scale.copy())
            return
        prev_location, prev_rotation, prev_scale = self.previous
        self.previous = (ue.lerp(prev_location[None], location[None], fraction)[0],
                         ue.slerp(prev_rotation[None], rotation[None], fraction)[0],
                         ue.lerp(prev_scale[None], scale[None], fraction)[0])
