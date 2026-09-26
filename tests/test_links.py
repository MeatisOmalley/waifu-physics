"""Link Chains: neighbours side by side, rings left open, and links that hold a skirt together."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, fresh_import

import bpy
import numpy as np
from mathutils import Vector

addon = fresh_import()
addon.register()
live = sys.modules["waifu_physics.runtime.live"]
draw = sys.modules["waifu_physics.ui.draw"]
scene = bpy.context.scene
PANELS = 6


def skirt(name):
    """Six panels of four bones hanging from a ring, panel k at angle 60k degrees."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    hips = data.edit_bones.new("hips")
    hips.head, hips.tail = (0, 0, 1.0), (0, 0, 1.1)
    for k in range(PANELS):
        angle = 2 * math.pi * k / PANELS
        parent, head = hips, Vector((0.2 * math.cos(angle), 0.2 * math.sin(angle), 1.0))
        for i in range(4):
            bone = data.edit_bones.new(f"panel{k}_{i}")
            bone.head = head
            bone.tail = head + Vector((0.03 * math.cos(angle), 0.03 * math.sin(angle), -0.12))
            bone.parent = parent
            parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    group = obj.waifu_physics.groups.add()
    for k in range(PANELS):
        group.roots.add().name = f"panel{k}_0"
    group.dummy_bone_length = 0.05
    return obj, group


def link(obj, selection_order, bone="2"):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = False
    for k in selection_order:
        obj.pose.bones[f"panel{k}_{bone}"].select = True   # any bone of a chain selects the chain
    result = bpy.ops.waifu_physics.link_chains()
    bpy.ops.object.mode_set(mode="OBJECT")
    return result


panel = lambda name: int(name[5:name.index("_")])
depth = lambda name: int(name[name.index("_") + 1:])


def linked_panels(group):
    return {(min(panel(l.bone_a), panel(l.bone_b)), max(panel(l.bone_a), panel(l.bone_b))) for l in group.links}


def ring_link(obj, group):
    """Link every panel round the ring: all of them, then the two end chains."""
    link(obj, range(PANELS))
    ends = [k for k in range(PANELS) if sum(k in pair for pair in linked_panels(group)) == 1]
    link(obj, ends)


obj, group = skirt("ring")
check("Link Whole Chains runs", link(obj, [3, 0, 5, 1, 4, 2]) == {"FINISHED"})
pairs = linked_panels(group)
around = {tuple(sorted((k, (k + 1) % PANELS))) for k in range(PANELS)}
check("a skirt's panels link to the panels beside them, whatever the selection order (no zig-zag)",
      pairs < around and len(pairs) == PANELS - 1, sorted(pairs))
check("... at every depth below the roots, bone by bone", len(group.links) == (PANELS - 1) * 3 and
      all(depth(l.bone_a) == depth(l.bone_b) and depth(l.bone_a) >= 1 for l in group.links), len(group.links))
check("linking again adds nothing new", link(obj, range(PANELS)) == {"FINISHED"} and len(group.links) == 15)
ends = [k for k in range(PANELS) if sum(k in pair for pair in pairs) == 1]
check("the ring is left open: two end chains, side by side", len(ends) == 2 and tuple(sorted(ends)) in around, ends)
link(obj, ends)
check("linking the two end chains closes it", linked_panels(group) == around and len(group.links) == PANELS * 3,
      sorted(linked_panels(group)))
bpy.data.objects.remove(obj)

# --- links hold the panels' spacing where unlinked panels splay apart
spread = {}
for linked in (False, True):
    for other in list(bpy.data.objects):
        bpy.data.objects.remove(other)
    obj, group = skirt("run")
    if linked:
        ring_link(obj, group)
    group.gravity = (3.0, 0.0, -1.0)
    group.stiffness = 0.02
    group.compliance = "CONCRETE"
    scene.frame_set(1)
    scene.waifu_physics.simulate = True
    rest = np.array([obj.pose.bones[f"panel{k}_3"].head for k in range(PANELS)])
    for frame in range(1, 60):
        scene.frame_set(frame)
    now = np.array([obj.pose.bones[f"panel{k}_3"].head for k in range(PANELS)])
    gaps = lambda p: np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
    spread[linked] = np.abs(gaps(now) - gaps(rest)).max()
    if linked:
        rt = live.runtime(scene)
        user_links = len(group.links)
        solver_links = len(rt.system.link_a)
    scene.waifu_physics.simulate = False
check("linked panels keep their spacing far better than unlinked ones", spread[True] < spread[False] * 0.25, spread)
check("the solver gets the links plus automatic links between the panels' tips",
      solver_links == user_links + PANELS, (user_links, solver_links))

group.bridge_count = 1
scene.waifu_physics.simulate = True
scene.frame_set(2)
bridges = int((live.runtime(scene).system.kind == 3).sum())
check("bridge points appear along the links", bridges == solver_links, (bridges, solver_links))
scene.waifu_physics.simulate = False

check("the link overlay is drawing", draw._handle is not None)
# --- a flat cape: its chains in a straight row, linked along it
from waifu_physics.data import links as chain_links
cape_data = bpy.data.armatures.new("cape")
cape = bpy.data.objects.new("cape", cape_data)
bpy.context.scene.collection.objects.link(cape)
bpy.context.view_layer.objects.active = cape
bpy.ops.object.mode_set(mode="EDIT")
back = cape_data.edit_bones.new("back")
back.head, back.tail = (0, 0, 1.5), (0, 0, 1.6)
cape_roots = []
for i, x in enumerate((0.1, -0.2, 0.2, -0.1, 0.0)):        # made out of order, in a straight row
    parent, head = back, Vector((x, 0.1, 1.5))
    for depth in range(3):
        bone = cape_data.edit_bones.new(f"cape{i}_{depth}")
        bone.head, bone.tail = head, head + Vector((0, 0, -0.15))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
    cape_roots.append(f"cape{i}_0")
bpy.ops.object.mode_set(mode="OBJECT")
row = [pair for pair in chain_links.pairs(cape, cape_roots)[0] if pair[0].endswith("_1")]
x_of = {f"cape{i}_1": x for i, x in enumerate((0.1, -0.2, 0.2, -0.1, 0.0))}
steps = sorted(round(abs(x_of[a] - x_of[b]), 3) for a, b in row)
check("a flat cape links each chain to its neighbour along the row, and not its two edges together",
      len(row) == 4 and steps == [0.1] * 4, row)

# --- a cape wrapped round the back: an arc, opened at its front gap
wrap_data = bpy.data.armatures.new("wrap")
wrap = bpy.data.objects.new("wrap", wrap_data)
bpy.context.scene.collection.objects.link(wrap)
bpy.context.view_layer.objects.active = wrap
bpy.ops.object.mode_set(mode="EDIT")
neck = wrap_data.edit_bones.new("neck")
neck.head, neck.tail = (0, 0, 1.5), (0, 0, 1.6)
degrees = (135, 45, 270, 90, 225, 180, 315)                # 45 to 315 in 45 degree steps, made out of order;
wrap_roots = []                                            # the front (0 degrees) is open
for i, angle in enumerate(degrees):
    a = math.radians(angle)
    parent, head = neck, Vector((0.15 * math.cos(a), 0.15 * math.sin(a), 1.5))
    for d in range(3):
        bone = wrap_data.edit_bones.new(f"wrap{i}_{d}")
        bone.head, bone.tail = head, head + Vector((0.02 * math.cos(a), 0.02 * math.sin(a), -0.15))
        bone.parent = parent
        parent, head = bone, bone.tail.copy()
    wrap_roots.append(f"wrap{i}_0")
bpy.ops.object.mode_set(mode="OBJECT")
found, order = chain_links.pairs(wrap, wrap_roots)
angle_of = lambda name: degrees[int(name[4:name.index("_")])]
check("a cape wrapped round the back links neighbours round it, and leaves its front open",
      sorted((angle_of(order[0]), angle_of(order[-1]))) == [45, 315]
      and all(abs(angle_of(a) - angle_of(b)) == 45 for a, b in found), [angle_of(r) for r in order])

# --- Link Selected Bones links only the bones selected
def link_bones(obj, names):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name in names
    try:
        result = bpy.ops.waifu_physics.link_bones()
    except RuntimeError:
        result = {"CANCELLED"}
    bpy.ops.object.mode_set(mode="OBJECT")
    return result


obj, group = skirt("single")
check("two selected bones make exactly one link, not their whole chains",
      link_bones(obj, ["panel0_2", "panel1_2"]) == {"FINISHED"} and len(group.links) == 1
      and {group.links[0].bone_a, group.links[0].bone_b} == {"panel0_2", "panel1_2"},
      [(l.bone_a, l.bone_b) for l in group.links])
group.links.clear()
check("four bones round the skirt make three links between neighbours, the ring left open",
      link_bones(obj, ["panel0_1", "panel3_1", "panel1_1", "panel2_1"]) == {"FINISHED"}
      and {tuple(sorted((panel(l.bone_a), panel(l.bone_b)))) for l in group.links} == {(0, 1), (1, 2), (2, 3)},
      [(l.bone_a, l.bone_b) for l in group.links])
group.links.clear()
check("a bone outside the group's chains is not linked",
      link_bones(obj, ["panel0_1", "hips"]) == {"CANCELLED"} and len(group.links) == 0)


def can_link(obj, names, op="link_bones"):
    """Is the button live for this selection (its poll)?"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name in names
    live_button = getattr(bpy.ops.waifu_physics, op).poll()
    bpy.ops.object.mode_set(mode="OBJECT")
    return live_button


check("Link Selected Bones is greyed out with one bone, or none",
      not can_link(obj, ["panel0_2"]) and not can_link(obj, []))
check("... for two bones of one chain: the chain already keeps them apart",
      not can_link(obj, ["panel0_1", "panel0_3"]) and not can_link(obj, ["panel0_1", "panel0_2"]))
check("... for roots, which follow the animation", not can_link(obj, ["panel0_0", "panel1_0"])
      and not can_link(obj, ["panel0_0", "panel1_2"]))
check("... and live for two bones of different chains, at any depths",
      can_link(obj, ["panel0_1", "panel1_1"]) and can_link(obj, ["panel0_1", "panel1_3"]))
check("two bones of different chains at different depths make one diagonal link",
      link_bones(obj, ["panel0_1", "panel1_3"]) == {"FINISHED"}
      and [{l.bone_a, l.bone_b} for l in group.links] == [{"panel0_1", "panel1_3"}])
check("... and once linked, the button greys out", not can_link(obj, ["panel0_1", "panel1_3"]))
group.links.clear()
check("whole chains selected link bone by bone at each depth, never along a chain",
      link_bones(obj, [f"panel{k}_{i}" for k in (0, 1) for i in range(4)]) == {"FINISHED"}
      and sorted(tuple(sorted((int(l.bone_a[-1]), int(l.bone_b[-1])))) for l in group.links) == [(1, 1), (2, 2), (3, 3)],
      [(l.bone_a, l.bone_b) for l in group.links])
group.links.clear()
group.excluded.add().name = "panel1_2"
check("a bone the group excludes (or one under it) is not linked", not can_link(obj, ["panel0_3", "panel1_3"]))
group.excluded.clear()
check("Link Whole Chains is greyed out until bones of two chains are selected",
      not can_link(obj, ["panel0_1", "panel0_2"], "link_chains") and can_link(obj, ["panel0_1", "panel1_2"], "link_chains"))

addon.unregister()
check("... and stops when unregistered", draw._handle is None)
finish()
