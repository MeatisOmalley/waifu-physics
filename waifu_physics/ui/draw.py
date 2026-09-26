"""Viewport overlay: a group's links drawn as lines between the bones they join (the link picked in the active
group's Links list in red, on top), and with Show Colliders, the chains' collision spheres (a circle facing the
view round each point, as big as it collides)."""
import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

_handle = None
COLOURS = ((0.35, 0.8, 1.0, 1.0), (1.0, 0.6, 0.25, 1.0), (0.6, 1.0, 0.4, 1.0), (1.0, 0.45, 0.8, 1.0))
PICKED = (1.0, 0.15, 0.15, 1.0)


SEGMENTS = 20
_CIRCLE = [(math.cos(2 * math.pi * k / SEGMENTS), math.sin(2 * math.pi * k / SEGMENTS)) for k in range(SEGMENTS + 1)]


def _simulated_spheres(scene):
    """(world position, world radius, group colour) of every simulated point, from the running simulation: what
    collides, where it is now. A cache plays its frames as bone channels only, so then the points are the
    shown frame's, as cached. None when nothing simulates, or a cache has no frame here (the pose shows)."""
    from ..runtime import live
    rt = live._runtimes.get(scene.as_pointer())
    if rt is None or getattr(rt, "system", None) is None:
        return None
    s = rt.system
    points = s.loc
    if live.is_cached(scene):
        snapshot = rt.cache.get(scene.frame_current)
        if snapshot is None or len(snapshot.loc) != len(points):
            return None                          # uncached frame, or a bake made for other chains
        points = snapshot.loc
    found = []
    try:
        worlds = [rig.obj.matrix_world for rig in rt.rigs]
        for i in np.flatnonzero(s.parent >= 0):
            rig, index = rt.group_props[s.group[i]]
            world = worlds[rig]
            found.append((world @ Vector(points[i] / rt.cm), float(s.radius[i]) / rt.cm * max(world.to_scale()),
                          COLOURS[index % len(COLOURS)]))
    except (ReferenceError, IndexError, AttributeError):
        return None
    return found


def _posed_spheres(scene):
    """The same from the pose, when nothing simulates: each chain bone's head below its root and each tip's
    tail, sized by the group's radius and its curve along the chain."""
    from ..data import curves
    from ..data.links import chain_subtree
    found = []
    for obj in scene.objects:
        if obj.type != "ARMATURE" or not obj.visible_get() or not len(obj.waifu_physics.groups):
            continue
        world, scale = obj.matrix_world, max(obj.matrix_world.to_scale())
        bones = obj.pose.bones
        for index, group in enumerate(obj.waifu_physics.groups):
            if not group.enabled:
                continue
            colour = COLOURS[index % len(COLOURS)]
            curve = curves.curve(group, "radius")
            excluded = [bone.name for bone in group.excluded]
            for root in group.roots:
                names = chain_subtree(obj, root.name, excluded)[1:]
                points, reach = [], {}
                for name in names:
                    bone = bones[name]
                    reach[name] = reach.get(bone.parent.name, 0.0) + bone.parent.length if bone.parent else 0.0
                    points.append((bone.head, reach[name]))
                    if not any(child.name in names for child in bone.children):
                        points.append((bone.tail, reach[name] + bone.length))
                if not points:
                    continue
                longest = max(distance for _head, distance in points) or 1.0
                rates = np.array([distance / longest for _head, distance in points], dtype=np.float32)
                sizes = curve.many(rates) if curve is not None else np.ones(len(points))
                for (head, _distance), size in zip(points, sizes):
                    found.append((world @ head, group.radius * float(size) * scale, colour))
    return found


def _draw():
    context = bpy.context
    scene = context.scene
    if scene is None:
        return
    settings = scene.waifu_physics
    from ..data import colliders
    if colliders.visible(scene) and context.region_data is not None:
        _draw_spheres(context, scene)
    if not settings.show_links:
        return
    lines, colours, picked = [], [], []
    from . import manager
    focus = {context.object}                       # the active armature, and the one the chain manager shows
    if manager.is_open(context.area):
        focus.add(manager.shown(context))
    for obj in scene.objects:
        if obj.type != "ARMATURE" or not obj.visible_get():
            continue
        bones = obj.pose.bones
        world = obj.matrix_world
        for index, group in enumerate(obj.waifu_physics.groups):
            if not group.enabled:
                continue
            colour = COLOURS[index % len(COLOURS)]
            chosen = group.active_link if obj in focus and index == obj.waifu_physics.active_group else -1
            for number, link in enumerate(group.links):
                a, b = bones.get(link.bone_a), bones.get(link.bone_b)
                if a is None or b is None:
                    continue
                if number == chosen:
                    picked += [world @ a.head, world @ b.head]
                    continue
                lines += [world @ a.head, world @ b.head]
                colours += [colour, colour]
    if not lines and not picked:
        return
    shader = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
    region = context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    for points, tints, width in ((lines, colours, 2.0), (picked, [PICKED] * len(picked), 4.0)):
        if points:
            shader.uniform_float("lineWidth", width)
            batch_for_shader(shader, "LINES", {"pos": [tuple(p) for p in points], "color": tints}).draw(shader)
    gpu.state.blend_set("NONE")


def _draw_spheres(context, scene):
    spheres = _simulated_spheres(scene) if scene.waifu_physics.simulate else None
    if spheres is None:
        spheres = _posed_spheres(scene)
    if not spheres:
        return
    rotation = context.region_data.view_rotation
    right, up = rotation @ Vector((1.0, 0.0, 0.0)), rotation @ Vector((0.0, 1.0, 0.0))
    lines, colours = [], []
    for centre, radius, colour in spheres:
        faded = colour[:3] + (0.55,)
        ring = [tuple(centre + (right * c + up * s) * radius) for c, s in _CIRCLE]
        for a, b in zip(ring, ring[1:]):
            lines += [a, b]
        colours += [faded] * (2 * SEGMENTS)
    shader = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
    region = context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 1.2)
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    batch_for_shader(shader, "LINES", {"pos": lines, "color": colours}).draw(shader)
    gpu.state.blend_set("NONE")


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
