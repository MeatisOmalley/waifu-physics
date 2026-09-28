"""Colliders: mesh objects parented to bones, their shape drawn by Geometry Nodes.

A collider belongs to the armature it is parented to; a group collides with
the colliders of its own armature, or of the armatures listed in its collider
sets. A collider on no armature (a ground; it may hang from any other object)
is the scene's, and every group collides with those unless told not to. Its shape and sizes are the inputs of its "Waifu Physics Collider" modifier
(animatable), and its object scale multiplies them:

    Sphere, Inner Sphere   radius  x the largest axis scale
    Capsule                radius  x the larger of the X and Z scales,
                           length  x the Y scale (capsules run along local Y, as bones do)
    Tapered Capsule        as the capsule; Radius at the +Y end, Radius 1 at -Y
    Box                    half extents x the X, Y and Z scales
    Plane                  the local XY plane, facing +Z (size is only for drawing)

The node group draws exactly that shape: it builds it at the effective sizes
and divides out the object's own scale, so what is drawn is what collides.
Kawaii's capsules run along Z; `shape_of` turns ours onto its axis.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ..solver.system import Shape, SPHERE_OUTER, SPHERE_INNER, CAPSULE, TAPERED, BOX, PLANE

TREE = ".Waifu Physics Collider"
MODIFIER = "Waifu Physics Collider"
TREE_VERSION = 1
SHAPES = ("Sphere", "Inner Sphere", "Capsule", "Tapered Capsule", "Box", "Plane")
SHAPE_ICONS = {"AUTO": "MOD_SHRINKWRAP", "Sphere": "MESH_UVSPHERE", "Inner Sphere": "SPHERE",
               "Capsule": "MESH_CAPSULE", "Tapered Capsule": "MESH_CONE", "Box": "MESH_CUBE", "Plane": "MESH_PLANE"}
SHAPE_NOTES = {"Sphere": "A ball the chains stay outside of",
               "Inner Sphere": "A ball the chains stay inside of",
               "Capsule": "A cylinder with round ends, along the bone",
               "Tapered Capsule": "A capsule thicker at one end than the other",
               "Box": "A box the chains stay outside of",
               "Plane": "An endless plane the chains stay above: a ground"}
# What a collider added to a bone may be: fitted to the skin, as one shape or the best fitting.
SHAPE_CHOICES = [("AUTO", "Auto (Best Fit)", "Pick the best-fitting shape for each bone",
                  SHAPE_ICONS["AUTO"], 0)] + [(s, s, SHAPE_NOTES[s], SHAPE_ICONS[s], n + 1)
                                              for n, s in enumerate(SHAPES)]
KINDS = {"Sphere": SPHERE_OUTER, "Inner Sphere": SPHERE_INNER, "Capsule": CAPSULE, "Tapered Capsule": TAPERED,
         "Box": BOX, "Plane": PLANE}
# Turns a shape running along local Y onto Kawaii's Z: X stays, Z goes to Y.
Y_TO_Z = Quaternion((1.0, 0.0, 0.0), -math.pi / 2)
COLLECTION = "Waifu Physics Colliders"


def node_group():
    """The collider node group, built once (and rebuilt if an older version is found)."""
    tree = bpy.data.node_groups.get(TREE)
    if tree is not None and tree.get("waifu_physics_version") == TREE_VERSION:
        return tree
    if tree is None:
        tree = bpy.data.node_groups.new(TREE, "GeometryNodeTree")
    tree.nodes.clear()
    tree.interface.clear()
    tree["waifu_physics_version"] = TREE_VERSION
    iface = tree.interface
    iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    iface.new_socket("Shape", in_out="INPUT", socket_type="NodeSocketMenu")
    for name, default in (("Radius", 0.05), ("Radius 1", 0.05), ("Length", 0.2)):
        socket = iface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        socket.default_value = default
        socket.min_value = 0.0
        socket.subtype = "DISTANCE"
    extent = iface.new_socket("Extent", in_out="INPUT", socket_type="NodeSocketVector")
    extent.default_value = (0.05, 0.05, 0.05)
    extent.subtype = "TRANSLATION"

    nodes, links = tree.nodes, tree.links
    new = nodes.new
    gi, go = new("NodeGroupInput"), new("NodeGroupOutput")

    def math_node(op, a, b=None):
        node = new("ShaderNodeMath")
        node.operation = op
        links.new(a, node.inputs[0])
        if b is not None:
            if isinstance(b, (int, float)):
                node.inputs[1].default_value = b
            else:
                links.new(b, node.inputs[1])
        return node.outputs[0]

    def vector_node(op, a, b=None):
        node = new("ShaderNodeVectorMath")
        node.operation = op
        links.new(a, node.inputs[0])
        if b is not None:
            if isinstance(b, (tuple, list)):
                node.inputs[1].default_value = b
            else:
                links.new(b, node.inputs[1])
        return node.outputs[0] if op not in ("DOT_PRODUCT", "LENGTH") else node.outputs[1]

    # The object's own scale, which the drawing divides out.
    info = new("GeometryNodeObjectInfo")
    info.transform_space = "ORIGINAL"
    links.new(new("GeometryNodeSelfObject").outputs[0], info.inputs["Object"])
    scale = info.outputs["Scale"]
    split = new("ShaderNodeSeparateXYZ")
    links.new(scale, split.inputs[0])
    ax, ay, az = (math_node("ABSOLUTE", split.outputs[k]) for k in range(3))
    most = math_node("MAXIMUM", math_node("MAXIMUM", ax, ay), az)
    most_xz = math_node("MAXIMUM", ax, az)
    radius, radius1, length = gi.outputs["Radius"], gi.outputs["Radius 1"], gi.outputs["Length"]

    def sphere(r):
        uv = new("GeometryNodeMeshUVSphere")
        uv.inputs["Segments"].default_value = 24
        uv.inputs["Rings"].default_value = 11            # odd: no ring on the equator, so capsules stretch cleanly
        if isinstance(r, (int, float)):
            uv.inputs["Radius"].default_value = r
        else:
            links.new(r, uv.inputs["Radius"])
        return uv.outputs["Mesh"]

    def along_y(geometry):
        turn = new("GeometryNodeTransform")
        turn.inputs["Rotation"].default_value = (-math.pi / 2, 0.0, 0.0)
        links.new(geometry, turn.inputs["Geometry"])
        return turn.outputs["Geometry"]

    def z_sign():
        position = new("GeometryNodeInputPosition")
        parts = new("ShaderNodeSeparateXYZ")
        links.new(position.outputs[0], parts.inputs[0])
        return math_node("SIGN", parts.outputs[2]), position.outputs[0]

    half_length = math_node("MULTIPLY", math_node("MULTIPLY", length, ay), 0.5)

    # Sphere and inner sphere.
    ball = sphere(math_node("MULTIPLY", radius, most))
    # Capsule: a sphere of the capsule's radius, its halves pushed apart along Z, then laid along Y.
    capsule_sphere = sphere(math_node("MULTIPLY", radius, most_xz))
    sign, _position = z_sign()
    offset = new("ShaderNodeCombineXYZ")
    links.new(math_node("MULTIPLY", sign, half_length), offset.inputs[2])
    stretched = new("GeometryNodeSetPosition")
    links.new(capsule_sphere, stretched.inputs["Geometry"])
    links.new(offset.outputs[0], stretched.inputs["Offset"])
    capsule = along_y(stretched.outputs[0])
    # Tapered capsule: the +Z half at Radius, the -Z half at Radius 1.
    unit = sphere(1.0)
    sign_t, position_t = z_sign()
    top = math_node("GREATER_THAN", sign_t, 0.0)
    r_top = math_node("MULTIPLY", radius, most_xz)
    r_bottom = math_node("MULTIPLY", radius1, most_xz)
    chosen = new("GeometryNodeSwitch")
    chosen.input_type = "FLOAT"
    links.new(top, chosen.inputs["Switch"])
    links.new(r_bottom, chosen.inputs["False"])
    links.new(r_top, chosen.inputs["True"])
    scaled = vector_node("SCALE", position_t)
    links.new(chosen.outputs[0], scaled.node.inputs["Scale"])
    offset_t = new("ShaderNodeCombineXYZ")
    links.new(math_node("MULTIPLY", sign_t, half_length), offset_t.inputs[2])
    placed = vector_node("ADD", scaled, offset_t.outputs[0])
    tapered_set = new("GeometryNodeSetPosition")
    links.new(unit, tapered_set.inputs["Geometry"])
    links.new(placed, tapered_set.inputs["Position"])
    tapered = along_y(tapered_set.outputs[0])
    # Box: twice the half extents, times the scale.
    size = vector_node("MULTIPLY", vector_node("MULTIPLY", gi.outputs["Extent"], (2.0, 2.0, 2.0)),
                       vector_node("ABSOLUTE", scale))
    cube = new("GeometryNodeMeshCube")
    links.new(size, cube.inputs["Size"])
    # Plane: a square the size of four radii, with a line along its normal.
    grid = new("GeometryNodeMeshGrid")
    grid.inputs["Vertices X"].default_value = 2
    grid.inputs["Vertices Y"].default_value = 2
    side = math_node("MULTIPLY", math_node("MULTIPLY", radius, most), 4.0)
    links.new(side, grid.inputs["Size X"])
    links.new(side, grid.inputs["Size Y"])
    normal = new("GeometryNodeMeshLine")
    normal.mode = "END_POINTS"
    normal.inputs["Count"].default_value = 2
    tip = new("ShaderNodeCombineXYZ")
    links.new(math_node("MULTIPLY", side, 0.5), tip.inputs[2])
    links.new(tip.outputs[0], normal.inputs["Offset"])          # shown as End Location in this mode
    plane = new("GeometryNodeJoinGeometry")
    links.new(normal.outputs[0], plane.inputs[0])
    links.new(grid.outputs["Mesh"], plane.inputs[0])

    menu = new("GeometryNodeMenuSwitch")
    menu.data_type = "GEOMETRY"
    menu.enum_items.clear()
    for name in SHAPES:
        menu.enum_items.new(name)
    links.new(gi.outputs["Shape"], menu.inputs["Menu"])
    for name, geometry in zip(SHAPES, (ball, ball, capsule, tapered, cube.outputs["Mesh"], plane.outputs[0])):
        links.new(geometry, menu.inputs[name])
    # Divide out the object's scale: the drawing is the shape the solver uses, not a scaled one.
    divide = new("ShaderNodeVectorMath")
    divide.operation = "DIVIDE"
    divide.inputs[0].default_value = (1.0, 1.0, 1.0)
    links.new(scale, divide.inputs[1])
    unscale = new("GeometryNodeTransform")
    links.new(menu.outputs[0], unscale.inputs["Geometry"])
    links.new(divide.outputs[0], unscale.inputs["Scale"])
    links.new(unscale.outputs["Geometry"], go.inputs["Geometry"])
    return tree


def is_collider(obj):
    return obj is not None and obj.type == "MESH" and obj.waifu_physics_collider.is_collider


def is_scene_collider(obj):
    """A collider on no armature: the scene's."""
    return is_collider(obj) and (obj.parent is None or obj.parent.type != "ARMATURE")


def scene_colliders(scene, enabled_only=True):
    """The scene's colliders (on no armature), by name; enabled ones only unless asked."""
    return sorted((obj for obj in scene.objects if is_scene_collider(obj)
                   and (obj.waifu_physics_collider.enabled or not enabled_only)), key=lambda obj: obj.name)


def _collection(scene):
    """The colliders' collection, made and put in the scene if need be (only when a collider is made)."""
    collection = bpy.data.collections.get(COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION)
    if collection not in scene.collection.children_recursive:
        scene.collection.children.link(collection)
    return collection


def _view_layer(scene):
    return bpy.context.view_layer if bpy.context.scene == scene else scene.view_layers[0]


def visible(scene):
    """Are the colliders (and the chains' collision spheres) shown: the Colliders panel's eye."""
    return scene.waifu_physics.show_colliders


def _layer_collection(view_layer):
    """The view layer's entry for the colliders' collection, or None."""
    pending = [view_layer.layer_collection]
    while pending:
        found = pending.pop()
        if found.collection.name == COLLECTION:
            return found
        pending += found.children
    return None


def apply_shown(scene):
    """Show or hide the colliders as visible() says, by their collection's eye: one switch, which new
    colliders join by being made in the collection. A collider's own eye stays the user's, and one moved out
    of the collection leaves the switch. Nothing is made when there are no colliders. The eye only hides:
    hidden colliders still follow their bones and collide (the monitor toggle, hide_viewport on the
    collection itself, would stop their evaluation)."""
    found = _layer_collection(_view_layer(scene))
    if found is not None and found.hide_viewport == visible(scene):
        found.hide_viewport = not visible(scene)


def remove(obj):
    """Delete a collider and its mesh."""
    mesh = obj.data
    bpy.data.objects.remove(obj)
    if mesh is not None and not mesh.users:
        bpy.data.meshes.remove(mesh)


# --------------------------------------------------------------------------- the collider the lists show as picked
# It is the viewport's selection, so the two always agree: the active object if it is a collider; in Pose Mode, a
# collider on the active bone. Picking one in a list selects it (in Pose Mode, its bone).

def picked(scene, stored=None):
    """The collider the viewport has picked, else the one last picked in a list (stored), else None."""
    view_layer = _view_layer(scene)
    active = view_layer.objects.active
    if is_collider(active):
        return active
    if active is not None and active.type == "ARMATURE" and active.mode == "POSE":
        bone = active.data.bones.active
        on_bone = [obj for obj in all_of(active) if bone is not None and obj.parent_bone == bone.name]
        if stored in on_bone:
            return stored
        if on_bone:
            return on_bone[0]
        return stored if stored is not None and is_scene_collider(stored) else None
    return stored if is_collider(stored) else None


def pick(scene, obj):
    """Select a collider in the viewport: the object itself, or in Pose Mode on its own armature, its bone."""
    view_layer = _view_layer(scene)
    active = view_layer.objects.active
    if active is not None and active.type == "ARMATURE" and active.mode == "POSE":
        if obj.parent == active and obj.parent_type == "BONE" and obj.parent_bone in active.pose.bones:
            for bone in active.pose.bones:
                bone.select = bone.name == obj.parent_bone
            active.data.bones.active = active.data.bones[obj.parent_bone]
        return
    if active is not None and active.mode != "OBJECT" or obj.name not in view_layer.objects:
        return                                            # editing something: its selection is left alone
    for other in view_layer.objects.selected:
        other.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj


def _posing(view_layer):
    """The armature in Pose Mode whose bones stand for its colliders, or None."""
    active = view_layer.objects.active
    return active if active is not None and active.type == "ARMATURE" and active.mode == "POSE" else None


def _bone_of(armature, obj):
    """The pose bone a collider hangs from on this armature, or None."""
    if obj.parent != armature or obj.parent_type != "BONE":
        return None
    return armature.pose.bones.get(obj.parent_bone)


def generation_bones(context, active_only=False):
    """Targets for Add/Regenerate: pose bones or selected bone-parented colliders.

    Use actual selection, not the panel's remembered collider. Scene colliders
    have no bone to fit to and do not enable these armature-collider commands.
    """
    obj = context.object
    if context.mode == "POSE" and obj is not None and obj.type == "ARMATURE":
        bones = [context.active_pose_bone] if active_only else (context.selected_pose_bones or ())
        return [b for b in bones if b is not None and b.id_data == obj]
    if context.mode != "OBJECT":
        return []
    selected = list(context.selected_objects)
    if active_only and obj in selected:
        selected.remove(obj)
        selected.insert(0, obj)
    found, seen = [], set()
    for collider in selected:
        parent = collider.parent
        if not is_collider(collider) or parent is None or parent.type != "ARMATURE":
            continue
        bone = _bone_of(parent, collider)
        if bone is not None and (parent.name, bone.name) not in seen:
            seen.add((parent.name, bone.name))
            found.append(bone)
            if active_only:
                break
    return found


def chosen(scene, stored=None):
    """Every collider the viewport has selected, the picked one (picked) among them: selected collider objects;
    in Pose Mode, the colliders on the armature's selected bones."""
    view_layer = _view_layer(scene)
    armature = _posing(view_layer)
    if armature is not None:
        found = [obj for obj in all_of(armature) if (bone := _bone_of(armature, obj)) is not None and bone.select]
    else:
        found = [obj for obj in view_layer.objects.selected if is_collider(obj)]
    current = picked(scene, stored)
    if current is not None and current not in found:
        found.append(current)
    return found


def select(scene, objs, state=True, activate=None):
    """Add colliders to the viewport's selection (state False: take them out), leaving the rest of it: the
    objects, or in Pose Mode their bones. activate: the collider to make picked (Pose Mode: its bone active)."""
    view_layer = _view_layer(scene)
    armature = _posing(view_layer)
    if armature is not None:
        for obj in objs:
            bone = _bone_of(armature, obj)
            if bone is not None:
                bone.select = state
        bone = _bone_of(armature, activate) if activate is not None else None
        if bone is not None:
            armature.data.bones.active = armature.data.bones[bone.name]
        return
    active = view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        return                                            # editing something: its selection is left alone
    for obj in objs:
        if obj.name in view_layer.objects:
            if state:
                obj.hide_set(False)
            obj.select_set(state)
    if activate is not None and activate.name in view_layer.objects:
        view_layer.objects.active = activate


def unpick(scene, obj):
    """Take the picked collider out of the selection: the pick moves to another selected collider, if any."""
    view_layer = _view_layer(scene)
    select(scene, [obj], False)
    rest = [other for other in chosen(scene) if other != obj]
    armature = _posing(view_layer)
    if armature is not None:
        bone = _bone_of(armature, rest[-1]) if rest else None
        armature.data.bones.active = armature.data.bones[bone.name] if bone is not None else None
    elif view_layer.objects.active == obj:
        view_layer.objects.active = rest[-1] if rest else None
    return rest[-1] if rest else None


def listed_armatures(context):
    """The armatures the Colliders page lists, in the scene's order: every one with colliders, and the active
    one, so it can be given its first. What is listed follows what exists, not what is selected."""
    active = armature_of(context)
    return [obj for obj in context.scene.objects
            if obj.type == "ARMATURE" and (obj == active or all_of(obj))]


def listed(context):
    """The colliders the Colliders page shows, top to bottom: those in open armature folders, then the scene's.
    Shift-click selects a run of them."""
    found = []
    for armature in listed_armatures(context):
        if armature.waifu_physics.colliders_expanded:
            found += all_of(armature)
    return found + scene_colliders(context.scene, enabled_only=False)


def modifier(obj):
    return obj.modifiers.get(MODIFIER)


def _identifiers(tree):
    return {item.name: item.identifier for item in tree.interface.items_tree if getattr(item, "in_out", "") == "INPUT"}


def values(obj):
    """The collider's shape and sizes, from its modifier."""
    md = modifier(obj)
    if md is None or md.node_group is None:
        return None
    ids = _identifiers(md.node_group)
    inputs = md.properties.inputs
    return {name: (getattr(inputs, ident).value if name != "Extent" else tuple(getattr(inputs, ident).value))
            for name, ident in ids.items()}


def set_value(obj, name, value):
    md = modifier(obj)
    setattr(getattr(md.properties.inputs, _identifiers(md.node_group)[name]), "value", value)
    obj.update_tag()                      # setting an input from Python does not re-evaluate by itself


def input_path(obj, name):
    """The animatable data path of one of the collider's inputs, for drawing or keying."""
    md = modifier(obj)
    return md, _identifiers(md.node_group)[name]


def default_sources(armature):
    """The armatures a group collides with when it names none: its own, and the armature it hangs from, if
    any (a garment rig parented to a body collides with the body's colliders)."""
    found = [armature]
    parent = armature.parent
    while parent is not None:
        if parent.type == "ARMATURE":
            found.append(parent)
            break
        parent = parent.parent
    return found


def sources(group):
    """The armatures whose colliders a group collides with: its list once edited (it may be empty; older files
    mark an edited list only by its entries), else the defaults."""
    if group.custom_collider_sets or len(group.collider_sets):
        return [item.armature for item in group.collider_sets if item.armature is not None]
    return default_sources(group.id_data)


def group_colliders(group, scene, enabled_only=True):
    """The colliders a group collides with: with Collide Against Every Collider, every collider in the scene; otherwise
    its armatures' (sources) and, with Scene Colliders, the scene's."""
    def wanted(obj):
        return obj.waifu_physics_collider.enabled or not enabled_only
    if group.use_all_colliders:
        return [obj for obj in scene.objects if is_collider(obj) and wanted(obj)]
    found = [obj for armature in sources(group) for obj in all_of(armature) if wanted(obj)]
    if group.use_scene_colliders:
        found += scene_colliders(scene, enabled_only)
    return found


def armature_of(context):
    """The armature whose colliders the panel shows: the active armature, or a selected collider's."""
    obj = context.object
    if obj is None:
        return None
    if obj.type == "ARMATURE":
        return obj
    if is_collider(obj) and obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    return None


def all_of(armature):
    """Every collider parented to the armature, on or off, by name."""
    return sorted((obj for obj in armature.children if is_collider(obj)), key=lambda obj: obj.name)


def colliders_of(armature):
    """Enabled colliders parented to this armature."""
    return [obj for obj in armature.children if is_collider(obj) and obj.waifu_physics_collider.enabled]


def shape_of(obj, armature, cm, matrix=None):
    """The collider as a solver Shape in the armature's space, centimetres. matrix: the collider's matrix in
    the armature's space, when the caller knows it better than Blender's last evaluation (read ahead)."""
    found = values(obj)
    if found is None:
        return None
    m = Matrix(matrix.tolist()) if matrix is not None else armature.matrix_world.inverted_safe() @ obj.matrix_world
    location, rotation, scale = m.decompose()
    sx, sy, sz = (abs(v) for v in scale)
    kind = KINDS.get(found["Shape"], SPHERE_OUTER)
    radius, radius1, length = found["Radius"], found["Radius 1"], found["Length"]
    if kind in (CAPSULE, TAPERED):
        rotation = rotation @ Y_TO_Z
        radius, radius1, length = radius * max(sx, sz), radius1 * max(sx, sz), length * sy
    elif kind in (SPHERE_OUTER, SPHERE_INNER):
        radius = radius * max(sx, sy, sz)
    extent = Vector(found["Extent"])
    extent = (extent.x * sx * cm, extent.y * sy * cm, extent.z * sz * cm)
    return Shape(kind, location=tuple(location * cm), rotation=(rotation.x, rotation.y, rotation.z, rotation.w),
                 radius=radius * cm, radius1=radius1 * cm, length=length * cm, extent=extent)


def short_name(bone_name):
    """A bone's name without VRoid's prefixes: J_Bip_C_Head -> Head, J_Bip_L_UpperArm -> L_UpperArm."""
    import re
    name = re.sub(r"^J_(Bip|Sec|Adj)_", "", bone_name)
    return re.sub(r"^C_", "", name) or bone_name


def _skinned_meshes(armature):
    return [obj for obj in bpy.data.objects if obj.type == "MESH" and not is_collider(obj)
            and any(md.type == "ARMATURE" and md.object == armature for md in obj.modifiers)]


def _coincides(bone, other):
    return (bone.head_local - other.head_local).length < 1e-5 and (bone.tail_local - other.tail_local).length < 1e-5


def _owner(bones, name):
    """The bone a vertex group's weights belong to: the group's bone, or the bone it lies on if it is a twin
    with the same head and tail (VRoid Swap's J_Scale_ bones scale the skin under each J_Bip_ bone)."""
    bone = bones.get(name)
    while bone is not None and bone.parent is not None and _coincides(bone, bone.parent):
        bone = bone.parent
    return bone.name if bone is not None else None


def skin_points(armature):
    """{bone name: the world positions, at rest, of the skin that belongs to it}: each vertex of the meshes the
    armature deforms goes to the bone with its largest weight, twins counting for the bone they lie on. So a
    bone's points are the skin both around it and weighted to it."""
    bones = armature.data.bones
    found = {}
    for obj in _skinned_meshes(armature):
        owners = [_owner(bones, group.name) for group in obj.vertex_groups]
        co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", co)
        mw = np.array(obj.matrix_world)
        world = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
        chosen = {}
        for vertex in obj.data.vertices:
            best, weight = None, 0.0
            for entry in vertex.groups:
                owner = owners[entry.group] if entry.group < len(owners) else None
                if owner is not None and entry.weight > weight:
                    best, weight = owner, entry.weight
            if best is not None:
                chosen.setdefault(best, []).append(vertex.index)
        for bone, indices in chosen.items():
            found.setdefault(bone, []).append(world[indices])
    return {bone: np.concatenate(parts) for bone, parts in found.items()}


def fit(armature, bone_name, shape="AUTO", points=None):
    """A collider_fit.Fit for the bone in its rest frame, or None when too little skin belongs to it."""
    from . import collider_fit
    if points is None:
        points = skin_points(armature)
    world = points.get(bone_name)
    if world is None:
        return None
    to_bone = np.linalg.inv(np.array(armature.matrix_world) @ np.array(armature.data.bones[bone_name].matrix_local))
    local = world @ to_bone[:3, :3].T + to_bone[:3, 3]
    return collider_fit.fit(local, shape)


def has_collider(armature, bone_name):
    return any(is_collider(obj) and obj.parent_bone == bone_name for obj in armature.children)


def simulated_bones(armature):
    """The bones the armature's groups simulate: their chains below the roots (a root keeps its animation).
    A collider on one would follow the simulation and push the chain that moves it, chasing it for ever (Kawaii's
    follow the animation instead), so none is added there."""
    from . import links
    found = set()
    for group in armature.waifu_physics.groups:
        excluded = [bone.name for bone in group.excluded]
        for root in group.roots:
            found.update(links.chain_subtree(armature, root.name, excluded)[1:])
    return found


def on_simulated_bone(obj):
    parent = obj.parent
    return (parent is not None and parent.type == "ARMATURE" and obj.parent_type == "BONE"
            and obj.parent_bone in simulated_bones(parent))


def from_bones(armature, bone_names, shape="AUTO", context=None, replace=False):
    """A collider fitted to the skin of each bone, simulated bones skipped. A bone with colliders already is
    skipped too, or with replace, has them replaced by the one fitted. Returns the new colliders."""
    points = skin_points(armature)
    skipped = simulated_bones(armature)
    made = []
    for name in bone_names:
        existing = [obj for obj in all_of(armature) if obj.parent_bone == name]
        if name in skipped or existing and not replace:
            continue
        for obj in existing:
            remove(obj)
        made.append(add(armature, name, shape, context, points))
    return made


def _new(name, context=None):
    """A collider object in the colliders' collection, in the scene and shown (a new one should be seen)."""
    scene = (context or bpy.context).scene
    collection = _collection(scene)
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    collection.objects.link(obj)
    obj.waifu_physics_collider.is_collider = True
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.hide_render = True
    md = obj.modifiers.new(MODIFIER, "NODES")
    md.node_group = node_group()
    apply_shown(scene)                       # the collection may be new, or linked into this scene just now
    return obj


def add_to_scene(shape="Plane", context=None, location=(0.0, 0.0, 0.0)):
    """A new scene collider at a location, upright: a Plane faces up, so it is a ground."""
    obj = _new("Ground Collider" if shape == "Plane" else f"{shape} Collider", context)
    obj.location = location
    set_value(obj, "Shape", shape)
    set_value(obj, "Radius", 1.0 if shape == "Plane" else 0.25)
    set_value(obj, "Radius 1", 0.25)
    set_value(obj, "Length", 0.5)
    set_value(obj, "Extent", (0.25, 0.25, 0.25))
    return obj


def add(armature, bone_name, shape="AUTO", context=None, points=None):
    """A new collider on a bone, fitted to the skin that belongs to it (AUTO: the shape that fits it best).
    Where no skin belongs to the bone: centred on it, a capsule along it, a quarter of its length thick."""
    obj = _new(f"{short_name(bone_name)} Collider", context)
    pose_bone = armature.pose.bones[bone_name]
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    bone_length = pose_bone.bone.length
    fitted = None
    if shape != "Plane":
        fitted = fit(armature, bone_name, "Sphere" if shape == "Inner Sphere" else shape, points)
    if fitted is None:
        center, kind = (0.0, bone_length / 2, 0.0), ("Capsule" if shape == "AUTO" else shape)
        radius = radius1 = bone_length * 0.25
        length, extent = bone_length, (bone_length * 0.25,) * 3
    else:
        center, kind = tuple(fitted.center), (shape if shape == "Inner Sphere" else fitted.shape)
        radius, radius1, length, extent = fitted.radius, fitted.radius1, fitted.length, tuple(fitted.extent)
    obj.matrix_world = armature.matrix_world @ pose_bone.matrix @ Matrix.Translation(center)
    set_value(obj, "Shape", kind)
    set_value(obj, "Radius", radius)
    set_value(obj, "Radius 1", radius1)
    set_value(obj, "Length", length)
    set_value(obj, "Extent", extent)
    return obj
