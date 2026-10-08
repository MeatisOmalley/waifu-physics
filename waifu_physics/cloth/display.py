"""Non-destructive cloth display, after skinning, with an independent skin input."""
import bpy
import numpy as np

GROUP = ".Waifu Cloth Display"
MODIFIER = "Waifu Cloth Preview"
SKINNED = "waifu_cloth_skinned"
POSITION = "waifu_cloth_position"
ACTIVE = "waifu_cloth_active"


def node_group():
    tree = bpy.data.node_groups.get(GROUP)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new(GROUP, "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    incoming = nodes.new("NodeGroupInput")
    output = nodes.new("NodeGroupOutput")
    position = nodes.new("GeometryNodeInputPosition")
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT_VECTOR", "POINT"
    store.inputs["Name"].default_value = SKINNED
    links.new(incoming.outputs["Geometry"], store.inputs["Geometry"])
    links.new(position.outputs["Position"], store.inputs["Value"])
    simulated = nodes.new("GeometryNodeInputNamedAttribute")
    simulated.data_type = "FLOAT_VECTOR"
    simulated.inputs["Name"].default_value = POSITION
    active = nodes.new("GeometryNodeInputNamedAttribute")
    active.data_type = "BOOLEAN"
    active.inputs["Name"].default_value = ACTIVE
    place = nodes.new("GeometryNodeSetPosition")
    links.new(store.outputs["Geometry"], place.inputs["Geometry"])
    links.new(simulated.outputs["Attribute"], place.inputs["Position"])
    links.new(active.outputs["Attribute"], place.inputs["Selection"])
    links.new(place.outputs["Geometry"], output.inputs["Geometry"])
    return tree


def _attribute(mesh, name, kind):
    found = mesh.attributes.get(name)
    if found is not None and (found.data_type != kind or found.domain != "POINT"):
        mesh.attributes.remove(found)
        found = None
    return found if found is not None else mesh.attributes.new(name, kind, "POINT")


def ensure(obj):
    if obj.data.users > 1:
        raise ValueError(f"{obj.name}: cloth preview requires single-user mesh data; make its Object Data single-user")
    _attribute(obj.data, POSITION, "FLOAT_VECTOR")
    _attribute(obj.data, ACTIVE, "BOOLEAN")
    modifier = next((md for md in obj.modifiers if md.type == "NODES" and md.node_group is not None
                     and md.node_group.name == GROUP), None)
    if modifier is None:
        modifier = obj.modifiers.new(MODIFIER, "NODES")
        modifier.node_group = node_group()
    index = obj.modifiers.find(modifier.name)
    if index != len(obj.modifiers) - 1:
        obj.modifiers.move(index, len(obj.modifiers) - 1)
    return modifier


def write(obj, positions):
    values = np.ascontiguousarray(positions, dtype=np.float32)
    if values.shape != (len(obj.data.vertices), 3):
        raise ValueError(f"{obj.name}: cloth output topology changed")
    _attribute(obj.data, POSITION, "FLOAT_VECTOR").data.foreach_set("vector", values.ravel())
    _attribute(obj.data, ACTIVE, "BOOLEAN").data.foreach_set("value", np.ones(len(values), dtype=bool))
    obj.data.update()
    obj.update_tag(refresh={"DATA"})


def clear(obj):
    active = obj.data.attributes.get(ACTIVE)
    if active is not None:
        active.data.foreach_set("value", np.zeros(len(active.data), dtype=bool))
        obj.data.update()
        obj.update_tag(refresh={"DATA"})


def read(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        if len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError(f"{obj.name}: cloth preview requires modifiers that preserve vertex count")
        values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        skinned = mesh.attributes.get(SKINNED)
        if skinned is None:
            mesh.vertices.foreach_get("co", values)
        else:
            skinned.data.foreach_get("vector", values)
        return values.reshape(-1, 3), np.array(evaluated.matrix_world, dtype=np.float32)
    finally:
        evaluated.to_mesh_clear()
