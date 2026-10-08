"""Shared cloth paint contract: display-space red freedom, green hold, blue ride."""
import numpy as np

ATTRIBUTE = "Waifu Cloth"
LEGACY_ATTRIBUTES = ("WS Cloth",)
RIDE_THRESHOLD = 0.5
MAX_DISTANCE_CM = 40.0


def attribute(mesh):
    found = mesh.color_attributes.get(ATTRIBUTE)
    if found is None:
        for name in LEGACY_ATTRIBUTES:
            found = mesh.color_attributes.get(name)
            if found is not None:
                found.name = ATTRIBUTE
                break
    return found


def channels(mesh):
    found = attribute(mesh)
    if found is None:
        return None
    values = np.empty(len(found.data) * 4, dtype=np.float32)
    found.data.foreach_get("color_srgb", values)
    values = values.reshape(-1, 4)[:, :3]
    if found.domain == "POINT":
        return values
    corners = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", corners)
    total = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    np.add.at(total, corners, values)
    count = np.bincount(corners, minlength=len(mesh.vertices))[:, None]
    return (total / np.maximum(count, 1)).astype(np.float32)


def ensure_attribute(mesh):
    found = attribute(mesh)
    if found is None:
        found = mesh.color_attributes.new(ATTRIBUTE, "FLOAT_COLOR", "POINT")
        values = np.zeros((len(found.data), 4), dtype=np.float32)
        values[:, 3] = 1
        found.data.foreach_set("color", values.ravel())
    mesh.color_attributes.active_color = found
    return found
