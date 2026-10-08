"""Blender I/O for the scene's cloth; the numerical hot loop lives in cloth/step.c."""
from dataclasses import fields

import bpy
import numpy as np

from ..cloth import convention, display, native
from ..cloth.build import Settings, build
from ..cloth.system import System
from ..data import colliders
from ..data.cloth_props import VALUE_NAMES
from .io import cm_per_unit


def objects(scene):
    return [obj for obj in scene.objects if obj.type == "MESH" and obj.waifu_cloth.enabled]


def overrides(scene):
    """Authoring integrations can identify chain groups the cloth replaces during preview."""
    return {(obj.waifu_cloth.override_armature.session_uid, entry.name)
            for obj in objects(scene) if obj.waifu_cloth.override_armature is not None
            for entry in obj.waifu_cloth.override_groups}


def settings(obj, cm):
    values = {field.name: getattr(obj.waifu_cloth, field.name) for field in fields(Settings)}
    for name in ("max_distance", "thickness"):
        values[name] *= cm
    return Settings(**values)


def _world(points, matrix, cm):
    return np.ascontiguousarray((points @ matrix[:3, :3].T + matrix[:3, 3]) * cm, dtype=np.float32)


def _fingerprint(obj):
    mesh = obj.data
    positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    indices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", indices)
    paint = convention.channels(mesh)
    return (obj.session_uid, mesh.session_uid, len(mesh.polygons), hash(positions.tobytes()),
            hash(indices.tobytes()), None if paint is None else hash(paint.tobytes()),
            tuple(getattr(obj.waifu_cloth, name) for name in VALUE_NAMES),
            tuple(entry.armature.session_uid if entry.armature is not None else None
                  for entry in obj.waifu_cloth.collider_sets))


def _reference_bone(obj):
    armature = obj.find_armature()
    if armature is None:
        return None, None
    bones = armature.data.bones
    used = [bones[group.name] for group in obj.vertex_groups if group.name in bones]
    if not used:
        return armature, None
    paths = [[bone.name, *(parent.name for parent in bone.parent_recursive)] for bone in used]
    common = set(paths[0]).intersection(*paths[1:])
    return armature, next((name for name in paths[0] if name in common), None)


class ClothRuntime:
    def __init__(self, scene):
        self.scene = scene
        self.cm = cm_per_unit(scene)
        self.meshes = objects(scene)
        for obj in self.meshes:
            if obj.data.users > 1:
                raise ValueError(f"{obj.name}: cloth preview requires single-user mesh data; make its Object Data single-user")
        self.identities = [(obj.session_uid, obj.data.session_uid, len(obj.data.vertices)) for obj in self.meshes]
        self.prints = [_fingerprint(obj) for obj in self.meshes]
        self.references = [_reference_bone(obj) for obj in self.meshes]
        self.backend = native.backend()
        built = []
        for obj in self.meshes:
            display.ensure(obj)
            mesh = obj.data
            positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", positions)
            mesh.calc_loop_triangles()
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", triangles)
            values = convention.channels(mesh)
            if values is None:
                values = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
            built.append(build(_world(positions.reshape(-1, 3), np.array(obj.matrix_world), self.cm),
                               triangles.reshape(-1, 3), values, settings(obj, self.cm)))
        self.system = System(built)
        self.previous_matrices = None
        self.targets = None
        self.input_prints = self.static_inputs()

    def alive(self):
        try:
            return all(obj.session_uid == uid and obj.data.session_uid == data_uid and len(obj.data.vertices) == count
                       and obj.data.users == 1
                       and obj.name in self.scene.objects
                       for obj, (uid, data_uid, count) in zip(self.meshes, self.identities))
        except ReferenceError:
            return False

    def changed(self, scene):
        if not self.alive() or [obj.session_uid for obj in objects(scene)] != [x[0] for x in self.identities]:
            return True
        return self.prints != [_fingerprint(obj) for obj in self.meshes]

    def input_ids(self):
        """IDs whose user edits change skinning, including rigs with every chain suppressed."""
        found, stack = set(), list(self.meshes)
        while stack:
            obj = stack.pop()
            if obj is None or obj.session_uid in found:
                continue
            found.add(obj.session_uid)
            if obj.data is not None:
                found.add(obj.data.session_uid)
                keys = getattr(obj.data, "shape_keys", None)
                if keys is not None:
                    found.add(keys.session_uid)
            stack.append(obj.parent)
            for modifier in obj.modifiers:
                stack.append(getattr(modifier, "object", None))
                tree = getattr(modifier, "node_group", None)
                if tree is not None and tree.name != display.GROUP:
                    found.add(tree.session_uid)
            for owner in [obj] + (list(obj.pose.bones) if obj.pose is not None else []):
                for constraint in owner.constraints:
                    stack.append(getattr(constraint, "target", None))
                    stack.extend(entry.target for entry in getattr(constraint, "targets", ()))
            if obj.animation_data is not None:
                for curve in obj.animation_data.drivers:
                    for variable in curve.driver.variables:
                        stack.extend(target.id for target in variable.targets if isinstance(target.id, bpy.types.Object))
        return found

    def static_inputs(self):
        # Unlike topology/settings, moving the component invalidates a bake without
        # resetting live inertia. Animation updates are covered by action/driver edits.
        result = []
        for obj in self.meshes:
            transform = tuple(v for row in obj.matrix_basis for v in row) if obj.animation_data is None else None
            keys = obj.data.shape_keys
            values = tuple(block.value for block in keys.key_blocks) if keys is not None and keys.animation_data is None else None
            result.append((transform, values))
        return result

    def _read(self, scene):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        targets, matrices = [], []
        for obj, (rig, bone) in zip(self.meshes, self.references):
            points, matrix = display.read(obj, depsgraph)
            targets.append(_world(points, matrix, self.cm))
            if rig is not None:
                evaluated = rig.evaluated_get(depsgraph)
                reference = evaluated.matrix_world.copy()
                if bone is not None and bone in evaluated.pose.bones:
                    reference = reference @ evaluated.pose.bones[bone].matrix
                matrix = np.array(reference, dtype=np.float32)
            matrix = matrix.copy()
            # Component motion is rigid; object/bone scale already belongs to the skinned targets.
            rotation, _, transpose = np.linalg.svd(matrix[:3, :3])
            matrix[:3, :3] = rotation @ transpose
            matrix[:3, 3] *= self.cm
            matrices.append(matrix)
        return targets, matrices

    def _shapes(self, scene):
        result = []
        for obj in self.meshes:
            props = obj.waifu_cloth
            if props.use_all_colliders:
                candidates = [other for other in scene.objects if colliders.is_collider(other)
                              and other.waifu_physics_collider.enabled]
            else:
                sources = [entry.armature for entry in props.collider_sets if entry.armature is not None]
                if not props.custom_collider_sets and not sources:
                    rig = obj.find_armature()
                    sources = colliders.default_sources(rig) if rig is not None else []
                candidates = [other for rig in sources for other in colliders.colliders_of(rig)]
                if props.use_scene_colliders:
                    candidates += colliders.scene_colliders(scene)
            shapes = [colliders.shape_of(other, obj, self.cm, matrix=np.array(other.matrix_world))
                      for other in candidates]
            result.append([shape for shape in shapes if shape is not None])
        return result

    def reset(self, scene):
        self.targets, self.previous_matrices = self._read(scene)
        self.system.reset(self.targets)
        self.write()

    def step(self, scene, dt):
        targets, matrices = self._read(scene)
        motions = []
        teleport = False
        for obj, matrix, previous in zip(self.meshes, matrices, self.previous_matrices or matrices):
            move = matrix @ np.linalg.inv(previous)
            distance = obj.waifu_cloth.teleport_distance * self.cm
            teleport |= distance > 0 and np.linalg.norm(matrix[:3, 3] - previous[:3, 3]) > distance
            motions.append((move, previous[:3, 3].copy()))
        if teleport:
            self.system.reset(targets)
        else:
            self.system.step_frame(dt, targets=targets, colliders=self._shapes(scene), motions=motions,
                                   backend=self.backend, gravity=np.array(scene.gravity) * self.cm
                                   if scene.use_gravity else np.zeros(3))
        self.targets, self.previous_matrices = targets, matrices
        self.write()

    def write(self):
        for index, obj in enumerate(self.meshes):
            world = self.system.render_positions(index) / self.cm
            matrix = np.array(obj.matrix_world.inverted_safe(), dtype=np.float32)
            display.write(obj, world @ matrix[:3, :3].T + matrix[:3, 3])

    def release(self):
        for obj, (uid, _data_uid, _count) in zip(self.meshes, self.identities):
            try:
                if obj.session_uid == uid:
                    display.clear(obj)
            except ReferenceError:
                pass

    def snapshot(self):
        positions = [self.system.render_positions(i).copy() for i in range(len(self.meshes))]
        local = []
        for obj, world in zip(self.meshes, positions):
            matrix = np.array(obj.matrix_world.inverted_safe(), dtype=np.float32)
            local.append((world / self.cm) @ matrix[:3, :3].T + matrix[:3, 3])
        return {"identities": self.identities.copy(), "system": self.system.snapshot(),
                "matrices": None if self.previous_matrices is None else [m.copy() for m in self.previous_matrices],
                "positions": positions, "local_positions": local}

    def restore(self, state, write=False):
        if state is None or state["identities"] != self.identities or not self.alive():
            return
        if write:
            for obj, local in zip(self.meshes, state["local_positions"]):
                display.write(obj, local)
        else:
            self.system.restore(state["system"])
            self.previous_matrices = state["matrices"]


def between(a, b, fraction):
    if a is None or b is None or a["identities"] != b["identities"]:
        return b
    result = dict(b)
    result["positions"] = [start + (end - start) * fraction for start, end in zip(a["positions"], b["positions"])]
    result["local_positions"] = [start + (end - start) * fraction
                                 for start, end in zip(a["local_positions"], b["local_positions"])]
    return result
