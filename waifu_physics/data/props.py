"""What Waifu Physics stores in a .blend: groups on armature objects, and scene settings.

A group is one Kawaii Physics node. Its per-bone settings (damping, stiffness,
world damping, radius, limit angle) are animatable and are read every frame;
everything that shapes the chains -- roots, exclusions, dummies, subdivision,
links -- rebuilds the simulation when it changes. Lengths are in Blender units
in the armature's own space (Kawaii measures in its component's space).
"""
import math

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from ..solver.system import COMPLIANCE_TYPES

COMPLIANCE_ITEMS = [(name, name.title(), f"Stretches like {name.lower()}") for name in COMPLIANCE_TYPES]
LINK_COMPLIANCE_ITEMS = [("GROUP", "Group's", "Use the group's compliance")] + COMPLIANCE_ITEMS
PLANAR_ITEMS = [("NONE", "None", "No planar constraint"),
                ("X", "X", "Bones swing only in the parent's YZ plane"),
                ("Y", "Y", "Bones swing only in the parent's XZ plane"),
                ("Z", "Z", "Bones swing only in the parent's XY plane")]


def _colliders():
    from . import colliders
    return colliders


def _picked_index(settings):
    objects = bpy.data.objects
    found = _colliders().picked(settings.id_data, objects.get(settings.last_collider))
    return objects.find(found.name) if found is not None else -1


def _pick_index(settings, index):
    objects = bpy.data.objects
    if 0 <= index < len(objects) and _colliders().is_collider(objects[index]):
        settings.last_collider = objects[index].name
        _colliders().pick(settings.id_data, objects[index])


def _structure_changed(self, context):
    """Something that shapes the chains changed: rebuild the simulation."""
    from ..runtime import live
    live.mark_dirty()


def _result_changed(self, context):
    """A setting read every frame changed: cached frames are stale."""
    from ..runtime import live
    live.invalidate(reason="a setting changed")


_propagating = False


def _setting_changed(name):
    """With Edit Selected Groups on, a change to one group's setting reaches every group
    holding a selected bone (Waifu Physics' version of Alt-editing). Any change drops cached frames."""
    def update(self, context):
        global _propagating
        from ..runtime import live
        live.invalidate(reason="a setting changed")
        if _propagating or context is None:
            return
        from ..ui.selection import groups_of_selected
        value = getattr(self, name)
        _propagating = True
        try:
            for group in groups_of_selected(context):
                if group != self and getattr(group, name) != value:
                    setattr(group, name, value)
        finally:
            _propagating = False
    return update


def _curve_toggled(name):
    def update(self, context):
        from . import curves
        if getattr(self, f"use_{name}_curve"):
            curves.node(self, name)
        if name == "radius":                # bridge and densified points are sized from it when built
            _structure_changed(self, context)
    return update


def _bone_named(then=None):
    """The update of every property that names a bone: remember where the bone is (bone_refs), so a later
    rename is followed whatever set the name (an operator, an import, a script); then the property's own."""
    def update(self, context):
        from . import bone_refs
        obj = self.id_data
        if obj is not None and obj.type == "ARMATURE" and obj.data is not None:
            bone_refs.remember(obj)
        if then is not None:
            then(self, context)
    return update


class WaifuPhysicsBoneName(PropertyGroup):
    name: StringProperty(update=_bone_named())


class WaifuPhysicsLink(PropertyGroup):
    """A distance constraint between two bones of the group (Kawaii's bone constraint)."""
    bone_a: StringProperty(name="Bone A", update=_bone_named(_structure_changed))
    bone_b: StringProperty(name="Bone B", update=_bone_named(_structure_changed))
    compliance: EnumProperty(name="Compliance", items=LINK_COMPLIANCE_ITEMS, default="GROUP",
                             update=_structure_changed)
    exclude_from_subdivision: BoolProperty(name="No Bridge Points", update=_structure_changed,
                                           description="Add no bridge points along this link")


class WaifuPhysicsColliderSet(PropertyGroup):
    """An armature whose colliders this group collides with."""
    armature: PointerProperty(type=bpy.types.Object, update=_structure_changed,
                              poll=lambda _self, obj: obj.type == "ARMATURE")


def _force_kind_changed(self, context):
    from . import curves
    if self.kind == "CURVE":
        for channel in FORCE_CHANNELS:
            curves.node(self, channel)
    _result_changed(self, context)


def _item_curve_toggled(name):
    def update(self, context):
        from . import curves
        if getattr(self, f"use_{name}_curve"):
            curves.node(self, name)
        _result_changed(self, context)
    return update


FORCE_CHANNELS = ("force_x", "force_y", "force_z")
FORCE_KINDS = [
    ("BASIC", "Push", "A steady push, or a pulse at intervals", "FORCE_FORCE", 0),
    ("GRAVITY", "Gravity", "Extra gravity, in world space",
     "FORCE_HARMONIC", 1),
    ("CURVE", "Curve", "A push that follows curves over time", "FCURVE", 2),
    # Kawaii's Wind external force read the scene's Wind fields; Blender Force Fields does that job now, and
    # files are converted on load. Kept so old files' values still read.
    ("WIND", "Field Wind", "Kawaii's wind from Wind force fields (replaced by Blender Force Fields)", "FORCE_WIND", 3),
    ("PROCEDURAL_WIND", "Procedural Wind", "Generated sway, ripples and gusts",
     "FORCE_WIND", 4)]
# What a force is, as its header shows it. Wind is Procedural Wind: Kawaii's field-reading Wind force became
# Blender Force Fields.
FORCE_CATEGORIES = [("PUSH", "Push", "A steady push, or a pulse every interval", "FORCE_FORCE", 0),
                    ("GRAVITY", "Gravity", "Extra gravity, in world space", "FORCE_HARMONIC", 1),
                    ("CURVE", "Curve", "A push that follows curves over time", "FCURVE", 2),
                    ("WIND", "Wind", "Generated sway, ripples and gusts. No force field needed", "FORCE_WIND", 3)]
_CATEGORY_OF_KIND = {"BASIC": 0, "GRAVITY": 1, "CURVE": 2, "WIND": 3, "PROCEDURAL_WIND": 3}
_KIND_OF_CATEGORY = ("BASIC", "GRAVITY", "CURVE", "PROCEDURAL_WIND")
FORCE_NAMES = {"BASIC": "Push", "GRAVITY": "Gravity", "CURVE": "Curve", "WIND": "Wind", "PROCEDURAL_WIND": "Wind"}


def _category_get(self):
    return _CATEGORY_OF_KIND[self.kind]


def _category_set(self, value):
    kind = _KIND_OF_CATEGORY[value]
    if kind != self.kind:
        if self.name == FORCE_NAMES.get(self.kind):        # a default name follows the type
            self.name = FORCE_NAMES[kind]
        self.kind = kind


FORCE_SPACES = [("COMPONENT", "Armature", "In the armature's space"),
                ("WORLD", "World", "In world space"),
                ("BONE", "Bone", "In each bone's own space, turning with it")]
CURVE_EVALUATE = [("SINGLE", "Single", "The curve's value at the current time"),
                  ("AVERAGE", "Average", "The average over the frame"),
                  ("MAX", "Max", "The largest value over the frame"),
                  ("MIN", "Min", "The smallest value over the frame")]
SYNC_DIRECTIONS = [("BOTH", "Both", "Follow movement either way along this axis"),
                   ("POSITIVE", "Positive", "Follow only movement toward +axis"),
                   ("NEGATIVE", "Negative", "Follow only movement toward -axis"),
                   ("NONE", "None", "Ignore movement along this axis")]


class WaifuPhysicsForce(PropertyGroup):
    """One Kawaii external force. Velocities are in Blender units a second; everything is animatable."""
    name: StringProperty(name="Name", default="Force")
    enabled: BoolProperty(name="Enabled", default=True, update=_result_changed)
    kind: EnumProperty(name="Type", items=FORCE_KINDS, default="BASIC", update=_force_kind_changed)
    category: EnumProperty(name="Type", items=FORCE_CATEGORIES, get=_category_get, set=_category_set,
                           options=set(), description="The kind of force")
    space: EnumProperty(name="Space", items=FORCE_SPACES, default="WORLD", update=_result_changed)
    apply_bones: CollectionProperty(type=WaifuPhysicsBoneName)
    ignore_bones: CollectionProperty(type=WaifuPhysicsBoneName)
    random_min: FloatProperty(name="Scale Min", default=1.0, update=_result_changed,
                              description="Scales the force by a random amount each frame, within this range")
    random_max: FloatProperty(name="Scale Max", default=1.0, update=_result_changed)
    curve_key: StringProperty(options={"HIDDEN"})
    use_rate_curve: BoolProperty(name="Force Curve", update=_item_curve_toggled("rate"),
                                 description="Scale the force along the chain, root to tip")

    direction: FloatVectorProperty(name="Direction", default=(0.0, 0.0, 0.0), size=3, update=_result_changed,
                                   description="The push, in units per second. For Gravity and Wind, its direction")
    interval: FloatProperty(name="Interval", default=0.0, min=0.0, subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE",
                            update=_result_changed, description="Seconds between pushes. 0 pushes constantly")
    override_direction: BoolProperty(name="Override Direction", update=_result_changed,
                                     description="Pull along Direction instead of straight down")
    duration: FloatProperty(name="Duration", default=1.0, min=0.001, subtype="TIME_ABSOLUTE",
                            unit="TIME_ABSOLUTE", update=_result_changed,
                            description="How long the curves run before repeating")
    amplitude: FloatVectorProperty(name="Amplitude", default=(1.0, 1.0, 1.0), size=3, update=_result_changed,
                                   description="The push at a curve value of 1, in units per second")
    time_scale: FloatProperty(name="Time Scale", default=1.0, update=_result_changed)
    evaluate: EnumProperty(name="Evaluate", items=CURVE_EVALUATE, default="SINGLE", update=_result_changed)
    substeps: IntProperty(name="Substeps", default=10, min=1, max=100, update=_result_changed)
    noise_angle: FloatProperty(name="Direction Noise", default=0.0, min=0.0, max=math.pi, subtype="ANGLE",
                               update=_result_changed, description="Randomly turns the wind's direction")
    noise_period: FloatProperty(name="Noise Period", subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE", default=1.0, min=0.01, update=_result_changed)
    constant: FloatProperty(name="Constant", unit="VELOCITY", default=0.0, update=_result_changed,
                            description="Steady wind strength")
    sway: FloatProperty(name="Sway", unit="VELOCITY", default=0.0, update=_result_changed,
                        description="Back-and-forth wind strength")
    sway_period: FloatProperty(name="Sway Period", subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE", default=1.0, min=0.01, update=_result_changed)
    sway_phase: FloatProperty(name="Sway Phase", default=0.0, subtype="ANGLE", update=_result_changed)
    ripple: FloatProperty(name="Ripple", unit="VELOCITY", default=0.0, update=_result_changed,
                          description="A wave running from root to tip")
    ripple_period: FloatProperty(name="Ripple Period", subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE", default=1.0, min=0.01, update=_result_changed)
    ripple_phase: FloatProperty(name="Ripple Phase", default=0.0, subtype="ANGLE", update=_result_changed)
    ripple_delay: FloatProperty(name="Ripple Tip Delay", default=math.pi, subtype="ANGLE", update=_result_changed,
                                description="How far the wave at the tip lags behind the root")
    cycle_min: FloatProperty(name="Gust Low", default=1.0, update=_result_changed,
                             description="Wind strength at the calmest point of a gust")
    cycle_max: FloatProperty(name="Gust High", default=1.0, update=_result_changed,
                             description="Wind strength at the strongest point of a gust")
    cycle_period: FloatProperty(name="Strength Period", subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE", default=10.0, min=0.01, update=_result_changed)
    cycle_phase: FloatProperty(name="Strength Phase", default=0.0, subtype="ANGLE", update=_result_changed)
    random: FloatProperty(name="Random", unit="VELOCITY", default=0.0, update=_result_changed,
                          description="Random turbulence strength")
    random_period: FloatProperty(name="Random Period", subtype="TIME_ABSOLUTE", unit="TIME_ABSOLUTE", default=0.5, min=0.01, update=_result_changed)
    seed: IntProperty(name="Seed", default=0, update=_result_changed)
    show_advanced: BoolProperty(name="Advanced", default=False)


class WaifuPhysicsSyncTarget(PropertyGroup):
    bone: StringProperty(name="Bone", update=_bone_named(_result_changed))
    include_children: BoolProperty(name="Children", default=True, update=_result_changed,
                                   description="Also move the bones below it")
    curve_key: StringProperty(options={"HIDDEN"})
    use_rate_curve: BoolProperty(name="Rate Curve", update=_item_curve_toggled("rate"),
                                 description="Scale the movement from this bone to the chain's tip")


class WaifuPhysicsSyncBone(PropertyGroup):
    """Kawaii's SyncBone: a bone outside the chains (a thigh) whose movement carries chain bones' poses."""
    name: StringProperty(name="Name", default="Sync")
    bone: StringProperty(name="Source Bone", update=_bone_named(_result_changed),
                         description="The bone whose movement the targets follow")
    targets: CollectionProperty(type=WaifuPhysicsSyncTarget)
    active_target: IntProperty()
    global_scale: FloatVectorProperty(name="Scale", default=(1.0, 1.0, 1.0), size=3, update=_result_changed)
    curve_key: StringProperty(options={"HIDDEN"})
    use_distance_curve: BoolProperty(name="Distance Curve", update=_item_curve_toggled("distance"),
                                     description="Scale the effect by how far the source has moved")
    distance: FloatProperty(name="Distance", default=0.3, min=0.001, subtype="DISTANCE", update=_result_changed,
                            description="The movement the right end of the curve stands for")
    direction_x: EnumProperty(name="X", items=SYNC_DIRECTIONS, default="BOTH", update=_result_changed)
    direction_y: EnumProperty(name="Y", items=SYNC_DIRECTIONS, default="BOTH", update=_result_changed)
    direction_z: EnumProperty(name="Z", items=SYNC_DIRECTIONS, default="BOTH", update=_result_changed)
    attenuation: BoolProperty(name="Distance Attenuation", update=_result_changed,
                              description="Weaken the effect on bones far from the source")
    inner_radius: FloatProperty(name="Inner Radius", default=0.0, min=0.0, subtype="DISTANCE",
                                update=_result_changed)
    outer_radius: FloatProperty(name="Outer Radius", default=0.0, min=0.0, subtype="DISTANCE",
                                update=_result_changed)
    max_attenuation: FloatProperty(name="Max Attenuation", default=1.0, min=0.0, update=_result_changed)


SETTLE_OFF = 1000.0                  # settle time shown for no stiffness: the chain never returns


def _step_rate():
    scene = getattr(bpy.context, "scene", None)
    return scene.waifu_physics.target_framerate if scene is not None else 60


def _settle_get(self):
    """Kawaii's stiffness as the seconds to get 95% of the way back to the pose: stiffness is the
    fraction of the gap closed each step (ApplyStiffnessPull), at the simulation rate."""
    s, rate = self.stiffness, _step_rate()
    if s <= 0.0:
        return SETTLE_OFF
    if s >= 1.0:
        return 1.0 / rate
    return min(SETTLE_OFF, math.log(0.05) / (rate * math.log(1.0 - s)))


def _settle_set(self, value):
    if value >= SETTLE_OFF:
        self.stiffness = 0.0
        return
    self.stiffness = 1.0 - 0.05 ** (1.0 / (_step_rate() * max(value, 1.0e-4)))


STIFFNESS_SPAN = 10.0      # seconds: the loosest a chain can be set, taking this long to settle


def _level_get(self):
    """Stiffness on a 0-10 scale: 10 minus the seconds to settle 95% of the way back to the pose."""
    return max(0.0, STIFFNESS_SPAN - min(_settle_get(self), STIFFNESS_SPAN))


def _level_set(self, value):
    _settle_set(self, STIFFNESS_SPAN - min(max(value, 0.0), STIFFNESS_SPAN))


# Damping on a 0-10 scale. How long a swing lasts goes as about 1 / damping, and durations are judged by
# ratio, so each step of the scale multiplies Kawaii's damping by the same factor. 0 is the floor: measured on
# VRoid hair at the loosest stiffness, less than this never settles (a frictionless tangle of pendulums);
# 0.03 settles in about ten seconds, the loosest the stiffness scale goes. 5 is Kawaii's default, 0.1.
DAMPING_FLOOR = 0.03
DAMPING_CEILING = 0.1 * 0.1 / DAMPING_FLOOR      # 0.33: the bounce is gone in about a tenth of a second
DAMPING_SPAN = 10.0


def _damping_level_get(self):
    d = self.damping
    if d <= DAMPING_FLOOR:
        return 0.0
    if d >= DAMPING_CEILING:
        return DAMPING_SPAN
    return DAMPING_SPAN * math.log(d / DAMPING_FLOOR) / math.log(DAMPING_CEILING / DAMPING_FLOOR)


def _damping_level_set(self, value):
    fraction = min(max(value, 0.0), DAMPING_SPAN) / DAMPING_SPAN
    self.damping = DAMPING_FLOOR * (DAMPING_CEILING / DAMPING_FLOOR) ** fraction


# Inertia on a 0-10 scale, like stiffness and damping: 10 times 1 - Kawaii's world damping, the share of the
# armature's movement the chains lag behind. The lag grows in proportion to it, so the scale is linear.
# Kawaii's default world damping, 0.8, is 2.
INERTIA_SPAN = 10.0


def _inertia_get(name):
    return lambda self: INERTIA_SPAN * (1.0 - getattr(self, name))


def _inertia_set(name):
    def setter(self, value):
        setattr(self, name, 1.0 - min(max(value, 0.0), INERTIA_SPAN) / INERTIA_SPAN)
    return setter


class WaifuPhysicsGroup(PropertyGroup):
    """One Kawaii Physics node: the chains under its root bones, and how they move."""
    name: StringProperty(name="Name", default="Group")
    enabled: BoolProperty(name="Enabled", default=True, update=_structure_changed)
    roots: CollectionProperty(type=WaifuPhysicsBoneName)
    excluded: CollectionProperty(type=WaifuPhysicsBoneName)
    links: CollectionProperty(type=WaifuPhysicsLink)
    collider_sets: CollectionProperty(type=WaifuPhysicsColliderSet)
    custom_collider_sets: BoolProperty(options={"HIDDEN"}, update=_structure_changed,
                                       description="The collider sets were edited: the list stands, even empty")
    use_all_colliders: BoolProperty(
        name="Collide Against Every Collider", default=True, update=_structure_changed,
        description="Collide with every collider in the scene. Turn off to choose armatures")
    use_scene_colliders: BoolProperty(
        name="Scene Colliders", default=True, update=_structure_changed,
        description="Collide with scene colliders, like a ground")
    active_link: IntProperty(default=-1)    # the link picked in the Links list, drawn red; -1 is none

    # FKawaiiPhysicsSettings, animatable. Radius is a length; limit angle an angle.
    damping: FloatProperty(name="Damping", default=0.1, min=0.0, max=1.0, update=_setting_changed("damping"),
                           description="How much velocity each point loses per step")
    stiffness: FloatProperty(name="Stiffness", default=0.05, min=0.0, max=1.0, update=_setting_changed("stiffness"),
                             description="How strongly points are pulled back to the animated pose")
    damping_level: FloatProperty(
        name="Damping", get=_damping_level_get, set=_damping_level_set, min=0.0, max=DAMPING_SPAN, precision=2,
        step=10, options=set(),
        description="How quickly swinging dies down. 0 swings longest, 10 stops almost at once")
    preset_name: StringProperty(options=set(), description="The preset last applied to the group")
    use_force_fields: BoolProperty(
        name="Blender Force Fields", default=False, update=_result_changed,
        description="React to the scene's force fields. Blender only: not exported")
    force_field_collection: PointerProperty(
        type=bpy.types.Collection, name="Collection", update=_result_changed,
        description="Only use force fields in this collection. Empty uses all of them")
    force_field_strength: FloatProperty(
        name="Strength", default=1.0, soft_min=0.0, soft_max=2.0, update=_result_changed,
        description="Scale the effect of all force fields on this group")
    stiffness_level: FloatProperty(
        name="Stiffness", get=_level_get, set=_level_set, min=0.0, max=STIFFNESS_SPAN, precision=2, step=10,
        options=set(),
        description="How quickly chains return to their pose. 10 snaps back, 0 takes ~10 s")
    world_location_inertia: FloatProperty(
        name="Moving Inertia", get=_inertia_get("world_damping_location"),
        set=_inertia_set("world_damping_location"), min=0.0, max=INERTIA_SPAN, precision=2, step=10,
        options=set(),
        description="How much chains lag and swing when the armature moves. 0 moves rigidly with it, 10 lags fully")
    world_rotation_inertia: FloatProperty(
        name="Rotating Inertia", get=_inertia_get("world_damping_rotation"),
        set=_inertia_set("world_damping_rotation"), min=0.0, max=INERTIA_SPAN, precision=2, step=10,
        options=set(),
        description="How much chains lag and swing when the armature turns. 0 turns rigidly with it, 10 lags fully")
    world_damping_location: FloatProperty(
        name="World Damping Location", default=0.8, min=0.0, max=1.0,
        update=_setting_changed("world_damping_location"),
        description="How little chains react to the armature moving")
    world_damping_rotation: FloatProperty(
        name="World Damping Rotation", default=0.8, min=0.0, max=1.0,
        update=_setting_changed("world_damping_rotation"),
        description="How little chains react to the armature turning")
    radius: FloatProperty(name="Chain Collision Radius", default=0.03, min=0.0, subtype="DISTANCE", precision=4,
                          update=_setting_changed("radius"),
                          description="Size of the collision sphere around each chain point")
    limit_angle: FloatProperty(name="Joint Limit", default=0.0, min=0.0, max=math.pi, subtype="ANGLE",
                               update=_setting_changed("limit_angle"),
                               description="How far bones may bend from their animated direction. 0 is no limit")
    # Curves along the chain, root to tip, multiplying each setting (Kawaii's *CurveData).
    curve_key: StringProperty(options={"HIDDEN"})
    use_damping_curve: BoolProperty(name="Damping Curve", update=_curve_toggled("damping"))
    use_stiffness_curve: BoolProperty(name="Stiffness Curve", update=_curve_toggled("stiffness"))
    use_world_damping_location_curve: BoolProperty(name="World Damping Location Curve",
                                                   update=_curve_toggled("world_damping_location"))
    use_world_damping_rotation_curve: BoolProperty(name="World Damping Rotation Curve",
                                                   update=_curve_toggled("world_damping_rotation"))
    use_radius_curve: BoolProperty(name="Radius Curve", update=_curve_toggled("radius"))
    use_limit_angle_curve: BoolProperty(name="Limit Angle Curve", update=_curve_toggled("limit_angle"))

    gravity: FloatVectorProperty(name="Gravity", default=(0.0, 0.0, -9.81), subtype="ACCELERATION", size=3,
                                 update=_result_changed,
                                 description="The group's own gravity")
    gravity_scale: FloatProperty(name="Gravity Scale", default=1.0, soft_min=0.0, soft_max=2.0,
                                 update=_result_changed,
                                 description="Multiplier on the scene's gravity. 1 is normal, 0 is none")
    use_scene_gravity: BoolProperty(name="Use Scene Gravity", default=True, update=_result_changed,
                                    description="Use the scene's gravity. Off uses the group's own")
    use_world_space_gravity: BoolProperty(name="World Space Gravity", default=True, update=_result_changed,
                                          description="Gravity points in world space, not the armature's")
    legacy_gravity: BoolProperty(name="Legacy Gravity", default=False, update=_structure_changed,
                                 description="Use the older gravity, added to position instead of velocity")

    dummy_bone_length: FloatProperty(name="Tip Length", default=0.0, min=0.0, subtype="DISTANCE",
                                     update=_structure_changed,
                                     description="Add a point past each chain's last bone, so that bone swings too")
    bone_subdivision_count: IntProperty(name="Subdivisions", default=0, min=0, max=10, update=_structure_changed,
                                        description="Extra collision points along each bone")
    bone_subdivision_collision_only: BoolProperty(name="Collision Only", default=True, update=_structure_changed,
                                                  description="Extra points collide but aren't simulated")
    bone_subdivision_densify_by_radius: BoolProperty(name="Densify by Radius", default=False,
                                                     update=_structure_changed,
                                                     description="Add points until they cover each bone at the collision radius")
    planar_constraint: EnumProperty(name="Planar Constraint", items=PLANAR_ITEMS, default="NONE",
                                    update=_structure_changed)
    compliance: EnumProperty(name="Link Compliance", items=COMPLIANCE_ITEMS, default="LEATHER",
                             update=_structure_changed,
                             description="How much links can stretch")
    iterations_before_collision: IntProperty(
        name="Iterations Before Collision", default=1, min=0, max=20, update=_structure_changed,
        description="Link passes before collision, settling the chains into shape. More holds spacing firmer")
    iterations_after_collision: IntProperty(
        name="Iterations After Collision", default=1, min=0, max=20, update=_structure_changed,
        description="Link passes after collision, spreading a collider's push to neighbours. More can clip")
    auto_child_dummy_links: BoolProperty(name="Link Tips", default=True, update=_structure_changed,
                                         description="Also link the tip and extra points")
    bridge_count: IntProperty(name="Bridge Points", default=0, min=0, max=10, update=_structure_changed,
                              description="Collision points along links, so colliders can't slip between chains")
    bridge_feedback: FloatProperty(name="Bridge Feedback", default=1.0, min=0.0, max=2.0,
                                   update=_structure_changed,
                                   description="How strongly bridge points push the bones they join")

    teleport_distance: FloatProperty(name="Teleport Distance", default=3.0, min=0.0, subtype="DISTANCE",
                                     update=_result_changed,
                                     description="Jumps farther than this in one frame are treated as teleports")
    teleport_rotation: FloatProperty(name="Teleport Rotation", default=math.radians(10.0), min=0.0,
                                     update=_result_changed,
                                     subtype="ANGLE",
                                     description="Turns larger than this in one frame are treated as teleports")
    warm_up_frames: IntProperty(name="Warm Up Frames", default=0, min=0, max=500, update=_result_changed,
                                description="Frames simulated before the start, so chains begin settled")

    show_chains: BoolProperty(name="Show Chains", default=True, description="Show this group's chains")
    list_links: BoolProperty(name="List Links", default=False, description="List this group's links in the chain manager")
    forces: CollectionProperty(type=WaifuPhysicsForce)
    active_force: IntProperty()
    sync_bones: CollectionProperty(type=WaifuPhysicsSyncBone)
    active_sync: IntProperty()
    simple_external_force: FloatVectorProperty(
        name="Simple Force", default=(0.0, 0.0, 0.0), size=3, subtype="VELOCITY", update=_result_changed,
        description="A constant push on every bone, in units per second")
    world_space_simple_external_force: BoolProperty(name="World Space", default=True, update=_result_changed)
    enable_wind: BoolProperty(name="Scene Wind", default=False, update=_result_changed,
                              description="Blow with the scene's Wind force fields, gusting at random")
    wind_scale: FloatProperty(name="Wind Scale", default=1.0, update=_result_changed)
    wind_direction_noise_angle: FloatProperty(name="Wind Direction Noise", default=0.0, min=0.0, max=math.pi,
                                              subtype="ANGLE", update=_result_changed)


class WaifuPhysicsCollider(PropertyGroup):
    """Marks a mesh object as a Waifu Physics collider (its shape lives on its Waifu Physics Collider modifier)."""
    is_collider: BoolProperty(options={"HIDDEN"})
    enabled: BoolProperty(name="Enabled", default=True, description="Use this collider")


def _group_picked(self, context):
    """Picking a group in another armature's list makes that armature the one the panels edit."""
    obj = self.id_data
    view_layer = getattr(context, "view_layer", None)
    if view_layer is not None and view_layer.objects.active != obj and obj.name in view_layer.objects:
        from ..ui.ops import make_active
        try:
            make_active(context, obj)
        except RuntimeError:                 # no mode switch possible from here: just make it active
            view_layer.objects.active = obj


class WaifuPhysicsKnownBone(PropertyGroup):
    """Where a bone the groups name sits at rest, so a renamed bone can be found again (data/bone_refs.py)."""
    name: StringProperty()
    parent: StringProperty()
    rest: FloatVectorProperty(size=6)          # head and tail, in armature space


class WaifuPhysicsArmature(PropertyGroup):
    groups: CollectionProperty(type=WaifuPhysicsGroup)
    known_bones: CollectionProperty(type=WaifuPhysicsKnownBone, options={"HIDDEN"})
    active_group: IntProperty(update=_group_picked)
    expanded: BoolProperty(name="Expanded", default=True, description="Show this armature's groups")
    colliders_expanded: BoolProperty(name="Expanded", default=True, description="Show this armature's colliders")


class WaifuPhysicsScene(PropertyGroup):
    simulate: BoolProperty(name="Simulate", default=False, update=lambda self, context: _simulate_changed(self),
                           description="Simulate the chains while the timeline plays")
    # The running simulation's clock follows these on its next frame, without a rebuild (Runtime.sync_clock).
    target_framerate: IntProperty(name="Steps per Second", default=60, min=1, max=480,
                                  description="Simulation steps per second. Changes how the settings feel")
    fixed_substepping: BoolProperty(
        name="Fixed Steps", default=True,
        description="Step at a fixed rate in live playback. Off steps once per frame")
    fast_evaluation: BoolProperty(
        name="Fast Evaluation (Live Preview)", default=False,
        description=("Live playback evaluates each frame once, reacting to the body a frame late. Off reacts on "
                     "time but evaluates each frame twice. The cache is always perfectly accurate"))
    edit_selected_groups: BoolProperty(
        name="Edit Selected Groups", default=True,
        description="Changes apply to every group with a selected bone")
    use_cache: BoolProperty(name="Cache", default=False, update=_result_changed,
                            description="Store simulated frames for scrubbing and rendering")
    show_links: BoolProperty(name="Show Links", default=True, description="Show links in the viewport")
    show_colliders: BoolProperty(
        name="Show Colliders", default=True, update=lambda self, context: _colliders().apply_shown(self.id_data),
        description="Show colliders and the chains' collision spheres in the viewport. Hidden, they still collide")
    # The Colliders list's pick is the viewport's selection (colliders.picked / pick); the list's index is
    # into bpy.data.objects.
    active_collider: IntProperty(name="Active Collider", options={"HIDDEN"},
                                 get=lambda self: _picked_index(self), set=lambda self, value: _pick_index(self, value))
    last_collider: StringProperty(options={"HIDDEN"})     # the name of the one last picked in the list
    collider_shape: EnumProperty(
        name="Shape", items=_colliders().SHAPE_CHOICES, default="AUTO",
        description="Shape for new bone colliders. Auto picks the best fit per bone")
    selected_only: BoolProperty(
        name="Selected Only", default=False,
        description="Only list the selected armatures")


def _simulate_changed(settings):
    from ..runtime import live
    live.set_simulating(bpy.context.scene, settings.simulate)


CLASSES = (WaifuPhysicsBoneName, WaifuPhysicsLink, WaifuPhysicsColliderSet, WaifuPhysicsForce, WaifuPhysicsSyncTarget, WaifuPhysicsSyncBone, WaifuPhysicsGroup,
           WaifuPhysicsCollider, WaifuPhysicsKnownBone, WaifuPhysicsArmature, WaifuPhysicsScene)
SETTING_NAMES = ("damping", "stiffness", "world_damping_location", "world_damping_rotation", "radius", "limit_angle")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.waifu_physics = PointerProperty(type=WaifuPhysicsArmature)
    bpy.types.Object.waifu_physics_collider = PointerProperty(type=WaifuPhysicsCollider)
    bpy.types.Scene.waifu_physics = PointerProperty(type=WaifuPhysicsScene)


def unregister():
    del bpy.types.Scene.waifu_physics
    del bpy.types.Object.waifu_physics_collider
    del bpy.types.Object.waifu_physics
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
