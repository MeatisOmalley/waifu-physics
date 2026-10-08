# Waifu Cloth numerical reference

This is an independently authored behavioral port of Unreal Engine **5.8.0,
55116800**, `Chaos::Softs::FEvolution`'s ordinary PBD cloth path. It does not
contain copied Epic source and does not run Unreal inside Blender. The Kawaii
chain solver is separate and unchanged.

## Supported configuration

The solver uses float32 particles, world centimetres, seconds and kilograms.
`cloth.build.Settings` defaults are density **0.35 kg/m²**, minimum vertex mass
**0.0001 kg**, gravity scale **1**, global damping **0.01**, local damping **0**,
linear/angular velocity scales **0.75**, **1** iteration at 60 Hz (maximum **10**),
**1** substep, edge/bending/area/tether stiffness **1**, tether scale **1**,
animation drive stiffness/damping high values **1**, collision thickness
**1 cm**, friction **0.8** and RGB maximum distance scale **40 cm**.

Each frame uses `clamp(round(60 * frame_dt * iterations), 1, max_iterations)`.
Substeps interpolate animation and collider transforms. Cloth meshes in a
System share the greatest requested substep count, matching scene-wide solver
substepping. The usual default uses one iteration and one substep.

The in-scope constraints are ordinary distance springs on triangle edges,
opposite-vertex distance bending, axial area springs, geodesic long range
attachments to anchor islands, spherical maximum distance, animation drive,
and body collisions against spheres, capsules, tapered capsules, oriented
boxes and one-sided planes. Wind/aerodynamics, backstop, self collision, CCD,
XPBD, anisotropic material, convex and triangle-mesh colliders, LOD and Newton
solving are outside this preview.

## Source references

Paths below are relative to the installed `UE_5.8/Engine` directory. Line numbers
refer to the installed release source; these document behavior rather than
granting redistribution of engine source.

| Behavior | Source |
|---|---|
| Modern initial guess, rule order, final velocity | `Source/Runtime/Experimental/Chaos/Private/Chaos/SoftsEvolution.cpp:60–98,149–161,463–529,553–590` |
| Modern edge, bend, area, max distance, animation, collision registration | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothConstraints.cpp:1385–1518` |
| Tethers once per substep, after prediction | Same file, `1606–1618` |
| Exponential PBD stiffness | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDStiffness.h:23–26,115–151` |
| Sixteen-entry truncated weight lookup | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDWeightMap.h:130–144,168–181` |
| Edge input order and distance correction | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDSpringConstraintsBase.h:111–175` |
| Color construction and application order | `Source/Runtime/Experimental/Chaos/Private/Chaos/GraphColoring.cpp:285–347`; `PBDSpringConstraints.cpp:37–80,107–210` |
| Axial barycentric choice and inverse mass | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDAxialSpringConstraintsBase.h:112–145,179–231` |
| Axial extra correction multiplier | `Source/Runtime/Experimental/Chaos/Private/Chaos/PBDAxialSpringConstraints.cpp:79–98` |
| Animation position pull then relative displacement damping | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDAnimDriveConstraint.h:119–122,198–214` |
| Dynamic-only centre of mass, inertia, local damping | `Source/Runtime/Experimental/Chaos/Private/Chaos/PerParticleDampVelocity.cpp:119–162`; public header `40–44` |
| Triangle area mass, density cm conversion, minimum mass | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothingSimulationSolver.cpp:1094–1109,1358–1405` |
| Kinematic threshold strictly below 0.1 cm | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothingSimulationCloth.cpp:287–300` |
| Geodesic seeds, paths, one closest seed per island, four attachments | `Source/Runtime/ClothingSystemRuntimeCommon/Private/ClothTetherData.cpp:474–671` |
| Tether unilateral correction | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDLongRangeConstraintsBase.h:104–145` |
| Residual rigid motion and velocity rotation | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothingSimulationSolver.cpp:1201–1211,1312–1348,1763–1773` |
| Teleport versus reset | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothingSimulationCloth.cpp:1090–1118` |
| Centrifugal external acceleration | `Source/Runtime/Experimental/Chaos/Public/Chaos/SoftsExternalForces.h:59–64` |
| Projection and position based Coulomb friction | `Source/Runtime/Experimental/Chaos/Private/Chaos/PBDSoftBodyCollisionConstraint.cpp:57–69,258–451` |
| Primitive collision reevaluated each iteration by default | `Source/Runtime/Experimental/Chaos/Public/Chaos/PBDSoftBodyCollisionConstraint.h:44–45` |
| Tapered capsule radius sampled along closest axis point | `Source/Runtime/Experimental/Chaos/Public/Chaos/TaperedCapsule.h:113–122` |
| Default optimized tapered capsule geometry construction | `Plugins/ChaosCloth/Source/ChaosCloth/Private/ChaosCloth/ChaosClothingSimulationCollider.cpp:35,191–194` |

## Numerical rules

For damping `d`, frequency 60 Hz and substep `h`, the predictor uses
`q = exp(log(1-d)*60*h)` and integrated velocity interval
`H = (q-1)/(log(1-d)*60)`. The zero-damping limit is `H=h`; near full damping
the predictor interval is zero. External acceleration first changes velocity,
then local damping acts on that velocity, then `P=X+V*H`. The final velocity is
always `(P-X)/h`, rather than retaining the damped predicted velocity.

PBD stiffness `s` maps through `fit=(1000^s-1)/999`, then
`k=1-(1-fit)^(120*h/iterations)`, with exact 0/1 endpoints. Green weights use
`floor(clamp(green)*15)/15` before this fit, matching the engine's 16-entry
stiffness table. Animation drive first pulls predicted position toward its
animation target, then damps displacement relative to animation velocity.

Tethers apply once to the prediction. Each iteration applies edge springs,
bending springs, axial area springs, max distance, animation drive, then body
collision. There are no supported post-collision constraints; self collision
would occupy that phase in Unreal. Collisions can therefore temporarily put a
particle outside its max distance if a collider and animation bound conflict,
which is the engine's constraint order rather than an extra repair rule.

Spring input follows triangle insertion order, then a deterministic greedy
graph coloring ignores kinematic conflicts. Both native and reference consume
the same contiguous color batches. This matches Unreal's development build,
which colors even below its 100-constraint parallel threshold. Unreal shipping
builds may retain uncolored order for small meshes; PBD order affects results.

Character motion pretransforms particles and old targets by the residual
`1-velocity_scale` rigid motion, rotating velocity without adding translation
velocity. Teleport uses full rigid motion and disables fictitious acceleration;
reset instead replaces positions with skin targets and zeros velocity. Ordinary
rotation adds the default centrifugal acceleration from full reference angular
velocity. A reference origin can be supplied to rotate around a moving bone.

## Workshop mesh convention

These are Workshop authoring choices, distinct from Chaos solver mathematics:

- Red maps to maximum distance (40 cm by default); values giving distance below
  0.1 cm are anchors. Green is animation-drive weight. Blue >=0.5 is riding
  geometry and excluded from simulation.
- Non-riding rest vertices weld in rounded 0.001 cm (0.01 mm) cells. A seam uses
  maximum red and green, matching `WaifuSim/Private/Mods/WaifuModCloth.cpp:110–145`.
  The first vertex is the skin-target sample. This is cell equality rather than
  a Euclidean distance search, so nearby vertices across cell boundaries stay
  separate and a painted dynamic seam can override a coincident anchor.
- Participating sections retain attached anchors, exclude invalid/degenerate
  and duplicate faces, and leave disconnected fixed-only sections animated.
- Riding vertices select the nearest participating finite triangle, then bind
  with unbounded plane-projection barycentric coordinates plus signed normal
  offset. This preserves positions beyond triangle boundaries, matching the
  projection behavior of `ClothingMeshUtils.cpp:829–834`. If no triangle exists,
  they stay animated. Transfer uses triangle face normals; Unreal's complete
  mesh-to-mesh skin transfer additionally uses source vertex normals.
- Blue riding geometry follows the simulated surface directly. Detailed
  differential skin deformation beyond barycentric/normal transfer is not
  represented in this preview.

## Standalone API and native build

```python
from waifu_physics.cloth.build import build, Settings
from waifu_physics.cloth.system import System
from waifu_physics.cloth import native, step_numpy

mesh = build(world_cm_positions, triangle_indices, rgb_weights, Settings())
scene = System([mesh], gravity=(0, 0, -980.665))
scene.step_frame(1/60, targets=[skinned_world_cm], colliders=[shapes],
                 motions=[previous_to_current_world_matrix], backend=native.backend())
vertices = scene.render_positions(0)
state = scene.snapshot()
scene.restore(state)
```

Colliders may be a shared flat Shape list or one list per cloth. Shape fields
are `kind` (0 sphere, 2 capsule, 3 tapered, 4 box, 5 plane), `location`,
`rotation` quaternion xyzw, `radius` (+Z), `radius1` (-Z), cylinder `length`,
half `extent` and `enabled`. Optional `velocity` and `angular_velocity` override
the velocities inferred from consecutive samples. Plane local +Z is its
normal. Kawaii's inner sphere is outside the Chaos primitive set and ignored.

`step_frame` accepts `gravity=`, `teleport=True`, `reset=True`; motion entries
may also be `(matrix, old_reference_origin_cm)`. `System.interpolate(a,b,t)`
returns a detached state interpolating particle/target positions and velocity;
restore it for display, then restore a canonical endpoint before advancing.

`native.backend()` returns a version-checked C backend or the numpy reference;
`native.reason()` explains fallback. The C ABI checks contiguous array dtypes
before binding and keeps arrays alive. `tools/release.py --dll-only` compiles
both Kawaii and cloth using MSVC `/O2 /fp:precise`; Linux CI compiles both with
`-O2 -ffp-contract=off -fno-fast-math` and tests before uploading distinct
artifacts. No claim of cross-platform bit identity is made.

## Validation and known limits

`tests/test_cloth_build.py` checks topology, density, anchors, geodesics, welding,
ride bindings, invalid geometry, integrated damping, max distance, gravity and
animation-drive fit. `tests/test_cloth_agreement.py` checks primitive projection,
friction including sampled collider movement, local damping force order,
teleport, frame snapshots, and 120 frames comparing both implementations with
every supported constraint/collider and moving targets. Windows MSVC measurement
on that fixture was **zero position/velocity difference**; this is measured
agreement on that fixture, not proof of universal or cross-platform identity.

The independent Unreal trajectories and fixture provenance live in
`tests/golden/chaos_cloth_ue58.json` and `tests/test_cloth_golden.py`. Native/numpy
agreement and actual Unreal agreement are separate checks. Solver parity is
bounded numerical approximation: small ordering, rest-length, geodesic tie and
float/SIMD rounding differences can grow in dynamic cloth, especially flags.
Unreal engine golden fixtures currently cover simple uncollided skirt/flag
surfaces at default iteration/substep settings; collisions, green maps and
rotating-character behavior have analytical and C/reference checks, not a
captured Unreal golden trajectory yet.

`tests/bench_cloth.py` reports build time plus median/p95 frame time for 1,520
particles at defaults and four iterations/two substeps. A Windows run with a
sphere and a plane, 300 frames per measurement, gave:

| Settings | C median / p95 | NumPy median / p95 |
|---|---|---|
| 1 iteration, 1 substep (default) | 0.440 / 0.606 ms | 3.479 / 4.718 ms |
| 4 iterations, 2 substeps | 1.042 / 1.248 ms | 22.185 / 24.992 ms |

Topology build took 312.7 ms. Timing excludes Blender skin evaluation,
depsgraph refresh and mesh display; those costs must be measured separately for
the full preview. The numpy backend is a correctness fallback, and dense cloth
previews should use the compiled native step.
