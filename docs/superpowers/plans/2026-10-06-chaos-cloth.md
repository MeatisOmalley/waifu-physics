# Waifu Cloth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each phase is expanded into step-level tasks when it starts; this document fixes the design and the order.

**Goal:** Preview in Blender, live, what a garment's cloth will do in a game running Chaos Cloth: a second solver in Waifu Physics, a faithful port of the Chaos Cloth simulation the game runs, driven by the same per-vertex paint the game reads, colliding with the same colliders.

**Architecture:** A port of the step Unreal's skeletal-mesh clothing runs (the legacy `FPBDEvolution` with PBD constraints, Chaos's defaults), as a pure function over flat arrays holding every cloth particle in the scene. It exists twice, in C (the same ctypes DLL pipeline as the Kawaii step) and in numpy (every platform, and the reference); constraints are coloured into batches that share no particle, and both backends walk the same batches in the same order, so they agree exactly. Blender I/O is one bulk read of skinned positions and one bulk write of simulated positions per mesh per frame, through a Geometry Nodes modifier, so nothing destructive touches the mesh. The Kawaii solver is untouched: bone chains keep Kawaii parity, cloth gets Chaos parity. Waifu Workshop's Cloth panel becomes a thin layer over this.

**Tech Stack:** Blender 5.2 (Python, numpy 2.3, bpy, Geometry Nodes), C via ctypes (MSVC 2022 Build Tools; Linux through the existing workflow), Unreal Engine 5.8's Chaos Cloth source as the behavioural specification, WaifuSim for golden data.

**Reference:** Chaos Cloth as shipped in Unreal Engine 5.8 (`C:\Program Files\Epic Games\UE_5.8`), `Engine/Plugins/ChaosCloth` and `Engine/Source/Runtime/Experimental/Chaos`. Behaviour is ported from that version; a later engine's change is a deliberate re-port. The engine source is Epic's licence, not open source: port behaviour (orders, formulas, defaults) and cite files, do not copy code verbatim. The Kawaii step's MIT notice is unaffected.

---

## Established by measurement

Recorded 2026-10-06 (WaifuSim branch history, `docs/superpowers/plans/2026-09-20-vrm-upload-pipeline.md` in WaifuSim, and the engine source) so implementation does not re-derive them.

| Finding | Evidence |
| --- | --- |
| The game's mod garments simulate on Unreal's skeletal-mesh clothing | WaifuSim `Mods/WaifuModCloth.cpp` builds a `UClothingAssetCommon` at runtime; a cooked Development build draws it simulating (`waifu.ClothSpike`). The Chaos Cloth Asset route is editor-only for render data |
| That path runs the **legacy** evolution | `bClothUseLegacyEvolution = true` (`ChaosClothingSimulation.cpp:69`, cvar `p.ChaosCloth.UseLegacyEvolution`) is passed to the solver (`:313`); `IsLegacySolver()` is `!!PBDEvolution` (`ChaosClothingSimulationSolver.h:65`). So the reference step is `FPBDEvolution::AdvanceOneTimeStep` (`Chaos/Private/Chaos/PBDEvolution.cpp`), not `Softs::FEvolution` |
| With defaults, one constraint pass per 60 Hz frame | Shared config: `IterationCount = 1`, `MaxIterationCount = 10`, `SubdivisionCount = 1`, `bUseXPBDConstraints = false` (`ChaosClothConfig.h:573-627`). Iterations per step = `round(60 * dt * IterationCount)` clamped to [1, 10] (`ChaosClothingSimulationSolver.cpp:2242-2255`); substeps divide dt |
| Legacy step order | `PBDEvolution.cpp`: per particle range, the pre-iteration update (forces, velocity fields, damping, predicted positions), the collision kinematic update, constraint inits, collision rule inits, then each iteration: constraint rules, collision, post-collision rules; then velocities from positions. Constraint rules register in the order `FClothConstraints` creates them (`ChaosClothConstraints.cpp`: edge springs, bending, area, long-range tethers, max distance, backstop, anim drive; read the exact order from the `Create*` calls) |
| Defaults the game uses (`UChaosClothConfig`, `ChaosClothConfig.h`) | Mass mode Density 0.35 (`:99`, `:120`), min particle mass 1e-4; edge, bending, area stiffness 1 (`:133`, `:142`, `:183`); tether stiffness 1, scale 1, geodesic tethers on (`:201-219`); collision thickness 1 cm, friction 0.8 (`:227-231`); damping 0.01 (`:289`); drag/lift 0.035 (`:318`, `:344`); gravity scale 1; anim drive {0, 1} weighted (`:390`); linear velocity scale 0.75, angular 0.75 (`:413`, `:446`) |
| How the paint reaches Chaos | MaxDistance is a weighted property {0, 1} whose map holds centimetres (`ChaosClothingSimulationConfig.cpp:239-241`): the card's red × max distance. A particle is kinematic when its max distance is under 0.1 cm (`ChaosClothingSimulationCloth.cpp:290-300`). Green is the AnimDriveStiffness map, scaled by {0, 1}. Blue (ride) vertices are not particles: they are bound to the sim mesh by `ClothingMeshUtils::GenerateMeshToMeshVertData` |
| Collision shapes Chaos cloth reads from a physics asset | Spheres, capsules (sphere pairs), tapered capsules (including extruded), boxes, convexes (`ChaosClothingSimulationCollider.cpp:595-690`). Tapered capsules are cloth-only in physics assets (Physics Asset Editor: "Clothing Only") |
| Cost in the game | A 3542-particle dress simulated at ~0.42 ms a frame on the user's 14-core machine without collision (WaifuSim `WaifuSim.Cloth.Spike.RuntimeBuild`). Workshop's meter guides 1,000 simulated vertices a garment, 1,500 a character |
| The paint and card contract exists | Workshop `cloth.py`: colour attribute "WS Cloth", read as displayed (sRGB): R simulate (0 anchor .. 1 free to 40 cm), G shape-hold (anim drive 0..1), B ride (≥ 0.5). Card item `cloth`: `max_distance_cm`, `ride_threshold`, `simulated_vertices`, `overrides`, attribute names. Point attributes `_WS_CLOTH_SIM/HOLD/RIDE` in the item's .glb |

## Design decisions

1. **A second solver, not a change to the first.** Waifu Physics keeps its charter per solver: chains are a faithful Kawaii port, cloth is a faithful Chaos Cloth port. New cloth features go in only where Chaos has them. The Kawaii step, its tests and its golden data are untouched.
2. **Port the legacy PBD evolution with the shared config's defaults.** That is what the game runs today. If WaifuSim ever switches `p.ChaosCloth.UseLegacyEvolution` off, the port follows deliberately; until then `Softs::FEvolution` is out of scope.
3. **v1 scope:** particles with mass from density, gravity, damping (global and local), velocity scales for character motion, PBD edge springs, bending springs, area springs, long-range tethers (geodesic, from the kinematic set), max distance from red, anim drive from green, collisions with sphere, capsule, tapered capsule, box and plane colliders (thickness, friction), substeps and time-dependent iterations, teleport and reset. **Later:** wind and aerodynamics (drag, lift, outer drag), backstop, self-collision, XPBD and anisotropic variants, convex colliders. **Never:** Chaos LOD transitions, the Dataflow asset.
4. **The step is a pure array function, twice, in agreement.** Constraints are graph-coloured once at build into batches that share no particle; numpy runs a batch as one vectorised update, C walks the same batches in the same order, so both are Gauss-Seidel by colour and agree to the bit (tested). Colouring changes the order relative to Chaos's sequential loop; that difference is accepted and measured against the golden data (decision 11).
5. **One cloth system for the scene.** Every simulated mesh's particles live in one set of arrays, like the chain system; meshes are particle ranges with their own settings.
6. **The paint is Waifu Physics' convention.** A colour attribute **"Waifu Cloth"** on the mesh: R simulate, G hold, B ride, read as displayed. It moves here from Workshop (which named it "WS Cloth"); Workshop renames old attributes when it meets them. Defining the channels once, in the library, is what keeps the preview, the card and the game in step.
7. **The opt-in is a cloth setup on the object.** A mesh simulates when it has an enabled cloth setup (its settings: max distance, stiffnesses, density, damping, collision thickness and friction, iterations, which collider sets it uses), stored the way Blender 5 stores add-on data (system properties, migrated on rename). Workshop's per-row checkbox mirrors into the instance's setup on Instantiate and on toggle, so rebuilds keep it.
8. **Display through Geometry Nodes, no destructive writes.** A hidden node group, added as the last modifier: it stores the incoming (skinned) positions as `waifu_cloth_skinned` and sets positions from `waifu_cloth_position` where the particle is simulated, passing riding vertices through their binding. Each frame the runtime reads `waifu_cloth_skinned` from the evaluated mesh, steps, and writes `waifu_cloth_position` on the original mesh in one bulk call. Renders and the viewport show the result; removing the modifier removes the cloth.
9. **Ride binding as the game does it.** Riding vertices are bound once, at build, to the nearest sim triangle in the same barycentric-plus-normal-offset form as `FMeshToMeshVertData`, and follow it each frame. Welding by position (0.01 mm) as the game does, so seams never tear.
10. **Colliders are Waifu Physics colliders.** Cloth setups pick collider sets like chain groups do. The body colliders Workshop fits become, in the game, the cloth-only physics asset with tapered capsules the user asked for (a WaifuSim task), so the preview and the game collide with the same shapes.
11. **Parity is measured against the game.** A WaifuSim automation test simulates fixed cloth fixtures (the test skirt; a flag) under scripted kinematic motion with the default config and writes per-frame particle positions to JSON. The Blender tests replay the same fixtures and compare. Order differences (colouring, float summation) make this a tolerance test, unlike Kawaii's bit-exact golden; the tolerance is set from the first measurement and recorded here.
12. **Live by default, cached like chains.** The cloth state joins the existing runtime: live while playing, the in-memory cache and Cache All, reset on teleports and on edits that change the setup.
13. **Settings travel in the card.** Workshop exports each cloth item's setup into the card's `cloth` entry; WaifuSim applies it to the runtime clothing config (today it uses Chaos's defaults). Same names and units as `UChaosClothConfig`, converted once (Blender metres to Unreal centimetres) at export.
14. **Performance is designed in.** One read and one write per mesh per frame; no per-particle Python; the C step's pointers bound once. Target: a 1,500-particle character under 1 ms a frame for the step in C.

## Module layout

```text
waifu_physics/
  cloth/
    convention.py            the "Waifu Cloth" attribute, channels, read-as-displayed, rename of old names
    build.py                 particles from a mesh: welding, kinematic set, masses, springs, bending pairs, area
                             triangles, geodesic tethers, ride bindings, constraint colouring
    system.py                the scene-wide cloth arrays and per-mesh ranges
    step_numpy.py            the reference step (legacy PBD evolution order)
    step.c                   the C step -> bin/waifu_cloth_step.dll / .so (same release tooling)
    native.py                loads the DLL, checks its version, binds once, falls back to numpy
    display.py               the hidden Geometry Nodes group and modifier
  runtime/
    cloth_live.py            read skinned positions, step, write; joins live.py's handlers and the cache
  data/
    cloth_props.py           the cloth setup PropertyGroup (system properties), defaults = UChaosClothConfig's
    serialize.py             + cloth setups, versioned
  ui/
    cloth_panel.py           a Cloth subpanel: enable, settings, paint, collider sets
tests/
  test_cloth_build.py, test_cloth_agreement.py, test_cloth_golden.py, test_cloth_runtime.py, bench_cloth.py
```

---

### Phase 1: Spec capture

- [ ] Read and record, with file and line, the legacy step: `FPBDEvolution::AdvanceOneTimeStep`, the pre-iteration update (damping, velocity scales), the constraint creation order in `FClothConstraints`, and each v1 constraint's apply (`PBDSpringConstraintsBase.h`, `PBDBendingConstraints*`, `PBDAxialSpringConstraints*` for area, `PBDLongRangeConstraintsBase.h`, `PBDSphericalConstraint.h`, `PBDAnimDriveConstraint.h`), mass from density, tether generation (`CalculateTethers`), and cloth collision against each shape. Append the findings to the table above.
- [ ] Build the WaifuSim golden fixture test (decision 11) and commit its JSON output here under `tests/golden/`.

### Phase 2: The numpy step

- [ ] `cloth/build.py` and `cloth/system.py` from a mesh and its paint; colouring.
- [ ] `cloth/step_numpy.py`: the v1 scope, legacy order.
- [ ] `test_cloth_golden.py` against the WaifuSim fixtures; record the tolerance.

### Phase 3: The C step

- [ ] `cloth/step.c` in lockstep; `native.py`; release tooling builds both DLLs and fetches both .so files.
- [ ] `test_cloth_agreement.py`: random fixtures, every feature, C equals numpy to the bit.
- [ ] `bench_cloth.py`: record numpy and C costs.

### Phase 4: Blender runtime

- [ ] `display.py` modifier; `cloth_live.py` read, step, write; join live playback, the cache, Cache All, teleport and reset; undo-safe references.
- [ ] `cloth_props.py`, the Cloth subpanel, serialize; `test_cloth_runtime.py`.

### Phase 5: Waifu Workshop

- [ ] Bump the submodule; `cloth.py` keeps only the Workshop parts (budget guide, overridden groups, card export) and calls Waifu Physics for the convention; rename "WS Cloth" to "Waifu Cloth" where met.
- [ ] The Cloth panel's checkbox mirrors into the instance's cloth setup; a Preview toggle plays it with the body colliders; per-garment settings in the panel.
- [ ] Export the setup into the card's `cloth` entry; `tools/verify_cloth_export.py` checks it.

### Phase 6: WaifuSim

- [ ] Apply the card's cloth settings to the runtime clothing config (`WaifuModCloth`).
- [ ] The cloth-only collision physics asset from the card's body colliders (tapered capsules).
- [ ] A collision benchmark on a mid-range target, to confirm or move Workshop's budget guide.

---

## Risks

- **Colouring is not Chaos's order.** Gauss-Seidel by colour converges like Chaos's sequential loop but not identically; with one iteration a frame the difference shows. Measure it first (Phase 2) and decide whether the C step should keep Chaos's sequential order with numpy matching through a slower path.
- **The engine may change the default evolution.** A future WaifuSim engine upgrade could make the new evolution the default; the golden test catches it.
- **One iteration a frame is soft.** Chaos's defaults are stretchy; the preview must look that way too, or modders will tune against a stiffer Blender than the game.
- **The display modifier evaluates twice.** Writing positions re-evaluates the mesh; keep the node group cheap and measure it with the existing benchmark method (interleaved A/B in one Blender process).
- **Epic's licence.** Port behaviour and cite; do not paste engine code.
