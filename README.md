# Waifu Physics

Bone-chain physics for Blender: hair, skirts, tails and accessories. The simulation is a faithful port of [Kawaii Physics](https://github.com/pafuhana1213/KawaiiPhysics), the Unreal Engine plugin, so physics tuned in Blender behaves the same in a game that runs Kawaii Physics. Around it sit the things Blender needs: keyframed settings, live playback with an optional cache, collider objects and selection-based editing.

**Status:** v1.0, the first public release. Every phase of the [plan](docs/superpowers/plans/2026-09-23-waifu-physics.md) is done: the simulation, colliders, links, the cache, saved setups, external forces, wind and sync bones.

## Using it

Everything is in the 3D Viewport sidebar, on the **Waifu Physics** tab.

1. Select an armature, enter Pose Mode, select the first bone of each chain and click **New Group**. A group is one Kawaii Physics node: one set of settings for all of its chains. To give part of a chain its own settings, select just that run of bones: the group ends where the selection ends. A group started partway down another group's chain splits it, and hangs from the upper group's result, as a second Kawaii node on the chain does (the bone where they meet rides rigidly on the upper half, and the upper half never feels the lower one).
2. Pick a **Preset** (Hair or Skirt) as a starting point, then press **Simulate** and play the timeline.
3. Tune in the **Physics** panel. Every setting can be keyframed. The curve button next to a setting varies it from root to tip. With **Edit Selected Groups** on, a change reaches every group that holds a selected bone.
The group list is a tree of armatures and their groups; the cursor toggle beside **New Group** limits it to the selected armature, and off it lists every armature with a group. Clicking a group edits it. In **Chains**, tick chains (or select their bones) to **Split** them into a new group with the same settings, **Move to** another group (moving all of them merges the two), remove them, or select them in the viewport. For finer work, a chain's arrow in the chain manager lists its bones, which select, drag and delete like chains: a run of bones moved out becomes a group of its own, and deleting a bone removes it and everything below it. Each group's Links row lists its links; clicking one draws it red in the viewport, and its x removes it. All of this stays folded until opened. The eye on the Links and Colliders headers shows or hides them in the viewport; Advanced > Setup File exports or imports an armature's groups and colliders.
4. For skirts and capes, select bones in the panels' chains and click **Link Whole Chains** in **Links**: neighbouring chains are linked at every bone, and linked chains keep their spacing. A ring is left open; link its two end chains to close it. **Link Selected Bones** joins single bones in different chains.
5. Add colliders in **Colliders**. A collider is a real object parented to a bone. Move, rotate and scale it like any object, and set its shape on its modifier.
6. Push chains around in **Forces and Wind**. There are Kawaii's five external forces: Basic, Gravity, Curve, Wind and Procedural Wind. There is also a simple constant force, and scene wind that gusts. Blender's Wind force fields are the scene's wind.
7. Keep a skirt out of the legs with **Sync Bones**. Make a thigh the active bone, select skirt bones, and click **+**. The skirt's pose then follows the thigh's movement.

Every bone of a chain swings, the last one too: its tip point is the bone's tail.

The input is the pose Blender evaluates each frame. It includes animation, constraints, drivers and IK, so hair under a head that copies another rig follows it. Before each frame, Waifu Physics clears its own output from the chain bones, so last frame's physics never feeds back. Keyed chain channels keep their animation, and unkeyed ones start from rest. Start a group below any constrained bones: Blender applies a constraint after the simulation's output, so a constrained bone in a chain cannot be moved. The panel warns when a group contains one.

**Speed.** Live playback evaluates each frame once wherever that stays exact. That means keyframes alone move the bones the chains hang from, the bones colliders hang from and the armature, and no collider or force field moves by itself. Waifu Physics then samples those keys and solves before Blender evaluates the frame. Where constraints, drivers, IK or NLA move that input, it solves after evaluation instead, which costs a second evaluation of whatever the chains move. **Fast Evaluation (Live Preview)**, in Advanced > Solver, solves before evaluation everywhere and accepts that such input arrives a frame late. The panel says which path playback is on. The cache is always exact.

While simulating, Waifu Physics takes over the chain bones' keyframes: it mutes those F-curves, records them on the armature, and samples them itself. The keys still drive the simulation. They are unmuted when Simulate is off, in every saved file, after an undo, and on load after a crash. A chain channel animated by an NLA strip or a driver can't be taken over, and that rig evaluates twice per frame. The cache evaluates only what the simulation reads (the armatures, their constraint and driver targets, colliders and force fields) and hides everything else while it bakes.

**Playback.** Live playback simulates as the timeline plays. Changing Steps per Second, the frame range or the frame rate while it plays carries on without a reset. **Cache** bakes the whole frame range on the scene's fixed simulation clock, to scrub and render. Click it again to clear the bake and go back to live. A change that affects the result keeps the bake playing and marks it outdated, with **Recache**; only clearing it throws it away. Frames outside the bake's range show the unsimulated pose.

**Sharing.** **Copy Settings** and **Paste Settings** move a group's settings and curves between groups. Export Setup and Import Setup (in F3 search) save an armature's groups, links, curves and colliders to JSON and load them back. A setup moved onto another armature with the same bone names behaves identically.

## Parity with Kawaii Physics

The rule: the simulation step does exactly what Kawaii Physics does, at the pinned commit, and anything Blender-specific lives outside it. The step reproduces Kawaii's own golden test bit for bit, in C (Windows) and in numpy (everywhere else), and the two backends agree bit for bit.

To see the same motion in a game:

- Run Kawaii Physics at commit `64cbc77`. A setup records the commit it follows.
- Convert lengths to centimetres: Blender units × 100 × the scene's unit scale. Gravity, radius, tip lengths and teleport distance are lengths. Damping, stiffness and world damping have no units.
- Curves are exported as the 65 linear keys the solver used, which Kawaii's `FRichCurve` evaluates to the same values.

Where Waifu Physics' defaults and conventions differ from Kawaii's:

- **Gravity.** It is on by default: (0, 0, −1) scaled by the scene's gravity. Kawaii's default is zero.
- **Tips.** A chain's tip point sits at its last bone's tail: the bone's rest length past it. Kawaii's bones are joints with no length, so it takes one `DummyBoneLength` per node, defaulting to none. Setups list each chain's tip under `tips` for the game.
- **Steps.** Steps per Second (Kawaii's `TargetFramerate`, 60 by default) is a scene setting, where Kawaii sets it per node. Fixed Steps matches Kawaii's project setting. Live preview may take more steps in a frame to keep up with Blender's timeline; the cache uses the fixed rate without dropping elapsed time.
- **Collider sizes.** They scale with the collider object and its bone. Kawaii does not scale collider radii.
- **Capsule axis.** Capsules run along the collider's local Y. Kawaii's capsules run along Z.
- **Random draws.** Random force scales, the scene wind's gust and the Wind force's noise are seeded by timeline frame in live preview and by fixed solver tick in the cache, so a baked motion remains reproducible across scene frame rates. Kawaii draws them from Unreal's unseeded random stream. Procedural wind is seeded in both and matches exactly.
- **Wind sources.** A Blender Wind force field stands in for Unreal's wind sources. It blows along its local Z, and its Strength is Unreal's wind Speed.

## Known limitations

- Rigs whose bones copy another rig (Copy Transforms, as Waifu Workshop's garment rigs do) evaluate twice per frame in live playback, or with Fast Evaluation arrive a frame late.
- With Fast Evaluation on, keyframed physics settings (stiffness and the like) take effect a frame late.
- The C step ships for Windows and Linux. Elsewhere, such as macOS, the numpy step runs the same simulation about five times slower.

## Development

```text
python tools/run_tests.py            run every headless test in Blender
python tools/run_tests.py register   run the tests whose file name contains "register"
python tools/release.py              compile the C step (Windows), validate, build dist/waifu_physics-<version>.zip
python tools/release.py --dll-only   only compile the C step into waifu_physics/bin/
```

`WAIFU_PHYSICS_BLENDER` points the tools at a Blender executable (default: the Steam install). `WAIFU_PHYSICS_VCVARS` points the release script at `vcvars64.bat` (default: Visual Studio 2022 Build Tools).

Tests are Blender scripts in `tests/`. Each prints one `[PASS]` or `[FAIL]` line per check and exits non-zero on failure; `tests/_harness.py` has the helpers.

## Credits and licence

The simulation follows Kawaii Physics by pafuhana1213, MIT licence, at commit `64cbc77ad4d75f6eb8c8f5673b4b4452f838ec21`. Its notice ships in [`waifu_physics/THIRD_PARTY_NOTICES.txt`](waifu_physics/THIRD_PARTY_NOTICES.txt). The interface takes ideas from Swingy Bone Physics but none of its code.

Waifu Physics is GPL-3.0-or-later ([LICENSE](LICENSE)), like other Blender add-ons.
