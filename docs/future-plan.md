# Future plan

Where Waifu Physics goes after 1.0: how it is released, what is left before and after, and the idea for a paid 2.0, "Waifu Physics Pro".

## Release and distribution (1.x)

Waifu Physics is free and stays free.

- **Licence:** GPL-3.0-or-later, like all Blender add-ons. The simulation follows Kawaii Physics (MIT), whose notice ships with the package.
- **Where:** GitHub (the source) and extensions.blender.org, which only lists free extensions and is the channel Blender itself shows its users.
- **Support:** an optional, pay-what-you-want listing on Gumroad for anyone who wants to support the work. Nothing is held back from the free version.
- **Scope:** a faithful port of Kawaii Physics, so physics tuned in Blender behaves the same in a game that runs Kawaii. New simulation features go into the solver only where Kawaii has them. Blender-specific features live around the solver: the live and cached playback, keyframed settings, colliders, setups and the UI.
- **Maintenance:** the add-on is maintained for WaifuSim's modding pipeline first. The README should say so, so expectations about support are clear.

### Before the public release

- Build the release zip (`python tools/release.py`) and test it in a clean Blender: install, simulate, cache, save, reload.
- Check extensions.blender.org's rules on bundled native libraries (the C step is a DLL / .so loaded with ctypes and falls back to numpy; its source ships with the package).
- Set up the Gumroad listing.

### After 1.0

- A native C step for macOS (a GitHub Actions workflow, as for Linux), so macOS is not left on the slower numpy step.
- The game side: Kawaii takes one `DummyBoneLength` per node, while Waifu Physics gives each chain its own tip (a setup's `tips`). WaifuSim's Kawaii copy may need a per-bone tip length.
- Waifu Workshop's garment rigs copy the base rig with Copy Transforms, so their live playback evaluates twice per frame. If that becomes a cost worth paying down, the fast path could learn to read Copy Transforms targets ahead of evaluation.

## Waifu Physics Pro (2.0, paid)

A separate, paid add-on that goes beyond what a game running Kawaii can reproduce: physically based rods for Blender animators who bake their motion, in film, stills and VTuber recordings. It is GPL too: buyers may share it, so it sells convenience, updates and support, not exclusivity.

### The solver: Cosserat rods

Kawaii treats a chain as points joined by fixed lengths, with a per-step pull toward the pose standing in for stiffness. A Cosserat rod also carries each segment's orientation (a material frame), which gives what Kawaii cannot express:

- **Bending and twisting stiffness** as material properties, independent of the step rate.
- **Twist** that travels along the rod: tails, ribbons, straps, anything with a visible cross-section.
- **Rest shapes** held by the material (curls, bends), not by a pull toward the animation.
- **Stretch and shear.** These are part of the Cosserat strain already, not an addition. The stretch-shear constraint, `C = (p2 - p1) / L - R(q) e3`, is the XPBD distance constraint the links use, coupled to the segment's orientation. Zero compliance makes an inextensible rod (hair, rope). Some compliance makes an elastic one that stretches under load and springs back. Separate compliances along and across the axis let a rod stretch without shearing.

The approach: XPBD Cosserat rods (Kugelstadt and Schömer, "Position and Orientation Based Cosserat Rods", 2016). It fits the XPBD machinery the links already use, runs in real time, and stays stable at high stiffness with sub-stepping. Discrete Elastic Rods (Bergou et al., 2008) is the more exact but heavier alternative.

### Features

- Stretch with a maximum strain (stretch limit), optional thinning (the collision radius shrinks as a rod stretches), and animatable rest lengths (growth, squash).
- Collision against real meshes, not only primitive colliders.
- Rods colliding with each other.
- Pinning and blending with the animation along a rod.
- Material properties per bone, not only per group and curve.
- Bake to keys as a first-class workflow.

### Architecture

- The solver step is a pure array function, and the groups, colliders, cache, UI and setup files do not depend on it. Pro shares all of that and brings its own solver and its own settings: a Cosserat stiffness means something physically different from Kawaii's per-step pull, so the two settings sets must not be mixed.
- **Writing stretch to bones:** a bone stretches through its scale or location, and scale Y is inherited by its children. Either compensate the children, or place them by location with Inherit Scale None or Aligned. The inheritance port in `runtime/io.py` already handles every mode.
- **Native builds on every platform,** macOS included. Rods need many more iterations per step than Kawaii, so the numpy fallback is not enough.

### Open questions

- Is there enough demand? The optional Gumroad payments for 1.x are the first signal.
- Should Pro's solver also be ported to Unreal, so WaifuSim can play Pro motion in the game and get back the Blender/game parity that Pro gives up?
