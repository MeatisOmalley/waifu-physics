"""Waifu Physics: bone-chain physics for Blender, ported from Kawaii Physics."""

# Reload Scripts re-runs this file with its old globals still in place; reload
# the submodules too (dependencies first), so an edit takes effect without a restart.
if "bpy" in locals():
    import importlib
    for _module in _RELOAD_ORDER:
        importlib.reload(_module)
else:
    from .solver import uemath, system, curves as solver_curves, step_numpy, native, build
    from .cloth import convention, display, build as cloth_build, system as cloth_system, step_numpy as cloth_numpy, native as cloth_native
    from .data import props, cloth_props, curves, colliders, links, serialize, presets, legacy
    from .runtime import keys, io, cache, cloth_live, live
    from .ui import selection, ops, manager, panels, cloth_panel, draw

import bpy  # noqa: E402,F401  (its presence marks a reload, above)

_RELOAD_ORDER = (uemath, system, solver_curves, step_numpy, native, build,
                 convention, display, cloth_build, cloth_numpy, cloth_native, cloth_system, props, cloth_props,
                 curves, colliders, links, serialize, presets, legacy, keys, io, cache, cloth_live, live,
                 selection, ops, manager, panels, cloth_panel, draw)

# Registered in this order, unregistered in reverse.
MODULES = (props, cloth_props, legacy, live, selection, ops, manager, panels, cloth_panel, draw)


def register():
    for module in MODULES:
        module.register()


def unregister():
    for module in reversed(MODULES):
        module.unregister()
