"""Versioned ctypes boundary for the optional float32 cloth hot loop."""
import ctypes
import os
import platform
import sys

import numpy as np

from . import step_numpy

VERSION = 1
LIBRARY = "waifu_cloth_step.dll" if sys.platform == "win32" else "waifu_cloth_step.so" if sys.platform.startswith("linux") else None
DLL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", LIBRARY or "waifu_cloth_step")
POINTER_FIELDS = (
    ("x", np.float32), ("p", np.float32), ("v", np.float32), ("target", np.float32),
    ("anim_vel", np.float32), ("inv_mass", np.float32), ("max_distance", np.float32),
    ("anim_stiff", np.float32), ("anim_damp", np.float32), ("acceleration", np.float32),
    ("local_dv", np.float32), ("starts", np.int32), ("iterations", np.int32), ("params", np.float32),
    ("edges", np.int32), ("edge_length", np.float32), ("edge_starts", np.int32),
    ("bending", np.int32), ("bending_length", np.float32), ("bending_starts", np.int32),
    ("area", np.int32), ("area_length", np.float32), ("area_bary", np.float32), ("area_starts", np.int32),
    ("tethers", np.int32), ("tether_length", np.float32), ("tether_starts", np.int32),
    ("shapes", np.float32), ("shape_starts", np.int32),
)


class ClothState(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int), ("ng", ctypes.c_int), ("dt", ctypes.c_float)] + [
        (name, ctypes.c_void_p) for name, _ in POINTER_FIELDS]


class CBackend:
    NAME = "c"

    def __init__(self, dll):
        self.dll = dll
        dll.waifu_cloth_step.argtypes = [ctypes.POINTER(ClothState)]
        dll.waifu_cloth_step.restype = None

    def simulate_once(self, system):
        arrays = [getattr(system, name) for name, _ in POINTER_FIELDS]
        key = tuple(id(a) for a in arrays)
        bound = getattr(system, "_cloth_bound", None)
        if bound is None or bound[0] != key:
            struct = ClothState(n=system.n, ng=system.ng)
            for (name, dtype), array in zip(POINTER_FIELDS, arrays):
                if array.dtype != dtype or not array.flags.c_contiguous:
                    raise TypeError(f"cloth {name} must be contiguous {np.dtype(dtype)}")
                setattr(struct, name, array.ctypes.data)
            bound = system._cloth_bound = (key, struct, arrays)
        bound[1].dt = float(system.step_dt)
        self.dll.waifu_cloth_step(ctypes.byref(bound[1]))


_backend = None
_reason = ""


def backend():
    global _backend, _reason
    if _backend is None:
        if LIBRARY is None or platform.machine().lower() not in ("amd64", "x86_64"):
            _reason = "no native cloth build for this platform"
        elif not os.path.exists(DLL_PATH):
            _reason = "the native cloth step has not been built"
        else:
            try:
                dll = ctypes.CDLL(DLL_PATH)
                dll.waifu_cloth_version.restype = ctypes.c_int
                version = dll.waifu_cloth_version()
                if version == VERSION:
                    _backend = CBackend(dll)
                else:
                    _reason = f"cloth library version {version}; expected {VERSION}"
            except (OSError, AttributeError) as exc:
                _reason = f"the native cloth step could not load: {exc}"
        if _backend is None:
            _backend = step_numpy
    return _backend


def reason():
    backend()
    return _reason
