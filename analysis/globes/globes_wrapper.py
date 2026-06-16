#!/usr/bin/env python3
"""
globes_wrapper.py — minimal GLoBES ctypes interface for the CP--CPT study.

Ported from:
    claude/5-DUNE-GLoBES/scripts/plot_dune_sensitivity.py

Provides the minimal surface that ``analysis/globes/experiment.py`` imports:
    load_globes()  — load libglobes.so and bind the function signatures used
                     by the band-plot / scan driver
    set_params()   — set oscillation parameters on a GLoBES params object
    GLB_ALL        — the GLoBES "all densities" sentinel (-1)
    TRUE_TH12, TRUE_TH13, TRUE_TH23, TRUE_DM21, TRUE_DM31
                   — true oscillation parameters (NuFit 5.0 / DUNE TDR values)

The full original module also defined scan/plot helpers for a DUNE sensitivity
figure; only the driver-facing surface above is ported here, verbatim in
physics (parameter values, GLoBES index conventions, signal/BG rate accessors).

Env / inputs:
    GLOBES_PREFIX — install prefix of GLoBES; libglobes is loaded from
                    ``$GLOBES_PREFIX/lib/libglobes.so``. (libgsl / libgslcblas
                    are loaded by SONAME from the system linker path.)

De-hardcode note:
    The original defaulted the GLoBES prefix to ``~/software/globes`` when
    ``GLOBES_PREFIX`` was unset. Here ``GLOBES_PREFIX`` is required (the
    portable repo resolves all machine-specific locations from env.sh); the
    user-home fallback was removed.
"""

import ctypes
import math
import os


# ============================================================================
# GLoBES interface
# ============================================================================

def load_globes():
    """Load GLoBES shared library and set up function signatures."""
    globes_prefix = os.environ.get("GLOBES_PREFIX")
    if not globes_prefix:
        raise EnvironmentError(
            "GLOBES_PREFIX is not set. Point it at your GLoBES install prefix "
            "(the directory containing lib/libglobes.so); see env.sh.example."
        )
    lib_path = os.path.join(globes_prefix, "lib", "libglobes.so")

    # Load GSL first
    for name in ["libgslcblas.so", "libgsl.so"]:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    # Set up signatures
    sigs = [
        ("glbInit", [ctypes.c_char_p], None),
        ("glbAllocParams", [], ctypes.c_void_p),
        ("glbFreeParams", [ctypes.c_void_p], None),
        ("glbDefineParams", [ctypes.c_void_p]+[ctypes.c_double]*6, ctypes.c_void_p),
        ("glbSetDensityParams", [ctypes.c_void_p, ctypes.c_double, ctypes.c_int], ctypes.c_int),
        ("glbSetOscillationParameters", [ctypes.c_void_p], ctypes.c_int),
        ("glbSetRates", [], ctypes.c_int),
        ("glbSetCentralValues", [ctypes.c_void_p], ctypes.c_int),
        ("glbSetInputErrors", [ctypes.c_void_p], ctypes.c_int),
        ("glbChiSys", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_double),
        ("glbChiTheta", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_double),
        ("glbChiAll", [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_double),
        ("glbAllocProjection", [], ctypes.c_void_p),
        ("glbFreeProjection", [ctypes.c_void_p], None),
        ("glbSetProjection", [ctypes.c_void_p], ctypes.c_int),
        ("glbSetProjectionFlag", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        ("glbSetDensityProjectionFlag", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        ("glbGetNumberOfRules", [ctypes.c_int], ctypes.c_int),
        ("glbGetNumberOfBins", [ctypes.c_int], ctypes.c_int),
        ("glbInitExperiment", [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        ("glbTotalRuleRate", [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double], ctypes.c_double),
        ("glbGetOscParams", [ctypes.c_void_p, ctypes.c_int], ctypes.c_double),
        ("glbGetSignalRatePtr", [ctypes.c_int, ctypes.c_int], ctypes.POINTER(ctypes.c_double)),
        ("glbGetBGRatePtr", [ctypes.c_int, ctypes.c_int], ctypes.POINTER(ctypes.c_double)),
        ("glbGetSignalFitRatePtr", [ctypes.c_int, ctypes.c_int], ctypes.POINTER(ctypes.c_double)),
        ("glbGetBGFitRatePtr", [ctypes.c_int, ctypes.c_int], ctypes.POINTER(ctypes.c_double)),
    ]
    for name, argtypes, restype in sigs:
        f = getattr(lib, name)
        f.argtypes = argtypes
        if restype is not None:
            f.restype = restype

    return lib


# GLoBES parameter indices
GLB_THETA_12 = 0
GLB_THETA_13 = 1
GLB_THETA_23 = 2
GLB_DELTA_CP = 3
GLB_DM_21    = 4
GLB_DM_31    = 5
GLB_ALL      = -1
GLB_FIXED    = 0
GLB_FREE     = 1


def set_params(lib, p, th12, th13, th23, dcp, dm21, dm31):
    """Set oscillation parameters on a GLoBES params object."""
    lib.glbDefineParams(p, th12, th13, th23, dcp, dm21, dm31)
    lib.glbSetDensityParams(p, 1.0, GLB_ALL)


# ============================================================================
# True oscillation parameters (NuFit 5.0, NO)
# ============================================================================

# Values used in DUNE TDR (arXiv:2002.03005, Table 1)
TRUE_S12 = 0.310       # sin^2(theta12) — NuFit 5.0
TRUE_S13 = 0.02240     # sin^2(theta13)
TRUE_S23 = 0.582       # sin^2(theta23) — upper octant
TRUE_DCP = -0.68*math.pi  # deltaCP ~ -130 deg (near maximal CPV)
TRUE_DM21 = 7.39e-5    # dm21^2 in eV^2
TRUE_DM31 = 2.525e-3   # dm31^2 in eV^2 (NO)

# Convert to GLoBES angle convention (radians)
TRUE_TH12 = math.asin(math.sqrt(TRUE_S12))
TRUE_TH13 = math.asin(math.sqrt(TRUE_S13))
TRUE_TH23 = math.asin(math.sqrt(TRUE_S23))
