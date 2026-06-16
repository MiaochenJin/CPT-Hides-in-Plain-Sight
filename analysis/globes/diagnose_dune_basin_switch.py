#!/usr/bin/env python3
"""
diagnose_dune_basin_switch.py — Diagnose basin-switch in DUNE method-d CPT bias scan.

Ported from:
    claude/3-CPT-violation/CP-CPT-degeneracy/DUNE/scripts/diagnose_basin_switch_method_d.py

Produces the basin-switch diagnostic figures (paper Suppl Fig 2). It both RUNS
GLoBES (Phase 2 event spectra, incl. the publication-quality "paper" version for
the Δ ≈ +0.76e-3 double-minimum point) and plots from the scan NPZs.

Three diagnostic phases:
  Phase 1 (pure analysis): Load task NPZ files, identify basin switches,
           plot chi2 profiles (Plot A) and transition summary (Plot D)
  Phase 2 (GLoBES): Compute event spectra at minima (Plot B)  [needs GLoBES]
  Phase 3 (nuSQuIDS): Oscillation probability comparison (Plot C)  [needs nuSQuIDS]

Env / inputs:
    --results-dir  directory containing task_000.npz ... task_099.npz from
                   run_dune_dcp_scan.py. NOT shipped with the repo.
    GLOBES_PREFIX  GLoBES install prefix (Phase 2 only).
    NUSQUIDS_DATA_PATH + nuSQuIDS python bindings (Phase 3 only).
    DUNE_GLoBES_CPT.glb config tree — resolved via
    paths.globes_config("dune_globes/DUNE_GLoBES_CPT.glb").

Usage:
    python3 diagnose_dune_basin_switch.py --results-dir <dir> [--skip-globes] [--skip-nusquids]

De-hardcode notes:
  - The original self-contained load_globes() defaulted the GLoBES prefix to
    ``~/software/globes``; here GLOBES_PREFIX is required (no home-dir fallback).
  - The original required ``cd <dune_globes_config_dir>`` before running so
    GLoBES could resolve the .glb's relative includes. We instead resolve the
    committed .glb via paths.globes_config and os.chdir into its directory only
    inside phase2_globes() (after results/output dirs are made absolute), so
    --results-dir / --output-dir stay relative to the original CWD.
"""

import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import argrelmin
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# Paper rcParams
# ============================================================================
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================================
# Constants (from plot_dune_sensitivity.py)
# ============================================================================
TRUE_S12 = 0.310
TRUE_S13 = 0.02240
TRUE_S23 = 0.582
TRUE_DM21 = 7.39e-5
TRUE_DM31 = 2.525e-3
TRUE_TH12 = math.asin(math.sqrt(TRUE_S12))
TRUE_TH13 = math.asin(math.sqrt(TRUE_S13))
TRUE_TH23 = math.asin(math.sqrt(TRUE_S23))
TRUE_DCP_0 = 0.0  # CP-conserving truth for this scan

# Method-d definitions
NU_RULES = [0, 2, 4, 6]
NUBAR_RULES = [1, 3, 5, 7]
PHYSICAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
PHYSICAL_NAMES = [
    r'$\nu_e$ app FHC',
    r'$\nu_e$ app RHC',
    r'$\nu_\mu$ dis FHC',
    r'$\nu_\mu$ dis RHC',
]
SIG_NORM_ERROR = [0.02, 0.02, 0.05, 0.05]
BG_NORM_ERROR = [0.05, 0.05, 0.10, 0.10]

# nuSQuIDS propagation constants
L_KM = 1284.9
RHO = 2.848
Y_E = 0.5


# ============================================================================
# Utility functions
# ============================================================================

def find_local_minima(chi2_arr, order=5):
    """Find local minima in chi2 array using scipy.signal.argrelmin."""
    rel_min_idx = argrelmin(chi2_arr, order=order)[0]
    global_min = np.argmin(chi2_arr)
    all_minima = set(rel_min_idx.tolist())
    all_minima.add(global_min)
    return sorted(all_minima)


def get_bin_edges(n_bins):
    """Reconstruct DUNE bin edges."""
    bin_widths = [0.125] * 64 + [1.0] * 2 + [2.0] * 5 + [10.0] * 9
    assert len(bin_widths) == n_bins, f"Expected {n_bins} bins, got {len(bin_widths)}"
    edges = np.zeros(n_bins + 1)
    for i, w in enumerate(bin_widths):
        edges[i + 1] = edges[i] + w
    return edges


# ============================================================================
# GLoBES functions (from run_dcp_scan_method_d.py)
# ============================================================================

def load_globes():
    """Load GLoBES shared library and set up function signatures."""
    import ctypes
    globes_prefix = os.environ.get("GLOBES_PREFIX")
    if not globes_prefix:
        raise EnvironmentError(
            "GLOBES_PREFIX is not set. Point it at your GLoBES install prefix "
            "(the directory containing lib/libglobes.so); see env.sh.example."
        )
    lib_path = os.path.join(globes_prefix, "lib", "libglobes.so")
    for name in ["libgslcblas.so", "libgsl.so"]:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
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
        ("glbGetNumberOfRules", [ctypes.c_int], ctypes.c_int),
        ("glbGetNumberOfBins", [ctypes.c_int], ctypes.c_int),
        ("glbInitExperiment", [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
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


GLB_ALL = -1


def set_params(lib, p, th12, th13, th23, dcp, dm21, dm31):
    lib.glbDefineParams(p, th12, th13, th23, dcp, dm21, dm31)
    lib.glbSetDensityParams(p, 1.0, GLB_ALL)


def init_dune_cpt(lib, glb_file="DUNE_GLoBES_CPT.glb"):
    import ctypes
    lib.glbInit(b"basin_diagnostics")
    num_exps = ctypes.c_int.in_dll(lib, "glb_num_of_exps")
    exp_list_addr = ctypes.addressof(ctypes.c_void_p.in_dll(lib, "glb_experiment_list"))
    lib.glbInitExperiment(glb_file.encode(), exp_list_addr, ctypes.byref(num_exps))
    n_rules = lib.glbGetNumberOfRules(0)
    assert n_rules == 8
    return n_rules, lib.glbGetNumberOfBins(0)


def extract_sig_bg(lib, n_bins, rule):
    sig_ptr = lib.glbGetSignalFitRatePtr(0, rule)
    bg_ptr = lib.glbGetBGFitRatePtr(0, rule)
    return (np.array([sig_ptr[i] for i in range(n_bins)]),
            np.array([bg_ptr[i] for i in range(n_bins)]))


def extract_truth_sig_bg(lib, n_bins, rule):
    sig_ptr = lib.glbGetSignalRatePtr(0, rule)
    bg_ptr = lib.glbGetBGRatePtr(0, rule)
    return (np.array([sig_ptr[i] for i in range(n_bins)]),
            np.array([bg_ptr[i] for i in range(n_bins)]))


def poisson_chi2_np(n_obs, n_exp):
    chi2 = 0.0
    mask = (n_exp > 0) & (n_obs > 0)
    chi2 += np.sum(2.0 * (n_exp[mask] - n_obs[mask] +
                          n_obs[mask] * np.log(n_obs[mask] / n_exp[mask])))
    mask2 = (n_exp > 0) & (n_obs == 0)
    chi2 += np.sum(2.0 * n_exp[mask2])
    return chi2


def generate_cpt_truth(lib, true_p, n_bins, delta_true):
    dm31_bar = TRUE_DM31 + delta_true
    truth_cache = {}
    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, TRUE_DCP_0, TRUE_DM21, TRUE_DM31)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()
    for r in NU_RULES:
        truth_cache[r] = extract_truth_sig_bg(lib, n_bins, r)
    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, TRUE_DCP_0, TRUE_DM21, dm31_bar)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()
    for r in NUBAR_RULES:
        truth_cache[r] = extract_truth_sig_bg(lib, n_bins, r)
    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, TRUE_DCP_0, TRUE_DM21, TRUE_DM31)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()
    return truth_cache


def chi2_method_d_cpt_conserving(lib, test_p, n_bins, truth_cache,
                                  th12, th13, th23, dcp, dm21, dm31_test):
    """Method (d) chi2 with CPT-conserving fit. Returns (chi2, nuisance_params)."""
    set_params(lib, test_p, th12, th13, th23, dcp, dm21, dm31_test)
    lib.glbChiSys(test_p, 0, 0)
    fit_all = {r: extract_sig_bg(lib, n_bins, r) for r in range(8)}

    total_chi2 = 0.0
    all_nuisance = []
    for pair_idx, (nu_rule, nubar_rule) in enumerate(PHYSICAL_PAIRS):
        fit_sig_nu, fit_bg_nu = fit_all[nu_rule]
        fit_sig_nubar, fit_bg_nubar = fit_all[nubar_rule]
        truth_sig_nu, truth_bg_nu = truth_cache[nu_rule]
        truth_sig_nubar, truth_bg_nubar = truth_cache[nubar_rule]

        truth_combined = (truth_sig_nu + truth_bg_nu) + (truth_sig_nubar + truth_bg_nubar)
        fit_sig_combined = fit_sig_nu + fit_sig_nubar
        fit_bg_combined = fit_bg_nu + fit_bg_nubar

        sig_err = SIG_NORM_ERROR[pair_idx]
        bg_err = BG_NORM_ERROR[pair_idx]

        def objective(x, _sig=fit_sig_combined, _bg=fit_bg_combined,
                      _truth=truth_combined, _se=sig_err, _be=bg_err):
            a_sig, a_bg = x
            fit_mod = (1 + a_sig) * _sig + (1 + a_bg) * _bg
            return poisson_chi2_np(_truth, fit_mod) + (a_sig / _se)**2 + (a_bg / _be)**2

        res = minimize(objective, [0.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-4, 'maxfev': 200})
        total_chi2 += res.fun
        all_nuisance.append(res.x.copy())

    return total_chi2, all_nuisance


def get_fit_rates_method_d(lib, test_p, n_bins, truth_cache,
                            th23, dcp, dm31_test, th13):
    """Get combined physical-channel rates at given fit point.

    Returns: truth_phys[4, n_bins], fit_phys[4, n_bins], fit_mod_phys[4, n_bins]
    """
    set_params(lib, test_p, TRUE_TH12, th13, th23, dcp, TRUE_DM21, dm31_test)
    lib.glbChiSys(test_p, 0, 0)
    fit_all = {r: extract_sig_bg(lib, n_bins, r) for r in range(8)}

    truth_phys = np.zeros((4, n_bins))
    fit_phys = np.zeros((4, n_bins))
    fit_mod_phys = np.zeros((4, n_bins))

    for pair_idx, (nu_rule, nubar_rule) in enumerate(PHYSICAL_PAIRS):
        fit_sig_nu, fit_bg_nu = fit_all[nu_rule]
        fit_sig_nubar, fit_bg_nubar = fit_all[nubar_rule]
        truth_sig_nu, truth_bg_nu = truth_cache[nu_rule]
        truth_sig_nubar, truth_bg_nubar = truth_cache[nubar_rule]

        truth_combined = (truth_sig_nu + truth_bg_nu) + (truth_sig_nubar + truth_bg_nubar)
        fit_sig_combined = fit_sig_nu + fit_sig_nubar
        fit_bg_combined = fit_bg_nu + fit_bg_nubar

        truth_phys[pair_idx] = truth_combined
        fit_phys[pair_idx] = fit_sig_combined + fit_bg_combined

        # Profile nuisance
        sig_err = SIG_NORM_ERROR[pair_idx]
        bg_err = BG_NORM_ERROR[pair_idx]

        def objective(x, _sig=fit_sig_combined, _bg=fit_bg_combined,
                      _truth=truth_combined, _se=sig_err, _be=bg_err):
            a_sig, a_bg = x
            fit_mod = (1 + a_sig) * _sig + (1 + a_bg) * _bg
            return poisson_chi2_np(_truth, fit_mod) + (a_sig / _se)**2 + (a_bg / _be)**2

        res = minimize(objective, [0.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-4, 'maxfev': 200})
        a_sig, a_bg = res.x
        fit_mod_phys[pair_idx] = (1 + a_sig) * fit_sig_combined + (1 + a_bg) * fit_bg_combined

    return truth_phys, fit_phys, fit_mod_phys


# ============================================================================
# nuSQuIDS propagation (from find_degeneracy_all_decompositions.py)
# ============================================================================

def propagate(energies, nu_type_str, dm31_val, dcp, use_matter):
    """Run nuSQuIDS propagation. Returns P(nu_mu -> nu_e)."""
    import nuSQuIDS as nsq
    units = nsq.Const()
    energies_eV = energies * units.GeV
    L_eV = L_KM * units.km
    n_e = len(energies)
    nu_type = nsq.NeutrinoType.neutrino if nu_type_str == 'nu' else nsq.NeutrinoType.antineutrino
    nusq = nsq.nuSQUIDS(energies_eV, 3, nu_type, False)
    nusq.Set_MixingAngle(0, 1, TRUE_TH12)
    nusq.Set_MixingAngle(0, 2, TRUE_TH13)
    nusq.Set_MixingAngle(1, 2, TRUE_TH23)
    nusq.Set_CPPhase(0, 2, dcp)
    nusq.Set_SquareMassDifference(1, TRUE_DM21)
    nusq.Set_SquareMassDifference(2, dm31_val)
    if use_matter:
        body = nsq.ConstantDensity(RHO, Y_E)
        track = nsq.ConstantDensity.Track(L_eV)
    else:
        body = nsq.Vacuum()
        track = nsq.Vacuum.Track(L_eV)
    nusq.Set_Body(body)
    nusq.Set_Track(track)
    init_state = np.zeros((n_e, 3))
    init_state[:, 1] = 1.0
    nusq.Set_initial_state(init_state, nsq.Basis.flavor)
    nusq.Set_rel_error(1e-9)
    nusq.Set_abs_error(1e-9)
    nusq.EvolveState()
    return np.array([nusq.EvalFlavorAtNode(0, i) for i in range(n_e)])


def propagate_custom(energies, nu_type_str, dm31_val, dcp, th23, th13, use_matter):
    """Like propagate() but with custom th23, th13."""
    import nuSQuIDS as nsq
    units = nsq.Const()
    energies_eV = energies * units.GeV
    L_eV = L_KM * units.km
    n_e = len(energies)
    nu_type = nsq.NeutrinoType.neutrino if nu_type_str == 'nu' else nsq.NeutrinoType.antineutrino
    nusq = nsq.nuSQUIDS(energies_eV, 3, nu_type, False)
    nusq.Set_MixingAngle(0, 1, TRUE_TH12)
    nusq.Set_MixingAngle(0, 2, th13)
    nusq.Set_MixingAngle(1, 2, th23)
    nusq.Set_CPPhase(0, 2, dcp)
    nusq.Set_SquareMassDifference(1, TRUE_DM21)
    nusq.Set_SquareMassDifference(2, dm31_val)
    if use_matter:
        body = nsq.ConstantDensity(RHO, Y_E)
        track = nsq.ConstantDensity.Track(L_eV)
    else:
        body = nsq.Vacuum()
        track = nsq.Vacuum.Track(L_eV)
    nusq.Set_Body(body)
    nusq.Set_Track(track)
    init_state = np.zeros((n_e, 3))
    init_state[:, 1] = 1.0
    nusq.Set_initial_state(init_state, nsq.Basis.flavor)
    nusq.Set_rel_error(1e-9)
    nusq.Set_abs_error(1e-9)
    nusq.EvolveState()
    return np.array([nusq.EvalFlavorAtNode(0, i) for i in range(n_e)])


# ============================================================================
# Phase 1: Pure analysis (numpy/scipy/matplotlib only)
# ============================================================================

def phase1_analysis(results_dir, output_dir):
    """Load task NPZs, identify basin switches, generate Plots A and D."""
    print("=" * 70)
    print("Phase 1: Pure Analysis — Loading results and identifying basin switches")
    print("=" * 70)

    # Load all task NPZs
    tasks = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith('task_') and fname.endswith('.npz'):
            path = os.path.join(results_dir, fname)
            data = np.load(path)
            tasks.append({
                'delta_true': float(data['delta_true']),
                'chi2_best': data['chi2_best'],
                'dcp_vals': data['dcp_vals'],
                'best_params': data['best_params'],  # [N_DCP, 3] = (th23, dm31, th13)
            })

    tasks.sort(key=lambda t: t['delta_true'])
    n_tasks = len(tasks)
    print(f"Loaded {n_tasks} task files")

    if n_tasks == 0:
        print("ERROR: No task files found!")
        return None

    dcp_vals = tasks[0]['dcp_vals']
    dcp_deg = np.degrees(dcp_vals)

    # Extract best-fit dCP and min chi2 for each task
    deltas = np.array([t['delta_true'] for t in tasks])
    best_dcp_deg = np.zeros(n_tasks)
    min_chi2 = np.zeros(n_tasks)
    secondary_dcp_all = []  # list of arrays

    for i, t in enumerate(tasks):
        chi2 = t['chi2_best']
        imin = np.argmin(chi2)
        best_dcp_deg[i] = dcp_deg[imin]
        min_chi2[i] = chi2[imin]

        # Find secondary minima
        local_min_idx = find_local_minima(chi2, order=5)
        dchi2 = chi2 - chi2.min()
        secondary = []
        for mi in local_min_idx:
            if mi != imin and dchi2[mi] <= 9.0:  # within 3sigma
                secondary.append(dcp_deg[mi])
        secondary_dcp_all.append(np.array(secondary))

    # Identify basin switches: adjacent Delta where best dCP jumps > 30 deg
    switch_indices = []
    for i in range(n_tasks - 1):
        jump = abs(best_dcp_deg[i + 1] - best_dcp_deg[i])
        if jump > 30:
            switch_indices.append(i)
            print(f"  Basin switch at Delta: {deltas[i]*1e3:+.3f} -> {deltas[i+1]*1e3:+.3f} x10^-3, "
                  f"dCP: {best_dcp_deg[i]:+.1f} -> {best_dcp_deg[i+1]:+.1f} deg (jump={jump:.0f} deg)")

    # Select ~5 representative Delta values
    if len(switch_indices) > 0:
        # Pre-transition, at-transition, post-transition, + large positive/negative
        representatives = []
        sw = switch_indices[0]  # primary switch

        # Large negative Delta
        idx_neg = max(0, int(n_tasks * 0.1))
        representatives.append(idx_neg)

        # Pre-transition (2 steps before switch)
        representatives.append(max(0, sw - 2))

        # At transition
        representatives.append(sw)
        representatives.append(min(n_tasks - 1, sw + 1))

        # Large positive Delta
        idx_pos = min(n_tasks - 1, int(n_tasks * 0.9))
        representatives.append(idx_pos)

        # Deduplicate and limit to 5
        seen = set()
        rep_unique = []
        for idx in representatives:
            if idx not in seen:
                seen.add(idx)
                rep_unique.append(idx)
        representatives = rep_unique[:5]
    else:
        # Fallback: use fixed Delta values
        fallback_deltas = [-1.5e-3, -0.5e-3, 0.3e-3, 0.7e-3, 1.5e-3]
        representatives = []
        for fd in fallback_deltas:
            idx = np.argmin(np.abs(deltas - fd))
            if idx not in representatives:
                representatives.append(idx)
        print("  No clear basin switch found, using fallback Delta values")

    print(f"\nRepresentative indices: {representatives}")
    for idx in representatives:
        t = tasks[idx]
        print(f"  Delta = {t['delta_true']*1e3:+.3f}e-3, best dCP = {best_dcp_deg[idx]:+.1f} deg, "
              f"min chi2 = {min_chi2[idx]:.2f}")

    # ================================================================
    # Plot A: Chi2 profiles at representative Delta values (3x2 grid)
    # ================================================================
    n_rep = len(representatives)
    n_panels = min(n_rep + 1, 6)  # 5 profiles + 1 summary
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes_flat = axes.flatten()

    for panel_idx in range(min(n_rep, 5)):
        ax = axes_flat[panel_idx]
        idx = representatives[panel_idx]
        t = tasks[idx]
        chi2 = t['chi2_best']
        dchi2 = chi2 - chi2.min()
        imin = np.argmin(chi2)

        ax.plot(dcp_deg, dchi2, 'k-', lw=1.5)

        # Global minimum
        ax.plot(dcp_deg[imin], dchi2[imin], 'r*', ms=14, zorder=10)
        ax.annotate(f'{dcp_deg[imin]:+.0f}' + r'$°$',
                    xy=(dcp_deg[imin], dchi2[imin]),
                    xytext=(12, 12), textcoords='offset points',
                    fontsize=9, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

        # Secondary local minima
        local_mins = find_local_minima(chi2, order=5)
        for mi in local_mins:
            if mi != imin and dchi2[mi] <= 9.0:
                ax.plot(dcp_deg[mi], dchi2[mi], 'bD', ms=7, zorder=9)
                ax.annotate(f'{dcp_deg[mi]:+.0f}' + r'$°$' + f'\n$\\Delta\\chi^2$={dchi2[mi]:.1f}',
                            xy=(dcp_deg[mi], dchi2[mi]),
                            xytext=(12, 8), textcoords='offset points',
                            fontsize=8, color='blue',
                            arrowprops=dict(arrowstyle='->', color='blue', lw=0.6))

        # Threshold lines
        for thresh, label, ls in [(1.0, r'1$\sigma$', '--'),
                                   (2.71, '90% CL', ':'),
                                   (4.0, r'2$\sigma$', '-.'),
                                   (9.0, r'3$\sigma$', '--')]:
            ax.axhline(thresh, color='gray', ls=ls, alpha=0.4, lw=0.7)

        ax.set_xlim(-180, 180)
        ax.set_ylim(bottom=-0.2)
        ax.set_ylim(top=min(max(dchi2.max(), 10), 50))
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.set_ylabel(r'$\Delta\chi^2$')
        ax.set_title(f'$\\Delta = {t["delta_true"]*1e3:+.2f}' + r'\times 10^{-3}$ eV$^2$',
                     fontsize=11)
        if panel_idx >= 3:
            ax.set_xlabel(r'$\delta_{CP}$ [degrees]')

    # Summary panel: all profiles overlaid
    ax = axes_flat[5] if n_panels == 6 else axes_flat[min(n_rep, 5)]
    colors_rep = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for panel_idx in range(min(n_rep, 5)):
        idx = representatives[panel_idx]
        t = tasks[idx]
        chi2 = t['chi2_best']
        dchi2 = chi2 - chi2.min()
        ax.plot(dcp_deg, dchi2, lw=1.5, color=colors_rep[panel_idx],
                label=f'$\\Delta$={t["delta_true"]*1e3:+.2f}')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.4, lw=0.7)
    ax.axhline(4.0, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.set_xlim(-180, 180)
    ax.set_ylim(bottom=-0.2, top=20)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.set_xlabel(r'$\delta_{CP}$ [degrees]')
    ax.set_ylabel(r'$\Delta\chi^2$')
    ax.set_title('All representative $\\Delta$ overlaid', fontsize=11)
    ax.legend(fontsize=8, loc='upper right', title=r'$\Delta$ [$\times 10^{-3}$]',
              title_fontsize=8)

    fig.suptitle(r'(A) $\chi^2$ Profiles at Representative $\Delta$ Values'
                 '\nMethod (d): combined spectra + shared nuisance profiling',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot_A_chi2_profiles.png'),
                dpi=200, bbox_inches='tight')
    print("Saved plot_A_chi2_profiles.png")
    plt.close()

    # ================================================================
    # Plot D: Transition summary (2 side-by-side panels)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    deltas_1e3 = deltas * 1e3

    # Left: best-fit dCP vs Delta_true
    ax1.plot(deltas_1e3, best_dcp_deg, 'ko-', ms=3, lw=1, label='Best-fit')

    # Secondary minima as gray dots
    for i in range(n_tasks):
        for sec_dcp in secondary_dcp_all[i]:
            ax1.plot(deltas_1e3[i], sec_dcp, '.', color='gray', ms=3, alpha=0.5)

    # Red verticals at switches
    for si in switch_indices:
        x_mid = 0.5 * (deltas_1e3[si] + deltas_1e3[si + 1])
        ax1.axvline(x_mid, color='red', ls='--', lw=1.5, alpha=0.7, zorder=3)

    ax1.axhline(0, color='green', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel(r'$\Delta_{\rm true}$ [$\times 10^{-3}$ eV$^2$]')
    ax1.set_ylabel(r'Best-fit $\delta_{CP}$ [degrees]')
    ax1.set_title('Best-fit $\\delta_{CP}$ vs $\\Delta_{\\rm true}$')
    ax1.set_ylim(-200, 200)
    ax1.yaxis.set_major_locator(MultipleLocator(45))

    handles = [
        plt.Line2D([0], [0], color='k', marker='o', ms=3, label='Global min'),
        plt.Line2D([0], [0], color='gray', marker='.', ls='', ms=5, label='Secondary min'),
        plt.Line2D([0], [0], color='red', ls='--', lw=1.5, label='Basin switch'),
    ]
    ax1.legend(handles=handles, fontsize=9, loc='upper left')

    # Right: min chi2 vs Delta_true
    ax2.plot(deltas_1e3, min_chi2, 'ko-', ms=3, lw=1)
    for si in switch_indices:
        x_mid = 0.5 * (deltas_1e3[si] + deltas_1e3[si + 1])
        ax2.axvline(x_mid, color='red', ls='--', lw=1.5, alpha=0.7)
    ax2.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax2.set_xlabel(r'$\Delta_{\rm true}$ [$\times 10^{-3}$ eV$^2$]')
    ax2.set_ylabel(r'$\chi^2_{\rm min}$')
    ax2.set_title(r'Minimum $\chi^2$ vs $\Delta_{\rm true}$')

    fig.suptitle('(D) Basin Switch Transition Summary\n'
                 r'Method (d): truth $\delta_{CP}=0$, CPT-conserving fit profiles $\theta_{23}$, $\Delta m^2_{31}$, $\theta_{13}$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot_D_transition_summary.png'),
                dpi=200, bbox_inches='tight')
    print("Saved plot_D_transition_summary.png")
    plt.close()

    # Return data for Phases 2 & 3
    return {
        'tasks': tasks,
        'deltas': deltas,
        'best_dcp_deg': best_dcp_deg,
        'min_chi2': min_chi2,
        'representatives': representatives,
        'switch_indices': switch_indices,
        'secondary_dcp_all': secondary_dcp_all,
    }


# ============================================================================
# Phase 2: GLoBES event spectra
# ============================================================================

def phase2_globes(phase1_data, output_dir):
    """Compute event spectra at minima for representative Delta values. Plot B."""
    print("\n" + "=" * 70)
    print("Phase 2: GLoBES Event Spectra at Minima")
    print("=" * 70)

    tasks = phase1_data['tasks']
    representatives = phase1_data['representatives']
    n_rep = len(representatives)

    # GLoBES resolves the .glb's relative includes (flux/xsec/eff/smr) from CWD;
    # chdir into the committed config dir. output_dir is already absolute (made so
    # in main()), so plot writes are unaffected.
    glb_path = paths.globes_config("dune_globes/DUNE_GLoBES_CPT.glb")
    os.chdir(os.path.dirname(glb_path))

    # Initialize GLoBES
    lib = load_globes()
    n_rules, n_bins = init_dune_cpt(lib)
    print(f"GLoBES initialized: {n_rules} rules, {n_bins} bins")

    true_p = lib.glbAllocParams()
    test_p = lib.glbAllocParams()
    err_p = lib.glbAllocParams()

    import ctypes
    set_params(lib, err_p,
               TRUE_TH12 * 0.024, TRUE_TH13 * 0.03, 0.0, 0.0,
               TRUE_DM21 * 0.024, 0.0)
    lib.glbSetDensityParams(err_p, 0.05, GLB_ALL)
    lib.glbSetInputErrors(err_p)

    bin_edges = get_bin_edges(n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    in_window = (bin_centers >= 0) & (bin_centers <= 8.0)
    bc_window = bin_centers[in_window]

    # Compute CPT-conserving truth (Delta=0) — baseline for comparison
    print("Computing CPT-conserving truth (Delta=0)...")
    cpt_conserving_cache = generate_cpt_truth(lib, true_p, n_bins, 0.0)
    cpt_conserving_phys = np.zeros((4, n_bins))
    for pair_idx, (nu_rule, nubar_rule) in enumerate(PHYSICAL_PAIRS):
        sig_nu, bg_nu = cpt_conserving_cache[nu_rule]
        sig_nubar, bg_nubar = cpt_conserving_cache[nubar_rule]
        cpt_conserving_phys[pair_idx] = (sig_nu + bg_nu) + (sig_nubar + bg_nubar)

    # For each representative Delta, generate truth and compute spectra at minima
    all_spectra = {}

    for panel_idx in range(min(n_rep, 5)):
        idx = representatives[panel_idx]
        t = tasks[idx]
        delta_true = t['delta_true']
        chi2 = t['chi2_best']
        dcp_vals = t['dcp_vals']
        dcp_deg = np.degrees(dcp_vals)
        params = t['best_params']  # [N_DCP, 3] = (th23, dm31, th13)

        print(f"\n  Delta = {delta_true*1e3:+.3f}e-3:")

        # Generate CPT-violating truth
        truth_cache = generate_cpt_truth(lib, true_p, n_bins, delta_true)

        # Find global minimum and secondary minima
        imin = np.argmin(chi2)
        local_mins = find_local_minima(chi2, order=5)
        dchi2 = chi2 - chi2.min()

        # Select top 2 minima (global + best secondary)
        minima_to_plot = [imin]
        secondaries = [(mi, dchi2[mi]) for mi in local_mins if mi != imin and dchi2[mi] <= 9.0]
        secondaries.sort(key=lambda x: x[1])
        if secondaries:
            minima_to_plot.append(secondaries[0][0])

        spectra_this = {}
        for k, mi in enumerate(minima_to_plot):
            th23_i, dm31_i, th13_i = params[mi]
            dcp_i = dcp_vals[mi]

            # Re-minimize at this dCP to get accurate rates
            def obj(x):
                c2, _ = chi2_method_d_cpt_conserving(
                    lib, test_p, n_bins, truth_cache,
                    TRUE_TH12, x[2], x[0], dcp_i, TRUE_DM21, x[1])
                return c2
            x0 = np.array([th23_i, dm31_i, th13_i])
            res = minimize(obj, x0, method='Powell',
                           bounds=[(0.6, 1.1), (2.0e-3, 3.0e-3), (0.1, 0.2)],
                           options={'ftol': 1e-4, 'maxfev': 400})

            th23_f, dm31_f, th13_f = res.x
            truth_phys, fit_phys, fit_mod_phys = get_fit_rates_method_d(
                lib, test_p, n_bins, truth_cache, th23_f, dcp_i, dm31_f, th13_f)

            spectra_this[k] = {
                'truth_phys': truth_phys,
                'fit_phys': fit_phys,
                'fit_mod_phys': fit_mod_phys,
                'dcp_deg': float(dcp_deg[mi]),
                'chi2': float(chi2[mi]),
                'params': res.x.copy(),
            }
            print(f"    Min {k+1}: dCP={dcp_deg[mi]:+.1f} deg, chi2={chi2[mi]:.2f}, "
                  f"s2th23={np.sin(th23_f)**2:.4f}, dm31={dm31_f*1e3:.4f}e-3")

        all_spectra[panel_idx] = {
            'delta_true': delta_true,
            'spectra': spectra_this,
        }

    # ================================================================
    # Plot B: One figure per Delta value (2 rows x 3 cols)
    #   Col 0: FHC channel, Col 1: RHC channel, Col 2: FHC+RHC summed
    #   Row 0: appearance, Row 1: disappearance
    # Only for points with a double minimum
    # ================================================================
    SUMMED_NAMES = [
        r'$\nu_e$ appearance (FHC+RHC)',
        r'$\nu_\mu$ disappearance (FHC+RHC)',
    ]
    # Channel indices: app = (0=FHC, 1=RHC), dis = (2=FHC, 3=RHC)
    CHANNEL_LAYOUT = [
        # (row, col, channel_indices, title)
        (0, 0, [0], PHYSICAL_NAMES[0]),
        (0, 1, [1], PHYSICAL_NAMES[1]),
        (0, 2, [0, 1], SUMMED_NAMES[0]),
        (1, 0, [2], PHYSICAL_NAMES[2]),
        (1, 1, [3], PHYSICAL_NAMES[3]),
        (1, 2, [2, 3], SUMMED_NAMES[1]),
    ]

    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    def make_spectra_figure(spectra, delta_val, cpt_conserving_phys, in_window,
                            bc_window, CHANNEL_LAYOUT, output_dir, paper=False):
        """Generate spectra figure. paper=True for publication quality."""
        if paper:
            saved_rc = matplotlib.rcParams.copy()
            matplotlib.rcParams.update({
                'font.size': 16,
                'axes.labelsize': 18,
                'axes.titlesize': 17,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 13,
            })

        fig = plt.figure(figsize=(21, 14))
        # Leave bottom margin for legend
        gs_outer = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28,
                            bottom=0.15 if paper else 0.08)

        legend_handles = []
        for layout_row, layout_col, ch_indices, title in CHANNEL_LAYOUT:
            gs_inner = gs_outer[layout_row, layout_col].subgridspec(
                2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(gs_inner[0])
            ax_res = fig.add_subplot(gs_inner[1], sharex=ax_main)

            truth = sum(spectra[0]['truth_phys'][ch][in_window] for ch in ch_indices)
            cpt_cons = sum(cpt_conserving_phys[ch][in_window] for ch in ch_indices)

            # --- Main panel ---
            ax_main.step(bc_window, truth, where='mid', color='black', lw=2.5)
            ax_main.step(bc_window, cpt_cons, where='mid', color='#2ca02c', lw=2, ls='-.')

            colors_min = ['#d62728', '#1f77b4']
            ls_min = ['--', ':']
            lw_main = [2.0, 2.0]
            fit_mods = {}
            for k in sorted(spectra.keys()):
                s = spectra[k]
                fit_mod = sum(s['fit_mod_phys'][ch][in_window] for ch in ch_indices)
                fit_mods[k] = fit_mod
                ax_main.step(bc_window, fit_mod, where='mid', color=colors_min[k],
                             lw=lw_main[k], ls=ls_min[k])

            ax_main.set_xlim(0, 8)
            ax_main.set_title(title)
            ax_main.set_ylabel('Events / bin')
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # --- Residual panel ---
            res_cons = cpt_cons - truth
            ax_res.step(bc_window, res_cons, where='mid', color='#2ca02c', lw=2, ls='-.')
            for k in sorted(fit_mods.keys()):
                res_fit = fit_mods[k] - truth
                ax_res.step(bc_window, res_fit, where='mid', color=colors_min[k],
                            lw=2, ls=ls_min[k])
            ax_res.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
            ax_res.set_xlim(0, 8)
            ax_res.set_xlabel('Energy [GeV]')
            ax_res.set_ylabel('Residual')

        # Build legend handles (once, using spectra info)
        if not legend_handles:
            legend_handles.append(
                Line2D([0], [0], color='black', lw=2.5,
                       label=(r'CPT-violating truth ($\Delta = '
                              + f'{delta_val:+.2f}'
                              + r' \times 10^{-3}$ eV$^2$)')))
            legend_handles.append(
                Line2D([0], [0], color='#2ca02c', lw=2, ls='-.',
                       label=r'CPT-conserving truth ($\Delta = 0$)'))
            for k in sorted(spectra.keys()):
                s = spectra[k]
                s2th23 = np.sin(s['params'][0])**2
                dm31_k = s['params'][1] * 1e3
                legend_handles.append(
                    Line2D([0], [0], color=colors_min[k], lw=2, ls=ls_min[k],
                           label=(r'Best fit (min ' + f'{k+1}' + r'): '
                                  + r'$\delta_{CP} = ' + f'{s["dcp_deg"]:+.0f}' + r'^{\circ}$, '
                                  + r'$\sin^2\theta_{23} = ' + f'{s2th23:.3f}' + r'$, '
                                  + r'$\Delta m^2_{31} = ' + f'{dm31_k:.3f}'
                                  + r' \times 10^{-3}$ eV$^2$')))

        leg_fontsize = 15 if paper else 10
        fig.legend(handles=legend_handles, loc='lower center',
                   ncol=1, fontsize=leg_fontsize,
                   bbox_to_anchor=(0.5, -0.01),
                   frameon=True, edgecolor='gray', fancybox=False)

        tag = f'{delta_val:+.2f}'.replace('+', 'p').replace('-', 'n').replace('.', 'd')
        suffix = '_paper' if paper else ''
        fname = f'plot_B_spectra_delta_{tag}{suffix}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
        print(f"Saved {fname}")
        plt.close()

        if paper:
            matplotlib.rcParams.update(saved_rc)

    for col_idx in range(min(n_rep, 5)):
        data = all_spectra[col_idx]
        spectra = data['spectra']
        delta_val = data['delta_true'] * 1e3
        has_double = len(spectra) >= 2

        if not has_double:
            print(f"  Skipping Delta={delta_val:+.2f}e-3 (no double minimum)")
            continue

        # Diagnostic version
        make_spectra_figure(spectra, delta_val, cpt_conserving_phys, in_window,
                            bc_window, CHANNEL_LAYOUT, output_dir, paper=False)

        # Paper version for Delta ~ +0.76
        if abs(delta_val - 0.76) < 0.05:
            make_spectra_figure(spectra, delta_val, cpt_conserving_phys, in_window,
                                bc_window, CHANNEL_LAYOUT, output_dir, paper=True)

    # Cleanup
    lib.glbFreeParams(true_p)
    lib.glbFreeParams(test_p)
    lib.glbFreeParams(err_p)

    return all_spectra


# ============================================================================
# Phase 3: nuSQuIDS oscillation probabilities
# ============================================================================

def phase3_nusquids(phase1_data, output_dir):
    """Compute oscillation probabilities at minima. Plot C."""
    print("\n" + "=" * 70)
    print("Phase 3: nuSQuIDS Oscillation Probabilities")
    print("=" * 70)

    tasks = phase1_data['tasks']
    representatives = phase1_data['representatives']
    n_rep = min(len(representatives), 5)

    energies = np.linspace(0.5, 10.0, 200)

    all_probs = {}

    for panel_idx in range(n_rep):
        idx = representatives[panel_idx]
        t = tasks[idx]
        delta_true = t['delta_true']
        chi2 = t['chi2_best']
        dcp_vals = t['dcp_vals']
        dcp_deg = np.degrees(dcp_vals)
        params = t['best_params']

        imin = np.argmin(chi2)
        local_mins = find_local_minima(chi2, order=5)
        dchi2 = chi2 - chi2.min()

        minima_to_plot = [imin]
        secondaries = [(mi, dchi2[mi]) for mi in local_mins if mi != imin and dchi2[mi] <= 9.0]
        secondaries.sort(key=lambda x: x[1])
        if secondaries:
            minima_to_plot.append(secondaries[0][0])

        dm31_bar = TRUE_DM31 + delta_true

        print(f"\n  Delta = {delta_true*1e3:+.3f}e-3:")

        # Truth probabilities
        p_nu_truth = propagate(energies, 'nu', TRUE_DM31, TRUE_DCP_0, True)
        p_nubar_truth = propagate(energies, 'nubar', dm31_bar, TRUE_DCP_0, True)

        # DP_CPT decomposition at truth
        p_nubar_nodelta = propagate(energies, 'nubar', TRUE_DM31, TRUE_DCP_0, True)
        dp_cpt = p_nubar_nodelta - p_nubar_truth

        probs_this = {
            'p_nu_truth': p_nu_truth,
            'p_nubar_truth': p_nubar_truth,
            'dp_cpt': dp_cpt,
            'minima': {},
        }

        for k, mi in enumerate(minima_to_plot):
            th23_i, dm31_i, th13_i = params[mi]
            dcp_i = dcp_vals[mi]

            p_nu_fit = propagate_custom(energies, 'nu', dm31_i, dcp_i, th23_i, th13_i, True)
            p_nubar_fit = propagate_custom(energies, 'nubar', dm31_i, dcp_i, th23_i, th13_i, True)

            probs_this['minima'][k] = {
                'p_nu': p_nu_fit,
                'p_nubar': p_nubar_fit,
                'dcp_deg': float(dcp_deg[mi]),
                'params': (th23_i, dm31_i, th13_i),
            }
            print(f"    Min {k+1}: dCP={dcp_deg[mi]:+.1f} deg, "
                  f"s2th23={np.sin(th23_i)**2:.4f}, dm31={dm31_i*1e3:.4f}e-3")

        all_probs[panel_idx] = {
            'delta_true': delta_true,
            'probs': probs_this,
        }

    # ================================================================
    # Plot C: Oscillation probabilities (2 rows x 5 cols)
    # ================================================================
    fig, axes = plt.subplots(2, n_rep, figsize=(24, 8))
    if n_rep == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_rep):
        data = all_probs[col]
        probs = data['probs']
        delta_true = data['delta_true']

        # Top row: P(nu_mu -> nu_e)
        ax = axes[0, col]
        ax.plot(energies, probs['p_nu_truth'], 'k-', lw=1.5, label=r'Truth $\nu$')
        colors_min = ['#d62728', '#1f77b4']
        ls_min = ['--', ':']
        for k in sorted(probs['minima'].keys()):
            m = probs['minima'][k]
            label = f'Fit {k+1}: {m["dcp_deg"]:+.0f}' + r'$°$'
            ax.plot(energies, m['p_nu'], color=colors_min[k], ls=ls_min[k], lw=1.2, label=label)

        ax.set_xlim(0.5, 10)
        ax.set_ylabel(r'$P(\nu_\mu \to \nu_e)$')
        delta_val = delta_true * 1e3
        ax.set_title(f'$\\Delta = {delta_val:+.2f}' + r'$\times 10^{-3}$', fontsize=10)
        if col == 0:
            ax.legend(fontsize=7, loc='upper right')

        # Bottom row: P(nubar_mu -> nubar_e)
        ax = axes[1, col]
        ax.plot(energies, probs['p_nubar_truth'], 'k-', lw=1.5, label=r'Truth $\bar{\nu}$')
        for k in sorted(probs['minima'].keys()):
            m = probs['minima'][k]
            label = f'Fit {k+1}: {m["dcp_deg"]:+.0f}' + r'$°$'
            ax.plot(energies, m['p_nubar'], color=colors_min[k], ls=ls_min[k], lw=1.2, label=label)

        # DP_CPT overlay (gray fill)
        dp_cpt_scaled = probs['dp_cpt'] * 5  # scale for visibility
        ax.fill_between(energies, 0, dp_cpt_scaled, alpha=0.15, color='purple',
                        label=r'$\Delta P_{\rm CPT} \times 5$')

        ax.set_xlim(0.5, 10)
        ax.set_xlabel('Energy [GeV]')
        ax.set_ylabel(r'$P(\bar{\nu}_\mu \to \bar{\nu}_e)$')
        if col == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(r'(C) Oscillation Probabilities $P(\nu_\mu \to \nu_e)$ at Basin-Switch Minima'
                 '\nBlack = CPT-violating truth, Colored = CPT-conserving fit at each minimum',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot_C_oscillation_probs.png'),
                dpi=200, bbox_inches='tight')
    print("\nSaved plot_C_oscillation_probs.png")
    plt.close()

    return all_probs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Diagnose basin-switch in method-d CPT bias scan')
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing task_000.npz ... task_099.npz')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: results-dir/diagnostics)')
    parser.add_argument('--skip-globes', action='store_true',
                        help='Skip Phase 2 (GLoBES event spectra)')
    parser.add_argument('--skip-nusquids', action='store_true',
                        help='Skip Phase 3 (nuSQuIDS oscillation probabilities)')
    args = parser.parse_args()

    # Resolve to absolute paths up-front: phase2_globes() chdir's into the
    # committed .glb directory, after which relative results/output paths would
    # otherwise break.
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(results_dir, 'diagnostics'))
    os.makedirs(output_dir, exist_ok=True)

    print("Basin Switch Diagnostics — Method (d) CPT Bias Scan")
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")

    # Phase 1: always runs
    phase1_data = phase1_analysis(results_dir, output_dir)
    if phase1_data is None:
        print("Phase 1 failed — no results found. Exiting.")
        sys.exit(1)

    # Phase 2: GLoBES event spectra
    all_spectra = None
    if not args.skip_globes:
        all_spectra = phase2_globes(phase1_data, output_dir)
    else:
        print("\n  [Skipping Phase 2 — GLoBES event spectra]")

    # Phase 3: nuSQuIDS oscillation probabilities
    all_probs = None
    if not args.skip_nusquids:
        all_probs = phase3_nusquids(phase1_data, output_dir)
    else:
        print("\n  [Skipping Phase 3 — nuSQuIDS oscillation probabilities]")

    # Save diagnostic data
    save_dict = {
        'deltas': phase1_data['deltas'],
        'best_dcp_deg': phase1_data['best_dcp_deg'],
        'min_chi2': phase1_data['min_chi2'],
        'representatives': np.array(phase1_data['representatives']),
        'switch_indices': np.array(phase1_data['switch_indices']),
    }
    np.savez(os.path.join(output_dir, 'basin_switch_diagnostics.npz'), **save_dict)
    print(f"\nSaved basin_switch_diagnostics.npz")

    print("\nDone!")


if __name__ == "__main__":
    main()
