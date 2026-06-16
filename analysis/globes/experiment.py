"""Shared method-d CPT-truth-bias machinery, parameterized over experiment.

Ported from:
    claude/3-CPT-violation/CP-CPT-degeneracy/_common/experiment.py

Used by the DUNE (Fig 4) and NOvA (Suppl Fig 1) band-plot runners. Extracted so
DUNE, NOvA, and future experiments share one implementation. The DUNE layout
(8 rules, paired by polarity for each of 4 physical channels) and the NOvA
layout (4 rules, each already a complete single-polarity physical channel) are
both supported via the ExperimentConfig.physical_pairs field.

Polarity classification:
  nu_rules     rules whose signal+bg channels use the '+' polarity (ν propagation)
  nubar_rules  rules whose signal+bg channels use the '-' polarity (ν̄ propagation)
  physical_pairs  tuples of rule indices whose rates sum to one physical
                  observable channel. Each tuple may contain 1 rule (NOvA-style:
                  each rule is a full physical channel) or 2 rules (DUNE-style:
                  one ν contribution + one ν̄ contribution per channel).

The normalization nuisance parameters (a_sig, a_bg) are applied to each
physical channel independently, then profiled per channel. sig_norm_error and
bg_norm_error are per-physical-channel (len == len(physical_pairs)).

Env / inputs:
  GLOBES_PREFIX — install prefix of GLoBES (read by globes_wrapper.load_globes).
  The worker mains must be invoked from a CWD containing the experiment's .glb
  file (GLoBES resolves the .glb's include directives relative to CWD); the
  runner shims chdir there using paths.globes_config(...).

De-hardcode note:
  The original bootstrapped sys.path to a sibling cluster directory
  (``../../../5-DUNE-GLoBES/scripts``) and imported the GLoBES ctypes surface
  from ``plot_dune_sensitivity``. That import is now ``from
  analysis.globes.globes_wrapper import ...``; no path bootstrap is needed
  because this module lives inside the ``analysis`` package.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from analysis.globes.globes_wrapper import (  # noqa: E402
    load_globes, set_params, GLB_ALL,
    TRUE_TH12, TRUE_TH13, TRUE_TH23, TRUE_DM21, TRUE_DM31,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    glb_file: str
    expected_n_rules: int
    nu_rules: List[int]
    nubar_rules: List[int]
    physical_pairs: List[Tuple[int, ...]]
    sig_norm_error: List[float]
    bg_norm_error: List[float]

    def __post_init__(self):
        n_phys = len(self.physical_pairs)
        assert len(self.sig_norm_error) == n_phys, \
            f"{self.name}: sig_norm_error len {len(self.sig_norm_error)} != {n_phys}"
        assert len(self.bg_norm_error) == n_phys, \
            f"{self.name}: bg_norm_error len {len(self.bg_norm_error)} != {n_phys}"
        all_rules = sorted(set(self.nu_rules + self.nubar_rules))
        pair_rules = sorted({r for pair in self.physical_pairs for r in pair})
        assert all_rules == pair_rules, \
            f"{self.name}: nu/nubar rules {all_rules} must match physical_pairs rules {pair_rules}"


def init_experiment_cpt(lib, cfg: ExperimentConfig, init_tag: bytes = b"cpt_method_d"):
    """Initialize GLoBES with the experiment's .glb and assert rule count.

    Must be called from a CWD containing `cfg.glb_file` (GLoBES resolves the
    .glb's include directives relative to CWD).
    """
    lib.glbInit(init_tag)
    num_exps = ctypes.c_int.in_dll(lib, "glb_num_of_exps")
    exp_list_addr = ctypes.addressof(ctypes.c_void_p.in_dll(lib, "glb_experiment_list"))
    lib.glbInitExperiment(cfg.glb_file.encode(), exp_list_addr, ctypes.byref(num_exps))
    n_rules = lib.glbGetNumberOfRules(0)
    assert n_rules == cfg.expected_n_rules, \
        f"{cfg.name}: got {n_rules} rules, expected {cfg.expected_n_rules}"
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


def generate_cpt_truth(lib, cfg: ExperimentConfig, true_p, n_bins,
                       delta_true, truth_dcp):
    """Generate CPT-violating Asimov truth rates.

    Since glbSetRates() sets truth for ALL rules at once, we call it twice:
      1. With dm31 (Δm²_ν) → extract and save nu-rule truth rates
      2. With dm31_bar (Δm²_ν̄ = dm31 + delta_true) → extract and save nubar-rule truth rates
    """
    dm31_bar = TRUE_DM31 + delta_true
    truth_cache = {}

    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, truth_dcp, TRUE_DM21, TRUE_DM31)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()
    for r in cfg.nu_rules:
        truth_cache[r] = extract_truth_sig_bg(lib, n_bins, r)

    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, truth_dcp, TRUE_DM21, dm31_bar)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()
    for r in cfg.nubar_rules:
        truth_cache[r] = extract_truth_sig_bg(lib, n_bins, r)

    # Restore CPT-conserving state for downstream fit-rate computations.
    set_params(lib, true_p, TRUE_TH12, TRUE_TH13, TRUE_TH23, truth_dcp, TRUE_DM21, TRUE_DM31)
    lib.glbSetOscillationParameters(true_p)
    lib.glbSetCentralValues(true_p)
    lib.glbSetRates()

    return truth_cache


def chi2_method_d_cpt_conserving(lib, cfg: ExperimentConfig, test_p, n_bins,
                                 truth_cache, th12, th13, th23, dcp, dm21, dm31_test):
    """Method-(d) chi² with CPT-conserving fit (single dm31 for all rules)."""
    set_params(lib, test_p, th12, th13, th23, dcp, dm21, dm31_test)
    lib.glbChiSys(test_p, 0, 0)  # triggers full rate recomputation
    fit_all = {r: extract_sig_bg(lib, n_bins, r) for r in range(cfg.expected_n_rules)}

    total_chi2 = 0.0
    for pair_idx, rules in enumerate(cfg.physical_pairs):
        fit_sig_combined = np.zeros(n_bins)
        fit_bg_combined = np.zeros(n_bins)
        truth_combined = np.zeros(n_bins)
        for r in rules:
            f_sig, f_bg = fit_all[r]
            t_sig, t_bg = truth_cache[r]
            fit_sig_combined += f_sig
            fit_bg_combined += f_bg
            truth_combined += t_sig + t_bg

        sig_err = cfg.sig_norm_error[pair_idx]
        bg_err = cfg.bg_norm_error[pair_idx]

        def objective(x, _sig=fit_sig_combined, _bg=fit_bg_combined,
                      _truth=truth_combined, _se=sig_err, _be=bg_err):
            a_sig, a_bg = x
            fit_mod = (1 + a_sig) * _sig + (1 + a_bg) * _bg
            return poisson_chi2_np(_truth, fit_mod) + (a_sig / _se) ** 2 + (a_bg / _be) ** 2

        res = minimize(objective, [0.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-4, 'maxfev': 200})
        total_chi2 += res.fun

    return total_chi2


def scan_dcp_warmstarted(lib, cfg: ExperimentConfig, test_p, n_bins,
                         truth_cache, dcp_vals, forward=True):
    """Warm-started bidirectional dCP scan, profiling (th23, dm31, th13)."""
    n = len(dcp_vals)
    chi2_arr = np.zeros(n)
    params_arr = np.zeros((n, 3))
    x0 = np.array([TRUE_TH23, TRUE_DM31, TRUE_TH13])
    indices = range(n) if forward else range(n - 1, -1, -1)

    for i in indices:
        dcp = dcp_vals[i]

        def obj(x):
            return chi2_method_d_cpt_conserving(
                lib, cfg, test_p, n_bins, truth_cache,
                TRUE_TH12, x[2], x[0], dcp, TRUE_DM21, x[1])

        res = minimize(obj, x0, method='Powell',
                       bounds=[(0.6, 1.1), (2.0e-3, 3.0e-3), (0.1, 0.2)],
                       options={'ftol': 1e-4, 'maxfev': 800})
        chi2_arr[i] = res.fun
        params_arr[i] = res.x
        x0 = res.x.copy()

    return chi2_arr, params_arr


def find_region(dcp_deg, dchi2, threshold):
    below = dchi2 <= threshold
    if not np.any(below):
        return (float('nan'), float('nan'))
    imin = int(np.argmin(dchi2))
    lo = imin
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = imin
    while hi < len(below) - 1 and below[hi + 1]:
        hi += 1
    return (float(dcp_deg[lo]), float(dcp_deg[hi]))


# ---------------------------------------------------------------------------
# Worker entry points — shared by DUNE and NOvA script shims.
# ---------------------------------------------------------------------------

def _setup_globes_and_priors(cfg: ExperimentConfig):
    lib = load_globes()
    n_rules, n_bins = init_experiment_cpt(lib, cfg)
    true_p = lib.glbAllocParams()
    test_p = lib.glbAllocParams()
    err_p = lib.glbAllocParams()
    set_params(lib, err_p,
               TRUE_TH12 * 0.024, TRUE_TH13 * 0.03, 0.0, 0.0,
               TRUE_DM21 * 0.024, 0.0)
    lib.glbSetDensityParams(err_p, 0.05, GLB_ALL)
    lib.glbSetInputErrors(err_p)
    return lib, n_rules, n_bins, true_p, test_p, err_p


def _free_params(lib, *params):
    for p in params:
        lib.glbFreeParams(p)


def run_band_plot_main(cfg: ExperimentConfig, delta_values, n_dcp: int = 201):
    """Band-plot worker: one SLURM task processes one delta_true index.

    Replaces the per-experiment main() in run_dcp_scan_method_d.py.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('task_id', nargs='?', default=None, type=int)
    parser.add_argument('--truth-dcp-deg', type=float, default=0.0,
                        help='Truth delta_CP in degrees (default: 0)')
    parser.add_argument('--delta', type=float, default=None,
                        help='Override delta_true value in eV^2 (bypass grid)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.delta is not None:
        delta_true = args.delta
        task_id = -1
    else:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',
                                     args.task_id if args.task_id is not None else 0))
        delta_true = float(delta_values[task_id])

    truth_dcp = math.radians(args.truth_dcp_deg)

    print(f"[{cfg.name}] Task {task_id}/{len(delta_values)-1}: "
          f"Delta_true = {delta_true:+.4e} eV^2")
    print(f"Truth: dCP={args.truth_dcp_deg:.1f} deg, dm31={TRUE_DM31:.4e}, "
          f"dm31_bar={TRUE_DM31+delta_true:.4e}")
    print(f"Fit: CPT conserving (single dm31), scan dCP, profile th23+dm31+th13")
    print(f"Chi2: Method (d) — combined spectra + shared nuisance profiling")

    lib, n_rules, n_bins, true_p, test_p, err_p = _setup_globes_and_priors(cfg)
    print(f"[{cfg.name}] {n_rules} rules, {n_bins} bins")

    print("Generating CPT-violating Asimov truth...")
    truth_cache = generate_cpt_truth(lib, cfg, true_p, n_bins, delta_true, truth_dcp)
    for r in range(cfg.expected_n_rules):
        sig, bg = truth_cache[r]
        print(f"  Rule {r}: sig={sig.sum():.1f}, bg={bg.sum():.1f}, total={sig.sum()+bg.sum():.1f}")

    chi2_at_truth_dm31 = chi2_method_d_cpt_conserving(
        lib, cfg, test_p, n_bins, truth_cache,
        TRUE_TH12, TRUE_TH13, TRUE_TH23, truth_dcp, TRUE_DM21, TRUE_DM31)
    print(f"Chi2 at truth dm31 (CPT-conserving fit): {chi2_at_truth_dm31:.4f}")
    print(f"  (Should be >0 for delta_true != 0)")

    dcp_vals = np.linspace(-math.pi, math.pi, n_dcp)
    print("Forward scan...")
    chi2_fwd, params_fwd = scan_dcp_warmstarted(
        lib, cfg, test_p, n_bins, truth_cache, dcp_vals, forward=True)
    print("Backward scan...")
    chi2_bwd, params_bwd = scan_dcp_warmstarted(
        lib, cfg, test_p, n_bins, truth_cache, dcp_vals, forward=False)

    chi2_best = np.minimum(chi2_fwd, chi2_bwd)
    dcp_deg = np.degrees(dcp_vals)
    dchi2 = chi2_best - chi2_best.min()
    imin = int(np.argmin(chi2_best))

    regions = {}
    for nsig, thresh in [(1, 1.0), (2, 4.0), (3, 9.0)]:
        regions[str(nsig)] = find_region(dcp_deg, dchi2, thresh)

    sig1_w = regions['1'][1] - regions['1'][0]
    maxdiff = float(np.max(np.abs(chi2_fwd - chi2_bwd)))
    bias = dcp_deg[imin] - args.truth_dcp_deg
    print(f"\nResults:")
    print(f"  best dCP = {dcp_deg[imin]:+.1f} deg (bias from truth: {bias:+.1f} deg)")
    print(f"  min chi2 = {chi2_best.min():.2f}")
    print(f"  1sig = [{regions['1'][0]:.0f}, {regions['1'][1]:.0f}] (w={sig1_w:.0f} deg)")
    print(f"  |fwd-bwd| max = {maxdiff:.2f}")

    result = {
        'experiment': cfg.name,
        'task_id': task_id,
        'delta_true': float(delta_true),
        'truth_dcp_deg': float(args.truth_dcp_deg),
        'min_chi2': float(chi2_best.min()),
        'best_dcp_deg': float(dcp_deg[imin]),
        'bias_deg': float(bias),
        'regions': regions,
    }
    if task_id >= 0:
        tag = f'task_{task_id:03d}'
    else:
        tag = f'task_delta_{delta_true:+.4e}'.replace('+', 'p').replace('-', 'n').replace('.', 'd')
    json_path = os.path.join(args.output_dir, f'{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    npz_path = os.path.join(args.output_dir, f'{tag}.npz')
    np.savez(npz_path, chi2_best=chi2_best, dcp_vals=dcp_vals,
             delta_true=delta_true, best_params=params_fwd)
    print(f"Saved {json_path}, {npz_path}")

    _free_params(lib, true_p, test_p, err_p)
    print("Done!")


def fit_one_point(lib, cfg: ExperimentConfig, true_p, test_p, n_bins,
                  dcp_true_deg, delta_true, n_dcp=201):
    """Generate CPT-violating truth at this point and return best-fit dCP."""
    truth_dcp = math.radians(dcp_true_deg)
    truth_cache = generate_cpt_truth(lib, cfg, true_p, n_bins, delta_true, truth_dcp)

    dcp_vals = np.linspace(-math.pi, math.pi, n_dcp)
    chi2_fwd, _ = scan_dcp_warmstarted(lib, cfg, test_p, n_bins, truth_cache,
                                       dcp_vals, forward=True)
    chi2_bwd, _ = scan_dcp_warmstarted(lib, cfg, test_p, n_bins, truth_cache,
                                       dcp_vals, forward=False)
    chi2_best = np.minimum(chi2_fwd, chi2_bwd)

    dcp_deg = np.degrees(dcp_vals)
    imin = int(np.argmin(chi2_best))
    dchi2 = chi2_best - chi2_best.min()

    return {
        'dcp_bestfit_deg': float(dcp_deg[imin]),
        'chi2_min': float(chi2_best.min()),
        'sig1_region': list(find_region(dcp_deg, dchi2, 1.0)),
        'sig3_region': list(find_region(dcp_deg, dchi2, 9.0)),
        'fwd_bwd_max_diff': float(np.max(np.abs(chi2_fwd - chi2_bwd))),
    }


def _load_csv_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
        assert header[0].strip() == 'dcp_true_deg' and header[1].strip() == 'dDel_true_eV2', \
            f'Unexpected CSV header: {header}'
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def run_manifold_point_main(cfg: ExperimentConfig, n_dcp: int = 201):
    """Grid/manifold worker: one SLURM task processes one (dcp_true, delta) row.

    Replaces the per-experiment main() in run_manifold_bestfit_dcp.py.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='CSV with columns dcp_true_deg, dDel_true_eV2')
    parser.add_argument('output_dir', help='Where to write per-task JSONs')
    parser.add_argument('task_id', nargs='?', default=None, type=int,
                        help='Override SLURM_ARRAY_TASK_ID')
    parser.add_argument('--task-offset', type=int,
                        default=int(os.environ.get('TASK_OFFSET', 0)),
                        help='Add to SLURM_ARRAY_TASK_ID (lets one CSV span multiple arrays).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = _load_csv_rows(args.csv_path)

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',
                                 args.task_id if args.task_id is not None else 0))
    task_id += args.task_offset
    if not (0 <= task_id < len(rows)):
        raise IndexError(f'task_id {task_id} out of range [0, {len(rows)-1}]')

    dcp_true_deg, delta_true = rows[task_id]
    print(f'[{cfg.name}] Task {task_id}/{len(rows)-1}: dcp_true={dcp_true_deg:+.2f} deg, '
          f'delta_true={delta_true:+.4e} eV^2')

    lib, n_rules, n_bins, true_p, test_p, err_p = _setup_globes_and_priors(cfg)

    result = fit_one_point(lib, cfg, true_p, test_p, n_bins,
                           dcp_true_deg, delta_true, n_dcp=n_dcp)
    result['experiment'] = cfg.name
    result['task_id'] = task_id
    result['dcp_true_deg'] = float(dcp_true_deg)
    result['dDel_true_eV2'] = float(delta_true)

    print(f'  best-fit dCP = {result["dcp_bestfit_deg"]:+.2f} deg, '
          f'chi2_min = {result["chi2_min"]:.3f}')

    out_path = os.path.join(args.output_dir, f'point_{task_id:04d}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved {out_path}')

    _free_params(lib, true_p, test_p, err_p)
