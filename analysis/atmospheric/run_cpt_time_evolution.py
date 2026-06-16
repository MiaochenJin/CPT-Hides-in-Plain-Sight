#!/usr/bin/env python3
"""
CPT time-evolution study (per-experiment Δχ² vs. exposure).

Paper figure: Fig 7 (CPT significance vs. year). For a single-experiment config,
evaluates the Δχ² profile over delta = Dm231 - Dm231_bar at multiple exposure
values. This reveals whether sensitivity scales linearly with livetime
(Poisson-dominated) or plateaus (MC-statistics-limited via Barlow-Beeston).

Run it once per experiment; the Fig-7 plotter combines the two JSONs:
- ``--config <ORCAFullEvtMC_Atm_CPT.xml>``  -> e.g. ``orcafull_evtmc_bb.json``
- ``--config <ICUp_Atm_CPT.xml>``           -> e.g. ``icupgrade_bb.json``
(Both XMLs are committed under ``configs/pynu/``; the names are passed to
``--config`` either directly as files or as a resolved-config name; see below.)

Config / env / inputs:
- ``--config`` is required: either an absolute XML path or, more portably, pass the
  bare config name (e.g. ``ORCAFullEvtMC_Atm_CPT.xml``) and it is resolved via
  ``paths.pynu_config`` (``${DATA_DIR}`` expanded).
- ``--output`` JSON path (one per experiment / mode).
- Env: PYNU_DIR/PYNU (set by ``paths.add_pynu_to_path``), DATA_DIR, NUSQUIDS_DATA_PATH.

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/sensitivity-scans/ORCA-full/scripts/run_cpt_time_evolution.py``):
- ``_find_pynu_parent`` walk + ``/n/holylfs05/.../Pynu`` fallback -> ``paths.add_pynu_to_path()``.
- ``--config`` previously took only an absolute path (supplied by the cluster submit
  script); it now also accepts a bare config name resolved via ``paths.pynu_config``.
  No other path was hardcoded (output is already an argparse arg).
"""

import sys
import os
import json
import numpy as np
import copy
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize

# --- repo bootstrap: make `analysis.lib` and `pynu` importable -------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths
paths.add_pynu_to_path()

from pynu import PyNuFit


TRUTH_DM = 2.511e-3
TRUTH_S23 = 0.572


def resolve_config(config):
    """Accept either an existing XML file path or a bare config name.

    A bare name (no directory separator, not an existing file) is resolved via
    ``paths.pynu_config`` so ``${DATA_DIR}`` placeholders get expanded.
    """
    if os.path.isfile(config):
        return config
    if os.sep not in config and "/" not in config:
        return paths.pynu_config(config)
    return config


def evaluate_delta_profile(pynufit, dm231, delta_values, exposure, no_mc_variance=False):
    """Evaluate chi2 at each delta value for a given exposure.

    Returns chi2 array and convergence array.
    Asimov is generated at delta=0 (CPT conserving) for this exposure.
    """
    # Override exposure
    for exp_name, exp in pynufit.Experiments.items():
        mc_exp = exp.TotalMCexposure
        exp.FitExposure = exposure
        exp.NORM = (exposure / mc_exp) * 1e4 * exp.SECONDS_PER_YEAR
        exp.BaseWeight = exp.Weight * exp.NORM

    # Set truth: CPT conserving (delta=0)
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_S23
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
        pt.OscillationTunes.Parameters["Dm231_bar"] = TRUTH_DM
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    # Generate Asimov at truth
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    asimov = copy.deepcopy(pynufit.Expectation)
    for k, v in asimov.items():
        if k in pynufit.MuonBackground and pynufit.MuonBackground[k] is not None:
            mu_counts, _ = pynufit.MuonBackground[k]
            asimov[k] = v + mu_counts

    # Set up likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for k in asimov:
        pynufit.LLH.observation[k] = asimov[k]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    if not no_mc_variance:
        pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    mc_var = None if no_mc_variance else pynufit.MCVariance

    # Nuisance setup
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    chi2_arr = np.full(len(delta_values), np.nan)
    conv_arr = np.zeros(len(delta_values), dtype=bool)
    x0 = nominal.copy()

    for j, delta in enumerate(delta_values):
        dm231_bar = TRUTH_DM - delta

        if dm231_bar <= 0:
            continue

        pynufit.StartPhysics()
        pynufit.StartNuisance()

        for name, pt in pynufit.physics_tunes.items():
            pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
            pt.OscillationTunes.Parameters["Dm231_bar"] = dm231_bar
            if hasattr(pt.OscillationTunes, 'reset_cache'):
                pt.OscillationTunes.reset_cache()

        pynufit.ApplyOscillations("Physics")
        pynufit.ApplyNuisanceWeights(nominal)
        pynufit.SetExpectedWeights()
        pynufit.SetBinnedExpectedEvents()
        if not no_mc_variance:
            pynufit.SetBinnedMCVariance()
            mc_var = pynufit.MCVariance

        def objective(nuisance):
            nonlocal mc_var
            pynufit.StartNuisance()
            pynufit.ApplyNuisanceWeights(nuisance)
            pynufit.SetExpectedWeights()
            pynufit.SetBinnedExpectedEvents()
            if not no_mc_variance:
                pynufit.SetBinnedMCVariance()  # refresh sigma^2_MC at the current point
                mc_var = pynufit.MCVariance
            return pynufit.LLH.stats_and_systematics(
                pynufit.Expectation, nuisance, mc_var
            )

        result = minimize(
            objective, x0,
            method='L-BFGS-B', jac=None, bounds=bounds,
            options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
        )

        chi2_arr[j] = result.fun
        conv_arr[j] = result.success
        if result.success:
            x0 = result.x.copy()

    return chi2_arr, conv_arr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="XML path, or a bare config name resolved via paths.pynu_config "
                             "(e.g. ORCAFullEvtMC_Atm_CPT.xml or ICUp_Atm_CPT.xml)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposures", nargs='+', type=float,
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    parser.add_argument("--n-delta", type=int, default=41)
    parser.add_argument("--delta-min", type=float, default=-1.0e-3)
    parser.add_argument("--delta-max", type=float, default=1.0e-3)
    parser.add_argument("--no-mc-variance", action="store_true")
    args = parser.parse_args()

    config = resolve_config(args.config)

    delta_values = np.linspace(args.delta_min, args.delta_max, args.n_delta)

    print(f"=== CPT Time Evolution Study ===")
    print(f"Config: {config}")
    print(f"Exposures: {args.exposures}")
    print(f"Delta points: {args.n_delta} in [{args.delta_min:.1e}, {args.delta_max:.1e}]")
    print(f"MC variance: {'disabled' if args.no_mc_variance else 'enabled (BB)'}")
    print(f"Started: {datetime.now().isoformat()}")

    # Initialize once
    pynufit = PyNuFit(config, verbosity=False)

    results = {
        'config': config,
        'delta_values': delta_values.tolist(),
        'exposures': [],
        'no_mc_variance': args.no_mc_variance,
    }

    for exp_val in args.exposures:
        print(f"\n--- Exposure: {exp_val} yr ---")
        t0 = datetime.now()

        chi2, conv = evaluate_delta_profile(
            pynufit, TRUTH_DM, delta_values, exp_val,
            no_mc_variance=args.no_mc_variance
        )

        dchi2 = chi2 - np.nanmin(chi2)
        dt = (datetime.now() - t0).total_seconds()

        results['exposures'].append({
            'exposure_yr': exp_val,
            'chi2': chi2.tolist(),
            'dchi2': dchi2.tolist(),
            'converged': conv.tolist(),
            'min_chi2': float(np.nanmin(chi2)),
            'time_s': dt,
        })

        # Report key delta values
        for target_delta in [0.05e-3, 0.1e-3, 0.2e-3, 0.5e-3]:
            idx = np.argmin(np.abs(delta_values - target_delta))
            if not np.isnan(dchi2[idx]):
                print(f"  Δχ²(δ={target_delta*1e3:.2f}e-3) = {dchi2[idx]:.4f} "
                      f"({np.sqrt(dchi2[idx]):.2f}σ)")

        print(f"  Time: {dt:.1f}s, Conv: {conv.sum()}/{len(conv)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
