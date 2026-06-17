#!/usr/bin/env python3
"""
Combined ORCA-Full + IceCube-Upgrade-7 CPT sensitivity-scan row worker.

Paper figure: Fig 5 (bottom panel, combined future-sensitivity projection) and
Suppl Fig 4 (projection curve). Produces the per-row JSONs that
``assemble_combined_future_sensitivity.py`` collects into the
``combined_fine_31x31`` result directory
(``dm_grid.npy`` / ``delta_grid.npy`` / ``chi2_grid.npy`` / ``converged_grid.npy``).

Asimov sensitivity scan with both experiments combined; chi2 is summed across both
by PyNuFit. No HS systematics, no muon background. Each worker handles one row of
the CPT grid: fixed Dm231 (from --row), scans all Delta = Dm231 - Dm231_bar.

Production grid (Fig 5 bottom): ORCA-Full 5 yr + IC-Upgrade-7 10 yr, 31x31 fine grid,
Dm231 in [2.3, 2.7]e-3, Delta in [-0.6, 0.6]e-3 (set via the CLI overrides below).

Config / env / inputs:
- XML config: ``Combined_ORCAFull_ICUp_CPT.xml`` (resolved via ``paths.pynu_config``;
  the committed template expands ``${DATA_DIR}``).
- Exposures: pass ``--orca-exposure 5.0 --icup-exposure 10.0`` (ORCA-Full 5 yr,
  IC-Upgrade-7 10 yr); without overrides the XML's own exposures are used.
- Env: PYNU_DIR/PYNU (set by ``paths.add_pynu_to_path``), DATA_DIR, NUSQUIDS_DATA_PATH.

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/sensitivity-scans/combined-future/scripts/run_combined_future_cpt_row_worker.py``):
- ``_find_pynu_parent`` walk + ``/n/holylfs05/.../Pynu`` fallback -> ``paths.add_pynu_to_path()``.
- ``--config`` was a bare required arg supplied with a cluster path by the submit
  script; it now defaults to ``paths.pynu_config("Combined_ORCAFull_ICUp_CPT.xml")``.
- ``--row-idx`` (renamed to ``--row``, ``--row-idx`` kept as an alias) defaults to
  ``$SLURM_ARRAY_TASK_ID`` if present, so it runs locally and on a cluster.
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

CONFIG_NAME = "Combined_ORCAFull_ICUp_CPT.xml"


def env_task_default():
    """Default --row from $SLURM_ARRAY_TASK_ID if present (None otherwise)."""
    v = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(v) if v not in (None, "") else None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Pynu XML config (default: resolved Combined_ORCAFull_ICUp_CPT.xml)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row", "--row-idx", dest="row", type=int,
                        default=env_task_default(),
                        help="Row index (Dm231 index); defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-delta", type=int, default=41)
    parser.add_argument("--dm-min", type=float, default=2.0e-3)
    parser.add_argument("--dm-max", type=float, default=3.0e-3)
    parser.add_argument("--delta-min", type=float, default=-1.0e-3)
    parser.add_argument("--delta-max", type=float, default=1.0e-3)
    parser.add_argument("--orca-exposure", type=float, default=None,
                        help="Override ORCA-Full exposure (years)")
    parser.add_argument("--icup-exposure", type=float, default=None,
                        help="Override ICUpgrade exposure (years)")
    args = parser.parse_args()

    if args.row is None:
        parser.error("--row is required (or set $SLURM_ARRAY_TASK_ID)")
    if args.config is None:
        args.config = paths.pynu_config(CONFIG_NAME)
    elif not os.path.exists(args.config):
        # A bare config name (e.g. "Combined_ORCAFull_ICUp_CPT.xml") -> resolve
        # from configs/pynu/ with ${DATA_DIR} expanded. An existing path is used as-is.
        args.config = paths.pynu_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    delta_grid = np.linspace(args.delta_min, args.delta_max, args.n_delta)

    row_idx = args.row
    dm_val = dm_grid[row_idx]

    print(f"=== Combined Future CPT Row Worker: row {row_idx}/{args.n_dm}, Dm231={dm_val:.4e} ===")
    print(f"Delta: {args.n_delta} pts in [{args.delta_min:.1e}, {args.delta_max:.1e}]")
    print(f"Config: {args.config}")
    print(f"Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit = PyNuFit(args.config, verbosity=False)

    # Override per-experiment exposure if requested
    for exp_name, exp in pynufit.Experiments.items():
        override = None
        if args.orca_exposure is not None and 'ORCA' in exp_name:
            override = args.orca_exposure
        elif args.icup_exposure is not None and 'ICUpgrade' in exp_name:
            override = args.icup_exposure
        if override is not None:
            mc_exp = exp.TotalMCexposure
            exp.FitExposure = override
            exp.NORM = (override / mc_exp) * 1e4 * exp.SECONDS_PER_YEAR
            exp.BaseWeight = exp.Weight * exp.NORM

    print(f"  Experiments: {list(pynufit.Experiments.keys())}")
    for exp_name, exp in pynufit.Experiments.items():
        print(f"  {exp_name}: exposure={exp.FitExposure}yr, MCexposure={exp.TotalMCexposure}yr")

    # Set truth parameters (CPT symmetric: Dm231 = Dm231_bar)
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
    for exp_name, obs in asimov.items():
        print(f"  Asimov events ({exp_name}): {obs.sum():.1f}")
    total_events = sum(v.sum() for v in asimov.values())
    print(f"  Total Asimov events: {total_events:.1f}")

    # Likelihood
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for k in asimov:
        pynufit.LLH.observation[k] = asimov[k]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Nuisance setup
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    print(f"  Nuisance parameters ({len(nominal)}): {pynufit.Analysis.NuisanceList}")

    chi2_row = np.full(args.n_delta, np.nan)
    converged_row = np.zeros(args.n_delta, dtype=bool)
    x0 = nominal.copy()

    t_start = datetime.now()

    for j, delta in enumerate(delta_grid):
        dm231_bar = dm_val - delta

        # Skip if dm231_bar is non-physical
        if dm231_bar <= 0:
            print(f"  [{row_idx},{j:2d}] delta={delta:.4e}: SKIP (dm231_bar={dm231_bar:.4e} <= 0)")
            continue

        pynufit.StartPhysics()
        pynufit.StartNuisance()

        for name, pt in pynufit.physics_tunes.items():
            pt.OscillationTunes.Parameters["Dm231"] = dm_val
            pt.OscillationTunes.Parameters["Dm231_bar"] = dm231_bar
            if hasattr(pt.OscillationTunes, 'reset_cache'):
                pt.OscillationTunes.reset_cache()

        pynufit.ApplyOscillations("Physics")
        pynufit.ApplyNuisanceWeights(nominal)
        pynufit.SetExpectedWeights()
        pynufit.SetBinnedExpectedEvents()
        pynufit.SetBinnedMCVariance()

        def objective(nuisance):
            pynufit.StartNuisance()
            pynufit.ApplyNuisanceWeights(nuisance)
            pynufit.SetExpectedWeights()
            pynufit.SetBinnedExpectedEvents()
            pynufit.SetBinnedMCVariance()  # refresh sigma^2_MC at the current point
            return pynufit.LLH.stats_and_systematics(
                pynufit.Expectation, nuisance, pynufit.MCVariance
            )

        # No analytical gradient — multi-experiment gradient has shape mismatch
        result = minimize(
            objective, x0,
            method='L-BFGS-B', jac=None, bounds=bounds,
            options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
        )

        chi2_row[j] = result.fun
        converged_row[j] = result.success
        if result.success:
            x0 = result.x.copy()

        print(f"  [{row_idx},{j:2d}] delta={delta:.4e} dm231_bar={dm231_bar:.4e}: "
              f"chi2={result.fun:.4f}, conv={result.success}")

    total_time = (datetime.now() - t_start).total_seconds()

    row_data = {
        'row_idx': row_idx,
        'dm231': float(dm_val),
        'chi2': chi2_row.tolist(),
        'converged': converged_row.tolist(),
        'total_time_s': total_time,
        'delta_range': [float(args.delta_min), float(args.delta_max)],
    }

    row_file = os.path.join(args.output_dir, f'row_{row_idx:03d}.json')
    with open(row_file, 'w') as f:
        json.dump(row_data, f, indent=2)

    print(f"Row {row_idx} done: min_chi2={np.nanmin(chi2_row):.4f}, "
          f"conv={converged_row.sum()}/{args.n_delta}, time={total_time:.1f}s")
    print(f"Saved: {row_file}")


if __name__ == "__main__":
    main()
