#!/usr/bin/env python3
"""
Combined ORCA-6 + IC DeepCore CPT 3D data-fit grid-scan row worker.

Paper figure: Fig 5 (top panel, combined data-fit contours) and Suppl Fig 4
(combined data-fit 1D profile). Produces the per-row JSONs that
``assemble_combined_cpt_datafit.py`` collects into the
``combined_cpt_3d_datafit_41x41x20`` result directory
(``dm_grid.npy`` / ``delta_grid.npy`` / ``delta_chi2_grid.npy`` /
``chi2_profiled.npy`` / ``s23_profiled.npy``).

3D grid scan over (Dm231, Delta, Sin2Theta23); theta23 is profiled post-hoc by
the assembler. Fits real data from both ORCA and IC DeepCore simultaneously.

Each worker handles one Dm231 row:
- Fixed Dm231 value (determined by --row)
- Outer loop: Sin2Theta23 values (--n-s23 points in [--s23-min, --s23-max])
- Inner loop: Delta values (--n-grid points in [--delta-min, --delta-max])
- At each (Dm231, Delta, S23) point, minimize over nuisance parameters only

Special handling:
- ORCA: standard event-level nuisance weights (7 det + muon_norm)
- IC DeepCore: hypersurface (HS) corrections at binned-histogram level (5 HS + muon_norm)
- Shared: 7 flux systematics applied at event level to both experiments

HS + CPT interaction:
  HS slopes are indexed by a single deltam31. In CPT mode (Dm231 != Dm231_bar),
  we use Dm231 (neutrino) for HS interpolation.

No analytical gradient — uses L-BFGS-B with finite differences (HS breaks gradient chain).

Config / env / inputs:
- XML config: ``Combined_ORCA_IC_CPT_datafit.xml`` (resolved via
  ``paths.pynu_config``; the committed template expands ``${DATA_DIR}``).
- Hypersurface CSVs: ``--hs-dir`` pointing at the ``hs_*.csv`` files
  (under ``$DATA_DIR``; defaults to ``$DATA_DIR/Pynu/data/IceCube``).
- Env: PYNU_DIR/PYNU (Pynu framework, set by ``paths.add_pynu_to_path``),
  DATA_DIR, NUSQUIDS_DATA_PATH (nuSQuIDS propagation tables).

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/data-fits/combined/scripts/run_combined_cpt_3d_row_worker.py``):
- ``_find_pynu_parent`` walk + ``/n/holylfs05/.../Pynu`` fallback -> ``paths.add_pynu_to_path()``.
- ``PROJECT_DIR = '/n/holylfs05/.../AtmNuDataFit'`` default config path
  -> ``paths.pynu_config("Combined_ORCA_IC_CPT_datafit.xml")``.
- ``DEFAULT_HS_DIR`` rooted at the cluster ``PROJECT_DIR`` -> ``$DATA_DIR/Pynu/data/IceCube``.
- ``--row-idx`` (renamed to ``--row``, with ``--row-idx`` kept as an alias) now
  defaults to ``$SLURM_ARRAY_TASK_ID`` if present, so it runs locally and on a cluster.
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize

# --- repo bootstrap: make `analysis.lib` and `pynu` importable -------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths
paths.add_pynu_to_path()

from pynu import PyNuFit
from pynu.Experiments.ICDeepCore import ICDeepCore_Atm

# Grid defaults
DM_MIN = 2.0e-3
DM_MAX = 3.0e-3
TRUTH_DM = 2.511e-3
TRUTH_THETA = 0.572

CONFIG_NAME = "Combined_ORCA_IC_CPT_datafit.xml"


def default_hs_dir():
    """Hypersurface CSV directory under the MC/data root ($DATA_DIR)."""
    return os.path.join(paths.data_dir(), "Pynu", "data", "IceCube")


def env_task_default():
    """Default --row from $SLURM_ARRAY_TASK_ID if present (None otherwise)."""
    v = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(v) if v not in (None, "") else None


def setup_pynufit_datafit(config, hs_dir):
    """Initialize PyNuFit with real data as observation for both experiments."""
    pynufit = PyNuFit(config, verbosity=False)

    # Load hypersurfaces for IC DeepCore
    for exp_name, exp in pynufit.Experiments.items():
        if isinstance(exp, ICDeepCore_Atm):
            print(f"  Loading hypersurfaces for {exp_name}...")
            exp.load_hypersurfaces(hs_dir)

    # Set initial physics params
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Sin2Theta23"] = TRUTH_THETA
        pt.OscillationTunes.Parameters["Dm231"] = TRUTH_DM
        pt.OscillationTunes.Parameters["Dm231_bar"] = TRUTH_DM
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    # Generate nominal expectation (needed for MC variance)
    pynufit.StartPhysics()
    pynufit.StartNuisance()
    pynufit.ApplyOscillations("Physics")
    pynufit.ApplyNuisanceWeights(pynufit.Analysis.NuisNominalList)
    pynufit.SetExpectedWeights()
    pynufit.SetBinnedExpectedEvents()
    pynufit.SetBinnedMCVariance()
    pynufit.SetMuonBackground()

    # Set observation from real data for BOTH experiments
    pynufit.Observation = {}
    for exp_name, exp in pynufit.Experiments.items():
        exp.SetObservedBinned()
        pynufit.Observation[exp_name] = exp.GetObservedBinned()

    # Set up likelihood with data observation
    pynufit.set_likelihood("BarlowBeestonLikelihood")
    for exp_name in pynufit.Observation:
        pynufit.LLH.observation[exp_name] = pynufit.Observation[exp_name]
    pynufit.LLH.set_muon_background(pynufit.MuonBackground)
    pynufit.LLH.set_mc_variance(pynufit.MCVariance)

    # Report observation stats
    n_data = {}
    for exp_name, obs in pynufit.Observation.items():
        n_data[exp_name] = float(np.sum(obs))
    n_muon = 0
    for exp_name in pynufit.MuonBackground:
        if pynufit.MuonBackground[exp_name] is not None:
            muon_counts, _ = pynufit.MuonBackground[exp_name]
            n_muon += np.sum(muon_counts)

    # Build HS parameter index map
    hs_names = ICDeepCore_Atm.HS_SLOPE_NAMES
    hs_indices = {}
    for name in hs_names:
        if name in pynufit.Analysis.NuisanceList:
            hs_indices[name] = pynufit.Analysis.NuisanceList.index(name)
    print(f"  HS parameter indices: {hs_indices}")

    return pynufit, n_data, n_muon, hs_indices


def run_one_point(pynufit, dm231, dm231_bar, s23, nominal, sigma, bounds, hs_indices, x0=None):
    """Run nuisance minimization at a single (Dm231, Dm231_bar, Sin2Theta23) grid point."""
    if x0 is None:
        x0 = nominal
    pynufit.StartPhysics()
    pynufit.StartNuisance()

    # Set all three oscillation parameters on all experiments
    for name, pt in pynufit.physics_tunes.items():
        pt.OscillationTunes.Parameters["Dm231"] = dm231
        pt.OscillationTunes.Parameters["Dm231_bar"] = dm231_bar
        pt.OscillationTunes.Parameters["Sin2Theta23"] = s23
        if hasattr(pt.OscillationTunes, 'reset_cache'):
            pt.OscillationTunes.reset_cache()

    pynufit.ApplyOscillations("Physics")

    def objective(nuisance):
        # 1. Apply nuisance weights (flux for both; ORCA det for ORCA; IC HS are no-ops)
        pynufit.StartNuisance()
        pynufit.ApplyNuisanceWeights(nuisance)
        pynufit.SetExpectedWeights()
        # 2. Bin events (uncorrected for both) and recompute the MC variance at
        #    the CURRENT point (Barlow-Beeston needs sigma^2_MC(eta)); snapshot.
        pynufit.SetBinnedExpectedEvents()
        pynufit.SetBinnedMCVariance()
        mc_var = {k: np.array(v, copy=True) for k, v in pynufit.MCVariance.items()}

        # 3. Build HS param dict from nuisance vector
        hs_params = {}
        for hs_name, idx in hs_indices.items():
            hs_params[hs_name] = nuisance[idx]

        # 4. Overwrite IC expectation with HS-corrected histograms and scale the
        #    IC MC variance by the HS bin-factor^2. ORCA is untouched.
        for exp_name, exp in pynufit.Experiments.items():
            if isinstance(exp, ICDeepCore_Atm):
                uncorr = np.array(pynufit.Expectation[exp_name], copy=True)
                corrected = exp.apply_hs_correction(dm231, hs_params)
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor = np.where(uncorr > 0, corrected / uncorr, 1.0)
                mc_var[exp_name] = mc_var[exp_name] * factor**2
                pynufit.Expectation[exp_name] = corrected

        # 5. Compute combined chi2 (summed over both experiments)
        return pynufit.LLH.stats_and_systematics(
            pynufit.Expectation, nuisance, mc_var
        )

    result = minimize(
        objective, x0,
        method='L-BFGS-B', jac=None, bounds=bounds,
        options={'ftol': 1e-5, 'gtol': 1e-5, 'maxiter': 200}
    )

    return result.fun, result.x, result.nit, result.success


def main():
    parser = argparse.ArgumentParser(
        description="Combined ORCA+IC CPT 3D data fit row worker (Dm231 x Delta x Sin2Theta23)")
    parser.add_argument("--row", "--row-idx", dest="row", type=int,
                        default=env_task_default(),
                        help="Row index (Dm231 index); defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--n-grid", type=int, default=41, help="Grid size for Dm231 and Delta axes")
    parser.add_argument("--output-dir", required=True, help="Output directory for row results")
    parser.add_argument("--config", default=None,
                        help="Pynu XML config (default: resolved Combined_ORCA_IC_CPT_datafit.xml)")
    parser.add_argument("--hs-dir", default=None,
                        help="Path to directory containing hs_*.csv files "
                             "(default: $DATA_DIR/Pynu/data/IceCube)")
    parser.add_argument("--dm-min", type=float, default=None, help="Override DM_MIN")
    parser.add_argument("--dm-max", type=float, default=None, help="Override DM_MAX")
    parser.add_argument("--delta-min", type=float, required=True,
                        help="Min Delta (Dm231-Dm231_bar)")
    parser.add_argument("--delta-max", type=float, required=True,
                        help="Max Delta")
    parser.add_argument("--s23-min", type=float, required=True,
                        help="Min Sin2Theta23")
    parser.add_argument("--s23-max", type=float, required=True,
                        help="Max Sin2Theta23")
    parser.add_argument("--n-s23", type=int, default=20,
                        help="Number of Sin2Theta23 grid points")
    args = parser.parse_args()

    if args.row is None:
        parser.error("--row is required (or set $SLURM_ARRAY_TASK_ID)")
    if args.config is None:
        args.config = paths.pynu_config(CONFIG_NAME)
    elif not os.path.exists(args.config):
        # A bare config name -> resolve from configs/pynu/ with ${DATA_DIR}
        # expanded. An existing path is used as-is.
        args.config = paths.pynu_config(args.config)
    if args.hs_dir is None:
        args.hs_dir = default_hs_dir()

    dm_min = args.dm_min if args.dm_min is not None else DM_MIN
    dm_max = args.dm_max if args.dm_max is not None else DM_MAX
    dm_grid = np.linspace(dm_min, dm_max, args.n_grid)
    delta_grid = np.linspace(args.delta_min, args.delta_max, args.n_grid)
    s23_grid = np.linspace(args.s23_min, args.s23_max, args.n_s23)
    dm231 = dm_grid[args.row]

    n_points = args.n_grid * args.n_s23
    print(f"[Combined 3D Worker {args.row}] Dm231={dm231:.5e}")
    print(f"  Delta range: [{args.delta_min:.2e}, {args.delta_max:.2e}] ({args.n_grid} points)")
    print(f"  S23 range: [{args.s23_min:.3f}, {args.s23_max:.3f}] ({args.n_s23} points)")
    print(f"  Total points this row: {n_points}")
    print(f"  HS dir: {args.hs_dir}")
    print(f"  Started: {datetime.now().isoformat()}")

    # Initialize
    pynufit, n_data, n_muon, hs_indices = setup_pynufit_datafit(args.config, args.hs_dir)
    n_nuis = len(pynufit.Analysis.NuisanceList)
    print(f"  Experiments: {list(pynufit.Experiments.keys())}")
    for exp_name, n_evt in n_data.items():
        print(f"  Data events ({exp_name}): {n_evt:.1f}")
    print(f"  Muon background: {n_muon:.1f}")
    print(f"  Nuisance parameters: {n_nuis}")
    print(f"  Params: {pynufit.Analysis.NuisanceList}")

    # Set up minimization bounds
    nominal = np.array(pynufit.Analysis.NuisNominalList)
    sigma = np.array(pynufit.Analysis.NuisSigmaList)
    lower = nominal - 5 * sigma
    upper = nominal + 5 * sigma
    for k in range(len(lower)):
        if nominal[k] > 0 and lower[k] < 0.01:
            lower[k] = 0.01
    bounds = list(zip(lower, upper))

    # Outer loop: Sin2Theta23, Inner loop: Delta (warm-start within each s23 slice)
    results = []
    for k, s23 in enumerate(s23_grid):
        x0 = nominal.copy()  # reset warm start for each new s23
        t_s23_start = datetime.now()
        for j, delta in enumerate(delta_grid):
            dm231_bar = dm231 - delta
            chi2, nuisance, nit, converged = run_one_point(
                pynufit, dm231, dm231_bar, s23, nominal, sigma, bounds, hs_indices, x0=x0
            )
            x0 = nuisance.copy()  # warm start within this s23 slice
            pull_max = np.max(np.abs((nuisance - nominal) / sigma))
            results.append({
                'i': args.row, 'j': j, 'k': k,
                'dm231': float(dm231), 'dm231_bar': float(dm231_bar),
                'delta': float(delta), 's23': float(s23),
                'chi2': float(chi2), 'nit': int(nit),
                'converged': bool(converged), 'max_pull': float(pull_max),
                'nuisance': nuisance.tolist()
            })
            print(f"  [i={args.row},k={k:2d},j={j:2d}] s23={s23:.4f} delta={delta:.5e}: "
                  f"chi2={chi2:8.4f}, iter={nit:3d}, conv={converged}, pull={pull_max:.3f}")

        dt = (datetime.now() - t_s23_start).total_seconds()
        n_conv_slice = sum(1 for r in results[-args.n_grid:] if r['converged'])
        print(f"  --- s23={s23:.4f} done: {n_conv_slice}/{args.n_grid} converged, {dt:.1f}s ---")

    # Save row results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"row_{args.row:03d}.json")
    n_data_total = sum(n_data.values())
    row_data = {
        'row_idx': args.row,
        'dm231': float(dm231),
        'n_grid': args.n_grid,
        'n_s23': args.n_s23,
        'dm_range': [float(dm_min), float(dm_max)],
        'delta_range': [float(args.delta_min), float(args.delta_max)],
        's23_range': [float(args.s23_min), float(args.s23_max)],
        'coordinate_system': 'delta',
        'truth_dm': float(TRUTH_DM),
        'truth_s23': float(TRUTH_THETA),
        'mode': 'datafit_3d',
        'n_data_events': float(n_data_total),
        'n_data_per_experiment': {k: float(v) for k, v in n_data.items()},
        'nuisance_names': pynufit.Analysis.NuisanceList,
        'experiments': list(pynufit.Experiments.keys()),
        'hs_indices': hs_indices,
        'points': results
    }
    with open(out_path, 'w') as f:
        json.dump(row_data, f, indent=2)

    n_conv = sum(1 for r in results if r['converged'])
    print(f"\n  Total convergence: {n_conv}/{len(results)}")
    print(f"  Saved: {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
