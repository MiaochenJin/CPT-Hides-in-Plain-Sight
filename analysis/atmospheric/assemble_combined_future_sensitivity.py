#!/usr/bin/env python3
"""
Assemble the combined ORCA-Full + IC-Upgrade-7 CPT delta-coordinate sensitivity scan.

Paper figure: Fig 5 (bottom panel) and Suppl Fig 4 (projection curve). Collects the
per-row ``row_*.json`` files written by ``run_combined_future_sensitivity.py`` into the
2D ``combined_fine_31x31`` result directory consumed by the plotters:
``chi2_grid.npy`` / ``converged_grid.npy`` / ``dm_grid.npy`` / ``delta_grid.npy`` /
``metadata.json``.

Config / env / inputs:
- Input: ``--input-dir`` holding the ``row_*.json`` files (no default; produced by the
  runner, not shipped with the repo).
- Output: ``--output-dir`` (defaults to ``--input-dir`` as in the original).
- The grid axes are reconstructed from the ``--n-dm`` / ``--n-delta`` / ``--dm-*`` /
  ``--delta-*`` args, which must match the runner (Fig 5 bottom: 31x31, Dm231 in
  [2.3, 2.7]e-3, Delta in [-0.6, 0.6]e-3).
- No Pynu / nuSQuIDS dependency (pure numpy post-processing).

This file is the SRC assembler
``claude/3-CPT-violation/sensitivity-scans/ORCA-full/scripts/assemble_orcafull_cpt_delta_results.py``
(the delta-coordinate 2D assembler whose output layout matches ``combined_fine_31x31``).
It contained no hardcoded absolute paths — input/output are already argparse-driven —
so the scientific logic is copied verbatim; only this docstring header was added.
"""

import sys
import os
import json
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-dm", type=int, default=41)
    parser.add_argument("--n-delta", type=int, default=41)
    parser.add_argument("--dm-min", type=float, default=2.0e-3)
    parser.add_argument("--dm-max", type=float, default=3.0e-3)
    parser.add_argument("--delta-min", type=float, default=-1.0e-3)
    parser.add_argument("--delta-max", type=float, default=1.0e-3)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)
    delta_grid = np.linspace(args.delta_min, args.delta_max, args.n_delta)

    chi2_grid = np.full((args.n_dm, args.n_delta), np.nan)
    converged_grid = np.zeros((args.n_dm, args.n_delta), dtype=bool)

    found = 0
    total_time = 0
    for i in range(args.n_dm):
        row_file = os.path.join(args.input_dir, f'row_{i:03d}.json')
        if not os.path.exists(row_file):
            print(f"  Missing row {i}")
            continue
        with open(row_file) as f:
            row = json.load(f)
        chi2_grid[i, :] = row['chi2']
        converged_grid[i, :] = row['converged']
        total_time += row.get('total_time_s', 0)
        found += 1

    print(f"Assembled {found}/{args.n_dm} rows")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'chi2_grid.npy'), chi2_grid)
    np.save(os.path.join(args.output_dir, 'converged_grid.npy'), converged_grid)
    np.save(os.path.join(args.output_dir, 'dm_grid.npy'), dm_grid)
    np.save(os.path.join(args.output_dir, 'delta_grid.npy'), delta_grid)

    meta = {
        'n_dm': args.n_dm, 'n_delta': args.n_delta,
        'dm_min': args.dm_min, 'dm_max': args.dm_max,
        'delta_min': args.delta_min, 'delta_max': args.delta_max,
        'rows_found': found,
        'convergence_rate': float(converged_grid.sum()) / (args.n_dm * args.n_delta),
        'min_chi2': float(np.nanmin(chi2_grid)),
        'total_time_s': total_time,
    }
    bf_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    meta['best_fit_dm'] = float(dm_grid[bf_idx[0]])
    meta['best_fit_delta'] = float(delta_grid[bf_idx[1]])

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"Min chi²: {meta['min_chi2']:.4f}")
    print(f"Best-fit: Dm231={meta['best_fit_dm']:.4e}, Delta={meta['best_fit_delta']:.4e}")
    print(f"Convergence: {meta['convergence_rate']*100:.1f}%")

    dchi2 = chi2_grid - np.nanmin(chi2_grid)
    dm_prof = np.nanmin(dchi2, axis=1)
    delta_prof = np.nanmin(dchi2, axis=0)
    dm_1s = dm_grid[dm_prof < 1.0]
    delta_1s = delta_grid[delta_prof < 1.0]
    if len(dm_1s) > 0:
        print(f"1σ Dm231: [{dm_1s[0]*1e3:.4f}, {dm_1s[-1]*1e3:.4f}] "
              f"(width {(dm_1s[-1]-dm_1s[0])*1e3:.4f}×10⁻³)")
    if len(delta_1s) > 0:
        print(f"1σ Delta: [{delta_1s[0]*1e3:.4f}, {delta_1s[-1]*1e3:.4f}] "
              f"(width {(delta_1s[-1]-delta_1s[0])*1e3:.4f}×10⁻³)")

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
