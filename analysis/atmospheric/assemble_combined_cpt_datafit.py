#!/usr/bin/env python3
"""
Assemble the combined ORCA-6 + IC DeepCore CPT 3D data-fit grid and profile over theta23.

Paper figure: Fig 5 (top panel) and Suppl Fig 4 (combined data-fit curve).
Collects the per-row ``row_*.json`` files written by
``run_combined_cpt_datafit.py`` into a 3D chi2 grid over
(Dm231, Delta, Sin2Theta23), profiles (minimizes) over sin2theta23, and writes
the result directory consumed by the plotters:
``dm_grid.npy`` / ``delta_grid.npy`` / ``chi2_3d.npy`` / ``converged_3d.npy`` /
``s23_grid.npy`` / ``chi2_profiled.npy`` / ``s23_profiled.npy`` /
``chi2_grid.npy`` / ``delta_chi2_grid.npy`` / ``metadata.json``
(this is the ``combined_cpt_3d_datafit_41x41x20`` layout).

Config / env / inputs:
- Input: ``--input-dir`` holding the ``row_*.json`` files (no default; produced
  by the runner, not shipped with the repo).
- Output: ``--output-dir`` for the assembled arrays and diagnostic plots.
- Optional overlay of an existing 2D result via ``--overlay-dir``.
- No Pynu / nuSQuIDS dependency (pure numpy/matplotlib post-processing).

This file is the SRC assembler
``claude/3-CPT-violation/data-fits/ORCA/scripts/assemble_and_profile_3d.py``
(the script that produced ``combined_cpt_3d_datafit_41x41x20``). It contained no
hardcoded absolute paths — input/output are already argparse-driven — so the
scientific logic is copied verbatim; only this docstring header was added.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


def load_rows(input_dir):
    """Load all row JSON files and build 3D chi2 grid."""
    row_files = sorted([f for f in os.listdir(input_dir) if f.startswith('row_') and f.endswith('.json')])
    if not row_files:
        raise FileNotFoundError(f"No row_*.json files found in {input_dir}")

    # Read first row to get grid parameters
    with open(os.path.join(input_dir, row_files[0])) as f:
        first = json.load(f)

    n_grid = first['n_grid']
    n_s23 = first['n_s23']
    dm_range = first['dm_range']
    delta_range = first['delta_range']
    s23_range = first['s23_range']
    truth_dm = first['truth_dm']
    truth_s23 = first.get('truth_s23', 0.572)

    dm_grid = np.linspace(dm_range[0], dm_range[1], n_grid)
    delta_grid = np.linspace(delta_range[0], delta_range[1], n_grid)
    s23_grid = np.linspace(s23_range[0], s23_range[1], n_s23)

    # 3D arrays: chi2[i, j, k] = chi2 at (dm_grid[i], delta_grid[j], s23_grid[k])
    chi2_3d = np.full((n_grid, n_grid, n_s23), np.nan)
    converged_3d = np.full((n_grid, n_grid, n_s23), False)
    best_s23_3d = np.full((n_grid, n_grid, n_s23), np.nan)
    nuisance_3d = None

    print(f"Loading {len(row_files)} row files...")
    for fname in row_files:
        with open(os.path.join(input_dir, fname)) as f:
            row = json.load(f)
        i = row['row_idx']
        for pt in row['points']:
            j = pt['j']
            k = pt['k']
            chi2_3d[i, j, k] = pt['chi2']
            converged_3d[i, j, k] = pt['converged']

    n_valid = np.sum(~np.isnan(chi2_3d))
    n_total = chi2_3d.size
    n_conv = np.sum(converged_3d)
    print(f"  Grid shape: {chi2_3d.shape}")
    print(f"  Valid points: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"  Converged: {n_conv}/{n_total} ({100*n_conv/n_total:.1f}%)")

    return chi2_3d, converged_3d, dm_grid, delta_grid, s23_grid, truth_dm, truth_s23


def profile_over_theta23(chi2_3d, s23_grid):
    """Profile (minimize) over sin2theta23 at each (Dm231, Delta) point."""
    # For each (i, j), find k that minimizes chi2
    best_k = np.nanargmin(chi2_3d, axis=2)
    n_dm, n_delta = best_k.shape

    chi2_profiled = np.full((n_dm, n_delta), np.nan)
    s23_profiled = np.full((n_dm, n_delta), np.nan)

    for i in range(n_dm):
        for j in range(n_delta):
            k = best_k[i, j]
            chi2_profiled[i, j] = chi2_3d[i, j, k]
            s23_profiled[i, j] = s23_grid[k]

    return chi2_profiled, s23_profiled


def load_overlay(overlay_dir):
    """Load an existing 2D result for overlay comparison."""
    chi2 = np.load(os.path.join(overlay_dir, 'chi2_grid.npy'))
    dm = np.load(os.path.join(overlay_dir, 'dm_grid.npy'))

    delta_path = os.path.join(overlay_dir, 'delta_grid.npy')
    if os.path.exists(delta_path):
        delta = np.load(delta_path)
    else:
        delta = None

    meta_path = os.path.join(overlay_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    dchi2 = chi2 - np.nanmin(chi2)
    return dchi2, dm, delta, meta


def plot_profiled_contours(chi2_profiled, s23_profiled, dm_grid, delta_grid,
                           truth_dm, truth_s23, output_dir,
                           overlay_dchi2=None, overlay_dm=None, overlay_delta=None,
                           overlay_label="Fixed θ₂₃"):
    """Plot profiled 2D contours and best-fit theta23 map."""
    dm_scale = 1e3
    delta_scale = 1e3

    min_chi2 = np.nanmin(chi2_profiled)
    dchi2 = chi2_profiled - min_chi2
    best_idx = np.unravel_index(np.nanargmin(chi2_profiled), chi2_profiled.shape)
    best_dm = dm_grid[best_idx[0]]
    best_delta = delta_grid[best_idx[1]]
    best_s23 = s23_profiled[best_idx[0], best_idx[1]]

    print(f"\nProfiled results:")
    print(f"  Min chi2: {min_chi2:.2f}")
    print(f"  Best-fit Dm231: {best_dm:.5e} eV2")
    print(f"  Best-fit Delta: {best_delta:.5e} eV2")
    print(f"  Best-fit Sin2Theta23: {best_s23:.4f}")

    # CL thresholds (2 DOF after profiling)
    cl_levels = {
        '68.3% (1σ)': 2.30,
        '90%': 4.61,
        '95.4% (2σ)': 6.18,
        '99.7% (3σ)': 11.83,
    }

    # --- Figure 1: Profiled 2D contours with overlay ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    dm_plot = dm_grid * dm_scale
    delta_plot = delta_grid * delta_scale

    # Smooth for cleaner contours
    dchi2_smooth = gaussian_filter(dchi2, sigma=0.8)

    levels_to_plot = [cl_levels['68.3% (1σ)'], cl_levels['90%'], cl_levels['95.4% (2σ)']]
    colors_profiled = ['#d62728', '#ff7f0e', '#2ca02c']
    labels_profiled = ['68.3% (1σ)', '90%', '95.4% (2σ)']

    for lev, col, lab in zip(levels_to_plot, colors_profiled, labels_profiled):
        cs = ax.contour(dm_plot, delta_plot, dchi2_smooth.T, levels=[lev],
                        colors=[col], linewidths=2.0)

    # Overlay old fixed-theta23 result
    if overlay_dchi2 is not None and overlay_delta is not None:
        ov_dm_plot = overlay_dm * dm_scale
        ov_delta_plot = overlay_delta * delta_scale
        ov_smooth = gaussian_filter(overlay_dchi2, sigma=0.8)
        for lev, col in zip(levels_to_plot[:2], ['#9467bd', '#9467bd']):
            ax.contour(ov_dm_plot, ov_delta_plot, ov_smooth.T, levels=[lev],
                       colors=[col], linewidths=1.5, linestyles='dashed')

    # Best-fit markers
    ax.plot(best_dm * dm_scale, best_delta * delta_scale, '*', color='red',
            markersize=15, markeredgecolor='k', markeredgewidth=0.5, zorder=10)

    # CPT symmetric line
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=13)
    ax.set_ylabel(r'$\Delta m^2_{31} - \Delta \bar{m}^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=13)
    ax.set_title('ORCA CPT Data Fit — Profiled over sin²θ₂₃', fontsize=14)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=colors_profiled[0], lw=2, label='68.3% (profiled)'),
        Line2D([0], [0], color=colors_profiled[1], lw=2, label='90% (profiled)'),
        Line2D([0], [0], color=colors_profiled[2], lw=2, label='95.4% (profiled)'),
    ]
    if overlay_dchi2 is not None:
        legend_elements.append(
            Line2D([0], [0], color='#9467bd', lw=1.5, linestyle='dashed',
                   label=f'68%/90% ({overlay_label})'))
    legend_elements.append(
        Line2D([0], [0], marker='*', color='red', lw=0, markersize=12,
               markeredgecolor='k', label='Best fit'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    fig.tight_layout()
    path1 = os.path.join(output_dir, 'cpt_profiled_contours.png')
    fig.savefig(path1, dpi=150)
    print(f"  Saved: {path1}")
    plt.close(fig)

    # --- Figure 2: Best-fit sin2theta23 map ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    im = ax2.pcolormesh(dm_plot, delta_plot, s23_profiled.T,
                        cmap='viridis', shading='auto')
    cbar = fig2.colorbar(im, ax=ax2, label='Best-fit sin²θ₂₃')
    ax2.plot(best_dm * dm_scale, best_delta * delta_scale, '*', color='red',
             markersize=15, markeredgecolor='white', markeredgewidth=1, zorder=10)
    ax2.axhline(0, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=13)
    ax2.set_ylabel(r'$\Delta m^2_{31} - \Delta \bar{m}^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=13)
    ax2.set_title('Best-fit sin²θ₂₃ at each (Δm²₃₁, Δ) point', fontsize=14)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, 'cpt_bestfit_s23_map.png')
    fig2.savefig(path2, dpi=150)
    print(f"  Saved: {path2}")
    plt.close(fig2)

    # --- Figure 3: 1D Delta profile ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    # Profile over Dm231 as well to get 1D Delta profile
    dchi2_1d = np.nanmin(dchi2, axis=0)  # min over Dm231 at each Delta
    ax3.plot(delta_plot, dchi2_1d, 'b-', linewidth=2, label='Profiled over θ₂₃')

    # Overlay old fixed-theta23 1D profile
    if overlay_dchi2 is not None and overlay_delta is not None:
        ov_dchi2_1d = np.nanmin(overlay_dchi2, axis=0)
        ax3.plot(overlay_delta * delta_scale, ov_dchi2_1d, '--', color='#9467bd',
                 linewidth=1.5, label=overlay_label)

    ax3.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, label='1σ (1 DOF)')
    ax3.axhline(2.71, color='gray', linestyle='--', linewidth=0.8, label='90% (1 DOF)')
    ax3.axhline(4.0, color='gray', linestyle='-.', linewidth=0.8, label='2σ (1 DOF)')
    ax3.axvline(0, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel(r'$\Delta = \Delta m^2_{31} - \Delta \bar{m}^2_{31}$ [$\times 10^{-3}$ eV$^2$]', fontsize=13)
    ax3.set_ylabel(r'$\Delta\chi^2$', fontsize=13)
    ax3.set_title('1D Δ Profile (profiled over Δm²₃₁ and sin²θ₂₃)', fontsize=13)
    ax3.set_ylim(-0.2, 15)
    ax3.legend(fontsize=10)
    fig3.tight_layout()
    path3 = os.path.join(output_dir, 'cpt_1d_profile_delta_profiled.png')
    fig3.savefig(path3, dpi=150)
    print(f"  Saved: {path3}")
    plt.close(fig3)

    # --- Print 1σ Delta range (1 DOF) ---
    # Interpolate to find where dchi2_1d crosses 1.0
    above = dchi2_1d > 1.0
    transitions = np.where(np.diff(above.astype(int)))[0]
    if len(transitions) >= 2:
        # Linear interpolation at crossings
        lo_idx = transitions[0]
        hi_idx = transitions[-1]
        lo_delta = np.interp(1.0, [dchi2_1d[lo_idx], dchi2_1d[lo_idx+1]],
                             [delta_plot[lo_idx], delta_plot[lo_idx+1]])
        hi_delta = np.interp(1.0, [dchi2_1d[hi_idx+1], dchi2_1d[hi_idx]],
                             [delta_plot[hi_idx+1], delta_plot[hi_idx]])
        print(f"\n  1σ Δ range (1 DOF): [{lo_delta:.2f}, {hi_delta:.2f}] ×10⁻³ eV² "
              f"(width {hi_delta - lo_delta:.2f}×10⁻³)")
    else:
        print(f"\n  Could not determine 1σ crossings from 1D profile")

    return min_chi2, best_dm, best_delta, best_s23


def main():
    parser = argparse.ArgumentParser(description="Assemble 3D CPT grid and profile over theta23")
    parser.add_argument("--input-dir", required=True, help="Directory with row_*.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory for arrays and plots")
    parser.add_argument("--overlay-dir", default=None,
                        help="Directory with old 2D result for overlay comparison")
    parser.add_argument("--overlay-label", default="Fixed θ₂₃",
                        help="Label for overlay contours")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and assemble 3D grid
    chi2_3d, converged_3d, dm_grid, delta_grid, s23_grid, truth_dm, truth_s23 = \
        load_rows(args.input_dir)

    # Save 3D arrays
    np.save(os.path.join(args.output_dir, 'chi2_3d.npy'), chi2_3d)
    np.save(os.path.join(args.output_dir, 'converged_3d.npy'), converged_3d)
    np.save(os.path.join(args.output_dir, 'dm_grid.npy'), dm_grid)
    np.save(os.path.join(args.output_dir, 'delta_grid.npy'), delta_grid)
    np.save(os.path.join(args.output_dir, 's23_grid.npy'), s23_grid)

    # Profile over theta23
    chi2_profiled, s23_profiled = profile_over_theta23(chi2_3d, s23_grid)
    np.save(os.path.join(args.output_dir, 'chi2_profiled.npy'), chi2_profiled)
    np.save(os.path.join(args.output_dir, 's23_profiled.npy'), s23_profiled)

    # Also save profiled result in standard format for compatibility
    dchi2_profiled = chi2_profiled - np.nanmin(chi2_profiled)
    np.save(os.path.join(args.output_dir, 'chi2_grid.npy'), chi2_profiled)
    np.save(os.path.join(args.output_dir, 'delta_chi2_grid.npy'), dchi2_profiled)

    # Load overlay if provided
    overlay_dchi2, overlay_dm, overlay_delta = None, None, None
    if args.overlay_dir and os.path.isdir(args.overlay_dir):
        print(f"\nLoading overlay from {args.overlay_dir}")
        overlay_dchi2, overlay_dm, overlay_delta, _ = load_overlay(args.overlay_dir)

    # Plot
    min_chi2, best_dm, best_delta, best_s23 = plot_profiled_contours(
        chi2_profiled, s23_profiled, dm_grid, delta_grid,
        truth_dm, truth_s23, args.output_dir,
        overlay_dchi2, overlay_dm, overlay_delta,
        overlay_label=args.overlay_label)

    # Save metadata
    meta = {
        'grid_shape': list(chi2_3d.shape),
        'dm_range': [float(dm_grid[0]), float(dm_grid[-1])],
        'delta_range': [float(delta_grid[0]), float(delta_grid[-1])],
        's23_range': [float(s23_grid[0]), float(s23_grid[-1])],
        'n_grid': len(dm_grid),
        'n_s23': len(s23_grid),
        'min_chi2': float(min_chi2),
        'best_fit': {
            'dm231': float(best_dm),
            'delta': float(best_delta),
            's23': float(best_s23),
        },
        'convergence': {
            'total': int(np.sum(converged_3d)),
            'total_points': int(converged_3d.size),
            'fraction': float(np.sum(converged_3d) / converged_3d.size),
        },
        'coordinate_system': 'delta',
        'mode': 'datafit_3d_profiled',
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved metadata: {os.path.join(args.output_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
