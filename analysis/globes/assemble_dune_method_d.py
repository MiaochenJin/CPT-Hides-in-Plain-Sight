#!/usr/bin/env python3
"""
assemble_dune_method_d.py — Assemble DUNE method-d task results and make band plot.

Ported from:
    claude/3-CPT-violation/CP-CPT-degeneracy/DUNE/scripts/assemble_method_d.py

Collects the per-delta task_*.json / task_*.npz produced by
``run_dune_dcp_scan.py``, writes a combined NPZ, and renders the CPT-bias band
plot (paper Fig 4 input; the final composite figure is built by
plots/plot_fig4_dune_cpt_bias.py).

Env / inputs:
    --results-dir  directory containing the task_*.json / task_*.npz produced by
                   run_dune_dcp_scan.py. NOT shipped with the repo; pass the
                   directory you scanned into.

Outputs (written into --results-dir, alongside the inputs, as in the original):
    cpt_bias_<dcp_tag>_method_d.npz       combined chi2 curves + regions
    cpt_bias_<dcp_tag>_method_d_band.png  band plot

Usage:
    python3 assemble_dune_method_d.py --results-dir <dir>

De-hardcode note:
  - The original read the results dir from ``sys.argv[1]``; it is now the
    required ``--results-dir`` argument (no cluster-assuming default). No
    absolute paths were present in the original; outputs are still written
    next to the inputs.
"""

import argparse
import sys
import os
import json
import numpy as np
import math
from scipy.signal import argrelmin

def main():
    parser = argparse.ArgumentParser(
        description="Assemble DUNE method-d task results and make band plot.")
    parser.add_argument('--results-dir', required=True,
                        help='Directory with task_*.json / task_*.npz from '
                             'run_dune_dcp_scan.py (not shipped; pass your scan dir).')
    args = parser.parse_args()
    results_dir = args.results_dir

    # Load all task JSONs
    results = []
    for f in sorted(os.listdir(results_dir)):
        if f.startswith('task_') and f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as fh:
                results.append(json.load(fh))

    results.sort(key=lambda r: r.get('delta', r.get('delta_true')))
    print(f"Loaded {len(results)} task results")

    # Detect truth dCP from JSON (default 0 for backwards compatibility)
    truth_dcp_deg = results[0].get('truth_dcp_deg', 0.0)
    truth_dcp_pi = truth_dcp_deg / 180.0
    print(f"Truth dCP = {truth_dcp_deg:.1f} deg ({truth_dcp_pi:.3f} pi)")

    # Summary table
    print(f"\n{'Delta (eV2)':>14} {'best dCP':>10} {'min chi2':>10} "
          f"{'1sig lo':>7} {'1sig hi':>7} {'1sig w':>6} {'3sig lo':>7} {'3sig hi':>7}")
    print("-" * 90)
    for r in results:
        s1 = r['regions']['1']
        s3 = r['regions']['3']
        s1w = s1[1] - s1[0]
        print(f"{r.get('delta', r.get('delta_true')):+14.4e} {r['best_dcp_deg']:+10.1f} deg {r['min_chi2']:10.1f} "
              f"{s1[0]:7.0f} {s1[1]:7.0f} {s1w:6.0f} {s3[0]:7.0f} {s3[1]:7.0f}")

    # Load chi2 curves for combined NPZ
    all_chi2 = {}
    dcp_vals = None
    for r in results:
        tid = r['task_id']
        if tid >= 0:
            npz_path = os.path.join(results_dir, f'task_{tid:03d}.npz')
        else:
            # Delta override task — find matching NPZ by glob
            delta_val = r.get('delta', r.get('delta_true'))
            npz_candidates = [f for f in os.listdir(results_dir)
                              if f.startswith('task_delta_') and f.endswith('.npz')]
            npz_path = None
            for c in npz_candidates:
                d = np.load(os.path.join(results_dir, c))
                if abs(float(d['delta_true']) - delta_val) < 1e-10:
                    npz_path = os.path.join(results_dir, c)
                    break
            if npz_path is None:
                print(f"  WARNING: no NPZ found for delta={delta_val}, skipping")
                continue
        data = np.load(npz_path)
        all_chi2[r.get('delta', r.get('delta_true'))] = data['chi2_best']
        if dcp_vals is None:
            dcp_vals = data['dcp_vals']

    # Save combined
    save_dict = {
        'dcp_vals': dcp_vals,
        'deltas': np.array([r.get('delta', r.get('delta_true')) for r in results]),
        'best_dcps': np.array([r['best_dcp_deg'] for r in results]),
        'min_chi2s': np.array([r['min_chi2'] for r in results]),
        'sig1_lo': np.array([r['regions']['1'][0] for r in results]),
        'sig1_hi': np.array([r['regions']['1'][1] for r in results]),
        'sig2_lo': np.array([r['regions']['2'][0] for r in results]),
        'sig2_hi': np.array([r['regions']['2'][1] for r in results]),
        'sig3_lo': np.array([r['regions']['3'][0] for r in results]),
        'sig3_hi': np.array([r['regions']['3'][1] for r in results]),
    }
    for r in results:
        tag = f"{r.get('delta', r.get('delta_true')):+.4e}".replace('+', 'p').replace('-', 'n').replace('.', 'd')
        save_dict[f'chi2_best_{tag}'] = all_chi2[r.get('delta', r.get('delta_true'))]

    dcp_tag = f"dcp{int(truth_dcp_deg)}" if truth_dcp_deg != 0 else "dcp0"
    combined_path = os.path.join(results_dir, f'cpt_bias_{dcp_tag}_method_d.npz')
    np.savez(combined_path, **save_dict)
    print(f"\nSaved combined: {combined_path}")

    # ================================================================
    # Band plot — matching paper conventions (plot_cpt_truth_bias_band_v11.py)
    # ================================================================
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 200,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.patches import Patch, Rectangle

    THRESH_90CL = 2.71
    THRESH_3SIG = 9.0

    DCP_LABEL = r'$\delta_{CP}$ [$\pi$]'
    DELTA_LABEL = r'$\Delta\bar{m}^2_{31} - \Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]'

    def find_all_regions(dchi2, dcp_pi, threshold):
        below = dchi2 <= threshold
        if not np.any(below):
            return []
        regions = []
        in_region = False
        lo = 0
        for i in range(len(below)):
            if below[i] and not in_region:
                lo = i
                in_region = True
            elif not below[i] and in_region:
                regions.append((dcp_pi[lo], dcp_pi[i - 1]))
                in_region = False
        if in_region:
            regions.append((dcp_pi[lo], dcp_pi[-1]))
        return regions

    deltas = save_dict['deltas']
    deltas_1e3 = deltas * 1e3
    n_delta = len(deltas)

    # Half-heights for boxes
    hh = np.zeros(n_delta)
    for j in range(n_delta):
        if j == 0:
            hh[j] = (deltas_1e3[1] - deltas_1e3[0]) / 2
        elif j == n_delta - 1:
            hh[j] = (deltas_1e3[-1] - deltas_1e3[-2]) / 2
        else:
            hh[j] = min(deltas_1e3[j] - deltas_1e3[j-1],
                        deltas_1e3[j+1] - deltas_1e3[j]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    for j in range(n_delta):
        tag = f"{deltas[j]:+.4e}".replace('+', 'p').replace('-', 'n').replace('.', 'd')
        key = f'chi2_best_{tag}'
        if key not in save_dict:
            continue
        chi2 = save_dict[key]
        chi2_min = chi2.min()
        dchi2 = chi2 - chi2_min
        dcp_pi = np.degrees(dcp_vals) / 180.0
        y = deltas_1e3[j]
        h = hh[j]

        # 3σ boxes
        for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_3SIG):
            ax.add_patch(Rectangle((lo, y - h), hi - lo, 2 * h,
                         facecolor='#d62728', alpha=0.25, edgecolor='none', zorder=1))
        # 90% CL boxes
        for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_90CL):
            ax.add_patch(Rectangle((lo, y - h), hi - lo, 2 * h,
                         facecolor='#ff7f0e', alpha=0.45, edgecolor='none', zorder=2))

        # Best-fit dot (single global minimum)
        imin = np.argmin(chi2)
        ax.plot(dcp_pi[imin], y, 'k.', ms=4, zorder=5)

        # Secondary minimum: use argrelmin to find true local minima,
        # pick the single deepest one that is distinct from global min
        rel_min_idx = argrelmin(chi2, order=5)[0]
        best_secondary = None
        best_secondary_dchi2 = np.inf
        for mi in rel_min_idx:
            if mi == imin:
                continue
            dc = chi2[mi] - chi2_min
            # Must be a meaningfully distinct minimum (Δχ² >= 0.5)
            # and within 3σ, and inside a 3σ allowed region
            if 0.5 <= dc <= THRESH_3SIG:
                # Check if this dCP is inside a 3σ region of meaningful width
                # (filter out single-bin noise spikes)
                in_band = False
                min_region_width = 3 * (dcp_pi[1] - dcp_pi[0])  # at least 3 grid points
                for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_3SIG):
                    if (hi - lo) >= min_region_width and lo <= dcp_pi[mi] <= hi:
                        in_band = True
                        break
                if in_band and dc < best_secondary_dchi2:
                    best_secondary = mi
                    best_secondary_dchi2 = dc
        if best_secondary is not None:
            ax.plot(dcp_pi[best_secondary], y, '.', color='gray', ms=3, zorder=4)

    # Reference lines
    ax.axvline(truth_dcp_pi, color='#2ca02c', ls='--', lw=2, alpha=0.7, zorder=3)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.3)
    ax.plot(truth_dcp_pi, 0, 'ko', ms=6, zorder=10)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-2.05, 2.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.set_title(r'Truth $\delta_{CP} = ' + f'{truth_dcp_deg:.0f}' + r'°$', fontsize=14)
    ax.set_xlabel(DCP_LABEL, fontsize=18)
    ax.set_ylabel(DELTA_LABEL, fontsize=17)

    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.25, label=r'3$\sigma$ allowed'),
        Patch(facecolor='#ff7f0e', alpha=0.45, label='90% CL allowed'),
        plt.Line2D([0], [0], marker='.', color='k', ls='', ms=6, label=r'Best-fit $\delta_{CP}$'),
        plt.Line2D([0], [0], marker='.', color='gray', ls='', ms=5, label='Secondary minimum'),
        plt.Line2D([0], [0], color='#2ca02c', ls='--', lw=2, label='Truth'),
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc='lower right',
              handlelength=0.8, handletextpad=0.3, borderpad=0.2,
              labelspacing=0.2)

    plt.tight_layout()
    outfile = os.path.join(results_dir, f'cpt_bias_{dcp_tag}_method_d_band.png')
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"Saved {outfile}")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
