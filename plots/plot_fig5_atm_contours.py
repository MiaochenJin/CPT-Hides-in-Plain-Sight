#!/usr/bin/env python3
"""
plot_fig5_atm_contours.py — Fig 5: 2-panel atmospheric CPT results figure.

Paper figure: Fig 5.
  (a) Combined ORCA-6 + IC DeepCore data-fit contours (top panel).
  (b) Combined ORCA-Full + IC-Upgrade-7 sensitivity projection (bottom panel).
Combined: 68% + 90% solid. Individual experiments: 90% only, dashed.

Inputs (all are assembled grid directories produced by the runners; NOT shipped
with this repo, so each is an explicit argparse arg with no cluster-assuming default):
  Top panel (data fit):
    --combined-datafit   combined ORCA-6 + IC DeepCore (combined_cpt_3d_datafit_41x41x20;
                         read via chi2_profiled.npy + dm_grid.npy + delta_grid.npy)
    --orca-datafit       ORCA-6 individual overlay   (dm_grid/delta_grid + chi2/delta_chi2)
    --ic-datafit         IC DeepCore individual overlay
  Bottom panel (sensitivity):
    --combined-sens      combined ORCA-Full + IC-Upgrade-7 (combined_fine_31x31)
    --orcafull-sens      ORCA-Full individual overlay
    --icup-sens          IC-Upgrade-7 individual overlay
  --output-dir           default: REPO_ROOT/outputs/figures (gitignored)

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/paper/scripts/plot_paper_cpt_2panel.py``):
- The single ``--base`` arg with six ``os.path.join(B, 'claude/3-CPT-violation/.../results/...')``
  result-directory constructions is replaced by six explicit ``--*-datafit`` /
  ``--*-sens`` directory args (the repo does not ship those grids).
- Output directory now defaults under ``REPO_ROOT/outputs/figures`` instead of CWD.
- Added the ``analysis.lib.paths`` bootstrap for ``REPO_ROOT``.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# --- repo bootstrap: make `analysis.lib` importable -------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.lib import paths

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 16,
    'axes.labelsize': 21,
    'axes.titlesize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 14,
    'figure.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Contour thresholds (2 dof)
THRESH_68 = 2.30
THRESH_90 = 4.61


def load_grid(base_dir, chi2_name=None):
    dm = np.load(os.path.join(base_dir, 'dm_grid.npy'))
    delta = np.load(os.path.join(base_dir, 'delta_grid.npy'))
    if chi2_name:
        chi2 = np.load(os.path.join(base_dir, chi2_name))
    else:
        # Datafit grids ship chi2_profiled.npy (profiled over s23); sensitivity
        # grids ship chi2_grid.npy; some ship a pre-subtracted delta_chi2_grid.npy.
        for cand in ('delta_chi2_grid.npy', 'chi2_grid.npy', 'chi2_profiled.npy'):
            p = os.path.join(base_dir, cand)
            if os.path.exists(p):
                chi2 = np.load(p)
                break
        else:
            raise FileNotFoundError(
                "no chi2 grid (delta_chi2_grid.npy / chi2_grid.npy / "
                f"chi2_profiled.npy) found in {base_dir}")
    return dm, delta, chi2 - np.nanmin(chi2)


def smooth_grid(dm, delta, dchi2, upsample=4, sigma=0.8):
    # Fill NaNs with nearest-neighbor before interpolation
    dchi2_clean = dchi2.copy()
    if np.any(np.isnan(dchi2_clean)):
        from scipy.ndimage import generic_filter
        mask = np.isnan(dchi2_clean)
        dchi2_clean[mask] = np.nanmax(dchi2_clean)
    interp = RegularGridInterpolator((dm, delta), dchi2_clean, method='cubic',
                                      bounds_error=False, fill_value=np.nan)
    dm_f = np.linspace(dm[0], dm[-1], len(dm) * upsample)
    de_f = np.linspace(delta[0], delta[-1], len(delta) * upsample)
    DM, DE = np.meshgrid(dm_f, de_f, indexing='ij')
    return dm_f, de_f, gaussian_filter(interp((DM, DE)), sigma=sigma)


def draw_contour_panel(ax, combined_dir, individual_dirs, is_datafit=True,
                       combined_chi2_name=None):
    dm, delta, dchi2 = load_grid(combined_dir, chi2_name=combined_chi2_name)
    dm_s, de_s, dchi2_s = smooth_grid(dm, delta, dchi2)

    # Combined: 68% and 90% (transpose: contour expects Z shape (n_y, n_x))
    ax.contour(dm_s*1e3, de_s*1e3, dchi2_s.T, levels=[THRESH_68],
               colors=['#d62728'], linestyles=['-'], linewidths=[2.5])
    ax.contour(dm_s*1e3, de_s*1e3, dchi2_s.T, levels=[THRESH_90],
               colors=['#ff7f0e'], linestyles=['-'], linewidths=[2.5])
    ax.plot([], [], '-', color='#d62728', lw=2.5, label=r'Combined 68.3% (1$\sigma$)')
    ax.plot([], [], '-', color='#ff7f0e', lw=2.5, label='Combined 90%')

    # Best fit / truth marker
    imin = np.unravel_index(np.nanargmin(dchi2), dchi2.shape)
    bf_dm = dm[imin[0]] * 1e3
    bf_delta = delta[imin[1]] * 1e3

    if is_datafit:
        ax.plot(bf_dm, bf_delta, '*', color='#d62728', ms=14, mew=1.5,
                zorder=10, label='Combined best fit')
    else:
        ax.plot(2.511, 0, 'P', color='k', ms=10, mew=2, zorder=10,
                label='Truth (2.511)')

    # Individual experiments: 90% only, with best-fit stars
    colors_ind = ['#9467bd', '#8c564b']
    for (name, idir), color in zip(individual_dirs.items(), colors_ind):
        dm_i, delta_i, dchi2_i = load_grid(idir)
        dm_is, de_is, dchi2_is = smooth_grid(dm_i, delta_i, dchi2_i)
        ax.contour(dm_is*1e3, de_is*1e3, dchi2_is.T, levels=[THRESH_90],
                   colors=[color], linestyles=['--'], linewidths=[1.5])
        # Individual best-fit star (data fit only — skip for sensitivity)
        if is_datafit:
            bf_i = np.unravel_index(np.nanargmin(dchi2_i), dchi2_i.shape)
            ax.plot(dm_i[bf_i[0]]*1e3, delta_i[bf_i[1]]*1e3, '*', color=color,
                    ms=10, mew=0.8, markeredgecolor='k', zorder=9)
        ax.plot([], [], '--', color=color, lw=1.5, label=f'{name} (90%)')

    # CPT symmetric line
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-datafit", required=True,
                    help="Combined ORCA-6 + IC DeepCore data-fit result dir "
                         "(combined_cpt_3d_datafit_41x41x20)")
    ap.add_argument("--orca-datafit", required=True,
                    help="ORCA-6 individual data-fit result dir (90%% overlay)")
    ap.add_argument("--ic-datafit", required=True,
                    help="IC DeepCore individual data-fit result dir (90%% overlay)")
    ap.add_argument("--combined-sens", required=True,
                    help="Combined ORCA-Full + IC-Upgrade-7 sensitivity dir (combined_fine_31x31)")
    ap.add_argument("--orcafull-sens", required=True,
                    help="ORCA-Full individual sensitivity dir (90%% overlay)")
    ap.add_argument("--icup-sens", required=True,
                    help="IC-Upgrade-7 individual sensitivity dir (90%% overlay)")
    ap.add_argument("--output-dir",
                    default=str(paths.REPO_ROOT / "outputs" / "figures"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Figure: 2 vertically stacked panels, independent axes
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8, 13))
    fig.subplots_adjust(hspace=0.28)

    # (a) Data fit
    draw_contour_panel(ax_a, args.combined_datafit,
                       {'ORCA-6': args.orca_datafit, 'IC DeepCore': args.ic_datafit},
                       is_datafit=True, combined_chi2_name='chi2_profiled.npy')
    ax_a.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]')
    ax_a.set_ylabel(r'$\Delta m^2_{31} - \Delta\bar{m}^2_{31}$ [$\times 10^{-3}$ eV$^2$]')
    ax_a.set_xlim(1.5, 3.0)
    ax_a.set_ylim(-1.0, 1.5)
    ax_a.legend(loc='upper left', fontsize=13, handlelength=1.5,
                handletextpad=0.4, labelspacing=0.3, framealpha=0.9)

    # (b) Sensitivity — zoomed to its natural scale
    draw_contour_panel(ax_b, args.combined_sens,
                       {'ORCA-Full': args.orcafull_sens, 'IC-Upgrade-7': args.icup_sens},
                       is_datafit=False)
    ax_b.set_xlabel(r'$\Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]')
    ax_b.set_ylabel(r'$\Delta m^2_{31} - \Delta\bar{m}^2_{31}$ [$\times 10^{-3}$ eV$^2$]')
    ax_b.set_xlim(2.3, 2.7)
    ax_b.set_ylim(-0.6, 0.6)
    ax_b.legend(loc='upper left', fontsize=13, handlelength=1.5,
                handletextpad=0.4, labelspacing=0.3, framealpha=0.9)

    out = os.path.join(args.output_dir, 'paper_cpt_2panel.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")

    out_pdf = os.path.join(args.output_dir, 'paper_cpt_2panel.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
