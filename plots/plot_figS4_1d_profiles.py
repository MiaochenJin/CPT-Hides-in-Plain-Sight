#!/usr/bin/env python3
"""
plot_figS4_1d_profiles.py — Suppl Fig 4: 1D profiled Δχ² on the CPT asymmetry.

Paper figure: Supplementary Fig 4. For each grid the chi2 is profiled (minimized)
over the Δm²₃₁ axis, giving the marginalized 1D profile chi2(δ) with
δ = Δm²₃₁ - Δm̄²₃₁. Two curves:
  - Current data fit:  combined ORCA-6 + IceCube DeepCore (3D, theta23-profiled)
  - Future projection: combined ORCA-Full + IC-Upgrade-7 (Asimov sensitivity)
Confidence levels are 1-dof thresholds (Δχ² = 1.0, 2.706, 3.841).

Inputs (assembled grid directories produced by the runners; NOT shipped with this repo):
  --data-dir   combined ORCA-6 + IC DeepCore data-fit dir (combined_cpt_3d_datafit_41x41x20;
               read via chi2_profiled.npy + dm_grid.npy + delta_grid.npy)
  --sens-dir   combined ORCA-Full + IC-Upgrade-7 sensitivity dir (combined_fine_31x31;
               read via chi2_grid.npy + dm_grid.npy + delta_grid.npy)
  --output-dir default: REPO_ROOT/outputs/figures (gitignored)

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/paper/scripts/plot_paper_cpt_1d_profiles.py``):
- The ``--base`` arg with two ``os.path.join(B, 'claude/3-CPT-violation/.../results/...')``
  constructions is replaced by explicit ``--data-dir`` / ``--sens-dir`` args.
- ``--output-dir`` default (a SRC-relative ``claude/3-CPT-violation/paper/plots/combined_fit``)
  -> ``REPO_ROOT/outputs/figures``.
- Added the ``analysis.lib.paths`` bootstrap for ``REPO_ROOT``.
- No physics / profiling / CL logic changed.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

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
    'axes.titlesize': 18,
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

# 1-dof confidence-level thresholds
CL = [(1.000, r'$1\sigma$'), (2.706, '90%'), (3.841, '95%')]
THR_90 = 2.706


def profiled_1d(base_dir, chi2_name):
    """Return (delta[1e-3 eV^2], dchi2_profiled_over_dm)."""
    dm = np.load(os.path.join(base_dir, 'dm_grid.npy'))
    delta = np.load(os.path.join(base_dir, 'delta_grid.npy')) * 1e3
    chi2 = np.load(os.path.join(base_dir, chi2_name))
    ax_dm = 0 if chi2.shape[0] == len(dm) else 1
    prof = np.nanmin(chi2, axis=ax_dm)
    order = np.argsort(delta)
    d = delta[order]
    y = prof[order] - np.nanmin(prof)
    return d, y


def smooth(d, y, n=4001):
    f = interp1d(d, y, kind='cubic')
    dd = np.linspace(d.min(), d.max(), n)
    return dd, f(dd)


def crossings(dd, yy, thr):
    """90%-style two-sided crossings about the minimum."""
    imin = int(np.argmin(yy))
    left, ly = dd[:imin], yy[:imin]
    right, ry = dd[imin:], yy[imin:]
    li = np.where(ly >= thr)[0]
    ri = np.where(ry >= thr)[0]
    neg = left[li[-1]] if len(li) else None
    pos = right[ri[0]] if len(ri) else None
    return neg, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Combined ORCA-6 + IC DeepCore data-fit dir "
                         "(combined_cpt_3d_datafit_41x41x20)")
    ap.add_argument("--sens-dir", required=True,
                    help="Combined ORCA-Full + IC-Upgrade-7 sensitivity dir "
                         "(combined_fine_31x31)")
    ap.add_argument("--output-dir",
                    default=str(paths.REPO_ROOT / "outputs" / "figures"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = args.data_dir
    sens_dir = args.sens_dir

    d_d, y_d = profiled_1d(data_dir, 'chi2_profiled.npy')
    d_s, y_s = profiled_1d(sens_dir, 'chi2_grid.npy')
    dd_d, yy_d = smooth(d_d, y_d)
    dd_s, yy_s = smooth(d_s, y_s)

    C_DATA = '#c0392b'   # current data fit  (red)
    C_SENS = '#1f77b4'   # future projection (blue)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # CPT-symmetric reference
    ax.axvline(0, color='gray', ls='-', lw=0.8, alpha=0.4, zorder=0)

    # CL threshold guide lines
    for thr, lab in CL:
        ax.axhline(thr, color='0.55', ls='--', lw=0.9, zorder=1)
        ax.text(0.984, thr + 0.06, lab, transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=12, color='0.4')

    # Shade each 90% allowed region (under the 90% line)
    for dd, yy, c in [(dd_d, yy_d, C_DATA), (dd_s, yy_s, C_SENS)]:
        neg, pos = crossings(dd, yy, THR_90)
        if neg is not None and pos is not None:
            ax.fill_between([neg, pos], 0, THR_90, color=c, alpha=0.10, zorder=0)
            for x in (neg, pos):
                ax.plot([x, x], [0, THR_90], color=c, ls=':', lw=1.2, alpha=0.7, zorder=2)

    # Profiles
    ax.plot(dd_s, yy_s, '-', color=C_SENS, lw=2.6, zorder=5,
            label=r'ORCA-Full + IC-Upgrade-7 (projection)')
    ax.plot(dd_d, yy_d, '-', color=C_DATA, lw=2.6, zorder=5,
            label=r'ORCA-6 + IC DeepCore (data)')

    # 90% CL numbers
    nd, pd = crossings(dd_d, yy_d, THR_90)
    ns, ps = crossings(dd_s, yy_s, THR_90)
    txt = (r'$90\%$ CL on $\delta\Delta m^2_{31}$' '\n'
           rf'data: $[{nd:+.2f},\,{pd:+.2f}]$' '\n'
           rf'proj.: $[{ns:+.2f},\,{ps:+.2f}]$' '\n'
           r'($\times10^{-3}$ eV$^2$)')
    ax.text(0.022, 0.035, txt, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=12.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', alpha=0.92))

    ax.set_xlabel(r'$\delta\Delta m^2_{31} \equiv \Delta\bar{m}^2_{31} - \Delta m^2_{31}$'
                  r' [$\times 10^{-3}$ eV$^2$]')
    ax.set_ylabel(r'$\Delta\chi^2$ (profiled over $\Delta m^2_{31}$)')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 5.0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=1,
              framealpha=0.92, handlelength=1.6, handletextpad=0.5, borderpad=0.5)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(args.output_dir, f'cpt_1d_profiles_combined.{ext}')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved {out}")
    plt.close()

    # console summary
    for name, dd, yy in [('DATA (ORCA-6+IC-DC)', dd_d, yy_d),
                         ('PROJ (ORCA-Full+IC-Up-7)', dd_s, yy_s)]:
        n, p = crossings(dd, yy, THR_90)
        print(f"{name}: 90% CL delta in [{n:+.3f}, {p:+.3f}] x1e-3 eV^2")


if __name__ == "__main__":
    main()
