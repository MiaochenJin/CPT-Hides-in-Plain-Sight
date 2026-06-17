#!/usr/bin/env python3
"""
Atmospheric CPT oscillograds — plot (paper Fig 6).

Draws the 2x3 panel of oscillation-probability derivative maps from the npz
produced by ``analysis/probability/run_oscillograds.py``:
  top row    — ∂Δm²₃₁ P(μ→μ) and ∂Δm²₃₁ P(μ→e), neutrinos, NO
  bottom row — ∂δCP P(μ→μ) (ν, NO), ∂δCP P̄(μ→e) (ν̄, NO), ∂δCP P̄(μ→e) (ν̄, IO)

No CHIC dependency here (only numpy/matplotlib) — all the computation lives in
the runner. Ported from ~/Downloads/atm_CPT_oscillograds.py (plot half): the only
changes are reading the arrays from ``--input`` instead of computing them inline,
and saving to ``--out`` instead of ``plt.show()``. The styling, colormap, contour
levels, derivative scalings (Δm² ×1e-4 eV², δCP ×1 rad), panel layout, and
colorbar are unchanged.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.lib import paths  # noqa: E402

RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'figure.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
}
matplotlib.rcParams.update(RC_PARAMS)

cmap = "cividis"
deltam = 0.1e-3
deltadcp = 1


def main():
    ap = argparse.ArgumentParser(description="Oscillograds plot (paper Fig 6)")
    ap.add_argument("--input", required=True,
                    help="oscillograds_data.npz from run_oscillograds.py")
    ap.add_argument("--out", default=str(paths.REPO_ROOT / "outputs" / "fig6_oscillograds.png"))
    args = ap.parse_args()

    d = np.load(args.input)
    energies, zens = d["energies"], d["zens"]
    dP_dm_mm_no_nu = d["dP_dm_mm_no_nu"]
    dP_dm_me_no_nu = d["dP_dm_me_no_nu"]
    dP_dcp_mm_no_nu = d["dP_dcp_mm_no_nu"]
    dP_dcp_me_no_nub = d["dP_dcp_me_no_nub"]
    dP_dcp_me_io_nub = d["dP_dcp_me_io_nub"]

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ED, CZ = np.meshgrid(energies, zens)
    levels = np.linspace(-1., 1., 31) / 8

    panels = [
        (gs[0, 0], dP_dm_mm_no_nu * deltam,    r'$\partial_{\Delta m^2_{31}}P_{\mu\rightarrow\mu}\times 10^{-4}~\mathrm{eV}^2$ (NO)'),
        (gs[0, 1], dP_dm_me_no_nu * deltam,    r'$\partial_{\Delta m^2_{31}}P_{\mu\rightarrow e}\times 10^{-4}~\mathrm{eV}^2$ (NO)'),
        (gs[1, 0], dP_dcp_mm_no_nu * deltadcp, r'$\partial_{\delta_{CP}}P_{\mu\rightarrow\mu}\times 1~\mathrm{rad}$ (NO)'),
        (gs[1, 1], dP_dcp_me_no_nub * deltadcp, r'$\partial_{\delta_{CP}}\overline{P}_{\mu\rightarrow e}\times 1~\mathrm{rad}$ (NO)'),
        (gs[1, 2], dP_dcp_me_io_nub * deltadcp, r'$\partial_{\delta_{CP}}\overline{P}_{\mu\rightarrow e}\times 1~\mathrm{rad}$ (IO)'),
    ]

    for spec, data, title in panels:
        ax = fig.add_subplot(spec)
        ax.contourf(ED, CZ, data, levels=levels, cmap=cmap, extend='both')
        ax.set_xlabel(r'$E_\nu$ (GeV)')
        ax.set_ylabel(r'$\cos\theta_\mathrm{zen}$')
        ax.set_title(title, fontsize=9)

    norm = Normalize(vmin=min(levels), vmax=max(levels))
    host_ax = fig.add_subplot(gs[0, 2])
    host_ax.set_visible(False)

    pos = host_ax.get_position()
    left = pos.x0 + 0.0 * pos.width
    bottom = pos.y0 + 0.05 * pos.height
    width = 0.05 * pos.width
    height = 0.90 * pos.height

    cbar_ax = fig.add_axes([left, bottom, width, height])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\partial_\zeta P_{\mu\rightarrow x} \times \delta$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
