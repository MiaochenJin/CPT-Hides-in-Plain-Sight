#!/usr/bin/env python3
"""
plot_fig4_dune_cpt_bias.py — Paper Fig 4: DUNE CP--CPT bias combined figure.

Ported from:
    claude/3-CPT-violation/paper/plots/DUNE/plot_dune_combined_v2.py

Combined DUNE figure (horizontal layout):
  Left panel: 2D grid heatmap of best-fit δ_CP with ΔP contours (real DUNE flux)
  Right panel: two band plots stacked vertically (δ_CP = -112° top, δ_CP = 0° bottom)

The ΔP contour physics (Cervera-style _P_mue / _P_mue_bar, oscillation
constants, flux integration window) is preserved verbatim.

Env / inputs (none shipped with the repo — pass them explicitly):
    --input-dcp112  combined NPZ for truth δ_CP = -112° (cpt_bias_dcp-112_method_d.npz),
                    produced by assemble_dune_method_d.py
    --input-dcp0    combined NPZ for truth δ_CP =   0°  (cpt_bias_dcp0_method_d.npz)
    --input-grid    grid_bestfit.npz (best-fit δ_CP over the dcp_true × dDel grid)
    --flux-dir      directory with DUNE GLoBES FD flux tables
                    (flux_dune_neutrino_FD_globes.txt,
                     flux_dune_antineutrino_FD_globes.txt)
    --smoother-dir  OPTIONAL override for the directory containing
                    smoothed_band_panel.py. By default the co-located
                    plots/smoothed_band_panel.py (shipped in this repo) is used;
                    only pass this to point at a different copy.
    --out           output figure stem (default: REPO_ROOT/outputs/fig4_dune_cpt_bias)
                    PNG + PDF are written.

De-hardcode notes:
  - The original derived RESULTS / FLUX_DIR / NPZ paths from the script's
    location inside the cluster source tree (Path(__file__).parents[...]).
    Those are now required argparse inputs (--input-dcp112/--input-dcp0/
    --input-grid/--flux-dir) with no cluster-assuming default.
  - The output was written next to the script (PLOTS dir); it now defaults under
    REPO_ROOT/outputs/ (gitignored), overridable via --out.
  - The original imported draw_smoothed_band_panel/FILL/FILL_ALPHA from a sibling
    cluster module ``smooth_right_panel/smoothed_band_panel.py``. That helper is
    now ported and shipped co-located at plots/smoothed_band_panel.py and is
    imported by default (no SRC tree needed); --smoother-dir is an optional
    override. Note: smoothed_band_panel additionally requires scikit-image
    (skimage).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Co-locate the shipped smoothed_band_panel.py (this plots/ dir) on sys.path so
# the default import below resolves without pointing at the SRC tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.lib import paths  # noqa: E402

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
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
import matplotlib.gridspec as gridspec

Y_LO, Y_HI = -1.25, 1.75

THRESH_90CL = 2.71
THRESH_3SIG = 9.0
THRESH_5SIG = 25.0

DCP_MEAS_LABEL = r'$\delta_{CP}^{\rm meas}$ [$\pi$]'
DELTA_LABEL = r'$\Delta\bar{m}^2_{31} - \Delta m^2_{31}$ [$\times 10^{-3}$ eV$^2$]'

# Oscillation constants
_s12_2 = 0.310; _s13_2 = 0.02240; _s23_2 = 0.582
_dm21 = 7.39e-5; _dm31 = 2.525e-3
_th12 = math.asin(math.sqrt(_s12_2))
_th13 = math.asin(math.sqrt(_s13_2))
_th23 = math.asin(math.sqrt(_s23_2))
_s23 = math.sin(_th23); _c23 = math.cos(_th23)
_sin2_2th13 = math.sin(2*_th13); _sin2_2th12 = math.sin(2*_th12)


def _P_mue(E, L, rho, dcp, dm31v):
    A = 7.63e-5 * rho * E
    D31 = 1.2669 * dm31v * L / E
    D21 = 1.2669 * _dm21 * L / E
    Ah = A / dm31v
    alp = _s23 * _sin2_2th13 * np.sin((1 - Ah) * D31) / (1 - Ah)
    bet = _c23 * _sin2_2th12 * np.sin(Ah * D21) / Ah
    return alp**2 + bet**2 + 2 * alp * bet * np.cos(D31 + dcp)


def _P_mue_bar(E, L, rho, dcp, dm31v):
    A = 7.63e-5 * rho * E
    D31 = 1.2669 * dm31v * L / E
    D21 = 1.2669 * _dm21 * L / E
    Ah = A / dm31v
    alpb = _s23 * _sin2_2th13 * np.sin((1 + Ah) * D31) / (1 + Ah)
    betb = _c23 * _sin2_2th12 * np.sin(Ah * D21) / Ah
    return alpb**2 + betb**2 + 2 * alpb * betb * np.cos(D31 - dcp)


try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def load_dune_fluxes(flux_dir):
    fhc = np.loadtxt(Path(flux_dir) / "flux_dune_neutrino_FD_globes.txt")
    rhc = np.loadtxt(Path(flux_dir) / "flux_dune_antineutrino_FD_globes.txt")
    E_fhc = fhc[:, 0]
    E_rhc = rhc[:, 0]
    numu_fhc = fhc[:, 2]
    numubar_rhc = rhc[:, 5]
    if not np.allclose(E_fhc, E_rhc):
        from scipy.interpolate import interp1d
        numubar_rhc = interp1d(E_rhc, numubar_rhc, bounds_error=False,
                               fill_value=0.0)(E_fhc)
    return E_fhc, numu_fhc, numubar_rhc


def compute_dune_dP_grid(dcp_deg_arr, dDel_arr, flux_dir):
    E_arr, phi_nu, phi_nubar = load_dune_fluxes(flux_dir)
    L, rho = 1284.9, 2.848
    mask = (E_arr >= 0.5) & (E_arr <= 10.0)
    E = E_arr[mask]
    pnu = phi_nu[mask]
    pnub = phi_nubar[mask]

    n_dcp = len(dcp_deg_arr)
    n_ddel = len(dDel_arr)
    result = np.zeros((n_dcp, n_ddel))

    for i, dcp_d in enumerate(dcp_deg_arr):
        dcp_r = np.radians(dcp_d)
        for j, ddel in enumerate(dDel_arr):
            Pnu = _P_mue(E, L, rho, dcp_r, _dm31)
            Pnub = _P_mue_bar(E, L, rho, dcp_r, _dm31 + ddel)
            avg_nu = _trapz(pnu * Pnu, E) / _trapz(pnu, E)
            avg_nub = _trapz(pnub * Pnub, E) / _trapz(pnub, E)
            result[i, j] = avg_nu - avg_nub
    return result


def main():
    parser = argparse.ArgumentParser(description="Paper Fig 4: DUNE CP-CPT bias combined figure.")
    parser.add_argument('--input-dcp112', required=True,
                        help='Combined NPZ for truth dCP = -112 deg '
                             '(cpt_bias_dcp-112_method_d.npz from assemble_dune_method_d.py).')
    parser.add_argument('--input-dcp0', required=True,
                        help='Combined NPZ for truth dCP = 0 deg '
                             '(cpt_bias_dcp0_method_d.npz from assemble_dune_method_d.py).')
    parser.add_argument('--input-grid', required=True,
                        help='grid_bestfit.npz (best-fit dCP over dcp_true x dDel grid).')
    parser.add_argument('--flux-dir', required=True,
                        help='Directory with DUNE GLoBES FD flux tables '
                             '(flux_dune_neutrino_FD_globes.txt, flux_dune_antineutrino_FD_globes.txt).')
    parser.add_argument('--smoother-dir', default=None,
                        help='OPTIONAL override for the directory containing smoothed_band_panel.py. '
                             'Defaults to the co-located plots/smoothed_band_panel.py shipped here.')
    parser.add_argument('--out', default=None,
                        help='Output stem (PNG+PDF). Default: REPO_ROOT/outputs/fig4_dune_cpt_bias')
    args = parser.parse_args()

    out_stem = Path(args.out) if args.out else (paths.REPO_ROOT / "outputs" / "fig4_dune_cpt_bias")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Compute ΔP grid
    # ============================================================
    print("Computing DUNE ΔP grid for contour overlay...")
    dcp_fine = np.linspace(-180, 180, 200)
    dDel_fine = np.linspace(Y_LO * 1e-3, Y_HI * 1e-3, 150)
    dP_grid = compute_dune_dP_grid(dcp_fine, dDel_fine, args.flux_dir)
    print("  done.")

    # ============================================================
    # Load data
    # ============================================================
    NPZ_112 = args.input_dcp112
    NPZ_0 = args.input_dcp0
    data_112 = np.load(NPZ_112, allow_pickle=True)
    data_0 = np.load(NPZ_0, allow_pickle=True)
    grid = np.load(args.input_grid, allow_pickle=True)

    # smoothed-panel renderer (ported, co-located at plots/smoothed_band_panel.py
    # and already on sys.path). --smoother-dir optionally overrides the location.
    if args.smoother_dir:
        sys.path.insert(0, str(Path(args.smoother_dir).resolve()))
    from smoothed_band_panel import draw_smoothed_band_panel, FILL, FILL_ALPHA  # noqa: E402

    # ============================================================
    # Combined figure — horizontal layout
    # ============================================================
    # Original: figsize=(11, 13), top=heatmap, bottom=[band_-112(1/3) | band_0(2/3)]
    # New: one row. Left=heatmap, Right=two band plots stacked vertically.
    # Original heatmap was ~11 wide x 6.5 tall. Band row was ~11 wide x 6.5 tall.
    # Now both sit side by side: total ~22 wide x 6.5 tall.

    fig = plt.figure(figsize=(22, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # --- Left: grid heatmap ---
    ax_grid = fig.add_subplot(gs[0])

    dcp_true = grid['dcp_true_deg']
    dDel_1e3 = grid['dDel_true_eV2'] * 1e3
    dcp_bf = grid['dcp_bestfit_deg']

    im = ax_grid.pcolormesh(dcp_true, dDel_1e3, dcp_bf.T,
                            cmap='twilight_shifted', vmin=-180, vmax=180,
                            shading='nearest', rasterized=True)
    cbar = fig.colorbar(im, ax=ax_grid, pad=0.02, fraction=0.046)
    cbar.set_label(r'Best-fit $\delta_{CP}^{\rm meas}$ [°]', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax_grid.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)

    # Overlay ΔP contours (real DUNE flux)
    cs = ax_grid.contour(dcp_fine, dDel_fine * 1e3, dP_grid.T,
                         levels=12, colors='white', linewidths=0.8,
                         alpha=0.6, zorder=6)
    ax_grid.clabel(cs, inline=True, fontsize=10, fmt='%.3f')

    # Vertical lines at truth δ_CP values
    ax_grid.axvline(-112, color='#2ca02c', ls='--', lw=2.5, alpha=0.9, zorder=7)
    ax_grid.axvline(0, color='#2ca02c', ls='--', lw=2.5, alpha=0.9, zorder=7)
    ax_grid.text(-108, Y_HI - 0.12, r'$-112°$', fontsize=15, color='#2ca02c',
                 ha='left', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    ax_grid.text(4, Y_HI - 0.12, r'$0°$', fontsize=15, color='#2ca02c',
                 ha='left', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    ax_grid.set_xlim(dcp_true[0], dcp_true[-1])
    ax_grid.set_ylim(Y_LO, Y_HI)
    ax_grid.xaxis.set_major_locator(MultipleLocator(60))
    ax_grid.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_grid.set_xlabel(r'$\delta_{CP}^{\rm true}$ [°]', fontsize=22)
    ax_grid.set_ylabel(DELTA_LABEL, fontsize=22)

    # --- Right: two band plots side by side (1:2 width ratio, same as original bottom) ---
    gs_band = gs[1].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.08)

    _SMOOTH_KW = dict(sigma=1.5, upsample=6, boundary_passes=30)

    ax_b1 = fig.add_subplot(gs_band[0])
    ax_b2 = fig.add_subplot(gs_band[1], sharey=ax_b1)

    draw_smoothed_band_panel(ax_b1, NPZ_112, truth_dcp_deg=-112.0,
                             xlim=(-1.0, 0.0), ylim=(Y_LO, Y_HI), **_SMOOTH_KW)
    ax_b1.set_xticks([-1.0, -0.5])  # 0.5 spacing; drop boundary 0.0 (collides with b2's -1.00)
    ax_b1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax_b1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_b1.set_title(r'Truth $\delta_{CP} = -112°$', fontsize=18)
    ax_b1.set_xlabel(DCP_MEAS_LABEL, fontsize=22)
    ax_b1.set_ylabel(DELTA_LABEL, fontsize=22)

    legend_handles = draw_smoothed_band_panel(ax_b2, NPZ_0, truth_dcp_deg=0.0,
                                              xlim=(-1.0, 1.0), ylim=(Y_LO, Y_HI),
                                              **_SMOOTH_KW)
    ax_b2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_b2.xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.setp(ax_b2.get_yticklabels(), visible=False)
    ax_b2.set_title(r'Truth $\delta_{CP} = 0°$', fontsize=18)
    ax_b2.set_xlabel(DCP_MEAS_LABEL, fontsize=22)

    # Legend placed horizontally BELOW the two band panels (outside the axes) so a
    # large font can be used without covering the confidence bands.
    fig.canvas.draw()  # finalize axes positions before anchoring the legend
    _pos1 = ax_b1.get_position()
    _pos2 = ax_b2.get_position()
    _xc = 0.5 * (_pos1.x0 + _pos2.x1)
    _ybot = min(_pos1.y0, _pos2.y0)
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(_xc, _ybot - 0.12), ncol=3, fontsize=16,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
               borderpad=0.5, labelspacing=0.5, framealpha=0.9)

    out = out_stem.with_suffix('.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")

    out_pdf = out_stem.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
