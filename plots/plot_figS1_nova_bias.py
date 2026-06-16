#!/usr/bin/env python3
"""
plot_figS1_nova_bias.py — Paper Suppl Fig 1: NOvA CP--CPT bias figure.

Ported from:
    claude/3-CPT-violation/paper/plots/NOvA/plot_nova_combined.py

Renders the NOvA best-fit δ_CP grid heatmap with real-flux ΔP contours overlaid.
The Cervera ΔP physics (_P_mue / _P_mue_bar, oscillation constants, NOvA
baseline/density and flux-integration window) is preserved verbatim.

Note: as in the original, the executed figure body draws only the single-panel
grid heatmap; the band-panel helpers (find_all_regions, draw_panel_xy, and the
imported draw_smoothed_band_panel) are retained for parity but are not invoked
when this script runs.

Env / inputs:
    --input-grid       grid_bestfit.npz (best-fit δ_CP over dcp_true × dDel grid).
                       NOT shipped with the repo; pass your NOvA grid output.
    --nova-config-dir  directory holding NOvAplus.dat / NOvAminus.dat. Defaults to
                       the committed configs/globes/nova in this repo.
    --smoother-dir     OPTIONAL override for the directory containing
                       smoothed_band_panel.py (now ported and co-located at
                       plots/smoothed_band_panel.py). Only relevant if you
                       re-enable the band panels; the default heatmap-only figure
                       does not call it.
    --out              output PNG path (default: REPO_ROOT/outputs/figS1_nova_bias.png)

De-hardcode notes:
  - The original derived RESULTS / NOVA_CONFIG_DIR / smoother dir from the
    script's location in the cluster source tree. --input-grid is now a required
    argparse arg (no default); --nova-config-dir defaults to the repo's committed
    NOvA configs; --smoother-dir is exposed but optional.
  - The original had a dead ``sys.path.insert(... "NOvA+T2K/scripts")`` before
    re-implementing the physics inline; that cluster-path insert is removed.
  - ``smoothed_band_panel`` (imported at module load in the original) is now
    ported and shipped co-located at plots/smoothed_band_panel.py (it needs
    scikit-image). Its import is kept lazy and guarded by --smoother-dir since
    the default heatmap-only figure does not use it; the co-located copy is on
    sys.path so re-enabling the band panels needs no SRC tree.
  - Output now defaults under REPO_ROOT/outputs/ (gitignored) via --out.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Co-locate the shipped smoothed_band_panel.py (this plots/ dir) on sys.path so
# the band-panel renderer resolves without pointing at the SRC tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.lib import paths  # noqa: E402

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
    'legend.fontsize': 13,
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


def draw_panel_xy(ax, data, true_dcp_deg):
    deltas = data['deltas']
    dcp_vals = data['dcp_vals']
    dcp_pi = np.degrees(dcp_vals) / 180.0
    true_dcp_pi = true_dcp_deg / 180.0
    deltas_1e3 = deltas * 1e3
    n_delta = len(deltas)

    hh = np.zeros(n_delta)
    for j in range(n_delta):
        if j == 0:
            hh[j] = (deltas_1e3[1] - deltas_1e3[0]) / 2
        elif j == n_delta - 1:
            hh[j] = (deltas_1e3[-1] - deltas_1e3[-2]) / 2
        else:
            hh[j] = min(deltas_1e3[j] - deltas_1e3[j-1],
                        deltas_1e3[j+1] - deltas_1e3[j]) / 2

    for j in range(n_delta):
        y = deltas_1e3[j]
        if y < Y_LO or y > Y_HI:
            continue
        tag = f"{deltas[j]:+.4e}".replace('+', 'p').replace('-', 'n').replace('.', 'd')
        key = f'chi2_best_{tag}'
        if key not in data:
            continue
        chi2 = data[key]
        chi2_min = chi2.min()
        dchi2 = chi2 - chi2_min
        h = hh[j]

        for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_3SIG):
            ax.add_patch(Rectangle((lo, y - h), hi - lo, 2 * h,
                         facecolor='#d62728', alpha=0.25, edgecolor='none', zorder=1))
        for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_90CL):
            ax.add_patch(Rectangle((lo, y - h), hi - lo, 2 * h,
                         facecolor='#ff7f0e', alpha=0.45, edgecolor='none', zorder=2))

        imin = np.argmin(chi2)
        ax.plot(dcp_pi[imin], y, 'k.', ms=4, zorder=5)

        rel_min_idx = argrelmin(chi2, order=5)[0]
        min_region_width = 3 * (dcp_pi[1] - dcp_pi[0])
        best_sec, best_sec_dc = None, np.inf
        for mi in rel_min_idx:
            if mi == imin:
                continue
            dc = chi2[mi] - chi2_min
            if 0.5 <= dc <= THRESH_3SIG:
                in_band = False
                for lo, hi in find_all_regions(dchi2, dcp_pi, THRESH_3SIG):
                    if (hi - lo) >= min_region_width and lo <= dcp_pi[mi] <= hi:
                        in_band = True
                        break
                if in_band and dc < best_sec_dc:
                    best_sec, best_sec_dc = mi, dc
        if best_sec is not None:
            ax.plot(dcp_pi[best_sec], y, '.', color='gray', ms=3, zorder=4)

    ax.axvline(true_dcp_pi, color='#2ca02c', ls='--', lw=2, alpha=0.7, zorder=3)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.3)
    ax.plot(true_dcp_pi, 0, 'ko', ms=6, zorder=10)


# ============================================================
# Cervera ΔP for NOvA contour overlay (real flux)
# ============================================================

# Import just the physics functions we need
# Re-implement minimally to avoid import side effects from the full script
def _load_globes_flux(filepath):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 2], data[:, 5]  # E, numu, numubar

def _load_nova_fluxes(nova_config_dir):
    fp = Path(nova_config_dir) / "NOvAplus.dat"
    fm = Path(nova_config_dir) / "NOvAminus.dat"
    Ep, numu_p, _ = _load_globes_flux(fp)
    Em, _, numubar_m = _load_globes_flux(fm)
    return Ep, numu_p, numubar_m

# Oscillation constants (same as degeneracy script)
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

def compute_nova_dP_grid(dcp_deg_arr, dDel_arr, nova_config_dir):
    """Compute flux-weighted ΔP = <P_nu>_Φν - <P_nubar>_Φν̄ on a 2D grid."""
    E_arr, phi_nu, phi_nubar = _load_nova_fluxes(nova_config_dir)
    L, rho = 810.0, 2.84
    mask = (E_arr >= 0.5) & (E_arr <= 5.0)
    E = E_arr[mask]
    pnu = phi_nu[mask]
    pnub = phi_nubar[mask]

    n_dcp = len(dcp_deg_arr)
    n_ddel = len(dDel_arr)
    result = np.zeros((n_dcp, n_ddel))

    for i, dcp_d in enumerate(dcp_deg_arr):
        dcp_r = np.radians(dcp_d)
        for j, ddel in enumerate(dDel_arr):
            dm31_nu = _dm31
            dm31_nubar = _dm31 + ddel
            Pnu = _P_mue(E, L, rho, dcp_r, dm31_nu)
            Pnub = _P_mue_bar(E, L, rho, dcp_r, dm31_nubar)
            avg_nu = _trapz(pnu * Pnu, E) / _trapz(pnu, E)
            avg_nub = _trapz(pnub * Pnub, E) / _trapz(pnub, E)
            result[i, j] = avg_nu - avg_nub
    return result


def main():
    parser = argparse.ArgumentParser(description="Paper Suppl Fig 1: NOvA CP-CPT bias figure.")
    parser.add_argument('--input-grid', required=True,
                        help='grid_bestfit.npz (best-fit dCP over dcp_true x dDel grid); '
                             'not shipped — pass your NOvA grid output.')
    parser.add_argument('--nova-config-dir',
                        default=str(paths.config_dir() / "globes" / "nova"),
                        help='Directory with NOvAplus.dat / NOvAminus.dat '
                             '(default: committed configs/globes/nova).')
    parser.add_argument('--smoother-dir', default=None,
                        help='OPTIONAL override for the directory containing smoothed_band_panel.py '
                             '(ported and co-located at plots/smoothed_band_panel.py). The default '
                             'heatmap-only figure does not use it.')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: REPO_ROOT/outputs/figS1_nova_bias.png)')
    args = parser.parse_args()

    out = Path(args.out) if args.out else (paths.REPO_ROOT / "outputs" / "figS1_nova_bias.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # smoothed-panel renderer (shared with DUNE figure) — only if a dir is given.
    # The default heatmap-only figure does not call it.
    if args.smoother_dir:
        sys.path.insert(0, str(Path(args.smoother_dir).resolve()))
        from smoothed_band_panel import draw_smoothed_band_panel, FILL, FILL_ALPHA  # noqa: F401

    print("Computing NOvA ΔP grid for contour overlay...")
    # Use a finer grid than the 25×25 GLoBES grid for smooth contours
    dcp_fine = np.linspace(-180, 180, 200)
    dDel_fine = np.linspace(Y_LO * 1e-3, Y_HI * 1e-3, 150)
    dP_grid = compute_nova_dP_grid(dcp_fine, dDel_fine, args.nova_config_dir)
    print("  done.")

    # ============================================================
    # Load data
    # ============================================================
    grid = np.load(args.input_grid, allow_pickle=True)

    # ============================================================
    # Single-panel figure: grid heatmap only
    # ============================================================
    fig, ax_grid = plt.subplots(figsize=(11, 6.5))

    dcp_true = grid['dcp_true_deg']
    dDel_1e3 = grid['dDel_true_eV2'] * 1e3
    dcp_bf = grid['dcp_bestfit_deg']

    im = ax_grid.pcolormesh(dcp_true, dDel_1e3, dcp_bf.T,
                            cmap='twilight_shifted', vmin=-180, vmax=180,
                            shading='nearest', rasterized=True)
    cbar = fig.colorbar(im, ax=ax_grid, pad=0.02, fraction=0.046)
    cbar.set_label(r'Best-fit $\delta_{CP}^{\rm meas}$ [°]', fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    ax_grid.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)

    # Overlay ΔP contours (real NOvA flux)
    n_levels = 12
    cs = ax_grid.contour(dcp_fine, dDel_fine * 1e3, dP_grid.T,
                         levels=n_levels, colors='white', linewidths=0.8,
                         alpha=0.6, zorder=6)
    ax_grid.clabel(cs, inline=True, fontsize=8, fmt='%.3f')

    # Vertical markers at reference δ_CP values
    ax_grid.axvline(-112, color='#2ca02c', ls='--', lw=2.5, alpha=0.9, zorder=7)
    ax_grid.axvline(0, color='#2ca02c', ls='--', lw=2.5, alpha=0.9, zorder=7)
    ax_grid.text(-108, Y_HI - 0.12, r'$-112°$', fontsize=12, color='#2ca02c',
                 ha='left', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    ax_grid.text(4, Y_HI - 0.12, r'$0°$', fontsize=12, color='#2ca02c',
                 ha='left', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    ax_grid.set_xlim(dcp_true[0], dcp_true[-1])
    ax_grid.set_ylim(Y_LO, Y_HI)
    ax_grid.xaxis.set_major_locator(MultipleLocator(60))
    ax_grid.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_grid.set_xlabel(r'$\delta_{CP}^{\rm true}$ [°]', fontsize=18)
    ax_grid.set_ylabel(DELTA_LABEL, fontsize=17)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
