#!/usr/bin/env python3
"""
plot_fig3_nova_t2k_manifolds.py — Fig 3 (CP-CPT degeneracy manifolds, T2K & NOvA).

Original source:
    AtmNuDataFit/claude/3-CPT-violation/paper/plots/NOvA+T2K/plot_flux_weighted_degeneracy.py

Figure:
    Fig 3 — "CP-CPT degeneracy manifolds for T2K (295 km) and NOvA (810 km)",
    two-panel figure:
      Left  ("Image"):    iso-<DP> contours for a CP-conserving truth with CPT
                          violation (δCP=0°, δΔm²=1e-3 eV²).
      Right ("Preimage"): CPT-violating truth that reconciles the T2K and NOvA
                          observed δCP under CPT conservation
                          (reconciling δCP≈109°, δΔm²≈1.48e-3 eV²).

Physics:
    Cervera analytic ν_e appearance probability with matter effects (NOT
    GLoBES / nuSQuIDS). All amplitudes (α, ᾱ, β, β̄), the matter potential, the
    δCP × δΔm² grid, baselines (T2K 295 km, NOvA 810 km), densities, and the
    truth/observed points are preserved verbatim from the original. Everything is
    computed inline, so this figure needs no separate runner.

Inputs / environment:
    - NOvA / T2K GLoBES flux .dat files, resolved via the repo's path helper:
        paths.globes_config("nova/NOvAplus.dat")
        paths.globes_config("nova/NOvAminus.dat")
        paths.globes_config("t2k/JHFplus.dat")
        paths.globes_config("t2k/JHFminus.dat")
      (shipped under REPO/configs/globes/{nova,t2k}/).
    - No nuSQuIDS / NUSQUIDS_DATA_PATH needed (analytic).

De-hardcoded vs. original:
    - Flux directories: original built NOVA_FLUX_DIR / T2K_FLUX_DIR from a
      SCRIPT_DIR.parents[2] / "CP-CPT-degeneracy" / {NOvA,T2K} / "configs"
      relative walk. Now resolved via paths.globes_config("nova/<f>") /
      paths.globes_config("t2k/<f>"). Same four flux files, same columns.
    - Output directory: original wrote next to the script (OUTPUT_DIR=SCRIPT_DIR).
      Now argparse `--output-dir`, default REPO_ROOT/outputs/fig3_nova_t2k_manifolds
      (gitignored).
    - Added repo-root sys.path bootstrap + `from analysis.lib import paths`.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np

# Repo-root bootstrap so `from analysis.lib import paths` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.lib import paths

# Compatibility: _trapz was added in NumPy 2.0; fall back to np.trapz
_trapz = getattr(np, 'trapezoid', np.trapz)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Flux configs ship under REPO/configs/globes/{nova,t2k}/ and are resolved via
# the repo path helper (see module docstring). Output dir is CLI-configurable.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    default=str(paths.REPO_ROOT / "outputs" / "fig3_nova_t2k_manifolds"),
    help="Directory for the output figure(s) "
         "(default: REPO_ROOT/outputs/fig3_nova_t2k_manifolds, gitignored).",
)
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Oscillation parameters  (NuFIT 5.2, Normal Ordering)
# ---------------------------------------------------------------------------
th12 = np.arcsin(np.sqrt(0.307))
th13 = np.arcsin(np.sqrt(0.0220))
th23 = np.arcsin(np.sqrt(0.545))

s23, c23 = np.sin(th23), np.cos(th23)
sin2_2th12 = np.sin(2 * th12)
sin2_2th13 = np.sin(2 * th13)

dm21 = 7.53e-5   # eV^2
dm31 = 2.453e-3  # eV^2

# ---------------------------------------------------------------------------
# Cervera analytic DP = P(nu_mu -> nu_e) - P(nubar_mu -> nubar_e)
# ---------------------------------------------------------------------------
def delta_P(E_GeV, L_km, rho_gcc, dcp_rad, dm31_nu, dm31_nubar):
    """Cervera formula for the CP asymmetry DP."""
    A = 7.63e-5 * rho_gcc * E_GeV          # matter potential
    D31 = 1.2669 * dm31_nu * L_km / E_GeV
    D31b = 1.2669 * dm31_nubar * L_km / E_GeV
    D21 = 1.2669 * dm21 * L_km / E_GeV
    Ah = A / dm31_nu
    Ahb = A / dm31_nubar

    # neutrino
    alp = s23 * sin2_2th13 * np.sin((1 - Ah) * D31) / (1 - Ah)
    bet = c23 * sin2_2th12 * np.sin(Ah * D21) / Ah
    P_nu = alp**2 + bet**2 + 2 * alp * bet * np.cos(D31 + dcp_rad)

    # antineutrino
    alpb = s23 * sin2_2th13 * np.sin((1 + Ahb) * D31b) / (1 + Ahb)
    betb = c23 * sin2_2th12 * np.sin(Ahb * D21) / Ahb
    P_nb = alpb**2 + betb**2 + 2 * alpb * betb * np.cos(D31b - dcp_rad)

    return P_nu - P_nb


def P_mue(E_GeV, L_km, rho_gcc, dcp_rad, dm31_val):
    """Cervera P(nu_mu -> nu_e) with matter effects."""
    A = 7.63e-5 * rho_gcc * E_GeV
    D31 = 1.2669 * dm31_val * L_km / E_GeV
    D21 = 1.2669 * dm21 * L_km / E_GeV
    Ah = A / dm31_val
    alp = s23 * sin2_2th13 * np.sin((1 - Ah) * D31) / (1 - Ah)
    bet = c23 * sin2_2th12 * np.sin(Ah * D21) / Ah
    return alp**2 + bet**2 + 2 * alp * bet * np.cos(D31 + dcp_rad)


def P_mue_bar(E_GeV, L_km, rho_gcc, dcp_rad, dm31_val):
    """Cervera P(nubar_mu -> nubar_e) with matter effects."""
    A = 7.63e-5 * rho_gcc * E_GeV
    D31 = 1.2669 * dm31_val * L_km / E_GeV
    D21 = 1.2669 * dm21 * L_km / E_GeV
    Ah = A / dm31_val
    alpb = s23 * sin2_2th13 * np.sin((1 + Ah) * D31) / (1 + Ah)
    betb = c23 * sin2_2th12 * np.sin(Ah * D21) / Ah
    return alpb**2 + betb**2 + 2 * alpb * betb * np.cos(D31 - dcp_rad)


# ---------------------------------------------------------------------------
# Load NOvA flux files (GLoBES format)
# ---------------------------------------------------------------------------
def load_globes_flux(filepath):
    """
    Load a GLoBES flux .dat file.
    Columns: E(GeV)  nue  numu  nutau  nuebar  numubar  nutaubar
    Returns (E, flux_numu, flux_numubar) arrays.
    """
    data = np.loadtxt(filepath)
    E = data[:, 0]
    flux_numu = data[:, 2]       # column index 2 = numu
    flux_numubar = data[:, 5]    # column index 5 = numubar
    return E, flux_numu, flux_numubar


def load_nova_fluxes():
    """Load NOvA nu-mode and nubar-mode fluxes, combine for appearance analysis."""
    fplus = paths.globes_config("nova/NOvAplus.dat")
    fminus = paths.globes_config("nova/NOvAminus.dat")
    E_p, numu_p, numubar_p = load_globes_flux(fplus)
    E_m, numu_m, numubar_m = load_globes_flux(fminus)
    assert np.allclose(E_p, E_m), "Energy grids must match"
    # For appearance: nu flux from nu-mode, nubar flux from nubar-mode
    return E_p, numu_p, numubar_m


def load_t2k_fluxes():
    """Load T2K nu-mode and nubar-mode fluxes, combine for appearance analysis."""
    fplus = paths.globes_config("t2k/JHFplus.dat")
    fminus = paths.globes_config("t2k/JHFminus.dat")
    E_p, numu_p, numubar_p = load_globes_flux(fplus)
    E_m, numu_m, numubar_m = load_globes_flux(fminus)
    # T2K flux files may have different energy grids; interpolate if needed
    if not np.allclose(E_p, E_m):
        from scipy.interpolate import interp1d
        E_common = E_p
        numubar_m = interp1d(E_m, numubar_m, bounds_error=False, fill_value=0.0)(E_common)
    else:
        E_common = E_p
    # For appearance: nu flux from nu-mode, nubar flux from nubar-mode
    return E_common, numu_p, numubar_m

# ---------------------------------------------------------------------------
# Flux-weighted <DP> functions
# ---------------------------------------------------------------------------
def avg_delta_P_flux(E_arr, flux_numu, flux_numubar,
                     L_km, rho_gcc, dcp_rad, dm31_nubar,
                     dm31_nu=dm31, E_range=(0.5, 5.0)):
    """
    Flux-weighted energy average of DP using actual beam fluxes.

    Uses separate physical flux weights for neutrino and antineutrino:
        <DP> = [int Phi_numu(E)*P_nu(E) dE / int Phi_numu(E) dE]
             - [int Phi_numubar(E)*P_nubar(E) dE / int Phi_numubar(E) dE]

    This is the physically correct treatment: each polarity's appearance
    probability is weighted by its own beam flux spectrum.
    """
    # Select energy range
    mask = (E_arr >= E_range[0]) & (E_arr <= E_range[1])
    E = E_arr[mask]
    phi_nu = flux_numu[mask]
    phi_nubar = flux_numubar[mask]

    if len(E) < 2:
        return 0.0

    ndim = np.ndim(dcp_rad)
    E_v = E.reshape((-1,) + (1,) * ndim)
    phi_nu_v = phi_nu.reshape((-1,) + (1,) * ndim)
    phi_nubar_v = phi_nubar.reshape((-1,) + (1,) * ndim)

    # P(nu_mu -> nu_e) with dm31_nu, weighted by nu flux
    P_nu = P_mue(E_v, L_km, rho_gcc, dcp_rad, dm31_nu)
    avg_P_nu = _trapz(phi_nu_v * P_nu, E, axis=0) / _trapz(phi_nu, E)

    # P(nubar_mu -> nubar_e) with dm31_nubar, weighted by nubar flux
    P_nubar = P_mue_bar(E_v, L_km, rho_gcc, dcp_rad, dm31_nubar)
    avg_P_nubar = _trapz(phi_nubar_v * P_nubar, E, axis=0) / _trapz(phi_nubar, E)

    return avg_P_nu - avg_P_nubar


def avg_delta_P_gaussian(L_km, rho_gcc, E_peak, E_sigma,
                         dcp_rad, dm31_nubar, dm31_nu=dm31, n_E=80):
    """Gaussian-flux-weighted energy average of DP (notebook convention)."""
    Ev = np.linspace(max(0.1, E_peak - 3 * E_sigma),
                     E_peak + 3 * E_sigma, n_E)
    flux = np.exp(-0.5 * ((Ev - E_peak) / E_sigma) ** 2)
    ndim = np.ndim(dcp_rad)
    Ev_v = Ev.reshape((-1,) + (1,) * ndim)
    fl_v = flux.reshape((-1,) + (1,) * ndim)
    dP = delta_P(Ev_v, L_km, rho_gcc, dcp_rad, dm31_nu, dm31_nubar)
    return _trapz(fl_v * dP, Ev, axis=0) / _trapz(flux, Ev)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Load fluxes
nova_E, nova_phi_numu, nova_phi_numubar = load_nova_fluxes()
t2k_E, t2k_phi_numu, t2k_phi_numubar = load_t2k_fluxes()

experiments = {
    'T2K': dict(
        L_km=295., rho_gcc=2.6,
        flux_type='real',
        E_arr=t2k_E, phi_numu=t2k_phi_numu, phi_numubar=t2k_phi_numubar,
        E_range=(0.2, 1.5),  # T2K narrow-band beam peaks at ~0.6 GeV
        # T2K best-fit dCP under CPT conservation (T2K 2020)
        dcp_obs_deg=-108.,
        color='#1f77b4', ls='-', lw=2.5,
        label='T2K',
    ),
    r'NO$\nu$A': dict(
        L_km=810., rho_gcc=2.84,
        flux_type='real',
        E_arr=nova_E, phi_numu=nova_phi_numu, phi_numubar=nova_phi_numubar,
        E_range=(0.5, 5.0),
        # NOvA best-fit dCP under CPT conservation (NOvA 2021)
        dcp_obs_deg=+21.,
        color='#d62728', ls='--', lw=2.5,
        label=r'NO$\nu$A',
    ),
}


def compute_avg_dP(exp, dcp_rad, dm31_nubar, dm31_nu=dm31):
    """Dispatch to appropriate flux-averaging function."""
    if exp['flux_type'] == 'gaussian':
        return avg_delta_P_gaussian(
            exp['L_km'], exp['rho_gcc'],
            exp['E_peak'], exp['E_sigma'],
            dcp_rad, dm31_nubar, dm31_nu=dm31_nu)
    else:
        return avg_delta_P_flux(
            exp['E_arr'], exp['phi_numu'], exp['phi_numubar'],
            exp['L_km'], exp['rho_gcc'],
            dcp_rad, dm31_nubar, dm31_nu=dm31_nu,
            E_range=exp.get('E_range', (0.5, 5.0)))


# ---------------------------------------------------------------------------
# Grid scan
# ---------------------------------------------------------------------------
print("Computing grids...")
N_dcp, N_dDel = 600, 400
dcp_vals = np.linspace(-np.pi, np.pi, N_dcp)
dDel_vals = np.linspace(-3e-3, 3e-3, N_dDel)
DCP, DDEL = np.meshgrid(dcp_vals, dDel_vals, indexing='ij')

grids = {}
for name, exp in experiments.items():
    grids[name] = compute_avg_dP(exp, DCP, dm31 + DDEL)
    print(f"  {name:12s} done  [{grids[name].min():.5f}, {grids[name].max():.5f}]")

# ---------------------------------------------------------------------------
# Piercing-point finder (for Image panel)
# ---------------------------------------------------------------------------
def find_piercings(exp, dcp_true, dDel_true):
    """Find dCP values where the degeneracy manifold crosses dDelta = 0."""
    dP_truth = float(compute_avg_dP(
        exp, dcp_true, dm31 + dDel_true))
    dP_axis = compute_avg_dP(exp, dcp_vals, dm31)
    diff = dP_axis - dP_truth
    crossings = []
    for i in range(len(diff) - 1):
        if np.sign(diff[i]) != np.sign(diff[i + 1]) and np.sign(diff[i]) != 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            pp = dcp_vals[i] + frac * (dcp_vals[i + 1] - dcp_vals[i])
            crossings.append(np.degrees(pp))
    return crossings

# ---------------------------------------------------------------------------
# Intersection finder (for Preimage panel)
# ---------------------------------------------------------------------------
def find_intersection(experiments, grids, targets):
    """Find the truth (dCP_true, dDel_true) that reconciles both experiments."""
    names = list(experiments.keys())
    scales = {name: max(abs(targets[name]), 1e-8) for name in names}
    residual_norm = sum(
        np.abs(grids[name] - targets[name]) / scales[name]
        for name in names
    )
    idx = np.unravel_index(np.argmin(residual_norm), residual_norm.shape)
    return (np.degrees(dcp_vals[idx[0]]),
            dDel_vals[idx[1]],
            residual_norm[idx])

# ---------------------------------------------------------------------------
# Compute targets for preimage (observed dCP under CPT conservation)
# ---------------------------------------------------------------------------
targets = {}
for name, exp in experiments.items():
    dcp_obs = np.radians(exp['dcp_obs_deg'])
    target = float(compute_avg_dP(exp, dcp_obs, dm31))
    targets[name] = target
    print(f"  {name:12s} DP target = {target:.5f}")

# Find intersection
dcp_int, dDel_int, res_int = find_intersection(experiments, grids, targets)
print(f"\nReconciling truth: dCP_true = {dcp_int:.1f} deg, "
      f"dDm2_true = {dDel_int*1e3:.2f} x 10^-3 eV^2  (residual={res_int:.4f})")

# ---------------------------------------------------------------------------
# Figure: two subplots
# ---------------------------------------------------------------------------
print("\nPlotting...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

# ===========================================================================
# LEFT PANEL: Image (degeneracy manifolds from a CP-conserving truth)
# ===========================================================================
dcp_true = 0.0        # CP-conserving truth
dDel_true = 1.0e-3    # eV^2

# Background: T2K DP field
cf = ax1.contourf(
    np.degrees(dcp_vals), dDel_vals * 1e3,
    grids['T2K'].T,
    levels=50, cmap='RdBu_r', alpha=0.30,
)
cbar1 = fig.colorbar(cf, ax=ax1, pad=0.02, fraction=0.04)
cbar1.set_label(r'$\langle\Delta P\rangle_{\rm T2K}$', fontsize=18)
cbar1.ax.tick_params(labelsize=11)

# Degeneracy manifolds and piercing points
pierce_dict = {}
for name, exp in experiments.items():
    dP_truth = float(compute_avg_dP(exp, dcp_true, dm31 + dDel_true))
    ax1.contour(
        np.degrees(dcp_vals), dDel_vals * 1e3,
        grids[name].T,
        levels=[dP_truth],
        colors=[exp['color']], linestyles=[exp['ls']],
        linewidths=exp['lw'],
    )
    piercings = find_piercings(exp, dcp_true, dDel_true)
    pierce_dict[name] = piercings
    for pp in piercings:
        ax1.plot(pp, 0, 'o', ms=8, color=exp['color'],
                 mec='k', mew=0.7, zorder=7, clip_on=False)

# Annotations for imposter dCP values
annotation_cfg = {
    'T2K': dict(ytext=0.55, va='bottom', color='#1f77b4'),
    r'NO$\nu$A': dict(ytext=-0.55, va='top', color='#d62728'),
}
for name, cfg in annotation_cfg.items():
    for pp in pierce_dict.get(name, []):
        if abs(pp - np.degrees(dcp_true)) < 8:
            continue
        ax1.annotate(
            f'${pp:.0f}' + r'^{\circ}$',
            xy=(pp, 0), xytext=(pp, cfg['ytext']),
            fontsize=11, color=cfg['color'], fontweight='bold',
            ha='center', va=cfg['va'],
            arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=1.1),
            zorder=8,
        )

# Truth star
ax1.plot(np.degrees(dcp_true), dDel_true * 1e3,
         '*', ms=16, color='gold', mec='k', mew=0.8, zorder=9)
ax1.annotate(
    r'truth ($\delta_{\rm CP}=0^{\circ}$)',
    xy=(np.degrees(dcp_true), dDel_true * 1e3),
    xytext=(40, 1.7),
    fontsize=11, color='k',
    arrowprops=dict(arrowstyle='->', color='k', lw=1.0),
)

# CPT-conserving axis
ax1.axhline(0, color='k', lw=1.0, ls=':', alpha=0.6, zorder=2)
ax1.text(174, 0.13, 'CPT conserved', fontsize=10, color='k', alpha=0.65, ha='right')

# Legend
legend1 = [
    Line2D([0], [0], color=exp['color'], lw=exp['lw'], ls=exp['ls'],
           label=f"{exp['label']} manifold")
    for name, exp in experiments.items()
] + [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markeredgecolor='k', markersize=13,
           label=r'Truth: $\delta_{\rm CP}=0^{\circ}$, $\delta\Delta=10^{-3}\,{\rm eV}^2$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='k', markersize=8,
           label=r'Imposter $\delta_{\rm CP}$ (CPT-cons. fit)'),
]
ax1.legend(handles=legend1, loc='lower left', framealpha=0.92, fontsize=12)

# Axes
ax1.set_xlabel(r'$\delta_{\rm CP}^{\rm true}\;[^{\circ}]$')
ax1.set_ylabel(r'$\delta\Delta m^2_{31}\;[10^{-3}\;{\rm eV}^2]$')
ax1.set_xlim(-180, 180)
ax1.set_ylim(-3, 3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(30))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

# ===========================================================================
# RIGHT PANEL: Preimage (reconciling truth)
# ===========================================================================

# Background: normalised residual
names = list(experiments.keys())
scales = {name: max(abs(targets[name]), 1e-8) for name in names}
residual_norm = sum(
    np.abs(grids[name] - targets[name]) / scales[name]
    for name in names
)
bg = np.clip(residual_norm, 0, 2.0)
cf2 = ax2.contourf(
    np.degrees(dcp_vals), dDel_vals * 1e3,
    bg.T, levels=40, cmap='Purples_r', alpha=0.30,
)
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.02, fraction=0.04)
cbar2.set_label('Normalised residual\n(darker = closer)', fontsize=18)
cbar2.ax.tick_params(labelsize=10)

# Preimage curves
for name, exp in experiments.items():
    ax2.contour(
        np.degrees(dcp_vals), dDel_vals * 1e3,
        grids[name].T,
        levels=[targets[name]],
        colors=[exp['color']], linestyles=[exp['ls']],
        linewidths=exp['lw'],
    )
    # Mark the observed point on dDelta=0
    ax2.plot(exp['dcp_obs_deg'], 0,
             's', ms=9, color=exp['color'], mec='k', mew=0.7,
             zorder=7, clip_on=False)
    offset = 0.30 if exp['ls'] == '-' else -0.30
    ax2.annotate(
        f"${exp['dcp_obs_deg']:.0f}" + r"^{\circ}$",
        xy=(exp['dcp_obs_deg'], 0),
        xytext=(exp['dcp_obs_deg'], offset),
        fontsize=11, color=exp['color'], fontweight='bold',
        ha='center', va='bottom' if offset > 0 else 'top',
        arrowprops=dict(arrowstyle='->', color=exp['color'], lw=1.0),
        zorder=8,
    )

# Intersection star
ax2.plot(dcp_int, dDel_int * 1e3,
         '*', ms=18, color='gold', mec='k', mew=0.9, zorder=9)

ax2.annotate(
    f'Reconciling truth\n'
    f'$\\delta_{{\\rm CP}}^{{\\rm true}}={dcp_int:.0f}' + r'^{\circ}$' + '\n'
    f'$\\delta\\Delta={dDel_int*1e3:.2f}' + r'\times10^{-3}\,{\rm eV}^2$',
    xy=(dcp_int, dDel_int * 1e3),
    xytext=(-140, 2.1),
    fontsize=10, color='k',
    arrowprops=dict(arrowstyle='->', color='k', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85),
    zorder=10,
)

# CPT-conserving axis
ax2.axhline(0, color='k', lw=1.0, ls=':', alpha=0.6, zorder=2)
ax2.text(174, 0.13, 'CPT conserved', fontsize=10, color='k', alpha=0.65, ha='right')

# Legend
legend2 = [
    Line2D([0], [0], color=exp['color'], lw=exp['lw'], ls=exp['ls'],
           label=f"{exp['label']} preimage")
    for name, exp in experiments.items()
] + [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markeredgecolor='k', markersize=9,
           label=r'Observed $\delta_{\rm CP}$ (CPT-cons. fit)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markeredgecolor='k', markersize=14,
           label='Reconciling truth'),
]
ax2.legend(handles=legend2, loc='lower left', framealpha=0.92, fontsize=10)

# Axes
ax2.set_xlabel(r'$\delta_{\rm CP}^{\rm true}\;[^{\circ}]$')
ax2.set_ylabel(r'$\delta\Delta m^2_{31}\;[10^{-3}\;{\rm eV}^2]$')
ax2.tick_params(labelleft=True)
ax2.set_xlim(-180, 180)
ax2.set_ylim(-3, 3)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(30))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout()
outpath = OUTPUT_DIR / "flux_weighted_degeneracy.png"
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# Also save PDF
outpath_pdf = OUTPUT_DIR / "flux_weighted_degeneracy.pdf"
plt.savefig(outpath_pdf, dpi=200, bbox_inches='tight')
print(f"Saved: {outpath_pdf}")

plt.close()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n--- Image panel (left) ---")
print(f"Truth: dCP = {np.degrees(dcp_true):.0f} deg, dDm2 = {dDel_true*1e3:.1f} x 10^-3 eV^2")
print("Imposter dCP values (CPT-conserving axis):")
for name, pps in pierce_dict.items():
    non_trivial = [p for p in pps if abs(p - np.degrees(dcp_true)) > 8]
    print(f"  {name:12s}: {[f'{p:.1f} deg' for p in non_trivial]}")

print("\n--- Preimage panel (right) ---")
print(f"T2K observed: dCP = {experiments['T2K']['dcp_obs_deg']:.0f} deg")
print(f"NOvA observed: dCP = {experiments[r'NO$' + chr(92) + 'nu$A']['dcp_obs_deg']:.0f} deg")
print(f"Reconciling truth: dCP = {dcp_int:.1f} deg, dDm2 = {dDel_int*1e3:.2f} x 10^-3 eV^2")
print(f"Residual: {res_int:.4f}")

# Note on flux weighting
print("\n--- Flux weighting ---")
print("NOvA: actual GLoBES flux (NOvAplus.dat / NOvAminus.dat)")
print("T2K:  actual GLoBES flux (JHFplus.dat / JHFminus.dat)")
print("Both use separate nu/nubar flux weights for P_nu and P_nubar.")
