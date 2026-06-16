#!/usr/bin/env python3
"""
plot_fig7_time_evolution.py — Fig 7: CPT significance vs. year (time evolution).

Paper figure: Fig 7. Reads the two time-evolution JSONs produced by
``analysis/atmospheric/run_cpt_time_evolution.py`` (one per experiment), fits a
saturating ``a*log(1+b*T)`` Δχ² model, and projects the per-experiment and
combined significance at δΔm²₃₁ = 0.05e-3 eV² across calendar years
(ICUpgrade start 2026, ORCA-Full start 2030). Also writes a diagnostic figure of
the data points + log/poly2 fits.

Inputs (runner outputs, NOT shipped with this repo):
  --icup-json      IC-Upgrade-7 time-evolution JSON  (e.g. icupgrade_bb.json)
  --orca-json      ORCA-Full EvtMC time-evolution JSON (e.g. orcafull_evtmc_bb.json)
  --output-dir     default: REPO_ROOT/outputs/figures (gitignored)

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/sensitivity-scans/ORCA-full/scripts/plot_cpt_time_evolution_paper.py``):
- ``RESULTS_DIR``/``TEVO_DIR`` derived from ``__file__``'s SRC subproject layout (which
  assumed the inputs lived in ``../results/cpt_time_evolution/`` and wrote outputs into
  ``../results/``) are replaced by explicit ``--icup-json`` / ``--orca-json`` inputs and
  an ``--output-dir`` (default under ``REPO_ROOT/outputs/figures``).
- Added the ``analysis.lib.paths`` bootstrap for ``REPO_ROOT``.
- No physics / fit logic changed (TARGET_DELTA, start years, log/poly2 fits intact).
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# --- repo bootstrap: make `analysis.lib` importable -------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.lib import paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

# Use serif fonts for paper
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 19,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

TARGET_DELTA = 0.05e-3  # eV^2

ICUP_START = 2026
ORCA_START = 2030  # full KM3NeT/ORCA completion (IN2P3 status report, Feb 2026)
YEARS = np.linspace(2026, 2036, 100)  # smooth curve


def load_time_evolution(filepath):
    with open(filepath) as f:
        data = json.load(f)
    delta_values = np.array(data['delta_values'])
    idx = np.argmin(np.abs(delta_values - TARGET_DELTA))
    exposures = []
    dchi2_at_target = []
    for entry in data['exposures']:
        exposures.append(entry['exposure_yr'])
        dchi2 = np.array(entry['dchi2'])
        dchi2_at_target.append(dchi2[idx])
    exp_arr = np.array(exposures)
    dchi2_arr = np.array(dchi2_at_target)
    # Remove 8yr point (minimizer artifact)
    mask = exp_arr != 8.0
    return exp_arr[mask], dchi2_arr[mask]


def fit_log(exposures, dchi2_values):
    """Fit dchi2 = a * log(1 + b*T). Monotonic and saturating."""
    from scipy.optimize import curve_fit

    def model(T, a, b):
        return a * np.log(1 + b * T)

    popt, _ = curve_fit(model, exposures, dchi2_values, p0=[0.5, 1.0],
                        bounds=([0, 0], [np.inf, np.inf]))
    return popt  # [a, b]


def eval_log(coeffs, T):
    T = max(T, 0.0)
    return coeffs[0] * np.log(1 + coeffs[1] * T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icup-json", required=True,
                    help="IC-Upgrade-7 time-evolution JSON (run_cpt_time_evolution.py output)")
    ap.add_argument("--orca-json", required=True,
                    help="ORCA-Full EvtMC time-evolution JSON (run_cpt_time_evolution.py output)")
    ap.add_argument("--output-dir",
                    default=str(paths.REPO_ROOT / "outputs" / "figures"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    icup_exp, icup_dchi2 = load_time_evolution(args.icup_json)
    orca_exp, orca_dchi2 = load_time_evolution(args.orca_json)

    # Fit log model: dchi2 = a * log(1 + b*T) — monotonic & saturating
    icup_coeffs = fit_log(icup_exp, icup_dchi2)
    orca_coeffs = fit_log(orca_exp, orca_dchi2)
    print(f"ICUpgrade fit: dchi2 = {icup_coeffs[0]:.4f} * log(1 + {icup_coeffs[1]:.4f}*T)")
    print(f"ORCA-Full fit: dchi2 = {orca_coeffs[0]:.4f} * log(1 + {orca_coeffs[1]:.4f}*T)")

    sigma_icup = []
    sigma_orca = []
    sigma_combined = []

    for year in YEARS:
        lt_icup = max(0, year - ICUP_START)
        lt_orca = max(0, year - ORCA_START)
        d_icup = eval_log(icup_coeffs, lt_icup)
        d_orca = eval_log(orca_coeffs, lt_orca)
        sigma_icup.append(np.sqrt(max(d_icup, 0)))
        sigma_orca.append(np.sqrt(max(d_orca, 0)))
        sigma_combined.append(np.sqrt(max(d_icup + d_orca, 0)))

    sigma_icup = np.array(sigma_icup)
    sigma_orca = np.array(sigma_orca)
    sigma_combined = np.array(sigma_combined)

    # ---- Diagnostic figure: data points + both fits ----
    fig_diag, (axd1, axd2) = plt.subplots(1, 2, figsize=(13, 5))

    T_smooth = np.linspace(0, 10, 200)

    for axd, exp_data, dchi2_data, coeffs, label, color in [
        (axd1, icup_exp, icup_dchi2, icup_coeffs, 'IceCube Upgrade', '#1f77b4'),
        (axd2, orca_exp, orca_dchi2, orca_coeffs, 'ORCA-Full EvtMC', '#d62728'),
    ]:
        # Raw data points
        sigma_data = np.sqrt(np.maximum(dchi2_data, 0))
        axd.plot(exp_data, sigma_data, 'o', color=color, ms=7, zorder=10, label='Data')

        # Log fit
        sigma_log = np.array([np.sqrt(max(eval_log(coeffs, t), 0)) for t in T_smooth])
        axd.plot(T_smooth, sigma_log, '-', color=color, lw=2, label=f'log fit: $a\\log(1+bT)$')

        # Poly2 fit for comparison
        A = np.column_stack([exp_data, exp_data**2])
        poly_coeffs, _, _, _ = np.linalg.lstsq(A, dchi2_data, rcond=None)
        sigma_poly = np.array([np.sqrt(max(poly_coeffs[0]*t + poly_coeffs[1]*t**2, 0)) for t in T_smooth])
        axd.plot(T_smooth, sigma_poly, '--', color=color, lw=1.5, alpha=0.7, label=f'poly2 fit: $aT+bT^2$')

        axd.set_xlabel('Exposure [years]', fontsize=13)
        axd.set_ylabel(r'$\sigma$', fontsize=14)
        axd.set_title(label, fontsize=13)
        axd.legend(fontsize=10)
        axd.set_xlim(-0.3, 10.5)
        axd.set_ylim(0, 1.2)
        axd.axhline(1, color='#cccccc', ls='-', lw=0.6)
        axd.tick_params(labelsize=11)

    fig_diag.tight_layout()
    outpath_diag = os.path.join(args.output_dir, 'cpt_time_evolution_fit_comparison.png')
    fig_diag.savefig(outpath_diag, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath_diag}")

    # ---- Paper figure ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Sigma band
    ax.axhline(1, color='#cccccc', ls='-', lw=0.6, zorder=0)
    ax.text(2036.15, 1, r'$1\sigma$', fontsize=13, color='#888888',
            va='center', ha='left')

    # ORCA-Full start line
    ax.axvline(ORCA_START, color='#d62728', ls=':', lw=0.8, alpha=0.4, zorder=0)
    ax.text(ORCA_START + 0.1, 1.42, 'ORCA start', fontsize=12, color='#d62728',
            alpha=0.6, rotation=90, va='top', ha='left')

    # Individual experiments (lighter)
    ax.plot(YEARS, sigma_icup, '-', color='#1f77b4', lw=1.5, alpha=0.5,
            label='IC-Upgrade-7')
    ax.plot(YEARS, sigma_orca, '-', color='#d62728', lw=1.5, alpha=0.5,
            label='ORCA-Full')

    # Combined (thick, prominent)
    ax.plot(YEARS, sigma_combined, '-', color='#2ca02c', lw=3.0,
            label='Combined', zorder=5)

    # Fill under combined for subtle emphasis
    ax.fill_between(YEARS, 0, sigma_combined, color='#2ca02c', alpha=0.06, zorder=0)

    ax.set_xlabel('Year', fontsize=19)
    ax.set_ylabel(r'Significance at $\delta\Delta m^2_{31} = 0.05 \times 10^{-3}\;$eV$^2$ [$\sigma$]',
                  fontsize=16)

    ax.set_xlim(2026, 2036)
    ax.set_ylim(0, 1.5)
    ax.set_xticks(np.arange(2026, 2037, 1))

    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    outpath = os.path.join(args.output_dir, 'cpt_time_evolution_paper.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    outpath_png = os.path.join(args.output_dir, 'cpt_time_evolution_paper.png')
    fig.savefig(outpath_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    print(f"Saved: {outpath_png}")


if __name__ == "__main__":
    main()
