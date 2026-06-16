#!/usr/bin/env python3
"""
plot_fig1_dune_dp_decomposition.py — plotter for Fig 1 (CP-CPT degeneracy at DUNE).

Original source:
    AtmNuDataFit/claude/5-DUNE-GLoBES/scripts/plot_paper_degeneracy_panels.py
    (the matplotlib half of that combined compute+plot script; the nuSQuIDS
     compute half is now analysis/probability/run_dune_degeneracy_panels.py).

Figure:
    Fig 1 — "CP-CPT degeneracy at DUNE", 3-panel ΔP decomposition.
      (a) ΔP decomposition at truth (δ_CP=-112°, Δ=1e-3): total / CP / CPT / matter
      (b) ΔP_total(truth) vs ΔP_total(imposter δ_CP=-84°), plus vacuum components
      (c) Residuals from (b)

Inputs / environment:
    - --input: path to degeneracy_panels_data.npz produced by
      analysis/probability/run_dune_degeneracy_panels.py. REQUIRED (the repo does
      not ship the npz). No cluster-assuming default.
    - No nuSQuIDS / env vars needed here (pure plotting from the npz).

De-hardcoded vs. original:
    - Split out of the combined script: this file no longer runs nuSQuIDS; it
      loads a precomputed npz instead of recomputing in-process.
    - Input npz: argparse `--input` (required), no default. Original recomputed
      the arrays in the same process and never read a file.
    - Output figure: argparse `--out`, default under
      REPO_ROOT/outputs/fig1_dune_degeneracy/ (gitignored). Original wrote
      "paper_degeneracy_3panel.png" into --output-dir (default ".").
    - All plot logic, labels, colors, axis limits, and the DUNE window
      (E_lo=1.3, E_hi=4.7) are preserved verbatim.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np

# Repo-root bootstrap so `from analysis.lib import paths` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.lib import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="Path to degeneracy_panels_data.npz "
             "(produced by analysis/probability/run_dune_degeneracy_panels.py).",
    )
    parser.add_argument(
        "--out",
        default=str(paths.REPO_ROOT / "outputs" / "fig1_dune_degeneracy"
                    / "paper_degeneracy_3panel.png"),
        help="Output figure path (default: "
             "REPO_ROOT/outputs/fig1_dune_degeneracy/paper_degeneracy_3panel.png, "
             "gitignored).",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Load precomputed arrays
    # ----------------------------------------------------------------
    d = np.load(args.input)
    energies = d['energies']
    dP_total_true = d['dP_total_true']
    dP_CP_true = d['dP_CP_true']
    dP_CPT_true = d['dP_CPT_true']
    dP_matter_true = d['dP_matter_true']
    dP_total_imp = d['dP_total_imp']
    dP_CP_imp = d['dP_CP_imp']
    dP_CP_plus_CPT_true = d['dP_CP_plus_CPT_true']
    delta = float(d['delta'])
    dcp_true_deg = float(d['dcp_true_deg'])
    dcp_imp_deg = float(d['dcp_imp_deg'])

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    # ================================================================
    # Plot
    # ================================================================
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 200,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 0.6], 'hspace': 0.08})

    # DUNE appearance window
    E_lo, E_hi = 1.3, 4.7

    # ---- Top: ΔP decomposition at truth ----
    ax = axes[0]
    ax.plot(energies, dP_total_true, 'k-', lw=2.5, label=r'$\Delta P_{\rm total}$')
    ax.plot(energies, dP_CP_true, '-', color='#2166ac', lw=1.8, label=r'$\Delta P_{CP}$')
    ax.plot(energies, dP_CPT_true, '-', color='#d62728', lw=1.8, label=r'$\Delta P_{CPT}$')
    ax.plot(energies, dP_matter_true, '-', color='#4daf4a', lw=1.8, label=r'$\Delta P_{\rm matter}$')
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.axvspan(E_lo, E_hi, alpha=0.08, color='gold')
    ax.set_ylabel(r'$\Delta P = P(\nu_\mu \to \nu_e) - P(\bar{\nu}_\mu \to \bar{\nu}_e)$')
    ax.set_title(r'$\Delta P$ decomposition at truth: $\delta_{CP}='
                 + f'{dcp_true_deg:.0f}' + r'^\circ$, $\Delta = '
                 + f'{delta*1e3:.1f}' + r'\times 10^{-3}$ eV$^2$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    # ---- Middle: truth vs imposter total ΔP, and vacuum components ----
    ax = axes[1]
    ax.plot(energies, dP_total_true, 'k-', lw=2.5,
            label=r'$\Delta P_{\rm total}$ (truth: $\delta_{CP}='
                  + f'{dcp_true_deg:.0f}' + r'^\circ$, $\Delta=' + f'{delta*1e3:.1f}' + r'\times 10^{-3}$)')
    ax.plot(energies, dP_total_imp, '--', color='#b2182b', lw=2.5,
            label=r"$\Delta P'_{\rm total}$ (imposter: $\delta_{CP}="
                  + f'{dcp_imp_deg:.0f}' + r"^\circ$, $\Delta=0$)")
    ax.plot(energies, dP_CP_plus_CPT_true, '-', color='#2166ac', lw=1.5,
            label=r'$\Delta P_{CP} + \Delta P_{CPT}$ (truth, vacuum)')
    ax.plot(energies, dP_CP_imp, '--', color='#6a5acd', lw=1.5,
            label=r"$\Delta P'_{CP}$ (imposter, vacuum)")
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.axvspan(E_lo, E_hi, alpha=0.08, color='gold')
    ax.set_ylabel(r'$\Delta P$')
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    # ---- Bottom: residuals ----
    ax = axes[2]
    res_total = dP_total_true - dP_total_imp
    res_vac = dP_CP_plus_CPT_true - dP_CP_imp
    ax.plot(energies, res_total, 'k-', lw=2,
            label=r'$\Delta P_{\rm total} - \Delta P^{\prime}_{\rm total}$')
    ax.plot(energies, res_vac, '-', color='#2166ac', lw=1.5,
            label=r'$(\Delta P_{CP}+\Delta P_{CPT}) - \Delta P^{\prime}_{CP}$')
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.axvspan(E_lo, E_hi, alpha=0.08, color='gold')
    ax.set_xlabel(r'Neutrino energy $E$ [GeV]')
    ax.set_ylabel('Residual')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.text(0.02, 0.92, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    for ax in axes:
        ax.set_xlim(0.5, 15)
        ax.set_xscale('log')

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

    # Print summary
    mask = (energies >= E_lo) & (energies <= E_hi)
    print(f"\nIn DUNE window [{E_lo}, {E_hi}] GeV:")
    print(f"  RMS(ΔP_total - ΔP'_total) = {np.sqrt(np.mean(res_total[mask]**2)):.4f}")
    print(f"  RMS(ΔP_vac - ΔP'_CP)      = {np.sqrt(np.mean(res_vac[mask]**2)):.4f}")
    print(f"  max|ΔP_total - ΔP'_total|  = {np.max(np.abs(res_total[mask])):.4f}")


if __name__ == "__main__":
    main()
