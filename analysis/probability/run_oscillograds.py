#!/usr/bin/env python3
"""
Atmospheric CPT oscillograds — CHIC compute step.

Paper figure: Fig 6 (2x3 oscillograds of the oscillation-probability derivatives
wrt Δm²₃₁ and δCP vs neutrino energy and zenith). Computes the five derivative
maps with the CHIC code (``pychic_earth``) and writes them to an npz that
``plots/plot_fig6_oscillograds.py`` draws.

Backend: CHIC (``pychic_earth`` / ``CHICEARTHDIFF``, Fernández-Menéndez). This is
an external dependency — see docs/INSTALL.md. No MC/data files are needed.

Ported from: ~/Downloads/atm_CPT_oscillograds.py (compute half). The original was
a single compute+``plt.show()`` script with no file I/O; here the CHIC compute is
split off and its arrays are saved (so re-plotting does not recompute the
500x500 derivative grids), and the grid resolution / output dir are CLI args.
The physics — PREM42 Earth model, the five derivative channels, the neutrino /
antineutrino modes, and the IO point Δm²₃₁ = -2.51e-3 — is unchanged.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths  # noqa: E402

import pychic_earth as pe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="CHIC oscillograds compute (paper Fig 6)")
    ap.add_argument("--output-dir",
                    default=str(paths.REPO_ROOT / "outputs" / "fig6_oscillograds"),
                    help="where to write oscillograds_data.npz")
    ap.add_argument("--n-energy", type=int, default=500)
    ap.add_argument("--n-zenith", type=int, default=500)
    ap.add_argument("--e-min", type=float, default=1.0, help="GeV")
    ap.add_argument("--e-max", type=float, default=10.0, help="GeV")
    ap.add_argument("--model", default="PREM42", help="CHIC Earth model")
    ap.add_argument("--dm231-io", type=float, default=-2.51e-3,
                    help="Δm²₃₁ for the inverted-ordering antineutrino panel")
    args = ap.parse_args()

    energies = np.linspace(args.e_min, args.e_max, args.n_energy)
    zens = np.linspace(-1.0, 0.0, args.n_zenith)

    dP_dm_mm_no_nu = np.zeros((energies.size, zens.size))
    dP_dcp_mm_no_nu = np.zeros((energies.size, zens.size))
    dP_dm_me_no_nu = np.zeros((energies.size, zens.size))
    dP_dcp_me_no_nub = np.zeros((energies.size, zens.size))
    dP_dcp_me_io_nub = np.zeros((energies.size, zens.size))

    # Neutrinos, normal ordering: ∂Δm²₃₁ (μ→μ, μ→e) and ∂δCP (μ→μ).
    dch = pe.CHICEARTHDIFF(model=args.model)
    for i, cz in enumerate(zens):
        for j, E in enumerate(energies):
            dPx = dch.compute_oscillations_derivatives("dm231", E, cz)
            dP_dm_mm_no_nu[i, j] = dPx[1, 1]
            dP_dm_me_no_nu[i, j] = dPx[0, 1]
            dPx = dch.compute_oscillations_derivatives("dcp", E, cz)
            dP_dcp_mm_no_nu[i, j] = dPx[1, 1]
    del dch

    # Antineutrinos, normal ordering: ∂δCP (μ̄→ē).
    dch = pe.CHICEARTHDIFF(mode="antineutrino", model=args.model)
    for i, cz in enumerate(zens):
        for j, E in enumerate(energies):
            dPx = dch.compute_oscillations_derivatives("dcp", E, cz)
            dP_dcp_me_no_nub[i, j] = dPx[0, 1]

    # Antineutrinos, inverted ordering: ∂δCP (μ̄→ē).
    dch.update_dm231(args.dm231_io)
    for i, cz in enumerate(zens):
        for j, E in enumerate(energies):
            dPx = dch.compute_oscillations_derivatives("dcp", E, cz)
            dP_dcp_me_io_nub[i, j] = dPx[0, 1]

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "oscillograds_data.npz")
    np.savez(
        out,
        energies=energies, zens=zens,
        dP_dm_mm_no_nu=dP_dm_mm_no_nu,
        dP_dm_me_no_nu=dP_dm_me_no_nu,
        dP_dcp_mm_no_nu=dP_dcp_mm_no_nu,
        dP_dcp_me_no_nub=dP_dcp_me_no_nub,
        dP_dcp_me_io_nub=dP_dcp_me_io_nub,
        model=args.model, dm231_io=args.dm231_io,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
