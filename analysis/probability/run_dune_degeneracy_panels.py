#!/usr/bin/env python3
"""
run_dune_degeneracy_panels.py — nuSQuIDS runner for Fig 1 (CP-CPT degeneracy at DUNE).

Original source:
    AtmNuDataFit/claude/5-DUNE-GLoBES/scripts/plot_paper_degeneracy_panels.py
    (the embedded nuSQuIDS `propagate()` + npz-writing logic of that combined
     compute+plot script; the plotting half is now plots/plot_fig1_dune_dp_decomposition.py)

Figure:
    Fig 1 — "CP-CPT degeneracy at DUNE", 3-panel ΔP decomposition (CP / CPT / matter).

What it does:
    Runs 10 nuSQuIDS 3-flavor propagations (constant-density matter, L=1285 km,
    ρ=2.848) at the truth (δ_CP=-112°, Δ=1.0e-3 eV²) and imposter (δ_CP=-84°,
    Δ=0) points, builds the ΔP = P(νμ→νe) - P(ν̄μ→ν̄e) decomposition, and writes
    a .npz that plots/plot_fig1_dune_dp_decomposition.py consumes.

Inputs / environment:
    - nuSQuIDS (Python module `nuSQuIDS`) must be importable.
    - $NUSQUIDS_DATA_PATH must point at the nuSQuIDS cross-section / data dir
      (read by nuSQuIDS itself; NOT hardcoded here).

De-hardcoded vs. original:
    - Added repo-root sys.path bootstrap + `from analysis.lib import paths`
      (so the runner lives inside the package; no behavioural change).
    - Output npz path: argparse `--output-dir`, default under
      REPO_ROOT/outputs/fig1_dune_degeneracy/ (gitignored). Original defaulted
      output-dir to "." — same npz filename ("degeneracy_panels_data.npz").
    - No absolute paths were present in the compute half; physics constants,
      grid, and the 10 propagations are preserved verbatim.
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np

# Repo-root bootstrap so `from analysis.lib import paths` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths

# ============================================================================
# Parameters
# ============================================================================
S12 = 0.310; S13 = 0.02240; S23 = 0.582
DM21 = 7.39e-5; DM31 = 2.525e-3

TH12 = math.asin(math.sqrt(S12))
TH13 = math.asin(math.sqrt(S13))
TH23 = math.asin(math.sqrt(S23))

L_KM = 1284.9; RHO = 2.848; Y_E = 0.5

DELTA_DM31 = 1e-3           # CPT violation strength
DCP_TRUE = -112.0            # true δ_CP (degrees)
DCP_IMPOSTER = -84.0         # imposter δ_CP (degrees)

DCP_TRUE_RAD = math.radians(DCP_TRUE)
DCP_IMPOSTER_RAD = math.radians(DCP_IMPOSTER)

DM31_BAR = DM31 + DELTA_DM31  # ν̄ mass splitting with CPT violation


def propagate(energies, dm31_val, dcp_rad, nu_type_str, use_matter):
    """Single nuSQuIDS propagation. Returns P(νμ→νe) and P(νμ→νμ)."""
    import nuSQuIDS as nsq

    units = nsq.Const()
    energies_eV = energies * units.GeV
    L_eV = L_KM * units.km

    if nu_type_str == 'nu':
        nu_type = nsq.NeutrinoType.neutrino
    else:
        nu_type = nsq.NeutrinoType.antineutrino

    nusq = nsq.nuSQUIDS(energies_eV, 3, nu_type, False)
    nusq.Set_MixingAngle(0, 1, TH12)
    nusq.Set_MixingAngle(0, 2, TH13)
    nusq.Set_MixingAngle(1, 2, TH23)
    nusq.Set_CPPhase(0, 2, dcp_rad)
    nusq.Set_SquareMassDifference(1, DM21)
    nusq.Set_SquareMassDifference(2, dm31_val)

    if use_matter:
        body = nsq.ConstantDensity(RHO, Y_E)
        track = nsq.ConstantDensity.Track(L_eV)
    else:
        body = nsq.Vacuum()
        track = nsq.Vacuum.Track(L_eV)

    nusq.Set_Body(body)
    nusq.Set_Track(track)

    n_e = len(energies)
    init_state = np.zeros((n_e, 3))
    init_state[:, 1] = 1.0  # start as νμ
    nusq.Set_initial_state(init_state, nsq.Basis.flavor)
    nusq.Set_rel_error(1e-9)
    nusq.Set_abs_error(1e-9)
    nusq.EvolveState()

    P_app = np.array([nusq.EvalFlavorAtNode(0, i) for i in range(n_e)])
    P_surv = np.array([nusq.EvalFlavorAtNode(1, i) for i in range(n_e)])

    return P_app, P_surv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(paths.REPO_ROOT / "outputs" / "fig1_dune_degeneracy"),
        help="Directory to write degeneracy_panels_data.npz (default: "
             "REPO_ROOT/outputs/fig1_dune_degeneracy, gitignored).",
    )
    parser.add_argument("--delta", type=float, default=DELTA_DM31)
    parser.add_argument("--dcp-true", type=float, default=DCP_TRUE, help="True dCP in degrees")
    parser.add_argument("--dcp-imposter", type=float, default=DCP_IMPOSTER, help="Imposter dCP in degrees")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    delta = args.delta
    dcp_true = math.radians(args.dcp_true)
    dcp_imp = math.radians(args.dcp_imposter)
    dm31_bar = DM31 + delta

    energies = np.geomspace(0.5, 20.0, 500)

    print(f"Computing oscillation probabilities...")
    print(f"  Δ = {delta:.1e} eV², dCP_true = {args.dcp_true}°, dCP_imp = {args.dcp_imposter}°")
    print(f"  Δm²₃₁ = {DM31:.4e}, Δm̄²₃₁ = {dm31_bar:.4e}")

    # ================================================================
    # 6 propagations at truth δ_CP (for decomposition)
    # ================================================================
    # 1. ν in matter with dm31 (truth)
    P_nu_matter_app, _ = propagate(energies, DM31, dcp_true, 'nu', True)
    # 2. ν̄ in matter with dm31_bar (truth + CPT)
    P_nubar_matter_cpt_app, _ = propagate(energies, dm31_bar, dcp_true, 'nubar', True)
    # 3. ν in vacuum with dm31
    P_nu_vac_app, _ = propagate(energies, DM31, dcp_true, 'nu', False)
    # 4. ν̄ in vacuum with dm31 (CPT-conserving reference)
    P_nubar_vac_app, _ = propagate(energies, DM31, dcp_true, 'nubar', False)
    # 5. ν̄ in vacuum with dm31_bar (CPT-violating vacuum)
    P_nubar_vac_cpt_app, _ = propagate(energies, dm31_bar, dcp_true, 'nubar', False)
    # 6. ν̄ in matter with dm31 (CPT-conserving matter, for reference)
    P_nubar_matter_app, _ = propagate(energies, DM31, dcp_true, 'nubar', True)

    # ΔP decomposition at truth
    dP_total_true = P_nu_matter_app - P_nubar_matter_cpt_app
    dP_CP_true = P_nu_vac_app - P_nubar_vac_app
    dP_CPT_true = P_nubar_vac_app - P_nubar_vac_cpt_app
    dP_matter_true = dP_total_true - dP_CP_true - dP_CPT_true

    print(f"  ΔP decomposition at truth computed")

    # ================================================================
    # 4 propagations at imposter δ_CP (no CPT violation)
    # ================================================================
    # 7. ν in matter with dm31, imposter δ_CP
    P_nu_matter_imp_app, _ = propagate(energies, DM31, dcp_imp, 'nu', True)
    # 8. ν̄ in matter with dm31 (no CPT), imposter δ_CP
    P_nubar_matter_imp_app, _ = propagate(energies, DM31, dcp_imp, 'nubar', True)
    # 9. ν in vacuum with dm31, imposter δ_CP
    P_nu_vac_imp_app, _ = propagate(energies, DM31, dcp_imp, 'nu', False)
    # 10. ν̄ in vacuum with dm31 (no CPT), imposter δ_CP
    P_nubar_vac_imp_app, _ = propagate(energies, DM31, dcp_imp, 'nubar', False)

    dP_total_imp = P_nu_matter_imp_app - P_nubar_matter_imp_app
    dP_CP_imp = P_nu_vac_imp_app - P_nubar_vac_imp_app

    dP_CP_plus_CPT_true = dP_CP_true + dP_CPT_true

    print(f"  Imposter probabilities computed")

    # Save data
    outfile = os.path.join(args.output_dir, 'degeneracy_panels_data.npz')
    np.savez(outfile,
             energies=energies,
             dP_total_true=dP_total_true,
             dP_CP_true=dP_CP_true,
             dP_CPT_true=dP_CPT_true,
             dP_matter_true=dP_matter_true,
             dP_total_imp=dP_total_imp,
             dP_CP_imp=dP_CP_imp,
             dP_CP_plus_CPT_true=dP_CP_plus_CPT_true,
             delta=delta,
             dcp_true_deg=args.dcp_true,
             dcp_imp_deg=args.dcp_imposter,
             dm31=DM31, dm31_bar=dm31_bar)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
