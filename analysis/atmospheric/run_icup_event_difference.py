#!/usr/bin/env python3
"""
IceCube-Upgrade-7 expected-event distribution difference (CPT vs. standard).

Paper figure: Suppl Fig 3 (IceCube Upgrade row). Loads the IceCube Upgrade MC
CSV, computes oscillated expected events for the standard and CPT-violated
scenarios using nuSQuIDS with Honda flux, and writes the per-panel 2D histograms
(``icup_event_difference_{cascade,track}_{true,reco}_2d.npz``) that the
Suppl-Fig-3 4-panel plotter combines, plus standalone diagnostic PNGs.

Adapted from the IC DeepCore version. CPT truth: Dm231 = 2.511e-3 (neutrinos),
Dm̄231 = 2.0e-3 (antineutrinos, --dm31-bar). IC-Upgrade-7 exposure: 10 yr.

Config / env / inputs:
- ``--mc-file`` IceCube Upgrade MC CSV (``neutrino_mc.csv``). No cluster-assuming
  default; if ``$DATA_DIR`` is set,
  ``$DATA_DIR/IceCubeUpgradeNeutrinoMCDataRelease-2/events/neutrino_mc.csv`` is used as
  a convenience default, otherwise ``--mc-file`` is required.
- ``--output-dir`` for the npz + PNG outputs.
- Requires nuSQuIDS, nuflux, pandas, matplotlib. Env: NUSQUIDS_DATA_PATH (nuSQuIDS tables).
  (Does not import the Pynu fitter — oscillations are computed directly with nuSQuIDS.)

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/sensitivity-scans/ICUpgrade/scripts/plot_icup_event_difference.py``):
- ``MC_FILE = '/n/holylfs05/.../IceCubeUpgradeNeutrinoMCDataRelease-2/events/neutrino_mc.csv'``
  (module constant) -> ``--mc-file`` arg (default
  ``$DATA_DIR/IceCubeUpgradeNeutrinoMCDataRelease-2/events/neutrino_mc.csv`` when DATA_DIR set).
- Added the ``analysis.lib.paths`` bootstrap only to resolve that default; no other
  absolute path was present.
"""

import os
import sys
import time
import argparse
import numpy as np
from math import asin, sqrt
from itertools import repeat
from pathlib import Path

# --- repo bootstrap: make `analysis.lib` importable (for the $DATA_DIR default) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths

import pandas as pd
import nuSQuIDS as nsq
import nuflux
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# Oscillation parameters (NuFIT 5.2)
OSC_PARAMS = {
    'sin2_12': 0.303,
    'sin2_13': 0.022,
    'sin2_23': 0.572,
    'dm21': 7.41e-5,
    'dcp': 1.36,
}

DM31_STD = 2.511e-3
DM31_BAR_CPT = 2.0e-3

# IceCube Upgrade exposure
EXPOSURE_YEARS = 10.0
SECONDS_PER_YEAR = 3.15576e7


def default_mc_file():
    """$DATA_DIR-rooted default for the IC Upgrade MC CSV, or None."""
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return None
    return os.path.join(
        data_dir, "IceCubeUpgradeNeutrinoMCDataRelease-2", "events", "neutrino_mc.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ICUp event distribution difference (CPT vs standard)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mc-file", default=default_mc_file(),
                        help="IC Upgrade MC CSV (default: "
                             "$DATA_DIR/IceCubeUpgradeNeutrinoMCDataRelease-2/events/"
                             "neutrino_mc.csv if DATA_DIR set)")
    parser.add_argument("--dm31-bar", type=float, default=DM31_BAR_CPT,
                        help="CPT-violated Dm31_bar (default: 2.0e-3)")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of bins per axis for true-level (default: 20)")
    args = parser.parse_args()
    if not args.mc_file:
        parser.error("--mc-file is required (set it or export DATA_DIR)")
    return args


def make_honda_flux(cth_nodes, energy_nodes, n_flavors=3):
    """Build Honda atmospheric initial flux for nuSQUIDSAtm."""
    flux_model = nuflux.makeFlux('IPhonda2014_spl_solmin')
    flux = np.zeros((len(cth_nodes), len(energy_nodes), 2, n_flavors))

    for ic, cz in enumerate(cth_nodes):
        for ie, E in enumerate(energy_nodes):
            flux[ic, ie, 0, 0] = flux_model.getFlux(nuflux.NuE, E, cz)
            flux[ic, ie, 0, 1] = flux_model.getFlux(nuflux.NuMu, E, cz)
            flux[ic, ie, 0, 2] = 0.0
            flux[ic, ie, 1, 0] = flux_model.getFlux(nuflux.NuEBar, E, cz)
            flux[ic, ie, 1, 1] = flux_model.getFlux(nuflux.NuMuBar, E, cz)
            flux[ic, ie, 1, 2] = 0.0

    return flux


def setup_and_propagate(cth_nodes, energy_nodes, dm31, initial_flux):
    """Create nuSQUIDSAtm, set parameters, propagate."""
    UNITS = nsq.Const()

    osc = nsq.nuSQUIDSAtm(
        cth_nodes,
        energy_nodes * UNITS.GeV,
        3,
        nsq.NeutrinoType.both,
        False,
    )
    osc.Set_rel_error(1e-8)
    osc.Set_abs_error(1e-8)

    osc.Set_MixingAngle(0, 1, asin(sqrt(OSC_PARAMS['sin2_12'])))
    osc.Set_MixingAngle(0, 2, asin(sqrt(OSC_PARAMS['sin2_13'])))
    osc.Set_MixingAngle(1, 2, asin(sqrt(OSC_PARAMS['sin2_23'])))
    osc.Set_SquareMassDifference(1, OSC_PARAMS['dm21'])
    osc.Set_SquareMassDifference(2, dm31)
    osc.Set_CPPhase(0, 2, OSC_PARAMS['dcp'])

    osc.Set_initial_state(initial_flux, nsq.Basis.flavor)
    osc.EvolveState()

    return osc


def eval_events(osc, cz_arr, E_arr, flavor_list, neutype_list):
    """Evaluate oscillated flux for each MC event via EvalFlavor."""
    UNITS = nsq.Const()

    weights = list(map(
        osc.EvalFlavor,
        flavor_list,
        cz_arr.astype(float).tolist(),
        (E_arr * UNITS.GeV).astype(float).tolist(),
        neutype_list,
        repeat(True),
    ))
    return np.asarray(weights)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    livetime = EXPOSURE_YEARS * SECONDS_PER_YEAR
    dm31_bar = args.dm31_bar
    n_bins = args.n_bins

    # =========================================================================
    # 1. Load IceCube Upgrade MC CSV
    # =========================================================================
    print("Loading IceCube Upgrade MC CSV...")
    df = pd.read_csv(args.mc_file)
    print(f"  Total events: {len(df)}")

    E_true = df['true_energy'].values
    cz_true = np.cos(df['true_zenith'].values)
    E_reco = df['reco_energy'].values
    cz_reco = np.cos(df['reco_zenith'].values)
    pdg = df['pdg'].values.astype(int)
    mc_weight = df['weight'].values  # in m^2
    current_type = df['current_type'].values.astype(int)
    pid = df['pid'].values.astype(int)  # 0=cascade, 1=track

    # Weight conversion: m^2 -> cm^2 (factor 1e4)
    mc_weight_cm2 = mc_weight * 1e4

    # nuSQuIDS flavor index: 0=nue, 1=numu, 2=nutau
    flavor = (0.5 * np.abs(pdg) - 6).astype(np.uint32)
    # neutype: 0=neutrino, 1=antineutrino
    neutype = np.zeros(len(pdg), dtype=np.uint32)
    neutype[pdg < 0] = 1
    is_nubar = pdg < 0

    # PID-based topology (from data release classifier)
    is_track = pid == 1
    is_cascade = pid == 0

    # Also define true topology (physics-based, for true-level plots)
    is_track_true = (current_type == 1) & (np.abs(pdg) == 14)
    is_cascade_true = ~is_track_true

    print(f"  E_true range: [{E_true.min():.2f}, {E_true.max():.1f}] GeV")
    print(f"  cos(z)_true range: [{cz_true.min():.3f}, {cz_true.max():.3f}]")
    print(f"  E_reco range: [{E_reco.min():.2f}, {E_reco.max():.1f}] GeV")
    print(f"  cos(z)_reco range: [{cz_reco.min():.3f}, {cz_reco.max():.3f}]")
    print(f"  Neutrinos: {np.sum(~is_nubar)}, Antineutrinos: {np.sum(is_nubar)}")
    print(f"  PID tracks: {np.sum(is_track)}, PID cascades: {np.sum(is_cascade)}")
    print(f"  True tracks (CC numu): {np.sum(is_track_true)}, "
          f"True cascades: {np.sum(is_cascade_true)}")

    # =========================================================================
    # 2. Set up nuSQuIDS grid and Honda flux
    # =========================================================================
    E_nodes = np.geomspace(0.1, 10000.0, 300)
    cz_nodes = np.linspace(-1.0, 1.0, 80)

    print("\nBuilding Honda initial flux...")
    t0 = time.time()
    initial_flux = make_honda_flux(cz_nodes, E_nodes)
    print(f"  Done in {time.time()-t0:.1f}s")

    # =========================================================================
    # 3. Standard propagation (dm31 = 2.511e-3 for all)
    # =========================================================================
    print(f"\nStandard propagation (Dm31 = {DM31_STD:.4e})...")
    t0 = time.time()
    osc_std = setup_and_propagate(cz_nodes, E_nodes, DM31_STD, initial_flux)
    osc_w_std = eval_events(osc_std, cz_true, E_true,
                            flavor.tolist(), neutype.tolist())
    print(f"  Done in {time.time()-t0:.1f}s")

    expected_std = mc_weight_cm2 * livetime * osc_w_std
    print(f"  Total expected events: {expected_std.sum():.1f}")

    # =========================================================================
    # 4. CPT propagation (dm31_bar for antineutrinos only)
    # =========================================================================
    delta_val = DM31_STD - dm31_bar
    print(f"\nCPT propagation (Dm31_bar = {dm31_bar:.4e}, "
          f"Delta = {delta_val:.4e})...")
    t0 = time.time()
    osc_cpt = setup_and_propagate(cz_nodes, E_nodes, dm31_bar, initial_flux)
    osc_w_cpt_bar = eval_events(osc_cpt, cz_true, E_true,
                                flavor.tolist(), neutype.tolist())
    print(f"  Done in {time.time()-t0:.1f}s")

    # Combine: neutrinos keep standard weights, antineutrinos use CPT weights
    osc_w_cpt = osc_w_std.copy()
    osc_w_cpt[is_nubar] = osc_w_cpt_bar[is_nubar]

    expected_cpt = mc_weight_cm2 * livetime * osc_w_cpt
    print(f"  Total expected events (CPT): {expected_cpt.sum():.1f}")
    print(f"  Total difference: {expected_cpt.sum() - expected_std.sum():.1f}")

    # =========================================================================
    # 5. Bin into histograms — per topology, for both true and reco
    # =========================================================================

    # True-level binning (fine grid, ICUp energy range)
    E_edges_true = np.geomspace(1.0, 100.0, n_bins + 1)
    cz_edges_true = np.linspace(-1.0, 1.0, n_bins + 1)

    # Reco-level binning: ICUp native analysis bins (20 E x 10 cosZ)
    E_edges_reco = np.logspace(np.log10(1.0), np.log10(100.0), 21)
    cz_edges_reco = np.linspace(-1.0, 1.0, 11)

    # For true-level, use physics-based topology; for reco-level, use PID
    var_spaces = {
        'true': {
            'E': E_true, 'cz': cz_true,
            'E_edges': E_edges_true, 'cz_edges': cz_edges_true,
            'topologies': {
                'cascade': is_cascade_true,
                'track': is_track_true,
            },
            'topo_labels': {
                'cascade': r'Cascades (NC + CC $\nu_e$/$\nu_\tau$)',
                'track': r'Tracks (CC $\nu_\mu$)',
            },
        },
        'reco': {
            'E': E_reco, 'cz': cz_reco,
            'E_edges': E_edges_reco, 'cz_edges': cz_edges_reco,
            'topologies': {
                'cascade': is_cascade,
                'track': is_track,
            },
            'topo_labels': {
                'cascade': 'Cascades (PID=0)',
                'track': 'Tracks (PID=1)',
            },
        },
    }

    for var_name, var_info in var_spaces.items():
        E_var = var_info['E']
        cz_var = var_info['cz']
        E_edges = var_info['E_edges']
        cz_edges = var_info['cz_edges']
        var_label = var_name
        E_axis_label = rf'$E^{{\rm {var_label}}}$ [GeV]'
        cz_axis_label = rf'cos($\theta_z^{{\rm {var_label}}}$)'
        n_E_bins = len(E_edges) - 1
        n_cz_bins = len(cz_edges) - 1
        E_mesh, CZ_mesh = np.meshgrid(E_edges, cz_edges)

        for topo_name, topo_mask in var_info['topologies'].items():
            E_sel = E_var[topo_mask]
            cz_sel = cz_var[topo_mask]
            w_std_sel = expected_std[topo_mask]
            w_cpt_sel = expected_cpt[topo_mask]

            hist_std, _, _ = np.histogram2d(
                E_sel, cz_sel, bins=[E_edges, cz_edges], weights=w_std_sel)
            hist_cpt, _, _ = np.histogram2d(
                E_sel, cz_sel, bins=[E_edges, cz_edges], weights=w_cpt_sel)
            hist_diff = hist_cpt - hist_std

            npz_out = os.path.join(
                args.output_dir,
                f"icup_event_difference_{topo_name}_{var_name}_2d.npz")
            np.savez(npz_out,
                     E_edges=E_edges, cz_edges=cz_edges,
                     hist_std=hist_std, hist_cpt=hist_cpt, hist_diff=hist_diff,
                     dm31=DM31_STD, dm31_bar=dm31_bar,
                     exposure_years=EXPOSURE_YEARS,
                     detector='IC-Upgrade', var=var_name, topo=topo_name)
            print(f"Saved: {npz_out}")

            topo_label = var_info['topo_labels'][topo_name]

            print(f"\n{n_E_bins}x{n_cz_bins} histogram ({var_name}, {topo_name}):")
            print(f"  Events in selection: {np.sum(topo_mask)}")
            print(f"  E bins: [{E_edges[0]:.2f}, {E_edges[-1]:.2f}] GeV")
            print(f"  cos(z) bins: [{cz_edges[0]:.2f}, {cz_edges[-1]:.2f}]")
            print(f"  Binned total (std): {hist_std.sum():.1f}")
            print(f"  Binned total (CPT): {hist_cpt.sum():.1f}")
            print(f"  Max |diff|: {np.abs(hist_diff).max():.2f} events")

            # --- Difference plot (CPT - standard) ---
            fig, ax = plt.subplots(figsize=(10, 7))

            vmax = np.max(np.abs(hist_diff))
            if vmax < 0.01:
                vmax = 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = ax.pcolormesh(E_mesh, CZ_mesh, hist_diff.T, cmap='RdBu_r',
                                norm=norm, shading='flat')
            ax.set_xscale('log')
            ax.set_xlabel(E_axis_label, fontsize=14)
            ax.set_ylabel(cz_axis_label, fontsize=14)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(r'$N_{\rm CPT} - N_{\rm std}$ (events)', fontsize=13)

            bin_info = f'{n_E_bins}$\\times${n_cz_bins} bins'
            ax.set_title(
                f'IceCube Upgrade: Event Difference (CPT $-$ Std) — '
                f'{topo_label} [{var_label}]\n'
                rf'$\Delta m^2_{{31}}={DM31_STD*1e3:.3f}$, '
                rf'$\Delta\bar{{m}}^2_{{31}}={dm31_bar*1e3:.1f}$ '
                rf'[$\times 10^{{-3}}$ eV$^2$], '
                f'{EXPOSURE_YEARS:.0f} yr, {bin_info}',
                fontsize=13
            )
            ax.tick_params(labelsize=12)

            plt.tight_layout()
            out = os.path.join(
                args.output_dir,
                f"icup_event_difference_{topo_name}_{var_name}_2d.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved: {out}")
            plt.close(fig)

            # --- Standard event distribution ---
            fig2, ax2 = plt.subplots(figsize=(10, 7))

            im2 = ax2.pcolormesh(E_mesh, CZ_mesh, hist_std.T, cmap='viridis',
                                   shading='flat')
            ax2.set_xscale('log')
            ax2.set_xlabel(E_axis_label, fontsize=14)
            ax2.set_ylabel(cz_axis_label, fontsize=14)
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Expected events', fontsize=13)
            ax2.set_title(
                f'IceCube Upgrade: Expected Events (Standard) — '
                f'{topo_label} [{var_label}]\n'
                rf'$\Delta m^2_{{31}}={DM31_STD*1e3:.3f}$ '
                rf'$\times 10^{{-3}}$ eV$^2$, '
                f'{EXPOSURE_YEARS:.0f} yr, {bin_info}',
                fontsize=13
            )
            ax2.tick_params(labelsize=12)

            plt.tight_layout()
            out2 = os.path.join(
                args.output_dir,
                f"icup_event_standard_{topo_name}_{var_name}_2d.png")
            fig2.savefig(out2, dpi=150, bbox_inches='tight')
            print(f"Saved: {out2}")
            plt.close(fig2)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
