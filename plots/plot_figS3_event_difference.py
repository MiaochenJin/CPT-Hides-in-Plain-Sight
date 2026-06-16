#!/usr/bin/env python3
"""
plot_figS3_event_difference.py — Suppl Fig 3: 4-panel track event-difference figure.

Paper figure: Supplementary Fig 3. A single 2x2 figure of the CPT-minus-standard
TRACK-sample event-difference distributions (the track sample has the larger
statistics):

    row 0: IC DeepCore   |  col 0: true variables   col 1: reconstructed
    row 1: IC-Upgrade-7   |

Each panel keeps its own colorbar (the per-detector event counts differ by more
than an order of magnitude). Loads the pre-computed track-sample histograms from
the ``.npz`` files written by the two event-difference runners:
  - ``analysis/atmospheric/run_ic_event_difference.py``   -> ic_event_difference_track_{true,reco}_2d.npz
  - ``analysis/atmospheric/run_icup_event_difference.py`` -> icup_event_difference_track_{true,reco}_2d.npz

Inputs (runner outputs, NOT shipped with this repo):
  --icdc-dir    directory holding the IC DeepCore npz files
  --icup-dir    directory holding the IC-Upgrade-7 npz files
  --output-dir  default: REPO_ROOT/outputs/figures (gitignored)

De-hardcoded vs. the original SRC
(``claude/3-CPT-violation/paper/scripts/plot_track_event_difference_4panel.py``):
- ``ROOT = "/Users/miaochenjin/Desktop/Harvard/AtmNuDataFit"`` and the three
  SRC-relative dirs (``ICDC``, ``ICUP``, ``OUT``) are replaced by ``--icdc-dir`` /
  ``--icup-dir`` inputs and an ``--output-dir`` (default under ``REPO_ROOT/outputs/figures``).
- Added the ``analysis.lib.paths`` bootstrap for ``REPO_ROOT``.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

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
    'axes.labelsize': 19,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

COL_HEADER = ["True", "Reconstructed"]
VAR_TEX = {"true": r"\rm true", "reco": r"\rm reco"}


def build_rows(icdc_dir, icup_dir):
    """(row_label, [ (npz path, var) for true, reco ]) for the two detectors."""
    return [
        ("IC DeepCore", [
            (os.path.join(icdc_dir, "ic_event_difference_track_true_2d.npz"), "true"),
            (os.path.join(icdc_dir, "ic_event_difference_track_reco_2d.npz"), "reco"),
        ]),
        ("IC-Upgrade-7", [
            (os.path.join(icup_dir, "icup_event_difference_track_true_2d.npz"), "true"),
            (os.path.join(icup_dir, "icup_event_difference_track_reco_2d.npz"), "reco"),
        ]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icdc-dir", required=True,
                    help="Directory with IC DeepCore ic_event_difference_track_*_2d.npz files")
    ap.add_argument("--icup-dir", required=True,
                    help="Directory with IC-Upgrade-7 icup_event_difference_track_*_2d.npz files")
    ap.add_argument("--output-dir",
                    default=str(paths.REPO_ROOT / "outputs" / "figures"))
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    rows = build_rows(args.icdc_dir, args.icup_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08,
                        wspace=0.62, hspace=0.26)

    for i, (row_label, panels) in enumerate(rows):
        for j, (path, var) in enumerate(panels):
            ax = axes[i, j]
            d = np.load(path, allow_pickle=True)
            E_edges = d["E_edges"]
            cz_edges = d["cz_edges"]
            hist = d["hist_diff"]
            E_mesh, CZ_mesh = np.meshgrid(E_edges, cz_edges)

            vmax = float(np.max(np.abs(hist)))
            if vmax < 1e-6:
                vmax = 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.pcolormesh(E_mesh, CZ_mesh, hist.T, cmap="RdBu_r",
                               norm=norm, shading="flat", rasterized=True)

            ax.set_xscale("log")
            ax.set_xlabel(rf"$E^{{{VAR_TEX[var]}}}$ [GeV]")
            ax.set_ylabel(rf"$\cos\theta_z^{{{VAR_TEX[var]}}}$")

            cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label(r"$N_{\rm CPT} - N_{\rm std}$ [events]", fontsize=16)
            cbar.ax.tick_params(labelsize=13)

            # column header on top row
            if i == 0:
                ax.set_title(COL_HEADER[j], fontsize=20, pad=10)

        # row label (experiment) on the far left, rotated
        axes[i, 0].annotate(
            row_label, xy=(-0.30, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=21, fontweight="bold")

    for ext in ("png", "pdf"):
        outpath = os.path.join(out, f"track_event_difference_4panel.{ext}")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
