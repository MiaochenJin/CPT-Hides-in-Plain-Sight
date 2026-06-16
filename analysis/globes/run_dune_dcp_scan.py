#!/usr/bin/env python3
"""run_dune_dcp_scan.py — DUNE CPT truth bias scan (method-d).

Ported from:
    claude/3-CPT-violation/CP-CPT-degeneracy/DUNE/scripts/run_dcp_scan_method_d.py

Produces the per-delta task outputs (task_NNN.json / .npz) that
``assemble_dune_method_d.py`` collects into the DUNE CPT-bias band plot
(paper Fig 4). Thin shim over analysis.globes.experiment: instantiates DUNE_CFG
and forwards to the generic band-plot worker. See experiment.py for the method
description.

Each invocation handles ONE delta_true index (SLURM_ARRAY_TASK_ID, the positional
task_id arg, or --delta to bypass the grid).

Env / inputs:
    GLOBES_PREFIX — GLoBES install prefix (read by the wrapper).
    DUNE_GLoBES_CPT.glb config tree — resolved from the repo via
    paths.globes_config("dune_globes/DUNE_GLoBES_CPT.glb").

Usage:
    python3 run_dune_dcp_scan.py <output_dir> [task_id] [--truth-dcp-deg 0] [--delta <eV^2>]

De-hardcode notes:
  - The original required ``cd <dune_globes_config_dir>`` before running and
    bootstrapped sys.path to a sibling cluster dir for ``_common.experiment``.
    Here we (a) put the repo root on sys.path so ``analysis`` imports resolve,
    and (b) os.chdir into the committed .glb's directory (resolved via
    paths.globes_config) so GLoBES still finds the .glb's relative includes.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.lib import paths  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

# Resolve the committed DUNE CPT .glb and chdir to its directory so GLoBES can
# resolve the .glb's relative include directives (flux/xsec/eff/smr).
_GLB = paths.globes_config("dune_globes/DUNE_GLoBES_CPT.glb")
os.chdir(os.path.dirname(_GLB))

from analysis.globes.experiment import ExperimentConfig, run_band_plot_main  # noqa: E402

DUNE_CFG = ExperimentConfig(
    name="DUNE",
    glb_file="DUNE_GLoBES_CPT.glb",
    expected_n_rules=8,
    nu_rules=[0, 2, 4, 6],
    nubar_rules=[1, 3, 5, 7],
    physical_pairs=[(0, 1), (2, 3), (4, 5), (6, 7)],
    sig_norm_error=[0.02, 0.02, 0.05, 0.05],
    bg_norm_error=[0.05, 0.05, 0.10, 0.10],
)

N_DCP = 201
DELTA_VALUES = np.linspace(-2e-3, 2e-3, 101)
DELTA_VALUES = DELTA_VALUES[DELTA_VALUES != 0.0]  # exclude exact zero → 100 points


if __name__ == "__main__":
    run_band_plot_main(DUNE_CFG, DELTA_VALUES, n_dcp=N_DCP)
