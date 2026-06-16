# Testing CPT invariance with atmospheric and accelerator neutrinos

Reproducibility companion to the CPT-violation paper. This repository holds the
configuration files and analysis/plotting scripts needed to reproduce the
oscillation fits and figures in the paper. It is **scripts-only**: no MC, no data
files, and no precomputed result grids are committed — those are acquired or
regenerated as described below.

The physics test: allow neutrinos and antineutrinos to have *separate*
atmospheric mass splittings (Δm²₃₁ and Δm̄²₃₁) and constrain the CPT-violating
difference **Δ ≡ Δm̄²₃₁ − Δm²₃₁**, using

- **atmospheric** neutrino samples (ORCA, IceCube DeepCore, IceCube Upgrade,
  Super-Kamiokande), fit with the [**Pynu**](https://github.com/pabloferm/Pynu)
  framework (`pheno-CPT` branch), and
- an **accelerator** long-baseline cross-check (DUNE) via **GLoBES**, used to
  study the CP–CPT degeneracy.

## Layout

```
configs/
  pynu/                 # Pynu XML fit configs (CPT), path-templated with ${DATA_DIR}
  globes/dune_globes/   # DUNE GLoBES setup — the 8-rule CPT config + flux/xsec/eff/smr tables
analysis/
  lib/paths.py          # env-var path/config resolver (no hardcoded paths anywhere)
  atmospheric/          # Pynu fit runners (ORCA / IC-DC / IC-Up / SK / combined)
  globes/               # GLoBES DUNE CPT-bias and degeneracy runners
plots/                  # scripts that generate the paper figures
external/Pynu/          # git submodule -> pabloferm/Pynu @ pheno-CPT
docs/INSTALL.md         # detailed install for Pynu deps + GLoBES
env.sh.example          # environment template (copy to env.sh, edit, source)
```

## Quick start

```bash
# 1. clone with the Pynu submodule
git clone --recurse-submodules <repo-url> cpt-neutrino-analysis
cd cpt-neutrino-analysis

# 2. install dependencies (see docs/INSTALL.md for details)
#    - Python: numpy<2, scipy, h5py, pandas/pyarrow, matplotlib, scikit-image, nuSQuIDS, nuflux
#    - GLoBES (only for the DUNE accelerator study)

# 3. configure your machine
cp env.sh.example env.sh
$EDITOR env.sh          # set DATA_DIR, NUSQUIDS_DATA_PATH, (GLOBES_PREFIX)
source env.sh

# 4. fetch the data files into $DATA_DIR (see "Data" below)

# 5. run a fit / make a figure (see "Running" below)
```

## Installation

See [`docs/INSTALL.md`](docs/INSTALL.md). In brief:

- **Pynu** is vendored as a submodule pinned to the `pheno-CPT` branch. Its
  scientific dependencies (nuSQuIDS + nuflux for oscillation/flux, `numpy<2`,
  scipy, h5py, pyarrow) are installed separately — the doc lists the PyPI-wheel
  route that works on a fresh machine.
- **GLoBES** (≥ 3.0) is built from source and located via `$GLOBES_PREFIX`. It is
  required only for the DUNE accelerator scripts under `analysis/globes/`.

## Environment variables

All machine-specific locations come from the environment (`env.sh`). The configs
contain `${DATA_DIR}`-style placeholders that `analysis/lib/paths.py` expands at
run time — there are no absolute or cluster-specific paths in the repo.

| Variable | Meaning | Default |
|---|---|---|
| `PYNU_DIR` | dir containing the `pynu` package | `external/Pynu` (submodule) |
| `PYNU` | Pynu SK-flux init path | `$PYNU_DIR/pynu` |
| `DATA_DIR` | root of the MC/data files | _(required)_ |
| `NUSQUIDS_DATA_PATH` | nuSQuIDS data tables | _(required for fits)_ |
| `GLOBES_PREFIX` | GLoBES install prefix | _(required for DUNE)_ |
| `CONFIG_DIR` | this repo's `configs/` | `configs/` |

## Data

The XML configs reference these files as `${DATA_DIR}/<path>`. Place them under
`$DATA_DIR` preserving the sub-paths shown. None are redistributed here.

| Path under `$DATA_DIR` | Sample | Availability |
|---|---|---|
| `Pynu/data/ORCA/ORCA_MC_dataverse_with_muons.parquet` | ORCA MC | KM3NeT/ORCA public data release |
| `Pynu/data/ORCA/ORCA_data_dataverse.parquet` | ORCA data | KM3NeT/ORCA public data release |
| `Pynu/data/IceCube/IC_MC.parquet` | IC DeepCore MC | IceCube DeepCore public data release |
| `Pynu/data/IceCube/IC_data.parquet` | IC DeepCore data | IceCube DeepCore public data release |
| `IceCubeUpgradeNeutrinoMCDataRelease-2/events/neutrino_mc.csv` | IC Upgrade MC | IceCube Upgrade public MC release (v01_00) |
| `SuperK_fullMC_6_4_x50_inv_flux_tune1.h5` | Super-K MC | Super-K collaboration (not public) |
| `Pynu/data/ORCAFull/ORCA_full_MC.parquet` | ORCA-Full MC | derived; collaboration-internal |
| `Pynu/data/ORCAFullEvtMC/ORCA_full_evtmc.csv` | ORCA-Full event MC | derived; collaboration-internal |

## Reproducing the analysis

Each paper figure is produced in two steps: an **analysis runner** (`analysis/`)
that regenerates the result file from MC/data + a framework (Pynu / GLoBES /
nuSQuIDS), then a **plot script** (`plots/`) that draws the figure from that
result. All scripts resolve frameworks, configs, and data through
`analysis/lib/paths.py` (env vars) — no paths are hardcoded. Run them from the
repo root with `env.sh` sourced; output figures default to `outputs/` (gitignored).

Runners take CLI args (grid indices, exposures, output dirs) and degrade to the
`SLURM_ARRAY_TASK_ID` env var when present, so the same script runs locally or as
a cluster array job. Plotters take their input result file via `--input` /
`--results-dir` (the heavy result grids are not shipped — regenerate them with the
runner, or point at your own).

### Figure → script → number map

| Fig | Content | Plot script (`plots/`) | Analysis runner(s) (`analysis/`) | Headline number |
|---|---|---|---|---|
| 1 | DUNE ΔP 3-panel decomposition (CP/CPT/matter) | `plot_fig1_dune_dp_decomposition.py` | `probability/run_dune_degeneracy_panels.py` (nuSQuIDS) | truth δCP=−112°, Δ=1.0×10⁻³; imposter δCP=−84°; residual ≲1.4% |
| 2 | CP/CPT/effective-CP vector schematic | — (TikZ in the manuscript, not code) | — | — |
| 3 | T2K + NOvA degeneracy manifolds | `plot_fig3_nova_t2k_manifolds.py` | inline (Cervera analytic) | reconciling δCP≈109°, δΔm²₃₁≈1.48×10⁻³ |
| 4 | DUNE CPT bias on δCP (heatmap + bands) | `plot_fig4_dune_cpt_bias.py` | `globes/run_dune_dcp_scan.py` → `globes/assemble_dune_method_d.py` | 8-rule GLoBES; basin switch δΔm²≈+0.76×10⁻³ |
| 5 | Atmospheric 2D contours (data + sensitivity) | `plot_fig5_atm_contours.py` | `atmospheric/run_combined_cpt_datafit.py`, `run_combined_future_sensitivity.py` (+ assemblers) | data ORCA-6+IC-DC (433 kt-yr / 7.74 yr); proj ORCA-Full 5 yr + IC-Up-7 10 yr |
| 6 | Oscillograds ∂P/∂Δm², ∂P/∂δCP | _pending (CHIC code, from P. Fernández-Menéndez)_ | — | NO/IO |
| 7 | Time-evolution sensitivity vs year | `plot_fig7_time_evolution.py` | `atmospheric/run_cpt_time_evolution.py` | at δΔm²=0.05×10⁻³; IC-Up 2026, ORCA-Full 2031 |
| S1 | NOvA CPT bias heatmap | `plot_figS1_nova_bias.py` | `globes/run_nova_dcp_scan.py` | NOvA 810 km, 25 kt, 3+3 yr; 4-rule |
| S2 | DUNE basin-switch event spectra (6-panel) | `globes/diagnose_dune_basin_switch.py --phase 2 --paper` | (embedded GLoBES) | min1 δCP=+169°, min2 δCP=−34°, s²θ₂₃=0.630 |
| S3 | IC-DC + IC-Up-7 event-count difference | `plot_figS3_event_difference.py` | `atmospheric/run_ic_event_difference.py`, `run_icup_event_difference.py` | Δm²₃₁=2.511×10⁻³, Δm̄²₃₁=2.0×10⁻³ |
| S4 | 1D Δχ² profile of δΔm²₃₁ | `plot_figS4_1d_profiles.py` | (reads Fig 5 grids) | data 90%: [−0.44,+0.57]×10⁻³; proj: [−0.14,+0.14]×10⁻³ |

**Headline result:** current world-leading constraint δΔm²₃₁ ∈ [−0.44, +0.57]×10⁻³ eV²
at 90% CL (ORCA-6 + IceCube DeepCore), reaching ≤ 10⁻⁴ eV² within a decade
(ORCA-Full + IceCube Upgrade).

## Status

Phases 1–2 in place: configs, Pynu submodule, env-var infrastructure, and the
ported analysis + plotting scripts for 9 of the paper's figures. Outstanding:
Fig 6 (CHIC oscillograds, awaited from a co-author) and the Fig 2 schematic
(TikZ, lives in the manuscript, not the code).

## Citation

_(paper reference to be added)_
