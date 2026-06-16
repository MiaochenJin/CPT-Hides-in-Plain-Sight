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
#    - Python: numpy<2, scipy, h5py, pandas/pyarrow, matplotlib, nuSQuIDS, nuflux
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

## Running the analysis

> The fit-runner and plotting scripts are ported from the production pipeline and
> finalized to the exact set of figures in the published paper. See **Status**.

**Atmospheric (Pynu) fit** — runners under `analysis/atmospheric/` resolve Pynu and
the config through `analysis/lib/paths.py`, e.g.:

```bash
source env.sh
python -m analysis.atmospheric.<runner> --config IC_Atm_CPT_datafit.xml [grid args...]
```

**Accelerator (GLoBES) DUNE study** — runners under `analysis/globes/` load the
8-rule CPT config (`configs/globes/dune_globes/DUNE_GLoBES_CPT.glb`).

## Producing the paper figures

`plots/` regenerates each paper figure from a result file (which you produce with
the runners above). A figure → script → quoted-number table is provided once the
figure set is finalized against the paper. See **Status**.

## Status

This repository is being assembled. Phase 1 (configs, submodule, env-var
infrastructure, install docs) is in place. Phase 2 — porting the exact fit-runner
and plotting scripts for the figures that appear in the final paper, with the
figure→script→number map — is in progress.

## Citation

_(paper reference to be added)_
