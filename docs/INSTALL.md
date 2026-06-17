# Installation

Two independent toolchains: **Pynu** (Python; the atmospheric fits) and
**GLoBES** (C; the DUNE accelerator cross-check). You only need GLoBES if you run
the `analysis/globes/` scripts.

---

## 1. Clone with the Pynu submodule

```bash
git clone --recurse-submodules <repo-url> cpt-neutrino-analysis
cd cpt-neutrino-analysis
# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

`external/Pynu` is pinned to the `pheno-CPT` branch of
[`pabloferm/Pynu`](https://github.com/pabloferm/Pynu) (the branch carrying the
CPT dual-propagation oscillator).

## 2. Pynu Python dependencies

Python 3.10–3.12. The one non-pure-Python dependency, **nuSQuIDS**, installs from
a PyPI wheel (pybind11 build, no source/GSL/Boost compile). `numpy` must be `<2`
because `nuflux` requires it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# core scientific stack (versions known to work together)
# NOTE: scikit-image must stay <0.25 — newer releases require numpy>=2, which
# conflicts with the numpy<2 that nuflux/nuSQuIDS need.
pip install "numpy<2" scipy h5py pandas pyarrow matplotlib iminuit KDEpy "scikit-image<0.25"

# oscillation + flux engines (native wheels)
pip install nusquids nuflux        # provides `import nuSQuIDS`, `import nuflux`
```

Pinned versions from a validated environment: numpy 1.26.4, scipy 1.17.1,
nuflux 2.0.7, nusquids 1.13.3, iminuit 2.32.0, KDEpy 1.1.12.

### nuSQuIDS data tables

The Earth model and cross-section tables are **not** in the wheel; extract them
from the nuSQuIDS sdist and point `NUSQUIDS_DATA_PATH` at them:

```bash
V=1.13.3
curl -fsSL "$(curl -s https://pypi.org/pypi/nusquids/$V/json \
  | python -c 'import sys,json;print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')" \
  -o nusquids.tar.gz
mkdir -p .venv/share
tar xzf nusquids.tar.gz nusquids-$V/data && mv nusquids-$V/data .venv/share/nuSQuIDS
# then in env.sh:  export NUSQUIDS_DATA_PATH="$PWD/.venv/share/nuSQuIDS"
```

## 3. GLoBES (only for the DUNE accelerator study)

GLoBES ≥ 3.x built from source; needs `gcc`, `make`, and **GSL** (`gsl-config` on
PATH).

```bash
GLB=3.2.18
curl -fsSLO "https://www.mpi-hd.mpg.de/personalhomes/globes/download/globes-$GLB.tar.gz"
tar xzf globes-$GLB.tar.gz && cd globes-$GLB
./configure --prefix="$HOME/software/globes"
make && make install
# then in env.sh:
#   export GLOBES_PREFIX="$HOME/software/globes"
#   export LD_LIBRARY_PATH="$GLOBES_PREFIX/lib:$LD_LIBRARY_PATH"
```

The Python GLoBES drivers in `analysis/globes/` call into `libglobes` via the
config in `configs/globes/dune_globes/` — that directory is self-contained
(flux, cross-section, efficiency, and smearing tables are all included).

## 4. Configure your machine

```bash
cp env.sh.example env.sh
$EDITOR env.sh        # set DATA_DIR, NUSQUIDS_DATA_PATH, GLOBES_PREFIX
source env.sh
```

## 5. Data

Fetch the MC/data files into `$DATA_DIR` per the table in the top-level
[`README.md`](../README.md#data). The XML configs reference them as
`${DATA_DIR}/...` and `analysis/lib/paths.py` expands the placeholders at run
time, so no path editing of the configs is needed.

## Verify

```bash
source env.sh
python -c "from analysis.lib import paths; paths.add_pynu_to_path(); import pynu; print('pynu OK')"
python -c "from analysis.lib import paths; print(paths.globes_config())"   # prints the 8-rule CPT config path
```
