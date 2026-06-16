"""
Path & environment resolution for the CPT neutrino analysis.

Single source of truth for locating the Pynu framework, the MC/data files, the
analysis configs, and external tools (GLoBES, nuSQuIDS). Nothing in this repo
hardcodes an absolute path or assumes a particular cluster: every machine-specific
location comes from an environment variable (see ``env.sh.example``), with
repo-relative defaults where one exists.

Typical use in an analysis script::

    from analysis.lib import paths
    paths.add_pynu_to_path()              # makes `import pynu` work
    xml = paths.pynu_config("ORCA_Atm_CPT_datafit.xml")   # env-vars expanded
    glb = paths.globes_config()           # the 8-rule DUNE CPT config
"""
import os
import sys
import tempfile
from pathlib import Path

# analysis/lib/paths.py  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise EnvironmentError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy env.sh.example to env.sh, edit it for your machine, then "
            f"`source env.sh`."
        )
    return val


def pynu_dir() -> Path:
    """Directory that CONTAINS the ``pynu`` package.

    Defaults to the bundled submodule at ``external/Pynu``; override with
    ``PYNU_DIR`` to point at an existing checkout.
    """
    return Path(_env("PYNU_DIR", str(REPO_ROOT / "external" / "Pynu")))


def add_pynu_to_path() -> str:
    """Put the Pynu framework on ``sys.path`` so ``import pynu`` resolves.

    Also defaults the ``PYNU`` env var (which Pynu's Super-K flux initialization
    reads) to ``<pynu_dir>/pynu`` if it is not already set.
    """
    p = str(pynu_dir())
    if p not in sys.path:
        sys.path.insert(0, p)
    os.environ.setdefault("PYNU", str(pynu_dir() / "pynu"))
    return p


def data_dir() -> str:
    """Root directory holding the MC/data files referenced by the XML configs.

    The configs use ``${DATA_DIR}/...`` placeholders; see docs/INSTALL.md for the
    expected sub-layout (Pynu/data/ORCA, IceCubeUpgrade.../events, SuperK_*.h5, ...).
    """
    return _env("DATA_DIR", required=True)


def config_dir() -> Path:
    """Directory holding the analysis configs (``configs/`` in this repo)."""
    return Path(_env("CONFIG_DIR", str(REPO_ROOT / "configs")))


def globes_prefix():
    """GLoBES install prefix (contains lib/libglobes, bin/globes). Optional."""
    return _env("GLOBES_PREFIX")


def pynu_config(name: str) -> str:
    """Return a ready-to-load Pynu XML config with ``${ENV}`` placeholders expanded.

    The committed XMLs under ``configs/pynu/`` are templated: they contain
    ``${DATA_DIR}/...`` instead of absolute paths. This reads the template,
    expands environment variables against the current process environment, and
    writes a fully-resolved copy to a temp file whose path is returned (Pynu reads
    the resolved file directly).
    """
    src = config_dir() / "pynu" / name
    if not src.exists():
        raise FileNotFoundError(f"Pynu config not found: {src}")
    resolved = os.path.expandvars(src.read_text())
    if "${" in resolved:
        near = resolved.split("${", 1)[1][:40]
        raise EnvironmentError(
            f"Unresolved ${{...}} placeholder in {name} after env expansion "
            f"(near '${{{near}'). Ensure DATA_DIR and any other referenced vars "
            f"are exported (source env.sh)."
        )
    out = Path(tempfile.gettempdir()) / f"resolved_{os.getpid()}_{name}"
    out.write_text(resolved)
    return str(out)


def globes_config(name: str = "dune_globes/DUNE_GLoBES_CPT.glb") -> str:
    """Path to a GLoBES AEDL config under ``configs/globes/``.

    The default is the 8-rule DUNE CPT configuration (4 appearance + 4
    disappearance rules, neutrino/antineutrino split) used for the
    accelerator CP--CPT degeneracy study.
    """
    p = config_dir() / "globes" / name
    if not p.exists():
        raise FileNotFoundError(f"GLoBES config not found: {p}")
    return str(p)
