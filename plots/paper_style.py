"""Canonical matplotlib style for CPT-paper figures.

Ported verbatim from SRC ``claude/3-CPT-violation/paper/scripts/paper_style.py``;
shared style used by the figure plotters in this directory. Contains no paths,
data, or environment dependencies — copied as-is (this note is the only change).

Matches paper/plots/DUNE/plot_dune_combined.py — the 2-panel DUNE figure
(top: δ_CP heatmap with twilight_shifted, bottom: band plot).

Usage:
    from paper_style import apply_paper_style, CYCLIC_CMAP, CYCLIC_VMIN, CYCLIC_VMAX
    apply_paper_style()
"""

import matplotlib

RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 15,
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
}

CYCLIC_CMAP = 'twilight_shifted'
CYCLIC_VMIN = -180
CYCLIC_VMAX = 180


def apply_paper_style(backend=None):
    if backend is not None:
        matplotlib.use(backend)
    matplotlib.rcParams.update(RC_PARAMS)
