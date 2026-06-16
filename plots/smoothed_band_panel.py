#!/usr/bin/env python3
"""Reusable smoothed-band-panel drawer. Call `draw_smoothed_band_panel(ax, ...)`
from any figure to render a paper-quality nested CL panel on an existing axes.

Ported from:
    claude/3-CPT-violation/CP-CPT-degeneracy/DUNE/scripts/smooth_right_panel/smoothed_band_panel.py

Shared smoothed-band renderer used by the DUNE Fig 4 and NOvA Suppl Fig 1
plotters (plot_fig4_dune_cpt_bias.py / plot_figS1_nova_bias.py). The only input
is the combined-scan NPZ path passed by the caller (via draw_smoothed_band_panel
/ load_dchi2_grid); nothing is read from or written to disk by this module
otherwise, so there are no paths to de-hardcode here. The smoothing / contour /
track-building logic is preserved verbatim from the original.

Dependency note:
    Requires scikit-image (``skimage``) for skimage.measure.find_contours, in
    addition to numpy / scipy / matplotlib.

De-hardcode note:
    Removed the unused ``from pathlib import Path`` import from the original
    (Path was never referenced); no absolute paths were present.
"""
import numpy as np
from scipy.ndimage import label
from scipy.signal import argrelmin, savgol_filter
from scipy.ndimage import gaussian_filter1d
from skimage import measure

from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MPLPath
import matplotlib.pyplot as plt


LEVELS = {'90cl': 2.71, '3sig': 9.0, '5sig': 25.0}

# Sampled from twilight_shifted at δ_CP^meas values on the top-panel colorbar:
# 90% CL = 100° (#a7464f), 3σ = 50° (#ca9b7e), 5σ = -25° (#bdcbd2).
FILL = {'5sig': '#bdcbd2', '3sig': '#ca9b7e', '90cl': '#a7464f'}
FILL_ALPHA = {'5sig': 1.0, '3sig': 1.0, '90cl': 1.0}
ZORDER = {'5sig': 0, '3sig': 1, '90cl': 2}


def _tag(d):
    return f"{d:+.4e}".replace('+', 'p').replace('-', 'n').replace('.', 'd')


def load_dchi2_grid(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    deltas = np.asarray(data['deltas'])
    dcp_vals = np.asarray(data['dcp_vals'])
    dcp_pi = np.degrees(dcp_vals) / 180.0
    n_d, n_dcp = len(deltas), len(dcp_vals)
    dchi2 = np.full((n_d, n_dcp), np.nan)
    for j, d in enumerate(deltas):
        key = f"chi2_best_{_tag(d)}"
        if key not in data:
            continue
        row = np.asarray(data[key])
        dchi2[j] = row - row.min()
    return dcp_pi, deltas * 1e3, dchi2


def _strategy_surface(dcp_pi, deltas_1e3, dchi2, sigma, upsample):
    from scipy.ndimage import gaussian_filter, zoom
    zoomed = zoom(dchi2, upsample, order=3, mode='nearest')
    sgrid = gaussian_filter(zoomed, sigma=sigma, mode='nearest')
    xg = np.linspace(dcp_pi[0], dcp_pi[-1], zoomed.shape[1])
    yg = np.linspace(deltas_1e3[0], deltas_1e3[-1], zoomed.shape[0])
    return xg, yg, sgrid


def _mask_for_level(sgrid, level, min_area_frac=0.003):
    mask = sgrid <= level
    labels, n = label(mask)
    total = mask.size
    keep = np.zeros_like(mask)
    for i in range(1, n + 1):
        if (labels == i).sum() / total >= min_area_frac:
            keep[labels == i] = True
    return keep


def _smooth_polyline(xs, ys, passes):
    for _ in range(passes):
        xs = (np.roll(xs, 1) + xs + np.roll(xs, -1)) / 3.0
        ys = (np.roll(ys, 1) + ys + np.roll(ys, -1)) / 3.0
    return xs, ys


def _polygons_from_mask(mask, xg, yg, passes=8, polyline_sigma_pts=0):
    labels, n = label(mask)
    dx = (xg[-1] - xg[0]) / (len(xg) - 1)
    dy = (yg[-1] - yg[0]) / (len(yg) - 1)
    out = []
    for i in range(1, n + 1):
        comp = (labels == i).astype(float)
        padded = np.pad(comp, 1, mode='constant')
        raw = measure.find_contours(padded, 0.5)
        if not raw:
            continue
        polys = []
        for c in raw:
            rows = c[:, 0] - 1
            cols = c[:, 1] - 1
            xs = np.clip(xg[0] + cols * dx, xg[0], xg[-1])
            ys = np.clip(yg[0] + rows * dy, yg[0], yg[-1])
            xs, ys = _smooth_polyline(xs, ys, passes)
            if polyline_sigma_pts > 0 and len(xs) > 10:
                xs = gaussian_filter1d(xs, sigma=polyline_sigma_pts, mode='wrap')
                ys = gaussian_filter1d(ys, sigma=polyline_sigma_pts, mode='wrap')
            polys.append(np.column_stack([xs, ys]))
        def _area(p):
            return 0.5 * np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1])
        outer = max(polys, key=lambda p: abs(_area(p)))
        holes = [p for p in polys if p is not outer]
        out.append((outer, holes))
    return out


def _compound_path(outer, holes):
    verts = [outer]
    codes = [np.r_[MPLPath.MOVETO,
                   np.full(len(outer) - 2, MPLPath.LINETO),
                   MPLPath.CLOSEPOLY]]
    for h in holes:
        verts.append(h)
        codes.append(np.r_[MPLPath.MOVETO,
                           np.full(len(h) - 2, MPLPath.LINETO),
                           MPLPath.CLOSEPOLY])
    return MPLPath(np.vstack(verts), np.concatenate(codes))


def _all_local_minima(row, order=3, within=25.0):
    rel = list(argrelmin(row, order=order)[0])
    imin = int(np.argmin(row))
    if imin not in rel:
        rel.append(imin)
    rmin = row.min()
    return sorted([i for i in set(rel) if row[i] - rmin <= within]), imin


def _build_tracks(dcp_pi, deltas_1e3, dchi2, max_gap_pi=0.18, max_row_gap=1,
                  track_threshold=25.0):
    per_row_pts = []
    for j, y in enumerate(deltas_1e3):
        row = dchi2[j]
        if np.any(np.isnan(row)):
            per_row_pts.append([])
            continue
        idxs, imin = _all_local_minima(row, within=track_threshold)
        pts = [{'delta': y, 'dcp': dcp_pi[i], 'is_global': (i == imin)} for i in idxs]
        per_row_pts.append(pts)
    tracks, last_j = [], []
    for j, pts in enumerate(per_row_pts):
        used = [False] * len(pts)
        for ti, tk in enumerate(tracks):
            if j - last_j[ti] > max_row_gap:
                continue
            last = tk[-1]
            best_k, best_dx = -1, np.inf
            for k, p in enumerate(pts):
                if used[k]:
                    continue
                dx = abs(p['dcp'] - last['dcp'])
                if dx < max_gap_pi and dx < best_dx:
                    best_k, best_dx = k, dx
            if best_k >= 0:
                tk.append(pts[best_k])
                last_j[ti] = j
                used[best_k] = True
        for k, p in enumerate(pts):
            if not used[k]:
                tracks.append([p])
                last_j.append(j)
    return [t for t in tracks if len(t) >= 4]


def _plot_track(ax, track, savgol_win=7, track_sigma_pts=0):
    xs = np.array([p['dcp'] for p in track])
    ys = np.array([p['delta'] for p in track])
    flags = np.array([p['is_global'] for p in track], dtype=bool)
    if len(xs) >= 5:
        w = min(savgol_win, len(xs) if len(xs) % 2 == 1 else len(xs) - 1)
        if w >= 3:
            xs = savgol_filter(xs, w, 2)
    if track_sigma_pts > 0 and len(xs) > 10:
        xs = gaussian_filter1d(xs, sigma=track_sigma_pts, mode='nearest')
    for i in range(len(xs) - 1):
        style = '-' if (flags[i] or flags[i + 1]) else ':'
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color='black', linestyle=style, lw=1.6,
                zorder=7, solid_capstyle='round', dash_capstyle='round')


def draw_smoothed_band_panel(ax, npz_path, truth_dcp_deg, xlim,
                             ylim=(-1.25, 1.75), sigma=0.8, upsample=4,
                             boundary_passes=8, polyline_sigma_pts=0,
                             track_savgol_win=7, track_sigma_pts=0,
                             title=None,
                             levels_to_draw=('5sig', '3sig', '90cl'),
                             track_threshold=None,
                             show_secondary_tracks=True):
    """Draw a nested-CL smoothed band panel on `ax` from `npz_path`.

    `levels_to_draw` lets callers drop levels that aren't relevant
    (e.g. NOvA uses only '3sig' and '90cl'). Returns legend handles.
    """
    dcp_pi, deltas_1e3, dchi2 = load_dchi2_grid(npz_path)
    xg, yg, sgrid = _strategy_surface(dcp_pi, deltas_1e3, dchi2, sigma, upsample)

    for lbl in levels_to_draw:
        level = LEVELS[lbl]
        mask = _mask_for_level(sgrid, level, min_area_frac=0.003)
        components = _polygons_from_mask(mask, xg, yg, passes=boundary_passes,
                                         polyline_sigma_pts=polyline_sigma_pts)
        for outer, holes in components:
            path = _compound_path(outer, holes)
            patch = PathPatch(path, facecolor=FILL[lbl], alpha=FILL_ALPHA[lbl],
                              edgecolor='none', linewidth=0, zorder=ZORDER[lbl])
            ax.add_patch(patch)

    # Default track threshold = the outermost drawn CL level — so secondary
    # tracks only appear within the visible band.
    if track_threshold is None:
        track_threshold = max(LEVELS[lbl] for lbl in levels_to_draw)
    tracks = _build_tracks(dcp_pi, deltas_1e3, dchi2,
                           track_threshold=track_threshold)
    if not show_secondary_tracks:
        # keep only the track containing the most global-best points
        def _n_global(tk):
            return sum(1 for p in tk if p['is_global'])
        tracks = [max(tracks, key=_n_global)] if tracks else []
    for tk in tracks:
        _plot_track(ax, tk, savgol_win=track_savgol_win,
                    track_sigma_pts=track_sigma_pts)

    truth_pi = truth_dcp_deg / 180.0
    ax.axvline(truth_pi, color='#2ca02c', ls='--', lw=2, alpha=0.85, zorder=5)
    ax.axhline(0.0, color='gray', ls='-', lw=0.8, alpha=0.5, zorder=3)
    ax.plot(truth_pi, 0.0, 'ko', ms=6, zorder=10)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=14)

    level_labels = {'5sig': r'5$\sigma$ allowed', '3sig': r'3$\sigma$ allowed',
                    '90cl': '90% CL allowed'}
    handles = [Patch(facecolor=FILL[lbl], alpha=FILL_ALPHA[lbl],
                     label=level_labels[lbl]) for lbl in levels_to_draw]
    handles.append(plt.Line2D([0], [0], color='black', ls='-', lw=1.6,
                              label=r'best-fit $\delta_{CP}$'))
    if show_secondary_tracks:
        handles.append(plt.Line2D([0], [0], color='black', ls=':', lw=1.6,
                                  label='secondary minimum'))
    handles.append(plt.Line2D([0], [0], color='#2ca02c', ls='--', lw=2, label='truth'))
    return handles
