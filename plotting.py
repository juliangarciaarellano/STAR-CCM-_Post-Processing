"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

plotting.py
===========
Plot generation for slices and comparisons.

Functions
---------
make_slice_plot(cp_grid, mask, ax_slice, plane_idx, val_mm,
                h_label, v_label, extent, scalar_key,
                fig_height=7.0, dpi=None)
    -> matplotlib Figure

make_diff_plot(grid_a, grid_b, mask_a, mask_b, ax_slice, plane_idx, val_mm,
               h_label, v_label, extent, scalar_key,
               label_a='Run A', label_b='Run B',
               fig_height=7.0, dpi=None,
               fixed_range=None)
    -> matplotlib Figure  (delta only — single panel)
    fixed_range : float or None
        If float, colormap is fixed to +/- this value for all planes.
        If None, auto-scales per plane to +/- max(|diff|).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import config as cfg

# Maximum figure width — prevents Tk pixmap overflow on wide planes.
# At 150 DPI, 24in = 3600px which stays within Tk's buffer limits.
MAX_FIG_WIDTH = 24.0


# ── Geometry overlay helpers ──────────────────────────────────────

def _solid_overlay(mask):
    """Single-run geometry: uniform dark gray."""
    r = int(cfg.FILL_COLOR[1:3], 16) / 255
    g = int(cfg.FILL_COLOR[3:5], 16) / 255
    b = int(cfg.FILL_COLOR[5:7], 16) / 255
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = [r, g, b, 1.0]
    return rgba


def _diff_geometry_overlay(mask_a, mask_b):
    """
    Three-colour geometry overlay for comparison plots.
      Gray    (#404040) — solid in BOTH runs (unchanged)
      Magenta (#CC00CC) — solid in Run B only (new geometry added)
      Brown   (#8B4513) — solid in Run A only (geometry removed)
    """
    rgba = np.zeros((*mask_a.shape, 4), dtype=np.float32)
    rgba[mask_a & mask_b]  = [0.25, 0.25, 0.25, 1.0]   # gray
    rgba[~mask_a & mask_b] = [0.80, 0.00, 0.80, 1.0]   # magenta
    rgba[mask_a & ~mask_b] = [0.55, 0.27, 0.07, 1.0]   # brown
    return rgba


def _apply_axes(ax, x_lo, x_hi, y_lo, y_hi, h_label, v_label, title):
    tick_m = cfg.TICK_SPACING_MM / 1000.0
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_m))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_m))
    ax.tick_params(which='major', length=4, width=0.8, colors="#555555",
                   labelsize=7, bottom=True, left=True, top=False, right=False)
    ax.grid(False)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(h_label, fontsize=9, color="#333333")
    ax.set_ylabel(v_label, fontsize=9, color="#333333")
    ax.set_title(title, fontsize=8, color="#222222", pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb"); spine.set_linewidth(0.8)


def _fig_size(fig_height, x_lo, x_hi, y_lo, y_hi):
    """Compute figure width capped at MAX_FIG_WIDTH."""
    aspect = (x_hi - x_lo) / max(y_hi - y_lo, 1e-6)
    fig_w  = fig_height * aspect + 1.8
    fig_w  = max(fig_w, 5.0)
    fig_w  = min(fig_w, MAX_FIG_WIDTH)
    return fig_w, fig_height


# ── Single-run slice plot ─────────────────────────────────────────

def make_slice_plot(cp_grid, mask, ax_slice, plane_idx, val_mm,
                    h_label, v_label, extent, scalar_key,
                    fig_height=7.0, dpi=None):
    scfg = cfg.SCALARS[scalar_key]
    h_min, h_max, v_min, v_max = extent
    pad = 0.04
    x_lo = h_min - pad*(h_max-h_min);  x_hi = h_max + pad*(h_max-h_min)
    y_lo = v_min - pad*(v_max-v_min);  y_hi = v_max + pad*(v_max-v_min)

    fig_w, fig_h = _fig_size(fig_height, x_lo, x_hi, y_lo, y_hi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi or cfg.DPI)
    fig.patch.set_facecolor(cfg.BG_COLOR)
    ax.set_facecolor(cfg.BG_COLOR)

    im = ax.imshow(cp_grid,
                   extent=[h_min, h_max, v_min, v_max],
                   origin='lower', cmap=scfg['cmap'],
                   vmin=scfg['vmin'], vmax=scfg['vmax'],
                   aspect='equal', interpolation='nearest', zorder=1)
    ax.imshow(_solid_overlay(mask),
              extent=[h_min, h_max, v_min, v_max],
              origin='lower', aspect='equal',
              interpolation='nearest', zorder=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(scfg['label'], fontsize=8, color="#333333")
    cbar.ax.tick_params(labelsize=7, colors="#555555")

    valid    = cp_grid[~np.isnan(cp_grid)]
    fluid_px = int((~np.isnan(cp_grid) & ~mask).sum())
    if valid.size:
        rng = f"[{valid.min():.3f}, {valid.max():.3f}]"
    else:
        rng = "[no data]"
    title = (f"Plane {cfg.AXIS_LABEL[ax_slice]} #{plane_idx:03d}  {val_mm:+.1f} mm  "
             f"{scalar_key} {rng}  "
             f"({fluid_px:,} fluid px)")

    _apply_axes(ax, x_lo, x_hi, y_lo, y_hi, h_label, v_label, title)
    return fig


# ── Delta-only comparison plot ────────────────────────────────────

def make_diff_plot(grid_a, grid_b, mask_a, mask_b,
                   ax_slice, plane_idx, val_mm,
                   h_label, v_label, extent, scalar_key,
                   label_a='Run A', label_b='Run B',
                   fig_height=7.0, dpi=None,
                   fixed_range=None):
    """
    Single-panel delta plot: (B - A) with three-colour geometry overlay.

    Parameters
    ----------
    fixed_range : float or None
        If float  -> colormap fixed to [-fixed_range, +fixed_range] for all planes.
        If None   -> auto-scale per plane to +/- max(|diff|).
    """
    h_min, h_max, v_min, v_max = extent
    pad = 0.04
    x_lo = h_min - pad*(h_max-h_min);  x_hi = h_max + pad*(h_max-h_min)
    y_lo = v_min - pad*(v_max-v_min);  y_hi = v_max + pad*(v_max-v_min)

    fig_w, fig_h = _fig_size(fig_height, x_lo, x_hi, y_lo, y_hi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi or cfg.DPI)
    fig.patch.set_facecolor(cfg.BG_COLOR)
    ax.set_facecolor(cfg.BG_COLOR)

    # Delta field — only where BOTH runs have fluid data
    diff = grid_b.astype(np.float32) - grid_a.astype(np.float32)
    either_solid = mask_a | mask_b
    diff[either_solid] = np.nan

    # Determine colormap range
    if fixed_range is not None:
        diff_abs  = float(fixed_range)
        range_str = f"fixed +/-{diff_abs:.4f}"
    else:
        diff_abs  = float(np.nanmax(np.abs(diff))) if not np.all(np.isnan(diff)) else 1.0
        diff_abs  = max(diff_abs, 1e-6)
        range_str = f"auto +/-{diff_abs:.4f}"

    im = ax.imshow(diff,
                   extent=[h_min, h_max, v_min, v_max],
                   origin='lower', cmap='RdBu_r',
                   vmin=-diff_abs, vmax=diff_abs,
                   aspect='equal', interpolation='nearest', zorder=1)

    ax.imshow(_diff_geometry_overlay(mask_a, mask_b),
              extent=[h_min, h_max, v_min, v_max],
              origin='lower', aspect='equal',
              interpolation='nearest', zorder=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"delta {scalar_key}  ({label_b} - {label_a})  [{range_str}]",
                   fontsize=8, color="#333333")
    cbar.ax.tick_params(labelsize=7, colors="#555555")

    legend_elements = [
        Patch(facecolor='#404040', label='Geometry (both)'),
        Patch(facecolor='#CC00CC', label=f'New geometry ({label_b} only)'),
        Patch(facecolor='#8B4513', label=f'Removed geometry ({label_a} only)'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               ncol=3,
               fontsize=7,
               framealpha=0.9,
               facecolor='white',
               edgecolor='#bbbbbb',
               bbox_transform=fig.transFigure)
    fig.subplots_adjust(bottom=0.08)

    valid = diff[~np.isnan(diff)]
    if len(valid):
        title = (f"delta {scalar_key}  {label_b} - {label_a}  |  "
                 f"Plane {cfg.AXIS_LABEL[ax_slice]} #{plane_idx:03d}"
                 f"  {val_mm:+.1f} mm  "
                 f"actual [{valid.min():.4f}, {valid.max():.4f}]")
    else:
        title = (f"delta {scalar_key}  {label_b} - {label_a}  |  "
                 f"Plane {cfg.AXIS_LABEL[ax_slice]} #{plane_idx:03d}"
                 f"  {val_mm:+.1f} mm  (no data)")

    _apply_axes(ax, x_lo, x_hi, y_lo, y_hi, h_label, v_label, title)
    return fig