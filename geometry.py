"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

geometry.py
===========
Geometry mask builder.

build_mask(geom_blocks, ax_slice, plane_val, h_min, h_max, v_min, v_max,
           px, plane_pts_2d=None)

For each enclosed region found after rasterising and dilating the geometry
slice segments, checks whether CFD data points exist inside it:
  - No CFD points  → solid geometry  (fill it)
  - CFD points present → fluid gap   (leave open)

This correctly handles multi-element wings, hollow aerofoil sections,
flat surfaces (undertray), and any geometry where simple flood-fill
would incorrectly fill fluid gaps between elements.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label as ndlabel
from skimage.draw import line as sk_line

import config as cfg


def build_mask(geom_blocks, ax_slice, plane_val,
               h_min, h_max, v_min, v_max, px,
               plane_pts_2d=None):
    """
    Parameters
    ----------
    geom_blocks   : list of pyvista blocks (geometry parts)
    ax_slice      : int  0=X 1=Y 2=Z
    plane_val     : float  plane position in metres
    h_min/h_max   : float  horizontal world bounds in metres
    v_min/v_max   : float  vertical world bounds in metres
    px            : float  pixels per metre  (1000 / RESOLUTION_MM)
    plane_pts_2d  : (N,2) float array of CFD point (h,v) coords on this plane,
                    or None — used to veto false-solid enclosed fluid regions

    Returns
    -------
    solid : bool ndarray (H, W)   True = solid geometry
    """
    ha, va = cfg.PLOT_AXES[ax_slice]
    W = int((h_max - h_min) * px) + 1
    H = int((v_max - v_min) * px) + 1
    line_mask = np.zeros((H, W), dtype=bool)

    origin = [0.0, 0.0, 0.0]
    origin[ax_slice] = plane_val

    # ── rasterise all geometry slice segments ─────────────────────
    for b in geom_blocks:
        if plane_val < b.bounds[ax_slice*2] or plane_val > b.bounds[ax_slice*2+1]:
            continue
        sl = b.slice(normal=cfg.AXIS_NORMAL[ax_slice], origin=origin)
        if sl is None or sl.n_cells == 0:
            continue

        pts2d = sl.points[:, [ha, va]]
        la    = sl.lines.reshape(-1, 3)
        a     = pts2d[la[:, 1]]
        bpt   = pts2d[la[:, 2]]

        col_a = np.clip(((a[:,0]   - h_min) * px).astype(int), 0, W-1)
        row_a = np.clip(((a[:,1]   - v_min) * px).astype(int), 0, H-1)
        col_b = np.clip(((bpt[:,0] - h_min) * px).astype(int), 0, W-1)
        row_b = np.clip(((bpt[:,1] - v_min) * px).astype(int), 0, H-1)

        for j in range(len(col_a)):
            rr, cc = sk_line(row_a[j], col_a[j], row_b[j], col_b[j])
            line_mask[np.clip(rr, 0, H-1), np.clip(cc, 0, W-1)] = True

    # ── rasterise CFD points for region veto ──────────────────────
    cfd_grid = np.zeros((H, W), dtype=bool)
    if plane_pts_2d is not None and len(plane_pts_2d) > 0:
        col_c = np.clip(((plane_pts_2d[:,0] - h_min)*px).astype(int), 0, W-1)
        row_c = np.clip(((plane_pts_2d[:,1] - v_min)*px).astype(int), 0, H-1)
        cfd_grid[row_c, col_c] = True

    # ── dilate to close hairline gaps in surface mesh ─────────────
    dilated = binary_dilation(line_mask, iterations=cfg.DILATION_PX)

    # ── label all open (non-line) regions ─────────────────────────
    comp_labels, n_comp = ndlabel(~dilated)

    # ── find exterior (border-connected) labels ───────────────────
    border_labels = set()
    for edge in [comp_labels[0,:],  comp_labels[-1,:],
                 comp_labels[:,0],   comp_labels[:,-1]]:
        border_labels.update(edge.ravel())
    border_labels.discard(0)

    # ── fill enclosed regions that contain no CFD data ────────────
    solid = dilated.copy()   # start with line pixels as solid

    for c in range(1, n_comp + 1):
        if c in border_labels:
            continue          # exterior fluid — always open

        region = comp_labels == c
        if region.sum() < 10:
            continue          # noise

        if (cfd_grid & region).sum() == 0:
            solid |= region   # no CFD data here → solid geometry

    return solid
