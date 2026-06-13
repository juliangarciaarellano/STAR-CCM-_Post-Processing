"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

interpolation.py
================
Per-fluid-region griddata linear interpolation.

interpolate_per_region(plane_pts_2d, plane_vals, mask, h_min, v_min, px)

Splits the fluid domain by connected region (separated by the solid mask)
and interpolates each region independently. This prevents data from one
side of a surface bleeding through to the other side.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label as ndlabel


def interpolate_per_region(plane_pts_2d, plane_vals, mask, h_min, v_min, px):
    """
    Parameters
    ----------
    plane_pts_2d : (N,2) float  CFD point (h,v) coordinates in metres
    plane_vals   : (N,)  float  scalar values at each CFD point
    mask         : (H,W) bool   True = solid geometry
    h_min        : float        horizontal world origin in metres
    v_min        : float        vertical world origin in metres
    px           : float        pixels per metre

    Returns
    -------
    result : float32 ndarray (H,W)   NaN where solid or outside data hull
    """
    H, W   = mask.shape
    result = np.full((H, W), np.nan, dtype=np.float32)

    fluid_labels, n_regions = ndlabel(~mask)

    for reg in range(1, n_regions + 1):
        rm = fluid_labels == reg
        if rm.sum() < 4:
            continue

        rows, cols = np.where(rm)
        gh = h_min + cols / px    # world h-coords of grid pixels
        gv = v_min + rows / px    # world v-coords of grid pixels

        # Bounding box + halo to select relevant CFD points
        halo = 0.05
        h_lo, h_hi = gh.min() - halo, gh.max() + halo
        v_lo, v_hi = gv.min() - halo, gv.max() + halo

        in_bbox = ((plane_pts_2d[:, 0] >= h_lo) & (plane_pts_2d[:, 0] <= h_hi) &
                   (plane_pts_2d[:, 1] >= v_lo) & (plane_pts_2d[:, 1] <= v_hi))
        lp = plane_pts_2d[in_bbox]
        lv = plane_vals[in_bbox]

        if len(lp) < 4:
            continue

        interp = griddata(lp, lv,
                          np.column_stack([gh, gv]),
                          method='linear')
        result[rows, cols] = interp

    return result
