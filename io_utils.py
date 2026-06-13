"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

io_utils.py
===========
File I/O helpers — loading geometry/CFD meshes and saving/loading NPZ results.

Functions
---------
load_geometry(case_path)       -> list of pyvista blocks
load_cfd(case_path)            -> pyvista MultiBlock
save_plane(out_dir, stem, cp_grids, mask, extent, meta)
load_plane(npz_path)           -> dict
find_planes(npz_dir, axis=None, scalar=None) -> list of paths
"""

import os
import numpy as np
import pyvista as pv

import config as cfg


# ── loaders ──────────────────────────────────────────────────────

def load_geometry(case_path=None):
    """
    Load geometry .case file and return list of non-empty pyvista blocks.

    Parameters
    ----------
    case_path : str or None   defaults to config.GEOMETRY_CASE

    Returns
    -------
    list of pyvista datasets (one per geometry part)
    """
    path = case_path or cfg.GEOMETRY_CASE
    pv.global_theme.allow_empty_mesh = True
    mesh = pv.read(path)
    blocks = [mesh[i] for i in range(mesh.n_blocks)
              if mesh[i] is not None and mesh[i].n_cells > 0]
    return blocks


def load_cfd(case_path=None):
    """
    Load CFD data .case file and return the raw MultiBlock mesh.

    Parameters
    ----------
    case_path : str or None   defaults to config.DATA_CASE

    Returns
    -------
    pyvista MultiBlock
    """
    path = case_path or cfg.DATA_CASE
    pv.global_theme.allow_empty_mesh = True
    return pv.read(path)


# ── NPZ save / load ───────────────────────────────────────────────

def save_plane(npz_dir, stem, scalar_grids, mask, extent,
               ax_slice, plane_val, plane_idx):
    """
    Save one plane's results to a compressed NPZ file.

    NPZ contents
    ------------
    mask           bool   (H,W)       True = solid geometry
    extent         float  [h_min,h_max,v_min,v_max]  metres
    resolution_mm  float              mm per pixel
    plane_axis     int8               0=X 1=Y 2=Z
    plane_value_m  float              plane position in metres
    plane_index    int16              1-based front-to-rear index
    h_axis         bytes              e.g. b'Y'
    v_axis         bytes              e.g. b'Z'
    <scalar_key>   float32  (H,W)     one array per scalar (e.g. 'Cp', 'CpT')

    Parameters
    ----------
    npz_dir      : str   output directory
    stem         : str   filename stem (no extension)
    scalar_grids : dict  {scalar_key: (H,W) float32 array}
    mask         : (H,W) bool
    extent       : (4,)  float [h_min, h_max, v_min, v_max]
    ax_slice     : int
    plane_val    : float  metres
    plane_idx    : int

    Returns
    -------
    path : str   full path to saved .npz
    """
    ha, va = cfg.PLOT_AXES[ax_slice]
    save_dict = {
        'mask':          mask,
        'extent':        np.array(extent, dtype=np.float32),
        'resolution_mm': np.float32(cfg.RESOLUTION_MM),
        'plane_axis':    np.int8(ax_slice),
        'plane_value_m': np.float32(plane_val),
        'plane_index':   np.int16(plane_idx),
        'h_axis':        np.bytes_(cfg.AXIS_LABEL[ha]),
        'v_axis':        np.bytes_(cfg.AXIS_LABEL[va]),
    }
    for key, grid in scalar_grids.items():
        save_dict[key] = grid.astype(np.float32)

    os.makedirs(npz_dir, exist_ok=True)
    path = os.path.join(npz_dir, stem + '.npz')
    np.savez_compressed(path, **save_dict)
    return path


def load_plane(npz_path):
    """
    Load a saved plane NPZ file.

    Returns
    -------
    dict with keys:
      mask, extent, resolution_mm, plane_axis, plane_value_m,
      plane_index, h_axis, v_axis,
      + one key per scalar (e.g. 'Cp', 'CpT')
    """
    raw = np.load(npz_path, allow_pickle=False)
    result = dict(raw)
    # Decode bytes fields
    for key in ['h_axis', 'v_axis']:
        if key in result:
            result[key] = result[key].item().decode('utf-8')
    return result


def find_planes(npz_root, axis=None, scalar=None):
    """
    Find all NPZ plane files under npz_root.

    Parameters
    ----------
    npz_root : str   root directory (contains X/, Y/, Z/ subdirs)
    axis     : str or None   filter to 'X', 'Y', or 'Z'
    scalar   : str or None   (unused — all scalars are in the same NPZ)

    Returns
    -------
    list of (axis_str, plane_idx, plane_val_mm, full_path)
    sorted by axis then plane_idx
    """
    results = []
    axes = [axis] if axis else ['X', 'Y', 'Z']
    for ax in axes:
        ax_dir = os.path.join(npz_root, ax)
        if not os.path.isdir(ax_dir):
            continue
        for fname in sorted(os.listdir(ax_dir)):
            if not fname.endswith('.npz'):
                continue
            # parse:  slice_X_014_+510.0mm.npz
            parts = fname.replace('.npz', '').split('_')
            try:
                pidx  = int(parts[2])
                pmm   = float(parts[3].replace('mm', ''))
            except (IndexError, ValueError):
                pidx, pmm = 0, 0.0
            results.append((ax, pidx, pmm,
                             os.path.join(ax_dir, fname)))
    results.sort(key=lambda x: (x[0], x[1]))
    return results
