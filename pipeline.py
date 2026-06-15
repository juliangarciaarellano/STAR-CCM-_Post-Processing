"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
================================================================================
"""

import os
import sys
import time
import numpy as np
import pyvista as pv

import config as cfg
import io_utils
import geometry as geom_mod
import interpolation as interp_mod
import plotting


# ── timer / logger ───────────────────────────────────────────────
_T0 = time.perf_counter()

def log(msg, t_ref=None):
    elapsed = time.perf_counter() - _T0
    suffix  = f"  (+{(time.perf_counter()-t_ref)*1000:.0f}ms)" if t_ref else ""
    print(f"  [{elapsed:7.2f}s] {msg}{suffix}")
    sys.stdout.flush()


# ── helpers ───────────────────────────────────────────────────────
def _extract_plane_jobs(cfd_mesh, scalars_to_run, axes_to_run):
    """
    Return list of plane job dicts from the CFD mesh.
    Each job carries the pts/vals arrays for every requested scalar.
    """
    jobs = []
    for block_idx in range(cfd_mesh.n_blocks):
        b = cfd_mesh[block_idx]
        if b is None or b.n_cells == 0:
            continue

        cc   = b.cell_centers()
        pts  = cc.points

        # ── Auto-detect the constant (slicing) axis ───────────────────────────
        # The slicing axis is the one with fewest unique coordinate values.
        # We also check the block name as a hint, but the data takes precedence.
        n_unique = [len(np.unique(np.round(pts[:, a], 5))) for a in range(3)]
        detected_ax = int(np.argmin(n_unique))

        # Block name hint — check if name contains X/Y/Z
        block_name = cfd_mesh.get_block_name(block_idx).strip().upper()
        hint_ax = None
        for a, label in [(0,'X'),(1,'Y'),(2,'Z')]:
            if label in block_name:
                hint_ax = a
                break

        # Use detected axis; warn if it disagrees with the name hint
        ax = detected_ax
        if hint_ax is not None and hint_ax != detected_ax:
            log(f"  [WARN] Block {block_idx} ('{cfd_mesh.get_block_name(block_idx).strip()}'): "
                f"name suggests axis {cfg.AXIS_LABEL[hint_ax]} but data is constant in "
                f"{cfg.AXIS_LABEL[detected_ax]} "
                f"(unique counts X={n_unique[0]:,} Y={n_unique[1]:,} Z={n_unique[2]:,}) "
                f"— using detected axis {cfg.AXIS_LABEL[detected_ax]}")
        elif hint_ax is None:
            log(f"  [INFO] Block {block_idx} ('{cfd_mesh.get_block_name(block_idx).strip()}'): "
                f"no axis hint in name, detected axis {cfg.AXIS_LABEL[detected_ax]} "
                f"(unique counts X={n_unique[0]:,} Y={n_unique[1]:,} Z={n_unique[2]:,})")

        ha, va = cfg.PLOT_AXES[ax]

        if cfg.AXIS_LABEL[ax] not in axes_to_run:
            continue

        # Collect arrays for all requested scalars.
        # First try exact match, then fall back to partial/substring match
        # so the pipeline works regardless of how the .case file was exported.
        scalar_vals = {}
        for skey in scalars_to_run:
            arr_name = cfg.SCALARS[skey]['array']
            if arr_name in cc.array_names:
                # Exact match
                scalar_vals[skey] = cc[arr_name]
            else:
                # Partial match — find any array whose name contains arr_name
                # or whose name is contained within arr_name
                matches = [a for a in cc.array_names
                           if arr_name.lower() in a.lower()
                           or a.lower() in arr_name.lower()]
                # Deduplicate (pyvista sometimes returns arrays twice)
                matches = list(dict.fromkeys(matches))
                if matches:
                    scalar_vals[skey] = cc[matches[0]]
                    if matches[0] != arr_name:
                        log(f"  [{skey}] array '{arr_name}' not found — "
                            f"using '{matches[0]}'")
                else:
                    log(f"  [{skey}] WARNING: no array matching '{arr_name}' "
                        f"in block {block_idx} — skipping")
        if not scalar_vals:
            continue

        unique = np.unique(np.round(pts[:, ax], 5))

        # Filter planes to car region using CAR_BOUNDS_2D
        # For each slice axis, the slicing coordinate range is
        # the extent of the OTHER two axes' bounds
        # e.g. X-slice: X range derived from Y-slice h_min/h_max isn't right —
        # use a dedicated per-axis range based on geometry bounds
        car_extent = {
            0: (cfg.CAR_BOUNDS_2D[1][0], cfg.CAR_BOUNDS_2D[1][1]),  # X: from Y-slice h range
            1: (cfg.CAR_BOUNDS_2D[0][0], cfg.CAR_BOUNDS_2D[0][1]),  # Y: from X-slice h range
            2: (cfg.CAR_BOUNDS_2D[0][2], cfg.CAR_BOUNDS_2D[0][3]),  # Z: from X-slice v range
        }
        lo, hi = car_extent[ax]
        unique_filtered = unique[(unique >= lo) & (unique <= hi)]

        if len(unique_filtered) == 0:
            log(f"  [WARN] Block {block_idx} axis {cfg.AXIS_LABEL[ax]}: "
                f"all {len(unique)} planes outside car bounds [{lo:.3f}, {hi:.3f}]m — "
                f"data range [{unique.min():.3f}, {unique.max():.3f}]m. "
                f"Check CAR_BOUNDS_2D in Settings.")
            continue
        unique = unique_filtered

        log(f"  Block {block_idx} (axis {cfg.AXIS_LABEL[ax]}): "
            f"{len(unique)} planes  scalars={list(scalar_vals.keys())}")

        for plane_idx, plane_val in enumerate(sorted(unique), start=1):
            jobs.append(dict(
                axis=ax, ha=ha, va=va,
                plane_idx=plane_idx,
                plane_val=float(plane_val),
                pts_all=pts,
                scalar_vals=scalar_vals,
            ))
    return jobs


# ── main entry point ──────────────────────────────────────────────
def run_pipeline(scalars=None, axes=None,
                 geom_case=None, data_case=None,
                 output_dir=None, resolution_mm=None,
                 fig_height=7.0, dpi=None):
    """
    Run the full slice → mask → interpolate → plot → save pipeline.

    Parameters
    ----------
    scalars       : list of str or None   e.g. ['Cp', 'CpT']
                    defaults to all keys in config.SCALARS
    axes          : list of str or None   e.g. ['X', 'Y']
                    defaults to ['X', 'Y', 'Z']
    geom_case     : str or None   override config.GEOMETRY_CASE
    data_case     : str or None   override config.DATA_CASE
    output_dir    : str or None   override config.OUTPUT_DIR
    resolution_mm : int or None   override config.RESOLUTION_MM
    """
    global _T0
    _T0 = time.perf_counter()

    scalars_to_run = scalars  or list(cfg.SCALARS.keys())
    axes_to_run    = axes     or ['X', 'Y', 'Z']
    out_dir        = output_dir    or cfg.OUTPUT_DIR
    res_mm         = resolution_mm or cfg.RESOLUTION_MM
    px             = 1000.0 / res_mm

    # Temporarily override config resolution for geometry/interp modules
    _orig_res = cfg.RESOLUTION_MM
    cfg.RESOLUTION_MM = res_mm

    # Create output dirs
    for scalar_key in scalars_to_run:
        for ax_label in axes_to_run:
            os.makedirs(os.path.join(out_dir, 'PNG', scalar_key, ax_label), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'NPZ'), exist_ok=True)
    for ax_label in axes_to_run:
        os.makedirs(os.path.join(out_dir, 'NPZ', ax_label), exist_ok=True)

    log("=" * 70)
    log(f"CFD Slicer Pipeline")
    log(f"  Scalars    : {scalars_to_run}")
    log(f"  Axes       : {axes_to_run}")
    log(f"  Resolution : {res_mm} mm/px")
    log(f"  Output     : {out_dir}")
    log("=" * 70)

    # ── STEP 1: load ─────────────────────────────────────────────
    log("STEP 1/4 — Loading geometry ...")
    t = time.perf_counter()
    geom_blocks = io_utils.load_geometry(geom_case)
    log(f"  {len(geom_blocks)} geometry parts", t)

    log("STEP 1/4 — Loading CFD data ...")
    t = time.perf_counter()
    cfd_mesh = io_utils.load_cfd(data_case)
    log(f"  {cfd_mesh.n_blocks} CFD blocks", t)

    # Print available arrays
    for i in range(cfd_mesh.n_blocks):
        b = cfd_mesh[i]
        if b is not None and b.n_cells > 0:
            log(f"  Block {i} arrays: {b.cell_centers().array_names}")

    # ── STEP 2: extract plane jobs ───────────────────────────────
    log("STEP 2/4 — Extracting plane positions ...")
    jobs  = _extract_plane_jobs(cfd_mesh, scalars_to_run, axes_to_run)
    total = len(jobs)
    log(f"  Total planes: {total}")

    # ── STEP 3+4: process each plane ────────────────────────────
    log(f"\nSTEP 3+4/4 — Processing {total} planes ...\n")

    t_mask_tot   = 0.0
    t_interp_tot = 0.0
    t_plot_tot   = 0.0
    t_save_tot   = 0.0

    for job_n, job in enumerate(jobs, start=1):
        ax        = job['axis']
        ha        = job['ha']
        va        = job['va']
        plane_idx = job['plane_idx']
        plane_val = job['plane_val']
        val_mm    = plane_val * 1000.0
        pts_all   = job['pts_all']
        scalar_vals = job['scalar_vals']

        ax_label         = cfg.AXIS_LABEL[ax]
        h_label, v_label = cfg.PLOT_LABELS[ax]
        h_min, h_max, v_min, v_max = cfg.CAR_BOUNDS_2D[ax]
        stem = f"slice_{ax_label}_{plane_idx:03d}_{val_mm:+.1f}mm"

        # Extract CFD points for this plane (use first available scalar for mask)
        first_vals = next(iter(scalar_vals.values()))
        on_plane  = np.abs(pts_all[:, ax] - plane_val) < 1e-4
        in_bounds = (on_plane &
                     (pts_all[:, ha] >= h_min) & (pts_all[:, ha] <= h_max) &
                     (pts_all[:, va] >= v_min) & (pts_all[:, va] <= v_max))
        plane_pts_2d = pts_all[in_bounds][:, [ha, va]]

        if len(plane_pts_2d) < 4:
            log(f"  [{job_n:3d}/{total}] {ax_label}#{plane_idx:03d} "
                f"{val_mm:+8.1f}mm  SKIP ({len(plane_pts_2d)} pts)")
            continue

        # 3a. geometry mask
        t0 = time.perf_counter()
        mask = geom_mod.build_mask(
            geom_blocks, ax, plane_val,
            h_min, h_max, v_min, v_max, px,
            plane_pts_2d=plane_pts_2d
        )
        t_mask_tot += time.perf_counter() - t0

        # 3b. interpolate all scalars
        t0 = time.perf_counter()
        scalar_grids = {}
        for skey, svals_all in scalar_vals.items():
            plane_svals = svals_all[in_bounds]
            scalar_grids[skey] = interp_mod.interpolate_per_region(
                plane_pts_2d, plane_svals, mask, h_min, v_min, px
            )
        t_interp_tot += time.perf_counter() - t0

        # 3c. save NPZ (one file per plane, all scalars inside)
        t0 = time.perf_counter()
        extent  = np.array([h_min, h_max, v_min, v_max], dtype=np.float32)
        npz_dir = os.path.join(out_dir, 'NPZ', ax_label)
        npz_path = io_utils.save_plane(
            npz_dir, stem, scalar_grids, mask, extent,
            ax, plane_val, plane_idx
        )
        npz_kb = os.path.getsize(npz_path) / 1024
        t_save_tot += time.perf_counter() - t0

        # 3d. save PNG per scalar
        t0 = time.perf_counter()
        for skey, cp_grid in scalar_grids.items():
            fig = plotting.make_slice_plot(
                cp_grid, mask, ax, plane_idx, val_mm,
                h_label, v_label, extent, skey,
                fig_height=fig_height,
                dpi=dpi,
            )
            png_dir  = os.path.join(out_dir, 'PNG', skey, ax_label)
            png_path = os.path.join(png_dir, stem + '.png')
            fig.savefig(png_path, bbox_inches='tight',
                        facecolor=cfg.BG_COLOR, dpi=dpi or cfg.DPI)
            import matplotlib.pyplot as plt
            plt.close(fig)
        t_plot_tot += time.perf_counter() - t0

        fluid_px = int((~np.isnan(next(iter(scalar_grids.values()))) & ~mask).sum())
        log(f"  [{job_n:3d}/{total}]  {ax_label}#{plane_idx:03d} {val_mm:+8.1f}mm  "
            f"solid={100*mask.mean():4.1f}%  fluid={fluid_px:>7,}px  "
            f"npz={npz_kb:4.0f}KB")

    cfg.RESOLUTION_MM = _orig_res

    t_wall = time.perf_counter() - _T0
    log("\n" + "=" * 70)
    log(f"DONE  —  {total} planes  |  {out_dir}")
    log(f"  Mask        : {t_mask_tot:.1f}s   avg {t_mask_tot/max(total,1)*1000:.0f}ms/plane")
    log(f"  Interpolate : {t_interp_tot:.1f}s   avg {t_interp_tot/max(total,1)*1000:.0f}ms/plane")
    log(f"  Plot        : {t_plot_tot:.1f}s   avg {t_plot_tot/max(total,1)*1000:.0f}ms/plane")
    log(f"  Save NPZ    : {t_save_tot:.1f}s   avg {t_save_tot/max(total,1)*1000:.0f}ms/plane")
    log(f"  Wall clock  : {t_wall:.1f}s")
    log("=" * 70)