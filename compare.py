"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
================================================================================

compare.py
==========
Compare two pipeline output directories plane-by-plane.

compare_runs(dir_a, dir_b, scalars=None, axes=None,
             output_dir=None, label_a='Run A', label_b='Run B',
             log_a=None, log_b=None,
             fig_height=7.0, dpi=None,
             delta_ranges=None)

delta_ranges : dict[scalar_key, float or None]
    Per-scalar fixed colormap range for delta plots.
    float -> fixed symmetric range +/- value
    None  -> auto per-plane range
    e.g. {'Cp': 0.1, 'CpT': None, 'Vi': 0.2}
    Missing keys default to None (auto).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import io_utils
import plotting


def load_and_diff(path_a, path_b, scalar_key):
    """
    Load two matching plane NPZ files.

    Returns
    -------
    grid_a, grid_b, mask_a, mask_b, extent, meta
    """
    da = io_utils.load_plane(path_a)
    db = io_utils.load_plane(path_b)

    if scalar_key not in da:
        raise KeyError(f"Scalar '{scalar_key}' not found in {path_a}")
    if scalar_key not in db:
        raise KeyError(f"Scalar '{scalar_key}' not found in {path_b}")

    grid_a = da[scalar_key].astype(np.float32)
    grid_b = db[scalar_key].astype(np.float32)
    mask_a = da['mask']
    mask_b = db['mask']
    extent = da['extent']

    if grid_a.shape != grid_b.shape:
        raise ValueError(
            f"Grid shape mismatch: {grid_a.shape} vs {grid_b.shape}. "
            "Both runs must use the same RESOLUTION_MM and CAR_BOUNDS_2D."
        )

    meta = {
        'plane_axis':    int(da['plane_axis']),
        'plane_value_m': float(da['plane_value_m']),
        'plane_index':   int(da['plane_index']),
        'h_axis':        da['h_axis'],
        'v_axis':        da['v_axis'],
    }
    return grid_a, grid_b, mask_a, mask_b, extent, meta


def _diff_stats(grid_a, grid_b, mask_a, mask_b):
    """Stats on delta over fluid pixels present in both runs."""
    diff  = grid_b.astype(np.float32) - grid_a.astype(np.float32)
    fluid = diff[~(mask_a | mask_b) & ~np.isnan(diff)]
    if len(fluid) == 0:
        return dict(n=0, mean=np.nan, std=np.nan,
                    min=np.nan, max=np.nan,
                    p05=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        n    = len(fluid),
        mean = float(np.mean(fluid)),
        std  = float(np.std(fluid)),
        min  = float(np.min(fluid)),
        max  = float(np.max(fluid)),
        p05  = float(np.percentile(fluid,  5)),
        p50  = float(np.percentile(fluid, 50)),
        p95  = float(np.percentile(fluid, 95)),
    )


def compare_runs(dir_a, dir_b,
                 scalars=None, axes=None,
                 output_dir=None,
                 label_a='Run A', label_b='Run B',
                 log_a=None, log_b=None,
                 fig_height=7.0, dpi=None,
                 delta_ranges=None):
    """
    Compare two run output directories.

    Parameters
    ----------
    dir_a / dir_b  : str   root output dirs (each contains NPZ/<axis>/)
    scalars        : list of str or None
    axes           : list of str or None
    output_dir     : str or None
    label_a/b      : str
    log_a / log_b  : str or None   .log paths for aero report
    fig_height     : float
    dpi            : int or None
    delta_ranges   : dict[str, float or None] or None
        Per-scalar fixed range for delta colormap.
        float -> fixed +/- value applied to every plane for that scalar.
        None  -> auto per-plane.
        Missing scalar keys default to None (auto).

    Returns
    -------
    summary : list of dicts
    """
    scalars_to_run = scalars  or list(cfg.SCALARS.keys())
    axes_to_run    = axes     or ['X', 'Y', 'Z']
    delta_ranges   = delta_ranges or {}

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(dir_b),
            f"compare_{os.path.basename(dir_b)}_vs_{os.path.basename(dir_a)}"
        )

    npz_root_a = os.path.join(dir_a, 'NPZ')
    npz_root_b = os.path.join(dir_b, 'NPZ')

    planes_a = {(ax, pidx): path
                for ax, pidx, pmm, path
                in io_utils.find_planes(npz_root_a)}
    planes_b = {(ax, pidx): path
                for ax, pidx, pmm, path
                in io_utils.find_planes(npz_root_b)}

    matching = sorted(set(planes_a.keys()) & set(planes_b.keys()))

    print(f"Comparing {len(matching)} matching planes  scalars={scalars_to_run}")
    print(f"  {label_a}: {dir_a}")
    print(f"  {label_b}: {dir_b}")
    print(f"  Output  : {output_dir}")
    print(f"  Delta ranges:")
    for skey in scalars_to_run:
        dr = delta_ranges.get(skey)
        print(f"    {skey}: {'auto per-plane' if dr is None else f'fixed +/-{dr}'}")
    print()

    summary = []

    for ax_str, pidx in matching:
        if ax_str not in axes_to_run:
            continue

        path_a = planes_a[(ax_str, pidx)]
        path_b = planes_b[(ax_str, pidx)]

        for skey in scalars_to_run:
            try:
                grid_a, grid_b, mask_a, mask_b, extent, meta = \
                    load_and_diff(path_a, path_b, skey)
            except KeyError as e:
                print(f"  [{ax_str}#{pidx:03d}] {skey}: SKIP -- {e}")
                continue

            ax_slice  = meta['plane_axis']
            plane_val = meta['plane_value_m']
            plane_idx = meta['plane_index']
            val_mm    = plane_val * 1000.0
            h_label, v_label = cfg.PLOT_LABELS[ax_slice]

            stats = _diff_stats(grid_a, grid_b, mask_a, mask_b)
            summary.append(dict(
                axis=ax_str, plane_idx=plane_idx, val_mm=val_mm,
                scalar=skey, label_a=label_a, label_b=label_b,
                **stats
            ))

            # Delta PNG
            png_dir = os.path.join(output_dir, 'PNG', skey, ax_str)
            os.makedirs(png_dir, exist_ok=True)
            stem = f"diff_{skey}_{ax_str}_{plane_idx:03d}_{val_mm:+.1f}mm"

            fig = plotting.make_diff_plot(
                grid_a, grid_b, mask_a, mask_b,
                ax_slice, plane_idx, val_mm,
                h_label, v_label, extent, skey,
                label_a=label_a, label_b=label_b,
                fig_height=fig_height, dpi=dpi,
                fixed_range=delta_ranges.get(skey),
            )
            fig.savefig(os.path.join(png_dir, stem + '.png'),
                        bbox_inches='tight',
                        facecolor=cfg.BG_COLOR,
                        dpi=dpi or cfg.DPI)
            plt.close(fig)

            range_note = (f"fixed+/-{delta_ranges[skey]}"
                          if skey in delta_ranges and delta_ranges[skey] is not None
                          else "auto")
            print(f"  [{ax_str}#{plane_idx:03d}  {val_mm:+.1f}mm]  {skey}  "
                  f"mean={stats['mean']:+.4f}  "
                  f"max={stats['max']:+.4f}  "
                  f"range={range_note}")

    # Summary CSV
    if summary:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'comparison_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"\nSummary CSV: {csv_path}")

    # Aero report
    if log_a and log_b:
        try:
            import aero_report as ar
            print("\nGenerating aero report...")
            n_a, d_a = ar.parse_log(log_a)
            n_b, d_b = ar.parse_log(log_b)
            ar.generate_aero_report(
                n_a, d_a, n_b, d_b,
                output_dir,
                parent_label=label_a,
                new_label=label_b
            )
        except Exception as e:
            print(f"[aero_report] ERROR: {e}")
            import traceback; traceback.print_exc()

    return summary