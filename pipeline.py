"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
================================================================================

pipeline.py
===========
Slice -> mask -> interpolate -> plot -> save, for every plane in the CFD data.

Recruits (parallel planes)  —  original beta by George Wei
--------------------------
run_pipeline(..., ram_budget_gb=G) processes several planes at once in
separate worker processes ("recruits"). The number of recruits comes from
recruits.plan(): RAM budget / (per-recruit overhead + geometry copy +
largest plane working set), capped by CPU cores and plane count.
ram_budget_gb=None (default) keeps the original one-plane-at-a-time path.
"""

import os
import sys
import time
import numpy as np

import config as cfg
import io_utils
import geometry as geom_mod
import interpolation as interp_mod
import plotting
import recruits as rc


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


def _plane_inputs(job):
    """
    Cut one plane's CFD points / values out of the full block arrays.
    Returns (meta, plane_pts_2d, {skey: plane_vals}) — only what a worker
    needs, so recruits never receive the full block.
    """
    ax, ha, va = job['axis'], job['ha'], job['va']
    plane_val  = job['plane_val']
    pts_all    = job['pts_all']
    h_min, h_max, v_min, v_max = cfg.CAR_BOUNDS_2D[ax]

    on_plane  = np.abs(pts_all[:, ax] - plane_val) < 1e-4
    in_bounds = (on_plane &
                 (pts_all[:, ha] >= h_min) & (pts_all[:, ha] <= h_max) &
                 (pts_all[:, va] >= v_min) & (pts_all[:, va] <= v_max))
    plane_pts_2d = np.ascontiguousarray(pts_all[in_bounds][:, [ha, va]])
    plane_svals  = {skey: np.ascontiguousarray(svals[in_bounds])
                    for skey, svals in job['scalar_vals'].items()}
    meta = dict(axis=ax, ha=ha, va=va,
                plane_idx=job['plane_idx'], plane_val=plane_val)
    return meta, plane_pts_2d, plane_svals


def process_plane(meta, plane_pts_2d, plane_svals, geom_blocks,
                  out_dir, fig_height=7.0, dpi=None):
    """
    Mask, interpolate, save NPZ and PNGs for one plane.
    Runs either in the main process or inside a recruit.

    Returns dict(msg, t_mask, t_interp, t_save, t_plot).
    """
    import matplotlib.pyplot as plt

    ax        = meta['axis']
    plane_idx = meta['plane_idx']
    plane_val = meta['plane_val']
    val_mm    = plane_val * 1000.0
    px        = 1000.0 / cfg.RESOLUTION_MM

    ax_label         = cfg.AXIS_LABEL[ax]
    h_label, v_label = cfg.PLOT_LABELS[ax]
    h_min, h_max, v_min, v_max = cfg.CAR_BOUNDS_2D[ax]
    stem = f"slice_{ax_label}_{plane_idx:03d}_{val_mm:+.1f}mm"

    # 3a. geometry mask
    t0 = time.perf_counter()
    mask = geom_mod.build_mask(
        geom_blocks, ax, plane_val,
        h_min, h_max, v_min, v_max, px,
        plane_pts_2d=plane_pts_2d
    )
    t_mask = time.perf_counter() - t0

    # 3b. interpolate all scalars
    t0 = time.perf_counter()
    scalar_grids = {}
    for skey, plane_svals_k in plane_svals.items():
        scalar_grids[skey] = interp_mod.interpolate_per_region(
            plane_pts_2d, plane_svals_k, mask, h_min, v_min, px
        )
    t_interp = time.perf_counter() - t0

    # 3c. save NPZ (one file per plane, all scalars inside)
    t0 = time.perf_counter()
    extent   = np.array([h_min, h_max, v_min, v_max], dtype=np.float32)
    npz_dir  = os.path.join(out_dir, 'NPZ', ax_label)
    npz_path = io_utils.save_plane(
        npz_dir, stem, scalar_grids, mask, extent,
        ax, plane_val, plane_idx
    )
    npz_kb = os.path.getsize(npz_path) / 1024
    t_save = time.perf_counter() - t0

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
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.join(png_dir, stem + '.png')
        fig.savefig(png_path, bbox_inches='tight',
                    facecolor=cfg.BG_COLOR, dpi=dpi or cfg.DPI)
        plt.close(fig)
    t_plot = time.perf_counter() - t0

    fluid_px = int((~np.isnan(next(iter(scalar_grids.values()))) & ~mask).sum())
    msg = (f"{ax_label}#{plane_idx:03d} {val_mm:+8.1f}mm  "
           f"solid={100*mask.mean():4.1f}%  fluid={fluid_px:>7,}px  "
           f"npz={npz_kb:4.0f}KB")
    return dict(msg=msg, t_mask=t_mask, t_interp=t_interp,
                t_save=t_save, t_plot=t_plot)


# ── recruit (worker process) side ─────────────────────────────────
_W = {}

def _worker_init(geom_case, snapshot):
    """Runs once in every recruit: restore config, load geometry."""
    if sys.stdout is None:                       # pythonw on Windows
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')
    cfg.apply_snapshot(snapshot)
    import matplotlib
    matplotlib.use('Agg')
    _W['geom'] = io_utils.load_geometry(geom_case)


def _worker_run(meta, plane_pts_2d, plane_svals, out_dir, fig_height, dpi):
    import traceback
    try:
        return process_plane(meta, plane_pts_2d, plane_svals, _W['geom'],
                             out_dir, fig_height, dpi)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc())


def _run_parallel(jobs, n_rec, geom_path, out_dir, fig_height, dpi, on_result):
    """
    Feed planes to n_rec recruits. Only ~2 planes per recruit are queued at
    a time so the main process never holds every plane's data at once.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from concurrent.futures.process import BrokenProcessPool

    ctx = mp.get_context('spawn')     # same behaviour on Windows / Linux / mac
    total = len(jobs)
    queue = iter(enumerate(jobs, start=1))
    in_flight = {}

    with ProcessPoolExecutor(max_workers=n_rec, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(geom_path, cfg.runtime_snapshot())) as ex:

        def _top_up():
            while len(in_flight) < n_rec * 2:
                try:
                    job_n, job = next(queue)
                except StopIteration:
                    return
                meta, pts2d, svals = _plane_inputs(job)
                if len(pts2d) < 4:
                    on_result(job_n, total, meta, None, skipped=len(pts2d))
                    continue
                fut = ex.submit(_worker_run, meta, pts2d, svals,
                                out_dir, fig_height, dpi)
                in_flight[fut] = (job_n, meta)

        _top_up()
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                job_n, meta = in_flight.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    raise RuntimeError(
                        "A recruit crashed (most likely out of RAM). "
                        "Move the RAM slider down and run again.")
                on_result(job_n, total, meta, res)
            _top_up()


# ── main entry point ──────────────────────────────────────────────
def run_pipeline(scalars=None, axes=None,
                 geom_case=None, data_case=None,
                 output_dir=None, resolution_mm=None,
                 fig_height=7.0, dpi=None,
                 ram_budget_gb=None, recruits=None):
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
    ram_budget_gb : float or None RAM the recruits may use in total.
                    None -> config.RECRUIT_RAM_GB (None = one plane at a time)
    recruits      : int or None   force an exact number of planes at once
                    (overrides the RAM budget)
    """
    global _T0
    _T0 = time.perf_counter()

    scalars_to_run = scalars  or list(cfg.SCALARS.keys())
    axes_to_run    = axes     or ['X', 'Y', 'Z']
    out_dir        = output_dir    or cfg.OUTPUT_DIR
    res_mm         = resolution_mm or cfg.RESOLUTION_MM
    geom_path      = geom_case or cfg.GEOMETRY_CASE
    if ram_budget_gb is None:
        ram_budget_gb = cfg.RECRUIT_RAM_GB
    if dpi:
        cfg.DPI = dpi

    # Temporarily override config resolution for geometry/interp modules
    _orig_res = cfg.RESOLUTION_MM
    cfg.RESOLUTION_MM = res_mm

    try:
        _run(scalars_to_run, axes_to_run, geom_path, data_case, out_dir,
             res_mm, fig_height, dpi, ram_budget_gb, recruits)
    finally:
        cfg.RESOLUTION_MM = _orig_res


def _run(scalars_to_run, axes_to_run, geom_path, data_case, out_dir,
         res_mm, fig_height, dpi, ram_budget_gb, recruits):

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
    geom_blocks = io_utils.load_geometry(geom_path)
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

    # ── Recruits: how many planes at once ────────────────────────
    n_rec = _decide_recruits(jobs, scalars_to_run, axes_to_run, res_mm,
                             fig_height, dpi, geom_path,
                             ram_budget_gb, recruits)

    # ── STEP 3+4: process each plane ────────────────────────────
    log(f"\nSTEP 3+4/4 — Processing {total} planes "
        f"({n_rec} at once) ...\n")

    totals = dict(t_mask=0.0, t_interp=0.0, t_plot=0.0, t_save=0.0)
    counts = dict(done=0, skipped=0, failed=0)

    def _on_result(job_n, total, meta, res, skipped=None):
        ax_label = cfg.AXIS_LABEL[meta['axis']]
        val_mm   = meta['plane_val'] * 1000.0
        if res is None:
            counts['skipped'] += 1
            log(f"  [{job_n:3d}/{total}] {ax_label}#{meta['plane_idx']:03d} "
                f"{val_mm:+8.1f}mm  SKIP ({skipped} pts)")
            return
        if 'error' in res:
            counts['failed'] += 1
            log(f"  [{job_n:3d}/{total}] {ax_label}#{meta['plane_idx']:03d} "
                f"{val_mm:+8.1f}mm  ERROR {res['error']}")
            print(res.get('trace', ''))
            return
        counts['done'] += 1
        for k in totals:
            totals[k] += res[k]
        log(f"  [{job_n:3d}/{total}]  {res['msg']}")

    if n_rec > 1:
        log(f"  {rc.CREDIT}\n")

    if n_rec <= 1:
        for job_n, job in enumerate(jobs, start=1):
            meta, pts2d, svals = _plane_inputs(job)
            if len(pts2d) < 4:
                _on_result(job_n, total, meta, None, skipped=len(pts2d))
                continue
            try:
                res = process_plane(meta, pts2d, svals, geom_blocks,
                                    out_dir, fig_height, dpi)
            except Exception as exc:
                import traceback
                res = dict(error=f"{type(exc).__name__}: {exc}",
                           trace=traceback.format_exc())
            _on_result(job_n, total, meta, res)
    else:
        del geom_blocks           # recruits load their own copy
        _run_parallel(jobs, n_rec, geom_path, out_dir, fig_height, dpi,
                      _on_result)

    t_wall = time.perf_counter() - _T0
    n_ok = max(counts['done'], 1)
    per = " (summed across recruits)" if n_rec > 1 else ""
    log("\n" + "=" * 70)
    log(f"DONE  —  {counts['done']}/{total} planes"
        f"{'  skipped=' + str(counts['skipped']) if counts['skipped'] else ''}"
        f"{'  FAILED=' + str(counts['failed']) if counts['failed'] else ''}"
        f"  |  {out_dir}")
    log(f"  Recruits    : {n_rec} plane(s) at once")
    log(f"  Mask        : {totals['t_mask']:.1f}s   avg {totals['t_mask']/n_ok*1000:.0f}ms/plane{per}")
    log(f"  Interpolate : {totals['t_interp']:.1f}s   avg {totals['t_interp']/n_ok*1000:.0f}ms/plane{per}")
    log(f"  Plot        : {totals['t_plot']:.1f}s   avg {totals['t_plot']/n_ok*1000:.0f}ms/plane{per}")
    log(f"  Save NPZ    : {totals['t_save']:.1f}s   avg {totals['t_save']/n_ok*1000:.0f}ms/plane{per}")
    log(f"  Wall clock  : {t_wall:.1f}s")
    log("=" * 70)


def _decide_recruits(jobs, scalars, axes, res_mm, fig_height, dpi,
                     geom_path, ram_budget_gb, recruits):
    """Number of planes processed at once, logged with the reasoning."""
    total = len(jobs)
    if total <= 1:
        return 1
    if recruits:
        n = max(1, min(int(recruits), total))
        log(f"  Recruits    : {n} (set explicitly)")
        return n
    if not ram_budget_gb:
        log("  Recruits    : off — one plane at a time")
        return 1

    # Same estimate the Run-tab slider shows ...
    p_gui = rc.plan(ram_budget_gb, axes, res_mm, len(scalars), fig_height,
                    dpi, geom_path, n_planes=total)
    n = p_gui['recruits']

    # ... then re-check with the real CFD point counts per plane.
    per_block = {}
    for job in jobs:                       # one pass per block, not per plane
        key = (id(job['pts_all']), job['axis'])
        if key not in per_block:
            _, cnt = np.unique(np.round(job['pts_all'][:, job['axis']], 5),
                               return_counts=True)
            per_block[key] = int(cnt.max()) if cnt.size else 0
    max_pts = max(per_block.values()) if per_block else 0
    p_real = rc.plan(ram_budget_gb, axes, res_mm, len(scalars), fig_height,
                     dpi, geom_path, n_pts=max_pts, n_planes=total)
    if p_real['recruits'] < n:
        log(f"  Recruits    : reduced {n} -> {p_real['recruits']}  "
            f"(largest plane has {max_pts:,} CFD points — more than the "
            f"slider estimate assumed)")
        n = p_real['recruits']
        p_gui = p_real

    why = {'ram': 'RAM budget', 'cpu': f"CPU cores (max {p_gui['cpu_max']})",
           'planes': 'number of planes', 'none': ''}[p_gui['limit']]
    log(f"  Recruits    : {n} planes at once  "
        f"(~{p_gui['per_recruit_gb']:.2f} GB each, "
        f"{n * p_gui['per_recruit_gb']:.1f} of {ram_budget_gb:.1f} GB budget"
        f"{'; limited by ' + why if why else ''})")
    return n
