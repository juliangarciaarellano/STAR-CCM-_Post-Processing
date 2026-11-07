"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
================================================================================

surface_cp.py
=============
Surface Cp (or any surface scalar) along a slice through the car — the
classic Cp-vs-chord plot — for any X / Y / Z position.

Full nodes, no interpolation
----------------------------
The CFD surface mesh is cut exactly at the slice plane. Every plotted value
is the value of a real CFD surface cell (one value per intersected cell, at
the segment midpoint) or, for node-based exports, the exact edge
intersection between two CFD nodes. Nothing is resampled, smoothed or
gridded.

Functions
---------
load_surface(surface_case, geom_case) -> list of {name, mesh}  (geometry parts only)
parse_positions("100, 250, 400:800:100") -> [100, 250, 400, ..., 800]  (mm)
section(parts, axis, pos_m, scalar_key, part_names=None)
                                      -> list of curve dicts
plot_section(curves, axis, pos_mm, scalar_key, ...) -> matplotlib Figure
write_csv(curves, path, axis, scalar_key)
export_sections(parts, axis, positions_mm, scalar_key, output_dir, ...)
                                      -> list of written files
"""

import os
import csv
import re
import time
import numpy as np

import config as cfg

AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}

# One colour per part (Okabe-Ito + extras, readable on white, colour-blind safe)
PART_COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
                '#56B4E9', '#000000', '#F0E442', '#882255', '#44AA99',
                '#999933', '#AA4499']


# ══════════════════════════════════════════════════════════════════
#  LOADING
# ══════════════════════════════════════════════════════════════════

def load_surface(surface_case=None, geom_case=None):
    """
    Load the surface parts to slice — the geometry's parts, as PolyData.

    geom_case    : geometry .case. Its parts are the parts listed and sliced,
                   always — whatever is (or is not) in the surface file.
    surface_case : optional .case with the surface Cp/Cf. Each geometry part
                   takes its data from the matching block here (by name, or
                   by position when the bounding boxes agree). Parts with no
                   match — or every part, when the file holds only plane
                   sections such as Raw_Data.case — use the Cp/Cf stored in
                   the geometry .case itself (UT26 Geometry.case carries it).
    With no geom_case, the non-plane blocks of surface_case are used.
    Plane sections (X_Planes, ...) are never sliced.
    """
    import pyvista as pv
    import io_utils
    import surface_render as sr
    pv.global_theme.allow_empty_mesh = True
    t0 = time.perf_counter()
    if not (surface_case or geom_case):
        raise FileNotFoundError("Set the Geometry .case (or a Surface data .case).")
    same = bool(surface_case and geom_case and
                os.path.abspath(surface_case) == os.path.abspath(geom_case))

    if geom_case:
        print(f"[surface_cp] Loading geometry parts: {geom_case}")
        io_utils.check_case(geom_case, 'Geometry')
        geom_parts = sr._blocks(pv.read(geom_case))
        matched = {}
        if surface_case and not same:
            print(f"[surface_cp] Loading surface data (full nodes): {surface_case}")
            io_utils.check_case(surface_case, 'Surface data')
            surf_parts = sr._blocks(pv.read(surface_case))
            matched, skipped = sr.match_to_geometry(geom_parts, surf_parts)
            if skipped:
                print(f"  not geometry parts — not sliced: {skipped}")
            if not matched:
                print("  [NOTE] the Surface data .case has no geometry parts "
                      "(plane sections such as Raw_Data.case are not surface "
                      "data) — using the Cp/Cf stored in the Geometry .case.")
        else:
            print("  using the surface data stored in the geometry .case")
        keep = [(gi, name, matched.get(gi, g), gi in matched)
                for gi, name, g in geom_parts]
    else:
        print(f"[surface_cp] Loading surface data (full nodes): {surface_case}")
        io_utils.check_case(surface_case, 'Surface data')
        keep = [(i, n, b, True) for i, n, b in sr._blocks(pv.read(surface_case))]

    planes = [n for _, n, _, _ in keep if sr.PLANE_NAME_RE.search(n)]
    if planes:
        print(f"  plane sections — not sliced: {planes}")
        keep = [k for k in keep if k[1] not in planes]

    parts = []
    for i, name, b, from_surf in keep:
        mesh = b if isinstance(b, pv.PolyData) else \
            b.extract_surface(algorithm='dataset_surface')
        has = [k for k in list(mesh.cell_data.keys()) + list(mesh.point_data.keys())
               if not k.startswith('vtkOriginal')]
        tag = ("surface data file" if (from_surf and geom_case) else
               "geometry .case data" if has else "no surface data")
        parts.append(dict(name=name, mesh=mesh))
        print(f"  {name}: {mesh.n_cells:,} cells  ({tag})")
    if not parts:
        src = geom_case or surface_case
        raise ValueError(
            f"No geometry parts found in {os.path.basename(src)} — it only "
            f"holds plane sections ({', '.join(planes) or 'nothing else'}). "
            f"Set the Geometry .case to the geometry export (e.g. Geometry.case).")
    print(f"[surface_cp] {len(parts)} parts in {time.perf_counter()-t0:.1f}s")
    return parts


def scalar_array_name(skey):
    sd = cfg.SURFACE_SCALARS.get(skey) or cfg.SCALARS.get(skey) or {}
    return sd.get('array', skey)


def _find_array(mesh, name):
    for where, data in (('cell', mesh.cell_data), ('point', mesh.point_data)):
        if name in data:
            return where, name
    for where, data in (('cell', mesh.cell_data), ('point', mesh.point_data)):
        for k in data.keys():
            if name.lower() in k.lower() or k.lower() in name.lower():
                return where, k
    return None, None


# ══════════════════════════════════════════════════════════════════
#  POSITIONS
# ══════════════════════════════════════════════════════════════════

def parse_positions(text):
    """
    Parse slice positions in mm.
      "300"                 -> [300]
      "100, 250, -400"      -> [100, 250, -400]
      "400:800:100"         -> [400, 500, 600, 700, 800]   (start:stop:step)
    Separators: comma, semicolon or whitespace. Duplicates removed, order kept.
    """
    out = []
    for tok in re.split(r'[,\s;]+', (text or '').strip()):
        if not tok:
            continue
        if ':' in tok:
            parts = [float(p) for p in tok.split(':')]
            if len(parts) == 2:
                start, stop, step = parts[0], parts[1], 10.0
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Bad range '{tok}' — use start:stop:step")
            if step == 0:
                raise ValueError(f"Step cannot be 0 in '{tok}'")
            step = abs(step) if stop >= start else -abs(step)
            n = int(np.floor((stop - start) / step + 1e-9)) + 1
            out += [round(start + i * step, 6) for i in range(n)]
        else:
            out.append(float(tok))
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


# ══════════════════════════════════════════════════════════════════
#  SLICING  (exact plane / CFD-surface intersection)
# ══════════════════════════════════════════════════════════════════

def _segments(sl):
    """
    Line segments of a slice result as (node_a, node_b, cell_id) arrays.
    Coincident points are merged so neighbouring segments share a node.
    """
    if sl.n_lines == 0:
        return None
    lines = np.asarray(sl.lines)
    a_list, b_list, c_list = [], [], []
    i, cell = 0, sl.n_verts                 # cell_data index: verts come first
    while i < len(lines):
        n = int(lines[i]); ids = lines[i + 1:i + 1 + n]
        for j in range(n - 1):
            a_list.append(ids[j]); b_list.append(ids[j + 1]); c_list.append(cell)
        i += n + 1; cell += 1
    a = np.asarray(a_list); b = np.asarray(b_list); c = np.asarray(c_list)

    pts = np.asarray(sl.points)
    scale = max(np.ptp(pts, axis=0).max(), 1e-9)
    key = np.round(pts / (scale * 1e-9)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    a, b = inv[a], inv[b]
    keep = a != b                            # drop zero-length segments
    first = np.zeros(inv.max() + 1, dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]   # representative point
    return a[keep], b[keep], c[keep], first


def _chains(a, b):
    """
    Order segments into polylines. Returns list of (node_seq, seg_seq, closed).
    Stops at open ends and at junctions (nodes shared by > 2 segments).
    """
    from collections import defaultdict
    adj = defaultdict(list)
    for s, (u, v) in enumerate(zip(a, b)):
        adj[u].append((s, v)); adj[v].append((s, u))
    used = np.zeros(len(a), bool)
    chains = []

    def walk(start, s0):
        nodes, segs = [start], []
        node, s = start, s0
        while True:
            used[s] = True; segs.append(s)
            u, v = a[s], b[s]
            node = v if u == node else u
            nodes.append(node)
            if len(adj[node]) != 2:
                return nodes, segs, False
            nxt = [t for t, _ in adj[node] if not used[t]]
            if not nxt:
                return nodes, segs, node == start
            s = nxt[0]

    # open chains first (start at ends / junctions), then closed loops
    for node in [n for n in adj if len(adj[n]) != 2]:
        for s, _ in adj[node]:
            if not used[s]:
                chains.append(walk(node, s))
    for s in range(len(a)):
        if not used[s]:
            chains.append(walk(a[s], s))
    return chains


def section(parts, axis, pos_m, scalar_key, part_names=None):
    """
    Cut every selected part at the plane and return ordered curves.

    Each curve dict:
      part, loop, closed, kind ('cell' | 'node')
      xyz     (n,3)  plotted positions: segment midpoints (cell data)
                     or edge intersections (node data)
      value   (n,)   exact CFD values at those positions
      seg_xyz (m,2,3) segment end points, for the profile drawing
      seg_val (m,)    value per segment
    """
    ax = AXIS_IDX.get(axis, axis)
    arr_name = scalar_array_name(scalar_key)
    origin = [0.0, 0.0, 0.0]; origin[ax] = pos_m
    curves = []
    for part in parts:
        if part_names and part['name'] not in part_names:
            continue
        mesh = part['mesh']
        lo, hi = mesh.bounds[2 * ax], mesh.bounds[2 * ax + 1]
        if pos_m < lo or pos_m > hi:
            continue
        where, arr = _find_array(mesh, arr_name)
        if where is None:
            print(f"  [WARN] '{arr_name}' not on '{part['name']}' — skipped")
            continue
        sl = mesh.slice(normal=cfg.AXIS_NORMAL[ax], origin=origin)
        if sl is None or sl.n_cells == 0:
            continue
        seg = _segments(sl)
        if seg is None:
            continue
        a, b, c, rep = seg
        pts = np.asarray(sl.points)
        node_xyz = pts[rep]
        if where == 'cell':
            cell_vals = np.asarray(sl.cell_data[arr], dtype=float)
            seg_vals = cell_vals[c]
        else:
            pvals = np.asarray(sl.point_data[arr], dtype=float)
            node_vals = pvals[rep]
            seg_vals = 0.5 * (node_vals[a] + node_vals[b])   # colour only

        for li, (nodes, segs, closed) in enumerate(_chains(a, b)):
            segs = np.asarray(segs); nodes = np.asarray(nodes)
            seg_xyz = np.stack([node_xyz[a[segs]], node_xyz[b[segs]]], axis=1)
            # orient each segment along the walk direction
            flip = a[segs] != nodes[:-1]
            seg_xyz[flip] = seg_xyz[flip][:, ::-1]
            if where == 'cell':
                xyz = seg_xyz.mean(axis=1)
                val = seg_vals[segs]
                kind = 'cell'
            else:
                xyz = node_xyz[nodes]
                val = node_vals[nodes]
                kind = 'node'
            curves.append(dict(part=part['name'], loop=li, closed=bool(closed),
                               kind=kind, xyz=xyz, value=val,
                               seg_xyz=seg_xyz, seg_val=seg_vals[segs]))
    return curves


# ══════════════════════════════════════════════════════════════════
#  PLOT + CSV
# ══════════════════════════════════════════════════════════════════

def plot_section(curves, axis, pos_mm, scalar_key, invert=True,
                 h_lim=None, val_lim=None, show_profile=True,
                 title_extra='', fig_size=(12.0, 8.0), dpi=150):
    """
    Top: scalar vs horizontal position, one colour per part (Cp axis
    inverted by default, suction up). Bottom: the section itself coloured
    by the scalar, drawn at true scale whenever it fits.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mc
    import plotting

    ax_i = AXIS_IDX.get(axis, axis)
    ha, va = cfg.PLOT_AXES[ax_i]
    h_label, v_label = cfg.PLOT_LABELS[ax_i]
    sd = cfg.SURFACE_SCALARS.get(scalar_key) or cfg.SCALARS.get(scalar_key) or {}
    label = sd.get('label', scalar_key)
    vmin, vmax = sd.get('vmin', -4.0), sd.get('vmax', 1.0)
    cmap = sd.get('cmap', 'viridis')

    segs = np.concatenate([c['seg_xyz'][:, :, [ha, va]] for c in curves])
    vals = np.concatenate([c['seg_val'] for c in curves])
    allh = segs[:, :, 0].ravel()
    if h_lim:
        h_lo, h_hi = h_lim
    else:
        span = max(allh.max() - allh.min(), 1e-6)
        h_lo, h_hi = allh.min() - 0.03 * span, allh.max() + 0.03 * span
    in_view = ((segs[:, :, 0] >= h_lo) & (segs[:, :, 0] <= h_hi)).any(axis=1)
    vv = segs[in_view][:, :, 1] if in_view.any() else segs[:, :, 1]
    v_lo, v_hi = float(vv.min()), float(vv.max())
    v_pad = 0.08 * max(v_hi - v_lo, 1e-3)
    v_lo, v_hi = v_lo - v_pad, v_hi + v_pad

    fig_w, fig_h = fig_size
    top_h = fig_h * 0.62
    prof_h, true_shape, exag = 0.0, True, 1.0
    if show_profile:
        axes_w = fig_w * 0.80                    # approx. plot width (in)
        need_h = axes_w * (v_hi - v_lo) / (h_hi - h_lo)
        prof_h = min(max(need_h, 0.9), 4.0)
        true_shape = need_h <= 4.0
        exag = prof_h / need_h if need_h > 0 else 1.0
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(fig_w, top_h + prof_h + 1.0), dpi=dpi, sharex=True,
            gridspec_kw=dict(height_ratios=[top_h, prof_h], hspace=0.08))
    else:
        fig, ax1 = plt.subplots(figsize=(fig_w, top_h + 0.8), dpi=dpi)
        ax2 = None
    fig.patch.set_facecolor('white')

    parts = list(dict.fromkeys(c['part'] for c in curves))
    colour = {p: PART_COLOURS[i % len(PART_COLOURS)] for i, p in enumerate(parts)}
    n_pts = sum(len(c['value']) for c in curves)

    for p in parts:
        first = True
        for c in (c for c in curves if c['part'] == p):
            h = c['xyz'][:, ha]; y = c['value']
            if c['closed'] and len(h) > 2:
                h = np.r_[h, h[:1]]; y = np.r_[y, y[:1]]
            ax1.plot(h, y, '-', lw=1.1, color=colour[p], marker='.', ms=2.2,
                     label=p if first else None)
            first = False

    ax1.axhline(0, color='#888', lw=0.6, zorder=0)
    ax1.grid(True, color='#e3e3e3', lw=0.6)
    ax1.set_ylabel(label, fontsize=10)
    if val_lim:
        ax1.set_ylim(*val_lim)
    if invert:
        ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc='best', framealpha=0.9,
               ncol=1 if len(parts) < 6 else 2)
    kinds = sorted({c['kind'] for c in curves})
    src = ('one value per CFD cell' if kinds == ['cell'] else
           'exact CFD node values' if kinds == ['node'] else 'CFD cells + nodes')
    ax1.set_title(f"Surface {label}  —  {cfg.AXIS_LABEL[ax_i]} = {pos_mm:+.1f} mm   "
                  f"({n_pts:,} points, {src}){title_extra}",
                  fontsize=10, loc='left')
    for s in ax1.spines.values():
        s.set_edgecolor('#bbbbbb')
    if cfg.FLIP_HORIZONTAL.get(ax_i, False):
        ax1.set_xlim(h_hi, h_lo)         # mirrored to match the sim exports
    else:
        ax1.set_xlim(h_lo, h_hi)

    if ax2 is not None:
        norm = mc.Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3.0,
                            capstyle='round')
        lc.set_array(vals)
        ax2.add_collection(lc)
        if true_shape:
            ax2.update_datalim([[h_lo, v_lo], [h_hi, v_hi]])
            ax2.autoscale_view(scalex=False)
            ax2.set_aspect('equal', adjustable='datalim')
        else:
            ax2.set_ylim(v_lo, v_hi)
            ax2.text(0.995, 0.03, f"vertical scale x{1/exag:.1f}",
                     transform=ax2.transAxes, ha='right', va='bottom',
                     fontsize=7, color='#777')
        ax2.set_xlabel(h_label, fontsize=10)
        ax2.set_ylabel(v_label, fontsize=10)
        ax2.grid(True, color='#eeeeee', lw=0.6)
        for s in ax2.spines.values():
            s.set_edgecolor('#bbbbbb')
        cbar = fig.colorbar(lc, ax=[ax1, ax2], fraction=0.025, pad=0.015)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        plotting.style_colorbar(cbar, vmin, vmax)
    else:
        ax1.set_xlabel(h_label, fontsize=10)
    return fig


def write_csv(curves, path, axis, scalar_key):
    ax_i = AXIS_IDX.get(axis, axis)
    ha, va = cfg.PLOT_AXES[ax_i]
    hn, vn = cfg.AXIS_LABEL[ha], cfg.AXIS_LABEL[va]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['part', 'loop', 'closed', 'order', 'source',
                    'x_m', 'y_m', 'z_m', f'h_{hn}_m', f'v_{vn}_m', scalar_key,
                    'seg_x0_m', 'seg_y0_m', 'seg_z0_m',
                    'seg_x1_m', 'seg_y1_m', 'seg_z1_m'])
        for c in curves:
            for k in range(len(c['value'])):
                x, y, z = c['xyz'][k]
                if c['kind'] == 'cell':
                    s0, s1 = c['seg_xyz'][k]
                    seg = [*s0, *s1]
                else:
                    seg = [''] * 6
                w.writerow([c['part'], c['loop'], int(c['closed']), k, c['kind'],
                            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                            f"{c['xyz'][k, ha]:.6f}", f"{c['xyz'][k, va]:.6f}",
                            f"{c['value'][k]:.6f}",
                            *[f"{v:.6f}" if v != '' else '' for v in seg]])


def export_sections(parts, axis, positions_mm, scalar_key, output_dir,
                    part_names=None, invert=True, h_lim=None, val_lim=None,
                    show_profile=True, write_csv_files=True,
                    fig_size=(12.0, 8.0), dpi=150):
    """
    Export one Cp-vs-position plot (+ CSV) per slice position.

    Output: <output_dir>/SurfaceCp/<scalar>/<axis>/surfcp_<scalar>_<axis>_<pos>mm.png/.csv
    Returns list of written file paths.
    """
    import matplotlib.pyplot as plt
    axis = axis.upper()
    out = os.path.join(output_dir, 'SurfaceCp', scalar_key, axis)
    os.makedirs(out, exist_ok=True)
    written = []
    print(f"[surface_cp] {scalar_key} on {len(positions_mm)} {axis}-slice(s)"
          f" -> {out}")
    for pos_mm in positions_mm:
        t0 = time.perf_counter()
        curves = section(parts, axis, pos_mm / 1000.0, scalar_key, part_names)
        if not curves:
            print(f"  {axis} = {pos_mm:+.1f} mm: no surface cut here — skipped")
            continue
        stem = f"surfcp_{scalar_key}_{axis}_{pos_mm:+.1f}mm"
        fig = plot_section(curves, axis, pos_mm, scalar_key, invert=invert,
                           h_lim=h_lim, val_lim=val_lim,
                           show_profile=show_profile, fig_size=fig_size, dpi=dpi)
        png = os.path.join(out, stem + '.png')
        fig.savefig(png, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        written.append(png)
        if write_csv_files:
            csv_path = os.path.join(out, stem + '.csv')
            write_csv(curves, csv_path, axis, scalar_key)
            written.append(csv_path)
        n = sum(len(c['value']) for c in curves)
        vals = np.concatenate([c['value'] for c in curves])
        parts_hit = sorted({c['part'] for c in curves})
        print(f"  {axis} = {pos_mm:+8.1f} mm: {n:6,d} pts  "
              f"{scalar_key} [{vals.min():+.3f}, {vals.max():+.3f}]  "
              f"parts={parts_hit}  ({time.perf_counter()-t0:.1f}s)")
    return written