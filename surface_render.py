"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
================================================================================

surface_render.py
=================
Renders CFD surface data on the CFD surface mesh itself — full nodes.

Every output (PNG views, per-part views, 3D export, interactive HTML) uses
the native CFD surface cells with their exact values: no nearest-cell
mapping onto the CAD geometry, no decimation, no cell->point averaging.
Only the parts of the geometry .case are drawn: surface-data blocks are
matched to them by name (plane sections and other non-geometry blocks are
skipped). With no separate surface file, the Cp/Cf stored in the geometry
.case is used.

Functions
---------
prepare_surfaces(geom_case, surface_case, scalars=None)
    -> list of dicts {name, surf, edges}
       Run once per dataset — loads full-node surfaces + edge extraction.

render_views(prepared, scalar_key, cmap, clim,
             views=None, window_size=(1600,1000),
             edge_color='black', edge_width=0.8,
             bg_color='white', feature_angle=25)
    -> dict[view_name, numpy RGB image]

render_delta_views(prepared_a, prepared_b, scalar_key,
                   views=None, window_size=(1600,1000),
                   fixed_range=None, feature_angle=25)
    -> dict[view_name, numpy RGB image]
    Run B is matched onto Run A cell-by-cell (exact when both runs share a
    mesh; nearest cell when the geometry changed between runs).

export_3d(prepared, scalar_keys, output_dir, fmt='vtp')
    Exports one file per part (VTP/VTK) or one merged file (STL/PLY/OBJ).

export_interactive_html(prepared, scalar_key, ...)
    Full-resolution Plotly HTML, one exact colour per CFD cell.

VIEWS
-----
  bottom    looking up from below   (-Z direction)
  top       looking down from above (+Z direction)
  iso_front isometric front-left
  iso_rear  isometric rear-right
  side_L    left side  (-Y direction)
  side_R    right side (+Y direction)
  front     face-on from front (-X direction)
  rear      face-on from rear  (+X direction)
"""

import os
import re
import time
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

import config as cfg

pv.global_theme.allow_empty_mesh = True


# ── Standard view definitions ─────────────────────────────────────
# (name, camera_position, focal_point, view_up)
VIEWS = {
    'bottom':    (( 0.5,  0.0, -6.5), (0.5, 0.0, 0.2), ( 0, -1, 0)),
    'top':       (( 0.5,  0.0,  6.5), (0.5, 0.0, 0.4), ( 0, -1, 0)),
    'iso_front': ((-4.0, -3.5,  2.5), (0.5, 0.0, 0.4), ( 0,  0, 1)),
    'iso_rear':  (( 5.5,  3.5,  2.5), (0.5, 0.0, 0.4), ( 0,  0, 1)),
    'side_L':    (( 0.5, -5.0,  0.5), (0.5, 0.0, 0.5), ( 0,  0, 1)),
    'side_R':    (( 0.5,  5.0,  0.5), (0.5, 0.0, 0.5), ( 0,  0, 1)),
    'front':     ((-5.0,  0.0,  0.5), (0.5, 0.0, 0.5), ( 0,  0, 1)),
    'rear':      (( 6.0,  0.0,  0.5), (0.5, 0.0, 0.5), ( 0,  0, 1)),
}

# Window sizes per view — landscape views get wider windows
VIEW_WINDOW_SIZES = {
    'bottom':    (2000, 1000),
    'top':       (2000, 1000),
    'iso_front': (1600, 1000),
    'iso_rear':  (1600, 1000),
    'side_L':    (2000, 1000),
    'side_R':    (2000, 1000),
    'front':     (1200, 1000),
    'rear':      (1200, 1000),
}

EXPORT_FORMATS = ['vtp', 'vtk', 'stl', 'ply', 'obj']

# FW + FW EPs -> 'Front wing',  RW + RW EPs -> 'Rear wing'
MERGE_GROUPS = {
    'Front wing': ['Front wing', 'Front wing EPs'],
    'Rear wing':  ['Rear wing',  'Rear wing EPs'],
}


# ── Colourmap ─────────────────────────────────────────────────────
def make_cp_cmap(mute=0.0, vmin=-4.0, vmax=1.0):
    """
    STAR-CCM+ Cp palette — same map as the Cp planes (config._make_cp_cmap).
    mute=0.0 -> exact STAR colours (default), mute=1.0 -> all gray.
    """
    base = cfg._make_cp_cmap(vmin=vmin, vmax=vmax)
    if mute <= 0:
        return base
    lut = base(np.linspace(0, 1, base.N))
    gray = np.array([160, 160, 160, 255]) / 255.0
    lut[:, :3] = lut[:, :3] * (1 - mute) + gray[:3] * mute
    return mcolors.ListedColormap(lut, name='cp_surface_muted')


def color_bands():
    """Number of solid colour bands for surface plots (0 = smooth)."""
    try:
        n = int(getattr(cfg, 'SURFACE_COLOR_BANDS', 0) or 0)
    except (TypeError, ValueError):
        return 0
    return n if n >= 2 else 0


def banded_cmap(cmap, n=None):
    """
    cmap -> n solid colours evenly spread over it (like STAR-CCM+'s
    'Number of colors'). The first and last band keep the end colours.
    n=None uses SURFACE_COLOR_BANDS; 0 returns the smooth map unchanged.
    """
    n = color_bands() if n is None else n
    if isinstance(cmap, str):
        import matplotlib
        cmap = matplotlib.colormaps[cmap]
    if not n or getattr(cmap, '_pp_bands', None) == n:
        return cmap
    out = mcolors.ListedColormap(cmap(np.linspace(0, 1, n)),
                                 name=f'{cmap.name}_{n}bands')
    out.set_under(cmap(0.0)); out.set_over(cmap(1.0)); out.set_bad((0, 0, 0, 0))
    out._pp_bands = n
    return out


def _lut_args(cmap):
    """add_mesh colour arguments: banded map + one LUT entry per band."""
    n = color_bands()
    return dict(cmap=banded_cmap(cmap), n_colors=n if n else 256)


def _find_array(mesh, name):
    """('cell'|'point', array_name) for an exact or partial array match."""
    for where, data in (('cell', mesh.cell_data), ('point', mesh.point_data)):
        if name in data:
            return where, name
    for where, data in (('cell', mesh.cell_data), ('point', mesh.point_data)):
        for k in data.keys():
            if name.lower() in k.lower() or k.lower() in name.lower():
                return where, k
    return None, None


def _colorbar(fig, ax, cmap, clim, label=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    import plotting
    sm = plt.cm.ScalarMappable(cmap=banded_cmap(cmap),
                               norm=mc.Normalize(vmin=clim[0], vmax=clim[1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    if label:
        cbar.set_label(label, fontsize=9)
    plotting.style_colorbar(cbar, clim[0], clim[1])
    return cbar


# ── Geometry-only part matching ──────────────────────────────────
PLANE_NAME_RE = re.compile(r'plane|section|slice', re.IGNORECASE)


def _norm_name(n):
    return ' '.join(str(n or '').lower().split())


def _blocks(mb):
    """[(index, name, block)] for every non-empty block of a MultiBlock."""
    if not isinstance(mb, pv.MultiBlock):
        mb = pv.MultiBlock([mb])
    out = []
    for i in range(mb.n_blocks):
        b = mb[i]
        if b is not None and b.n_cells > 0:
            out.append((i, (mb.get_block_name(i) or f"Part {i}").strip(), b))
    return out


def _bounds_match(a, b, tol=0.05):
    """True if two bounding boxes agree within tol x the part size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    size = max(float(np.max(a[1::2] - a[0::2])), 1e-3)
    return bool(np.all(np.abs(a - b) <= tol * size))


def match_to_geometry(geom_parts, surf_parts):
    """
    Pair surface-data blocks with geometry parts.

    1. by part name (case / spacing ignored)
    2. by block position, only if the bounding boxes agree — so plane
       sections or any other non-geometry block are never picked up.

    Returns ({geom_index: surface_block}, [names of skipped surface blocks]).
    """
    by_name = {}
    for i, n, b in surf_parts:
        by_name.setdefault(_norm_name(n), (i, n, b))
    used, result = set(), {}
    for gi, gn, gb in geom_parts:
        hit = by_name.get(_norm_name(gn))
        if hit and hit[0] not in used:
            result[gi] = hit[2]; used.add(hit[0])
    by_index = {i: (i, n, b) for i, n, b in surf_parts}
    for gi, gn, gb in geom_parts:
        if gi in result:
            continue
        cand = by_index.get(gi)
        if (cand and cand[0] not in used and not PLANE_NAME_RE.search(cand[1])
                and _bounds_match(gb.bounds, cand[2].bounds)):
            result[gi] = cand[2]; used.add(cand[0])
    skipped = [n for i, n, b in surf_parts if i not in used]
    return result, skipped


# ── Core preparation ──────────────────────────────────────────────
def prepare_surfaces(geom_case, surface_case=None, scalars=None,
                     feature_angle=25):
    """
    Build the surfaces to draw — ONLY the parts of the geometry .case.

    Surface data comes from the full-node CFD surface blocks that match a
    geometry part (by name, or by position when the bounding boxes agree).
    Blocks that are not geometry parts — plane sections such as X_Planes,
    or anything else in the surface file — are skipped and listed.

    surface_case None (or the geometry .case itself): use the Cp/Cf stored
    in the geometry .case, if it has any.

    Returns
    -------
    list of dicts, one per geometry part:
        name    : str
        surf    : pyvista PolyData   native cells + cell_data
        edges   : pyvista PolyData or None   feature edges
        source  : 'cfd' (surface data file) | 'geometry' (geometry .case)
    """
    import io_utils
    t0 = time.perf_counter()
    print("[surface_render] Loading geometry...")
    io_utils.check_case(geom_case, 'Geometry')
    geom_parts = _blocks(pv.read(geom_case))

    same = (surface_case is not None and
            os.path.abspath(surface_case) == os.path.abspath(geom_case))
    matched = {}
    if surface_case is not None and not same:
        print("[surface_render] Loading surface data (full nodes)...")
        io_utils.check_case(surface_case, 'Surface data')
        surf_parts = _blocks(pv.read(surface_case))
        matched, skipped = match_to_geometry(geom_parts, surf_parts)
        if skipped:
            print(f"  not geometry parts — not drawn: {skipped}")
        if not matched:
            print("  [WARN] none of the parts in the Surface data .case match "
                  "the geometry parts — using the geometry .case's own data. "
                  "(Plane sections such as Raw_Data.case are not surface data; "
                  "leave Surface data empty to use the Cp/Cf in Geometry.case.)")
    else:
        print("[surface_render] Using the surface data stored in the "
              "geometry .case")

    print("[surface_render] Preparing surfaces (geometry parts only, "
          "full nodes, no interpolation)...")
    prepared = []
    for gi, name, g in geom_parts:
        s_block = matched.get(gi)
        src = s_block if s_block is not None else g
        surf = src.extract_surface(algorithm='dataset_surface')
        if scalars:
            for k in list(surf.cell_data.keys()):
                if k not in scalars:
                    surf.cell_data.remove(k)
            for k in list(surf.point_data.keys()):
                if k not in scalars:
                    surf.point_data.remove(k)
        data = [k for k in list(surf.cell_data.keys()) + list(surf.point_data.keys())
                if not k.startswith('vtkOriginal')]
        source = 'cfd' if s_block is not None else 'geometry'

        edges = surf.extract_feature_edges(
            boundary_edges=True, feature_edges=True,
            feature_angle=feature_angle,
            non_manifold_edges=False, manifold_edges=False,
        )
        edges = edges if edges.n_cells > 0 else None

        prepared.append(dict(name=name, surf=surf, edges=edges, source=source))
        tag = ("surface data file, full nodes" if source == 'cfd' else
               "geometry .case data" if data else "no surface data — grey")
        print(f"  {name}: {surf.n_cells:,} cells  ({tag})")

    prepared = _merge_parts(prepared, MERGE_GROUPS)

    print(f"[surface_render] Prepared {len(prepared)} parts "
          f"in {time.perf_counter()-t0:.1f}s")
    return prepared


def _merge_parts(prepared, groups):
    """
    Merge named parts into combined entries.

    groups : dict[new_name, list of part names to merge]
    Parts not listed in any group are kept as-is.
    Parts that are consumed by a merge are removed from the output.
    """
    # Case-insensitive: the UT26 export names the FW endplates 'Front Wing EPs'
    def _norm(n):
        return ' '.join(str(n).lower().split())

    consumed = set()
    merged_entries = []

    for group_name, members in groups.items():
        members_n = {_norm(m) for m in members}
        parts_in_group = [p for p in prepared if _norm(p['name']) in members_n]
        if len(parts_in_group) < 2:
            continue  # nothing to merge

        all_surfs = [p['surf']  for p in parts_in_group]
        all_edges = [p['edges'] for p in parts_in_group if p['edges'] is not None]

        merged_surf  = pv.merge(all_surfs, merge_points=False)
        merged_edges = pv.merge(all_edges) if all_edges else None
        source = ('cfd' if all(p.get('source') == 'cfd' for p in parts_in_group)
                  else 'geometry')

        merged_entries.append(dict(
            name=group_name,
            surf=merged_surf,
            edges=merged_edges,
            source=source,
        ))
        for p in parts_in_group:
            consumed.add(p['name'])
        print(f"  merged: {[p['name'] for p in parts_in_group]} -> '{group_name}'")

    result = []
    inserted = set()
    for p in prepared:
        if p['name'] not in consumed:
            result.append(p)
        else:
            for group_name, members in groups.items():
                if (_norm(p['name']) in {_norm(m) for m in members}
                        and group_name not in inserted):
                    merged = next(e for e in merged_entries
                                  if e['name'] == group_name)
                    result.append(merged)
                    inserted.add(group_name)
                    break

    return result


def _add_scalar_mesh(pl, surf, scalar_key, cmap, clim, part_name='?'):
    """Add one part coloured by its own (cell or node) values."""
    where, arr = _find_array(surf, scalar_key)
    if where is not None:
        pl.add_mesh(surf, scalars=arr, preference=where,
                    clim=list(clim), **_lut_args(cmap),
                    smooth_shading=False, interpolate_before_map=False,
                    show_scalar_bar=False, lighting=False, show_edges=False)
        return True
    print(f"  [WARN] '{scalar_key}' not found on '{part_name}'"
          f" — check Surface data .case path is set in the GUI")
    pl.add_mesh(surf, color='#aaaaaa',
                smooth_shading=True, lighting=True,
                ambient=0.4, show_edges=False)
    return False


def _set_projection(pl):
    """Parallel (orthographic) or perspective, from config."""
    if getattr(cfg, 'SURFACE_PARALLEL_PROJECTION', True):
        pl.enable_parallel_projection()
    else:
        pl.disable_parallel_projection()


def _fit_camera(pl, parts, pos, focal, up, wsize, target_fill=0.85):
    """Point the camera along the view direction and zoom to fill."""
    import vtk as _vtk
    _set_projection(pl)
    pl.view_vector(vector=tuple(p - f for p, f in zip(pos, focal)), viewup=up)
    pl.reset_camera()
    all_bounds = np.array([p['surf'].bounds for p in parts])
    wmin = all_bounds[:, [0, 2, 4]].min(axis=0)
    wmax = all_bounds[:, [1, 3, 5]].max(axis=0)
    corners = np.array([[x, y, z] for x in [wmin[0], wmax[0]]
                                   for y in [wmin[1], wmax[1]]
                                   for z in [wmin[2], wmax[2]]])
    coord = _vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    spts = []
    for pt in corners:
        coord.SetValue(*pt)
        spts.append(coord.GetComputedDisplayValue(pl.renderer)[:2])
    spts = np.array(spts, dtype=float)
    W, H = wsize
    fill = max((spts[:, 0].max() - spts[:, 0].min()) / W,
               (spts[:, 1].max() - spts[:, 1].min()) / H)
    if fill > 0:
        pl.camera.zoom(target_fill / fill)


def _render_one_view(parts, scalar_key, cmap, clim,
                     vname, window_size, edge_color, edge_width,
                     bg_color, zoom=False, zoom_factor=1.8):
    """
    Render a single view of the given parts list.
    If zoom=True, the parts fill ~85% of the window (aspect-corrected).
    Returns RGB numpy array.
    """
    pos, focal, up = VIEWS[vname]
    wsize = VIEW_WINDOW_SIZES.get(vname, window_size) if window_size == (1600, 1000) \
            else window_size
    pl = pv.Plotter(off_screen=True, window_size=list(wsize))
    pl.set_background(bg_color)

    for part in parts:
        _add_scalar_mesh(pl, part['surf'], scalar_key, cmap, clim,
                         part.get('name', '?'))
        if part['edges'] is not None:
            pl.add_mesh(part['edges'], color=edge_color,
                        line_width=edge_width, lighting=False)

    if zoom:
        _fit_camera(pl, parts, pos, focal, up, wsize)
    else:
        _set_projection(pl)
        pl.camera.position    = pos
        pl.camera.focal_point = focal
        pl.camera.view_up     = up
        pl.reset_camera_clipping_range()

    img = pl.screenshot(None, return_img=True)
    pl.close()
    return img


def _save_view_png(img, cmap, clim, title, out_path, bg_color, label=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.imshow(img); ax.axis('off')
    _colorbar(fig, ax, cmap, clim, label)
    ax.set_title(title, fontsize=11,
                 color='black' if bg_color == 'white' else 'white')
    fig.savefig(out_path, dpi=150, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)


# ── Single-run render ─────────────────────────────────────────────
def render_views(prepared, scalar_key, cmap=None, clim=(-4.0, 1.0),
                 views=None, window_size=(1600, 1000),
                 edge_color='black', edge_width=0.8,
                 bg_color='white',
                 output_dir=None, file_stem='surface', label=None):
    """
    Render a set of views for a single run.
    Saves one PNG per view if output_dir is provided.

    Parameters
    ----------
    prepared    : list from prepare_surfaces()
    scalar_key  : str   array name to colour by
    cmap        : matplotlib colormap or None (uses make_cp_cmap())
    clim        : (vmin, vmax)
    views       : list of view name strings or None (all views)
    window_size : (W, H)
    output_dir  : str or None   if set, saves individual PNGs here
    file_stem   : str           prefix for output filenames

    Returns
    -------
    dict[view_name, np.ndarray (H, W, 3)]
    """
    if cmap is None:
        cmap = make_cp_cmap(vmin=clim[0], vmax=clim[1])
    views_to_render = views or list(VIEWS.keys())

    png_dir = os.path.join(output_dir, 'PNG') if output_dir else None
    npz_dir = os.path.join(output_dir, 'NPZ') if output_dir else None
    if png_dir: os.makedirs(png_dir, exist_ok=True)
    if npz_dir: os.makedirs(npz_dir, exist_ok=True)

    imgs = {}
    for vname in views_to_render:
        if vname not in VIEWS:
            print(f"[surface_render] Unknown view '{vname}', skipping")
            continue
        t0  = time.perf_counter()
        img = _render_one_view(prepared, scalar_key, cmap, clim,
                               vname, window_size, edge_color, edge_width,
                               bg_color, zoom=True)
        imgs[vname] = img

        if output_dir:
            out_path = os.path.join(png_dir, f'{file_stem}_{vname}.png')
            _save_view_png(img, cmap, clim,
                           f'{file_stem}  --  {vname.replace("_"," ").title()}',
                           out_path, bg_color, label)
            print(f"  saved: PNG/{os.path.basename(out_path)}")

            npz_path = os.path.join(npz_dir, f'{file_stem}_{vname}.npz')
            np.savez_compressed(
                npz_path,
                image      = img.astype(np.uint8),
                view       = np.bytes_(vname),
                scalar_key = np.bytes_(scalar_key),
                clim       = np.array(clim, dtype=np.float32),
                file_stem  = np.bytes_(file_stem),
            )
            print(f"  saved: NPZ/{os.path.basename(npz_path)}")

        print(f"  {vname}: {time.perf_counter()-t0:.1f}s")

    return imgs


def render_part_views(prepared, part_names, scalar_key,
                      cmap=None, clim=(-4.0, 1.0),
                      views=None, window_size=(1600, 1000),
                      edge_color='black', edge_width=0.8,
                      bg_color='white',
                      output_dir=None, file_stem='surface',
                      zoom_factor=1.4, label=None):
    """
    Render individual views for each requested part, auto-zoomed to fit.
    Saves one PNG per (part, view) combination.

    Returns
    -------
    dict[(part_name, view_name), np.ndarray]
    """
    if cmap is None:
        cmap = make_cp_cmap(vmin=clim[0], vmax=clim[1])
    views_to_render = views or list(VIEWS.keys())

    png_dir = os.path.join(output_dir, 'PNG') if output_dir else None
    npz_dir = os.path.join(output_dir, 'NPZ') if output_dir else None
    if png_dir: os.makedirs(png_dir, exist_ok=True)
    if npz_dir: os.makedirs(npz_dir, exist_ok=True)

    part_lookup = {p['name']: p for p in prepared}

    results = {}
    for pname in part_names:
        if pname not in part_lookup:
            print(f"[surface_render] Part '{pname}' not found, skipping")
            continue

        part = part_lookup[pname]
        safe_name = pname.replace(' ', '_').replace('/', '_')

        for vname in views_to_render:
            if vname not in VIEWS:
                continue
            t0  = time.perf_counter()
            img = _render_one_view([part], scalar_key, cmap, clim,
                                   vname, window_size, edge_color, edge_width,
                                   bg_color, zoom=True, zoom_factor=zoom_factor)
            results[(pname, vname)] = img

            if output_dir:
                out_path = os.path.join(png_dir,
                                        f'{file_stem}_{safe_name}_{vname}.png')
                _save_view_png(img, cmap, clim,
                               f'{pname}  --  {vname.replace("_"," ").title()}',
                               out_path, bg_color, label)
                print(f"  saved: PNG/{os.path.basename(out_path)}")

                npz_path = os.path.join(npz_dir,
                                        f'{file_stem}_{safe_name}_{vname}.npz')
                np.savez_compressed(
                    npz_path,
                    image      = img.astype(np.uint8),
                    view       = np.bytes_(vname),
                    part_name  = np.bytes_(pname),
                    scalar_key = np.bytes_(scalar_key),
                    clim       = np.array(clim, dtype=np.float32),
                    file_stem  = np.bytes_(file_stem),
                )
                print(f"  saved: NPZ/{os.path.basename(npz_path)}")

            print(f"  {pname} / {vname}: {time.perf_counter()-t0:.1f}s")

    return results


# ── Delta render ──────────────────────────────────────────────────
def render_delta_views(prepared_a, prepared_b, scalar_key,
                       label_a='Run A', label_b='Run B',
                       views=None, window_size=(1600, 1000),
                       fixed_range=None, feature_angle=25,
                       bg_color='white'):
    """
    Render delta views (B - A) with three-colour geometry overlay.

    Geometry colours:
      Gray    (#404040) — solid in both
      Magenta (#CC00CC) — new in B
      Brown   (#8B4513) — removed (only in A)

    Cells are matched A -> B by cell centre. When both runs share the same
    surface mesh this is an exact 1:1 match; when the geometry changed it is
    the nearest CFD cell of run B.

    Returns
    -------
    (dict[view_name, np.ndarray], clim)
    """
    views_to_render = views or list(VIEWS.keys())

    parts_a = {p['name']: p for p in prepared_a}
    parts_b = {p['name']: p for p in prepared_b}
    all_names = sorted(set(parts_a.keys()) | set(parts_b.keys()))

    delta_parts  = []
    GRAY    = '#404040'
    MAGENTA = '#CC00CC'
    BROWN   = '#8B4513'

    for name in all_names:
        pa = parts_a.get(name)
        pb = parts_b.get(name)

        if pa is not None and pb is not None:
            geom_color = GRAY
            wa, ka = _find_array(pa['surf'], scalar_key)
            wb, kb = _find_array(pb['surf'], scalar_key)
            surf_d = pa['surf'].copy()
            edges  = pa['edges']
            if wa == 'cell' and wb == 'cell':
                va = pa['surf'].cell_data[ka]
                vb = pb['surf'].cell_data[kb]
                cc_a = pa['surf'].cell_centers().points
                cc_b = pb['surf'].cell_centers().points
                if len(cc_a) == len(cc_b) and np.allclose(cc_a, cc_b, atol=1e-9):
                    idx = np.arange(len(cc_a))              # same mesh: exact
                else:
                    _, idx = cKDTree(cc_b).query(cc_a, k=1, workers=-1)
                surf_d.cell_data['delta'] = vb[idx] - va
            elif wa == 'point' and wb == 'point':
                va = pa['surf'].point_data[ka]
                vb = pb['surf'].point_data[kb]
                pa_pts, pb_pts = pa['surf'].points, pb['surf'].points
                if len(pa_pts) == len(pb_pts) and np.allclose(pa_pts, pb_pts, atol=1e-9):
                    idx = np.arange(len(pa_pts))
                else:
                    _, idx = cKDTree(pb_pts).query(pa_pts, k=1, workers=-1)
                surf_d.point_data['delta'] = vb[idx] - va

        elif pb is not None:
            geom_color = MAGENTA
            surf_d = pb['surf'].copy()
            edges  = pb['edges']
        else:
            geom_color = BROWN
            surf_d = pa['surf'].copy()
            edges  = pa['edges']

        where = ('cell' if 'delta' in surf_d.cell_data else
                 'point' if 'delta' in surf_d.point_data else None)
        delta_parts.append(dict(
            surf=surf_d, edges=edges,
            geom_color=geom_color,
            has_delta=where is not None, where=where,
        ))

    if fixed_range is None:
        all_deltas = []
        for p in delta_parts:
            if p['has_delta']:
                data = p['surf'].cell_data if p['where'] == 'cell' else p['surf'].point_data
                all_deltas.append(np.asarray(data['delta']))
        if all_deltas:
            dmax = float(np.nanmax(np.abs(np.concatenate(all_deltas))))
            clim = (-dmax, dmax)
        else:
            clim = (-1.0, 1.0)
    else:
        clim = (-fixed_range, fixed_range)

    imgs = {}
    for vname in views_to_render:
        if vname not in VIEWS:
            continue
        pos, focal, up = VIEWS[vname]
        t0 = time.perf_counter()

        wsize = VIEW_WINDOW_SIZES.get(vname, window_size) if window_size == (1600, 1000) \
                else window_size
        pl = pv.Plotter(off_screen=True, window_size=list(wsize))
        pl.set_background(bg_color)

        for part in delta_parts:
            if part['has_delta']:
                pl.add_mesh(part['surf'], scalars='delta',
                            preference=part['where'],
                            clim=list(clim), **_lut_args('RdBu_r'),
                            smooth_shading=False,
                            interpolate_before_map=False,
                            show_scalar_bar=False,
                            lighting=False, show_edges=False)
            else:
                pl.add_mesh(part['surf'], color=part['geom_color'],
                            opacity=0.8, lighting=True,
                            ambient=0.4, show_edges=False)

            if part['edges'] is not None:
                pl.add_mesh(part['edges'], color='black',
                            line_width=0.8, lighting=False)

        _fit_camera(pl, delta_parts, pos, focal, up, wsize)
        imgs[vname] = pl.screenshot(None, return_img=True)
        pl.close()
        print(f"  {vname}: {time.perf_counter()-t0:.1f}s")

    return imgs, clim


# ── 3D export ─────────────────────────────────────────────────────
def export_3d(prepared, scalar_keys=None, output_dir='.',
              fmt='vtp', merged=False):
    """
    Export prepared (full-node) surface meshes to a 3D file format.

    Parameters
    ----------
    prepared    : list from prepare_surfaces()
    scalar_keys : list of str or None   scalars to include in export
                  (only relevant for VTP/VTK — ignored for STL/PLY/OBJ)
    output_dir  : str
    fmt         : str   one of 'vtp', 'vtk', 'stl', 'ply', 'obj'
    merged      : bool  if True, merge all parts into one file

    Returns
    -------
    list of exported file paths
    """
    fmt = fmt.lower().lstrip('.')
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"fmt must be one of {EXPORT_FORMATS}")

    os.makedirs(output_dir, exist_ok=True)
    scalar_supports = fmt in ('vtp', 'vtk')
    exported = []

    def _strip(s):
        for key in list(s.cell_data.keys()):
            s.cell_data.remove(key)
        for key in list(s.point_data.keys()):
            s.point_data.remove(key)

    if merged:
        all_surfs = []
        for part in prepared:
            s = part['surf'].copy()
            if not scalar_supports:
                _strip(s)
            all_surfs.append(s)

        merged_mesh = pv.merge(all_surfs, merge_points=False)
        out_path = os.path.join(output_dir, f'surface_merged.{fmt}')
        merged_mesh.save(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[surface_render] Exported merged -> {out_path}  ({size_mb:.1f}MB)")
        exported.append(out_path)

    else:
        for part in prepared:
            s    = part['surf'].copy()
            name = part['name'].replace(' ', '_').replace('/', '_')

            if not scalar_supports:
                _strip(s)
            elif scalar_keys is not None:
                for key in list(s.cell_data.keys()):
                    if key not in scalar_keys and key != 'vtkOriginalCellIds':
                        s.cell_data.remove(key)
                for key in list(s.point_data.keys()):
                    if key not in scalar_keys:
                        s.point_data.remove(key)

            out_path = os.path.join(output_dir, f'{name}.{fmt}')
            s.save(out_path)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  {name}.{fmt}  ({size_mb:.1f}MB)")
            exported.append(out_path)

    return exported


# ── Interactive HTML export ───────────────────────────────────────


# ── Interactive HTML — compact, lossless ─────────────────────────────────────
#
# Plotly's own HTML writer stores one entry per *triangle*: three int32
# indices plus a float per triangle, uncompressed. STAR surfaces are mostly
# hexagons (≈4 triangles each), so a full car came out at ~400 MB.
#
# Instead each part is stored as its CFD polygons (one value per cell),
# delta + byte-shuffle encoded and deflated, then unpacked by the browser
# when the page opens (DecompressionStream). Every node, every cell and every
# value is kept exactly (float32, bit-for-bit) — the file is just ~5x smaller.
# The browser splits each polygon into a triangle fan; the few concave
# polygons where a plain fan would fold are rotated or ear-clipped here first.

HTML_ZLIB_LEVEL = 6


def _shuffle(a):
    """Byte-shuffle: byte k of every element stored together (helps deflate)."""
    a = np.ascontiguousarray(a)
    b = a.view(np.uint8).reshape(-1, a.dtype.itemsize)
    return np.ascontiguousarray(b.T).tobytes()


def _delta_zz(a):
    """int32 values -> zigzag-encoded deltas (uint32). Exactly reversible."""
    d = np.diff(np.asarray(a, dtype=np.int64), prepend=0)
    d = (d + 2**31) % 2**32 - 2**31                         # wrap to int32
    return ((d << 1) ^ (d >> 31)).astype(np.uint32)


def _fan(off, conn):
    """Fan triangles (v0, vk, vk+1) for every polygon + owning polygon index."""
    sizes = np.diff(off)
    ntri  = sizes - 2
    cell  = np.repeat(np.arange(len(sizes)), ntri)
    k     = np.arange(len(cell)) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    st    = off[:-1][cell]
    return np.stack([conn[st], conn[st + k + 1], conn[st + k + 2]], 1), cell


def _newell(P):
    """Area-weighted normal of a polygon (rows of P in order)."""
    Q = np.roll(P, -1, axis=0)
    return np.array([np.sum((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])),
                     np.sum((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])),
                     np.sum((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1]))])


def _fan_ok(P, n):
    """True if the fan from vertex 0 has no triangle facing against n."""
    a = P[1:-1] - P[0]
    b = P[2:] - P[0]
    t = np.cross(a, b)
    return bool(np.all(t @ n >= -1e-6 * np.linalg.norm(t, axis=1)
                       * np.linalg.norm(n)))


def _ear_clip(P, ids, n):
    """Ear-clip one polygon (projected on its own plane) -> list of triangles."""
    ax = int(np.argmax(np.abs(n)))
    uv = np.delete(P, ax, axis=1)
    if n[ax] < 0:
        uv[:, 0] = -uv[:, 0]                       # make the polygon CCW
    idx  = list(range(len(ids)))
    tris = []

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    guard = 0
    while len(idx) > 3 and guard < 4 * len(ids) ** 2:
        guard += 1
        for m in range(len(idx)):
            i0, i1, i2 = idx[m - 1], idx[m], idx[(m + 1) % len(idx)]
            A, B, C = uv[i0], uv[i1], uv[i2]
            if cross(A, B, C) <= 0:
                continue                           # reflex corner
            if any(cross(A, B, uv[q]) >= 0 and cross(B, C, uv[q]) >= 0
                   and cross(C, A, uv[q]) >= 0
                   for q in idx if q not in (i0, i1, i2)):
                continue                           # another vertex inside
            tris.append((ids[i0], ids[i1], ids[i2]))
            idx.pop(m)
            break
        else:
            break                                  # degenerate: fan the rest
    for q in range(1, len(idx) - 1):
        tris.append((ids[idx[0]], ids[idx[q]], ids[idx[q + 1]]))
    return tris


def _html_polygons(surf, pts64):
    """
    Polygons of a surface, ready for a browser-side fan split.
    Returns (sizes, conn, src, n_rot, n_clip): src[i] is the original
    polygon index each output polygon takes its value from.
    """
    from vtk.util.numpy_support import vtk_to_numpy
    off  = vtk_to_numpy(surf.GetPolys().GetOffsetsArray()).astype(np.int64)
    conn = vtk_to_numpy(surf.GetPolys().GetConnectivityArray()).astype(np.int64)
    sizes = np.diff(off)
    src   = np.arange(len(sizes))

    keep = sizes >= 3                              # lines/points draw nothing
    if not keep.all():
        conn  = conn[np.repeat(keep, sizes)]
        sizes, src = sizes[keep], src[keep]
        off = np.r_[0, np.cumsum(sizes)]

    F, cell = _fan(off, conn)
    t  = np.cross(pts64[F[:, 1]] - pts64[F[:, 0]], pts64[F[:, 2]] - pts64[F[:, 0]])
    nc = np.stack([np.bincount(cell, t[:, c], len(sizes)) for c in range(3)], 1)
    d  = np.einsum('ij,ij->i', t, nc[cell])
    bad_t = d < -1e-6 * np.linalg.norm(t, axis=1) * np.linalg.norm(nc[cell], axis=1)
    bad = np.unique(cell[bad_t])

    n_rot = 0
    clip_cells, clip_tris = [], []
    for c in bad:
        a, b = off[c], off[c + 1]
        ids = conn[a:b].copy()
        P   = pts64[ids]
        n   = _newell(P)
        for r in range(1, len(ids)):
            if _fan_ok(np.roll(P, -r, axis=0), n):
                conn[a:b] = np.roll(ids, -r)       # same polygon, new start
                n_rot += 1
                break
        else:
            clip_cells.append(c)
            clip_tris.append(_ear_clip(P, ids, n))

    if clip_cells:
        drop = np.zeros(len(sizes), bool)
        drop[clip_cells] = True
        conn = conn[~np.repeat(drop, sizes)]
        extra_src = [src[c] for c, tl in zip(clip_cells, clip_tris) for _ in tl]
        extra_con = [v for tl in clip_tris for tri in tl for v in tri]
        sizes = np.r_[sizes[~drop], np.full(len(extra_src), 3)]
        src   = np.r_[src[~drop], np.asarray(extra_src, np.int64)]
        conn  = np.r_[conn, np.asarray(extra_con, np.int64)]
    return sizes, conn, src, n_rot, len(clip_cells)


def _encode_part(surf, where, arr):
    """Pack one part -> (base64 text, meta dict, number of triangles)."""
    import base64
    import zlib
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    if surf.n_strips:
        surf = surf.triangulate()

    pts   = np.ascontiguousarray(surf.points, dtype=np.float32)
    sizes, conn, src, n_rot, n_clip = _html_polygons(surf, pts.astype(np.float64))

    if where == 'cell':
        base = surf.n_verts + surf.n_lines        # polys follow verts + lines
        vals = np.asarray(surf.cell_data[arr], np.float32)[base:][src]
        mode = 'cell'
    else:
        vals = np.asarray(surf.point_data[arr], np.float32)
        mode = 'vertex'

    sw = 1 if sizes.max(initial=3) <= 255 else 2
    streams = [_shuffle(_delta_zz(pts[:, c].copy().view(np.int32)))
               for c in range(3)]
    streams.append(sizes.astype(np.uint8).tobytes() if sw == 1
                   else _shuffle(sizes.astype(np.uint16)))
    streams.append(_shuffle(_delta_zz(conn)))
    streams.append(_shuffle(np.ascontiguousarray(vals, np.float32)))

    blob = zlib.compress(b''.join(streams), HTML_ZLIB_LEVEL)
    meta = dict(n_pts=int(len(pts)), n_poly=int(len(sizes)),
                n_conn=int(len(conn)), n_tri=int((sizes - 2).sum()),
                sw=sw, mode=mode, lens=[len(s) for s in streams],
                n_rot=n_rot, n_clip=n_clip)
    return base64.b64encode(blob).decode('ascii'), meta


_HTML_JS = r"""
(async function () {
  const status = document.getElementById('pp-status');
  try {
    if (typeof DecompressionStream === 'undefined')
      throw new Error('this browser cannot unpack the data');
    const META = JSON.parse(document.getElementById('pp-meta').textContent);
    const FIG  = JSON.parse(document.getElementById('pp-fig').textContent);

    const b64 = s => {
      if (Uint8Array.fromBase64) return Uint8Array.fromBase64(s);
      const bin = atob(s), u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      return u;
    };
    const inflate = async u8 => new Uint8Array(await new Response(
      new Blob([u8]).stream().pipeThrough(new DecompressionStream('deflate'))
    ).arrayBuffer());
    const unshuffle = (src, off, n, w) => {
      const out = new Uint8Array(n * w);
      for (let k = 0; k < w; k++) {
        const base = off + k * n;
        for (let i = 0; i < n; i++) out[i * w + k] = src[base + i];
      }
      return out.buffer;
    };
    const undelta = buf => {
      const u = new Uint32Array(buf), v = new Int32Array(buf);
      let acc = 0;
      for (let i = 0; i < u.length; i++) {
        const x = u[i];
        acc = (acc + ((x >>> 1) ^ -(x & 1))) | 0;
        v[i] = acc;
      }
      return v;
    };

    for (let q = 0; q < META.length; q++) {
      const p = META[q];
      status.textContent = `Unpacking ${p.name} (${q + 1}/${META.length}) …`;
      await new Promise(r => setTimeout(r, 0));
      const el  = document.getElementById('pp-d' + q);
      const raw = await inflate(b64(el.textContent.trim()));
      el.textContent = '';
      let o = 0;
      const xyz = [];
      for (let c = 0; c < 3; c++) {
        xyz.push(new Float32Array(undelta(unshuffle(raw, o, p.n_pts, 4)).buffer));
        o += p.lens[c];
      }
      const sizes = p.sw === 1 ? raw.slice(o, o + p.n_poly)
                               : new Uint16Array(unshuffle(raw, o, p.n_poly, 2));
      o += p.lens[3];
      const conn = undelta(unshuffle(raw, o, p.n_conn, 4));
      o += p.lens[4];
      const cell = p.mode === 'cell';
      const vals = new Float32Array(unshuffle(raw, o, cell ? p.n_poly : p.n_pts, 4));

      const I = new Int32Array(p.n_tri), J = new Int32Array(p.n_tri),
            K = new Int32Array(p.n_tri);
      const inten = cell ? new Float32Array(p.n_tri) : vals;
      let t = 0, c0 = 0;
      for (let a = 0; a < p.n_poly; a++) {
        const n = sizes[a], v0 = conn[c0];
        for (let m = 1; m < n - 1; m++) {
          I[t] = v0; J[t] = conn[c0 + m]; K[t] = conn[c0 + m + 1];
          if (cell) inten[t] = vals[a];
          t++;
        }
        c0 += n;
      }
      Object.assign(FIG.data[q], {x: xyz[0], y: xyz[1], z: xyz[2],
                                  i: I, j: J, k: K, intensity: inten});
    }
    status.textContent = 'Drawing …';
    await new Promise(r => setTimeout(r, 0));
    await Plotly.newPlot('pp-plot', FIG.data, FIG.layout, {responsive: true});
    status.remove();
    window.PP_READY = true;
  } catch (e) {
    status.textContent = 'Could not open this file: ' + e.message +
      ' — use an up-to-date Chrome, Edge, Firefox or Safari (internet needed for plotly.js).';
    status.style.color = '#b00';
    window.PP_ERROR = String(e);
  }
})();
"""


def export_interactive_html(prepared, scalar_key, cmap=None,
                            clim=(-4.0, 1.0), label='Cp',
                            output_path='surface.html'):
    """
    Export an interactive Plotly 3D HTML file at full resolution.

    Every CFD surface cell is kept (no decimation) and coloured by its own
    value (intensitymode='cell' — no cell->point averaging). The mesh is
    stored compactly and losslessly (polygons + one value per cell, packed
    and deflated) and unpacked by the browser when the page opens, so the
    file is ~5x smaller than Plotly's default writer with identical data.

    Returns
    -------
    str   path to saved HTML file
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")
    import json

    if cmap is None:
        cmap = make_cp_cmap(vmin=clim[0], vmax=clim[1])

    vmin, vmax = clim
    def _rgb(c):
        return f'rgb({int(round(c[0]*255))},{int(round(c[1]*255))},{int(round(c[2]*255))})'
    n_bands = color_bands()
    if n_bands:                       # hard-edged bands: two stops per band
        cols = banded_cmap(cmap, n_bands)(np.linspace(0, 1, n_bands))
        colorscale = []
        for k, c in enumerate(cols):
            colorscale += [[k / n_bands, _rgb(c)], [(k + 1) / n_bands, _rgb(c)]]
    else:
        positions  = np.linspace(0, 1, 256)
        colorscale = [[float(p), _rgb(c)] for p, c in zip(positions, cmap(positions))]

    print("[surface_render] Building interactive HTML (full nodes, every cell, "
          "packed losslessly)...")
    t_all  = time.perf_counter()
    traces, metas, blobs = [], [], []
    n_total = 0

    for part in prepared:
        surf = part['surf']
        name = part['name']
        where, arr = _find_array(surf, scalar_key)
        if where is None:
            continue

        t0 = time.perf_counter()
        text, meta = _encode_part(surf, where, arr)
        meta['name'] = name
        first = not traces
        traces.append(go.Mesh3d(
            intensitymode=meta['mode'],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            showscale=first,
            colorbar=dict(
                title=dict(text=label, side='right'),
                thickness=18, len=0.6,
                tickfont=dict(size=11),
            ) if first else None,
            name=name,
            lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
            flatshading=True,
            hovertemplate=(f'<b>{name}</b><br>'
                           f'{label}: %{{intensity:.3f}}<extra></extra>'),
            showlegend=True,
        ))
        metas.append(meta)
        blobs.append(text)
        n_total += meta['n_tri']
        fix = ''
        if meta['n_rot'] or meta['n_clip']:
            fix = f"  [{meta['n_rot']} concave cells re-fanned, {meta['n_clip']} ear-clipped]"
        print(f"  {name}: {surf.n_cells:,} cells -> {meta['n_tri']:,} tris, "
              f"{len(text)/1e6:.1f}MB  ({time.perf_counter()-t0:.1f}s){fix}")

    if not traces:
        raise ValueError(f"No part carries '{scalar_key}' — nothing to export.")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f'UT26 — Surface {label}  (full mesh)  |  '
                  f'drag to rotate  ·  scroll to zoom  ·  '
                  f'click legend to toggle parts'),
            font=dict(size=13)),
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor='#f8f8f8',
                       gridcolor='#ddd', showbackground=True),
            yaxis=dict(title='Y (m)', backgroundcolor='#f8f8f8',
                       gridcolor='#ddd', showbackground=True),
            zaxis=dict(title='Z (m)', backgroundcolor='#f8f8f8',
                       gridcolor='#ddd', showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=-1.2, y=-1.2, z=0.7),
                        projection=dict(type='orthographic'
                                        if getattr(cfg, 'SURFACE_PARALLEL_PROJECTION', True)
                                        else 'perspective')),
        ),
        paper_bgcolor='white',
        legend=dict(x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1,
                    font=dict(size=11)),
        width=1400, height=900,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    def _safe(s):                     # JSON inside <script>: never close the tag
        return s.replace('</', '<\\/')

    try:
        from plotly.io._utils import plotly_cdn_url
        cdn = plotly_cdn_url()
    except Exception:
        from plotly.offline import get_plotlyjs_version
        cdn = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"

    title = f"Surface {label} — interactive"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
                 f'<title>{title}</title>\n'
                 f'<script src="{cdn}" charset="utf-8"></script>\n'
                 '<style>body{margin:0;font-family:sans-serif;background:#fff}'
                 '#pp-status{padding:12px 16px;color:#444;font-size:14px}</style>\n'
                 '</head><body>\n'
                 '<div id="pp-status">Unpacking surface data …</div>\n'
                 '<div id="pp-plot"></div>\n')
        fh.write('<script type="application/json" id="pp-meta">'
                 + _safe(json.dumps(metas)) + '</script>\n')
        fh.write('<script type="application/json" id="pp-fig">'
                 + _safe(pio.to_json(fig, validate=False)) + '</script>\n')
        for q, text in enumerate(blobs):
            fh.write(f'<script type="application/octet-stream" id="pp-d{q}">')
            fh.write(text)
            fh.write('</script>\n')
        fh.write('<script>' + _HTML_JS + '</script>\n</body></html>\n')

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[surface_render] Saved interactive HTML -> {output_path}  "
          f"({size_mb:.1f}MB, {n_total:,} triangles, "
          f"{time.perf_counter()-t_all:.1f}s)")
    if size_mb > 200:
        print("  [NOTE] Large file — very big HTML files can be slow to open "
              "in a browser. Chrome/Edge handle them best.")
    return output_path