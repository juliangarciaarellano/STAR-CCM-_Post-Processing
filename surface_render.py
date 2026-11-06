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
The geometry .case only supplies part names and a grey fallback for parts
that have no surface data.

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
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mc.Normalize(vmin=clim[0], vmax=clim[1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    if label:
        cbar.set_label(label, fontsize=9)
    plotting.style_colorbar(cbar, clim[0], clim[1])
    return cbar


# ── Core preparation ──────────────────────────────────────────────
def prepare_surfaces(geom_case, surface_case=None, scalars=None,
                     feature_angle=25):
    """
    Load the CFD surface mesh (full nodes) and the geometry part names.

    Parameters
    ----------
    geom_case    : str   path to geometry .case file (part names + fallback)
    surface_case : str or None   path to surface CFD .case file
                   If None, renders geometry only (no scalar colouring).
    scalars      : list of str or None   which arrays to keep
                   defaults to all available in surface_case
    feature_angle: float   angle threshold for feature edge extraction

    Returns
    -------
    list of dicts, one per part:
        name    : str
        surf    : pyvista PolyData   CFD surface, native cells + cell_data
        edges   : pyvista PolyData or None   feature edges
        source  : 'cfd' (full-node CFD surface) or 'geometry' (no data)
    """
    import io_utils
    t0 = time.perf_counter()
    print("[surface_render] Loading geometry...")
    io_utils.check_case(geom_case, 'Geometry')
    geom = pv.read(geom_case)

    surf_mesh = None
    scalars_to_use = []
    if surface_case is not None:
        print("[surface_render] Loading surface data (full nodes)...")
        io_utils.check_case(surface_case, 'Surface data')
        surf_mesh = pv.read(surface_case)

        available = set()
        for i in range(surf_mesh.n_blocks):
            b = surf_mesh[i]
            if b is not None and b.n_cells > 0:
                available.update(b.cell_data.keys())
                available.update(b.point_data.keys())
        scalars_to_use = scalars if scalars else sorted(available)
        print(f"[surface_render] Scalars: {scalars_to_use}")

    print("[surface_render] Preparing surfaces (full nodes, no interpolation)...")
    prepared = []
    n_blocks = geom.n_blocks
    if surf_mesh is not None:
        n_blocks = max(n_blocks, surf_mesh.n_blocks)

    for i in range(n_blocks):
        g = geom[i] if i < geom.n_blocks else None
        s = (surf_mesh[i] if surf_mesh is not None and i < surf_mesh.n_blocks
             else None)
        has_g = g is not None and g.n_cells > 0
        has_s = s is not None and s.n_cells > 0
        if not has_g and not has_s:
            continue
        name = (geom.get_block_name(i) if has_g else
                surf_mesh.get_block_name(i)) or f"Part {i}"
        name = name.strip()

        if has_s:
            surf = s.extract_surface(algorithm='dataset_surface')
            # keep only the requested arrays (plus nothing interpolated)
            for k in list(surf.cell_data.keys()):
                if k not in scalars_to_use:
                    surf.cell_data.remove(k)
            for k in list(surf.point_data.keys()):
                if k not in scalars_to_use:
                    surf.point_data.remove(k)
            source = 'cfd'
        else:
            surf = g.extract_surface(algorithm='dataset_surface')
            source = 'geometry'

        edges = surf.extract_feature_edges(
            boundary_edges=True, feature_edges=True,
            feature_angle=feature_angle,
            non_manifold_edges=False, manifold_edges=False,
        )
        edges = edges if edges.n_cells > 0 else None

        prepared.append(dict(name=name, surf=surf, edges=edges, source=source))
        tag = "full nodes" if source == 'cfd' else "geometry only, no data"
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
                    cmap=cmap, clim=list(clim),
                    smooth_shading=False, interpolate_before_map=False,
                    show_scalar_bar=False, lighting=False, show_edges=False)
        return True
    print(f"  [WARN] '{scalar_key}' not found on '{part_name}'"
          f" — check Surface data .case path is set in the GUI")
    pl.add_mesh(surf, color='#aaaaaa',
                smooth_shading=True, lighting=True,
                ambient=0.4, show_edges=False)
    return False


def _fit_camera(pl, parts, pos, focal, up, wsize, target_fill=0.85):
    """Point the camera along the view direction and zoom to fill."""
    import vtk as _vtk
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
        pl.camera.position    = pos
        pl.camera.focal_point = focal
        pl.camera.view_up     = up

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
                            cmap='RdBu_r', clim=list(clim),
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
def export_interactive_html(prepared, scalar_key, cmap=None,
                            clim=(-4.0, 1.0), label='Cp',
                            output_path='surface.html'):
    """
    Export an interactive Plotly 3D HTML file at full resolution.

    Every CFD surface cell is kept (no decimation) and coloured by its own
    value (Plotly intensitymode='cell' — no cell->point averaging). Polygons
    are split into triangles that each carry their parent cell's value.
    Large surfaces give large files; the size is printed when done.

    Returns
    -------
    str   path to saved HTML file
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    if cmap is None:
        cmap = make_cp_cmap(vmin=clim[0], vmax=clim[1])

    vmin, vmax = clim
    positions  = np.linspace(0, 1, 256)
    rgba       = cmap(positions)
    colorscale = [[float(p), f'rgb({int(round(r*255))},{int(round(g*255))},{int(round(b*255))})']
                  for p, (r, g, b, _) in zip(positions, rgba)]

    print("[surface_render] Building interactive HTML (full nodes, every cell)...")
    traces = []
    first  = True
    n_total = 0

    for part in prepared:
        surf = part['surf']
        name = part['name']
        where, arr = _find_array(surf, scalar_key)
        if where is None:
            continue

        t0 = time.perf_counter()
        s = surf.copy()
        for k in list(s.cell_data.keys()):
            if k != arr:
                s.cell_data.remove(k)
        for k in list(s.point_data.keys()):
            if k != arr:
                s.point_data.remove(k)

        tri   = s.triangulate()                    # cell data copied per triangle
        pts   = np.asarray(tri.points, dtype=np.float32)
        faces = tri.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        if where == 'cell':
            intensity = np.asarray(tri.cell_data[arr], dtype=np.float32)
            imode = 'cell'
        else:
            intensity = np.asarray(tri.point_data[arr], dtype=np.float32)
            imode = 'vertex'

        trace = go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=intensity,
            intensitymode=imode,
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
        )
        traces.append(trace)
        first = False
        n_total += len(faces)
        print(f"  {name}: {s.n_cells:,} cells -> {len(faces):,} tris  "
              f"({time.perf_counter()-t0:.1f}s)")

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
            camera=dict(eye=dict(x=-1.2, y=-1.2, z=0.7)),
        ),
        paper_bgcolor='white',
        legend=dict(x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1,
                    font=dict(size=11)),
        width=1400, height=900,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[surface_render] Saved interactive HTML -> {output_path}  "
          f"({size_mb:.1f}MB, {n_total:,} triangles)")
    if size_mb > 200:
        print("  [NOTE] Large file — very big HTML files can be slow to open "
              "in a browser. Chrome/Edge handle them best.")
    return output_path