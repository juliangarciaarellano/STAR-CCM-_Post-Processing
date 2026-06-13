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
Renders CFD surface data onto the geometry mesh.

Functions
---------
prepare_surfaces(geom_case, surface_case, scalars=None)
    -> list of (surf_mesh, edges_mesh, part_name)
       Run once per dataset — KD-tree matching + edge extraction.

render_views(prepared, scalar_key, cmap, clim,
             views=None, window_size=(1600,1000),
             edge_color='black', edge_width=0.8,
             bg_color='white', feature_angle=25)
    -> dict[view_name, numpy RGB image]

render_delta_views(prepared_a, prepared_b, scalar_key,
                   views=None, window_size=(1600,1000),
                   fixed_range=None, feature_angle=25)
    -> dict[view_name, numpy RGB image]

export_3d(prepared, scalar_keys, output_dir, fmt='vtp')
    Exports one file per part (VTP/VTK) or one merged file (STL/PLY/OBJ).

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


# ── Colourmap ─────────────────────────────────────────────────────
def make_cp_cmap(mute=0.45, vmin=-4.0, vmax=1.0):
    """
    STAR-CCM+ Cp palette desaturated toward gray.
    mute=0.0 -> full saturation, mute=1.0 -> all gray.

    Control points are defined as (position_0_to_1, r, g, b) where position
    is a fraction of the full [vmin, vmax] range.  Grey is pinned at exactly
    the position corresponding to Cp=0, regardless of vmin/vmax.
    """
    # Position of Cp=0 in the normalised [0,1] range
    zero_pos = (0.0 - vmin) / (vmax - vmin)   # e.g. 0.80 for vmin=-4, vmax=1

    # Control points defined in normalised [0,1] space
    # Negative half: 13 evenly spaced points from 0.0 to zero_pos (blue->grey)
    # Positive half: proportional points from zero_pos to 1.0 (grey->red)
    # Below zero (suction): blue -> cyan -> green, ordered darkest->grey
    neg_colours = [
        (  0,   0, 139),   # deep blue       (most negative Cp)
        (  0,  50, 200),
        (  0, 120, 240),
        ( 30, 180, 230),   # cyan
        ( 50, 220, 180),
        (100, 240, 100),   # green
        (180, 240,  60),
        (240, 230,  40),   # yellow-green    (just below zero)
    ]
    # Above zero (stagnation/pressure): orange -> red
    pos_colours = [
        (240, 160,  20),   # orange          (just above zero)
        (230,  80,   0),
        (200,  20,   0),
        (140,   0,   0),   # deep red        (most positive Cp)
    ]

    # Build list of (position, r, g, b)
    n_neg = len(neg_colours)
    n_pos = len(pos_colours)
    ctrl = []
    for i, (r, g, b) in enumerate(neg_colours):
        p = zero_pos * i / n_neg
        ctrl.append((p, r, g, b))
    ctrl.append((zero_pos, 185, 185, 185))   # grey exactly at Cp=0
    for i, (r, g, b) in enumerate(pos_colours):
        p = zero_pos + (1.0 - zero_pos) * (i + 1) / n_pos
        ctrl.append((p, r, g, b))

    gray = np.array([160, 160, 160])
    muted = []
    for p, r, g, b in ctrl:
        # Pin grey exactly — don't blend
        if abs(p - zero_pos) < 1e-9:
            muted.append((p, 185, 185, 185))
        else:
            muted.append((p,
                          int(r*(1-mute)+gray[0]*mute),
                          int(g*(1-mute)+gray[1]*mute),
                          int(b*(1-mute)+gray[2]*mute)))

    cd = {k: [(muted[i][0], muted[i][j]/255, muted[i][j]/255)
              for i in range(len(muted))]
          for k, j in [('red',1), ('green',2), ('blue',3)]}
    return mcolors.LinearSegmentedColormap('cp_surface', cd, N=512)


# ── Core preparation ──────────────────────────────────────────────
def prepare_surfaces(geom_case, surface_case=None, scalars=None,
                     feature_angle=25):
    """
    Load geometry and optionally surface CFD data.
    Runs KD-tree matching and feature edge extraction once.

    Parameters
    ----------
    geom_case    : str   path to geometry .case file
    surface_case : str or None   path to surface CFD .case file
                   If None, renders geometry only (no scalar colouring).
    scalars      : list of str or None   which arrays to transfer
                   defaults to all available in surface_case
    feature_angle: float   angle threshold for feature edge extraction

    Returns
    -------
    list of dicts, one per geometry part:
        name    : str
        surf    : pyvista PolyData   surface mesh with scalar cell_data
        edges   : pyvista PolyData or None   feature edges
    """
    t0 = time.perf_counter()
    print("[surface_render] Loading geometry...")
    geom = pv.read(geom_case)

    surf_mesh = None
    if surface_case is not None:
        print("[surface_render] Loading surface data...")
        surf_mesh = pv.read(surface_case)

        # Discover available scalars
        available = set()
        for i in range(surf_mesh.n_blocks):
            b = surf_mesh[i]
            if b is not None and b.n_cells > 0:
                available.update(b.cell_data.keys())
        scalars_to_use = scalars if scalars else sorted(available)
        print(f"[surface_render] Scalars: {scalars_to_use}")

    print("[surface_render] Preparing surfaces (KD-tree + edges)...")
    prepared = []

    for i in range(geom.n_blocks):
        g    = geom[i]
        name = geom.get_block_name(i).strip()
        if g is None or g.n_cells == 0:
            continue

        g_copy = g.copy()

        # Transfer scalars via KD-tree
        if surf_mesh is not None:
            s = surf_mesh[i]
            if s is not None and s.n_cells > 0:
                s_cc   = s.cell_centers().points
                g_cc   = g.cell_centers().points
                tree   = cKDTree(s_cc)
                _, idx = tree.query(g_cc, k=1, workers=-1)

                for skey in scalars_to_use:
                    if skey in s.cell_data:
                        g_copy.cell_data[skey] = s.cell_data[skey][idx]

        # Extract surface
        surf = g_copy.extract_surface(algorithm='dataset_surface')
        if 'vtkOriginalCellIds' in surf.cell_data:
            orig = surf.cell_data['vtkOriginalCellIds']
            for skey in (scalars_to_use if surf_mesh else []):
                if skey in g_copy.cell_data:
                    surf.cell_data[skey] = g_copy.cell_data[skey][orig]

        # Feature edges
        edges = surf.extract_feature_edges(
            boundary_edges=True, feature_edges=True,
            feature_angle=feature_angle,
            non_manifold_edges=False, manifold_edges=False,
        )
        edges = edges if edges.n_cells > 0 else None

        prepared.append(dict(name=name, surf=surf, edges=edges))
        print(f"  {name}: {surf.n_cells:,} cells")

    # ── Merge related parts into logical groups ───────────────────
    # FW + FW EPs -> 'Front wing'
    # RW + RW EPs -> 'Rear wing'
    MERGE_GROUPS = {
        'Front wing': ['Front wing', 'Front wing EPs'],
        'Rear wing':  ['Rear wing',  'Rear wing EPs'],
    }
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
    consumed = set()
    merged_entries = []

    for group_name, members in groups.items():
        parts_in_group = [p for p in prepared if p['name'] in members]
        if len(parts_in_group) < 2:
            continue  # nothing to merge

        # Merge surfaces and edges
        all_surfs = [p['surf']  for p in parts_in_group]
        all_edges = [p['edges'] for p in parts_in_group if p['edges'] is not None]

        merged_surf  = pv.merge(all_surfs)
        merged_edges = pv.merge(all_edges) if all_edges else None

        merged_entries.append(dict(
            name=group_name,
            surf=merged_surf,
            edges=merged_edges,
        ))
        for p in parts_in_group:
            consumed.add(p['name'])
        print(f"  merged: {members} -> '{group_name}'")

    # Keep unconsumed parts, inserting merged groups at the position
    # of the first consumed member
    result = []
    inserted = set()
    for p in prepared:
        if p['name'] not in consumed:
            result.append(p)
        else:
            # Find which group this belongs to and insert it once
            for group_name, members in groups.items():
                if p['name'] in members and group_name not in inserted:
                    merged = next(e for e in merged_entries
                                  if e['name'] == group_name)
                    result.append(merged)
                    inserted.add(group_name)
                    break

    return result


def _render_one_view(parts, scalar_key, cmap, clim,
                     vname, window_size, edge_color, edge_width,
                     bg_color, zoom=False, zoom_factor=1.8):
    """
    Render a single view of the given parts list.
    If zoom=True, uses reset_camera() then zooms in so the part fills
    ~85% of the shorter window dimension, correcting for aspect ratio.
    zoom_factor is ignored when zoom=True — fill logic takes over.
    Returns RGB numpy array.
    """
    pos, focal, up = VIEWS[vname]
    wsize = VIEW_WINDOW_SIZES.get(vname, window_size) if window_size == (1600, 1000) \
            else window_size
    pl = pv.Plotter(off_screen=True, window_size=list(wsize))
    pl.set_background(bg_color)

    for part in parts:
        surf  = part['surf']
        edges = part['edges']
        if scalar_key in surf.cell_data:
            pl.add_mesh(surf, scalars=scalar_key, preference='cell',
                        cmap=cmap, clim=list(clim),
                        smooth_shading=False, interpolate_before_map=True,
                        show_scalar_bar=False, lighting=False, show_edges=False)
        else:
            # Scalar missing — surface_case was not loaded or array not found
            # Print a warning so the user knows which part is missing data
            print(f"  [WARN] '{scalar_key}' not found on '{parts[0].get('name','?') if len(parts)==1 else 'merged'}'"
                  f" — check Surface data .case path is set in the GUI")
            pl.add_mesh(surf, color='#aaaaaa',
                        smooth_shading=True, lighting=True,
                        ambient=0.4, show_edges=False)
        if edges is not None:
            pl.add_mesh(edges, color=edge_color,
                        line_width=edge_width, lighting=False)

    if zoom:
        # Step 1: set view direction and fit camera to content
        pl.view_vector(
            vector=tuple(p - f for p, f in zip(pos, focal)),
            viewup=up
        )
        pl.reset_camera()

        # Step 2: compute projected screen bounds to calculate fill ratio
        import numpy as np
        import vtk as _vtk
        all_bounds = np.array([p['surf'].bounds for p in parts])
        world_min = all_bounds[:, [0,2,4]].min(axis=0)
        world_max = all_bounds[:, [1,3,5]].max(axis=0)
        corners_3d = np.array([[x,y,z]
            for x in [world_min[0],world_max[0]]
            for y in [world_min[1],world_max[1]]
            for z in [world_min[2],world_max[2]]])

        coord = _vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        screen_pts = []
        for pt in corners_3d:
            coord.SetValue(*pt)
            sp = coord.GetComputedDisplayValue(pl.renderer)
            screen_pts.append(sp[:2])
        screen_pts = np.array(screen_pts, dtype=float)

        W, H = wsize
        pt_w = screen_pts[:,0].max() - screen_pts[:,0].min()
        pt_h = screen_pts[:,1].max() - screen_pts[:,1].min()

        fill_w = pt_w / W
        fill_h = pt_h / H

        TARGET_FILL = 0.85
        current_fill = max(fill_w, fill_h)
        if current_fill > 0:
            zoom_in = TARGET_FILL / current_fill
            pl.camera.zoom(zoom_in)
    else:
        pl.camera.position    = pos
        pl.camera.focal_point = focal
        pl.camera.view_up     = up

    img = pl.screenshot(None, return_img=True)
    pl.close()
    return img


# ── Single-run render ─────────────────────────────────────────────
def render_views(prepared, scalar_key, cmap=None, clim=(-3.0, 1.0),
                 views=None, window_size=(1600, 1000),
                 edge_color='black', edge_width=0.8,
                 bg_color='white',
                 output_dir=None, file_stem='surface'):
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
    edge_color  : str
    edge_width  : float
    bg_color    : str
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
                               bg_color, zoom=True)   # fill logic for all views
        imgs[vname] = img

        if output_dir:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.colors as mc

            # ── PNG ──────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
            ax.imshow(img); ax.axis('off')
            sm = plt.cm.ScalarMappable(cmap=cmap,
                 norm=mc.Normalize(vmin=clim[0], vmax=clim[1]))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(f'{file_stem}  --  {vname.replace("_"," ").title()}',
                         fontsize=11, color='black' if bg_color=='white' else 'white')
            out_path = os.path.join(png_dir, f'{file_stem}_{vname}.png')
            fig.savefig(out_path, dpi=150, facecolor=bg_color,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"  saved: PNG/{os.path.basename(out_path)}")

            # ── NPZ ──────────────────────────────────────────────
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
                      cmap=None, clim=(-3.0, 1.0),
                      views=None, window_size=(1600, 1000),
                      edge_color='black', edge_width=0.8,
                      bg_color='white',
                      output_dir=None, file_stem='surface',
                      zoom_factor=1.4):
    """
    Render individual views for each requested part, auto-zoomed to fit.
    Saves one PNG per (part, view) combination.

    Parameters
    ----------
    part_names  : list of str   part names to render individually
                  must match the 'name' keys in prepared
    output_dir  : str or None   where to save PNGs
    file_stem   : str           prefix e.g. 'surface_Cp'

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

    # Build name -> part lookup
    part_lookup = {p['name']: p for p in prepared}

    results = {}
    for pname in part_names:
        if pname not in part_lookup:
            print(f"[surface_render] Part '{pname}' not found, skipping")
            continue

        part = part_lookup[pname]
        parts_single = [part]
        safe_name = pname.replace(' ', '_').replace('/', '_')

        for vname in views_to_render:
            if vname not in VIEWS:
                continue
            t0  = time.perf_counter()
            img = _render_one_view(parts_single, scalar_key, cmap, clim,
                                   vname, window_size, edge_color, edge_width,
                                   bg_color, zoom=True, zoom_factor=zoom_factor)
            results[(pname, vname)] = img

            if output_dir:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import matplotlib.colors as mc

                # ── PNG ──────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
                ax.imshow(img); ax.axis('off')
                sm = plt.cm.ScalarMappable(cmap=cmap,
                     norm=mc.Normalize(vmin=clim[0], vmax=clim[1]))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
                cbar.ax.tick_params(labelsize=8)
                ax.set_title(f'{pname}  --  {vname.replace("_"," ").title()}',
                             fontsize=11,
                             color='black' if bg_color=='white' else 'white')
                out_path = os.path.join(png_dir,
                                        f'{file_stem}_{safe_name}_{vname}.png')
                fig.savefig(out_path, dpi=150, facecolor=bg_color,
                            bbox_inches='tight')
                plt.close(fig)
                print(f"  saved: PNG/{os.path.basename(out_path)}")

                # ── NPZ ──────────────────────────────────────────
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

    Parameters
    ----------
    prepared_a/b : lists from prepare_surfaces()
    scalar_key   : str
    fixed_range  : float or None   symmetric clim for delta colormap
    views        : list of str or None

    Returns
    -------
    dict[view_name, np.ndarray]
    """
    views_to_render = views or list(VIEWS.keys())

    # Build name->part lookup
    parts_a = {p['name']: p for p in prepared_a}
    parts_b = {p['name']: p for p in prepared_b}
    all_names = sorted(set(parts_a.keys()) | set(parts_b.keys()))

    # Compute per-part delta meshes and geometry overlay
    delta_parts  = []   # (surf_with_delta, edges, geom_color)
    GRAY    = '#404040'
    MAGENTA = '#CC00CC'
    BROWN   = '#8B4513'

    for name in all_names:
        pa = parts_a.get(name)
        pb = parts_b.get(name)

        if pa is not None and pb is not None:
            geom_color = GRAY
            # Compute delta — requires matching cells
            if (scalar_key in pa['surf'].cell_data and
                    scalar_key in pb['surf'].cell_data):
                va = pa['surf'].cell_data[scalar_key]
                vb = pb['surf'].cell_data[scalar_key]
                # Match by cell centre KD-tree
                cc_a = pa['surf'].cell_centers().points
                cc_b = pb['surf'].cell_centers().points
                _, idx = cKDTree(cc_b).query(cc_a, k=1, workers=-1)
                delta = vb[idx] - va
                surf_d = pa['surf'].copy()
                surf_d.cell_data['delta'] = delta
                edges = pa['edges']
            else:
                surf_d = pa['surf'].copy()
                edges  = pa['edges']

        elif pb is not None:
            geom_color = MAGENTA
            surf_d = pb['surf'].copy()
            edges  = pb['edges']
        else:
            geom_color = BROWN
            surf_d = pa['surf'].copy()
            edges  = pa['edges']

        delta_parts.append(dict(
            surf=surf_d, edges=edges,
            geom_color=geom_color,
            has_delta=('delta' in surf_d.cell_data),
        ))

    # Auto-range if needed
    if fixed_range is None:
        all_deltas = []
        for p in delta_parts:
            if p['has_delta']:
                all_deltas.append(p['surf'].cell_data['delta'])
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
                pl.add_mesh(part['surf'], scalars='delta', preference='cell',
                            cmap='RdBu_r', clim=list(clim),
                            smooth_shading=False,
                            interpolate_before_map=True,
                            show_scalar_bar=False,
                            lighting=False, show_edges=False)
            else:
                pl.add_mesh(part['surf'], color=part['geom_color'],
                            opacity=0.8, lighting=True,
                            ambient=0.4, show_edges=False)

            if part['edges'] is not None:
                pl.add_mesh(part['edges'], color='black',
                            line_width=0.8, lighting=False)

        # Fill-based zoom
        pl.view_vector(tuple(p-f for p,f in zip(pos,focal)), viewup=up)
        pl.reset_camera()
        import vtk as _vtk
        all_bounds = np.array([p['surf'].bounds for p in delta_parts])
        wmin = all_bounds[:,[0,2,4]].min(axis=0)
        wmax = all_bounds[:,[1,3,5]].max(axis=0)
        corners = np.array([[x,y,z] for x in [wmin[0],wmax[0]]
                                     for y in [wmin[1],wmax[1]]
                                     for z in [wmin[2],wmax[2]]])
        coord = _vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        spts = []
        for pt in corners:
            coord.SetValue(*pt)
            spts.append(coord.GetComputedDisplayValue(pl.renderer)[:2])
        spts = np.array(spts, dtype=float)
        W, H = wsize
        fill = max((spts[:,0].max()-spts[:,0].min())/W,
                   (spts[:,1].max()-spts[:,1].min())/H)
        if fill > 0:
            pl.camera.zoom(0.85 / fill)

        imgs[vname] = pl.screenshot(None, return_img=True)
        pl.close()
        print(f"  {vname}: {time.perf_counter()-t0:.1f}s")

    return imgs, clim


# ── 3D export ─────────────────────────────────────────────────────
def export_3d(prepared, scalar_keys=None, output_dir='.',
              fmt='vtp', merged=False):
    """
    Export prepared surface meshes to a 3D file format.

    Parameters
    ----------
    prepared    : list from prepare_surfaces()
    scalar_keys : list of str or None   scalars to include in export
                  (only relevant for VTP/VTK — ignored for STL/PLY/OBJ)
    output_dir  : str
    fmt         : str   one of 'vtp', 'vtk', 'stl', 'ply', 'obj'
    merged      : bool  if True, merge all parts into one file
                        if False, one file per part (recommended for VTP/VTK)

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

    if merged:
        # Single merged file — geometry only for STL/PLY/OBJ
        all_surfs = []
        for part in prepared:
            s = part['surf'].copy()
            if not scalar_supports:
                # Strip all data arrays
                for key in list(s.cell_data.keys()):
                    s.cell_data.remove(key)
                for key in list(s.point_data.keys()):
                    s.point_data.remove(key)
            all_surfs.append(s)

        merged_mesh = pv.merge(all_surfs)
        out_path = os.path.join(output_dir, f'surface_merged.{fmt}')
        merged_mesh.save(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[surface_render] Exported merged -> {out_path}  ({size_mb:.1f}MB)")
        exported.append(out_path)

    else:
        # One file per part
        for part in prepared:
            s    = part['surf'].copy()
            name = part['name'].replace(' ', '_').replace('/', '_')

            if not scalar_supports:
                for key in list(s.cell_data.keys()):
                    s.cell_data.remove(key)
                for key in list(s.point_data.keys()):
                    s.point_data.remove(key)
            else:
                # Keep only requested scalars
                if scalar_keys is not None:
                    for key in list(s.cell_data.keys()):
                        if key not in scalar_keys and key != 'vtkOriginalCellIds':
                            s.cell_data.remove(key)

            out_path = os.path.join(output_dir, f'{name}.{fmt}')
            s.save(out_path)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  {name}.{fmt}  ({size_mb:.1f}MB)")
            exported.append(out_path)

    return exported


# ── Interactive HTML export ───────────────────────────────────────
def export_interactive_html(prepared, scalar_key, cmap=None,
                            clim=(-3.0, 1.0), label='Cp',
                            output_path='surface.html',
                            triangles_per_part=60_000):
    """
    Export an interactive Plotly 3D HTML file.
    Opens in any browser — no software installation required.
    Drag to rotate, scroll to zoom, click legend to toggle parts.

    Parameters
    ----------
    prepared          : list from prepare_surfaces()
    scalar_key        : str   array name to colour by
    cmap              : matplotlib colormap or None (uses make_cp_cmap())
    clim              : (vmin, vmax)
    label             : str   colorbar label
    output_path       : str   path to .html output file
    triangles_per_part: int   decimate each part to this many triangles
                              for browser performance (default 60_000)

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

    # Build Plotly colorscale from matplotlib colormap
    vmin, vmax = clim
    positions  = np.linspace(0, 1, 64)
    rgba       = cmap(positions)
    colorscale = [[float(p), f'rgb({int(r*255)},{int(g*255)},{int(b*255)})']
                  for p, (r, g, b, _) in zip(positions, rgba)]

    print(f"[surface_render] Building interactive HTML "
          f"({triangles_per_part:,} tris/part)...")
    traces = []
    first  = True

    for part in prepared:
        surf = part['surf']
        name = part['name']
        if scalar_key not in surf.cell_data:
            continue

        t0 = time.perf_counter()

        # Copy only the scalar we need
        s = surf.copy()
        for k in list(s.cell_data.keys()):
            if k != scalar_key:
                s.cell_data.remove(k)

        # Fast grid-based simplification via vtkQuadricClustering
        # Works on mixed cell types (quads + tris), no triangulation needed
        # Target ~triangles_per_part cells by tuning grid divisions
        import vtk as _vtk
        divisions = max(30, int((triangles_per_part / 3) ** (1/2)))
        qc = _vtk.vtkQuadricClustering()
        qc.SetInputData(s)
        qc.SetNumberOfDivisions(divisions, divisions, divisions)
        qc.AutoAdjustNumberOfDivisionsOn()
        qc.Update()
        simplified = pv.wrap(qc.GetOutput())

        # Triangulate the simplified mesh for Plotly (needs triangles)
        tri    = simplified.triangulate()
        pts    = tri.points
        faces  = tri.faces.reshape(-1, 4)[:, 1:]

        # cell_data_to_point_data for smooth vertex colours
        tri_pt  = tri.cell_data_to_point_data()
        pt_vals = tri_pt.point_data.get(
            scalar_key, np.zeros(len(pts), dtype=np.float32)
        ).astype(np.float32)

        trace = go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=pt_vals,
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
            flatshading=False,
            hovertemplate=(f'<b>{name}</b><br>'
                           f'{label}: %{{intensity:.3f}}<extra></extra>'),
            showlegend=True,
        )
        traces.append(trace)
        first = False
        print(f"  {name}: {len(faces):,} tris  ({time.perf_counter()-t0:.1f}s)")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f'UT26 — Surface {label}  |  '
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
    print(f"[surface_render] Saved interactive HTML -> {output_path}  ({size_mb:.1f}MB)")
    return output_path