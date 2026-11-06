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

import re
import numpy as np
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════
#  FILE PATHS
# ══════════════════════════════════════════════════════════════════
GEOMETRY_CASE = r""
DATA_CASE     = r""
OUTPUT_DIR    = r""


# ══════════════════════════════════════════════════════════════════
#  SCALARS
#  Each entry defines one field to process.
#  Keys:
#    array   : exact array name inside the CFD file
#    vmin    : colormap lower bound (clipped)
#    vmax    : colormap upper bound (clipped)
#    cmap    : matplotlib colormap name OR a Colormap object
#    label   : axis / colorbar label
# ══════════════════════════════════════════════════════════════════

# STAR-CCM+ "Mean of Pressure Coefficient" legend, sampled pixel-by-pixel
# from the reference colour bar (range -4 .. +1) and reduced to the control
# points below (mean error ~1.6/255 against the reference).
#   (Cp value, R, G, B)
CP_REF_RANGE = (-4.0, 1.0)
CP_STAR_CTRL = [
    (-4.000,  51,  30,  70),   # deep indigo   (< -4)
    (-3.947,  52,  34,  77),
    (-3.834,  57,  48, 107),
    (-3.685,  63,  61, 137),
    (-3.536,  69,  72, 163),
    (-3.308,  77,  86, 196),
    (-3.081,  81, 107, 216),   # royal blue
    (-2.511,  88, 152, 253),
    (-2.380,  87, 163, 247),   # sky blue
    (-1.767,  82, 211, 202),   # cyan / teal
    (-1.443,  92, 226, 174),
    (-1.005, 104, 245, 128),   # bright green
    (-0.664,  86, 203, 101),
    (-0.357,  63, 155,  69),
    (-0.182,  47, 121,  42),
    (-0.095,  34, 100,  14),   # dark forest green (hard edge)
    (-0.086,  82, 123,  77),
    (-0.060, 140, 157, 138),
    (-0.025, 196, 197, 193),   # neutral grey  (Cp ~ 0)
    (+0.019, 208, 207, 168),
    (+0.063, 214, 214, 121),
    (+0.098, 219, 218,  65),   # yellow
    (+0.124, 222, 215,  55),
    (+0.370, 242, 165,  52),   # orange
    (+0.422, 237, 153,  50),
    (+0.597, 209, 109,  38),
    (+0.772, 178,  45,  18),   # red
    (+0.956, 129,  27,  14),
    (+1.000, 122,  23,  14),   # dark red      (> 1)
]


def _make_cp_cmap(vmin=-4.0, vmax=1.0, N=1024):
    """
    STAR-CCM+ Cp palette (matches the 'Mean of Pressure Coefficient' legend).

    At the reference range (-4 .. +1) the colours are identical to the STAR
    legend. For any other vmin / vmax the negative side is stretched over
    [vmin, 0] and the positive side over [0, vmax], so the grey band always
    sits at Cp ~ 0 regardless of the range.
    """
    ref_lo, ref_hi = CP_REF_RANGE
    cv = np.array([c[0] for c in CP_STAR_CTRL])
    rgb = np.array([c[1:] for c in CP_STAR_CTRL], dtype=float) / 255.0

    v = np.linspace(vmin, vmax, N)
    vref = np.zeros_like(v)
    neg, pos = v < 0, v > 0
    if vmin < 0:
        vref[neg] = v[neg] / vmin * ref_lo
    else:
        vref[neg] = v[neg]
    if vmax > 0:
        vref[pos] = v[pos] / vmax * ref_hi
    else:
        vref[pos] = v[pos]
    vref = np.clip(vref, ref_lo, ref_hi)

    colours = np.stack([np.interp(vref, cv, rgb[:, k]) for k in range(3)], 1)
    cmap = mcolors.ListedColormap(colours, name='cp_star')
    cmap.set_under(colours[0])
    cmap.set_over(colours[-1])
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _make_cpt_cmap():
    """CpT colormap — dark blue at -0.5, cyan/green mid, dark red at +1."""
    ctrl = [
        (-0.50,   0,   0, 139),
        (-0.25,   0,  80, 220),
        ( 0.00,   0, 180, 220),
        ( 0.25,  50, 220, 150),
        ( 0.50, 200, 220,  50),
        ( 0.75, 240, 140,  20),
        ( 1.00, 160,  10,   5),
    ]
    vmin, vmax = -0.5, 1.0
    positions = [(cp - vmin) / (vmax - vmin) for cp, r, g, b in ctrl]
    cdict = {
        'red':   [(positions[i], ctrl[i][1]/255, ctrl[i][1]/255) for i in range(len(ctrl))],
        'green': [(positions[i], ctrl[i][2]/255, ctrl[i][2]/255) for i in range(len(ctrl))],
        'blue':  [(positions[i], ctrl[i][3]/255, ctrl[i][3]/255) for i in range(len(ctrl))],
    }
    return mcolors.LinearSegmentedColormap('cpt_custom', cdict, N=512)


def _make_cf_cmap(vmax=0.03):
    """
    Skin friction colormap.
    Cf = 0 exactly: hard pink/magenta sentinel colour (not blended).
    Cf > 0: blue gradient from light to dark.
    vmax: upper bound (default 0.03).
    """
    # A very thin sliver at position 0 is pure pink
    # then immediately jumps to the blue gradient above a tiny epsilon
    eps = 1.0 / 512   # one LUT step above zero

    # Blue gradient control points (positions eps..1)
    blue_ctrl = [
        (eps,   230, 240, 255),   # very light blue — just above zero
        (0.15,  180, 215, 245),
        (0.35,  100, 170, 230),
        (0.55,   40, 110, 210),
        (0.75,   10,  60, 185),
        (1.00,    0,  20, 140),   # darkest blue — max Cf
    ]

    # Build cdict with hard step at zero:
    # at pos=0:   pink
    # at pos=eps: light blue (hard jump — two entries at same or adjacent positions)
    pink = (220/255, 80/255, 160/255)
    cdict = {'red': [], 'green': [], 'blue': []}

    # Pin pink at exactly 0 — use two entries at 0 and eps to create hard edge
    cdict['red']  += [(0.0,   pink[0], pink[0]),
                      (eps,   blue_ctrl[0][1]/255, blue_ctrl[0][1]/255)]
    cdict['green']+= [(0.0,   pink[1], pink[1]),
                      (eps,   blue_ctrl[0][2]/255, blue_ctrl[0][2]/255)]
    cdict['blue'] += [(0.0,   pink[2], pink[2]),
                      (eps,   blue_ctrl[0][3]/255, blue_ctrl[0][3]/255)]

    # Rest of blue gradient
    for p, r, g, b in blue_ctrl[1:]:
        cdict['red'].append((p, r/255, r/255))
        cdict['green'].append((p, g/255, g/255))
        cdict['blue'].append((p, b/255, b/255))

    return mcolors.LinearSegmentedColormap('cf_custom', cdict, N=512)


SCALARS = {
    'Cp': {
        'array': 'MeanPressureCoeffromVarianceofPressureC',
        'vmin':  -4.0,
        'vmax':   1.0,
        'cmap':  _make_cp_cmap(),
        'label': 'Cp',
    },
    'CpT': {
        'array': 'MeanCpTinRRFfromVarianceofCpTinRRF',
        'vmin':  -0.5,
        'vmax':   1.0,
        'cmap':  _make_cpt_cmap(),
        'label': 'CpT',
    },
    'Vi': {
        'array': 'VelocityMeanNormalizedi',
        'vmin':  -2.0,
        'vmax':   2.0,
        'cmap':  'RdBu_r',
        'label': 'Vi (norm)',
    },
    'Vj': {
        'array': 'VelocityMeanNormalizedj',
        'vmin':  -1.0,
        'vmax':   1.0,
        'cmap':  'RdBu_r',
        'label': 'Vj (norm)',
    },
    'Vk': {
        'array': 'VelocityMeanNormalizedk',
        'vmin':  -1.0,
        'vmax':   1.0,
        'cmap':  'RdBu_r',
        'label': 'Vk (norm)',
    },
}

# Scalars whose colormap is rebuilt (not replaced) when vmin/vmax are edited
# in the Settings tab, so they keep their physics-informed palette.
CMAP_FACTORIES = {
    'Cp': _make_cp_cmap,
}

# Scalars that only exist on surface exports (Cp, Cf from surface .case)
# Used by surface_render.py / surface_cp.py — NOT included in the Run tab
SURFACE_SCALARS = {
    'Cp': SCALARS['Cp'],   # Cp exists in both — share the definition
    'Cf': {
        'array': 'MeanSkinFrictionCoeffromVarianceofSkinF',
        'vmin':   0.0,
        'vmax':   0.03,
        'cmap':  _make_cf_cmap(vmax=0.03),
        'label': 'Cf',
    },
}


# ══════════════════════════════════════════════════════════════════
#  COMPARISON DEFAULTS
#  Default symmetric delta colormap range (+/- value) used by the
#  Compare tab, the surface delta render and `cli.py compare`.
#    +/-0.5  everything (Cp, CpT, normalised velocities, ...)
#    +/-500  vorticity  (UT26 VorticityMagnitude: median ~100, p99 ~70,000 1/s)
#  Vorticity is recognised by name — any key or array containing
#  'Vorticity' (any case), e.g. VorticityMagnitude / 'VorticityMag'.
# ══════════════════════════════════════════════════════════════════
DELTA_RANGE_DEFAULT   = 0.5
DELTA_RANGE_VORTICITY = 500.0


def is_vorticity(key, array=""):
    """True if a scalar key or array name is a vorticity field."""
    return any('vorticity' in re.sub(r'[^a-z0-9]', '', str(s or '').lower())
               for s in (key, array))


def default_delta_range(key):
    """Default +/- delta range: 500 for vorticity, else 0.5."""
    sd = SCALARS.get(key) or SURFACE_SCALARS.get(key) or {}
    if is_vorticity(key, sd.get('array', '')):
        return DELTA_RANGE_VORTICITY
    return DELTA_RANGE_DEFAULT


# ══════════════════════════════════════════════════════════════════
#  RECRUITS  (parallel plane processing)
#  The Run tab RAM slider decides how many planes are processed at
#  once. Each recruit is a separate worker process with its own copy
#  of the geometry, so RAM — not CPU — is usually the limit.
# ══════════════════════════════════════════════════════════════════
RECRUIT_RAM_GB      = None    # None = one plane at a time (no recruits)
RECRUIT_BASE_MB     = 300     # per-recruit overhead: Python + VTK + libs (measured ~170)
RECRUIT_RESERVE_GB  = 2.0     # RAM always left free for the OS / GUI


# ══════════════════════════════════════════════════════════════════
#  DOMAIN BOUNDS
#  Clips CFD data and sets plot extents.
#  Format per axis: (h_min, h_max, v_min, v_max) in metres
#    axis 0 (X-slice): h=Y,  v=Z
#    axis 1 (Y-slice): h=X,  v=Z
#    axis 2 (Z-slice): h=X,  v=Y
# ══════════════════════════════════════════════════════════════════
CAR_BOUNDS_2D = {
    0: (-0.90,  0.90, -0.10, 1.30),   # X-slice
    1: (-1.10,  2.20, -0.10, 1.30),   # Y-slice
    2: (-1.10,  2.20, -0.90, 0.90),   # Z-slice
}


# ══════════════════════════════════════════════════════════════════
#  GRID & MASK
# ══════════════════════════════════════════════════════════════════
RESOLUTION_MM = 1     # mm per pixel — increase to 2 or 5 for speed
DILATION_PX   = 1     # px to dilate geometry lines (closes surface mesh gaps)


# ══════════════════════════════════════════════════════════════════
#  PLOT APPEARANCE
# ══════════════════════════════════════════════════════════════════
DPI             = 150
FILL_COLOR      = "#404040"   # solid geometry fill
BG_COLOR        = "white"
TICK_SPACING_MM = 100         # axis tick marks every N mm
COLORBAR_TICKS  = 5           # STAR-style: evenly spaced, '<' / '>' on the ends


# ══════════════════════════════════════════════════════════════════
#  VIEW ORIENTATION
#  Mirror the horizontal plot axis per slice direction so the plots use
#  the same axis system as the STAR-CCM+ sim exports.
#    X-slices (Y-Z plane): viewed from the front, looking rearward along +X
#                          -> Y points LEFT, Z up   (matches the sim triad)
#  Display only — NPZ grids are stored un-mirrored, so comparisons between
#  runs are unaffected.
# ══════════════════════════════════════════════════════════════════
FLIP_HORIZONTAL = {
    0: True,    # X-slice  (h = Y): Y increases to the left
    1: False,   # Y-slice  (h = X)
    2: False,   # Z-slice  (h = X)
}


# ══════════════════════════════════════════════════════════════════
#  AXIS HELPERS  (not user-editable — derived constants)
# ══════════════════════════════════════════════════════════════════
AXIS_LABEL  = {0: "X",  1: "Y",  2: "Z"}
AXIS_NORMAL = {0: "x",  1: "y",  2: "z"}
PLOT_AXES   = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
PLOT_LABELS = {
    0: ("Y (m)", "Z (m)"),
    1: ("X (m)", "Z (m)"),
    2: ("X (m)", "Y (m)"),
}
# CFD block index -> true slice axis (block names in file are misleading)
BLOCK_AXIS  = {0: 0, 1: 2, 2: 1}


def runtime_snapshot():
    """
    Everything a worker process needs to reproduce the current (possibly
    GUI-edited) configuration. Used by the recruits feature, since worker
    processes re-import this module from disk and would otherwise lose
    any changes made in the Settings tab.
    """
    return dict(
        SCALARS=SCALARS, SURFACE_SCALARS=SURFACE_SCALARS,
        CAR_BOUNDS_2D=dict(CAR_BOUNDS_2D),
        RESOLUTION_MM=RESOLUTION_MM, DILATION_PX=DILATION_PX,
        DPI=DPI, FILL_COLOR=FILL_COLOR, BG_COLOR=BG_COLOR,
        TICK_SPACING_MM=TICK_SPACING_MM, COLORBAR_TICKS=COLORBAR_TICKS,
        FLIP_HORIZONTAL=dict(FLIP_HORIZONTAL),
    )


def apply_snapshot(snap):
    """Inverse of runtime_snapshot() — called inside each worker."""
    g = globals()
    for k, v in snap.items():
        g[k] = v
