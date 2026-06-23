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
#    cmap    : matplotlib colormap name OR a LinearSegmentedColormap object
#    label   : axis / colorbar label
# ══════════════════════════════════════════════════════════════════

def _make_cp_cmap(vmin=-4.0, vmax=1.0):
    """
    STAR-CCM+ Cp palette.
    Control points are evenly distributed across [vmin, 0] and [0, vmax].
    White is pinned exactly at Cp=0 regardless of vmin/vmax: the negative side
    ends in green and the positive side begins in yellow.
    """
    zero_pos = (0.0 - vmin) / (vmax - vmin)

    # Below zero (suction): deep blue -> cyan -> teal -> green
    # Ends at green just below zero, so white (Cp=0) sits between green and yellow.
    neg_colours = [
        (  0,   0, 139),   # deep blue       (most negative Cp)
        (  0,  50, 200),
        (  0, 120, 240),
        ( 30, 180, 230),   # cyan
        ( 50, 220, 180),
        (110, 225, 120),   # green           (just below zero)
    ]
    # Above zero (stagnation/pressure): yellow -> orange -> red
    # Yellow is now the FIRST positive colour, just above white.
    pos_colours = [
        (250, 235,  70),   # yellow          (just above zero)
        (245, 165,  30),   # orange
        (235,  90,  10),
        (200,  25,   0),
        (135,   0,   0),   # deep red         (most positive Cp)
    ]

    ctrl = []
    for i, (r, g, b) in enumerate(neg_colours):
        ctrl.append((zero_pos * i / len(neg_colours), r, g, b))
    ctrl.append((zero_pos, 250, 250, 250))   # white pinned exactly at Cp=0
    for i, (r, g, b) in enumerate(pos_colours):
        ctrl.append((zero_pos + (1.0 - zero_pos) * (i + 1) / len(pos_colours), r, g, b))

    cdict = {
        'red':   [(ctrl[i][0], ctrl[i][1]/255, ctrl[i][1]/255) for i in range(len(ctrl))],
        'green': [(ctrl[i][0], ctrl[i][2]/255, ctrl[i][2]/255) for i in range(len(ctrl))],
        'blue':  [(ctrl[i][0], ctrl[i][3]/255, ctrl[i][3]/255) for i in range(len(ctrl))],
    }
    return mcolors.LinearSegmentedColormap('cp_custom', cdict, N=512)


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
        'vmin':  -1.0,
        'vmax':   1.0,
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

# Scalars that only exist on surface exports (Cp, Cf from surface .case)
# Used by surface_render.py — NOT included in the slice pipeline Run tab
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
DILATION_PX   = 3     # px to dilate geometry lines (closes surface mesh gaps)


# ══════════════════════════════════════════════════════════════════
#  PLOT APPEARANCE
# ══════════════════════════════════════════════════════════════════
DPI             = 150
FILL_COLOR      = "#404040"   # solid geometry fill
BG_COLOR        = "white"
TICK_SPACING_MM = 100         # axis tick marks every N mm


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