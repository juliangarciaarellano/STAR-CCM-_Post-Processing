"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

recruits.py
===========
RAM budgeting for parallel plane processing ("recruits").

Credit: Recruits — original beta by George Wei.

Each recruit is a worker process that holds its own copy of the geometry and
processes one plane at a time. The RAM slider in the Run tab sets a budget;
this module turns that budget into a number of planes processed at once.

Kept free of heavy imports (no pyvista / vtk) so the GUI can call it live
while the slider moves.

Functions
---------
system_ram()                       -> (total_bytes, available_bytes)
geometry_ram_bytes(geom_case)      -> estimated bytes for one geometry copy
plane_ram_bytes(axis, res_mm, n_scalars, fig_height, dpi, n_pts=None)
plan(ram_gb, axes, res_mm, n_scalars, fig_height, dpi, geom_case, n_pts=None)
                                   -> dict(recruits, per_recruit_gb, ...)
"""

import os
import re
import glob

import config as cfg

CREDIT = "Recruits — original beta by George Wei"

GB = 1024 ** 3
MB = 1024 ** 2

# ── Calibration ───────────────────────────────────────────────────────
# Measured peak RSS of one plane (Y-slice, 3.3 x 1.4 m, DPI 150):
#   1 mm, 200k pts, 1 scalar ->  678 MB      2 mm, 200k pts, 1 scalar -> 221 MB
#   1 mm, 800k pts, 3 scalar ->  908 MB      2 mm, 200k pts, 3 scalar -> 306 MB
# A plane peaks in one of two stages, so the estimate is the larger of:
#   grid stage   : mask build + per-scalar grids + PNG rendering
#   interp stage : Qhull Delaunay of the plane's CFD points + query arrays
BYTES_PER_PX        = 100     # grid stage, per pixel
BYTES_PER_PX_SCALAR = 32      # grid stage, per pixel per scalar
FIG_BYTES_PER_PX    = 24      # matplotlib canvas + resampling, per output px
BYTES_PER_CFD_PT    = 700     # interp stage, scipy griddata (measured 670-750)
BYTES_PER_PX_INTERP = 40      # interp stage, region coords / query points
# CFD points per mm^2 of plane area assumed when the real count is not known
# yet (GUI estimate before the data is loaded). UT26 STAR export: densest
# plane 0.046 pts/mm^2 (X), 0.023 (Y / Z). The pipeline re-checks with the
# real point counts once the planes are extracted.
ASSUMED_PTS_PER_MM2 = 0.05
# Geometry in RAM vs its size on disk (UT26 Geometry00000.geo: 158 MB on
# disk -> 362 MB loaded = 2.3x; slicing temporaries on top)
GEOM_RAM_FACTOR     = 2.5


# ══════════════════════════════════════════════════════════════════
#  SYSTEM RAM
# ══════════════════════════════════════════════════════════════════

def system_ram():
    """Return (total_bytes, available_bytes); either may be None if unknown."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    if os.name == 'nt':
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        except Exception:
            return None, None

    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.strip().split()[0]) * 1024
        return info.get('MemTotal'), info.get('MemAvailable')
    except Exception:
        pass

    try:   # macOS
        import subprocess
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
        return total, None
    except Exception:
        return None, None


def cpu_limit():
    """Max recruits by CPU: one per logical core."""
    return max(1, os.cpu_count() or 1)


# ══════════════════════════════════════════════════════════════════
#  ESTIMATES
# ══════════════════════════════════════════════════════════════════

def _case_model_files(case_path):
    """Geometry files referenced by an EnSight .case 'model:' line."""
    try:
        with open(case_path, 'r', errors='replace') as f:
            text = f.read()
    except Exception:
        return []
    m = re.search(r'^\s*model\s*:(.*)$', text, flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return []
    tokens = m.group(1).split()
    if not tokens:
        return []
    name = tokens[-1]
    if name.lower() in ('change_coords_only',) and len(tokens) > 1:
        name = tokens[-2]
    base = os.path.dirname(os.path.abspath(case_path))
    pattern = os.path.join(base, name.replace('*', '?'))
    hits = sorted(glob.glob(pattern)) if ('*' in name or '?' in name) \
        else [os.path.join(base, name)]
    return [h for h in hits if os.path.isfile(h)][:1]


_GEOM_CACHE = {}


def geometry_ram_bytes(geom_case):
    """Rough RAM for one loaded copy of the geometry (cached per file)."""
    if not geom_case or not os.path.exists(geom_case):
        return 0
    key = (os.path.abspath(geom_case), os.path.getmtime(geom_case))
    if key not in _GEOM_CACHE:
        _GEOM_CACHE[key] = _geometry_ram_bytes(geom_case)
    return _GEOM_CACHE[key]


def _geometry_ram_bytes(geom_case):
    files = []
    if geom_case.lower().endswith('.case'):
        files = _case_model_files(geom_case)
    if not files:
        files = [geom_case]
        stem_dir = os.path.splitext(geom_case)[0]      # e.g. .vtm data folder
        if os.path.isdir(stem_dir):
            for dp, _, fs in os.walk(stem_dir):
                files += [os.path.join(dp, f) for f in fs]
    size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    return int(size * GEOM_RAM_FACTOR)


def grid_shape(axis, res_mm):
    """(H, W) of the interpolation grid for an axis at a resolution."""
    h_min, h_max, v_min, v_max = cfg.CAR_BOUNDS_2D[axis]
    px = 1000.0 / res_mm
    return int((v_max - v_min) * px) + 1, int((h_max - h_min) * px) + 1


def _fig_pixels(axis, fig_height, dpi):
    h_min, h_max, v_min, v_max = cfg.CAR_BOUNDS_2D[axis]
    aspect = (h_max - h_min) / max(v_max - v_min, 1e-6)
    fig_w = min(max(fig_height * aspect + 1.8, 5.0), 24.0)
    return fig_w * fig_height * (dpi or cfg.DPI) ** 2


def plane_ram_bytes(axis, res_mm, n_scalars=1, fig_height=7.0, dpi=None,
                    n_pts=None):
    """
    Peak working-set estimate for processing one plane.

    n_pts : CFD points on the plane, or None to assume a typical density.
    """
    H, W = grid_shape(axis, res_mm)
    n_px = H * W
    if n_pts is None:
        n_pts = n_px * (res_mm ** 2) * ASSUMED_PTS_PER_MM2   # plane area in mm^2
    grid_stage = (n_px * (BYTES_PER_PX + BYTES_PER_PX_SCALAR * max(1, n_scalars))
                  + _fig_pixels(axis, fig_height, dpi) * FIG_BYTES_PER_PX)
    interp_stage = n_pts * BYTES_PER_CFD_PT + n_px * BYTES_PER_PX_INTERP
    b = max(grid_stage, interp_stage)
    return int(b)


_AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}


def plan(ram_gb, axes, res_mm, n_scalars=1, fig_height=7.0, dpi=None,
         geom_case=None, n_pts=None, n_planes=None):
    """
    Turn a RAM budget into a number of planes processed at once.

    Returns dict:
      recruits        planes processed at once (>= 1)
      per_recruit_gb  estimated RAM per recruit (base + geometry + largest plane)
      plane_gb        largest single-plane working set
      geom_gb         one copy of the geometry
      used_gb         recruits * per_recruit_gb
      limit           'ram' | 'cpu' | 'planes' | 'none'
      cpu_max         CPU cap
    """
    axes_idx = [_AXIS_IDX.get(a, a) for a in (axes or ['X', 'Y', 'Z'])]
    plane_b = max(plane_ram_bytes(a, res_mm, n_scalars, fig_height, dpi,
                                  n_pts=n_pts) for a in axes_idx)
    geom_b  = geometry_ram_bytes(geom_case)
    per_b   = cfg.RECRUIT_BASE_MB * MB + geom_b + plane_b

    cpu_max = cpu_limit()
    if ram_gb is None or ram_gb <= 0:
        n_ram = 1
    else:
        n_ram = max(1, int(ram_gb * GB // per_b))

    n, limit = n_ram, 'ram'
    if n > cpu_max:
        n, limit = cpu_max, 'cpu'
    if n_planes is not None and n > max(1, n_planes):
        n, limit = max(1, n_planes), 'planes'
    if ram_gb is None or ram_gb <= 0:
        limit = 'none'

    return dict(recruits=int(n),
                per_recruit_gb=per_b / GB,
                plane_gb=plane_b / GB,
                geom_gb=geom_b / GB,
                used_gb=n * per_b / GB,
                limit=limit,
                cpu_max=cpu_max)
