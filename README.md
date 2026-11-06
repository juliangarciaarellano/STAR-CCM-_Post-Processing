# UT26 CFD Post-Processing Pipeline
**Author:** Julian G-A  
**Team:** University of Toronto Formula Racing (UTFR)  
**Vehicle:** UT26 Formula SAE  
**Recruits (parallel planes):** original beta by George Wei

A desktop application and Python pipeline for post-processing STAR-CCM+ aerodynamic simulation data exported in EnSight Gold format.

---

## Features

### Slice Pipeline
Interpolates volumetric CFD scalar fields onto regular 2D grids across X, Y, and Z planes. Each plane is masked against the geometry to correctly handle solid regions, multi-element wings, and hollow surfaces using a CFD-point-veto flood fill. Outputs PNG images and NPZ data files.

**Scalars supported:** Cp, CpT, Vi, Vj, Vk (and any custom array via auto-detection, e.g. VorticityMagnitude)

### Recruits — several planes at once
*Original beta by George Wei.*

The Run tab has a **RAM for recruits** slider. Each recruit is a separate worker process that holds its own copy of the geometry and processes one plane; the slider sets the total RAM they may use and the tab shows, live, how many planes will run at once:

```
2 planes at once   (3.6 of 4.0 GB)
~1.79 GB per recruit = 0.29 base + 0.00 geometry + 1.49 largest plane (1 mm, 5 scalars)
System RAM 32.0 GB · 21.4 GB free now · 16 CPU threads
```

- **0 GB = off** — one plane at a time (the original behaviour).
- The count is capped by CPU threads and by the number of planes; the label says when.
- The estimate is calibrated on measured peak memory (grid size × scalars, CFD points per plane, figure size, geometry size) and on the UT26 STAR export (≤0.05 CFD points/mm² per plane; geometry ≈2.3× its `.geo` size in RAM). On that export, 60 planes × 2 scalars at 2 mm ran in 58 s with 2 recruits. Once the data is loaded the pipeline re-checks with the real CFD point counts and lowers the count if the planes are denser than assumed — the Log tab reports the final number.
- Output is bit-identical to a one-at-a-time run.
- If a recruit runs out of memory the run stops with a clear message: move the slider down.

CLI: `python cli.py run --ram 12` (budget) or `--recruits 4` (exact count).

### Comparison Pipeline
Computes plane-by-plane deltas between two simulation runs. Delta images use a three-colour geometry overlay:
- **Gray** — geometry present in both runs
- **Magenta** — new geometry (Run B only)
- **Brown** — removed geometry (Run A only)

**Default delta range:** fixed **±0.5** for every scalar, **±500 for vorticity** (recognised by name: any key or array containing `Vorticity`, e.g. `VorticityMagnitude`; on UT26 it runs ~100 median to ~70,000 1/s at p99). Each scalar can be switched to *auto* (±max|diff| per plane) or given its own value; *Reset ranges to defaults* restores them. The surface delta render uses the same defaults.

Optionally parses STAR-CCM+ `.log` files to generate an aerodynamic balance delta report.

### Surface Rendering — full nodes
Renders the **CFD surface mesh itself**: every CFD surface cell with its exact value. No nearest-cell mapping onto the CAD geometry, no decimation, no cell→point averaging. The geometry `.case` only supplies part names (and a grey fallback for parts without surface data). Produces:
- Full-car renders across 8 views (bottom, top, iso front/rear, side L/R, front, rear)
- Per-component zoomed renders with auto-fit framing
- Front wing + EPs and Rear wing + EPs merged into single parts (name match is case-insensitive, e.g. `Front Wing EPs`)

### Surface Cp along any slice (Surface Cp tab)
Exports the classic Cp-vs-position plot for any X, Y or Z slice through the car:
- **Top panel:** Cp vs horizontal position, one colour per part, Cp axis inverted (suction up).
- **Bottom panel:** the section itself, coloured by Cp with the STAR colour scale, drawn at true scale whenever it fits.
- **CSV** with every point: part, loop, order, x/y/z, value and the CFD segment end points.

The CFD surface mesh is cut **exactly** at the plane. Each point is the value of a real CFD surface cell (one per intersected cell, at the segment midpoint) — or the exact edge intersection for node-based exports. Nothing is resampled, gridded or smoothed.

Positions accept lists and ranges in mm: `0, 150, 300` or `0:600:50`. Parts can be ticked on/off; horizontal limits zoom onto one component.

CLI: `python cli.py surfcp Surface.case --axis Y --pos 0:600:150 --out out/`

### Interactive 3D Export
Exports a self-contained `.html` file using Plotly at **full resolution** — every CFD cell, coloured by its own value (`intensitymode='cell'`). Opens in any browser. Large cars give large files (size is printed in the log).

### 3D File Export
- **VTP / VTK** — preserves scalar data, opens in ParaView
- **STL / PLY / OBJ** — geometry only, opens in Fusion360, Meshmixer, Windows 3D Viewer

---

## Colormaps

### Cp — STAR-CCM+ legend
Matches the STAR-CCM+ *Mean of Pressure Coefficient* legend, sampled pixel-by-pixel from the reference colour bar (29 control points, 1024-level LUT; mean error ≈1.6/255). Used for both the Cp planes and the surface plots. Colour bars use STAR-style ticks: `< -4, -2.75, -1.5, -0.25, > 1`.

| Cp | Colour |
|---|---|
| < -4 | Deep indigo |
| -3 | Royal blue |
| -2.4 | Sky blue |
| -1.8 | Cyan / teal |
| -1.0 | Bright green |
| -0.1 | Dark forest green (hard edge) |
| ≈ 0 | Neutral grey |
| +0.1 | Yellow |
| +0.4 | Orange |
| > 1 | Dark red |

For any other vmin/vmax the negative side is stretched over [vmin, 0] and the positive side over [0, vmax], so the grey band always sits at Cp ≈ 0. Editing the Cp range in Settings keeps this palette.

### Cf
Sequential scale with a hard sentinel colour at zero.

| Value | Colour |
|---|---|
| Cf = 0 (separation / stagnation) | Pink (hard step, not blended) |
| Low Cf | Light blue |
| High Cf (attached flow) | Deep blue |

Range: 0 – 0.03

---

## View orientation
Plots use the same axis system as the STAR-CCM+ sim exports. **X-slices (Y–Z plane) are seen from the front, looking rearward along +X: Y points left, Z up.** Set per slice direction in **Settings → View Orientation** (or `FLIP_HORIZONTAL` in `config.py`). This is display only — NPZ grids are stored un-mirrored, so comparisons are unaffected.

---

## Installation

```bash
pip install pyvista numpy scipy matplotlib plotly scikit-image psutil
```

`psutil` is optional (used for the RAM readout; Windows/Linux fall back to the OS directly). `pillow` is needed for `viz_viewer.py`.

Tkinter is included with most Python distributions. If missing:
```bash
# Ubuntu/Debian
sudo apt install python3-tk
```

---

## File Structure

```
cfd_slicer/
├── config.py          # All user settings — paths, scalars, colormaps, bounds,
│                      #   compare defaults, recruits, view orientation
├── geometry.py        # Geometry masking (CFD-point-veto flood fill)
├── interpolation.py   # Per-region griddata interpolation
├── plotting.py        # Slice and delta plot generation, STAR-style colour bars
├── io_utils.py        # Case file loading, NPZ save/load, plane discovery
├── pipeline.py        # Slice pipeline orchestration (+ recruits)
├── recruits.py        # RAM budgeting for parallel planes (original beta: George Wei)
├── compare.py         # Run comparison and delta rendering
├── aero_report.py     # Log file parser and aero balance report
├── surface_render.py  # Full-node surface rendering, part views, HTML and 3D export
├── surface_cp.py      # Surface Cp along slices (plots + CSV)
├── viz_viewer.py      # Standalone PNG browser for parent / new / delta sets
├── cli.py             # Command-line interface
└── gui.py             # Desktop GUI (Tkinter)
```

---

## Output Structure

```
output/
├── PNG/
│   ├── Cp/
│   │   ├── X/   slice_X_001_+100.0mm.png
│   │   ├── Y/
│   │   └── Z/
│   └── CpT/
├── NPZ/
│   ├── X/
│   ├── Y/
│   └── Z/
└── comparison_summary.csv
```

Surface renders:
```
output/
├── PNG/
│   ├── surface_Cp_bottom.png
│   ├── surface_Cp_iso_front.png
│   ├── surface_Cp_Front_wing_bottom.png   ← per-component, auto-zoomed
│   └── ...
├── NPZ/
│   └── ...
└── surface_Cp.html                         ← interactive 3D (full mesh)
```

Surface Cp along slices:
```
output/
└── SurfaceCp/
    └── Cp/
        └── Y/
            ├── surfcp_Cp_Y_+300.0mm.png
            └── surfcp_Cp_Y_+300.0mm.csv
```

---

## Usage

### GUI
```bash
python gui.py
```

Tabs: **Run** · **Compare** · **Surface** · **Surface Cp** · **Settings** · **Log**

### CLI
```bash
# Run slice pipeline
python cli.py run --scalars Cp CpT --axes X Y Z --res 2

# ... with recruits (12 GB budget, or exactly 4 planes at once)
python cli.py run --ram 12
python cli.py run --recruits 4

# Compare two runs (defaults: ±0.5, vorticity ±500)
python cli.py compare DIR_A DIR_B --label-a "Baseline" --label-b "Winglet"
python cli.py compare DIR_A DIR_B --auto-range
python cli.py compare DIR_A DIR_B --range Cp=0.3 CpT=auto

# Surface Cp along Y-slices every 150 mm
python cli.py surfcp Surface.case --axis Y --pos 0:600:150 --out out/

# Inspect a saved plane
python cli.py info path/to/plane.npz

# List all saved planes
python cli.py list output/NPZ --axis Y
```

> Scripts that call `pipeline.run_pipeline(..., ram_budget_gb=...)` directly need an `if __name__ == "__main__":` guard (recruits are spawned processes). `gui.py` and `cli.py` already have one.

---

## Configuration

All settings live in `config.py`. The most commonly changed values:

```python
GEOMETRY_CASE  = r"path/to/Geometry.case"
DATA_CASE      = r"path/to/Raw_Data.case"
OUTPUT_DIR     = r"path/to/output"

RESOLUTION_MM  = 1       # grid resolution (1mm recommended)
DPI            = 150     # output image DPI
DILATION_PX    = 3       # geometry mask dilation (px)

DELTA_RANGE_DEFAULT = 0.5     # compare: fixed ± range
DELTA_RANGE_VORTICITY = 500.0 # compare: vorticity ± range

RECRUIT_RAM_GB      = None    # default recruits budget (None = off)
RECRUIT_BASE_MB     = 300     # per-recruit overhead
RECRUIT_RESERVE_GB  = 2.0     # slider never offers this last bit of RAM

FLIP_HORIZONTAL = {0: True, 1: False, 2: False}   # X-slices: Y points left

CAR_BOUNDS_2D  = {
    0: (-0.90,  0.90, -0.10, 1.30),  # X-slice: h=Y, v=Z
    1: (-1.10,  2.20, -0.10, 1.30),  # Y-slice: h=X, v=Z
    2: (-1.10,  2.20, -0.90, 0.90),  # Z-slice: h=X, v=Y
}
```

New scalar arrays can be added in the Settings tab or via the **Detect from CFD file** button, which reads array names directly from the loaded `.case` file.

---

## Export

### Exporting from STAR-CCM+ (inputs)

Post Pro reads **EnSight Gold** (`.case` + binary `.geo` + one file per variable), single time step, scalars **per element**. Keep every `.case` in the same folder as its `.geo` and variable files — the `.case` refers to them by name.

| Export | Contents | Used by |
|---|---|---|
| `Geometry.case` → `Geometry00000.geo` | Surface mesh, one part per component | Slice masks (Run), part names, surface renders |
| `Raw_Data.case` → `Raw_Data00000.geo` + `Raw_Data00000.<variable>` | Three parts of plane sections: `X_Planes`, `Y_Planes`, `Z_Planes` | Run tab (plane slices) |
| Surface Cp / Cf | `MeanPressureCoeffromVarianceofPressureC`, `MeanSkinFrictionCoeffromVarianceofSkinF` on the surface mesh | Surface + Surface Cp tabs |

**Geometry parts** (UT26 export, 13): Bodywork, Driver, Front wing, Front Wing EPs, Radiators and Fans, Rear wing, Rear wing EPs, Undertray, Wheel LHF, Wheel LHR, Wheel RHF, Wheel RHR, Whiskers. FW + EPs and RW + EPs are merged for surface renders (name match ignores case).

**Plane sections** (UT26 export): 30 X-planes (−1042 … +1958 mm, ~103 mm apart), 18 Y-planes (−870 … +930 mm, ~106 mm), 13 Z-planes (0 … 1300 mm, ~108 mm). Sections can span the whole domain — the pipeline clips them to `CAR_BOUNDS_2D` and detects each part's slice axis from the data, so the part names are only a hint.

**Plane variables** (per element):

| STAR export name | Post Pro scalar | Compare default |
|---|---|---|
| `MeanPressureCoeffromVarianceofPressureC` | Cp | ±0.5 |
| `MeanCpTinRRFfromVarianceofCpTinRRF` | CpT | ±0.5 |
| `VelocityMeanNormalizedi` / `j` / `k` | Vi / Vj / Vk | ±0.5 |
| `VorticityMagnitude` | add with **Settings → Detect from CFD file** (key `VorticityMag`) | ±500 |

STAR shortens field names in the export (e.g. *Mean of Pressure Coefficient from Variance of Pressure Coefficient* → `MeanPressureCoeffromVarianceofPressureC`). The `array` entries in `config.py` must match these names (an exact or partial match is accepted).

**Surface data:** if the surface Cp/Cf variables are exported with the geometry (as in the UT26 `Geometry.case`), use that same `Geometry.case` as the *Surface data .case* in the Surface and Surface Cp tabs. A separate `Surface_XXXXX.case` with the same parts works too. If a `.case` lists variables whose files are missing, the mesh still loads but without data — surfaces render grey and the Log shows `Unable to open file`.

### What Post Pro exports (outputs)

| Output | Path | Format | From |
|---|---|---|---|
| Plane images | `<out>/PNG/<scalar>/<axis>/slice_<axis>_<idx>_<pos>mm.png` | PNG | Run tab · `cli.py run` |
| Plane data | `<out>/NPZ/<axis>/slice_<axis>_<idx>_<pos>mm.npz` | NPZ — `mask`, `extent`, `resolution_mm`, plane metadata, one float32 grid per scalar | Run tab · `cli.py run` |
| Delta images | `<compare out>/PNG/<scalar>/<axis>/diff_<scalar>_<axis>_<idx>_<pos>mm.png` | PNG | Compare tab · `cli.py compare` |
| Delta statistics | `<compare out>/comparison_summary.csv` | CSV — per plane & scalar: n, mean, std, min, max, p05, p50, p95 | Compare tab · `cli.py compare` |
| Aero report | `<compare out>/aero_deltas.txt` | TXT | Compare tab (with both `.log` files) |
| Surface views | `<out>/PNG/surface_<scalar>_<view>.png` (+ `NPZ/` image copies) | PNG + NPZ | Surface tab |
| Per-part views | `<out>/PNG/surface_<scalar>_<Part>_<view>.png` | PNG | Surface tab (ticked parts) |
| Surface delta | `<out>/surface_delta_<scalar>.png` | PNG | Surface tab |
| Surface Cp along slices | `<out>/SurfaceCp/<scalar>/<axis>/surfcp_<scalar>_<axis>_<pos>mm.png` + `.csv` | PNG + CSV — every point: part, loop, order, x/y/z, value, CFD segment end points | Surface Cp tab · `cli.py surfcp` |
| 3D mesh | `<out>/<Part>.<fmt>` or `surface_merged.<fmt>` | VTP / VTK (with scalars), STL / PLY / OBJ (geometry only) | Surface tab |
| Interactive 3D | chosen `.html` file | HTML (Plotly, full mesh) | Surface tab |

Plane, delta and Surface Cp plots follow **Settings → View Orientation**; NPZ and CSV data are never mirrored.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyvista` | Mesh I/O, surface extraction, slicing, offscreen rendering |
| `numpy` | Grid operations, masking, NPZ I/O |
| `scipy` | KD-tree matching, griddata interpolation |
| `scikit-image` | Geometry line rasterising |
| `matplotlib` | Plot generation, colormaps |
| `plotly` | Interactive 3D HTML export |
| `psutil` | RAM readout for recruits (optional) |
| `tkinter` | Desktop GUI |

Python 3.10+ required.

---

## Credits

- **Julian G-A** — CFD post-processing pipeline, GUI
- **George Wei** — original beta of Recruits (parallel plane processing)

---

*Do not distribute without permission.*  
*Julian G-A*