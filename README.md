# UT26 CFD Post-Processing Pipeline
**Author:** Julian G-A  
**Team:** University of Toronto Formula Racing (UTFR)  

A desktop application and Python pipeline for post-processing STAR-CCM+ aerodynamic simulation data exported in EnSight Gold format.

---

## Features

### Slice Pipeline
Interpolates volumetric CFD scalar fields onto regular 2D grids across X, Y, and Z planes. Each plane is masked against the geometry to correctly handle solid regions, multi-element wings, and hollow surfaces using a CFD-point-veto flood fill. Outputs PNG images and NPZ data files.

**Scalars supported:** Cp, CpT, Vi, Vj, Vk (and any custom array via auto-detection)

### Comparison Pipeline
Computes plane-by-plane deltas between two simulation runs. Delta images use a three-colour geometry overlay:
- **Gray** — geometry present in both runs
- **Magenta** — new geometry (Run B only)
- **Brown** — removed geometry (Run A only)

Optionally parses STAR-CCM+ `.log` files to generate an aerodynamic balance delta report.

### Surface Rendering
Maps surface CFD scalar data (Cp, Cf) directly onto the geometry mesh via KD-tree cell matching. Produces:
- Full-car renders across 8 views (bottom, top, iso front/rear, side L/R, front, rear)
- Per-component zoomed renders with auto-fit framing
- Front wing + EPs and Rear wing + EPs merged into single parts

### Interactive 3D Export
Exports a self-contained `.html` file using Plotly. Opens in any browser — no software installation required. Supports rotation, zoom, and per-part toggling via the legend.

### 3D File Export
- **VTP / VTK** — preserves scalar data, opens in ParaView
- **STL / PLY / OBJ** — geometry only, opens in Fusion360, Meshmixer, Windows 3D Viewer

### Visualization Viewer
A standalone Tkinter PNG browser (`viz_viewer.py`) for scrolling through pipeline and compare output folders. Pairs parent, new, and delta slices on (axis, plane index) rather than filename, so mismatched stems still line up. View one set at a time or all three side by side, switch scalar/axis, and navigate with arrow keys, mouse wheel, or PgUp/PgDn. Requires Pillow.

---

## Colormaps

### Cp
Physics-informed diverging scale. Grey is pinned exactly at Cp = 0 regardless of the data range.

| Range | Colour |
|---|---|
| Most negative (suction) | Deep blue |
| Mid suction | Cyan → green |
| Zero | Neutral grey |
| Positive (pressure) | Orange → deep red |

### Cf
Sequential scale with a hard sentinel colour at zero.

| Value | Colour |
|---|---|
| Cf = 0 (separation / stagnation) | Pink (hard step, not blended) |
| Low Cf | Light blue |
| High Cf (attached flow) | Deep blue |

Range: 0 – 0.03

---

## Installation

```bash
pip install pyvista numpy scipy matplotlib plotly pillow scikit-image
```

Tkinter is included with most Python distributions. If missing:
```bash
# Ubuntu/Debian
sudo apt install python3-tk
```

---

## File Structure