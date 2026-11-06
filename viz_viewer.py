"""
================================================================================
  CFD Slicer — Visualization Viewer
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Standalone PNG browser for pipeline / compare output folders.

  Point each field at a full output directory (the one that CONTAINS a `PNG/`
  subfolder). Layout expected (created by pipeline.py / compare.py):

      <output>/PNG/<scalar>/<axis>/<stem>.png

  Stems differ between sets:
      pipeline : slice_<axis>_<idx>_<val>mm.png
      compare  : diff_<scalar>_<axis>_<idx>_<val>mm.png
  So slices are paired on (axis, plane index), NOT on the raw filename.

  Pick a scalar + axis, then scroll through the matching slices. View one set
  at a time or all three together.

  Keys:
      TAB            toggle single-pane  <->  three-pane (all) view
      Up / Down      in single view: change which set is focal (parent/new/delta)
                     in all view:    move the highlight between the three panes
      Left / Right   previous / next slice  (mouse wheel also scrolls slices)
      PgUp / PgDn    jump 10 slices
  Do not distribute without permission.
================================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── style (mirrors gui.py) ───────────────────────────────────────────
PAD, PAD_SM = 8, 4
FONT   = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
FONT_M = ("Consolas", 9)
BG, BG_D = "#f5f5f5", "#e0e0e0"
ACCENT = "#1976D2"
ACCENT_D = "#0D47A1"
BTN_SEC = {"bg": "#757575", "fg": "white", "font": FONT,
           "relief": "flat", "padx": 10, "pady": 5, "cursor": "hand2"}

IMG_EXTS = (".png", ".jpg", ".jpeg")
SETS = ("parent", "new", "delta")
SET_TITLE = {"parent": "PARENT", "new": "NEW", "delta": "DELTA"}

# matches  ..._<axis>_<idx>_<val>mm   pulling out axis + zero-padded index.
# works for both  slice_X_012_+10.0mm  and  diff_Cp_X_012_+10.0mm
_KEY_RE = re.compile(r"_(?P<axis>[XYZxyz])_(?P<idx>\d+)_[-+]?[\d.]+mm$")


def _plane_key(stem, axis_hint):
    """
    Reduce a filename stem to a (axis, plane_index) pairing key so that the
    differently-named parent/new/delta files line up on the same slice.
    Falls back to the raw stem if the pattern is not recognised.
    """
    m = _KEY_RE.search(stem)
    if m:
        return (m.group("axis").upper(), int(m.group("idx")))
    return (axis_hint, stem)


def _png_root(folder):
    """Resolve the PNG/ root whether the user picked the output dir or PNG/."""
    if not folder or not os.path.isdir(folder):
        return None
    if os.path.basename(os.path.normpath(folder)).upper() == "PNG":
        return folder
    cand = os.path.join(folder, "PNG")
    return cand if os.path.isdir(cand) else folder


def _index(folder):
    """
    Walk <root>/PNG/<scalar>/<axis>/*.png.
    Return {scalar: {axis: {plane_key: (sort_idx, abspath)}}}.
    plane_key is (axis, plane_index); sort_idx orders slices numerically.
    """
    root = _png_root(folder)
    tree = {}
    if not root:
        return tree
    for dp, _, files in os.walk(root):
        imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        rel = os.path.relpath(dp, root).split(os.sep)
        if len(rel) >= 2:
            scalar, axis = rel[-2], rel[-1]
        elif len(rel) == 1 and rel[0] != ".":
            scalar, axis = rel[0], "-"
        else:
            scalar, axis = "(root)", "-"
        for f in imgs:
            stem = os.path.splitext(f)[0]
            key = _plane_key(stem, axis)
            sort_idx = key[1] if isinstance(key[1], int) else 1 << 30
            tree.setdefault(scalar, {}).setdefault(axis, {})[key] = (
                sort_idx, os.path.join(dp, f))
    return tree


def _label(key):
    """Human label for a plane key."""
    if isinstance(key[1], int):
        return f"{key[0]}  #{key[1]:03d}"
    return str(key[1])


class Pane(tk.Frame):
    """One titled image panel that scales its image to fit."""
    def __init__(self, parent, title):
        super().__init__(parent, bg=BG, highlightthickness=2,
                         highlightbackground=BG, highlightcolor=BG)
        self.title_lbl = tk.Label(self, text=title, font=FONT_B, bg=ACCENT,
                                  fg="white", anchor="center")
        self.title_lbl.pack(fill="x")
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._raw = None
        self._tk = None
        self._missing = False
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def set_focal(self, on):
        self.configure(highlightbackground=ACCENT_D if on else BG)
        self.title_lbl.configure(bg=ACCENT_D if on else ACCENT)

    def set_path(self, path):
        self._missing = path is not None and not os.path.exists(path)
        if path and os.path.exists(path) and _HAS_PIL:
            try:
                self._raw = Image.open(path)
            except Exception:
                self._raw = None
        else:
            self._raw = None
        self._redraw()

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 2 or h < 2:
            return
        if self._raw is None:
            msg = ("missing" if self._missing else
                   "no image" if _HAS_PIL else "Pillow not installed")
            c.create_text(w // 2, h // 2, text=msg, fill="#888", font=FONT)
            return
        iw, ih = self._raw.size
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = self._raw.resize((nw, nh), Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(img)
        c.create_image(w // 2, h // 2, image=self._tk, anchor="center")


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CFD Slicer — Viewer")
        self.configure(bg=BG)
        self.geometry("1280x820")
        self.minsize(900, 560)

        self.dirs = {s: tk.StringVar() for s in SETS}
        self.trees = {s: {} for s in SETS}
        self.keys = []                            # ordered plane keys
        self.paths = {s: {} for s in SETS}        # plane_key -> abspath
        self.idx = 0
        self.show_all = False                     # single vs three-pane
        self.focal = 0                            # which set is focal (0..2)

        self.v_scalar = tk.StringVar()
        self.v_axis = tk.StringVar()

        self._build_header()
        self._build_paths()
        self._build_nav()
        self._build_panes()
        self._build_statusbar()

        self.bind("<Left>",  lambda e: self._step(-1))
        self.bind("<Right>", lambda e: self._step(1))
        self.bind("<Prior>", lambda e: self._step(-10))
        self.bind("<Next>",  lambda e: self._step(10))
        self.bind("<Up>",    lambda e: (self._focal_step(-1), "break")[1])
        self.bind("<Down>",  lambda e: (self._focal_step(1), "break")[1])
        self.bind("<Tab>",   lambda e: (self._toggle_all(), "break")[1])
        self.bind_all("<MouseWheel>", self._on_wheel)
        self.bind_all("<Button-4>", lambda e: self._step(-1))
        self.bind_all("<Button-5>", lambda e: self._step(1))

    # ── UI ───────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=ACCENT, height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  Visualization Viewer",
                 font=("Segoe UI", 13, "bold"), bg=ACCENT, fg="white")\
            .pack(side="left", padx=PAD)
        tk.Label(hdr, text="TAB single/all · ↑↓ focal · ←→/wheel slice",
                 font=("Segoe UI", 9), bg=ACCENT, fg="#BBDEFB")\
            .pack(side="left", padx=4)

    def _path_row(self, parent, label, key, row):
        tk.Label(parent, text=label, font=FONT, bg=BG, width=8, anchor="w")\
            .grid(row=row, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        tk.Entry(parent, textvariable=self.dirs[key], font=FONT)\
            .grid(row=row, column=1, sticky="ew", padx=(0, PAD), pady=PAD_SM)
        tk.Button(parent, text="...", font=FONT, relief="flat", bg=BG_D,
                  cursor="hand2", command=lambda: self._browse(key))\
            .grid(row=row, column=2, padx=(0, PAD), pady=PAD_SM)

    def _build_paths(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x")
        f.columnconfigure(1, weight=1)
        self._path_row(f, "Parent", "parent", 0)
        self._path_row(f, "New",    "new",    1)
        self._path_row(f, "Delta",  "delta",  2)
        tk.Label(f, text="output folder (contains PNG/)",
                 font=("Segoe UI", 8), bg=BG, fg="#999")\
            .grid(row=3, column=1, sticky="w", padx=(0, PAD))
        tk.Button(f, text="Load", command=self._load, **BTN_SEC)\
            .grid(row=0, column=3, rowspan=3, padx=PAD, pady=PAD_SM, sticky="ns")

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG_D)
        nav.pack(fill="x")
        tk.Label(nav, text="Scalar", font=FONT, bg=BG_D)\
            .pack(side="left", padx=(PAD, PAD_SM))
        self.cb_scalar = ttk.Combobox(nav, state="readonly", width=10,
                                      textvariable=self.v_scalar, font=FONT)
        self.cb_scalar.pack(side="left")
        self.cb_scalar.bind("<<ComboboxSelected>>", lambda e: self._refresh_axis())
        tk.Label(nav, text="Axis", font=FONT, bg=BG_D)\
            .pack(side="left", padx=(PAD, PAD_SM))
        self.cb_axis = ttk.Combobox(nav, state="readonly", width=6,
                                    textvariable=self.v_axis, font=FONT)
        self.cb_axis.pack(side="left")
        self.cb_axis.bind("<<ComboboxSelected>>", lambda e: self._rebuild_keys())

        tk.Button(nav, text="◀", command=lambda: self._step(-1),
                  **BTN_SEC).pack(side="left", padx=(PAD, 0), pady=PAD_SM)
        tk.Button(nav, text="▶", command=lambda: self._step(1),
                  **BTN_SEC).pack(side="left", pady=PAD_SM)
        self.combo = ttk.Combobox(nav, state="readonly", font=FONT_M, width=24)
        self.combo.pack(side="left", padx=PAD, fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._on_combo)

        self.mode_lbl = tk.Label(nav, text="view: PARENT", font=FONT_B,
                                 bg=BG_D, fg=ACCENT)
        self.mode_lbl.pack(side="right", padx=PAD)
        self.counter = tk.Label(nav, text="0 / 0", font=FONT_B, bg=BG_D,
                                fg="#555")
        self.counter.pack(side="right", padx=PAD)

    def _build_panes(self):
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        self.body.rowconfigure(0, weight=1)
        self.panes = {s: Pane(self.body, SET_TITLE[s]) for s in SETS}
        self._layout_panes()

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_D)
        bar.pack(fill="x", side="bottom")
        self.status = tk.Label(bar, text="Set output folders and click Load.",
                               font=FONT, bg=BG_D, fg="#555", anchor="w")
        self.status.pack(side="left", padx=PAD)
        tk.Label(bar, text="TAB single/all · ↑↓ focal · wheel/←→ scroll",
                 font=("Segoe UI", 8), bg=BG_D, fg="#999")\
            .pack(side="right", padx=PAD)

    # ── pane layout ──────────────────────────────────────────────────
    def _layout_panes(self):
        for p in self.panes.values():
            p.grid_forget()
        for i in range(3):
            self.body.columnconfigure(i, weight=0)
        if self.show_all:
            for i, s in enumerate(SETS):
                self.body.columnconfigure(i, weight=1)
                padx = (0, PAD_SM) if i == 0 else \
                       (PAD_SM, 0) if i == 2 else PAD_SM
                self.panes[s].grid(row=0, column=i, sticky="nsew", padx=padx)
        else:
            self.body.columnconfigure(0, weight=1)
            self.panes[SETS[self.focal]].grid(row=0, column=0, sticky="nsew")
        self._update_focus_marks()

    def _update_focus_marks(self):
        for i, s in enumerate(SETS):
            # highlight focal only in all-view; single-view pane needs no border
            self.panes[s].set_focal(self.show_all and i == self.focal)
        cur = SETS[self.focal].upper()
        self.mode_lbl.config(
            text=("view: ALL  ·  focal " + cur) if self.show_all
            else f"view: {cur}")

    def _toggle_all(self):
        self.show_all = not self.show_all
        self._layout_panes()
        self._show()

    def _focal_step(self, d):
        self.focal = (self.focal + d) % 3
        if self.show_all:
            self._update_focus_marks()
        else:
            self._layout_panes()
            self._show()

    # ── logic ────────────────────────────────────────────────────────
    def _browse(self, key):
        p = filedialog.askdirectory()
        if p:
            self.dirs[key].set(p)

    def _load(self):
        if not _HAS_PIL:
            messagebox.showerror("Missing dependency",
                                 "Pillow is required.\n\npip install Pillow")
            return
        for s in SETS:
            self.trees[s] = _index(self.dirs[s].get().strip())
        scalars = sorted({sc for t in self.trees.values() for sc in t})
        if not scalars:
            self.status.config(text="No PNG/<scalar>/<axis> images found.")
            self.cb_scalar["values"] = []
            self.cb_axis["values"] = []
            self.keys = []
            self._show()
            self.counter.config(text="0 / 0")
            return
        self.cb_scalar["values"] = scalars
        if self.v_scalar.get() not in scalars:
            self.v_scalar.set(scalars[0])
        counts = {s: sum(len(a) for t in self.trees[s].values()
                         for a in t.values()) for s in SETS}
        self.status.config(
            text=f"Loaded  parent:{counts['parent']}  new:{counts['new']}  "
                 f"delta:{counts['delta']}")
        self._refresh_axis()

    def _refresh_axis(self):
        sc = self.v_scalar.get()
        axes = sorted({ax for s in SETS for ax in self.trees[s].get(sc, {})})
        self.cb_axis["values"] = axes
        if self.v_axis.get() not in axes:
            self.v_axis.set(axes[0] if axes else "")
        self._rebuild_keys()

    def _rebuild_keys(self):
        sc, ax = self.v_scalar.get(), self.v_axis.get()
        # collect per-set {plane_key: path} and order keys by numeric index
        order = {}
        self.paths = {s: {} for s in SETS}
        for s in SETS:
            for key, (sidx, path) in self.trees[s].get(sc, {}).get(ax, {}).items():
                self.paths[s][key] = path
                order[key] = sidx
        self.keys = sorted(order, key=lambda k: (order[k], str(k)))
        self.idx = 0
        self.combo["values"] = [_label(k) for k in self.keys]
        self._show()

    def _step(self, d):
        if not self.keys:
            return
        self.idx = max(0, min(len(self.keys) - 1, self.idx + d))
        self._show()

    def _on_combo(self, _evt):
        sel = self.combo.current()
        if sel >= 0:
            self.idx = sel
            self._show()

    def _on_wheel(self, event):
        self._step(-1 if event.delta > 0 else 1)

    def _show(self):
        if not self.keys:
            for p in self.panes.values():
                p.set_path(None)
            self.counter.config(text="0 / 0")
            return
        key = self.keys[self.idx]
        for s in SETS:
            self.panes[s].set_path(self.paths[s].get(key))
        self.counter.config(text=f"{self.idx + 1} / {len(self.keys)}")
        if self.combo.current() != self.idx:
            self.combo.current(self.idx)


def main():
    Viewer().mainloop()


if __name__ == "__main__":
    main()
