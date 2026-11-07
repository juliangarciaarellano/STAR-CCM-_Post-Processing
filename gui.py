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

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import multiprocessing
import queue
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import recruits as rc

# ══════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════
PAD    = 8
PAD_SM = 4
FONT   = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
FONT_S = ("Segoe UI", 8)
FONT_M = ("Consolas", 9)
BG     = "#f5f5f5"
BG_D   = "#e0e0e0"
ACCENT = "#1976D2"
OK_FG  = "#2E7D32"
ERR_FG = "#C62828"
WARN_FG = "#E65100"
BTN_RUN = {"bg": "#1976D2", "fg": "white", "font": FONT_B,
            "relief": "flat", "padx": 12, "pady": 6, "cursor": "hand2"}
BTN_SEC = {"bg": "#757575", "fg": "white", "font": FONT,
            "relief": "flat", "padx": 10, "pady": 5, "cursor": "hand2"}


# ══════════════════════════════════════════════════════════════════
#  SCROLLABLE FRAME
#  Wraps any frame in a canvas+scrollbar so tabs can scroll
# ══════════════════════════════════════════════════════════════════

class ScrollableFrame(tk.Frame):
    """
    A frame that adds a vertical scrollbar.
    Put widgets inside self.inner instead of self.
    """
    def __init__(self, parent, bg=BG, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self._canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self.inner = tk.Frame(self._canvas, bg=bg)
        self._window = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._window, width=event.width)

    def _bind_mousewheel(self, event):
        self._canvas.bind_all("<MouseWheel>",    self._on_mousewheel)
        self._canvas.bind_all("<Button-4>",      self._on_mousewheel)
        self._canvas.bind_all("<Button-5>",      self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def browse_file(var, filetypes=None):
    ft = filetypes or [("Case files", "*.case"), ("All files", "*.*")]
    p = filedialog.askopenfilename(filetypes=ft)
    if p: var.set(p)

def browse_log(var):
    p = filedialog.askopenfilename(
        filetypes=[("Log files", "*.log"), ("All files", "*.*")])
    if p: var.set(p)

def browse_dir(var):
    p = filedialog.askdirectory()
    if p: var.set(p)

def labeled_entry(parent, label, var, row, col=0, width=52, browse=None):
    tk.Label(parent, text=label, font=FONT, bg=BG, anchor="w")\
        .grid(row=row, column=col, sticky="w", padx=PAD, pady=PAD_SM)
    e = tk.Entry(parent, textvariable=var, width=width, font=FONT)
    e.grid(row=row, column=col+1, sticky="ew", padx=(0, PAD), pady=PAD_SM)
    if browse:
        tk.Button(parent, text="...", font=FONT, command=browse,
                  relief="flat", bg=BG_D, cursor="hand2")\
            .grid(row=row, column=col+2, padx=(0, PAD), pady=PAD_SM)
    return e

def section_label(parent, text, row, col=0, colspan=3):
    tk.Frame(parent, bg=ACCENT, height=2)\
        .grid(row=row, column=col, columnspan=colspan,
              sticky="ew", padx=PAD, pady=(PAD*2, 0))
    tk.Label(parent, text=text, font=FONT_B, bg=BG, fg=ACCENT)\
        .grid(row=row+1, column=col, sticky="w", padx=PAD, pady=(2, PAD_SM))

def hint(parent, text, row, col=0, colspan=3, fg="#666"):
    tk.Label(parent, text=text, font=FONT_S, bg=BG, fg=fg, justify="left")\
        .grid(row=row, column=col, columnspan=colspan, sticky="w",
              padx=PAD, pady=(0, PAD_SM))

def surface_scalar_cfg(skey):
    """Surface scalars first (Cp, Cf), then volume scalars."""
    return cfg.SURFACE_SCALARS.get(skey) or cfg.SCALARS.get(skey) or {}

def _float_or_none(s):
    s = (s or "").strip()
    return float(s) if s else None


# ══════════════════════════════════════════════════════════════════
#  LOG REDIRECTOR
# ══════════════════════════════════════════════════════════════════

class LogRedirector:
    """
    stdout -> Log tab. Worker threads only queue their text (they never
    touch Tk, so a busy GUI can't slow them down); the main thread inserts
    it from App._pump_ui.
    """
    def __init__(self, widget, original):
        self.widget = widget; self.original = original
        self._q = queue.Queue()
        self._main = threading.main_thread()

    def write(self, text):
        if self.original is not None:
            try: self.original.write(text)
            except Exception: pass
        if threading.current_thread() is self._main:
            self.drain()
            self._insert(text)
        else:
            self._q.put(text)

    def drain(self):
        chunks = []
        try:
            while True:
                chunks.append(self._q.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self._insert(''.join(chunks))

    def _insert(self, text):
        if not text:
            return
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", text)
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except Exception:
            pass

    def flush(self):
        if self.original is not None:
            try: self.original.flush()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════
#  SCALAR EDITOR DIALOG
# ══════════════════════════════════════════════════════════════════

class ScalarEditor(tk.Toplevel):
    def __init__(self, parent, key="", scalar_dict=None, on_save=None):
        super().__init__(parent)
        self.title("Edit Scalar" if key else "Add Scalar")
        self.configure(bg=BG); self.resizable(False, False)
        self.on_save = on_save; self.orig_key = key
        sd = scalar_dict or {}
        self.v_key   = tk.StringVar(value=key)
        self.v_array = tk.StringVar(value=sd.get('array', ''))
        self.v_vmin  = tk.StringVar(value=str(sd.get('vmin', -3.0)))
        self.v_vmax  = tk.StringVar(value=str(sd.get('vmax',  1.0)))
        self.v_label = tk.StringVar(value=sd.get('label', key))
        for r, (lbl, var) in enumerate([
            ("Short name (e.g. Cp)", self.v_key),
            ("Array name in .case file", self.v_array),
            ("Color min (vmin)", self.v_vmin),
            ("Color max (vmax)", self.v_vmax),
            ("Colorbar label", self.v_label),
        ]):
            tk.Label(self, text=lbl, font=FONT, bg=BG, anchor="w")\
                .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
            tk.Entry(self, textvariable=var, width=38, font=FONT)\
                .grid(row=r, column=1, sticky="ew", padx=PAD, pady=PAD_SM)
        bf = tk.Frame(self, bg=BG)
        bf.grid(row=5, column=0, columnspan=2, pady=PAD, padx=PAD, sticky="e")
        tk.Button(bf, text="Cancel", command=self.destroy, **BTN_SEC)\
            .pack(side="right", padx=(PAD_SM, 0))
        tk.Button(bf, text="Save", command=self._save, **BTN_RUN).pack(side="right")

    def _save(self):
        key = self.v_key.get().strip()
        if not key:
            messagebox.showerror("Error", "Short name cannot be empty."); return
        try:
            vmin, vmax = float(self.v_vmin.get()), float(self.v_vmax.get())
        except ValueError:
            messagebox.showerror("Error", "vmin and vmax must be numbers."); return
        if self.on_save:
            self.on_save(dict(key=key, orig_key=self.orig_key,
                              array=self.v_array.get().strip(),
                              vmin=vmin, vmax=vmax,
                              label=self.v_label.get().strip() or key))
        self.destroy()


# ══════════════════════════════════════════════════════════════════
#  TAB: RUN
# ══════════════════════════════════════════════════════════════════

class RunTab(ScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build(self.inner)

    def _build(self, f):
        f.columnconfigure(1, weight=1)
        r = 0

        section_label(f, "Input / Output Paths", r); r += 2
        self.v_geom = tk.StringVar(value=cfg.GEOMETRY_CASE)
        self.v_data = tk.StringVar(value=cfg.DATA_CASE)
        self.v_out  = tk.StringVar(value=cfg.OUTPUT_DIR)
        labeled_entry(f, "Geometry .case", self.v_geom, r,
                      browse=lambda: browse_file(self.v_geom)); r += 1
        labeled_entry(f, "CFD data .case", self.v_data, r,
                      browse=lambda: browse_file(self.v_data)); r += 1
        labeled_entry(f, "Output directory", self.v_out, r,
                      browse=lambda: browse_dir(self.v_out)); r += 1

        section_label(f, "Scalars to process", r); r += 2
        self.scalar_frame = tk.Frame(f, bg=BG)
        self.scalar_frame.grid(row=r, column=0, columnspan=3,
                                sticky="ew", padx=PAD)
        self.scalar_vars = {}
        self._rebuild_scalar_checks()
        r += 1

        section_label(f, "Axes to slice", r); r += 2
        self.axis_vars = {}
        af = tk.Frame(f, bg=BG)
        af.grid(row=r, column=1, sticky="w", padx=PAD)
        for ax in ['X', 'Y', 'Z']:
            v = tk.BooleanVar(value=True); self.axis_vars[ax] = v
            tk.Checkbutton(af, text=ax, variable=v, font=FONT, bg=BG,
                           command=self._update_recruits)\
                .pack(side="left", padx=PAD)
        r += 1

        section_label(f, "Resolution & Image Size", r); r += 2
        tk.Label(f, text="mm per pixel", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_res = tk.StringVar(value=str(cfg.RESOLUTION_MM))
        rf = tk.Frame(f, bg=BG)
        rf.grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM)
        for val in [1, 2, 5]:
            tk.Radiobutton(rf, text=f"{val} mm", variable=self.v_res,
                           value=str(val), font=FONT, bg=BG)\
                .pack(side="left", padx=PAD)
        tk.Label(rf, text="custom:", font=FONT, bg=BG)\
            .pack(side="left", padx=(PAD*2, PAD_SM))
        tk.Entry(rf, textvariable=self.v_res, width=6, font=FONT)\
            .pack(side="left")
        tk.Label(rf, text="mm", font=FONT, bg=BG).pack(side="left", padx=PAD_SM)
        r += 1

        tk.Label(f, text="Figure height (in)", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_figh = tk.StringVar(value="7.0")
        tk.Entry(f, textvariable=self.v_figh, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        tk.Label(f, text="DPI", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_dpi = tk.StringVar(value=str(cfg.DPI))
        tk.Entry(f, textvariable=self.v_dpi, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        # ── Recruits ──────────────────────────────────────────────
        section_label(f, "Recruits  (process several planes at once)", r); r += 2
        hint(f, ("Each recruit is a separate worker that holds its own copy of the "
                 "geometry and works on one plane.\nGive them more RAM and more planes "
                 "run at the same time. 0 GB = off (one plane at a time)."), r); r += 1
        hint(f, "Recruits — original beta by George Wei.", r, fg="#888"); r += 1

        total_b, _ = rc.system_ram()
        max_gb = 64.0
        if total_b:
            max_gb = max(1.0, round((total_b / rc.GB - cfg.RECRUIT_RESERVE_GB) * 2) / 2)
        self.v_ram = tk.DoubleVar(value=float(cfg.RECRUIT_RAM_GB or 0.0))
        tk.Label(f, text="RAM for recruits", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        sf = tk.Frame(f, bg=BG)
        sf.grid(row=r, column=1, columnspan=2, sticky="ew", padx=PAD)
        sf.columnconfigure(0, weight=1)
        self.ram_scale = tk.Scale(sf, from_=0.0, to=max_gb, resolution=0.5,
                                  orient="horizontal", variable=self.v_ram,
                                  showvalue=False, bg=BG, highlightthickness=0,
                                  troughcolor=BG_D, activebackground=ACCENT,
                                  sliderlength=18, length=320,
                                  command=lambda _v: self._update_recruits())
        self.ram_scale.grid(row=0, column=0, sticky="ew")
        self.ram_val_lbl = tk.Label(sf, text="", font=FONT_B, bg=BG, width=9,
                                    anchor="e")
        self.ram_val_lbl.grid(row=0, column=1, padx=(PAD_SM, 0))
        r += 1

        self.rec_lbl = tk.Label(f, text="", font=FONT_B, bg=BG, fg=ACCENT,
                                anchor="w", justify="left")
        self.rec_lbl.grid(row=r, column=1, columnspan=2, sticky="w", padx=PAD)
        r += 1
        self.rec_detail = tk.Label(f, text="", font=FONT_S, bg=BG, fg="#666",
                                   anchor="w", justify="left")
        self.rec_detail.grid(row=r, column=1, columnspan=2, sticky="w",
                             padx=PAD, pady=(0, PAD_SM))
        r += 1

        for var in (self.v_res, self.v_figh, self.v_dpi, self.v_geom):
            var.trace_add("write", lambda *_: self._update_recruits())

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3, pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf, text="Run Pipeline", command=self._run,
                  **BTN_RUN).pack(side="right")
        self.status_lbl = tk.Label(bf, text="", font=FONT, bg=BG, fg="#555")
        self.status_lbl.pack(side="right", padx=PAD)

        self._update_recruits()

    def _rebuild_scalar_checks(self):
        for w in self.scalar_frame.winfo_children(): w.destroy()
        self.scalar_vars = {}
        for skey in cfg.SCALARS:
            v = tk.BooleanVar(value=True); self.scalar_vars[skey] = v
            tk.Checkbutton(self.scalar_frame, text=skey, variable=v,
                           font=FONT, bg=BG, command=self._update_recruits)\
                .pack(side="left", padx=PAD)

    def refresh_scalars(self):
        self._rebuild_scalar_checks()
        self._update_recruits()

    # ── recruits estimate (live) ─────────────────────────────────
    def _ram_budget(self):
        gb = float(self.v_ram.get() or 0.0)
        return gb if gb > 0 else None

    def _update_recruits(self):
        if not hasattr(self, 'rec_lbl'):
            return
        gb = float(self.v_ram.get() or 0.0)
        self.ram_val_lbl.config(text="off" if gb <= 0 else f"{gb:.1f} GB")
        try:
            res  = float(self.v_res.get())
            figh = float(self.v_figh.get())
            dpi  = int(self.v_dpi.get())
            assert res > 0
        except Exception:
            self.rec_lbl.config(text="(enter a valid resolution / size / DPI)",
                                fg="#888")
            self.rec_detail.config(text="")
            return
        axes = [k for k, v in self.axis_vars.items() if v.get()] or ['X', 'Y', 'Z']
        n_sc = max(1, sum(1 for v in self.scalar_vars.values() if v.get()))
        p = rc.plan(gb if gb > 0 else None, axes, res, n_sc, figh, dpi,
                    self.v_geom.get().strip())

        total_b, avail_b = rc.system_ram()
        sysinfo = []
        if total_b:
            sysinfo.append(f"System RAM {total_b/rc.GB:.1f} GB")
        if avail_b:
            sysinfo.append(f"{avail_b/rc.GB:.1f} GB free now")
        sysinfo.append(f"{p['cpu_max']} CPU threads")
        per = p['per_recruit_gb']
        detail = (f"~{per:.2f} GB per recruit = {cfg.RECRUIT_BASE_MB/1024:.2f} base"
                  f" + {p['geom_gb']:.2f} geometry + {p['plane_gb']:.2f} largest plane"
                  f" ({res:g} mm, {n_sc} scalar{'s' if n_sc != 1 else ''})"
                  f"\n" + "  ·  ".join(sysinfo))

        if gb <= 0:
            self.rec_lbl.config(text="Off — 1 plane at a time", fg="#555")
        elif gb < per:
            self.rec_lbl.config(
                text=f"1 plane at a time — one recruit needs ~{per:.2f} GB",
                fg=WARN_FG)
        else:
            n = p['recruits']
            txt = (f"{n} plane{'s' if n != 1 else ''} at once   "
                   f"({p['used_gb']:.1f} of {gb:.1f} GB)")
            if p['limit'] == 'cpu':
                txt += f"   — capped by CPU ({p['cpu_max']} threads)"
            colour = ACCENT
            if avail_b and gb * rc.GB > avail_b:
                txt += "   ⚠ more than free RAM"
                colour = WARN_FG
            self.rec_lbl.config(text=txt, fg=colour)
        self.rec_detail.config(text=detail)

    def _run(self):
        scalars = [k for k, v in self.scalar_vars.items() if v.get()]
        axes    = [k for k, v in self.axis_vars.items()   if v.get()]
        if not scalars:
            messagebox.showwarning("Nothing to do", "Select at least one scalar."); return
        if not axes:
            messagebox.showwarning("Nothing to do", "Select at least one axis."); return
        try:
            res = float(self.v_res.get()); assert res > 0
        except Exception:
            messagebox.showerror("Invalid resolution",
                                 "mm per pixel must be a positive number."); return
        try:
            figh = float(self.v_figh.get()); dpi = int(self.v_dpi.get())
        except Exception:
            messagebox.showerror("Invalid size",
                                 "Figure height and DPI must be numbers."); return

        cfg.GEOMETRY_CASE = self.v_geom.get()
        cfg.DATA_CASE     = self.v_data.get()
        cfg.OUTPUT_DIR    = self.v_out.get()
        cfg.DPI           = dpi
        ram_gb = self._ram_budget()

        self.status_lbl.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import pipeline
                pipeline.run_pipeline(scalars=scalars, axes=axes,
                                       resolution_mm=res,
                                       fig_height=figh, dpi=dpi,
                                       ram_budget_gb=ram_gb)
                self.status_lbl.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_lbl.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
#  TAB: COMPARE
# ══════════════════════════════════════════════════════════════════

class CompareTab(ScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build(self.inner)

    def _build(self, f):
        f.columnconfigure(1, weight=1)
        r = 0

        section_label(f, "Run Directories", r); r += 2
        self.v_dir_a   = tk.StringVar()
        self.v_dir_b   = tk.StringVar()
        self.v_out     = tk.StringVar()
        self.v_label_a = tk.StringVar(value="Run A")
        self.v_label_b = tk.StringVar(value="Run B")
        labeled_entry(f, "Run A (baseline)", self.v_dir_a, r,
                      browse=lambda: browse_dir(self.v_dir_a)); r += 1
        labeled_entry(f, "Run B (new)",      self.v_dir_b, r,
                      browse=lambda: browse_dir(self.v_dir_b)); r += 1
        labeled_entry(f, "Output directory", self.v_out,   r,
                      browse=lambda: browse_dir(self.v_out));   r += 1
        labeled_entry(f, "Label for Run A",  self.v_label_a, r, width=28); r += 1
        labeled_entry(f, "Label for Run B",  self.v_label_b, r, width=28); r += 1

        section_label(f, "Scalars & Axes", r); r += 2
        self.scalar_frame = tk.Frame(f, bg=BG)
        self.scalar_frame.grid(row=r, column=0, columnspan=3,
                                sticky="ew", padx=PAD)
        self.scalar_vars = {}
        self._rebuild_scalar_checks()
        r += 1

        self.axis_vars = {}
        af = tk.Frame(f, bg=BG)
        af.grid(row=r, column=1, sticky="w", padx=PAD)
        for ax in ['X', 'Y', 'Z']:
            v = tk.BooleanVar(value=True); self.axis_vars[ax] = v
            tk.Checkbutton(af, text=ax, variable=v, font=FONT, bg=BG)\
                .pack(side="left", padx=PAD)
        r += 1

        section_label(f, "Delta Colormap Range", r); r += 2
        hint(f, (f"Fixed: same symmetric range for every plane of that scalar  "
                 f"(defaults: ±{cfg.DELTA_RANGE_DEFAULT:g}, "
                 f"vorticity ±{cfg.DELTA_RANGE_VORTICITY:g})  |  "
                 f"Auto: ±max(|diff|) per plane"), r); r += 1
        self.range_frame = tk.Frame(f, bg=BG)
        self.range_frame.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD)
        self.range_vars = {}
        self._rebuild_range_rows()
        r += 1
        tk.Button(f, text="Reset ranges to defaults", font=FONT, bg=BG_D,
                  relief="flat", cursor="hand2",
                  command=self._reset_ranges)\
            .grid(row=r, column=0, columnspan=2, sticky="w", padx=PAD,
                  pady=(PAD_SM, 0))
        r += 1

        section_label(f, "Delta Image Size", r); r += 2
        tk.Label(f, text="Figure height (in)", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_figh = tk.StringVar(value="7.0")
        tk.Entry(f, textvariable=self.v_figh, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1
        tk.Label(f, text="DPI", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_dpi = tk.StringVar(value=str(cfg.DPI))
        tk.Entry(f, textvariable=self.v_dpi, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        section_label(f, "Aero Report  (optional)", r); r += 2
        self.v_aero = tk.BooleanVar(value=False)
        tk.Checkbutton(f,
                       text="Generate aero delta report from .log files",
                       variable=self.v_aero, font=FONT, bg=BG,
                       command=self._toggle_aero)\
            .grid(row=r, column=0, columnspan=2, sticky="w", padx=PAD); r += 1
        self.aero_frame = tk.Frame(f, bg=BG)
        self.aero_frame.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self.v_log_a = tk.StringVar(); self.v_log_b = tk.StringVar()
        labeled_entry(self.aero_frame, "Log file -- Run A", self.v_log_a, 0,
                      browse=lambda: browse_log(self.v_log_a))
        labeled_entry(self.aero_frame, "Log file -- Run B", self.v_log_b, 1,
                      browse=lambda: browse_log(self.v_log_b))
        self.aero_frame.grid_remove()

        tk.Label(f,
                 text=("Geometry key:  Gray = both  |  "
                       "Magenta = new (Run B only)  |  Brown = removed (Run A only)"),
                 font=FONT_S, bg=BG, fg="#555", justify="left")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3,
                pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf, text="Run Comparison", command=self._run,
                  **BTN_RUN).pack(side="right")
        self.status_lbl = tk.Label(bf, text="", font=FONT, bg=BG, fg="#555")
        self.status_lbl.pack(side="right", padx=PAD)

    def _rebuild_scalar_checks(self):
        for w in self.scalar_frame.winfo_children(): w.destroy()
        self.scalar_vars = {}
        for skey in cfg.SCALARS:
            v = tk.BooleanVar(value=True); self.scalar_vars[skey] = v
            tk.Checkbutton(self.scalar_frame, text=skey, variable=v,
                           font=FONT, bg=BG).pack(side="left", padx=PAD)

    def _rebuild_range_rows(self):
        """Rows live in their own frame, so adding scalars never overlaps."""
        old = {k: (m.get(), v.get()) for k, (m, v, _) in self.range_vars.items()}
        for w in self.range_frame.winfo_children():
            w.destroy()
        self.range_vars = {}
        for col, hdr in enumerate(["Scalar", "Mode", "Fixed ± value"]):
            tk.Label(self.range_frame, text=hdr, font=FONT_B, bg=BG_D)\
                .grid(row=0, column=col, sticky="ew",
                      padx=1, pady=1, ipadx=6, ipady=3)
        for r, skey in enumerate(cfg.SCALARS, start=1):
            mode0, val0 = old.get(skey, ("fixed",
                                         f"{cfg.default_delta_range(skey):g}"))
            mode_var  = tk.StringVar(value=mode0)
            value_var = tk.StringVar(value=val0)
            tk.Label(self.range_frame, text=skey, font=FONT, bg=BG, anchor="w")\
                .grid(row=r, column=0, sticky="w", padx=PAD, pady=2)
            mf = tk.Frame(self.range_frame, bg=BG)
            mf.grid(row=r, column=1, sticky="w", padx=PAD)
            for mode in ["fixed", "auto"]:
                tk.Radiobutton(mf, text=mode, variable=mode_var,
                               value=mode, font=FONT, bg=BG,
                               command=lambda sk=skey: self._toggle_range(sk))\
                    .pack(side="left", padx=PAD_SM)
            entry = tk.Entry(self.range_frame, textvariable=value_var, width=8,
                             font=FONT)
            entry.grid(row=r, column=2, padx=PAD, pady=2, sticky="w")
            self.range_vars[skey] = (mode_var, value_var, entry)
            self._toggle_range(skey)

    def _toggle_range(self, skey):
        mode_var, _, entry = self.range_vars[skey]
        if mode_var.get() == "fixed":
            entry.config(state="normal", bg="white")
        else:
            entry.config(state="disabled", bg=BG_D)

    def _reset_ranges(self):
        for skey, (mode_var, value_var, _) in self.range_vars.items():
            mode_var.set("fixed")
            value_var.set(f"{cfg.default_delta_range(skey):g}")
            self._toggle_range(skey)

    def refresh_scalars(self):
        self._rebuild_scalar_checks()
        self._rebuild_range_rows()

    def _toggle_aero(self):
        if self.v_aero.get(): self.aero_frame.grid()
        else: self.aero_frame.grid_remove()

    def _get_delta_ranges(self):
        result = {}
        for skey, (mode_var, value_var, _) in self.range_vars.items():
            if mode_var.get() == "fixed":
                try:
                    val = float(value_var.get())
                    assert val > 0
                    result[skey] = val
                except (ValueError, AssertionError):
                    messagebox.showerror("Invalid range",
                        f"Fixed range for '{skey}' must be a positive number.")
                    return None
            else:
                result[skey] = None
        return result

    def _run(self):
        dir_a = self.v_dir_a.get().strip()
        dir_b = self.v_dir_b.get().strip()
        if not dir_a or not dir_b:
            messagebox.showwarning("Missing paths", "Set both run directories."); return
        scalars = [k for k, v in self.scalar_vars.items() if v.get()]
        axes    = [k for k, v in self.axis_vars.items()   if v.get()]
        try:
            figh = float(self.v_figh.get()); dpi = int(self.v_dpi.get())
        except Exception:
            messagebox.showerror("Invalid size",
                                 "Figure height and DPI must be numbers."); return
        delta_ranges = self._get_delta_ranges()
        if delta_ranges is None: return
        log_a = self.v_log_a.get().strip() or None
        log_b = self.v_log_b.get().strip() or None
        if self.v_aero.get() and (not log_a or not log_b):
            messagebox.showwarning("Missing log files",
                "Provide both log files or uncheck 'Generate aero report'.")
            return
        out_dir = self.v_out.get().strip() or None
        self.status_lbl.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import compare as cmp
                cmp.compare_runs(
                    dir_a=dir_a, dir_b=dir_b,
                    scalars=scalars or None, axes=axes or None,
                    output_dir=out_dir,
                    label_a=self.v_label_a.get() or "Run A",
                    label_b=self.v_label_b.get() or "Run B",
                    log_a=log_a if self.v_aero.get() else None,
                    log_b=log_b if self.v_aero.get() else None,
                    fig_height=figh, dpi=dpi,
                    delta_ranges=delta_ranges,
                )
                self.status_lbl.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_lbl.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
#  TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════

class SettingsTab(ScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build(self.inner)

    def _build(self, f):
        f.columnconfigure(1, weight=1)
        r = 0

        section_label(f, "Scalars", r); r += 2
        tk.Label(f, text=(
            "Array name must match what pyvista reports in the Log tab\n"
            "(look for 'Block X arrays: [...]' after loading CFD data).\n"
            "Use 'Detect from CFD file' to add missing scalars automatically."),
            font=FONT, bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

        self.scalar_list_frame = tk.Frame(f, bg=BG)
        self.scalar_list_frame.grid(row=r, column=0, columnspan=3,
                                     sticky="ew", padx=PAD); r += 1
        self._refresh_scalar_list()

        br = tk.Frame(f, bg=BG)
        br.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD, pady=PAD_SM)
        tk.Button(br, text="+ Add Scalar", command=self._add_scalar,
                  **BTN_SEC).pack(side="left")
        tk.Button(br, text="Detect from CFD file",
                  command=self._detect_scalars,
                  bg="#1B5E20", fg="white", font=FONT,
                  relief="flat", padx=10, pady=5, cursor="hand2")\
            .pack(side="left", padx=(PAD, 0))
        self.detect_lbl = tk.Label(br, text="", font=FONT, bg=BG, fg="#555")
        self.detect_lbl.pack(side="left", padx=PAD)
        r += 1

        section_label(f, "Domain Bounds  (metres)", r); r += 2
        tk.Label(f,
                 text="h = horizontal axis, v = vertical axis per slice direction.",
                 font=FONT, bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1
        self.bounds_vars = {}
        for col, h in enumerate(["Axis", "h min", "h max", "v min", "v max"]):
            tk.Label(f, text=h, font=FONT_B, bg=BG_D)\
                .grid(row=r, column=col, sticky="ew",
                      padx=1, pady=1, ipadx=6, ipady=3)
        r += 1
        ax_names = {0: "X-slice (h=Y, v=Z)",
                    1: "Y-slice (h=X, v=Z)",
                    2: "Z-slice (h=X, v=Y)"}
        for ax in range(3):
            row_vars = []
            tk.Label(f, text=ax_names[ax], font=FONT, bg=BG)\
                .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
            for col, val in enumerate(cfg.CAR_BOUNDS_2D[ax]):
                v = tk.StringVar(value=str(val)); row_vars.append(v)
                tk.Entry(f, textvariable=v, width=8, font=FONT)\
                    .grid(row=r, column=col+1, padx=PAD_SM, pady=PAD_SM)
            self.bounds_vars[ax] = row_vars; r += 1

        section_label(f, "View Orientation  (match the sim exports)", r); r += 2
        hint(f, ("Mirror the horizontal plot axis per slice direction (display only — "
                 "NPZ data is unchanged).\nDefault: X-slices are seen from the front "
                 "looking rearward, so Y points left and Z up, like the STAR triad."),
             r); r += 1
        self.flip_vars = {}
        ff = tk.Frame(f, bg=BG)
        ff.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD)
        for ax, txt in [(0, "X-slices: Y points left"),
                        (1, "Y-slices: X points left"),
                        (2, "Z-slices: X points left")]:
            v = tk.BooleanVar(value=bool(cfg.FLIP_HORIZONTAL.get(ax, False)))
            self.flip_vars[ax] = v
            tk.Checkbutton(ff, text=txt, variable=v, font=FONT, bg=BG)\
                .pack(side="left", padx=(0, PAD * 2))
        r += 1

        section_label(f, "Plot Defaults", r); r += 2
        self.v_dpi   = tk.StringVar(value=str(cfg.DPI))
        self.v_fill  = tk.StringVar(value=cfg.FILL_COLOR)
        self.v_tick  = tk.StringVar(value=str(cfg.TICK_SPACING_MM))
        self.v_dilat = tk.StringVar(value=str(cfg.DILATION_PX))
        labeled_entry(f, "DPI",                    self.v_dpi,   r, width=8); r += 1
        labeled_entry(f, "Geometry fill color",    self.v_fill,  r, width=8); r += 1
        labeled_entry(f, "Tick spacing (mm)",      self.v_tick,  r, width=8); r += 1
        labeled_entry(f, "Geometry dilation (px)", self.v_dilat, r, width=8)
        tk.Label(f, text="increase to close gaps, decrease to reduce bloat",
                 font=FONT_S, bg=BG, fg="#888")\
            .grid(row=r, column=2, sticky="w", padx=PAD_SM)
        r += 1

        section_label(f, "Aero Report Settings", r); r += 2
        self.v_aero_thresh  = tk.StringVar(value="0.01")
        self.v_aero_target  = tk.StringVar(value="0.45")
        self.v_aero_failure = tk.StringVar(value="500")
        labeled_entry(f, "Uncertainty threshold", self.v_aero_thresh,  r, width=8); r += 1
        labeled_entry(f, "Aerobalance target",    self.v_aero_target,  r, width=8); r += 1
        labeled_entry(f, "Failure iterations",    self.v_aero_failure, r, width=8); r += 1

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3, pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf, text="Apply Settings", command=self._apply,
                  **BTN_RUN).pack(side="right")
        self.status_lbl = tk.Label(bf, text="", font=FONT, bg=BG, fg="#555")
        self.status_lbl.pack(side="right", padx=PAD)

    def _refresh_scalar_list(self):
        for w in self.scalar_list_frame.winfo_children(): w.destroy()
        for col, h in enumerate(["Key", "Array name", "vmin", "vmax", "Label", ""]):
            tk.Label(self.scalar_list_frame, text=h, font=FONT_B, bg=BG_D)\
                .grid(row=0, column=col, sticky="ew",
                      padx=1, pady=1, ipadx=6, ipady=3)
        for row, (skey, sd) in enumerate(cfg.SCALARS.items(), start=1):
            for col, val in enumerate([skey, sd['array'],
                                        str(sd['vmin']), str(sd['vmax']),
                                        sd['label']]):
                tk.Label(self.scalar_list_frame, text=val,
                         font=FONT, bg=BG, anchor="w")\
                    .grid(row=row, column=col, sticky="w", padx=PAD_SM, pady=2)
            bf = tk.Frame(self.scalar_list_frame, bg=BG)
            bf.grid(row=row, column=5, padx=PAD_SM)
            tk.Button(bf, text="Edit",
                      command=lambda k=skey: self._edit_scalar(k),
                      font=FONT, bg=BG_D, relief="flat", cursor="hand2")\
                .pack(side="left", padx=2)
            tk.Button(bf, text="X",
                      command=lambda k=skey: self._remove_scalar(k),
                      font=FONT, bg="#FFCDD2", relief="flat", cursor="hand2")\
                .pack(side="left", padx=2)

    def _refresh_all_tabs(self):
        self._refresh_scalar_list()
        self.app.run_tab.refresh_scalars()
        self.app.compare_tab.refresh_scalars()
        self.app.surface_tab.refresh_scalars()
        self.app.surface_cp_tab.refresh_scalars()

    @staticmethod
    def _generic_cmap(key, vmin, vmax):
        import matplotlib.colors as mcolors
        mid = (vmin+vmax)/2
        pos = [0.0, (mid-vmin)/(vmax-vmin), 1.0]
        cdict = {
            'red':   [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.8,0.8)],
            'green': [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
            'blue':  [(pos[0],0.8,0.8),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
        }
        return mcolors.LinearSegmentedColormap(f"auto_{key}", cdict, N=256)

    def _detect_scalars(self):
        data_path = self.app.run_tab.v_data.get().strip()
        if not data_path or not os.path.exists(data_path):
            messagebox.showwarning("No file",
                "Set the CFD data .case path in the Run tab first.")
            return
        self.detect_lbl.config(text="Loading...", fg=ACCENT)
        self.update()
        try:
            import pyvista as pv
            pv.global_theme.allow_empty_mesh = True
            mesh = pv.read(data_path)
            found = []
            for i in range(mesh.n_blocks):
                b = mesh[i]
                if b is not None and b.n_cells > 0:
                    cc = b.cell_centers()
                    for arr in cc.array_names:
                        if arr not in found:
                            found.append(arr)
            seen = set(); unique_arrays = []
            for arr in found:
                if arr not in seen:
                    seen.add(arr); unique_arrays.append(arr)
            existing_arrays = {sd['array'] for sd in cfg.SCALARS.values()}
            added = []
            for arr in unique_arrays:
                already = any(arr == ea or arr in ea or ea in arr
                              for ea in existing_arrays)
                if already: continue
                key = arr
                for suffix in ['fromVariance', 'Mean', 'Normalized',
                                'Coefficient', 'Pressure', 'Velocity']:
                    key = key.replace(suffix, '')
                key = key.strip('_').strip()[:12] or arr[:12]
                base_key = key; n = 2
                while key in cfg.SCALARS:
                    key = f"{base_key}{n}"; n += 1
                arr_lower = arr.lower()
                if 'velocity' in arr_lower:     vmin, vmax = -1.0, 2.0
                elif 'pressure' in arr_lower or 'cp' in arr_lower:
                    vmin, vmax = -3.0, 1.0
                else:                           vmin, vmax = -1.0, 1.0
                cfg.SCALARS[key] = {'array': arr, 'vmin': vmin, 'vmax': vmax,
                                    'cmap': self._generic_cmap(key, vmin, vmax),
                                    'label': key}
                existing_arrays.add(arr); added.append(key)
            self._refresh_all_tabs()
            if added:
                self.detect_lbl.config(text=f"Added: {', '.join(added)}", fg=OK_FG)
            else:
                self.detect_lbl.config(text="All arrays already present", fg="#555")
        except Exception as exc:
            self.detect_lbl.config(text="Error -- see Log tab", fg=ERR_FG)
            print(f"[detect_scalars] {exc}")
            import traceback; traceback.print_exc()

    def _add_scalar(self):
        ScalarEditor(self.inner, on_save=self._on_scalar_saved)

    def _edit_scalar(self, key):
        ScalarEditor(self.inner, key=key, scalar_dict=cfg.SCALARS[key],
                     on_save=self._on_scalar_saved)

    def _remove_scalar(self, key):
        if messagebox.askyesno("Remove", f"Remove scalar '{key}'?"):
            del cfg.SCALARS[key]
            self._refresh_all_tabs()

    def _on_scalar_saved(self, result):
        vmin, vmax = result['vmin'], result['vmax']
        key, orig = result['key'], result['orig_key']
        old = cfg.SCALARS.get(orig) if orig else None

        # Keep physics-informed palettes (e.g. the STAR Cp map) when editing:
        # rebuild them for the new range instead of replacing them.
        factory = cfg.CMAP_FACTORIES.get(orig or key)
        if factory is not None:
            cmap = factory(vmin=vmin, vmax=vmax)
        elif old is not None and old.get('vmin') == vmin and old.get('vmax') == vmax:
            cmap = old['cmap']
        else:
            cmap = self._generic_cmap(key, vmin, vmax)

        if orig and orig != key:
            cfg.SCALARS.pop(orig, None)
        new = {'array': result['array'], 'vmin': vmin, 'vmax': vmax,
               'cmap': cmap, 'label': result['label']}
        cfg.SCALARS[key] = new
        # Cp is shared with the surface tabs — keep them in sync
        for sk, sd in list(cfg.SURFACE_SCALARS.items()):
            if old is not None and sd is old:
                cfg.SURFACE_SCALARS[sk] = new
        self._refresh_all_tabs()

    def _apply(self):
        try:
            for ax in range(3):
                cfg.CAR_BOUNDS_2D[ax] = tuple(
                    float(v.get()) for v in self.bounds_vars[ax])
            cfg.DPI             = int(self.v_dpi.get())
            cfg.FILL_COLOR      = self.v_fill.get().strip()
            cfg.TICK_SPACING_MM = int(self.v_tick.get())
            cfg.DILATION_PX     = int(self.v_dilat.get())
            for ax, v in self.flip_vars.items():
                cfg.FLIP_HORIZONTAL[ax] = bool(v.get())
            import aero_report as ar
            ar.UNCERTAINTY_THRESHOLD = float(self.v_aero_thresh.get())
            ar.AEROBALANCE_TARGET    = float(self.v_aero_target.get())
            ar.FAILURE_ITERATIONS    = int(self.v_aero_failure.get())
            self.app.run_tab._update_recruits()
            self.status_lbl.config(text="Applied", fg=OK_FG)
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            self.status_lbl.config(text="Error", fg=ERR_FG)


# ══════════════════════════════════════════════════════════════════
#  TAB: LOG  (not scrollable — the text widget handles its own scroll)
# ══════════════════════════════════════════════════════════════════

class LogTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG); self.app = app; self._build()

    def _build(self):
        self.rowconfigure(1, weight=1); self.columnconfigure(0, weight=1)
        ctrl = tk.Frame(self, bg=BG)
        ctrl.grid(row=0, column=0, sticky="ew", padx=PAD, pady=PAD_SM)
        tk.Button(ctrl, text="Clear log", font=FONT, bg=BG_D,
                  relief="flat", cursor="hand2", command=self._clear)\
            .pack(side="right")
        tk.Label(ctrl, text="Pipeline output", font=FONT_B, bg=BG).pack(side="left")
        self.log_text = scrolledtext.ScrolledText(
            self, font=FONT_M, state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", wrap="none", height=30)
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=(0, PAD))

    def _clear(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════
#  TAB: SURFACE
# ══════════════════════════════════════════════════════════════════

class SurfaceTab(ScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build(self.inner)

    def _build(self, f):
        f.columnconfigure(1, weight=1)
        r = 0

        # ── Single run ────────────────────────────────────────────
        section_label(f, "Single Run Surface Render", r); r += 2
        hint(f, ("Full nodes: the CFD surface cells are rendered as-is with their "
                 "exact values —\nno mapping onto the CAD geometry, no decimation, "
                 "no interpolation.\nOnly the parts of the Geometry .case are drawn "
                 "(plane sections are never shown).\nLeave 'Surface data .case' empty "
                 "to use the Cp/Cf stored in the Geometry .case."), r); r += 1
        self.v_geom    = tk.StringVar(value=cfg.GEOMETRY_CASE)
        self.v_surface = tk.StringVar()
        self.v_out_png = tk.StringVar(value=cfg.OUTPUT_DIR)
        labeled_entry(f, "Geometry .case",     self.v_geom,    r,
                      browse=lambda: browse_file(self.v_geom));    r += 1
        labeled_entry(f, "Surface data .case", self.v_surface, r,
                      browse=lambda: browse_file(self.v_surface)); r += 1
        labeled_entry(f, "Output directory",   self.v_out_png, r,
                      browse=lambda: browse_dir(self.v_out_png));  r += 1

        tk.Label(f, text="Scalar", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_scalar = tk.StringVar(
            value=list(cfg.SURFACE_SCALARS.keys())[0] if cfg.SURFACE_SCALARS else '')
        self.scalar_menu = ttk.Combobox(f, textvariable=self.v_scalar,
                                         values=list(cfg.SURFACE_SCALARS.keys()),
                                         width=20, state='readonly')
        self.scalar_menu.grid(row=r, column=1, sticky="w",
                              padx=PAD, pady=PAD_SM)
        self.scalar_menu.bind("<<ComboboxSelected>>",
                              lambda e: self._on_scalar_change()); r += 1

        tk.Label(f, text="Views", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="nw", padx=PAD, pady=PAD_SM)
        vf = tk.Frame(f, bg=BG)
        vf.grid(row=r, column=1, columnspan=2, sticky="w", padx=PAD)
        self.view_vars = {}
        all_views = ['bottom','top','iso_front','iso_rear',
                     'side_L','side_R','front','rear']
        for i, v in enumerate(all_views):
            bv = tk.BooleanVar(value=True); self.view_vars[v] = bv
            tk.Checkbutton(vf, text=v, variable=bv, font=FONT, bg=BG)\
                .grid(row=i//4, column=i%4, sticky="w", padx=PAD_SM)
        r += 3

        tk.Label(f, text="Figure height (in)", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_figh = tk.StringVar(value="7.0")
        tk.Entry(f, textvariable=self.v_figh, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        tk.Label(f, text="DPI", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_dpi = tk.StringVar(value=str(cfg.DPI))
        tk.Entry(f, textvariable=self.v_dpi, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        tk.Label(f, text="Feature angle (deg)", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_angle = tk.StringVar(value="25")
        tk.Entry(f, textvariable=self.v_angle, width=6, font=FONT)\
            .grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM); r += 1

        self.v_parallel = tk.BooleanVar(value=bool(cfg.SURFACE_PARALLEL_PROJECTION))
        tk.Checkbutton(f, text="Parallel projection (orthographic) — "
                               "untick for perspective",
                       variable=self.v_parallel, font=FONT, bg=BG,
                       command=lambda: setattr(cfg, 'SURFACE_PARALLEL_PROJECTION',
                                               bool(self.v_parallel.get())))\
            .grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD); r += 1

        tk.Label(f, text="Colour bands", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        bandf = tk.Frame(f, bg=BG)
        bandf.grid(row=r, column=1, columnspan=2, sticky="w", padx=PAD)
        self.v_bands = tk.StringVar(value=str(cfg.SURFACE_COLOR_BANDS))
        tk.Spinbox(bandf, from_=0, to=256, increment=1, width=6, font=FONT,
                   textvariable=self.v_bands).pack(side="left")
        tk.Label(bandf, text="solid colours, evenly spread over the range "
                             "(renders, delta, HTML) — 0 = smooth",
                 font=FONT_S, bg=BG, fg="#888").pack(side="left", padx=PAD)

        def _set_bands(*_):
            try:
                cfg.SURFACE_COLOR_BANDS = max(0, int(self.v_bands.get()))
            except ValueError:
                pass
        self.v_bands.trace_add("write", _set_bands)
        r += 1

        # ── Per-part individual exports ───────────────────────────
        section_label(f, "Individual Part Exports  (auto-zoomed)", r); r += 2
        hint(f, ("Ticked parts get their own separate PNGs per view,\n"
                 "auto-zoomed to fit just that component."), r, fg="#555"); r += 1

        # Known parts — uses merged names for FW and RW
        self.PART_NAMES = [
            'Bodywork', 'Driver',
            'Front wing',           # FW + FW EPs merged
            'Radiators and Fans',
            'Rear wing',            # RW + RW EPs merged
            'Side Wings', 'Undertray',
            'Wheel LHF', 'Wheel LHR', 'Wheel RHF', 'Wheel RHR',
            'Whiskers',
        ]
        self.part_vars = {}
        pf = tk.Frame(f, bg=BG)
        pf.grid(row=r, column=0, columnspan=3, sticky="ew", padx=PAD)
        for i, pname in enumerate(self.PART_NAMES):
            v = tk.BooleanVar(value=False)
            self.part_vars[pname] = v
            tk.Checkbutton(pf, text=pname, variable=v,
                           font=FONT, bg=BG)\
                .grid(row=i//3, column=i%3, sticky="w", padx=PAD_SM, pady=1)
        r += (len(self.PART_NAMES) + 2) // 3 + 1

        btn_sel = tk.Frame(f, bg=BG)
        btn_sel.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD)
        tk.Button(btn_sel, text="Select all",  font=FONT, bg=BG_D,
                  relief="flat", cursor="hand2",
                  command=lambda: [v.set(True)
                                   for v in self.part_vars.values()])\
            .pack(side="left", padx=PAD_SM)
        tk.Button(btn_sel, text="Select none", font=FONT, bg=BG_D,
                  relief="flat", cursor="hand2",
                  command=lambda: [v.set(False)
                                   for v in self.part_vars.values()])\
            .pack(side="left", padx=PAD_SM)
        r += 1

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3, pady=PAD, padx=PAD, sticky="e")
        tk.Button(bf, text="Render Surface", command=self._run_single,
                  **BTN_RUN).pack(side="right")
        self.status_single = tk.Label(bf, text="", font=FONT, bg=BG, fg="#555")
        self.status_single.pack(side="right", padx=PAD); r += 1

        # ── Delta render ──────────────────────────────────────────
        section_label(f, "Surface Delta Render  (B - A)", r); r += 2
        self.v_surf_a    = tk.StringVar()
        self.v_surf_b    = tk.StringVar()
        self.v_out_delta = tk.StringVar()
        self.v_label_a   = tk.StringVar(value="Run A")
        self.v_label_b   = tk.StringVar(value="Run B")
        labeled_entry(f, "Surface A (baseline)", self.v_surf_a, r,
                      browse=lambda: browse_file(self.v_surf_a)); r += 1
        labeled_entry(f, "Surface B (new)",      self.v_surf_b, r,
                      browse=lambda: browse_file(self.v_surf_b)); r += 1
        labeled_entry(f, "Output directory",     self.v_out_delta, r,
                      browse=lambda: browse_dir(self.v_out_delta)); r += 1
        labeled_entry(f, "Label for A", self.v_label_a, r, width=20); r += 1
        labeled_entry(f, "Label for B", self.v_label_b, r, width=20); r += 1

        tk.Label(f, text="Delta range +/-", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        dr_frame = tk.Frame(f, bg=BG)
        dr_frame.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_delta_mode = tk.StringVar(value="fixed")
        tk.Radiobutton(dr_frame, text="Fixed:", variable=self.v_delta_mode,
                       value="fixed", font=FONT, bg=BG,
                       command=self._toggle_delta_range).pack(side="left")
        self.v_delta_val = tk.StringVar(
            value=f"{cfg.default_delta_range(self.v_scalar.get()):g}")
        self.delta_entry = tk.Entry(dr_frame, textvariable=self.v_delta_val,
                                    width=7, font=FONT)
        self.delta_entry.pack(side="left", padx=PAD_SM)
        tk.Radiobutton(dr_frame, text="Auto", variable=self.v_delta_mode,
                       value="auto", font=FONT, bg=BG,
                       command=self._toggle_delta_range)\
            .pack(side="left", padx=(PAD, 0))
        r += 1

        tk.Label(f,
                 text="Geometry:  Gray=both  |  Magenta=new (B only)  |  Brown=removed (A only)",
                 font=FONT_S, bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

        bf2 = tk.Frame(f, bg=BG)
        bf2.grid(row=r, column=0, columnspan=3, pady=PAD, padx=PAD, sticky="e")
        tk.Button(bf2, text="Render Delta", command=self._run_delta,
                  **BTN_RUN).pack(side="right")
        self.status_delta = tk.Label(bf2, text="", font=FONT, bg=BG, fg="#555")
        self.status_delta.pack(side="right", padx=PAD); r += 1

        # ── 3D Export ─────────────────────────────────────────────
        section_label(f, "3D Export", r); r += 2
        self.v_export_out = tk.StringVar()
        labeled_entry(f, "Output directory", self.v_export_out, r,
                      browse=lambda: browse_dir(self.v_export_out)); r += 1

        tk.Label(f, text="Format", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        fmt_frame = tk.Frame(f, bg=BG)
        fmt_frame.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_fmt = tk.StringVar(value="vtp")
        for i, (label, val) in enumerate([
            ("VTP — scalars preserved, ParaView", "vtp"),
            ("VTK — scalars preserved",           "vtk"),
            ("STL — geometry only, universal",    "stl"),
            ("PLY — geometry only, compact",      "ply"),
            ("OBJ — geometry only, universal",    "obj"),
        ]):
            tk.Radiobutton(fmt_frame, text=label, variable=self.v_fmt,
                           value=val, font=FONT, bg=BG)\
                .grid(row=i, column=0, sticky="w", padx=PAD_SM)
        r += 6

        self.v_merged = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Merge all parts into one file",
                       variable=self.v_merged, font=FONT, bg=BG)\
            .grid(row=r, column=0, columnspan=2, sticky="w", padx=PAD); r += 1

        bf3 = tk.Frame(f, bg=BG)
        bf3.grid(row=r, column=0, columnspan=3, pady=PAD, padx=PAD, sticky="e")
        tk.Button(bf3, text="Export 3D", command=self._run_export,
                  **BTN_RUN).pack(side="right")
        self.status_export = tk.Label(bf3, text="", font=FONT, bg=BG, fg="#555")
        self.status_export.pack(side="right", padx=PAD); r += 1

        # ── Interactive HTML export ───────────────────────────────
        section_label(f, "Interactive HTML  (open in any browser)", r); r += 2
        hint(f, ("Exports a self-contained .html file — drag to rotate, scroll to zoom,\n"
                 "click legend to toggle parts on/off. Full mesh: every CFD cell with "
                 "its exact value\n(no decimation) — large cars give large files."),
             r, fg="#555"); r += 1

        self.v_html_out = tk.StringVar()
        labeled_entry(f, "Output .html file", self.v_html_out, r,
                      browse=lambda: self.v_html_out.set(
                          filedialog.asksaveasfilename(
                              defaultextension=".html",
                              filetypes=[("HTML files", "*.html")],
                              initialfile="surface_Cp.html"))); r += 1

        bf4 = tk.Frame(f, bg=BG)
        bf4.grid(row=r, column=0, columnspan=3, pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf4, text="Export Interactive HTML",
                  command=self._run_html_export,
                  bg="#7B1FA2", fg="white", font=FONT_B,
                  relief="flat", padx=12, pady=6, cursor="hand2")\
            .pack(side="right")
        self.status_html = tk.Label(bf4, text="", font=FONT, bg=BG, fg="#555")
        self.status_html.pack(side="right", padx=PAD)

    def _toggle_delta_range(self):
        if self.v_delta_mode.get() == "fixed":
            self.delta_entry.config(state="normal", bg="white")
        else:
            self.delta_entry.config(state="disabled", bg=BG_D)

    def _on_scalar_change(self):
        self.v_delta_val.set(f"{cfg.default_delta_range(self.v_scalar.get()):g}")

    def refresh_scalars(self):
        keys = list(cfg.SURFACE_SCALARS.keys())
        self.scalar_menu['values'] = keys
        if keys and self.v_scalar.get() not in keys:
            self.v_scalar.set(keys[0])
            self._on_scalar_change()

    def _get_views(self):
        return [v for v, bv in self.view_vars.items() if bv.get()] or None

    def _angle(self):
        try:
            return float(self.v_angle.get())
        except ValueError:
            return 25.0

    def _run_single(self):
        geom    = self.v_geom.get().strip()
        surface = self.v_surface.get().strip() or None
        out_dir = self.v_out_png.get().strip()
        skey    = self.v_scalar.get().strip()
        views   = self._get_views()
        parts_to_export = [p for p, v in self.part_vars.items() if v.get()]
        if not geom:
            messagebox.showwarning("Missing", "Set geometry .case path."); return
        try:
            float(self.v_figh.get()); int(self.v_dpi.get())
            angle = float(self.v_angle.get())
        except Exception:
            messagebox.showerror("Invalid", "Height/DPI/angle must be numbers."); return

        self.status_single.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr
                scfg = surface_scalar_cfg(skey)
                clim = (scfg.get('vmin', -4.0), scfg.get('vmax', 1.0))
                cmap = scfg.get('cmap') or sr.make_cp_cmap(vmin=clim[0], vmax=clim[1])
                arr  = scfg.get('array', skey)
                label = scfg.get('label', skey)
                stem = f'surface_{skey}'

                prepared = sr.prepare_surfaces(
                    geom, surface, feature_angle=angle)

                print(f"\n[surface] Rendering full-car views...")
                sr.render_views(prepared, arr, cmap=cmap, clim=clim,
                                views=views, window_size=(1600, 1000),
                                bg_color='white',
                                output_dir=out_dir, file_stem=stem, label=label)

                if parts_to_export:
                    print(f"\n[surface] Rendering {len(parts_to_export)} "
                          f"individual parts...")
                    sr.render_part_views(
                        prepared, parts_to_export, arr,
                        cmap=cmap, clim=clim,
                        views=views, window_size=(1600, 1000),
                        bg_color='white',
                        output_dir=out_dir, file_stem=stem, label=label)

                self.status_single.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_single.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()

    def _run_delta(self):
        geom   = self.v_geom.get().strip()
        surf_a = self.v_surf_a.get().strip()
        surf_b = self.v_surf_b.get().strip()
        out    = self.v_out_delta.get().strip()
        skey   = self.v_scalar.get().strip()
        views  = self._get_views()
        if not geom or not surf_a or not surf_b:
            messagebox.showwarning("Missing",
                "Set geometry and both surface .case paths."); return
        fixed_range = None
        if self.v_delta_mode.get() == "fixed":
            try:
                fixed_range = float(self.v_delta_val.get()); assert fixed_range > 0
            except Exception:
                messagebox.showerror("Invalid", "Fixed range must be a positive number.")
                return
        scfg = surface_scalar_cfg(skey)
        arr  = scfg.get('array', skey)
        la   = self.v_label_a.get() or "Run A"
        lb   = self.v_label_b.get() or "Run B"
        angle = self._angle()
        self.status_delta.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr, matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt, matplotlib.colors as mc2
                import plotting
                print("[surface] Preparing Run A...")
                prep_a = sr.prepare_surfaces(geom, surf_a, feature_angle=angle)
                print("[surface] Preparing Run B...")
                prep_b = sr.prepare_surfaces(geom, surf_b, feature_angle=angle)
                imgs, clim = sr.render_delta_views(
                    prep_a, prep_b, arr, label_a=la, label_b=lb,
                    views=views, fixed_range=fixed_range, bg_color='white')
                os.makedirs(out or '.', exist_ok=True)
                n = len(imgs); cols = min(n,3); rows = (n+cols-1)//cols
                figh = float(self.v_figh.get()) if self.v_figh.get() else 7.0
                dpi  = int(self.v_dpi.get()) if self.v_dpi.get() else 150
                fig, axes = plt.subplots(rows, cols,
                                          figsize=(cols*10, rows*figh))
                fig.patch.set_facecolor('white')
                af = list(axes.flat) if hasattr(axes,'flat') else [axes]
                for ax,(vname,img) in zip(af,imgs.items()):
                    ax.imshow(img); ax.axis('off')
                    ax.set_title(vname.replace('_',' ').title(),
                                 color='black', fontsize=11)
                for ax in af[n:]: ax.axis('off')
                sm = plt.cm.ScalarMappable(cmap=sr.banded_cmap('RdBu_r'),
                     norm=mc2.Normalize(vmin=clim[0],vmax=clim[1]))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=af, fraction=0.01, pad=0.02,
                                    label=f'delta {skey}  ({lb} - {la})')
                plotting.style_colorbar(cbar, clim[0], clim[1])
                plt.suptitle(f'Surface Delta -- {skey}  {lb} - {la}', fontsize=14)
                out_path = os.path.join(out or '.', f'surface_delta_{skey}.png')
                fig.savefig(out_path, dpi=dpi, facecolor='white',
                            bbox_inches='tight')
                plt.close(fig)
                print(f"[surface] Saved -> {out_path}")
                self.status_delta.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_delta.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()

    def _run_export(self):
        geom    = self.v_geom.get().strip()
        surface = self.v_surface.get().strip() or None
        out_dir = self.v_export_out.get().strip()
        fmt     = self.v_fmt.get()
        merged  = self.v_merged.get()
        if not geom:
            messagebox.showwarning("Missing", "Set geometry .case path."); return
        if not out_dir:
            messagebox.showwarning("Missing", "Set output directory."); return
        angle = self._angle()
        self.status_export.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr
                prepared = sr.prepare_surfaces(geom, surface, feature_angle=angle)
                scalar_keys = None
                if surface:
                    scalar_keys = sorted({sd['array'] for sd in
                                          list(cfg.SCALARS.values()) +
                                          list(cfg.SURFACE_SCALARS.values())})
                sr.export_3d(prepared, scalar_keys=scalar_keys,
                             output_dir=out_dir, fmt=fmt, merged=merged)
                self.status_export.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_export.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()

    def _run_html_export(self):
        geom    = self.v_geom.get().strip()
        surface = self.v_surface.get().strip() or None
        out_path = self.v_html_out.get().strip()
        skey    = self.v_scalar.get().strip()

        if not geom:
            messagebox.showwarning("Missing", "Set geometry .case path."); return
        if not out_path:
            messagebox.showwarning("Missing", "Set output .html file path."); return

        scfg  = surface_scalar_cfg(skey)
        arr   = scfg.get('array', skey)
        cmap  = scfg.get('cmap', None)
        clim  = (scfg.get('vmin', -4.0), scfg.get('vmax', 1.0))
        label = scfg.get('label', skey)
        angle = self._angle()

        self.status_html.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr
                prepared = sr.prepare_surfaces(geom, surface, feature_angle=angle)
                sr.export_interactive_html(
                    prepared, arr,
                    cmap=cmap, clim=clim, label=label,
                    output_path=out_path,
                )
                self.status_html.config(text="Done", fg=OK_FG)
            except Exception as exc:
                self.status_html.config(text="Error", fg=ERR_FG)
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
#  TAB: SURFACE CP  (Cp along any slice — export)
# ══════════════════════════════════════════════════════════════════

class SurfaceCpTab(ScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._parts = None          # cached surface (full nodes)
        self._parts_path = None
        self._busy = False
        self._build(self.inner)

    def _build(self, f):
        f.columnconfigure(1, weight=1)
        r = 0

        section_label(f, "Surface Data", r); r += 2
        hint(f, ("Cuts the CFD surface mesh exactly at each slice. Every point is a "
                 "real CFD cell value (or exact node\nvalue for node-based exports) — "
                 "full nodes, nothing interpolated, resampled or smoothed.\n"
                 "The Geometry .case parts are always the parts listed. Leave 'Surface data "
                 ".case' empty to use the Cp/Cf\nstored in the Geometry .case — plane "
                 "files such as Raw_Data.case are ignored here."), r); r += 1
        self.v_geom    = tk.StringVar(value=cfg.GEOMETRY_CASE)
        self.v_surface = tk.StringVar()
        self.v_out     = tk.StringVar(value=cfg.OUTPUT_DIR)
        labeled_entry(f, "Geometry .case", self.v_geom, r,
                      browse=lambda: browse_file(self.v_geom)); r += 1
        labeled_entry(f, "Surface data .case", self.v_surface, r,
                      browse=lambda: browse_file(self.v_surface)); r += 1
        labeled_entry(f, "Output directory", self.v_out, r,
                      browse=lambda: browse_dir(self.v_out)); r += 1

        lf = tk.Frame(f, bg=BG)
        lf.grid(row=r, column=1, columnspan=2, sticky="w", padx=PAD, pady=PAD_SM)
        tk.Button(lf, text="Load parts", command=self._load_parts,
                  **BTN_SEC).pack(side="left")
        tk.Button(lf, text="Use paths from Surface tab", font=FONT, bg=BG_D,
                  relief="flat", cursor="hand2",
                  command=self._copy_from_surface_tab)\
            .pack(side="left", padx=PAD)
        self.load_lbl = tk.Label(lf, text="", font=FONT, bg=BG, fg="#555")
        self.load_lbl.pack(side="left", padx=PAD)
        r += 1

        section_label(f, "Slice", r); r += 2
        tk.Label(f, text="Scalar", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_scalar = tk.StringVar(
            value=list(cfg.SURFACE_SCALARS.keys())[0] if cfg.SURFACE_SCALARS else 'Cp')
        self.scalar_menu = ttk.Combobox(f, textvariable=self.v_scalar,
                                        values=list(cfg.SURFACE_SCALARS.keys()),
                                        width=20, state='readonly')
        self.scalar_menu.grid(row=r, column=1, sticky="w", padx=PAD, pady=PAD_SM)
        r += 1

        tk.Label(f, text="Slice normal", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        axf = tk.Frame(f, bg=BG)
        axf.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_axis = tk.StringVar(value="Y")
        for ax, desc in [("X", "X  (plot vs Y — spanwise)"),
                         ("Y", "Y  (plot vs X — chordwise)"),
                         ("Z", "Z  (plot vs X — plan view)")]:
            tk.Radiobutton(axf, text=desc, variable=self.v_axis, value=ax,
                           font=FONT, bg=BG).pack(side="left", padx=(0, PAD))
        r += 1

        tk.Label(f, text="Positions (mm)", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        self.v_positions = tk.StringVar(value="0, 150, 300, 450")
        tk.Entry(f, textvariable=self.v_positions, width=40, font=FONT)\
            .grid(row=r, column=1, sticky="ew", padx=(PAD, PAD), pady=PAD_SM)
        r += 1
        hint(f, ("Comma-separated and/or ranges start:stop:step   "
                 "e.g.   100, 250, -400   or   0:600:50"), r, col=1, colspan=2); r += 1

        section_label(f, "Parts", r); r += 2
        self.parts_frame = tk.Frame(f, bg=BG)
        self.parts_frame.grid(row=r, column=0, columnspan=3, sticky="ew", padx=PAD)
        self.part_vars = {}
        self._show_parts_msg("Parts load automatically from the Geometry .case when "
                             "this tab opens (or click 'Load parts').")
        r += 1
        sel = tk.Frame(f, bg=BG)
        sel.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD)
        tk.Button(sel, text="Select all", font=FONT, bg=BG_D, relief="flat",
                  cursor="hand2",
                  command=lambda: [v.set(True) for v in self.part_vars.values()])\
            .pack(side="left", padx=PAD_SM)
        tk.Button(sel, text="Select none", font=FONT, bg=BG_D, relief="flat",
                  cursor="hand2",
                  command=lambda: [v.set(False) for v in self.part_vars.values()])\
            .pack(side="left", padx=PAD_SM)
        r += 1

        section_label(f, "Plot Options", r); r += 2
        of = tk.Frame(f, bg=BG)
        of.grid(row=r, column=0, columnspan=3, sticky="w", padx=PAD)
        self.v_invert  = tk.BooleanVar(value=True)
        self.v_profile = tk.BooleanVar(value=True)
        self.v_csv     = tk.BooleanVar(value=True)
        tk.Checkbutton(of, text="Invert value axis (suction up)",
                       variable=self.v_invert, font=FONT, bg=BG)\
            .pack(side="left", padx=(0, PAD))
        tk.Checkbutton(of, text="Section profile coloured by value",
                       variable=self.v_profile, font=FONT, bg=BG)\
            .pack(side="left", padx=(0, PAD))
        tk.Checkbutton(of, text="Export CSV (every point)",
                       variable=self.v_csv, font=FONT, bg=BG)\
            .pack(side="left")
        r += 1

        def _pair(label, row, a, b, unit):
            tk.Label(f, text=label, font=FONT, bg=BG, anchor="w")\
                .grid(row=row, column=0, sticky="w", padx=PAD, pady=PAD_SM)
            pf = tk.Frame(f, bg=BG)
            pf.grid(row=row, column=1, sticky="w", padx=PAD)
            tk.Label(pf, text="min", font=FONT, bg=BG).pack(side="left")
            tk.Entry(pf, textvariable=a, width=8, font=FONT).pack(side="left", padx=PAD_SM)
            tk.Label(pf, text="max", font=FONT, bg=BG).pack(side="left", padx=(PAD, 0))
            tk.Entry(pf, textvariable=b, width=8, font=FONT).pack(side="left", padx=PAD_SM)
            tk.Label(pf, text=unit, font=FONT_S, bg=BG, fg="#888").pack(side="left", padx=PAD)

        self.v_hmin, self.v_hmax = tk.StringVar(), tk.StringVar()
        self.v_cmin, self.v_cmax = tk.StringVar(), tk.StringVar()
        _pair("Horizontal limits", r, self.v_hmin, self.v_hmax,
              "metres, blank = auto (use to zoom on one component)"); r += 1
        _pair("Value axis limits", r, self.v_cmin, self.v_cmax,
              "blank = auto"); r += 1

        tk.Label(f, text="Figure size (in) / DPI", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        sf = tk.Frame(f, bg=BG)
        sf.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_figw = tk.StringVar(value="12")
        self.v_figh = tk.StringVar(value="8")
        self.v_dpi  = tk.StringVar(value=str(cfg.DPI))
        for lbl, var in (("W", self.v_figw), ("H", self.v_figh), ("DPI", self.v_dpi)):
            tk.Label(sf, text=lbl, font=FONT, bg=BG).pack(side="left")
            tk.Entry(sf, textvariable=var, width=6, font=FONT)\
                .pack(side="left", padx=(PAD_SM, PAD))
        r += 1

        hint(f, ("Output:  <output>/SurfaceCp/<scalar>/<axis>/"
                 "surfcp_<scalar>_<axis>_<pos>mm.png  (+ .csv)"), r, fg="#888"); r += 1

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3, pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf, text="Export Surface Cp Plots", command=self._run,
                  **BTN_RUN).pack(side="right")
        self.status_lbl = tk.Label(bf, text="", font=FONT, bg=BG, fg="#555")
        self.status_lbl.pack(side="right", padx=PAD)

    # ── helpers ───────────────────────────────────────────────────
    def refresh_scalars(self):
        keys = list(cfg.SURFACE_SCALARS.keys())
        self.scalar_menu['values'] = keys
        if keys and self.v_scalar.get() not in keys:
            self.v_scalar.set(keys[0])

    def _copy_from_surface_tab(self):
        st = self.app.surface_tab
        geom, surf = st.v_geom.get().strip(), st.v_surface.get().strip()
        if not geom and not surf:
            messagebox.showinfo("Surface tab",
                                "No paths are set in the Surface tab.")
            return
        self.v_geom.set(geom)
        self.v_surface.set(surf)
        self._load_parts()

    def _paths(self, quiet=False):
        """(surface_case or None, geom_case or None); None if neither is set."""
        geom = self.v_geom.get().strip() or None
        surf = self.v_surface.get().strip() or None
        if not geom and not surf:
            if not quiet:
                messagebox.showwarning("Missing",
                    "Set the Geometry .case (and optionally a Surface data .case).")
            return None
        return surf, geom

    def auto_load(self):
        """Load the part list when the tab opens, if the paths point at files."""
        paths = self._paths(quiet=True)
        if paths is None or self._busy:
            return
        if not all(os.path.isfile(p) for p in paths if p):
            return
        if self.part_vars and self._parts_path == paths:
            return
        self._load_parts()

    def _show_parts_msg(self, text, fg="#888"):
        for w in self.parts_frame.winfo_children():
            w.destroy()
        tk.Label(self.parts_frame, text=text, font=FONT_S, bg=BG, fg=fg,
                 justify="left", wraplength=760)\
            .grid(row=0, column=0, columnspan=3, sticky="w")

    def _ensure_loaded(self, surf, geom):
        """Load (or reuse) the full-node surface. Runs in a worker thread."""
        import surface_cp as sc
        key = (surf, geom)
        if self._parts is None or self._parts_path != key:
            self._parts = sc.load_surface(surf, geom)
            self._parts_path = key
        return self._parts

    def _populate_parts(self, parts):
        for w in self.parts_frame.winfo_children():
            w.destroy()
        old = {k: v.get() for k, v in self.part_vars.items()}
        self.part_vars = {}
        if not parts:
            self._show_parts_msg("No parts found — see the Log tab.", ERR_FG)
            return
        for i, p in enumerate(parts):
            v = tk.BooleanVar(value=old.get(p['name'], True))
            self.part_vars[p['name']] = v
            tk.Checkbutton(self.parts_frame,
                           text=f"{p['name']}  ({p['mesh'].n_cells:,})",
                           variable=v, font=FONT, bg=BG)\
                .grid(row=i // 3, column=i % 3, sticky="w", padx=PAD_SM, pady=1)

    def _load_parts(self):
        paths = self._paths()
        if paths is None:
            return
        if self._busy:
            return
        self._busy = True
        self.load_lbl.config(text="Loading parts...", fg=ACCENT)
        self._show_parts_msg("Loading parts...", ACCENT)

        def _worker():
            try:
                parts = self._ensure_loaded(*paths)
                n = sum(p['mesh'].n_cells for p in parts)
                self.app.ui(lambda: (self._populate_parts(parts),
                                       self.load_lbl.config(
                                           text=f"{len(parts)} parts · {n:,} cells",
                                           fg=OK_FG)))
            except Exception as exc:
                msg = str(exc)
                self.app.ui(lambda: (self.load_lbl.config(
                    text="Error — see Log tab", fg=ERR_FG),
                    self._show_parts_msg(f"Could not load parts: {msg}", ERR_FG)))
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
            finally:
                self._busy = False
        threading.Thread(target=_worker, daemon=True).start()

    def _run(self):
        import surface_cp as sc
        paths = self._paths()
        out  = self.v_out.get().strip()
        if paths is None:
            return
        if not out:
            messagebox.showwarning("Missing", "Set the output directory."); return
        try:
            positions = sc.parse_positions(self.v_positions.get())
            assert positions
        except Exception as exc:
            messagebox.showerror("Positions",
                                 f"Could not read the slice positions.\n\n{exc}")
            return
        try:
            h_lim = (_float_or_none(self.v_hmin.get()), _float_or_none(self.v_hmax.get()))
            h_lim = h_lim if None not in h_lim else None
            v_lim = (_float_or_none(self.v_cmin.get()), _float_or_none(self.v_cmax.get()))
            v_lim = v_lim if None not in v_lim else None
            fig_size = (float(self.v_figw.get()), float(self.v_figh.get()))
            dpi = int(self.v_dpi.get())
        except ValueError:
            messagebox.showerror("Invalid", "Limits, figure size and DPI must be numbers.")
            return
        if self._busy:
            return
        selected = [k for k, v in self.part_vars.items() if v.get()] or None
        if self.part_vars and not selected:
            messagebox.showwarning("Parts", "Select at least one part."); return

        skey = self.v_scalar.get().strip()
        axis = self.v_axis.get()
        opts = dict(part_names=selected, invert=self.v_invert.get(),
                    h_lim=h_lim, val_lim=v_lim,
                    show_profile=self.v_profile.get(),
                    write_csv_files=self.v_csv.get(),
                    fig_size=fig_size, dpi=dpi)
        self._busy = True
        self.status_lbl.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                parts = self._ensure_loaded(*paths)
                if not self.part_vars:
                    self.app.ui(lambda: self._populate_parts(parts))
                files = sc.export_sections(parts, axis, positions, skey, out, **opts)
                n_png = sum(1 for p in files if p.endswith('.png'))
                self.app.ui(lambda: self.status_lbl.config(
                    text=f"Done — {n_png} plot{'s' if n_png != 1 else ''}", fg=OK_FG))
            except Exception as exc:
                self.app.ui(lambda: self.status_lbl.config(text="Error", fg=ERR_FG))
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
            finally:
                self._busy = False
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CFD Slicer")
        self.configure(bg=BG)
        self.geometry("860x720")
        self.minsize(700, 500)

        hdr = tk.Frame(self, bg=ACCENT, height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  CFD Slicer",
                 font=("Segoe UI", 13, "bold"), bg=ACCENT, fg="white")\
            .pack(side="left", padx=PAD)
        tk.Label(hdr, text="slice  .  interpolate  .  compare",
                 font=("Segoe UI", 9), bg=ACCENT, fg="#BBDEFB")\
            .pack(side="left", padx=4)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", font=FONT_B, padding=(12, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", "white"), ("!selected", BG_D)],
                  foreground=[("selected", ACCENT),  ("!selected", "#333")])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=PAD, pady=PAD)

        self.run_tab        = RunTab(self.notebook, self)
        self.compare_tab    = CompareTab(self.notebook, self)
        self.surface_tab    = SurfaceTab(self.notebook, self)
        self.surface_cp_tab = SurfaceCpTab(self.notebook, self)
        self.settings_tab   = SettingsTab(self.notebook, self)
        self.log_tab        = LogTab(self.notebook, self)

        self.notebook.add(self.run_tab,        text="  Run  ")
        self.notebook.add(self.compare_tab,    text="  Compare  ")
        self.notebook.add(self.surface_tab,    text="  Surface  ")
        self.notebook.add(self.surface_cp_tab, text="  Surface Cp  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.notebook.add(self.settings_tab,   text="  Settings  ")
        self.notebook.add(self.log_tab,        text="  Log  ")

        redir = LogRedirector(self.log_tab.log_text, sys.stdout)
        self._log_redir = redir
        sys.stdout = redir

        # Worker threads hand UI updates to the main thread through a queue
        self._ui_queue = queue.Queue()
        self._pump_ui()

        # ── Watermark / status bar ───────────────────────────────
        bar = tk.Frame(self, bg=BG_D)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, text="Ready", font=FONT, bg=BG_D,
                 fg="#555", anchor="w")            .pack(side="left", padx=PAD)
        tk.Label(bar, text="Julian G-A  ·  UTFR",
                 font=FONT_S, bg=BG_D,
                 fg="#999", anchor="e")            .pack(side="right", padx=PAD)


    def _on_tab_changed(self, _event=None):
        try:
            if self.notebook.select() == str(self.surface_cp_tab):
                self.surface_cp_tab.auto_load()
        except Exception as exc:
            print(f"[ui] {exc}")

    def ui(self, fn):
        """Run fn() on the Tk main thread (safe to call from worker threads)."""
        self._ui_queue.put(fn)

    def _pump_ui(self):
        redir = getattr(self, '_log_redir', None)
        if redir is not None:
            redir.drain()
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                try:
                    fn()
                except Exception as exc:
                    print(f"[ui] {exc}")
        except queue.Empty:
            pass
        self.after(50, self._pump_ui)


def main():
    multiprocessing.freeze_support()     # recruits in a frozen .exe
    App().mainloop()


if __name__ == "__main__":
    main()