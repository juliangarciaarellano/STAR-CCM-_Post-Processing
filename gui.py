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
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ══════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════
PAD    = 8
PAD_SM = 4
FONT   = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
FONT_M = ("Consolas", 9)
BG     = "#f5f5f5"
BG_D   = "#e0e0e0"
ACCENT = "#1976D2"
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


# ══════════════════════════════════════════════════════════════════
#  LOG REDIRECTOR
# ══════════════════════════════════════════════════════════════════

class LogRedirector:
    def __init__(self, widget, original):
        self.widget = widget; self.original = original
    def write(self, text):
        self.original.write(text)
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", text)
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except Exception:
            pass
    def flush(self): self.original.flush()


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
            tk.Checkbutton(af, text=ax, variable=v, font=FONT, bg=BG)\
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

        bf = tk.Frame(f, bg=BG)
        bf.grid(row=r, column=0, columnspan=3, pady=PAD*2, padx=PAD, sticky="e")
        tk.Button(bf, text="Run Pipeline", command=self._run,
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

    def refresh_scalars(self):
        self._rebuild_scalar_checks()

    def _run(self):
        scalars = [k for k, v in self.scalar_vars.items() if v.get()]
        axes    = [k for k, v in self.axis_vars.items()   if v.get()]
        if not scalars:
            messagebox.showwarning("Nothing to do", "Select at least one scalar."); return
        if not axes:
            messagebox.showwarning("Nothing to do", "Select at least one axis."); return
        try:
            res = float(self.v_res.get()); assert res > 0
        except:
            messagebox.showerror("Invalid resolution",
                                 "mm per pixel must be a positive number."); return
        try:
            figh = float(self.v_figh.get()); dpi = int(self.v_dpi.get())
        except:
            messagebox.showerror("Invalid size",
                                 "Figure height and DPI must be numbers."); return

        cfg.GEOMETRY_CASE = self.v_geom.get()
        cfg.DATA_CASE     = self.v_data.get()
        cfg.OUTPUT_DIR    = self.v_out.get()
        cfg.DPI           = dpi

        self.status_lbl.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import pipeline
                pipeline.run_pipeline(scalars=scalars, axes=axes,
                                       resolution_mm=res,
                                       fig_height=figh, dpi=dpi)
                self.status_lbl.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_lbl.config(text="Error", fg="#C62828")
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
        tk.Label(f,
                 text=("Auto: scale +/-max(|diff|) per plane  |  "
                       "Fixed: same symmetric range for every plane of that scalar"),
                 font=("Segoe UI", 8), bg=BG, fg="#666")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=(0, PAD_SM))
        r += 1
        for col, hdr in enumerate(["Scalar", "Mode", "Fixed +/- value"]):
            tk.Label(f, text=hdr, font=FONT_B, bg=BG_D)\
                .grid(row=r, column=col, sticky="ew",
                      padx=1, pady=1, ipadx=6, ipady=3)
        r += 1
        self.range_frame_row = r
        self.range_vars = {}
        self._rebuild_range_rows(f)
        r += len(cfg.SCALARS)

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
        self.v_images = tk.BooleanVar(value=True)
        tk.Checkbutton(f,
                       text="Compute plane image comparison (delta PNGs + CSV)",
                       variable=self.v_images, font=FONT, bg=BG)\
            .grid(row=r, column=0, columnspan=2, sticky="w", padx=PAD); r += 1
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
                 font=("Segoe UI", 8), bg=BG, fg="#555", justify="left")\
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

    def _rebuild_range_rows(self, f=None):
        if f is None:
            f = self.inner
        # Clean up old widgets
        for skey, (mv, vv, ew) in self.range_vars.items():
            try:
                for widget in f.grid_slaves():
                    if int(widget.grid_info().get('row', -1)) == \
                       self.range_frame_row + list(cfg.SCALARS.keys()).index(skey):
                        widget.destroy()
            except Exception:
                pass
        self.range_vars = {}
        r = self.range_frame_row
        for skey in cfg.SCALARS:
            mode_var  = tk.StringVar(value="auto")
            value_var = tk.StringVar(value="0.1")
            tk.Label(f, text=skey, font=FONT, bg=BG, anchor="w")\
                .grid(row=r, column=0, sticky="w", padx=PAD, pady=2)
            mf = tk.Frame(f, bg=BG)
            mf.grid(row=r, column=1, sticky="w", padx=PAD)
            for mode in ["auto", "fixed"]:
                tk.Radiobutton(mf, text=mode, variable=mode_var,
                               value=mode, font=FONT, bg=BG,
                               command=lambda mv=mode_var, vv=value_var,
                                              sk=skey: self._toggle_range(mv, vv, sk))\
                    .pack(side="left", padx=PAD_SM)
            entry = tk.Entry(f, textvariable=value_var, width=8,
                             font=FONT, state="disabled", bg=BG_D)
            entry.grid(row=r, column=2, padx=PAD, pady=2, sticky="w")
            self.range_vars[skey] = (mode_var, value_var, entry)
            r += 1

    def _toggle_range(self, mode_var, value_var, skey):
        _, _, entry = self.range_vars[skey]
        if mode_var.get() == "fixed":
            entry.config(state="normal", bg="white")
        else:
            entry.config(state="disabled", bg=BG_D)

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
                    result[skey] = float(value_var.get())
                except ValueError:
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
        except:
            messagebox.showerror("Invalid size",
                                 "Figure height and DPI must be numbers."); return
        delta_ranges = self._get_delta_ranges()
        if delta_ranges is None: return
        do_images = self.v_images.get()
        log_a = self.v_log_a.get().strip() or None
        log_b = self.v_log_b.get().strip() or None
        if self.v_aero.get() and (not log_a or not log_b):
            messagebox.showwarning("Missing log files",
                "Provide both log files or uncheck 'Generate aero report'.")
            return
        if not do_images and not self.v_aero.get():
            messagebox.showwarning("Nothing to do",
                "Enable image comparison, the aero report, or both.")
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
                    images=do_images,
                )
                self.status_lbl.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_lbl.config(text="Error", fg="#C62828")
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
                 font=("Segoe UI", 8), bg=BG, fg="#888")\
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
            import matplotlib.colors as mcolors
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
                mid = (vmin+vmax)/2
                pos = [0.0, (mid-vmin)/(vmax-vmin), 1.0]
                cdict = {
                    'red':   [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.8,0.8)],
                    'green': [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
                    'blue':  [(pos[0],0.8,0.8),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
                }
                cmap = mcolors.LinearSegmentedColormap(f"auto_{key}", cdict, N=256)
                cfg.SCALARS[key] = {'array': arr, 'vmin': vmin, 'vmax': vmax,
                                    'cmap': cmap, 'label': key}
                existing_arrays.add(arr); added.append(key)
            self._refresh_scalar_list()
            self.app.run_tab.refresh_scalars()
            self.app.compare_tab.refresh_scalars()
            if added:
                self.detect_lbl.config(text=f"Added: {', '.join(added)}", fg="#2E7D32")
            else:
                self.detect_lbl.config(text="All arrays already present", fg="#555")
        except Exception as exc:
            self.detect_lbl.config(text="Error -- see Log tab", fg="#C62828")
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
            self._refresh_scalar_list()
            self.app.run_tab.refresh_scalars()
            self.app.compare_tab.refresh_scalars()

    def _on_scalar_saved(self, result):
        import matplotlib.colors as mcolors
        vmin, vmax = result['vmin'], result['vmax']
        mid = (vmin+vmax)/2
        pos = [0.0, (mid-vmin)/(vmax-vmin), 1.0]
        cdict = {
            'red':   [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.8,0.8)],
            'green': [(pos[0],0.0,0.0),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
            'blue':  [(pos[0],0.8,0.8),(pos[1],1.0,1.0),(pos[2],0.0,0.0)],
        }
        cmap = mcolors.LinearSegmentedColormap(f"auto_{result['key']}", cdict, N=256)
        if result['orig_key'] and result['orig_key'] != result['key']:
            cfg.SCALARS.pop(result['orig_key'], None)
        cfg.SCALARS[result['key']] = {
            'array': result['array'], 'vmin': result['vmin'],
            'vmax':  result['vmax'],  'cmap': cmap,
            'label': result['label'],
        }
        self._refresh_scalar_list()
        self.app.run_tab.refresh_scalars()
        self.app.compare_tab.refresh_scalars()

    def _apply(self):
        try:
            for ax in range(3):
                cfg.CAR_BOUNDS_2D[ax] = tuple(
                    float(v.get()) for v in self.bounds_vars[ax])
            cfg.DPI             = int(self.v_dpi.get())
            cfg.FILL_COLOR      = self.v_fill.get().strip()
            cfg.TICK_SPACING_MM = int(self.v_tick.get())
            cfg.DILATION_PX     = int(self.v_dilat.get())
            import aero_report as ar
            ar.UNCERTAINTY_THRESHOLD = float(self.v_aero_thresh.get())
            ar.AEROBALANCE_TARGET    = float(self.v_aero_target.get())
            ar.FAILURE_ITERATIONS    = int(self.v_aero_failure.get())
            self.status_lbl.config(text="Applied", fg="#2E7D32")
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            self.status_lbl.config(text="Error", fg="#C62828")


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
                              padx=PAD, pady=PAD_SM); r += 1

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

        # ── Per-part individual exports ───────────────────────────
        section_label(f, "Individual Part Exports  (auto-zoomed)", r); r += 2
        tk.Label(f, text=(
            "Ticked parts get their own separate PNGs per view,\n"
            "auto-zoomed to fit just that component."),
            font=("Segoe UI", 8), bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

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

        # Select all / none helpers
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
        self.v_delta_mode = tk.StringVar(value="auto")
        tk.Radiobutton(dr_frame, text="Auto", variable=self.v_delta_mode,
                       value="auto", font=FONT, bg=BG,
                       command=self._toggle_delta_range).pack(side="left")
        tk.Radiobutton(dr_frame, text="Fixed:", variable=self.v_delta_mode,
                       value="fixed", font=FONT, bg=BG,
                       command=self._toggle_delta_range)\
            .pack(side="left", padx=(PAD,0))
        self.v_delta_val = tk.StringVar(value="0.1")
        self.delta_entry = tk.Entry(dr_frame, textvariable=self.v_delta_val,
                                    width=7, font=FONT,
                                    state="disabled", bg=BG_D)
        self.delta_entry.pack(side="left", padx=PAD_SM); r += 1

        tk.Label(f,
                 text="Geometry:  Gray=both  |  Magenta=new (B only)  |  Brown=removed (A only)",
                 font=("Segoe UI", 8), bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

        # Interactive delta-HTML output + quality (shares A/B/labels/range above)
        self.v_delta_html_out = tk.StringVar()
        labeled_entry(f, "Delta .html file (optional)", self.v_delta_html_out, r,
                      browse=lambda: self.v_delta_html_out.set(
                          filedialog.asksaveasfilename(
                              defaultextension=".html",
                              filetypes=[("HTML files", "*.html")],
                              initialfile="surface_delta_Cp.html"))); r += 1

        tk.Label(f, text="Delta HTML quality", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        dqf = tk.Frame(f, bg=BG)
        dqf.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_delta_html_quality = tk.StringVar(value="medium")
        for i, (lbl, val) in enumerate([
            ("Low", "low"), ("Medium", "medium"), ("High", "high")]):
            tk.Radiobutton(dqf, text=lbl, variable=self.v_delta_html_quality,
                           value=val, font=FONT, bg=BG)\
                .grid(row=0, column=i, padx=PAD)
        r += 1

        bf2 = tk.Frame(f, bg=BG)
        bf2.grid(row=r, column=0, columnspan=3, pady=PAD, padx=PAD, sticky="e")
        tk.Button(bf2, text="Delta HTML", command=self._run_delta_html,
                  bg="#7B1FA2", fg="white", font=FONT_B,
                  relief="flat", padx=12, pady=6, cursor="hand2")\
            .pack(side="right", padx=(PAD_SM, 0))
        tk.Button(bf2, text="Render Delta PNG", command=self._run_delta,
                  **BTN_RUN).pack(side="right")
        self.status_delta = tk.Label(bf2, text="", font=FONT, bg=BG, fg="#555")
        self.status_delta.pack(side="right", padx=PAD); r += 1

        # ── Interactive HTML export ───────────────────────────────
        section_label(f, "Interactive HTML  (open in any browser)", r); r += 2
        tk.Label(f, text=(
            "Exports a self-contained .html file — drag to rotate, scroll to zoom,\n"
            "click legend to toggle parts on/off. No software installation required."),
            font=("Segoe UI", 8), bg=BG, fg="#555")\
            .grid(row=r, column=0, columnspan=3, sticky="w",
                  padx=PAD, pady=PAD_SM); r += 1

        self.v_html_out = tk.StringVar()
        labeled_entry(f, "Output .html file", self.v_html_out, r,
                      browse=lambda: self.v_html_out.set(
                          filedialog.asksaveasfilename(
                              defaultextension=".html",
                              filetypes=[("HTML files", "*.html")],
                              initialfile="surface_Cp.html"))); r += 1

        tk.Label(f, text="Quality", font=FONT, bg=BG, anchor="w")\
            .grid(row=r, column=0, sticky="w", padx=PAD, pady=PAD_SM)
        qf = tk.Frame(f, bg=BG)
        qf.grid(row=r, column=1, sticky="w", padx=PAD)
        self.v_html_quality = tk.StringVar(value="medium")
        for i, (lbl, val, tip) in enumerate([
            ("Low  (~75MB, fast)",    "low",    "20k tris/part"),
            ("Medium  (~100MB)",      "medium", "60k tris/part"),
            ("High  (~200MB, slow)",  "high",   "200k tris/part"),
        ]):
            tk.Radiobutton(qf, text=lbl, variable=self.v_html_quality,
                           value=val, font=FONT, bg=BG)\
                .grid(row=0, column=i, padx=PAD)
        r += 1

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

    def refresh_scalars(self):
        keys = list(cfg.SURFACE_SCALARS.keys())
        self.scalar_menu['values'] = keys
        if keys and self.v_scalar.get() not in keys:
            self.v_scalar.set(keys[0])

    def _get_views(self):
        return [v for v, bv in self.view_vars.items() if bv.get()] or None

    def _run_single(self):
        geom    = self.v_geom.get().strip()
        surface = self.v_surface.get().strip() or None
        out_dir = self.v_out_png.get().strip()
        skey    = self.v_scalar.get().strip()
        views   = self._get_views()
        parts_to_export = [p for p, v in self.part_vars.items() if v.get()]
        if not geom:
            messagebox.showwarning("Missing", "Set geometry .case path."); return
        if not surface:
            if not messagebox.askyesno("No surface data",
                "Surface data .case is empty — geometry will render without "
                "any scalar colouring (all grey).\n\nContinue anyway?"):
                return
        try:
            figh  = float(self.v_figh.get())
            dpi   = int(self.v_dpi.get())
            angle = float(self.v_angle.get())
        except:
            messagebox.showerror("Invalid", "Height/DPI/angle must be numbers."); return

        self.status_single.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr, os
                scfg = cfg.SURFACE_SCALARS.get(skey, cfg.SCALARS.get(skey, {}))
                cmap = scfg.get('cmap', sr.make_cp_cmap())
                clim = (scfg.get('vmin', -3.0), scfg.get('vmax', 1.0))
                arr  = scfg.get('array', skey)
                stem = f'surface_{skey}'

                prepared = sr.prepare_surfaces(
                    geom, surface, feature_angle=angle)

                # Full car — one PNG per view
                print(f"\n[surface] Rendering full-car views...")
                sr.render_views(prepared, arr, cmap=cmap, clim=clim,
                                views=views, window_size=(1600, 1000),
                                bg_color='white',
                                output_dir=out_dir, file_stem=stem)

                # Per-part individual exports (auto-zoomed)
                if parts_to_export:
                    print(f"\n[surface] Rendering {len(parts_to_export)} "
                          f"individual parts...")
                    sr.render_part_views(
                        prepared, parts_to_export, arr,
                        cmap=cmap, clim=clim,
                        views=views, window_size=(1600, 1000),
                        bg_color='white',
                        output_dir=out_dir, file_stem=stem)

                self.status_single.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_single.config(text="Error", fg="#C62828")
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()
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
            try: fixed_range = float(self.v_delta_val.get())
            except:
                messagebox.showerror("Invalid", "Fixed range must be a number.")
                return
        scfg = cfg.SCALARS.get(skey, {})
        arr  = scfg.get('array', skey)
        la   = self.v_label_a.get() or "Run A"
        lb   = self.v_label_b.get() or "Run B"
        self.status_delta.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr, matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt, matplotlib.colors as mc2, os
                angle = float(self.v_angle.get()) if self.v_angle.get() else 25.0
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
                sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                     norm=mc2.Normalize(vmin=clim[0],vmax=clim[1]))
                sm.set_array([])
                fig.colorbar(sm, ax=af, fraction=0.01, pad=0.02,
                             label=f'delta {skey}  ({lb} - {la})')
                plt.suptitle(f'Surface Delta -- {skey}  {lb} - {la}', fontsize=14)
                plt.tight_layout()
                out_path = os.path.join(out or '.', f'surface_delta_{skey}.png')
                fig.savefig(out_path, dpi=dpi, facecolor='white',
                            bbox_inches='tight')
                plt.close(fig)
                print(f"[surface] Saved -> {out_path}")
                self.status_delta.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_delta.config(text="Error", fg="#C62828")
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()

    def _run_delta_html(self):
        geom   = self.v_geom.get().strip()
        surf_a = self.v_surf_a.get().strip()
        surf_b = self.v_surf_b.get().strip()
        out_path = self.v_delta_html_out.get().strip()
        skey   = self.v_scalar.get().strip()
        quality = self.v_delta_html_quality.get()

        if not geom or not surf_a or not surf_b:
            messagebox.showwarning("Missing",
                "Set geometry and both surface .case paths."); return
        if not out_path:
            messagebox.showwarning("Missing",
                "Set the output Delta .html file path."); return

        fixed_range = None
        if self.v_delta_mode.get() == "fixed":
            try: fixed_range = float(self.v_delta_val.get())
            except:
                messagebox.showerror("Invalid", "Fixed range must be a number.")
                return

        quality_map = {'low': 20_000, 'medium': 60_000, 'high': 200_000}
        tris = quality_map.get(quality, 60_000)

        scfg  = cfg.SCALARS.get(skey, {})
        arr   = scfg.get('array', skey)
        label = scfg.get('label', skey)
        la    = self.v_label_a.get() or "Run A"
        lb    = self.v_label_b.get() or "Run B"

        self.status_delta.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr
                angle = float(self.v_angle.get()) if self.v_angle.get() else 25.0
                print("[surface] Preparing Run A...")
                prep_a = sr.prepare_surfaces(geom, surf_a, feature_angle=angle)
                print("[surface] Preparing Run B...")
                prep_b = sr.prepare_surfaces(geom, surf_b, feature_angle=angle)
                sr.export_interactive_delta_html(
                    prep_a, prep_b, arr,
                    label_a=la, label_b=lb, label=label,
                    fixed_range=fixed_range,
                    output_path=out_path,
                    triangles_per_part=tris,
                )
                self.status_delta.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_delta.config(text="Error", fg="#C62828")
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()

    def _run_html_export(self):
        geom    = self.v_geom.get().strip()
        surface = self.v_surface.get().strip() or None
        out_path = self.v_html_out.get().strip()
        skey    = self.v_scalar.get().strip()
        quality = self.v_html_quality.get()

        if not geom:
            messagebox.showwarning("Missing", "Set geometry .case path."); return
        if not out_path:
            messagebox.showwarning("Missing", "Set output .html file path."); return
        if not surface:
            messagebox.showwarning("Missing",
                "Set a surface data .case file to colour by scalars."); return

        quality_map = {'low': 20_000, 'medium': 60_000, 'high': 200_000}
        tris = quality_map.get(quality, 60_000)

        scfg  = cfg.SCALARS.get(skey, {})
        arr   = scfg.get('array', skey)
        cmap  = scfg.get('cmap', None)
        clim  = (scfg.get('vmin', -3.0), scfg.get('vmax', 1.0))
        label = scfg.get('label', skey)

        self.status_html.config(text="Running...", fg=ACCENT)
        self.app.notebook.select(self.app.log_tab)

        def _worker():
            try:
                import surface_render as sr
                angle = float(self.v_angle.get()) if self.v_angle.get() else 25.0
                prepared = sr.prepare_surfaces(geom, surface, feature_angle=angle)
                sr.export_interactive_html(
                    prepared, arr,
                    cmap=cmap, clim=clim, label=label,
                    output_path=out_path,
                    triangles_per_part=tris,
                )
                self.status_html.config(text="Done", fg="#2E7D32")
            except Exception as exc:
                self.status_html.config(text="Error", fg="#C62828")
                print(f"\n[ERROR] {exc}")
                import traceback; traceback.print_exc()
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CFD Slicer")
        self.configure(bg=BG)
        self.geometry("820x700")
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

        self.run_tab      = RunTab(self.notebook, self)
        self.compare_tab  = CompareTab(self.notebook, self)
        self.surface_tab  = SurfaceTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)
        self.log_tab      = LogTab(self.notebook, self)

        self.notebook.add(self.run_tab,      text="  Run  ")
        self.notebook.add(self.compare_tab,  text="  Compare  ")
        self.notebook.add(self.surface_tab,  text="  Surface  ")
        self.notebook.add(self.settings_tab, text="  Settings  ")
        self.notebook.add(self.log_tab,      text="  Log  ")

        redir = LogRedirector(self.log_tab.log_text, sys.stdout)
        sys.stdout = redir

        # ── Watermark / status bar ───────────────────────────────
        bar = tk.Frame(self, bg=BG_D)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, text="Ready", font=FONT, bg=BG_D,
                 fg="#555", anchor="w")            .pack(side="left", padx=PAD)
        tk.Label(bar, text="Julian G-A  ·  UTFR",
                 font=("Segoe UI", 8), bg=BG_D,
                 fg="#999", anchor="e")            .pack(side="right", padx=PAD)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()