"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.

aero_report.py
==============
Parses STAR-CCM+ .log files and generates aerodynamic delta reports.

Functions
---------
parse_log(log_path)
    -> (n_iterations, data_dict)  where data_dict maps column name -> float

generate_aero_report(parent_n_iter, parent_data, new_n_iter, new_data,
                     output_dir, parent_label, new_label)
    -> str   path to saved report file
"""

import os
import re

# ── Column definitions ────────────────────────────────────────────
COLUMN_HEADERS = [
    "Iteration", "Continuity", "X-momentum", "Y-momentum", "Z-momentum",
    "Tke", "Sdr", "CdA", "ClA", "CyA", "CdA_mean", "ClA_mean", "CyA_mean",
    "aerobalance_front", "aerobalance_sideforce_front",
    "Cmx", "Cmy", "Cmz", "Cmx_mean", "Cmy_mean", "Cmz_mean",
    "Solver Iteration Elapsed Time (s)", "Total Solver Elapsed Time (s)",
    "ClA_convergence_monitor",
    "RW ClA", "RW CdA", "FW CdA", "FW ClA",
    "Undertray CdA", "Undertray ClA",
    "rad_mass_flow (kg/s)",      # optional — not always present
    "Whiskers CdA", "Whiskers ClA",
    "Bodywork ClA", "Bodywork CdA",
]

LOWER_IS_BETTER = {
    "CdA", "ClA", "CdA_mean", "ClA_mean",
    "RW ClA", "RW CdA", "FW CdA", "FW ClA",
    "Undertray CdA", "Undertray ClA",
    "Whiskers CdA", "Whiskers ClA",
    "Bodywork ClA", "Bodywork CdA",
}

HIGHER_IS_BETTER = set()

AEROBALANCE_TARGET    = 0.5
UNCERTAINTY_THRESHOLD = 0.01
FAILURE_ITERATIONS    = 500

REPORT_PROPERTIES = {
    "CdA", "ClA", "CdA_mean", "ClA_mean",
    "aerobalance_front", "aerobalance_sideforce_front",
    "RW ClA", "RW CdA", "FW CdA", "FW ClA",
    "Undertray CdA", "Undertray ClA",
    "Whiskers CdA", "Whiskers ClA",
    "Bodywork ClA", "Bodywork CdA",
}


# ── Log parser ────────────────────────────────────────────────────

def parse_log(log_path):
    """
    Parse a STAR-CCM+ .log file and extract the final iteration's values.

    Strategy
    --------
    1. Find the LAST header line (starts with whitespace + 'Iteration  Continuity')
    2. Determine which known columns are present in that header
    3. Extract the last numeric data row after the header
    4. Map tokens positionally to present column names

    Parameters
    ----------
    log_path : str

    Returns
    -------
    n_iter : int                  final iteration number (0 if not found)
    data   : dict[str, float]     column name -> value from last iteration
    """
    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[aero_report] Cannot read {log_path}: {e}")
        return 0, {}

    # Find last header line
    header_pattern = re.compile(r'^\s+Iteration\s+Continuity')
    last_header_idx = None
    for i, line in enumerate(lines):
        if header_pattern.match(line):
            last_header_idx = i

    if last_header_idx is None:
        print(f"[aero_report] No data header found in {log_path}")
        return 0, {}

    header_line = lines[last_header_idx]

    # Determine which known columns are present, preserving order
    present_cols = [c for c in COLUMN_HEADERS if c in header_line]

    # Find last numeric data row after the header
    data_pattern = re.compile(r'^\s+\d+\s+[\d.e+\-]')
    last_data_line = None
    for line in lines[last_header_idx + 1:]:
        if data_pattern.match(line):
            last_data_line = line

    if last_data_line is None:
        print(f"[aero_report] No data rows found after last header in {log_path}")
        return 0, {}

    tokens = last_data_line.strip().split()

    # Map tokens to columns positionally
    data   = {}
    n_iter = 0
    for col, tok in zip(present_cols, tokens):
        try:
            val = float(tok)
            data[col] = val
            if col == 'Iteration':
                n_iter = int(val)
        except ValueError:
            pass

    return n_iter, data


# ── Report generator ──────────────────────────────────────────────

def generate_aero_report(parent_n_iter, parent_data,
                          new_n_iter,    new_data,
                          output_dir,
                          parent_label="Parent",
                          new_label="New"):
    """
    Generate a structured aerodynamic delta report as a .txt file.

    Returns
    -------
    output_path : str
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "aero_deltas.txt")

    parent_failed = (parent_n_iter >= FAILURE_ITERATIONS)
    new_failed    = (new_n_iter    >= FAILURE_ITERATIONS)

    lines = []

    # ── Raw values for spreadsheet copy-paste ────────────────────
    lines.append(f"{parent_label}:")
    lines.append("  " + "  ".join(
        f"{k}={v:.6g}" for k, v in parent_data.items()
        if k in REPORT_PROPERTIES
    ))
    lines.append(f"{new_label}:")
    if not new_failed and new_data:
        lines.append("  " + "  ".join(
            f"{k}={v:.6g}" for k, v in new_data.items()
            if k in REPORT_PROPERTIES
        ))
    else:
        lines.append(f"  FAILED ({new_n_iter} iterations) -- no data")
    lines.append("")

    # ── Header ────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("AERODYNAMIC DELTA REPORT")
    lines.append(
        f"  Parent : {parent_label}  "
        f"(Iteration {parent_n_iter}"
        f"{'  *** FAILED ***' if parent_failed else ''})"
    )
    lines.append(
        f"  New    : {new_label}  "
        f"(Iteration {new_n_iter}"
        f"{'  *** FAILED ***' if new_failed else ''})"
    )
    if parent_failed or new_failed:
        lines += [
            "",
            "  WARNING: One or both sims hit 500 iterations -- considered failures.",
            "  Deltas not shown. Parent values listed for reference only.",
        ]
    lines.append("=" * 72)

    # ── Failure mode — parent values only ────────────────────────
    if parent_failed or new_failed:
        lines += [
            "",
            "PARENT VALUES (no deltas -- comparison invalid)",
            "-" * 72,
            f"{'Property':<42} {'Parent':>12}",
            "-" * 72,
        ]
        for name in COLUMN_HEADERS:
            if name in REPORT_PROPERTIES and name in parent_data:
                lines.append(f"{name:<42} {parent_data[name]:>12.4f}")
        lines.append("=" * 72)
        _write(output_path, lines)
        return output_path

    # ── Build comparison rows ─────────────────────────────────────
    improved, degraded, neutral, uncertain = [], [], [], []
    key_rows = []

    for name in COLUMN_HEADERS:
        if name not in REPORT_PROPERTIES:
            continue
        p_val = parent_data.get(name)
        n_val = new_data.get(name)
        if p_val is None or n_val is None:
            continue

        delta    = n_val - p_val
        low_conf = abs(delta) < UNCERTAINTY_THRESHOLD

        if low_conf:
            result = "LOW CONF"
            uncertain.append((name, delta))
        elif name == "aerobalance_front":
            p_dist = abs(p_val - AEROBALANCE_TARGET)
            n_dist = abs(n_val - AEROBALANCE_TARGET)
            if n_dist < p_dist:
                result = "BETTER -> 45%"
                improved.append((name, delta))
            elif n_dist > p_dist:
                result = "WORSE -> 45%"
                degraded.append((name, delta))
            else:
                result = "UNCHANGED"
        elif name in LOWER_IS_BETTER:
            if delta < 0:
                result = "BETTER"
                improved.append((name, delta))
            elif delta > 0:
                result = "WORSE"
                degraded.append((name, delta))
            else:
                result = "UNCHANGED"
        elif name in HIGHER_IS_BETTER:
            if delta > 0:
                result = "BETTER"
                improved.append((name, delta))
            elif delta < 0:
                result = "WORSE"
                degraded.append((name, delta))
            else:
                result = "UNCHANGED"
        else:
            result = "---"
            neutral.append((name, delta))

        key_rows.append((name, p_val, n_val, delta, result))

    # ── Verdict ───────────────────────────────────────────────────
    lines += ["", "=" * 72, "VERDICT", "-" * 72]
    if improved:
        lines.append("  Improved:")
        for name, delta in improved:
            lines.append(f"    {name:<40} {delta:>+12.4f}")
    if degraded:
        lines.append("  Degraded:")
        for name, delta in degraded:
            lines.append(f"    {name:<40} {delta:>+12.4f}")
    if neutral:
        lines.append("  Informational (no target):")
        for name, delta in neutral:
            lines.append(f"    {name:<40} {delta:>+12.4f}")
    if uncertain:
        lines.append(f"  Low confidence (|delta| < {UNCERTAINTY_THRESHOLD}):")
        for name, delta in uncertain:
            lines.append(f"    {name:<40} {delta:>+12.4f}")

    # ── Key properties summary ────────────────────────────────────
    lines += [
        "", "=" * 72,
        "KEY PROPERTIES  (delta = new - parent)",
        "-" * 72,
        f"{'Property':<42} {'Parent':>10} {'New':>10} {'Delta':>10}  Result",
        "-" * 72,
    ]
    for name, p_val, n_val, delta, result in key_rows:
        lines.append(
            f"{name:<42} {p_val:>10.4f} {n_val:>10.4f} {delta:>+10.4f}  {result}"
        )

    # ── Full delta table ──────────────────────────────────────────
    all_names = [c for c in COLUMN_HEADERS
                 if c in parent_data and c in new_data]
    lines += [
        "", "=" * 72,
        "FULL DELTA TABLE  (delta = new - parent)",
        "-" * 72,
        f"{'Property':<42} {'Parent':>10} {'New':>10} {'Delta':>10}  Note",
        "-" * 72,
    ]
    for name in all_names:
        p_val = parent_data[name]
        n_val = new_data[name]
        delta = n_val - p_val
        note  = "LOW CONF" if abs(delta) < UNCERTAINTY_THRESHOLD else ""
        lines.append(
            f"{name:<42} {p_val:>10.4f} {n_val:>10.4f} {delta:>+10.4f}  {note}"
        )
    lines.append("=" * 72)

    _write(output_path, lines)
    return output_path


def _write(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aero_report] Saved -> {path}")