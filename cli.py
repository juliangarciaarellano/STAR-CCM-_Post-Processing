"""
================================================================================
  CFD Slicer Pipeline
  Author  : Julian G-A
  Project : UTFR
--------------------------------------------------------------------------------
  Slice, interpolate and compare STAR-CCM+ EnSight Gold exports.
  Do not distribute without permission.
  
cli.py
======
Command-line interface for the CFD slicer.

Usage
-----
  python cli.py run   [options]
  python cli.py compare DIR_A DIR_B [options]
  python cli.py info  NPZ_PATH [options]
  python cli.py list  NPZ_ROOT [options]

Run examples
------------
  # Process all scalars, all axes, default resolution
  python cli.py run

  # Only Cp, only X-planes, 2mm resolution
  python cli.py run --scalars Cp --axes X --res 2

  # Only CpT and Cp, Y and Z planes
  python cli.py run --scalars Cp CpT --axes Y Z

  # Override paths without editing config.py
  python cli.py run --geom path/to/Geometry.case \\
                    --data path/to/Data.case \\
                    --out  path/to/output

Compare examples
----------------
  # Compare two run output directories
  python cli.py compare path/to/run1 path/to/run2 \\
                        --label-a "Baseline" --label-b "New config" \\
                        --scalars Cp --axes X

Info / list
-----------
  # Print metadata from a saved NPZ file
  python cli.py info path/to/NPZ/X/slice_X_014_+510.0mm.npz

  # List all available planes in an output directory
  python cli.py list path/to/output/NPZ
  python cli.py list path/to/output/NPZ --axis X
"""

import argparse
import sys
import os

import config as cfg


def cmd_run(args):
    import pipeline
    pipeline.run_pipeline(
        scalars       = args.scalars  or None,
        axes          = args.axes     or None,
        geom_case     = args.geom     or None,
        data_case     = args.data     or None,
        output_dir    = args.out      or None,
        resolution_mm = args.res      or None,
    )


def cmd_compare(args):
    import compare
    compare.compare_runs(
        dir_a      = args.dir_a,
        dir_b      = args.dir_b,
        scalars    = args.scalars or None,
        axes       = args.axes    or None,
        output_dir = args.out     or None,
        label_a    = args.label_a,
        label_b    = args.label_b,
        log_a      = args.log_a   or None,
        log_b      = args.log_b   or None,
        images     = not args.no_images,
    )


def cmd_info(args):
    import io_utils
    import numpy as np

    d = io_utils.load_plane(args.npz_path)
    print(f"\nFile: {args.npz_path}")
    print(f"  Plane axis     : {cfg.AXIS_LABEL[int(d['plane_axis'])]}")
    print(f"  Plane position : {float(d['plane_value_m'])*1000:+.1f} mm")
    print(f"  Plane index    : {int(d['plane_index'])}")
    print(f"  Resolution     : {float(d['resolution_mm']):.1f} mm/px")
    ext = d['extent']
    print(f"  Extent H       : [{ext[0]:.3f}, {ext[1]:.3f}] m")
    print(f"  Extent V       : [{ext[2]:.3f}, {ext[3]:.3f}] m")
    print(f"  Grid size      : {d['mask'].shape[1]} x {d['mask'].shape[0]} px")
    print(f"  Solid pixels   : {d['mask'].sum():,}  ({100*d['mask'].mean():.1f}%)")
    print(f"  Scalars stored :")
    for key in cfg.SCALARS:
        if key in d:
            arr = d[key]
            valid = arr[~np.isnan(arr) & ~d['mask']]
            print(f"    {key:10s}  shape={arr.shape}  "
                  f"fluid range=[{valid.min():.4f}, {valid.max():.4f}]  "
                  f"mean={valid.mean():.4f}")


def cmd_list(args):
    import io_utils
    planes = io_utils.find_planes(args.npz_root, axis=args.axis or None)
    print(f"\nPlanes in {args.npz_root}:")
    print(f"  {'Axis':4} {'#':4} {'Position':>12}  Path")
    print("  " + "-" * 65)
    for ax, pidx, pmm, path in planes:
        print(f"  {ax:4} {pidx:4d} {pmm:+10.1f}mm  {os.path.basename(path)}")
    print(f"\n  Total: {len(planes)} planes")


def build_parser():
    parser = argparse.ArgumentParser(
        prog='cli.py',
        description='CFD slice & interpolate pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── run ───────────────────────────────────────────────────────
    p_run = sub.add_parser('run', help='Run the slice + interpolate pipeline')
    p_run.add_argument('--scalars', nargs='+', metavar='NAME',
                       help=f'Scalars to process (default: all). '
                            f'Available: {list(cfg.SCALARS.keys())}')
    p_run.add_argument('--axes', nargs='+', choices=['X','Y','Z'],
                       metavar='AXIS',
                       help='Axes to slice (default: X Y Z)')
    p_run.add_argument('--res', type=int, metavar='MM',
                       help=f'Resolution in mm/px (default: {cfg.RESOLUTION_MM})')
    p_run.add_argument('--geom', metavar='PATH',
                       help='Override geometry .case path')
    p_run.add_argument('--data', metavar='PATH',
                       help='Override CFD data .case path')
    p_run.add_argument('--out', metavar='DIR',
                       help='Override output directory')

    # ── compare ───────────────────────────────────────────────────
    p_cmp = sub.add_parser('compare',
                            help='Compare two run output directories')
    p_cmp.add_argument('dir_a', metavar='DIR_A',
                       help='First run output directory (baseline)')
    p_cmp.add_argument('dir_b', metavar='DIR_B',
                       help='Second run output directory (new)')
    p_cmp.add_argument('--scalars', nargs='+', metavar='NAME',
                       help='Scalars to compare (default: all)')
    p_cmp.add_argument('--axes', nargs='+', choices=['X','Y','Z'],
                       metavar='AXIS',
                       help='Axes to compare (default: X Y Z)')
    p_cmp.add_argument('--out', metavar='DIR',
                       help='Output directory for diff plots + CSV')
    p_cmp.add_argument('--label-a', default='Run A',
                       help='Label for DIR_A in plots')
    p_cmp.add_argument('--label-b', default='Run B',
                       help='Label for DIR_B in plots')
    p_cmp.add_argument('--log-a', metavar='PATH',
                       help='STAR-CCM+ .log for DIR_A (enables aero report)')
    p_cmp.add_argument('--log-b', metavar='PATH',
                       help='STAR-CCM+ .log for DIR_B (enables aero report)')
    p_cmp.add_argument('--no-images', action='store_true',
                       help='Skip plane-image/NPZ diffing. Use with --log-a/--log-b '
                            'to produce an aero (job) comparison on its own.')

    # ── info ──────────────────────────────────────────────────────
    p_info = sub.add_parser('info',
                             help='Print metadata from a saved NPZ file')
    p_info.add_argument('npz_path', metavar='NPZ_PATH',
                        help='Path to .npz file')

    # ── list ──────────────────────────────────────────────────────
    p_list = sub.add_parser('list',
                             help='List all available planes in an output dir')
    p_list.add_argument('npz_root', metavar='NPZ_ROOT',
                        help='Root output directory (contains NPZ/ subdir)')
    p_list.add_argument('--axis', choices=['X','Y','Z'],
                        help='Filter to one axis')

    return parser


def main():
    parser = build_parser()

    # No arguments -> default to a full run
    if len(sys.argv) == 1:
        sys.argv.append('run')

    args = parser.parse_args()

    dispatch = {
        'run':     cmd_run,
        'compare': cmd_compare,
        'info':    cmd_info,
        'list':    cmd_list,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()