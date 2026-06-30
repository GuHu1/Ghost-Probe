#!/usr/bin/env python3
"""
run_pipeline.py
---------------
End-to-end pipeline runner for ghost vehicle data mining.

Usage:
    python run_pipeline.py --dataroot /path/to/nuscenes --version v1.0-mini

Steps (in order, stop on any failure):
    1. Unit tests              — coordinate math must be correct
    2. OSZ single-sample viz   — visually verify OSZ geometry
    3. Ghost vehicle mining    — mine all events from mini set
    4. Event visualization     — verify positive/negative samples visually

Do NOT skip step 1 and 2.  The visual check in step 2 is the fastest way
to catch a coordinate convention mistake before it silently corrupts step 3.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

OUT_DIR = Path('/home/claude/phantom_agent')


def run_step(label: str, cmd: list) -> None:
    print(f"\n{'━'*60}")
    print(f"  STEP: {label}")
    print(f"{'━'*60}")
    result = subprocess.run(cmd, cwd=str(OUT_DIR))
    if result.returncode != 0:
        print(f"\n[FATAL] Step '{label}' failed. Fix errors before continuing.")
        sys.exit(1)
    print(f"  ✓ {label} complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--sample_idx', type=int, default=5,
                        help='Sample index for OSZ visualization check')
    parser.add_argument('--lookback',   type=int, default=4)
    parser.add_argument('--min_osz',    type=int, default=1)
    parser.add_argument('--skip_steps', nargs='*', type=int, default=[],
                        help='Step numbers to skip (1-4). Use only if you '
                             'know what you are doing.')
    args = parser.parse_args()

    events_path = OUT_DIR / 'ghost_events_mini.json'
    py = sys.executable

    steps = [
        (1, "Unit tests",
         [py, 'test_units.py']),

        (2, "OSZ geometry visualization",
         [py, 'osz_geometry.py',
          '--dataroot', args.dataroot,
          '--version',  args.version,
          '--sample_idx', str(args.sample_idx)]),

        (3, "Ghost vehicle mining",
         [py, 'ghost_vehicle_miner.py',
          '--dataroot', args.dataroot,
          '--version',  args.version,
          '--out',      str(events_path),
          '--lookback', str(args.lookback),
          '--min_osz',  str(args.min_osz)]),

        (4, "Event visualization",
         [py, 'visualize_events.py',
          '--dataroot', args.dataroot,
          '--version',  args.version,
          '--events',   str(events_path),
          '--max_events', '12']),
    ]

    print(f"\nPhantom Agent — Ghost Vehicle Mining Pipeline")
    print(f"  dataroot : {args.dataroot}")
    print(f"  version  : {args.version}")
    print(f"  output   : {OUT_DIR}")

    for step_num, label, cmd in steps:
        if step_num in args.skip_steps:
            print(f"\n  [SKIP] Step {step_num}: {label}")
            continue
        run_step(f"[{step_num}/4] {label}", cmd)

    print(f"\n{'━'*60}")
    print("  Pipeline complete. Check these outputs:")
    print(f"  {OUT_DIR / 'osz_sample_viz.png'}   ← OSZ geometry check")
    print(f"  {OUT_DIR / 'events_positive.png'}   ← Ghost events (should show")
    print(f"                                         red dots inside red OSZ)")
    print(f"  {OUT_DIR / 'events_negative.png'}   ← Negatives (should be clean)")
    print(f"  {events_path}      ← Raw event data")
    print(f"{'━'*60}\n")


if __name__ == '__main__':
    main()
