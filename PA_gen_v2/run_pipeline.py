#!/usr/bin/env python3
"""
run_pipeline.py
---------------
End-to-end pipeline runner for ghost vehicle data mining.

Usage:
    python run_pipeline.py --dataroot /path/to/nuscenes --version v1.0-mini

Steps (in order, stop on any failure):
    1. Unit tests              — coordinate math + trajectory logic must be correct
    2. OSZ single-sample viz   — visually verify OSZ geometry (and OSZ-vs-GT
                                  agreement) via osz_source.py
    3. Ghost vehicle mining    — mine all events from the dataset
    4. Event visualization     — verify positive/negative samples visually

Do NOT skip step 1 and 2. The visual check in step 2 is the fastest way to
catch a coordinate convention mistake before it silently corrupts step 3.

Changes vs. the original version
---------------------------------
- OUT_DIR is no longer hardcoded to /home/claude/phantom_agent; it defaults
  to filter/output/ inside this repo, and can be overridden with --outdir.
- --dataroot has no fake default that would just fail deep inside NuScenes()
  — it's a required argument, same as before, but every downstream script
  now shares that same requirement instead of silently trying
  /data/nuscenes.
- Step 2 now calls osz_source_viz.py (OSZ/modules/ray_casting.py via
  osz_source.py) instead of the old osz_geometry.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent   # filter/ itself


def run_step(label: str, cmd: list, cwd: Path) -> None:
    print(f"\n{'━'*60}")
    print(f"  STEP: {label}")
    print(f"{'━'*60}")
    result = subprocess.run(cmd, cwd=str(cwd))
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
    parser.add_argument('--outdir', default=str(_THIS_DIR / 'output'),
                        help='Where to write all pipeline outputs '
                             '(default: filter/output/ inside this repo)')
    parser.add_argument('--skip_steps', nargs='*', type=int, default=[],
                        help='Step numbers to skip (1-4). Use only if you '
                             'know what you are doing.')
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / 'ghost_events_mini.json'
    py = sys.executable

    steps = [
        (1, "Unit tests",
         [py, 'test_units.py']),

        (2, "OSZ geometry visualization (osz_source.py -> OSZ/)",
         [py, 'osz_source_viz.py',
          '--dataroot', args.dataroot,
          '--version',  args.version,
          '--sample_idx', str(args.sample_idx),
          '--outdir', str(out_dir)]),

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
          '--max_events', '12',
          '--out_pos', str(out_dir / 'events_positive.png'),
          '--out_neg', str(out_dir / 'events_negative.png')]),
    ]

    print(f"\nGhost-Probe — Ghost Vehicle Mining Pipeline")
    print(f"  dataroot : {args.dataroot}")
    print(f"  version  : {args.version}")
    print(f"  output   : {out_dir}")

    for step_num, label, cmd in steps:
        if step_num in args.skip_steps:
            print(f"\n  [SKIP] Step {step_num}: {label}")
            continue
        run_step(f"[{step_num}/4] {label}", cmd, cwd=_THIS_DIR)

    print(f"\n{'━'*60}")
    print("  Pipeline complete. Check these outputs:")
    print(f"  {out_dir / 'osz_sample_viz.png'}     ← OSZ geometry + GT agreement check")
    print(f"  {out_dir / 'events_positive.png'}    ← Ghost events (should show")
    print(f"                                          red dots inside red OSZ)")
    print(f"  {out_dir / 'events_negative.png'}    ← Negatives (should be blue-only)")
    print(f"  {events_path}      ← Raw event data")
    print(f"{'━'*60}\n")


if __name__ == '__main__':
    main()
