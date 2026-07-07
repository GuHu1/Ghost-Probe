#!/usr/bin/env python3
"""
create_pa_labels_mini.py
========================
v1.0-mini split entry point for PA label generation.

Usage:
  python create_pa_labels_mini.py              # labels + vis + file copy
  python create_pa_labels_mini.py --no_copy    # pkls only, skip file copy
  python create_pa_labels_mini.py --vis_n 10   # render 10 previews (default 5)
  python create_pa_labels_mini.py --vis_n -1   # render all positive frames
  python create_pa_labels_mini.py --dataroot /data/sets/nuscenes \
                                   --outdir_base /data/output/pa_labels

Output tree:
  OUTDIR_BASE/
    full/
      maps/  samples/  sweeps/  v1.0-mini/
      phantom_labels_full.pkl          ← all frames, for training
    positive/
      phantom_labels_positive.pkl      ← distribution analysis only
    negative/
      phantom_labels_negative.pkl      ← distribution analysis only
    preview/
      *_cam.png   three-camera pos/neg visualization
      *_bev.png   BEV heatmap + vector overlay
    stats.txt

Implementation note:
  All data processing lives in `pa_labels_common.py`. This file only keeps
  the v1.0-mini-specific version, paths, and output layout.
"""

from pa_labels_common import parse_and_run

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — only edit here for the mini split
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT    = "/data/sets/nuscenes"
OUTDIR_BASE = "./output/pa_labels"
VERSION     = "v1.0-mini"


if __name__ == '__main__':
    parse_and_run(VERSION, DATAROOT, OUTDIR_BASE)
