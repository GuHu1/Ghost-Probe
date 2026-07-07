#!/usr/bin/env python3
"""
create_pa_labels_full.py
========================
v1.0-trainval split entry point for PA label generation.

Same logic as create_pa_labels_mini.py; only the split changes:
  VERSION     = "v1.0-trainval"
  DATAROOT    = "/data/sets/nuscenes"
  OUTDIR_BASE = "./output/pa_labels"

Runtime scale:
  mini:     404 samples  ≈ minutes
  trainval: 34149 samples ≈ hours (use --no_copy to test logic first)

Usage:
  python create_pa_labels_full.py
  python create_pa_labels_full.py --no_copy    # pkls only, skip file copy
  python create_pa_labels_full.py --vis_n 20   # 20 preview images
  python create_pa_labels_full.py --dataroot /data/sets/nuscenes \
                                   --outdir_base /data/output/pa_labels

Implementation note:
  All data processing lives in `pa_labels_common.py`. This file only keeps
  the v1.0-trainval-specific version, paths, and output layout.
"""

from pa_labels_common import parse_and_run

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — only edit here for the trainval split
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT    = "/data/sets/nuscenes"
OUTDIR_BASE = "./output/pa_labels"
VERSION     = "v1.0-trainval"


if __name__ == '__main__':
    parse_and_run(VERSION, DATAROOT, OUTDIR_BASE)
