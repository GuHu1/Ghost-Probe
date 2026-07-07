#!/usr/bin/env python3
"""
create_bevdet_pkl_full.py
=========================
v1.0-trainval split entry point for BEVDet-format .pkl generation.

Same logic as create_bevdet_pkl_mini.py; only the split changes:
  VERSION     = "v1.0-trainval"
  DATAROOT    = "/data/sets/nuscenes"
  OUT_DIR     = "./output/bevdet_pkls"
  scenes:     train / val (instead of mini_train / mini_val)

Runtime scale:
  mini:     404 samples  ≈ minutes
  trainval: 34149 samples ≈ hours

Usage:
  python create_bevdet_pkl_full.py
  python create_bevdet_pkl_full.py --dataroot /data/sets/nuscenes \
                                   --out_dir /data/output/trainval_pkls

Output:
  OUT_DIR/
    bevdet-nuscenes-trainval-train.pkl
    bevdet-nuscenes-trainval-val.pkl

Implementation note:
  All data processing lives in `bevdet_pkl_common.py`. This file only keeps
  the v1.0-trainval-specific version, paths, and pkl filenames.
"""

from nuscenes.utils.splits import train, val

from bevdet_pkl_common import parse_and_run


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — only edit here for the trainval split
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT = "/data/sets/nuscenes"
OUT_DIR  = "./output/bevdet_pkls"
VERSION  = "v1.0-trainval"

TRAIN_PKL_NAME = "bevdet-nuscenes-trainval-train.pkl"
VAL_PKL_NAME   = "bevdet-nuscenes-trainval-val.pkl"


if __name__ == '__main__':
    parse_and_run(
        version=VERSION,
        dataroot=DATAROOT,
        out_dir=OUT_DIR,
        train_scenes=train,
        val_scenes=val,
        train_pkl_name=TRAIN_PKL_NAME,
        val_pkl_name=VAL_PKL_NAME,
    )
