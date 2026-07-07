#!/usr/bin/env python3
"""
create_bevdet_pkl_mini.py
=========================
v1.0-mini split entry point for BEVDet-format .pkl generation.

Does not depend on BEVNeXt's internal create_data_bevdet.py; directly
produces pkl files that NuScenesDataset can load.

Usage (can run from any directory):
  python create_bevdet_pkl_mini.py
  python create_bevdet_pkl_mini.py --dataroot /data/sets/nuscenes \
                                   --out_dir /data/output/mini_pkls

Output:
  OUT_DIR/
    bevdet-nuscenes-mini-train.pkl
    bevdet-nuscenes-mini-val.pkl

Implementation note:
  All data processing lives in `bevdet_pkl_common.py`. This file only keeps
  the v1.0-mini-specific version, paths, and pkl filenames.
"""

from nuscenes.utils.splits import mini_train, mini_val

from bevdet_pkl_common import parse_and_run


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — only edit here for the mini split
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT = "/data/sets/nuscenes"
OUT_DIR  = "./output/bevdet_pkls"
VERSION  = "v1.0-mini"

TRAIN_PKL_NAME = "bevdet-nuscenes-mini-train.pkl"
VAL_PKL_NAME   = "bevdet-nuscenes-mini-val.pkl"


if __name__ == '__main__':
    parse_and_run(
        version=VERSION,
        dataroot=DATAROOT,
        out_dir=OUT_DIR,
        train_scenes=mini_train,
        val_scenes=mini_val,
        train_pkl_name=TRAIN_PKL_NAME,
        val_pkl_name=VAL_PKL_NAME,
    )
