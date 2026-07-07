#!/usr/bin/env python3
"""
create_bevdet_pkl_full.py  —  v1.0-trainval 版
==========================================================
与 create_bevdet_pkl_mini.py 逻辑完全一致，仅 split 不同：
  VERSION     = "v1.0-trainval"
  DATAROOT    = "/data/sets/nuscenes"
  OUT_DIR     = "./output/bevdet_pkls"
  split 来源: train / val（而非 mini_train / mini_val）

数据量差异：
  mini:     404 samples  ≈ 分钟级
  trainval: 34149 samples ≈ 数小时

运行：
  python create_bevdet_pkl_full.py
  python create_bevdet_pkl_full.py --dataroot /data/sets/nuscenes \
                                   --out_dir /data/output/trainval_pkls

输出：
  OUT_DIR/
    bevdet-nuscenes-trainval-train.pkl
    bevdet-nuscenes-trainval-val.pkl

实现说明：
  所有数据处理逻辑已抽取到 `bevdet_pkl_common.py`。
  本文件仅保留 v1.0-trainval split 专用的版本、路径和 pkl 文件名。
"""

from nuscenes.utils.splits import train, val

from bevdet_pkl_common import parse_and_run


# ═══════════════════════════════════════════════════════════════════════════
#  ★ 配置区（trainval 版唯一需要改的地方）
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
