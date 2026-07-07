#!/usr/bin/env python3
"""
create_bevdet_pkl_mini.py  —  v1.0-mini 版
==========================================================
不依赖 BEVNeXt 内部的 create_data_bevdet.py，
直接生成 BEVDet 格式的 pkl，可被 NuScenesDataset 直接加载。

运行（在任意目录，不需要进入 BEVNeXt）：
  python create_bevdet_pkl_mini.py
  python create_bevdet_pkl_mini.py --dataroot /data/sets/nuscenes \
                                   --out_dir /data/output/mini_pkls

输出：
  OUT_DIR/
    bevdet-nuscenes-mini-train.pkl
    bevdet-nuscenes-mini-val.pkl

实现说明：
  所有数据处理逻辑已抽取到 `bevdet_pkl_common.py`。
  本文件仅保留 v1.0-mini split 专用的版本、路径和 pkl 文件名。
"""

from nuscenes.utils.splits import mini_train, mini_val

from bevdet_pkl_common import parse_and_run


# ═══════════════════════════════════════════════════════════════════════════
#  ★ 配置区（mini 版唯一需要改的地方）
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
