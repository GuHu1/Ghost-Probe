#!/usr/bin/env python3
"""
create_pa_labels_full.py  —  v1.0-trainval 版
========================================================
与 create_pa_labels_mini.py 逻辑完全一致，仅 split 不同：
  VERSION     = "v1.0-trainval"
  DATAROOT    = "/data/sets/nuscenes"
  OUTDIR_BASE = "./output/pa_labels"

数据量差异：
  mini:     404 samples  ≈ 分钟级
  trainval: 34149 samples ≈ 数小时（建议 --no_copy 先测逻辑）

运行：
  python create_pa_labels_full.py
  python create_pa_labels_full.py --no_copy    # 仅 pkl，跳过文件复制
  python create_pa_labels_full.py --vis_n 20   # 20 张预览图
  python create_pa_labels_full.py --dataroot /data/sets/nuscenes \
                                   --outdir_base /data/output/pa_labels

实现说明：
  所有数据处理逻辑已抽取到 `pa_labels_common.py`。
  本文件仅保留 v1.0-trainval split 专用的路径/版本配置。
"""

from pa_labels_common import parse_and_run

# ═══════════════════════════════════════════════════════════════════════════
#  ★ 配置区（trainval 版唯一需要改的地方）
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT    = "/data/sets/nuscenes"
OUTDIR_BASE = "./output/pa_labels"
VERSION     = "v1.0-trainval"


if __name__ == '__main__':
    parse_and_run(VERSION, DATAROOT, OUTDIR_BASE)
