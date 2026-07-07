#!/usr/bin/env python3
"""
create_pa_labels_mini.py  —  v1.0-mini 版
========================================================
运行：
  python create_pa_labels_mini.py              # 标注 + 可视化 + 复制文件
  python create_pa_labels_mini.py --no_copy    # 仅生成 pkl，跳过文件复制
  python create_pa_labels_mini.py --vis_n 10   # 生成 10 张预览图（默认 5）
  python create_pa_labels_mini.py --vis_n -1   # 所有正样本帧都出图
  python create_pa_labels_mini.py --dataroot /data/sets/nuscenes \
                                   --outdir_base /data/output/pa_labels

输出目录：
  OUTDIR_BASE/
    full/
      maps/  samples/  sweeps/  v1.0-mini/
      phantom_labels_full.pkl          ← 所有帧，训练用
    positive/
      phantom_labels_positive.pkl      ← 仅分布分析
    negative/
      phantom_labels_negative.pkl      ← 仅分布分析
    preview/
      *_cam.png   三相机正负样本可视化
      *_bev.png   BEV 热力图 + 矢量叠加图
    stats.txt

实现说明：
  所有数据处理逻辑已抽取到 `pa_labels_common.py`。
  本文件仅保留 v1.0-mini  split 专用的路径/版本配置。
"""

from pa_labels_common import parse_and_run

# ═══════════════════════════════════════════════════════════════════════════
#  ★ 配置区（mini 版唯一需要改的地方）
# ═══════════════════════════════════════════════════════════════════════════
DATAROOT    = "/data/sets/nuscenes"
OUTDIR_BASE = "./output/pa_labels"
VERSION     = "v1.0-mini"


if __name__ == '__main__':
    parse_and_run(VERSION, DATAROOT, OUTDIR_BASE)
