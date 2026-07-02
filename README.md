# Ghost-Probe

BEV 遮挡阴影区（OSZ）+ Phantom Agent 数据挖掘。

## 目录结构

```
common/       项目唯一的 BEV 网格配置来源（改一个数字，OSZ/PA_gen_v1/PA_gen_v2 全部联动）
OSZ/          相机深度图 → 3D 体素投射 → 2D BEV 射线投射，得到遮挡阴影区
PA_gen_v1/   基于 nuScenes 标注 visibility_token 挖 Phantom Agent 训练标签
PA_gen_v2/       基于几何遮挡（复用 OSZ/ 的计算）挖"幽灵车辆出现事件"，方法与用途见 PA_gen_v2/README.md
```

## BEV 网格大小怎么改

三个模块以前各自写死了自己的 BEV 分辨率（OSZ 用 0.4m、preprocess 用 0.5m、
filter 用 0.2m），格子对不齐，任何"把 OSZ 结果喂给 Phantom Agent 训练"的用法都会
悄悄错位。现在统一改成从 `common/bev_config.py` 读取：

```python
# common/bev_config.py
BEV_EXTENT_M     = 50.0   # ego 为中心的半径范围（米）
BEV_RESOLUTION_M = 0.2    # 每格多少米 —— 改这一个数字，全项目跟着变
```

跑 `python common/bev_config.py` 可以直接看到当前生效的网格参数：

```
BEV grid: 500x500 cells @ 0.2m/cell (±50.0m range, 250,000 cells total)
```

`OSZ/run_osz_pipeline.py`、`PA_gen_v1/create_pa_labels_mini.py`、
`PA_gen_v1/create_pa_labels_full.py`、`PA_gen_v2/osz_source.py` 都从这里取值。
如果某次实验确实需要单独用不同分辨率跑 OSZ（比如做分辨率消融实验），
`OSZ/run_osz_pipeline.py` 的 `--bev_res` 等参数仍然可以在命令行单独覆盖，
只是覆盖之后这一次跑的网格就不再和 preprocess/filter 对齐了，除非你也
同步改了它们的分辨率。

`PA_gen_v1/create_bevdet_pkl_mini.py`、`create_bevdet_pkl_full.py`、
`PA_gen_v1/pa_visible.py` 不涉及 BEV 网格化（只产出原始 3D 框数据，或者是纯
标注可视化），不需要跟着这个配置改动。

## 各模块简述

- **OSZ/**：给定每个相机的稠密深度图，先做 3D 体素投射找出真实遮挡物表面
  （`OSZ/modules/ray_casting.py`），再对这个实心 BEV 占据网格做一次以自车为中心
  的 2D 射线投射，得到遮挡阴影区（OSZ）。`OSZ/run_osz_pipeline.py` 是主入口，
  `--mock` 模式下不需要真实 nuScenes 数据也能跑一遍验证几何逻辑。

- **PA_gen_v1/**：直接读 nuScenes 标注自带的 `visibility_token`，找"过去被
  强遮挡、之后突然变清晰可见"的车辆标注链，生成 Phantom Agent 的 BEV 热力图
  训练标签。不依赖任何自己算的遮挡几何。

- **PA_gen_v2/**：和 preprocess/ 目标类似（挖 Phantom Agent 正/负样本），但用
  OSZ/ 算出来的真实几何遮挡区去判断，而不是依赖标注自带的可见度字段。
  现在 `PA_gen_v2/` 的 OSZ 计算直接调用 `OSZ/modules/ray_casting.py`（通过
  `PA_gen_v2/osz_source.py` 这层薄适配），不再维护自己的第二套实现。
  详细说明、坐标轴约定的坑、以及"回溯帧没标注就默认判定被遮挡"这个隐藏假设的
  修复方案，见 [`filter/README.md`](PA_gen_v2/README.md)。
