# Ghost-Probe

Ghost-Probe 是一个面向 nuScenes 数据集的 **Bird's-Eye-View（BEV）遮挡阴影区（Occlusion Shadow Zone, OSZ）** 计算与 **Phantom Agent（幽灵交通参与者）训练样本挖掘** 工具链。

它解决的核心问题：

- **OSZ 计算**：给定自车周视相机的稠密深度图，用 3D 体素投射 + 2D 自车射线投射，算出“从自车视角看被真实遮挡物挡住的区域”。
- **Phantom Agent 样本挖掘**：找出“之前被遮挡、当前突然可见”的真实车辆/行人，作为感知模型“预测盲区中可能存在的对象”的训练正样本。

---

## 目录结构

```
Ghost-Probe/
├── common/              # 项目唯一的 BEV 网格配置来源
│   └── bev_config.py    # 改一个数字，OSZ/PA_gen_v1/PA_gen_v2 全部联动
├── OSZ/                 # 几何 OSZ 计算（相机深度 → 体素 → BEV 阴影区）
│   ├── modules/         # ray_casting, drivable_filter, crf_refine
│   ├── utils/           # nuscenes_loader
│   ├── visualize/       # BEV / 相机可视化
│   └── run_osz_pipeline.py
├── PA_gen_v1/           # 基于 nuScenes 标注 visibility_token 生成 PA 训练标签
│   ├── pa_labels_common.py
│   ├── create_pa_labels_mini.py / create_pa_labels_full.py
│   ├── bevdet_pkl_common.py
│   ├── create_bevdet_pkl_mini.py / create_bevdet_pkl_full.py
│   └── pa_visible.py
├── PA_gen_v2/           # 基于几何 OSZ 挖掘“幽灵车辆出现事件”
│   ├── osz_source.py
│   ├── ghost_vehicle_miner.py
│   ├── trajectory.py
│   ├── visualize_events.py
│   ├── osz_source_viz.py
│   ├── test_units.py
│   ├── test_synthetic_e2e.py
│   └── run_pipeline.py
├── requirements.txt     # 依赖说明（含可选依赖）
└── README.md
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 中区分了**必需**与**可选**依赖：

| 依赖 | 作用 | 是否必需 |
|---|---|---|
| `numpy`, `matplotlib`, `scipy`, `Pillow`, `tqdm`, `pyquaternion` | 核心几何 / 可视化 | 是 |
| `nuscenes-devkit` | nuScenes 数据接口 | 是 |
| `shapely` | HD-map 可行驶区域过滤 | 否（缺则退化为不过滤） |
| `torch` | CNN 边界精修 / `--train_cnn` | 否（纯几何 OSZ 不需要） |

---

## BEV 网格：唯一的配置来源

项目所有模块共享同一个 BEV 网格定义：`common/bev_config.py`。

```python
BEV_EXTENT_M     = 50.0   # ego 为中心的半径范围（米）
BEV_RESOLUTION_M = 0.2    # 每格多少米 —— 改这一个数字，全项目跟着变
```

运行以下命令查看当前生效的网格参数：

```bash
python common/bev_config.py
```

输出示例：

```
BEV grid: 500x500 cells @ 0.2m/cell (±50.0m range, 250,000 cells total)
```

> **注意**：如果某次实验确实需要单独用不同分辨率跑 OSZ（例如做分辨率消融），
> `OSZ/run_osz_pipeline.py` 的 `--bev_res`、`--bev_range` 仍可在命令行单独覆盖。
> 但覆盖后需同步调整 PA_gen_v1/PA_gen_v2 的分辨率，否则网格不再对齐。

---

## 快速开始

### 1. OSZ 几何流程（无需真实数据即可验证）

```bash
cd OSZ/
python run_osz_pipeline.py --mock
```

`--mock` 使用合成数据跑通 3D 体素投射、2D 射线投射、可视化全流程，不依赖 nuScenes。

### 2. OSZ 几何流程（真实 nuScenes 数据）

```bash
cd OSZ/
python run_osz_pipeline.py --dataroot /data/sets/nuscenes --version v1.0-mini
```

### 3. PA_gen_v2 幽灵车辆挖掘（完整流程）

```bash
cd PA_gen_v2/
python run_pipeline.py --dataroot /data/sets/nuscenes --version v1.0-mini
```

这会依次执行：单元测试 → 单帧 OSZ 可视化检查 → 全量挖掘 → 事件可视化。

### 4. PA_gen_v1 基于标注 visibility 生成标签

```bash
cd PA_gen_v1/
python create_pa_labels_mini.py --dataroot /data/sets/nuscenes \
                                --outdir_base ./output/pa_labels
```

---

## 各模块说明

### `OSZ/` — 几何遮挡阴影区

核心流程：

1. **Stage 1+2**：对每路相机做 3D 体素投射，得到 `V_occ^c`（相机视角下的真实遮挡物表面体素）。
2. **Stage 3**：沿 Z 轴 max-pool，得到每路相机的 BEV 遮挡掩码 `M_occ^c`。
3. **Stage 4a**：以自车为中心做 2D BEV 射线投射，得到原始几何 OSZ `M_OSZ`。
4. **Stage 4c（可选）**：将原始 OSZ 与 HD-map 可行驶区域取交集，得到 **PA-relevant OSZ**。

> 原始几何 OSZ 在密集城区可能覆盖 70-80% BEV 区域，这是**预期行为**（建筑物阴影）。
> 真正用于 Phantom Agent 挖掘的是“原始 OSZ ∩ 可行驶区域”，不是原始 OSZ。

关键文件：

| 文件 | 作用 |
|---|---|
| `OSZ/modules/ray_casting.py` | 3D 体素投射 + 2D 自车射线投射（纯 numpy，无 torch 强依赖） |
| `OSZ/modules/drivable_filter.py` | 原始 OSZ ∩ HD-map 可行驶区域 = PA-relevant OSZ |
| `OSZ/modules/crf_refine.py` | 可选 CNN/CRF 边界精修（依赖 torch） |
| `OSZ/utils/nuscenes_loader.py` | 按 sample_token 构建单帧相机深度图、内外参 |
| `OSZ/run_osz_pipeline.py` | 主入口，支持 `--mock`、CLI 覆盖 BEV 参数 |

### `PA_gen_v1/` — 基于 nuScenes visibility 标签

直接读取 nuScenes 标注的 `visibility_token`，找出“过去被强遮挡、当前清晰可见”的标注链，生成 Phantom Agent 的 BEV 热力图训练标签。

- 不依赖自己计算的几何 OSZ。
- 运行快，但完全受限于标注质量。

| 文件 | 作用 |
|---|---|
| `create_pa_labels_mini.py` / `create_pa_labels_full.py` | 生成 PA 训练标签 pkl |
| `bevdet_pkl_common.py` | 共享的 BEVDet 格式 pkl 生成逻辑 |
| `create_bevdet_pkl_mini.py` / `create_bevdet_pkl_full.py` | 生成 BEVDet 格式 train/val pkl |
| `pa_visible.py` | 直接可视化遮挡标注 |

所有入口脚本都支持 `--dataroot` / `--outdir_base` 命令行覆盖。

### `PA_gen_v2/` — 基于几何 OSZ 的幽灵车辆挖掘

不读 `visibility_token`，而是用 `OSZ/` 算出的真实几何遮挡区去判断车辆是否“被挡住”，再挖掘“从遮挡区冒出来”的幽灵车辆出现事件。

详细说明（含坐标轴约定、可行驶区域过滤、回溯帧三态判定的修复）见：

- **[PA_gen_v2/README.md](PA_gen_v2/README.md)**

| 文件 | 作用 |
|---|---|
| `osz_source.py` | 桥接层：把 `OSZ/` 包装成 PA_gen_v2 的单帧查询接口 |
| `ghost_vehicle_miner.py` | 主挖掘逻辑，输出 JSON 事件列表 |
| `trajectory.py` | 完整轨迹插值，修复“缺标注就默认被遮挡”的隐藏假设 |
| `visualize_events.py` | 事件可视化 |
| `osz_source_viz.py` | 单帧 OSZ 几何检查 |
| `test_units.py` / `test_synthetic_e2e.py` | 单元测试与合成场景端到端测试 |
| `run_pipeline.py` | 一键跑完整流程 |

---

## 坐标轴约定（重要）

`OSZ/` 与 `PA_gen_v2/` 使用一致的 ego-centric BEV 约定：

- ego-x：车辆前进方向（BEV 数组 axis-0）
- ego-y：车辆左侧（BEV 数组 axis-1）
- BEV 数组形状：`(nx, ny)`，即 `array[i, j]` 中 `i ↔ ego-x`，`j ↔ ego-y`

旧的 `PA_gen_v2/osz_geometry_legacy.py` 使用图像式 `(row, col) = (ego-y, ego-x)`，已废弃。
**这两个约定在正方形网格下互换不会报错，只会把结果沿对角线镜像**，务必注意。

---

## 可选依赖的优雅降级

- **缺少 `shapely`**：`drivable_filter.py` 无法加载，但几何 OSZ 仍可运行；PA-relevant OSZ 退化为原始 OSZ，并打印一次性警告。
- **缺少 `torch`**：`crf_refine.py` 与 `--train_cnn` 不可用；纯几何 OSZ 与 PA_gen_v2 不受影响。

---

## 测试

```bash
cd PA_gen_v2/
python test_units.py
python test_synthetic_e2e.py
```

`test_units.py` 与 `test_synthetic_e2e.py` **不需要真实 nuScenes 数据**，可快速验证坐标变换、OSZ 单遮挡物、坐标轴顺序、可行驶区域降级、轨迹三态逻辑等。

---

## 常见问题

**Q：为什么原始 OSZ 覆盖了 70-80% 的 BEV？是不是 bug？**

A：在密集城区，建筑物、停放的车辆会在 BEV 上投下大片几何阴影，这是**正确的几何结果**。
Phantom Agent 挖掘用的是 `PA-relevant OSZ = 原始 OSZ ∩ 可行驶区域`，已过滤掉建筑物内部等车辆不可能出现的位置。
运行 `PA_gen_v2/osz_source_viz.py` 时重点看第 4 格（PA-relevant OSZ）和第 6 格（GT 车辆框对照）。

**Q：我可以不用 nuScenes 地图过滤吗？**

A：可以。不安装 `shapely` 即可自动退化；或者安装后也可以直接调用 `get_osz_for_sample()` 获取原始 OSZ。

**Q：BEV 分辨率改多少合适？**

A：默认 0.2m/格在 50m 半径下得到 500×500 网格。分辨率越低（格子越大）速度越快但细节越差；分辨率越高越慢。
改 `common/bev_config.py` 中的 `BEV_RESOLUTION_M` 即可全局生效。

---

## 许可证

本项目代码按 MIT 许可证发布（如无特殊说明）。
