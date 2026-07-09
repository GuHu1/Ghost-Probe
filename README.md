# Ghost-Probe

Ghost-Probe 是一个面向 nuScenes 数据集的 **Bird's-Eye-View（BEV）遮挡阴影区（Occlusion Shadow Zone, OSZ）** 计算与 **Phantom Agent（幽灵交通参与者）训练样本挖掘** 工具链。

它解决的核心问题：

- **OSZ 计算**：给定自车周视相机的稠密深度图，用 3D 体素投射 + 2D 自车射线投射，算出"从自车视角看被真实遮挡物挡住的区域"。
- **Phantom Agent 样本挖掘**：找出"之前被遮挡、当前突然可见"的真实车辆/行人，作为感知模型"预测盲区中可能存在的对象"的训练正样本。

---

## 目录结构

```
Ghost-Probe/
├── common/                  # 项目唯一的 BEV 网格配置来源
│   └── bev_config.py        # 改一个数字，OSZ/PA_gen_v1/PA_gen_v2 全部联动
│
├── OSZ/                     # 几何 OSZ 计算（相机深度 → 体素 → BEV 阴影区）
│   ├── modules/             # ray_casting, drivable_filter
│   ├── utils/               # nuscenes_loader
│   ├── visualize/           # BEV / 相机可视化
│   └── run_osz_pipeline.py
│
├── PA_gen_v1/               # 基于 nuScenes 标注 visibility_token 生成 PA 训练标签
│   ├── pa_labels_common.py
│   ├── create_pa_labels_mini.py / create_pa_labels_full.py
│   ├── bevdet_pkl_common.py
│   ├── create_bevdet_pkl_mini.py / create_bevdet_pkl_full.py
│   └── pa_visible.py
│
├── PA_gen_v2/               # 基于几何 OSZ 挖掘"幽灵车辆出现事件"
│   ├── osz_source.py        # 桥接层：包装 OSZ/ 为单帧查询接口 + 磁盘缓存
│   ├── ghost_vehicle_miner.py    # 主挖掘逻辑
│   ├── trajectory.py        # 完整轨迹插值（KNOWN/INTERPOLATED/NO_EVIDENCE 三态）
│   ├── visualize_events.py  # 事件可视化（GUI/headless/web 三模式 + 交互式 Event Browser）
│   ├── osz_source_viz.py    # 单帧 OSZ 几何检查
│   ├── analyze_distances.py # 距离分布统计 + 数据驱动阈值建议
│   ├── test_units.py / test_synthetic_e2e.py
│   ├── run_pipeline.py
│   └── osz_geometry_legacy.py  # 已废弃，仅作历史参考
│
├── requirements.txt         # 依赖说明（含可选依赖）
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
| `tornado` | matplotlib WebAgg 后端 | 否（仅 --browse 浏览器模式需要） |
| `tkinter` | matplotlib TkAgg 后端 | 通常随 CPython 自带 |

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
> `PA_gen_v2/osz_source.py` 的 OSZ 磁盘缓存会按 BEV/Z 参数的 config_hash
> 自动隔离，不同分辨率不会读到对方的脏缓存。

---

## 快速开始

### 1. PA_gen_v2 幽灵车辆挖掘（完整流程，推荐）

```bash
cd PA_gen_v2/
python run_pipeline.py --dataroot /path/to/nuscenes --version v1.0-mini
```

依次执行：单元测试(16项) → 单帧OSZ可视化检查 → 全量挖掘 → 事件可视化。

输出：
- `output/osz_sample_viz.png`     — OSZ几何 + GT车框对比
- `output/ghost_events_mini.json` — 挖掘出的事件数据
- `output/events_positive.png`    — 正样本网格
- `output/events_negative.png`    — 负样本网格

可选参数：
- `--sample_idx 5`   用于可视化的样本索引
- `--lookback 4`    回溯帧数
- `--min_osz 1`     最少OSZ帧数要求
- `--skip_steps 1 2` 跳过某些步骤（1=单元测试, 2=可视化, 3=挖掘, 4=事件图）

### 2. OSZ 几何流程（独立运行）

```bash
cd OSZ/
python run_osz_pipeline.py --dataroot /path/to/nuscenes --version v1.0-mini

# 或不用真实数据，纯合成 mock：
python run_osz_pipeline.py --mock
```

输出 PNG（在 `OSZ/output/`）：
- `frame_XXXX_osz.png`        — OSZ explained：障碍物(橙)+阴影(黑)+道路(深灰)
- `frame_XXXX_pa.png`         — GT车框 + phantom候选
- `frame_XXXX_comparison.png` — 相机深度图 vs BEV OSZ 对比

### 3. 事件质检（交互式）

```bash
cd PA_gen_v2/

# GUI 模式（弹出 matplotlib 窗口，n/p/r/q 翻页）
python visualize_events.py --dataroot /path/to/nuscenes --browse

# 终端+PNG 模式（无 GUI 依赖）
python visualize_events.py --dataroot /path/to/nuscenes --browse --headless

# 浏览器图廊模式（最推荐，绕开 matplotlib 窗口卡顿）
python visualize_events.py --dataroot /path/to/nuscenes --web
```

### 4. 距离分布统计 + 阈值建议

```bash
cd PA_gen_v2/
python analyze_distances.py
```

### 5. PA_gen_v1 基于标注 visibility 生成标签

```bash
cd PA_gen_v1/
python create_pa_labels_mini.py --dataroot /path/to/nuscenes \
                                --outdir_base ./output/pa_labels
```

---

## 各模块说明

### `OSZ/` — 几何遮挡阴影区

核心流程：

1. **LiDAR 多帧聚合**：当前帧 + 前 N 帧历史 sweep（仅 past），ego-motion 补偿后地面过滤。
2. **直接点云体素化**：绕过深度图中转，点云直接 bin 到 3D 体素网格（x, y, z）。
3. **Z 轴 max-pool**：得到 solid BEV 障碍物掩码 `bev_occ`。
4. **2D 射线投射**：从 ego 在各方向发 ray，碰到 `bev_occ` 后标记为 OSZ。
5. **可行驶区域过滤**：原始 OSZ ∩ HD-map 可行驶区域 = PA-relevant OSZ（仅道路上的阴影）。

> CRF 后处理已移除；不再依赖深度图插值。OSZ 纯靠 LiDAR 点云驱动。

关键文件：

| 文件 | 作用 |
|---|---|
| `OSZ/modules/ray_casting.py` | 3D 体素投射 + **向量化** 2D BEV 射线投射 + 直接点云体素化 |
| `OSZ/modules/drivable_filter.py` | 原始 OSZ ∩ HD-map 可行驶区域 = PA-relevant OSZ |
| `OSZ/utils/nuscenes_loader.py` | LiDAR 加载、地面过滤、**多帧聚合（仅 past sweep）** |
| `OSZ/visualize/bev_viz.py` | BEV 可视化，统一 PA_gen_v2 调色板 |
| `OSZ/run_osz_pipeline.py` | OSZ 主入口，支持 `--mock` |

### `PA_gen_v1/` — 基于 nuScenes visibility 标签

直接读取 nuScenes 标注的 `visibility_token`，找出"过去被强遮挡、当前清晰可见"的标注链，生成 Phantom Agent 的 BEV 热力图训练标签。

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

用 `OSZ/` 算出的真实几何遮挡区判断车辆是否"被挡住"，挖掘"从遮挡区冒出来"的幽灵车辆出现事件。

**PA 判定逻辑（关键）：**
- 车辆 BEV 框栅格化到网格上，逐格判断。
- **LiDAR 命中排除**：框内任一一格被 `bev_occ`（LiDAR 表面）覆盖 → 车辆被传感器看到 → **不是 PA**（它是 occluder 本人）。
- **全部在阴影中**：框内所有格都在 `osz_pa` 内且零格命中 → **真 PA**。
- 仅支持 **vehicle.\*（含 bicycle/motorcycle）**；行人已移除。

| 文件 | 作用 |
|---|---|
| `osz_source.py` | 桥接层：包装 OSZ/ 为单帧查询接口，含 `is_box_occluded_not_occluder`、磁盘缓存 |
| `ghost_vehicle_miner.py` | 主挖掘逻辑，KNOW 帧整框检查 + INTERPOLATED 帧中心点回退 |
| `trajectory.py` | 轨迹插值（KNOWN/INTERPOLATED/NO_EVIDENCE 三态） |
| `visualize_events.py` | 事件可视化，GUI/headless/web 三模式 |
| `osz_source_viz.py` | 单帧 OSZ 几何检查 |
| `analyze_distances.py` | 距离分布统计 |
| `test_units.py` | 16 项单元测试 |
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

## 可视化三模式

`PA_gen_v2/visualize_events.py` 提供三种浏览方式，覆盖本地 GUI 用户、远程服务器用户、想看大量事件的用户：

| 模式 | 启动方式 | 优点 | 缺点 |
|---|---|---|---|
| **GUI 浏览器** | `--browse` | 交互流畅，原生窗口 n/p/r/q 翻页 | 需要 tkinter/Qt；远程服务器无显示时不可用 |
| **Headless 终端** | `--browse --headless` | 零 GUI 依赖；每事件独立 PNG | 终端输入 + 看图软件双窗口 |
| **Web 图廊** | `--web` | 浏览器翻页，零窗口卡顿；最推荐 | 首次渲染 5-15s/事件（第二次秒出） |

三种模式画的内容**完全一致**：2×3 布局，每帧显示该帧**自己的** PA-relevant OSZ + ego marker + 车辆位置（颜色按 was_in_osz 判定）+ drivable 路面 + 全场景 BEV 真值框（青色车辆 + 黄色行人 + 被追踪车辆高亮框 + 朝向箭头）+ HD 地图车道线。详见 [PA_gen_v2/README.md](PA_gen_v2/README.md)。

---

## 性能特性

- **`cast_osz_2d` 向量化**：`OSZ/modules/ray_casting.py` 的 2D 阴影射线投射已从纯 Python 双 for 循环改为 numpy 向量化，500×500 网格**单 sample 从 25s 降到 0.5s（提速 50 倍）**。20 次随机小网格 bit-for-bit 一致测试通过。
- **OSZ 磁盘缓存**：`PA_gen_v2/osz_source.py` 的 `get_pa_relevant_osz_for_sample()` 采用三层缓存：内存（per-session）→ 磁盘（per-config-hash 的 .npz 文件）→ 计算。第二次跑 `visualize_events.py` 秒出（直接 load npz）。改 BEV/Z 参数时 config_hash 自动切换子目录，不会读到旧配置脏缓存。

---

## 可选依赖的优雅降级

- **缺少 `shapely`**：`drivable_filter.py` 无法加载，但几何 OSZ 仍可运行；PA-relevant OSZ 退化为原始 OSZ，并打印一次性警告。
- **缺少 `tornado`**：matplotlib WebAgg 后端不可用，GUI 浏览器自动尝试 TkAgg/QtAgg。
- **缺少 `tkinter` / `Qt`**：GUI 浏览器自动降级为 headless 终端模式或 WebAgg。

---

## 测试

```bash
cd PA_gen_v2/
python test_units.py
python test_synthetic_e2e.py
```

`test_units.py` 与 `test_synthetic_e2e.py` **不需要真实 nuScenes 数据**，可快速验证坐标变换、OSZ 单遮挡物、坐标轴顺序、可行驶区域降级、轨迹三态逻辑等。

---

## 已知踩坑经验（重要）

这里列出在 nuScenes / OSZ 集成中**已踩过、且容易复现**的陷阱：

1. **正方形 BEV 网格的轴顺序静默转置**：`OSZ/` 的 `(nx, ny)` 数组是 `array[i, j]` 中 `i=ego-x`、`j=ego-y`；图像式 `(row, col) = (ego-y, ego-x)` 是相反的。**正方形网格下两者数值完全一样，只是镜像**——写错不会报错，结果沿对角线翻转。`osz_source.py` 只暴露 `bev_xy_to_ij()` / `ij_to_bev_xy()` 两个显式命名的转换函数，避免 `mask[j, i]` 这种容易顺手写反的接口。`test_units.py` 用**左右不对称**的遮挡物专门做回归测试（对称墙测试查不出转置 bug——两边看起来一样）。

2. **`matplotlib.use('Agg')` 破坏 `plt.show()`**：模块顶部一次性 `use('Agg')` 会让 `savefig` 工作但 `plt.show()` 静默失效。`visualize_events.py` 故意不在导入时强制 backend，由 `_ensure_gui_backend()` 按需切换。

3. **`from x import y` 后再用 `x.y()` 触发 NameError**：这种"函数级导入 + 模块名调用"的混用是常见笔误，IDE 通常不报警。`HeadlessEventBrowser._render()` 首次实现时就踩了这个坑——静态审计时再发现 `LOOKBACK_K` 未定义、`axes[2,2]` 越界、`_draw_info_panel` 参数顺序错、`build_instance_trajectories` 缺 `instance_tokens` 等 4 处同类隐患（这些都只存在于我新写的 headless 类里，GUI 版对应调用全部正确）。**经验：任何函数级导入，调用点必须用导入的函数名；任何数组索引，必须先看 `shape` 再用。**

4. **`build_web_gallery` 重构时漏写 `hb.out_dir = out_dir`**：注释说"redirect PNGs into the gallery dir"，但实际代码漏了这一行——PNG 全部进 `output/browser/`，`index.html` 在 `output/web/`，用户看不到任何文件。**经验：重构前后必须看一遍函数所有引用点都还连得上，**特别是看似"显而易见"的赋值语句。

5. **`_GALLERY_HTML` 模板错用 `{{` 占位符**：模板为 `.format()` 留了 `{{` 转义，但实际走 `str.replace`（不解析 format）——`{{` 原样进 HTML 把 CSS/JS 全崩。**经验：占位符语法要和替换方式匹配——`str.replace` 用单括号，`.format()` 才用双括号。**

6. **OSZ 原始几何覆盖 70-80% BEV 不是 bug**：密集城区建筑物阴影会投出大块几何阴影，是**正确的几何结果**。`PA_gen_v2/` 实际用的是 `PA-relevant OSZ = 原始 OSZ ∩ 可行驶区域`，已过滤掉建筑物内部等车辆不可能出现的位置。`osz_source_viz.py` 跑出大覆盖百分比时不要怀疑代码。

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
改 `common/bev_config.py` 中的 `BEV_RESOLUTION_M` 即可全局生效。OSZ 磁盘缓存会按 config hash 自动隔离。

**Q：可视化卡死 / output 目录没文件？**

A：先看终端是否在打印 `[ 1/20]` 之类的进度——首次运行每个事件约 5-15 秒（3D 光线投射），不是卡死。第二次运行走磁盘缓存秒出。如果是 GUI 模式 `plt.show()` 阻塞终端是正常的，原生窗口可能藏在任务栏里。推荐改用 `--web` 浏览器图廊模式（最稳）。

**Q：跑 `--web` 想覆盖所有 762 个 positive 事件？**

A：用 `--web_max 0` 限制关闭；首次会跑较久（约 1-2 小时），但 OSZ 缓存写盘后第二次秒出。日常翻看用默认 `--web_max 20`（约 5 分钟）即可。

---

## 许可证

本项目代码按 MIT 许可证发布（如无特殊说明）。
