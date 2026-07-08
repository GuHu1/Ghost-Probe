# PA_gen_v2/ — Ghost Vehicle Mining

## 这是做什么的

`PA_gen_v2/` 挖掘"幽灵车辆出现事件"（ghost vehicle emergence events）：某辆车在过去
几帧被 OSZ（遮挡阴影区）挡住看不见，当前帧突然出现在可见区域——这就是一次
"从盲区冒出来"的事件，可以作为 Phantom Agent 的训练正样本（"这里应该预测出一个
可能存在的幽灵车"）。

这和 `PA_gen_v1/` 做的事情目标相同（都是给 Phantom Agent 挖正/负样本），但方法
完全不同：

- `PA_gen_v1/`：直接读 nuScenes 标注自带的 `visibility_token` 字段（人工标注的
  可见度），不做任何几何推理。简单、快，但完全依赖标注质量。
- `PA_gen_v2/`：不看标注的可见度字段，而是用 LiDAR/相机深度自己算一遍"这个位置
  从自车视角看是否被遮挡"（也就是 OSZ），再拿真实车辆位置去对照这个自己算出来
  的遮挡区。更复杂，但能验证"几何上是否真的被挡住"，也能在标注可见度字段不可靠
  或缺失时作为交叉验证。

`PA_gen_v2/` 的 OSZ 计算不再自己重新实现一遍，而是直接调用 `OSZ/` 模块
（`OSZ/modules/ray_casting.py`）。原因和改法见下面「PA_gen_v2 如何使用 OSZ 模块」。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `osz_source.py` | **桥接层**：把 `OSZ/` 模块（原始几何 OSZ + 可行驶区域过滤）包装成 `PA_gen_v2/` 需要的单帧查询接口。**含 OSZ 磁盘缓存**（第二次运行秒出）。所有 OSZ 相关计算都应经过这里，不要再自己写一遍。 |
| `trajectory.py` | 修复"回溯帧没标注 = 默认判定被遮挡"这个隐藏假设的核心逻辑（纯函数，无 nuScenes 依赖，见下文）。 |
| `ghost_vehicle_miner.py` | 主挖掘逻辑：扫描每个 scene，对每辆车判断"当前可见 + 过去被遮挡（按 PA-relevant OSZ）"是否成立，输出事件列表（JSON）。 |
| `visualize_events.py` | 把挖出来的事件画成图，肉眼检查"红点是否真的落在红色 PA-relevant OSZ 里"。**在信任挖掘结果之前必须先看这个。** 支持 GUI/headless/web 三种浏览模式 + 全场景 BEV 真值 + HD 地图车道线。 |
| `osz_source_viz.py` | 单帧 OSZ 几何检查：原始 OSZ vs 可行驶区域过滤后的 PA-relevant OSZ 对比 + OSZ-vs-GT 车辆框对照图，回答"OSZ 生成得对不对"。 |
| `analyze_distances.py` | 正样本 emergence 距离分布统计 + 数据驱动阈值建议（生成 2x2 图、CSV、终端报告）。 |
| `test_units.py` | 单元测试：坐标变换、OSZ 单遮挡物测试、坐标轴顺序回归测试、可行驶区域过滤的优雅降级测试、`trajectory.py` 的三种状态测试。 |
| `test_synthetic_e2e.py` | 不需要真实 nuScenes 数据的合成场景端到端测试（一堵墙 + 一辆藏在墙后的车）。 |
| `run_pipeline.py` | 一键跑完整流程：单测 → 单帧可视化 → 挖掘 → 事件可视化，任何一步失败就停。 |
| `osz_geometry_legacy.py` | 旧版本 PA_gen_v2/ 自己的 OSZ 实现，**不再被任何活跃代码引用**，只作历史参考保留。 |

---

## 怎么用

```bash
cd PA_gen_v2/

# 第一步：跑单测（不需要真实数据，几秒钟）
python test_units.py
python test_synthetic_e2e.py   # 生成 output/synthetic_validation.png，肉眼看一眼

# 第二步：单帧 OSZ 检查（需要真实 nuScenes 数据）
python osz_source_viz.py --dataroot /path/to/nuscenes --version v1.0-mini --sample_idx 5
# 打开 output/osz_sample_viz.png，是 2x3 六格图：
#   占据网格 | 原始OSZ | 可行驶区域 | PA-relevant OSZ | 叠加图 | GT车辆框对照
# 重点看第 4 格（PA-relevant OSZ）和第 6 格（GT vs PA-relevant OSZ）：
#   原始OSZ在密集城区覆盖 70-80%+ 是正常的（建筑物阴影），不用因为这个数字
#   大就怀疑代码；PA-relevant OSZ 才是挖掘实际用的东西，看这个是否合理。
#   第6格 绿框 = 有车，OSZ 判定可见 —— 正常
#            红框 = 有车，OSZ 判定被遮挡 —— 这就是"幽灵车候选"

# 第三步：一键跑完整流程（单测 + 单帧检查 + 全量挖掘 + 事件可视化）
python run_pipeline.py --dataroot /path/to/nuscenes --version v1.0-mini

# 也可以单独跑挖掘 + 可视化：
python ghost_vehicle_miner.py --dataroot /path/to/nuscenes --version v1.0-mini \
    --out output/ghost_events_mini.json
python visualize_events.py --dataroot /path/to/nuscenes --version v1.0-mini \
    --events output/ghost_events_mini.json
```

`--dataroot` 是**必填参数**，不再有指向不存在路径的假默认值——宁可让程序
立刻报错提示你忘记传参，也不要让它试图打开一个不存在的目录、在 `NuScenes()`
构造函数深处才报一个看不懂的错。所有输出默认写到 `PA_gen_v2/output/`，可以用
`--outdir`（`run_pipeline.py`）或各脚本自己的 `--out` / `--events` / `--out_pos`
等参数改路径。

---

## 可视化三模式（`visualize_events.py`）

`visualize_events.py` 提供三种浏览方式，覆盖本地 GUI 用户、远程服务器用户、想看大量事件的用户。**所有模式画的内容完全一致**——2×3 布局，每帧显示该帧**自己的** PA-relevant OSZ + ego marker + 车辆位置（颜色按 was_in_osz 判定）+ drivable 路面 + 全场景 BEV 真值框（青色车辆 + 黄色行人 + 被追踪车辆高亮框 + 朝向箭头）+ HD 地图车道线。

### 1. GUI 浏览器模式（`--browse`，默认尝试 GUI 后端）

```bash
python visualize_events.py --dataroot <nusc> --browse
```

弹出 matplotlib 窗口，按键操作：

| 按键 | 功能 |
|---|---|
| `n` / `→` | 下一个事件 |
| `p` / `←` | 上一个事件 |
| `r` | 重绘当前 |
| `q` / `Esc` | 退出 |

**适用场景**：本地图形界面用户。后端优先级 TkAgg → QtAgg → WebAgg（最后兜底）。

### 2. Headless 终端模式（`--browse --headless`）

```bash
python visualize_events.py --dataroot <nusc> --browse --headless
```

终端输入命令，每个事件独立存 PNG 到 `output/browser/event_0000.png`、`event_0001.png` ... 用看图软件翻文件夹即可（Windows 照片 / IrfanView / VS Code 均支持 ←/→ 翻页）。

| 按键 | 功能 |
|---|---|
| `n` / `→` | 下一个事件 |
| `p` / `←` | 上一个事件 |
| `j` / `+` | 跳 +10 |
| `k` / `-` | 跳 -10 |
| 数字 | 跳到指定索引 |
| `r` | 重绘 |
| `q` | 退出 |

**适用场景**：远程服务器无 GUI、或不想被 matplotlib 窗口卡顿困扰时。

### 3. Web 图廊模式（`--web`，最推荐）

```bash
python visualize_events.py --dataroot <nusc> --web
# 可选：--web_max N  限制渲染数（默认 20，0=全部）；--label_filter 1/0/-1
```

把每个事件渲成 PNG + 生成 `index.html`，自动用默认浏览器打开。浏览器里键盘操作：

| 按键 | 功能 |
|---|---|
| `←` / `→` 或 `↑` / `↓` | 翻页 |
| `j` / `k` | 跳 ±10 |
| 输入数字 + Go | 跳到指定编号 |
| `q` | 关闭 |

**输出**：
```
PA_gen_v2/output/web/
├── event_0000.png
├── event_0001.png
├── ...
└── index.html   ← 浏览器打开这个
```

**适用场景**：想快速翻看大量事件、不想被 matplotlib 窗口拖累时。**首次运行**每个事件约 5-15 秒（3D 光线投射），**第二次运行走 OSZ 磁盘缓存秒出**。远程服务器：`scp -r output/web <laptop>:/tmp/` 拉回本地打开，或 `python -m http.server --directory output/web 8080` 起服务后浏览器访问 `http://<server>:8080/`。

### 三模式对比

| 模式 | 启动方式 | GUI 依赖 | 单事件渲染 | 推荐场景 |
|---|---|---|---|---|
| **GUI 浏览器** | `--browse` | 需要 tkinter/Qt | 秒级（开窗后） | 本地快速翻看少量 |
| **Headless 终端** | `--browse --headless` | 零 | 5-15s（首次）/ 秒级（缓存） | 远程服务器、看图软件翻 |
| **Web 图廊** | `--web` | 零 | 5-15s（首次）/ 秒级（缓存） | **大量事件、最推荐** |

---

## 可视化增强：每帧画什么

`visualize_events.py` 的每个 BEV 面板（`[t-4][t-3][t-2] / [t-1][t]`）都画以下几层（按 zorder 从低到高）：

1. **Drivable area（路面）**：暗绿色底，从 `drivable_mask` 提取
2. **Bev obstacles（障碍）**：灰色，voxel cast 出的真障碍表面
3. **PA-relevant OSZ（遮挡阴影）**：红色，`raw OSZ ∩ drivable area`
4. **HD 地图车道线**：细蓝线，从 `NuScenesMap.get_records_in_radius` 加载，按 location 缓存
5. **其他车辆**：青色半透明 BEV 框，含正确朝向
6. **行人**：黄色小点（不画框——BEV 尺度下太小）
7. **被追踪的幽灵车辆**：verdict 颜色框（红/蓝/灰/绿）+ 白色朝向箭头，高亮区别于其他车辆
8. **Ego marker**：白色三角 + 朝向箭头（在每帧自己的 ego 系原点）

**典型解读**（以 positive ghost 事件为例）：

- **t-4 / t-3**：被追踪车辆（红框）落在红色 OSZ 内 → 它确实被挡住了
- **t-2**：OSZ 变小、车辆到边缘 → 准备出来了
- **t-1**：OSZ 已经够不到它（蓝框 visible）→ 已经能看到
- **t**：绿框（emerged）+ 朝向箭头 → "从盲区冒出来"的瞬间
- 青色框告诉你**是哪些车/墙投出的 OSZ**——如果 OSZ 形状对不上青色框，挖掘就有问题

`visualize_events.py` 的右侧 info 面板会标注每帧的 OSZ 覆盖率、verdict 颜色含义、键盘说明。

---

## analyze_distances.py 距离分布分析

```bash
python analyze_distances.py
# 可选：--bin 5（箱宽米数，默认 5） --out <png> --label_filter 1/0/-1
```

读 `output/ghost_events_*.json`，输出：

1. `output/distance_analysis.png` — 2×2 图：
   - 左上：emergence 距离直方图（带 mean 线）
   - 右上：每箱 positive 计数
   - 左下：每箱 Unknown（无证据帧）占比
   - 右下：每箱 OSZ overlap 均值 + strong% (n_osz≥2) 比例
2. `output/distance_analysis.csv` — 每箱统计表
3. 终端报告 — 累计留存表 + 数据驱动的阈值建议

**典型结论**（基于 nuScenes v1.0-mini 762 个 positive 样本）：

- 距离范围 3.6m ~ 66.3m，均值 31.5m，中位数 31.1m
- Unknown 比例在 ≤30m 仅 ~5%，30-40m 约 11-13%，**无明显 35m/40m 断崖**
- OSZ overlap 均值 ≤30m 约 2.6-4.0，30-40m 约 2.3-2.4
- **推荐阈值**：35-40m 是较实用的平衡点（保留 62-76% 样本，strong% 仍约 75%）。收紧到 25-30m 会损失 54-69% 样本，收益有限。

阈值分析逻辑：从峰值质量区间（10-15m）向外扫描，找持续 ≥2 个箱退化（unknown≥15% 或 osz_mean<2 或 strong%<60%），避免误把 0-5m 近距离低 OSZ 重叠判为崩塌点。

---

## PA_gen_v2 如何使用 OSZ 模块

以前 `PA_gen_v2/osz_geometry.py` 是完全独立的一份 OSZ 实现：直接把 LiDAR 点云按高度
裁剪后扔进 BEV 网格，再做一次 2D 射线投射。这和 `OSZ/modules/ray_casting.py` 做的
事情概念上一样，但具体做法更简单粗糙——没有相机深度重投影，也没有 `OSZ/`
里那种「先用体素判定真实遮挡物表面、再对这个实心网格做射线投射」的两阶段设计
（`OSZ/modules/ray_casting.py` 的模块注释详细解释了为什么直接按点云密度分箱会在
远处产生间隙、进而导致"影子被当成新墙"的连锁扩散 bug）。

既然 OSZ 模块本来就是设计给别的模块用的，`PA_gen_v2/` 现在改成直接调用它：

```
PA_gen_v2/ghost_vehicle_miner.py
        │
        ▼  get_pa_relevant_osz_for_sample(nusc, sample_token)
PA_gen_v2/osz_source.py  ← 唯一的桥接层（含 OSZ 磁盘缓存）
        │
        ├─ OSZ.utils.nuscenes_loader.NuScenesOSZLoader.build_frame_for_token(...)
        │     单帧构建相机深度图（复用 OSZ/ 已有逻辑，不重新实现）
        │
        ├─ OSZ.modules.ray_casting.build_bev_occ_from_voxel_cast(...)
        │  OSZ.modules.ray_casting.cast_osz_2d(...)
        │     真正的 3D 体素投射 + 2D 射线投射（原始几何 OSZ），
        │     和 OSZ/run_osz_pipeline.py 用的是同一套代码
        │
        └─ OSZ.modules.drivable_filter.build_drivable_mask(...)
           OSZ.modules.drivable_filter.filter_osz_by_drivable(...)
              原始几何 OSZ ∩ 可行驶区域 = PA-relevant OSZ（见下一节）
```

具体改动：

1. **`OSZ/utils/nuscenes_loader.py`** 新增了 `build_frame(sample)` /
   `build_frame_for_token(sample_token)`，把原来只能在"遍历整个数据集"场景下
   使用的单帧相机深度图构建逻辑，拆成一个可以按 `sample_token` 单独调用的方法。
   `NuScenesOSZLoader.__iter__` 内部现在也是调用这个方法，行为完全不变，只是
   变成可复用的了。
2. **`PA_gen_v2/osz_source.py`**（新文件）包一层薄适配：拿到调用方已经建好的
   `nusc` 对象，构造一个轻量的 `NuScenesOSZLoader` 壳子（不重新打开一次
   nuScenes 数据库），调用 `build_frame_for_token` 拿到相机深度图，再交给
   `OSZ/modules/ray_casting.py` 的 `build_bev_occ_from_voxel_cast` +
   `cast_osz_2d` 算出 `(bev_occ, osz_raw)`。
3. 因为 `mine_ghost_events` 对同一帧、重叠的回溯窗口会重复查询 OSZ（第 t 帧
   要看 t-1..t-4，第 t+1 帧要看 t-3..t，中间有重叠），`osz_source.py` 内置了
   一个按 scene 清空的**内存缓存**，避免同一帧的相机深度重投影 + 体素投射被
   重复算好几遍——这一步比旧实现（纯 LiDAR 分箱）贵不少，缓存在这里是必须的，
   不是锦上添花。
4. **跨 session 的磁盘缓存**：`get_pa_relevant_osz_for_sample()` 还会在首次
   计算后把 `(bev_occ, osz_raw, osz_pa, drivable_mask)` 写盘到
   `output/osz_cache/{config_hash}/{sample_token}.npz`。第二次运行（无论是重挖
   还是用 `visualize_events.py` 翻事件）直接 load npz，**首次 OSZ 计算从 25s
   降到 <0.1s**。`config_hash` 含 BEV_RANGE/RESOLUTION/Z 参数，改配置自动失效。
5. **坐标轴顺序是这次改动里最容易踩的坑**，专门写了警告：`OSZ/` 的数组是
   `(nx, ny)`，`array[i,j]` 里 `i` 对应 ego-x（前）、`j` 对应 ego-y（左）；
   旧的 `PA_gen_v2/osz_geometry.py` 用的是反过来的图像式 `(row, col)` =
   `(ego-y, ego-x)`。因为网格是正方形，这类轴顺序搞反的 bug **不会报错，只会
   把 OSZ 结果沿对角线镜像/转置**，跑起来看着"好像没问题"，实际上完全错了。
   `osz_source.py` 因此不提供 `col/row` 这种容易顺手写反的接口，只提供命名
   明确的 `bev_xy_to_ij()` / `ij_to_bev_xy()`，并在 `test_units.py` 里专门用
   一个**左右不对称**的遮挡物做了回归测试（对称的墙测试查不出转置 bug——两边
   看起来一样）。

`OSZ/modules/ray_casting.py` 顺带修了一个真实的可移植性 bug：它以前在文件顶部
无条件 `import torch`，但这个文件里完全没用到 torch（是纯 numpy 几何代码）。这会
导致任何没装 torch 的纯 CPU 环境（比如 `PA_gen_v2/` 这种只需要 numpy 几何计算的场景）
在 `from OSZ.modules.ray_casting import ...` 这一步就直接失败。已去掉这个多余的
import；需要 torch 的 CNN 精修部分完全在 `OSZ/modules/crf_refine.py` 里，那边本来
就有独立的 import 保护。

`OSZ/modules/ray_casting.py` 的 `cast_osz_2d` 2D 阴影射线投射还做了**向量化优化**：
原实现用纯 Python 双 for 循环（n_angles=12566 × max_steps=2000 = 2500 万次迭代，
单 sample 约 25 秒），改为 numpy 向量化后 500×500 网格降到 0.5 秒（**提速 50 倍**），
20 次随机小网格测试 bit-for-bit 一致通过。

---

## 原始几何 OSZ vs. PA-relevant OSZ（可行驶区域过滤）

**这是实测数据跑出来后新加的一步，不要跳过。** 原始几何 OSZ（`cast_osz_2d` 直接
算出来的那个）会把"建筑物挡住之后的整片阴影"也算成遮挡区——这在几何上没错（那片
区域确实被挡住看不见），但对幽灵车辆挖掘毫无意义：车不可能出现在建筑物内部，
building 阴影再大也不该被当成"车被挡住了"的证据。这正是 `OSZ/modules/
drivable_filter.py` 存在的原因（同样的道理，`OSZ/run_osz_pipeline.py` 的
Stage 4c 一直都在做这一步），也是这个项目 README 里提到的"被建筑物包围 = 70%
OSZ"问题——在密集城区场景下，原始几何 OSZ 覆盖 70-80%+ 是**符合预期的正常现象**，
不是 bug，`osz_source_viz.py` 跑出这个数字时不用怀疑代码。

`PA_gen_v2/` 一开始只接了原始几何 OSZ 这一步（`get_osz_for_sample`），没有接
可行驶区域过滤——这是这次重构漏掉的一环，不是故意简化，已经补上：

- **`osz_source.py` 新增 `get_drivable_mask_for_sample(nusc, sample_token)`**：
  返回 `(nx, ny)` bool 数组，True = 车辆物理上可能出现的位置（drivable_area /
  carpark_area，按道路边缘再膨胀 ~1.5m）。直接调用
  `OSZ/modules/drivable_filter.py` 的 `build_drivable_mask`。
- **`osz_source.py` 新增 `get_pa_relevant_osz_for_sample(nusc, sample_token)`**：
  一次性返回 `(bev_occ, osz_raw, osz_pa, drivable_mask)`，其中
  `osz_pa = osz_raw ∩ drivable_mask`。**`ghost_vehicle_miner.py` 现在所有的
  遮挡判断都用 `osz_pa`，不再用原始 `osz_raw`**——正样本、负样本、"是否仍在
  遮挡区"的判定全部基于 PA-relevant OSZ。
- **依赖处理**：`OSZ/modules/drivable_filter.py` 文件顶部无条件 `import
  shapely`，这是可行驶区域过滤真实需要的依赖，但不该因为缺它就让整个
  `osz_source.py`（包括完全不需要 shapely 的原始几何 OSZ）都导入失败。所以
  `osz_source.py` 用 `try/except` 包了这个 import：如果 `shapely` 没装，
  `get_pa_relevant_osz_for_sample` 会退化成"不过滤"（`drivable_mask` 全
  `True`，等价于原始 OSZ），并只打印**一次**警告（不会在挖掘几千帧的时候刷屏），
  不会让整条流水线跑不起来。`osz_source.drivable_filter_available()` 可以
  查询当前这个过滤到底是真的生效了还是被静默跳过了——`ghost_vehicle_miner.py`
  启动时会打印这个状态，`osz_source_viz.py` 也会在过滤实际未生效时明确提示。
- **`osz_source_viz.py`** 现在是 6 格图：占据网格 | 原始 OSZ | 可行驶区域 |
  PA-relevant OSZ | 叠加图 | GT 车辆框对照（对照的是 PA-relevant OSZ，不是
  原始 OSZ）。跑一遍就能直接看到过滤前后的差异，而不是只看一个百分比数字。
- **`visualize_events.py`** 背景显示也换成了 PA-relevant OSZ——因为这才是
  `ghost_vehicle_miner.py` 真正用来做判断的东西，用原始 OSZ 画图会让本来正确的
  挖掘结果看起来"红点怎么到处都是"，产生误导。

---

## "回溯帧没有标注就默认判定被遮挡"这个隐藏假设是怎么修的

### 问题

旧代码（`ghost_vehicle_miner.py`）里有这么一行：

```python
if lb_xyz_global is None:
    # Vehicle was not annotated in this lookback frame.
    # Treat as "effectively in OSZ" (unseen = possibly hidden)
    was_in_osz_per_frame.append(True)
```

"这一帧没有该车的标注"其实可能对应三种完全不同的情况，旧代码把它们混为一谈：

1. **真的被遮挡到几乎不可见**（nuScenes 标注习惯是遮挡到 40% 可见度以下仍然会
   标注，也就是说真正"完全不可见"才会没有标注框）——这种情况判定为"被遮挡"是对的。
2. **这辆车的轨迹在这一帧根本还没开始 / 已经结束**——车可能压根还没出现在场景里，
   跟"被遮挡"毫无关系。
3. **这一帧这辆车开到了 BEV 网格范围之外**——只是超出了感知范围，不是被遮挡。

把情况 2、3 也当成"被遮挡"，会让"幽灵车挖掘"混入大量假阳性：车只是暂时开出视野
或者轨迹还没开始，就被算成"从阴影里冒出来的幽灵车"。

### 修法

`PA_gen_v2/trajectory.py`：对每辆车，不只看挖掘时用到的 k 帧回溯窗口，而是把这辆车
**完整的标注链**（`first_annotation_token` 沿 `next` 一路走到底，和
`preprocess/create_pa_labels_mini.py` 建轨迹用的是同一套机制）都建出来。对于回溯
窗口里某一帧缺标注的情况，去查这辆车的完整轨迹：

- **如果查询时刻正好被轨迹的前后两个已知点夹住**（前后都有这辆车的标注，只是
  中间这一帧没有）→ 说明轨迹是连续的，中间没标注基本可以确定就是完全不可见 →
  这才是真正的遮挡证据。同时会用前后两点线性插值出这一帧大概在哪，再检查这个
  插值位置是不是还在 BEV 网格范围内（哪怕轨迹连续，车也可能在这一帧正好开出了
  网格——比如一个大转弯），不在范围内的话仍然不算证据。
- **如果查询时刻在轨迹开始之前、或结束之后**（没有前后夹住）→ 明确返回"没有
  证据"（`NO_EVIDENCE`），既不算被遮挡也不算可见，交给调用方决定怎么处理，
  而不是替调用方悄悄猜一个结果。

`ghost_vehicle_miner.py` 里，每个回溯帧的判定结果现在是 `True`（确认被遮挡）/
`False`（确认可见）/ `None`（没有证据）三态，而不是原来的布尔值：

- 正样本条件不变：确认被遮挡的帧数 ≥ `min_osz_overlap`。
- 负样本条件变严格了：以前是"确认被遮挡的帧数 = 0"就算负样本，现在要求
  **k 帧全部都有确凿证据、且全部证据都是"可见"**才算负样本——如果某帧是
  `None`（没证据），不能因为"反正没被判定为遮挡"就顺手当成负样本，那同样是在
  猜。这种"有一部分帧没证据、又没有任何一帧确认被遮挡"的模糊情况会被直接丢弃
  （`ghost_vehicle_miner.py` 运行时会打印 `Dropped (ambiguous, no evidence)`
  的计数，方便你知道这种情况出现的频率）。

`visualize_events.py` 也相应更新：回溯帧现在会画成三种颜色——红色实心圆=确认
被遮挡，蓝色实心圆=确认可见，灰色叉号=没有证据——而不是把没有证据的情况偷偷
画成某种看似确定的颜色。

`test_units.py` 和 `test_synthetic_e2e.py` 里都有针对这个修复的单测，不需要真实
nuScenes 数据就能跑（`trajectory.py` 是纯函数，不直接访问 nuScenes API）。

---

## 已知踩坑经验（已修复，避免重蹈覆辙）

`visualize_events.py` 经历多轮 GUI/headless/web 模式适配时，曾出现过一组同类静态笔误——IDE 不报警，单元测试也覆盖不到，**只在首次运行时炸**。这些坑全部记录在根 README 的"已知踩坑经验"小节中。**特别关注本模块的两点**：

1. **`HeadlessEventBrowser._render()` 5 处笔误**（同一函数里集中出现）：
   - `trajectory.build_instance_trajectories(...)` → `build_instance_trajectories(...)`（顶部是 `from trajectory import` 函数级导入，模块名 `trajectory` 未导入）
   - `LOOKBACK_K` 未定义 → 从 `ghost_vehicle_miner` 导入 `LOOKBACK_FRAMES` 并替换
   - `axes[2, 2]` 越界（subplots(2,3) 只有 0/1 两行）→ `axes[1, 2]`
   - `_draw_info_panel(...)` 参数顺序错位（frames/coverages/label_filter）→ 对齐签名
   - `build_instance_trajectories(self.nusc)` 缺必需参数 → 加 `{instance_tok}`

   GUI 版 `EventBrowser` 的对应调用本来就全部正确，全部是 headless 版新写时的笔误。

2. **`build_web_gallery` 漏写 `hb.out_dir = out_dir`**：注释说"redirect PNGs"，但实际代码漏了这一行——PNG 全进 `output/browser/`，`index.html` 在 `output/web/`，用户看不到任何文件。**经验：重构前后必须看一遍函数所有引用点都还连得上，特别是看似"显而易见"的赋值语句。**

3. **`_GALLERY_HTML` 模板错用 `{{` 占位符**：模板为 `.format()` 留了 `{{` 转义，但实际走 `str.replace`——`{{` 原样进 HTML 把 CSS/JS 全崩。**经验：占位符语法要和替换方式匹配。**
