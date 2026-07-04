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

**现在 `PA_gen_v2/` 的 OSZ 计算不再自己重新实现一遍，而是直接调用 `OSZ/` 模块**
（`OSZ/modules/ray_casting.py`）。原因和改法见下面「PA_gen_v2 如何使用 OSZ 模块」。

## 文件说明

| 文件 | 作用 |
|---|---|
| `osz_source.py` | **桥接层**：把 `OSZ/` 模块（原始几何 OSZ + 可行驶区域过滤）包装成 `PA_gen_v2/` 需要的单帧查询接口。所有 OSZ 相关计算都应该经过这里，不要再自己写一遍。 |
| `trajectory.py` | 修复"回溯帧没标注 = 默认判定被遮挡"这个隐藏假设的核心逻辑（纯函数，无 nuScenes 依赖，见下文）。 |
| `ghost_vehicle_miner.py` | 主挖掘逻辑：扫描每个 scene，对每辆车判断"当前可见 + 过去被遮挡（按 PA-relevant OSZ）"是否成立，输出事件列表（JSON）。 |
| `visualize_events.py` | 把挖出来的事件画成图，肉眼检查"红点是否真的落在红色 PA-relevant OSZ 里"。**在信任挖掘结果之前必须先看这个**。 |
| `osz_source_viz.py` | 单帧 OSZ 几何检查：原始 OSZ vs 可行驶区域过滤后的 PA-relevant OSZ 对比 + OSZ-vs-GT 车辆框对照图，回答"OSZ 生成得对不对"。 |
| `test_units.py` | 单元测试：坐标变换、OSZ 单遮挡物测试、坐标轴顺序回归测试、可行驶区域过滤的优雅降级测试、`trajectory.py` 的三种状态测试。 |
| `test_synthetic_e2e.py` | 不需要真实 nuScenes 数据的合成场景端到端测试（一堵墙 + 一辆藏在墙后的车）。 |
| `run_pipeline.py` | 一键跑完整流程：单测 → 单帧可视化 → 挖掘 → 事件可视化，任何一步失败就停。 |
| `osz_geometry_legacy.py` | 旧版本 PA_gen_v2/ 自己的 OSZ 实现，**不再被任何活跃代码引用**，只作历史参考保留。 |

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

`--dataroot` 现在是**必填参数**，不再有一个指向不存在路径的假默认值——宁可让程序
立刻报错提示你忘记传参，也不要让它试图打开一个不存在的目录、在 `NuScenes()`
构造函数深处才报一个看不懂的错。所有输出默认写到 `PA_gen_v2/output/`，可以用
`--outdir`（`run_pipeline.py`）或各脚本自己的 `--out` / `--events` / `--out_pos`
等参数改路径。

## PA_gen_v2 如何使用 OSZ 模块

以前 `PA_gen_v2/osz_geometry.py` 是完全独立的一份 OSZ 实现：直接把 LiDAR 点云按高度
裁剪后扔进 BEV 网格，再做一次 2D 射线投射。这和 `OSZ/modules/ray_casting.py` 做的
事情概念上一样，但具体做法更简单粗糙——没有相机深度重投影，也没有 `OSZ/`
里那种「先用体素判定真实遮挡物表面、再对这个实心网格做射线投射」的两阶段设计
（`OSZ/modules/ray_casting.py` 的模块注释详细解释了为什么直接按点云密度分箱会在
远处产生间隙、进而导致"影子被当成新墙"的连锁扩散 bug）。

既然你说 OSZ 模块本来就是设计给别的模块用的，`PA_gen_v2/` 现在改成直接调用它：

```
PA_gen_v2/ghost_vehicle_miner.py
        │
        ▼  get_pa_relevant_osz_for_sample(nusc, sample_token)
PA_gen_v2/osz_source.py  ← 唯一的桥接层
        │
        ├─ OSZ.utils.nuscenes_loader.NuScenesOSZLoader.build_frame_for_token(...)
        │     单帧构建相机深度图（复用 OSZ/ 已有逻辑，不重新实现）
        │
        ├─ OSZ.modules.ray_casting.build_bev_occ_from_voxel_cast(...)
        │  OSZ.modules.ray_casting.cast_osz_2d(...)
        │     真正的 3D 体素投射 + 2D 射线投射（原始几何 OSZ），和
        │     OSZ/run_osz_pipeline.py 用的是同一套代码
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
   一个按 scene 清空的缓存，避免同一帧的相机深度重投影 + 体素投射被重复算好几遍
   ——这一步比旧实现（纯 LiDAR 分箱）贵不少，缓存在这里是必须的，不是锦上添花。
4. **坐标轴顺序是这次改动里最容易踩的坑**，专门写了警告：`OSZ/` 的数组是
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
