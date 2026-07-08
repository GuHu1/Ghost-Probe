# Ghost-Probe — 项目长期记忆

## 项目目标
从 nuScenes 挖掘"幽灵车辆涌现事件"（ghost vehicle emergence）：
车辆从 OSZ（遮挡阴影区）中驶出，用于生成 phantom-vehicle 训练样本。

## 目录结构
- `common/bev_config.py` — BEV 网格唯一定义（BEV_EXTENT_M=50, BEV_RESOLUTION_M=0.2）。
  改网格只改这一个文件。
- `OSZ/` — 几何遮挡计算（ray_casting, drivable_filter, bev_viz）。
- `PA_gen_v2/` — 幽灵车辆挖掘主流程（当前活跃版本）。
  - `osz_source.py` — OSZ 单一入口，缓存 per-frame 结果。
  - `ghost_vehicle_miner.py` — 事件挖掘核心。
  - `trajectory.py` — 轨迹插值（KNOWN/INTERPOLATED/NO_EVIDENCE 三态）。
  - `visualize_events.py` — 事件可视化（离线 PNG + 交互式 Event Browser）。

## 关键坐标约定（极易踩坑）
- BEV 数组 (nx,ny)：axis-0=ego-x(前), axis-1=ego-y(左)。索引 mask[i,j]。
  **绝不要换成 mask[j,i]**（方阵会静默转置）。
- matplotlib 画图：x-axis=ego-y(水平), y-axis=ego-x(前=上)，即 plot(ey, ex)。
  与 imshow extent `[y_max,y_min,x_min,x_max]` 配合，origin='lower'。
- ego 恒在自己的坐标系原点 (0,0)，朝 +x（前）。

## OSZ 两种口径（重要）
- raw OSZ：纯几何遮挡，含建筑阴影，城市密集场景覆盖 70-80%+ 网格。
- PA-relevant OSZ = raw OSZ ∩ drivable area（nuScenes HD 地图）。
  **车辆遮挡判定一律用 PA-relevant OSZ**（miner 这么做，可视化也要展示这个，
  否则正确的挖掘判定看起来是错的）。需要 shapely。

## 事件数据结构
{scene_token, emerge_sample(t), instance_token, emerge_bev_xy(帧t ego系),
 lookback_tokens[t-k..t-1], was_in_osz[True|False|None](逐帧判定),
 n_osz_frames, n_evidence_frames, label(1=ghost/0=visible)}
- was_in_osz: True=确认在OSZ, False=确认可见, None=无证据(NO_EVIDENCE)。
- lookback_k=4, MIN_OSZ_OVERLAP=1。

## Event Browser（交互式质检工具）
`python PA_gen_v2/visualize_events.py --dataroot <nusc> --browse`
- 2x3 布局 [t-4 t-3 t-2 / t-1 t info]，每帧用各自 ego 系显示自己的 OSZ。
- 键盘 n/p/r/q。lookback 车辆位置用 trajectory 插值（与 miner 一致）。
- **自动降级**：GUI 后端不可用时（远程服务器/无 tkinter/Qt），
  自动进入 HeadlessEventBrowser（终端输入 + PNG 输出到 `output/_browser_current.png`）。
  额外支持 j/k（±10跳转）和数字索引跳转。
  WebAgg 后端（需 tornado）也可用于浏览器内交互。
- 离线导出走不加 `--browse`（默认）：make_event_grid → PNG。

## 环境
- managed python 3.13.12；无项目 venv（依赖 nuscenes-devkit/shapely 等未装）。
- 依赖见 requirements.txt。
