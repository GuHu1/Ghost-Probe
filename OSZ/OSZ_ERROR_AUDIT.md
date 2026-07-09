# OSZ 质量误差审计与修复方案

> 审计日期：2026-07-09  
> 审计范围：`OSZ/modules/ray_casting.py` + `OSZ/utils/nuscenes_loader.py`  
> 输入数据：nuScenes VLP-32 LiDAR（~30k points/sweep, 360°）

---

## 管线总览

```
LiDAR 点云
  │
  ▼
① project_lidar_to_camera  ──→  稀疏深度图 (900×1600, ~0.35% 覆盖率)
  │
  ▼
② densify_depth_map        ──→  稠密深度图 (最近邻插值 + 深度不连续保护)
  │
  ▼
③ RayCaster3D.cast         ──→  V_occ (500×500×10, 每相机独立)
  │                               体素表面判定: |z_voxel - d_obs| ≤ 0.3m
  ▼
④ voxel_to_bev_maxpool     ──→  M_occ^c (500×500 bool, per-camera)
  │      + 6 相机取 OR
  ▼
⑤ cast_osz_2d              ──→  osz_mask (500×500 bool, 向量化 360° 射线投射)
  │
  ▼
⑥ filter_osz_by_drivable   ──→  osz_pa (500×500 bool, ∩ drivable area)
```

---

## 误差源详析

### 误差 ①：LiDAR 点密度极低（底层输入瓶颈）

**现状**：
- nuScenes VLP-32 LiDAR：~30,000 个点 / 帧（360° 全覆盖）
- 6 台相机分摊：每台约 5,000 个有效点
- 图像分辨率：1600 × 900 = 1,440,000 像素
- **实际覆盖率：~0.35%**

**后果**：后面所有步骤（densify → voxel cast → ray cast）99.65% 的数据来自**插值**，不是真实测量。插值误差逐级放大。

**修复方向**：
- 使用多帧 LiDAR 聚合（nuScenes 支持 sweep 聚合，当前 N_SWEEPS 仅用于文件复制，未用于深度图）
- 多帧点云拼接后投影 → 单帧深度图密度提升 5-10x
- 实现位置：`OSZ/utils/nuscenes_loader.py:build_frame_for_token()` 的 LiDAR 加载段

---

### 误差 ②：地面点混入深度图（导致虚假障碍物）

**现状**：
`project_lidar_to_camera` 无差别投射**所有** LiDAR 点到相机深度图，包括地面点。

```
地面 LiDAR 点：  x=15m, z≈0      → 投影深度 ≈15m
体素中心：      x=15m, z=0.4m  → 投影深度 ≈15m
|15-15| = 0 ≤ 0.3 → 标记为"障碍物" ✓  ← 错误！
```

**后果**：道路表面被当成障碍物 → 虚假 BEV occ 覆盖路面 → 射线在这些位置"击中障碍物" → 路面后方被错误标记为 OSZ。

**修复**：
- 在 `build_frame_for_token` 中，LiDAR 点从 sensor frame 转到 ego frame 后，过滤 `z_ego < z_ground_thresh`（建议 0.2m）的点
- 或用 RANSAC 提取地面平面后剔除地面点
- 涉及文件：`OSZ/utils/nuscenes_loader.py` 第 261 行附近

---

### 误差 ③：densify 边缘保护过激（核心问题）

**现状**：
```python
depth_discontinuity_thresh = 1.5   # 米
k_neighbors = 4
# 前景障碍物深度=10m, 背景深度=30m
# K=4 近邻中有前景也有背景点 → spread=20m > 1.5m → 判定为"深度不连续"
# → 边缘像素不填补, depth=0
```

**后果**：每个障碍物边界周围有一圈**深度空洞**。体素投射到此像素时 `d_obs=0` → 不标记 → BEV occ 有边界空洞 → 射线从空洞穿过 → OSZ 不连续。

这是造成"OSZ 不匹配障碍物轮廓"的**最主要原因**。

**修复**：
```python
# 方案 A：放宽阈值（改动最小）
depth_discontinuity_thresh = 4.0   # 从 1.5 → 4.0

# 方案 B：减少近邻数（降低误判）
k_neighbors = 2                    # 从 4 → 2
# K=2 时 spread = |depth_1 - depth_2|，障碍物边缘两个近邻大概率都是前景或都是背景
```
涉及文件：`OSZ/utils/nuscenes_loader.py` `densify_depth_map()` 调用处

---

### 误差 ④：surface_tolerance 对插值深度太严格（核心问题）

**现状**：
```python
surface_tolerance = max(bev_res, z_res) * 1.5 = 0.3m
on_surface = |z_voxel - d_obs| ≤ 0.3
```

densify 的最近邻插值精度有限（最远 8px ≈ 数米误差），插值深度与真实深度的偏差经常 > 0.3m。

**后果**：大量体素因 `|z_voxel - d_interpolated| > 0.3` 而不被标记 → BEV occ 呈"碎片化的薄壳" → 射线从碎片间隙穿过。

**此外**：当前 `on_surface` 只标记**体素深度 ≈ 观测深度**的体素（即障碍物表面的精确位置），不标记障碍物**内部**或**后方**。这是为了防止"阴影变墙壁"的 bug，但代价是 BEV occ 永远只是一层壳。

**修复**：
```python
# 方案 A：放宽 tolerance（最简单）
surface_tolerance = 0.6   # 或 max(bev_res, z_res) * 3.0

# 方案 B：标记障碍物体积而非仅表面（改判定逻辑）
# 将 on_surface: |z_voxel - d_obs| ≤ tol
# 改为 on_or_behind: z_voxel ≥ d_obs - tol AND z_voxel ≤ d_obs + margin
# 这样障碍物"后方"的体素也会被标记（但严格限制在 margin 内，防止变阴影）
on_or_behind = (z_valid >= d_obs - surface_tolerance) & (z_valid <= d_obs + 1.5)
```
涉及文件：`OSZ/modules/ray_casting.py` `RayCaster3D.cast()` 第 307 行

---

### 误差 ⑤：高度门 z ∈ [0.3, 2.2]m 太窄（漏检大型/低矮障碍物）

| 对象类型 | 典型高度 | 当前覆盖？ | 影响 |
|---------|---------|-----------|------|
| 轿车 | 1.4-1.6m | ✓ | — |
| SUV | 1.7-2.0m | ✓ (上限临界) | 车顶可能漏 |
| 公交车/卡车 | 2.5-4.0m | ✗ | **完全漏掉 → 无 OSZ** |
| 路沿/低护栏 | 0.1-0.3m | ✗ (下限临界) | 射线从下方穿过 |
| 行人 | 1.5-1.8m | ✓ | — |

**后果**：大型车辆（公交、卡车）不产生 OSZ，而它们恰恰是城市路口最大的遮挡源。

**修复**：
```python
z_min = 0.1   # 降低
z_max = 4.5   # 抬升
z_res = 0.3   # 略粗，保持 nz 可控
# nz = (4.5 - 0.1) / 0.3 = 14（原 10 层，增量可接受）
```
涉及文件：`OSZ/modules/ray_casting.py` `RayCaster3D.__init__()` 参数，及所有调用 `RayCaster3D()` 的地方（含 `osz_source.py`、`run_osz_pipeline.py`）

---

### 误差 ⑥：max_radius=8 使大物体中空

**现状**：`densify_depth_map(max_radius=8)` → 距离最近 LiDAR 点 > 8px 的像素不填补。

一辆近距离的大卡车可占 200-400px 宽，LiDAR 只打中了车身几个离散点。其余像素距离最近有效点 > 8px → 不填补 → 深度=0 → 体素投射为空洞。

**后果**：大型障碍物的 BEV 投影呈"中空结构"，射线穿过中心空洞 → 障碍物后方部分区域**不被遮挡** → OSZ 缺失。

**修复**：
```python
max_radius = 16   # 从 8 增大到 16
```
涉及文件：`OSZ/utils/nuscenes_loader.py` 中 `densify_depth_map()` 调用处

---

### 误差 ⑦：投影坐标 int32 截断（亚像素误差）

**现状**：
```python
u = (uvw[:, 0] / z_cam).astype(np.int32)
v = (uvw[:, 1] / z_cam).astype(np.int32)
```

`astype(np.int32)` 是**截断**而非四舍五入：`100.9 → 100`。

**后果**：体素中心投影到 (100.9, 200.1)，实际读取深度图中 (100, 200) 的深度值。该位置的深度可能来自 0.5 像素之外的 LiDAR 点。0.5px 偏移在 BEV 上对应约 0.1m 位置偏差。

**修复**：
```python
u = np.rint(uvw[:, 0] / z_cam).astype(np.int32)   # 四舍五入
v = np.rint(uvw[:, 1] / z_cam).astype(np.int32)
```
涉及文件：`OSZ/modules/ray_casting.py` 第 289-290 行

---

### 误差 ⑧：相机拼缝（FOV 边界空洞）

**现状**：6 台相机各自独立计算 `V_occ`，最后 OR。相机之间的 FOV 边界处没有深度覆盖 → 体素从该区域投影时 `in_image=False` → 不标记。

**后果**：BEV occ 在相机 FOV 缝隙处有**从自车辐射出的扇形空洞**。射线从这些扇形缝隙穿过 → OSZ 在这些方位不连续。

**修复**：
- nuscenes 6 相机 FOV 有重叠区域，当前 OR 逻辑已经利用了重叠。但如果仍有缝隙，可在 `cast()` 中增加 `in_image` 边界扩展（margin=5px）
- 或对 BEV occ 做 morphological closing（已在 `visualize_events.py` 的 `_solidify_obstacles` 中实现，但对 OSZ 计算无效，仅影响渲染）

---

### 误差 ⑨：depth_map 中近点覆盖远点的信息丢失

**现状**：
```python
order = np.argsort(-z)   # 按深度降序排列
depth_map[v[order], u[order]] = z[order]   # 远处的先写，近处的覆盖
```

同一像素上有多个 LiDAR 点时，只保留最近的那个。

**后果**：一个像素只能记录一个深度值。如果障碍物前方有其他物体（如路灯杆在前、建筑在后），后方的障碍物深度信息丢失 → 后方障碍物不产生 OSZ。这在密集城市场景中常见。

**修复**：
- 无简单修复（这是深度图表示的根本限制）
- 可改用多帧聚合或多传感器融合缓解

---

### 误差 ⑩：无帧间时序一致性

**现状**：每帧 OSZ 独立计算，帧间无任何关联。

**后果**：
- OSZ 形状帧间跳动（尤其是滑过停放的车辆时，shadows 跳变）
- 短暂进入相机盲区再出来的障碍物，重新出现时 OSZ 可能完全不同
- 可视化上表现为"闪烁"

**修复**：
```python
# EMA 平滑（最简单）
osz_smoothed = 0.8 * osz_current + 0.2 * osz_previous
osz_smoothed = osz_smoothed > 0.5  # 二值化

# 或论文方法：维护"曾可见区域"mask，OSZ 只在新出现的不可见区域中成立
# 见 Möller et al., "From Shadows to Safety", Algorithm 2
```
涉及文件：`OSZ/modules/ray_casting.py` 或 `PA_gen_v2/osz_source.py`

---

## 修复优先级排序

| 优先级 | 误差源 | 改动量 | 预期效果 |
|--------|-------|--------|---------|
| **P0** | ③ densify 边缘保护 + ④ surface_tolerance | 2 行 | BEV occ 从碎片壳变实体，OSZ 连续性大幅改善 |
| **P0** | ② 地面点过滤 | ~5 行 | 消除路面上的虚假 OSZ |
| **P1** | ⑤ 高度门放宽 | ~3 行 | 覆盖卡车/公交，城市路口场景大幅改善 |
| **P1** | ⑥ max_radius 增大 | 1 行 | 大型障碍物 BEV occ 更实心 |
| **P1** | ⑦ int32 → rint | 2 行 | 减少亚像素对齐误差 |
| **P2** | ① 多帧 LiDAR 聚合 | ~20 行 | 深度图密度提升 5-10x → 所有下游步骤质量提升 |
| **P2** | ⑩ 帧间时序一致性 | ~15 行 | 消除 OSZ 抖动 |
| **P3** | ⑧ 相机拼缝 | ~3 行 | 填补 FOV 边界空洞 |
| **P3** | ⑨ 近点覆盖远点 | — | 深度图表示的根本限制，短期不修复 |

---

## 修复后预期效果

**当前**：
```
BEV occ: 碎片化薄壳，有边界空洞，有地面假点
OSZ:     不连续条纹，与障碍物轮廓不匹配，路面有虚假阴影
```

**修复 P0+P1 后**：
```
BEV occ: 较完整实体，边界连续，地面假点消除
OSZ:     连续阴影区，与障碍物轮廓大致匹配，路面无虚假阴影
BEV occ 仍然不会完美实心（LiDAR 稀疏+单帧的固有限制），但足以支撑合格的几何 OSZ
```

---

## 相关文件索引

| 文件 | 相关误差源 |
|------|----------|
| `OSZ/utils/nuscenes_loader.py` | ① ② ③ ⑥ |
| `OSZ/modules/ray_casting.py` | ④ ⑤ ⑦ |
| `OSZ/modules/drivable_filter.py` | — (OSZ 过滤层，不受上述误差影响) |
| `PA_gen_v2/osz_source.py` | ⑩ (OSZ 缓存和接口层) |
| `PA_gen_v2/visualize_events.py` | — (渲染层，已有 `_smooth_osz_mask` 和 `_solidify_obstacles`) |

---

---

## 优化方案：多帧 LiDAR 聚合 + 直接体素化

### 思路

当前管线的最大瓶颈是"稀疏 LiDAR → 深度图 → densify → 体素投射"这条长链路，每一环都在累积误差。如果能**砍掉深度图这一步**，直接在 3D 空间做障碍物体素化，同时用多帧历史点云补密度，质量会有质的提升。

### 约束

只使用**历史帧**（不能偷看未来）。nuScenes 的 LiDAR sweep 是单向链表：`sample_data['prev']` → 上一个 sweep → `prev` → 再上一个。当前帧 t=0，历史 sweep 来自 t=-1, t=-2, ...。LiDAR 频率 20Hz，每个 sweep 间隔约 0.05s。

### 管线对比

```
【当前管线】深度图路径（长链路，多误差源）
  LiDAR (当前帧) → 投影到相机深度图(稀疏) → densify(插值) → voxel cast(表面匹配)
                → maxpool → BEV occ → ray cast → OSZ
  问题: ①稀疏 ②地面 ③边缘保护 ④tolerance ⑥中空 ⑦截断 ⑧拼缝

【优化管线】多帧聚合 + 直接体素化（短链路）
  多帧 LiDAR (t=0..t-N) → 自车运动补偿 → 地面点过滤 → 3D 体素化(点数>阈值=occupied)
                         → maxpool → BEV occ → ray cast → OSZ
  消除的误差源: ②③④⑥⑦ 全部绕过
  改进的误差源: ①⑧
```

### 自车运动补偿

每个历史 sweep 的 LiDAR 点被采集时 ego 处于不同的全局位置。要将历史点并入当前 ego 坐标系：

```
历史 LiDAR 点 → T_lidar2ego(历史) → 历史 ego 坐标系
              → T_ego2global(历史 ego_pose) → 全局坐标系
              → inv(T_ego2global(当前 ego_pose)) → 当前 ego 坐标系
              → 地面点过滤(z < 0.2m 丢弃)
              → 拼入总点云
```

### 动态物体的影响

历史帧里的**移动车辆**在 N 个 sweep 前的位置与当前不同。自车运动补偿后这些点的位置会有偏差：

| 聚合 sweep 数 | 回看时长 | 点密度提升 | 动态物体位移(30km/h) | 建议 |
|---------------|---------|-----------|---------------------|------|
| N=0 (仅当前) | 0s | 1x | 0m | 底线 |
| N=1 | 0.05s | ~1.7x | ~0.4m | 安全 |
| N=3 | 0.15s | ~3x | ~1.25m | **推荐** |
| N=5 | 0.25s | ~4x | ~2.1m | 有风险 |
| N=9 | 0.45s | ~6x | ~3.8m | 不推荐 |

**推荐 N=3**：
- 静态障碍物（建筑、停放车辆、护栏）的全局位置不变 → 聚合后精确对齐，3x 密度大幅改善；
- 移动车辆的历史点偏差 ~1.25m，产生的虚假障碍物是**保守的**（额外的 OSZ → 系统更谨慎），下一帧就会被当前深度覆盖修正；
- 3 个 sweep 的自车位置变化（30km/h × 0.15s = 1.25m）运动补偿误差可控。

### 体素化方案

```
输入: N+1 帧聚合后的点云 (所有点在当前 ego 坐标系)
参数: voxel_size = BEV_RESOLUTION_M = 0.2m (xy), z_res = 0.2m (z)
      z_min, z_max 同当前
      occupancy_threshold: 体素内最少点数（建议 1-2）

流程:
  1. 对聚合点云做 3D 栅格化: voxel_idx = floor((pts - origin) / voxel_size)
  2. 统计每个体素的点数
  3. 点数 >= threshold → occupied
  4. maxpool z → BEV occ
  5. 对 BEV occ 做 morphological closing (填孔) → 实体 BEV 障碍物
  6. cast_osz_2d 照常
```

### 对比当前管线的误差消除

| 误差源 | 当前管线 | 多帧+体素化后 |
|--------|---------|--------------|
| ① LiDAR 稀疏 | 0.35% 像素有深度 → 99.65% 插值 | 3x 点密度 + 体素化不需要深度图 → 绕过 |
| ② 地面假点 | 无过滤，全部投深度图 | 聚合前过滤 z_ego < 0.2m → **消除** |
| ③ densify 边缘保护 | 边界留空 → BEV occ 有孔 | **不再使用 densify → 消除** |
| ④ surface_tolerance | 插值深度 vs 真实深度 > 0.3m → 漏标 | **不再做逐体素深度匹配 → 消除** |
| ⑤ 高度门太窄 | z ∈ [0.3, 2.2] 漏卡车 | z ∈ [0.1, 4.5] 覆盖全部（nz 略增） |
| ⑥ 大物体中空 | max_radius=8 限制填充 | 多帧多角度覆盖 + 体素化 → **消除** |
| ⑦ int32 截断 | 投影取整误差 | **不投影到图像 → 消除** |
| ⑧ 相机拼缝 | FOV 边界空洞 | 仍在（体素化在 ego 空间，与相机无关） |
| ⑨ 近点覆盖远点 | 单像素单深度 | **消除**（体素化不经过像素层） |
| ⑩ 帧间时序 | 独立计算，无关联 | 多帧聚合本身引入了时序信息 |

### 实现涉及文件

- `OSZ/utils/nuscenes_loader.py`: `build_frame_for_token()` 新增 `n_sweeps` 参数 + 自车运动补偿逻辑
- `OSZ/modules/ray_casting.py`: 新增 `build_bev_occ_from_pointcloud()` 函数，替代 `build_bev_occ_from_voxel_cast`
- `PA_gen_v2/osz_source.py`: 透传 `n_sweeps` 参数

### 关键伪代码

```python
def aggregate_lidar_sweeps(nusc, sample_token, n_sweeps=3):
    """多帧 LiDAR 聚合：当前帧 + 历史 N 个 sweep，全部转到当前 ego 坐标"""
    sample = nusc.get('sample', sample_token)
    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_tok)
    ego_cur = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    T_cur2global = make_tf(ego_cur['translation'], ego_cur['rotation'])

    all_pts_ego = []
    
    # 当前帧
    pc = LidarPointCloud.from_file(dataroot + '/' + lidar_sd['filename'])
    T_l2e = _get_transform(nusc, lidar_sd)
    pts = transform_and_filter(pc, T_l2e)  # LiDAR→ego, 过滤地面
    all_pts_ego.append(pts)

    # 历史 sweep（沿 prev 链回退）
    tok = lidar_sd['prev']
    for _ in range(n_sweeps):
        if not tok:
            break
        sd = nusc.get('sample_data', tok)
        ego_past = nusc.get('ego_pose', sd['ego_pose_token'])
        
        pc_p = LidarPointCloud.from_file(dataroot + '/' + sd['filename'])
        T_l2e_p = _get_transform(nusc, sd)
        pts_p = transform_and_filter(pc_p, T_l2e_p)  # past ego 坐标
        
        # 自车运动补偿: past ego → global → current ego
        T_p2g = make_tf(ego_past['translation'], ego_past['rotation'])
        pts_global = (T_p2g @ to_homogeneous(pts_p).T).T[:, :3]
        pts_cur = (inv(T_cur2global) @ to_homogeneous(pts_global).T).T[:, :3]
        
        pts_cur = pts_cur[pts_cur[:, 2] > 0.2]  # 地面再过滤
        all_pts_ego.append(pts_cur)
        
        tok = sd['prev']
    
    return np.concatenate(all_pts_ego, axis=0)


def build_bev_occ_from_pointcloud(pts_ego, caster, min_points=1):
    """直接体素化：将 ego 坐标系点云转为 BEV occupancy"""
    # 3D 栅格化
    xi = np.floor((pts_ego[:, 0] - caster.bev_range[0]) / caster.bev_res).astype(int)
    yi = np.floor((pts_ego[:, 1] - caster.bev_range[2]) / caster.bev_res).astype(int)
    zi = np.floor((pts_ego[:, 2] - caster.z_min) / caster.z_res).astype(int)
    
    in_grid = (xi >= 0) & (xi < caster.nx) & (yi >= 0) & (yi < caster.ny) & (zi >= 0) & (zi < caster.nz)
    xi, yi, zi = xi[in_grid], yi[in_grid], zi[in_grid]
    
    # 统计每体素点数
    voxel_counts = np.zeros((caster.nx, caster.ny, caster.nz), dtype=np.int16)
    np.add.at(voxel_counts, (xi, yi, zi), 1)
    
    # Occupancy: 体素内点数 >= min_points
    V_occ = voxel_counts > 0  # min_points=1 时等价于至少一个点
    
    # Maxpool z → BEV occ
    bev_occ = V_occ.any(axis=2)
    
    # Morphological closing 填孔（地面过滤后仍可能有微小空洞）
    if _SCIPY_AVAILABLE:
        from scipy.ndimage import binary_closing
        bev_occ = binary_closing(bev_occ, iterations=2)
    
    return bev_occ
```

---

## 多模态 OSZ 质量提升路线

以上是多帧 LiDAR 聚合 + 体素化（纯 LiDAR 方案 B）。以下补充多模态方案，供后续论文实验参考。

### 方案矩阵

```
方案 A：纯 LiDAR 参数调优（审计文档 P0+P1 修复）
  ┌─────┐    ┌────────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │LiDAR│ →  │densify │ →  │voxel │ →  │ray   │ →  │ OSZ  │
  │点云 │    │(改参数) │    │cast  │    │cast  │    │      │
  └─────┘    └────────┘    │(放宽) │    └──────┘    └──────┘
                           └──────┘
  改动: 3-5个参数  |  效果: ★★★  |  风险: 极低

方案 B：多帧 LiDAR + 直接体素化（推荐先行）
  ┌──────────┐    ┌──────────┐    ┌──────┐    ┌──────┐
  │多帧LiDAR │ →  │3D体素化  │ →  │ray   │ →  │ OSZ  │
  │聚合+补偿 │    │(直接占栅)│    │cast  │    │      │
  └──────────┘    └──────────┘    └──────┘    └──────┘
  改动: ~50行新增  |  效果: ★★★★  |  风险: 低

方案 C：LiDAR + Camera 深度补全（Camera-guided depth completion）
  ┌─────┐  ┌──────────────┐    ┌──────┐    ┌──────┐
  │LiDAR│→│NLSPN/PENet   │ →  │voxel │ →  │ OSZ  │
  │点云 │  │深度补全      │    │cast  │    │      │
  ├─────┤  │(RGB引导填孔) │    └──────┘    └──────┘
  │RGB  │  └──────────────┘
  └─────┘
  改动: 引入预训练模型  |  效果: ★★★★  |  风险: 中(模型依赖)

方案 D：LiDAR + 语义分割（避开地面 + 确认障碍物）
  ┌─────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌──────┐
  │LiDAR│→│深度图     │→│障碍物mask│→│voxel │→│ OSZ  │
  │点云 │  └──────────┘  │(语义分割)│  │cast  │  │      │
  ├─────┤                └──────────┘  └──────┘  └──────┘
  │RGB  │→ OneFormer/SegFormer
  └─────┘   road=不标记, vehicle/building/wall=标记
  改动: 引入分割模型     |  效果: ★★★★  |  风险: 中

方案 E：纯视觉 OSZ（论文实验目标）
  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌──────┐
  │RGB   │→│ZoeDepth  │→│障碍物mask│→│voxel │→│ OSZ  │
  │(环视)│  │度量深度  │  │(语义分割)│  │cast  │  │      │
  │      │  ├──────────┤  └──────────┘  └──────┘  └──────┘
  │      │  │OneFormer │
  └──────┘  │语义分割  │
            └──────────┘
  改动: 引入两个模型     |  效果: ★★★★★  |  风险: 高(实验性质)
```

### 方案 C 详解：深度补全

**原理**：LiDAR 提供稀疏但精确的深度锚点，RGB 图像提供语义和纹理引导，网络学会"颜色相似的区域深度也连续"。输出是稠密度量深度图，天然保持物体边界。

**代表模型**：
- **NLSPN** (Park et al., ECCV 2020)：非局部空间传播网络，在 KITTI depth completion benchmark 上曾是 SOTA
- **PENet** (Hu et al., ICCV 2021)：结合双分支编码器，速度更快
- **P3D-SC**：专为自动驾驶设计的实时深度补全

**对 OSZ 的改进**：输出的深度图稠密且边界清晰 → 不需要 densify → 消除误差源 ③④⑥；同时 RGB 提供的语义信息自然地让网络不在地面上产生虚假深度 → 间接解决误差 ②。

### 方案 D 详解：语义分割辅助

**原理**：对 6 台相机图像分别做全景/语义分割，将 road/sidewalk/terrain/sky 等类别的像素深度设 0（不参与体素标记），只保留 {vehicle, truck, bus, building, wall, pole, barrier, traffic_sign} 等类别的像素。

**对 OSZ 的改进**：
- 精确消除地面假点（误差 ②）
- 区分"透明的"障碍物（玻璃建筑）和真实的不透明障碍物
- 减少植被（树叶等）对深度图的噪声

### 方案 E 详解：纯视觉 OSZ

**动机**：论文实验想证明"不依赖 LiDAR 也能做 OSZ"，这在学术上有区分度。

**输入**：6 台环视相机 RGB 图像（nuScenes 标准配置）

**输出**：度量深度图 + per-pixel 语义标签 → BEV occ → ray cast → OSZ

**模型推荐**：
- 深度：ZoeDepth (ZoeD-M12-NK)，在 nuScenes 上 zero-shot 可用
- 语义：OneFormer (Swin-B backbone)，Cityscapes/Mapillary 预训练可迁移到 nuScenes

**挑战**：
1. 度量深度在远距离的绝对误差较 LiDAR 大（50m 处可能 5-10m 偏差）
2. 语义分割对 nuScenes 类别不完全匹配
3. 两个模型叠加的推理延迟（需要优化 batch inference）

### 推荐实施顺序

```
Phase 1 (当前) → 方案 A: 纯 LiDAR 参数调优（~5 个参数，改动量最小）
Phase 2 (当前) → 方案 B: 多帧聚合 + 直接体素化（~50 行，最大改进/代码比）
Phase 3 (实验前) → 方案 D: LiDAR + 语义分割（引入分割模型，验证多模态增益）
Phase 4 (论文) → 方案 E: 纯视觉 OSZ（目标方案，写论文的核心实验）
```

方案 C（深度补全）可在 Phase 3 作为 D 的替代或补充，视模型可用性和效果决定。

## 参考文献

1. Möller, K., Schwarzmeier, L., & Betz, J. (2024). *From Shadows to Safety: Occlusion Tracking and Risk Mitigation for Urban Autonomous Driving.* IV 2024 / arXiv:2504.01408.
   — 提供了基于障碍物多边形的 OSZ 计算和时序跟踪的 reference 实现。

2. Wang, L., Burger, C., & Stiller, C. (2021). *Reasoning about Potential Hidden Traffic Participants by Tracking Occluded Areas.* ITSC 2021.
   — 最早将 OSZ 跟踪与车道拓扑结合的论文。

3. Sanchez, J. M. G., et al. (2022). *Foresee the Unseen: Sequential Reasoning about Hidden Obstacles for Safe Driving.* IV 2022.
   — 质点运动模型传播 OSZ，"previously visible = safe" 的时序排除逻辑。

4. Bhat, S. F., et al. (2023). *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth.* arXiv:2302.12288.
   — 度量单目深度估计，可作为纯视觉 OSZ 的深度输入。
