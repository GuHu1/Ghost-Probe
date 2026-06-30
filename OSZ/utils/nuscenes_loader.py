"""
nuScenes Data Utilities
=======================
Provides:
  - NuScenesOSZLoader : iterates samples, returns per-camera depth maps,
                        intrinsics, extrinsics, and optional GT OSZ masks
                        (computed from LiDAR-based ray casting as pseudo-GT).

nuScenes coordinate conventions
--------------------------------
  - All sensor poses are stored as  sensor → ego → global.
  - We work entirely in the EGO frame at each timestamp.
  - depth_map : LiDAR points projected onto each camera image, depth = z_cam.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("[WARN] nuscenes-devkit not found; using synthetic mock data.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from scipy.spatial import cKDTree
from matplotlib import cm


# Cameras used in nuScenes (6-camera rig)
NUSCENES_CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def densify_depth_map(depth_map: np.ndarray,
                      max_radius: int = 8,
                      depth_discontinuity_thresh: float = 1.5) -> np.ndarray:
    """
    对稀疏深度图进行最近邻插值，并对深度不连续边界做保护。

    朴素最近邻插值的问题
    --------------------
    如果直接对所有无效像素取最近的有效像素深度值，会在物体边缘产生"涂抹"
    伪影：car 边缘外侧本属于背景的空洞像素，会被错误地填上 car 的深度值
    （因为 2D 图像距离最近的有效点恰好在 car 表面上）。这会让 car 在深度图
    里的轮廓比真实物理边界更宽，反投影到 3D 后 occluder voxel 也跟着被
    横向拉伸，导致 2D ray casting 阶段把这个虚假的"加宽阴影角宽"投射出去
    ——这正是"被多辆车包围时 OSZ 异常扩散覆盖远超车辆本身宽度的区域"的
    根本原因之一。

    修复方法
    --------
    两阶段最近邻插值：
      1. 先找到每个无效像素最近的 K 个有效像素（K=4）。
      2. 如果这 K 个候选深度值彼此差异超过 depth_discontinuity_thresh
         （说明该像素恰好处于深度不连续边界，比如 car 边缘与背景的交界），
         则不插值，保持 0（未知），而不是随便选一个最近邻深度值。
      3. 否则使用真正的最近邻深度值（K个候选中距离最近的那个）。

    这样可以保证：car 的深度边界不会因为插值而"涂抹"扩散到背景区域。
    """
    H, W = depth_map.shape
    valid = depth_map > 0
    if valid.sum() == 0:
        return depth_map.copy()

    coords = np.array(np.nonzero(valid)).T          # (N, 2)  (y, x)
    values = depth_map[valid]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    tree = cKDTree(coords)

    # 查询 K=4 个最近邻，用来检测该位置是否处于深度不连续边界
    k_neighbors = min(4, len(coords))
    dist_k, idx_k = tree.query(grid_coords, k=k_neighbors)
    if k_neighbors == 1:
        dist_k = dist_k[:, None]
        idx_k  = idx_k[:, None]

    values_k = values[idx_k]                          # (H*W, k) candidate depths
    depth_spread = values_k.max(axis=1) - values_k.min(axis=1)  # (H*W,)

    # 最近邻（k=1）深度值和距离，作为默认插值结果
    nearest_dist  = dist_k[:, 0]
    nearest_depth = values_k[:, 0]

    # 判定该像素是否处于深度不连续区域：
    # K 个最近有效像素的深度跨度超过阈值 → 该像素夹在两个不同深度的物体
    # 之间（例如 car 边缘与背景交界），插值不可信，保持未知（0）。
    is_discontinuous = depth_spread > depth_discontinuity_thresh

    dense = nearest_depth.copy()
    dense[is_discontinuous] = 0.0
    dense = dense.reshape(H, W)
    dist_map = nearest_dist.reshape(H, W)

    # 距离太远的不填充
    dense_mask = dist_map <= max_radius
    dense = dense * dense_mask.astype(np.float32)

    # 保留原始有效像素
    dense[valid] = depth_map[valid]
    return dense


def _get_transform(nusc, record: Dict) -> np.ndarray:
    """Calibrated sensor → ego  (4×4 float64)."""
    cs = nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
    T = transform_matrix(
        cs['translation'],
        Quaternion(cs['rotation']),
        inverse=False,
    )
    return T.astype(np.float32)


def _get_intrinsic(nusc, cam_token: str) -> np.ndarray:
    """Returns (3, 3) camera intrinsic K."""
    sample_data = nusc.get('sample_data', cam_token)
    cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    K = np.array(cs['camera_intrinsic'], dtype=np.float32)
    return K


def project_lidar_to_camera(
    points_ego: np.ndarray,   # (N, 3) LiDAR points in ego frame
    K: np.ndarray,             # (3, 3)
    T_cam2ego: np.ndarray,     # (4, 4)
    img_h: int,
    img_w: int,
    min_dist: float = 1.0,
) -> np.ndarray:
    """
    Project LiDAR points (ego frame) onto a camera image.
    Returns dense depth map (img_h, img_w) float32, 0 = no measurement.
    """
    T_ego2cam = np.linalg.inv(T_cam2ego)

    # Transform to camera frame
    pts_h = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    pts_cam = (T_ego2cam @ pts_h.T).T[:, :3]  # (N, 3)

    # Keep points in front of camera
    mask = pts_cam[:, 2] > min_dist
    pts_cam = pts_cam[mask]

    # Project
    uvw = (K @ pts_cam.T).T  # (M, 3)
    z = uvw[:, 2]
    u = (uvw[:, 0] / z).astype(np.int32)
    v = (uvw[:, 1] / z).astype(np.int32)

    # Filter to image bounds
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[in_img], v[in_img], z[in_img]

    # Build depth map (keep closest point per pixel)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    # Sort by descending depth so closer points overwrite farther ones
    order = np.argsort(-z)
    depth_map[v[order], u[order]] = z[order]
    return depth_map


class NuScenesOSZLoader:
    """
    Iterates nuScenes samples and returns everything needed for OSZ computation.

    Usage:
        loader = NuScenesOSZLoader(dataroot='/data/nuscenes', version='v1.0-mini')
        for frame in loader:
            # frame['cameras'] = { 'CAM_FRONT': { 'depth_map', 'K', 'T_cam2ego' }, ... }
            # frame['sample_token'] = str

    If nuScenes is not available, yields synthetic mock frames instead.
    """

    def __init__(
        self,
        dataroot: str = '/data/sets/nuscenes',
        version: str = 'v1.0-mini',
        cameras: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        img_h: int = 900,
        img_w: int = 1600,
    ):
        self.cameras = cameras or NUSCENES_CAMERAS
        self.max_samples = max_samples
        self.img_h = img_h
        self.img_w = img_w

        if NUSCENES_AVAILABLE and Path(dataroot).exists():
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            self.samples = self.nusc.sample
            if max_samples:
                self.samples = self.samples[:max_samples]
            self._use_mock = False
        else:
            print(f"[INFO] nuScenes data not found at {dataroot}. Using synthetic mock.")
            self._use_mock = True
            self.n_mock = max_samples or 3

    def __len__(self):
        return self.n_mock if self._use_mock else len(self.samples)

    def __iter__(self):
        if self._use_mock:
            yield from self._mock_iter()
        else:
            yield from self._nuscenes_iter()

    # ── Real nuScenes ────────────────────────────────────────────────────────
    def _nuscenes_iter(self):
        for sample in self.samples:
            frame = {'sample_token': sample['token'], 'cameras': {}}

            # ── Load LiDAR (ego frame) ────────────────────────────────────
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_sd = self.nusc.get('sample_data', lidar_token)
            lidar_path = self.nusc.dataroot + '/' + lidar_sd['filename']
            pc = LidarPointCloud.from_file(lidar_path)

            T_lidar2ego = _get_transform(self.nusc, lidar_sd)
            pts_h = np.concatenate(
                [pc.points[:3].T, np.ones((pc.points.shape[1], 1))], axis=1
            )
            pts_ego = (T_lidar2ego @ pts_h.T).T[:, :3]  # (N, 3) ego frame

            # ── Per-camera ────────────────────────────────────────────────
            for cam_name in self.cameras:
                if cam_name not in sample['data']:
                    continue
                cam_token = sample['data'][cam_name]
                cam_sd    = self.nusc.get('sample_data', cam_token)
                T_cam2ego = _get_transform(self.nusc, cam_sd)
                K         = _get_intrinsic(self.nusc, cam_token)

                depth_sparse = project_lidar_to_camera(
                    pts_ego, K, T_cam2ego,
                    self.img_h, self.img_w,
                )
                depth_dense = densify_depth_map(depth_sparse)

                # Load camera image
                image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
                if PIL_AVAILABLE:
                    try:
                        img_path = self.nusc.dataroot + '/' + cam_sd['filename']
                        image = np.array(Image.open(img_path).convert('RGB'))
                    except Exception:
                        pass

                frame['cameras'][cam_name] = {
                    'depth_map': depth_dense,        # (H, W) metres, densified
                    'depth_map_sparse': depth_sparse,# (H, W) original sparse
                    'image':     image,              # (H, W, 3) RGB
                    'K':         K,                  # (3, 3)
                    'T_cam2ego': T_cam2ego,          # (4, 4)
                    'img_h':     self.img_h,
                    'img_w':     self.img_w,
                }

            yield frame

    # ── Synthetic mock (when no data available) ──────────────────────────────
    def _mock_iter(self):
        """
        Synthetic scene in nuScenes ego frame convention (x=fwd, y=left, z=up).
        Camera frame convention: z=optical axis (fwd), x=right, y=down.
        One solid box occluder at ~12m forward + background objects behind it.
        """
        rng = np.random.default_rng(42)

        K = np.array([
            [1266.4, 0,      816.0],
            [0,      1266.4, 491.5],
            [0,      0,      1.0 ],
        ], dtype=np.float32)

        # nuScenes front camera extrinsic:
        # cam_z(fwd) -> ego_x(fwd),  cam_x(right) -> -ego_y,  cam_y(down) -> -ego_z
        def make_cam2ego(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
            """yaw around z-axis (ego frame), then fixed cam-to-ego rotation."""
            # Base rotation: cam optical axis = ego forward
            R_base = np.array([
                [ 0, 0, 1],   # cam_z -> ego_x
                [-1, 0, 0],   # cam_x -> -ego_y  (camera right = ego right = -ego_left)
                [ 0,-1, 0],   # cam_y -> -ego_z
            ], dtype=np.float32)
            # Yaw rotation in ego frame
            yaw = np.deg2rad(yaw_deg)
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,            0,           1],
            ], dtype=np.float32)
            R = Rz @ R_base
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            return T

        # 3 front cameras with yaw offsets
        cam_configs = [
            # (name,            yaw_deg, tx,  ty,  tz)
            ('CAM_FRONT',            0,  1.5,  0.0, 1.5),
            ('CAM_FRONT_LEFT',      55,  1.5,  0.5, 1.5),
            ('CAM_FRONT_RIGHT',    -55,  1.5, -0.5, 1.5),
        ]

        for i in range(self.n_mock):
            frame = {'sample_token': f'mock_{i:04d}', 'cameras': {}}

            ox = 12.0 + rng.uniform(-1.0, 1.0)   # occluder x (forward)
            oy =  1.5 + rng.uniform(-0.3, 0.3)   # occluder y (lateral)

            for cam_name, yaw_deg, tx, ty, tz in cam_configs:
                if cam_name not in self.cameras:
                    continue
                T_cam2ego = make_cam2ego(yaw_deg, tx, ty, tz)

                # Occluder front face (dense, facing ego)
                box_pts = []
                for dy in np.linspace(-1.5, 1.5, 50):
                    for dz in np.linspace(0.05, 1.75, 35):
                        box_pts.append([ox, oy + dy, dz])
                # Occluder side walls
                for dx in np.linspace(0, 4.0, 25):
                    for dz in np.linspace(0.05, 1.75, 25):
                        box_pts.append([ox + dx, oy - 1.5, dz])
                        box_pts.append([ox + dx, oy + 1.5, dz])
                box_pts = np.array(box_pts, dtype=np.float32)

                # Ground plane
                xs_g = np.linspace(1, 50, 80)
                ys_g = np.linspace(-10, 10, 40)
                xx, yy = np.meshgrid(xs_g, ys_g)
                gnd = np.stack([xx.ravel(), yy.ravel(),
                                np.zeros(xx.size)], axis=1).astype(np.float32)

                # Background objects BEHIND occluder (should be shadow zone)
                bg_pts = []
                for dx in np.linspace(0, 5, 20):
                    for dy in np.linspace(-1.2, 1.2, 20):
                        for dz in np.linspace(0.1, 1.6, 10):
                            bg_pts.append([ox + 5 + dx, oy + dy, dz])
                bg_pts = np.array(bg_pts, dtype=np.float32)

                pts_ego = np.concatenate([box_pts, gnd, bg_pts], axis=0)
                depth_sparse = project_lidar_to_camera(
                    pts_ego, K, T_cam2ego, self.img_h, self.img_w
                )
                depth_dense = densify_depth_map(depth_sparse)

                # Synthetic reference image from dense depth visualization
                depth_norm = depth_dense / (depth_dense.max() + 1e-6)
                image = (cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)

                frame['cameras'][cam_name] = {
                    'depth_map': depth_dense,
                    'depth_map_sparse': depth_sparse,
                    'image':     image,
                    'K':         K,
                    'T_cam2ego': T_cam2ego,
                    'img_h':     self.img_h,
                    'img_w':     self.img_w,
                }

            yield frame
