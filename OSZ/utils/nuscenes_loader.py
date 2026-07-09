"""
OSZ/utils/nuscenes_loader.py
============================
nuScenes Data Utilities

Provides:
  - NuScenesOSZLoader : iterates samples, returns per-camera depth maps,
                        intrinsics, extrinsics, and optional GT OSZ masks
                        (computed from LiDAR-based ray casting as pseudo-GT).
  - build_frame(sample) / build_frame_for_token(sample_token):
        single-frame camera-depth builders used by both the iterator
        AND by external callers (PA_gen_v2/osz_source.py uses
        build_frame_for_token to query one sample at a time without
        iterating the whole dataset).

nuScenes coordinate conventions
-------------------------------
  - All sensor poses are stored as  sensor → ego → global.
  - We work entirely in the EGO frame at each timestamp.
  - depth_map : LiDAR points projected onto each camera image, depth = z_cam.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

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
                      max_radius: int = 16,
                      depth_discontinuity_thresh: float = 4.0) -> np.ndarray:
    """
    Fill sparse depth map with nearest-neighbour interpolation while
    preserving depth discontinuities.

    Naive nearest-neighbour interpolation smears object boundaries: a
    background hole next to a car can be filled with the car's depth
    because its nearest valid pixel lies on the car surface. That widens
    the car silhouette, inflates the occluder voxels after back-projection,
    and makes OSZ spread far beyond the real vehicle width.

    Instead, for each invalid pixel we look at its K=4 nearest valid
    neighbours. If their depths differ by more than
    depth_discontinuity_thresh, the pixel sits on a depth edge and is left
    unknown (0). Otherwise the closest neighbour's depth is used.

    Parameters tuned per OSZ_ERROR_AUDIT.md P0:
      max_radius increased 8→16 to fill larger object interiors.
      depth_discontinuity_thresh increased 1.5→4.0 to reduce edge gaps
      that caused fragmented BEV occupancy.
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

    # query K=4 nearest neighbors to detect depth-discontinuity boundaries
    k_neighbors = min(4, len(coords))
    dist_k, idx_k = tree.query(grid_coords, k=k_neighbors)
    if k_neighbors == 1:
        dist_k = dist_k[:, None]
        idx_k  = idx_k[:, None]

    values_k = values[idx_k]                          # (H*W, k) candidate depths
    depth_spread = values_k.max(axis=1) - values_k.min(axis=1)  # (H*W,)

    # nearest-neighbor (k=1) depth and distance are the default interpolation
    nearest_dist  = dist_k[:, 0]
    nearest_depth = values_k[:, 0]

    # pixel is at a depth discontinuity if its K nearest valid neighbors span
    # a large depth range, e.g. it sits between a car edge and the background.
    # In that case interpolation is unreliable, so keep it unknown (0).
    is_discontinuous = depth_spread > depth_discontinuity_thresh

    dense = nearest_depth.copy()
    dense[is_discontinuous] = 0.0
    dense = dense.reshape(H, W)
    dist_map = nearest_dist.reshape(H, W)

    # do not fill pixels far from any valid measurement
    dense_mask = dist_map <= max_radius
    dense = dense * dense_mask.astype(np.float32)

    # keep original valid pixels unchanged
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


def filter_ground_points(pts_ego: np.ndarray,
                         z_thresh: float = 0.2) -> np.ndarray:
    """Remove points at or below ground level.

    Ground points projected to camera depth maps create false obstacle
    voxels (the ground at 15m has the same depth as a 0.4m-tall obstacle
    at 15m). Filtering them BEFORE projection eliminates the largest
    source of phantom OSZ on the road surface.

    Args:
        pts_ego: (N, 3) LiDAR points in ego frame (x=fwd, y=left, z=up).
        z_thresh: points with z_ego < z_thresh are ground and removed.

    Returns:
        (M, 3) filtered points with z_ego >= z_thresh.
    """
    return pts_ego[pts_ego[:, 2] >= z_thresh]


def _ego_pose_tf(nusc, sample_token: str):
    """Build ego→global 4×4 transform from the LiDAR sample_data's ego_pose."""
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    T = transform_matrix(ep['translation'], Quaternion(ep['rotation']), inverse=False)
    return T.astype(np.float64)


def aggregate_lidar_sweeps(nusc, sample_token: str,
                           n_sweeps: int = 3,
                           ground_z_thresh: float = 0.2) -> np.ndarray:
    """Aggregate current + historical LiDAR sweeps into one ego-frame point cloud.

    Loads the keyframe LiDAR sweep plus up to n_sweeps previous sweeps (via
    sample_data['prev'] chain — past only, never future). Each historical
    sweep's points are ego-motion-compensated (past_ego → global → current_ego)
    and ground-filtered before being concatenated.

    This bypasses the depth-map pipeline entirely: the caller can voxelize
    the returned points directly in 3D, eliminating densify/voxel-cast errors.

    Args:
        nusc: NuScenes instance.
        sample_token: keyframe sample token (the 'current' frame).
        n_sweeps: how many PAST sweeps to include (0 = keyframe only).
        ground_z_thresh: z_ego cutoff for ground removal (metres).

    Returns:
        (M, 3) float32 array — all points in CURRENT ego frame, ground removed.
    """
    sample = nusc.get('sample', sample_token)
    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_tok)

    # Current ego → global
    T_cur_ego2global = _ego_pose_tf(nusc, sample_token)

    all_pts = []

    # ── Current keyframe sweep ────────────────────────────────────────
    pc = LidarPointCloud.from_file(nusc.dataroot + '/' + lidar_sd['filename'])
    T_l2e = _get_transform(nusc, lidar_sd)
    pts_h = np.concatenate([pc.points[:3].T, np.ones((pc.points.shape[1], 1))], axis=1)
    pts_ego = (T_l2e @ pts_h.T).T[:, :3]
    pts_ego = filter_ground_points(pts_ego, ground_z_thresh)
    all_pts.append(pts_ego.astype(np.float32))

    # ── Historical sweeps (prev chain, past-only) ─────────────────────
    tok = lidar_sd['prev']
    for _ in range(n_sweeps):
        if not tok:
            break
        sd = nusc.get('sample_data', tok)

        # Load past sweep points → past ego frame
        pc_p = LidarPointCloud.from_file(nusc.dataroot + '/' + sd['filename'])
        T_l2e_p = _get_transform(nusc, sd)
        pts_p_h = np.concatenate([pc_p.points[:3].T,
                                  np.ones((pc_p.points.shape[1], 1))], axis=1)
        pts_p_ego = (T_l2e_p @ pts_p_h.T).T[:, :3]

        # Ground filter BEFORE motion compensation (z in past ego frame)
        pts_p_ego = filter_ground_points(pts_p_ego, ground_z_thresh)
        if len(pts_p_ego) == 0:
            tok = sd['prev']
            continue

        # Ego-motion compensation: past_ego → global → current_ego
        T_past_ego2global = _ego_pose_tf(nusc, sd['token'])
        pts_global_h = (T_past_ego2global @
                        np.concatenate([pts_p_ego,
                                        np.ones((len(pts_p_ego), 1))], axis=1).T).T
        pts_cur_ego_h = (np.linalg.inv(T_cur_ego2global) @ pts_global_h.T).T
        pts_cur_ego = pts_cur_ego_h[:, :3].astype(np.float32)

        all_pts.append(pts_cur_ego)

        tok = sd['prev']

    print(f"  [aggregate] {len(all_pts)} sweeps, "
          f"{sum(len(p) for p in all_pts)} points total")
    return np.concatenate(all_pts, axis=0)


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
    def build_frame_for_token(self, sample_token: str) -> dict:
        """
        Build and return a single frame dict (the same shape __iter__ yields)
        for the given sample_token, WITHOUT iterating the whole dataset.

        This is the token-based entry point osz_source.py's _get_loader shim
        uses: that shim wraps a caller-supplied NuScenes object via __new__
        (so it doesn't re-open the nuScenes tables a second time) and only
        sets self.nusc / self.cameras / self.img_h / self.img_w / self._use_mock
        — it does NOT set self.samples. This method therefore looks up the
        sample record directly via self.nusc.get('sample', sample_token)
        rather than scanning self.samples, so it works on both the __init__
        path (self.samples populated) and the shim path (self.samples absent).

        Returns:
            frame : {'sample_token': str, 'cameras': {cam_name: {...}}}
            'cameras' is empty if no camera data could be loaded; callers
            (get_osz_for_sample) check `if not cams` and raise a clear error.
        """
        sample = self.nusc.get('sample', sample_token)
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

        # Ground point filtering (P1 fix): ground points create false
        # obstacle voxels on the road surface. Remove before projection.
        pts_ego = filter_ground_points(pts_ego)

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

        return frame

    def _nuscenes_iter(self):
        for sample in self.samples:
            yield self.build_frame_for_token(sample['token'])

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
