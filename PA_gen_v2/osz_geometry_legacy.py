"""
osz_geometry_legacy.py  (formerly osz_geometry.py)
====================================================
*** NOT USED BY THE ACTIVE PIPELINE. Kept for reference only. ***

This was PA_gen_v2/'s own independent occlusion-shadow-zone implementation:
raw LiDAR points binned directly into a BEV grid (simple height clip, no
camera reprojection), then a 2D ray cast over that grid.

The active pipeline now gets OSZ from OSZ/modules/ray_casting.py via
PA_gen_v2/osz_source.py instead — see that file's module docstring for why
(short version: per-camera 3D voxel casting against measured depth finds
true occluder surfaces and avoids the range-dependent gaps that plain
point-density binning produces; OSZ/ is the project's single source of
truth for "what is occluded", so PA_gen_v2/ now consumes it directly rather
than maintaining a second, divergent implementation).

Nothing in ghost_vehicle_miner.py, visualize_events.py, run_pipeline.py,
test_units.py, or test_synthetic_e2e.py imports this file anymore. It is
left here in case you want to A/B compare the two geometric approaches on
the same frame — its BEV_SIZE/BEV_RESOLUTION here happen to still match
common/bev_config.py's current settings (both were 0.2m/cell at the time
of this rewrite), but nothing keeps that in sync automatically, so don't
assume it without checking common/bev_config.py's BEV_RESOLUTION_M first.

Coordinate convention note (do not port code from here into osz_source.py
without translating this): this file uses image-style (row, col) indexing
= (ego-y index, ego-x index), i.e. the OPPOSITE axis order from OSZ/'s
(nx, ny) indexing='ij' = (ego-x index, ego-y index). See osz_source.py's
module docstring for the full explanation of why this matters.

---- Original docstring below ----

Occlusion Shadow Zone (OSZ) generation via BEV ray casting.

Core idea:
  For each sample in nuScenes, cast rays from the ego vehicle position
  outward in all BEV directions. A ray is "blocked" the first time it
  hits an occupied voxel (from LiDAR points). Everything BEHIND that
  blocking point along the same ray is the OSZ.

Design principles (Karpathy style):
  - Every function is independently testable.
  - No silent failures: assert shapes and value ranges explicitly.
  - Visualize intermediate outputs in __main__ before trusting them.
  - Keep it simple: no fancy sparse ops, just dense numpy grids.
"""

import numpy as np
from typing import Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import pyquaternion


# ---------------------------------------------------------------------------
# BEV grid parameters — kept as module-level constants so every downstream
# file uses exactly the same grid without any argument passing mistakes.
#
# LEGACY WARNING: these are now independent of common/bev_config.py. The
# active pipeline's grid size lives in common/bev_config.py; changing that
# file will NOT change these numbers. That's intentional — this file is
# frozen as a historical reference implementation, not a live consumer.
# ---------------------------------------------------------------------------
BEV_RANGE_M   = 51.2          # meters, symmetric around ego in x and y
BEV_RESOLUTION = 0.2          # meters per pixel
BEV_SIZE       = int(2 * BEV_RANGE_M / BEV_RESOLUTION)   # 512 x 512
BEV_CENTER     = BEV_SIZE // 2                            # pixel index of ego

# Ray casting parameters
N_RAYS         = 720           # angular resolution: 0.5° per ray
RAY_STEP_M     = BEV_RESOLUTION / 2   # sub-pixel stepping to avoid gaps
MAX_RANGE_M    = BEV_RANGE_M - 1.0   # don't go to the very edge


def _lidar_to_ego(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load LiDAR points for one sample and transform them into the
    ego vehicle frame (flat on the ground plane, z=0 is ground).

    Returns:
        pts: (N, 3) float32 array of [x, y, z] in ego frame.
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data  = nusc.get('sample_data', lidar_token)

    # Load raw points (4 x N: x, y, z, intensity) in lidar sensor frame
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

    # Transform: lidar frame → ego frame
    cs_record = nusc.get('calibrated_sensor',
                         lidar_data['calibrated_sensor_token'])
    rotation    = pyquaternion.Quaternion(cs_record['rotation'])
    translation = np.array(cs_record['translation'])

    pts = pc.points[:3, :].T.astype(np.float32)   # (N, 3)
    pts = (rotation.rotation_matrix @ pts.T).T + translation

    assert pts.shape[1] == 3, f"Expected (N,3), got {pts.shape}"
    return pts


def build_bev_occupancy(pts_ego: np.ndarray,
                        height_min: float = -1.5,
                        height_max: float =  3.0) -> np.ndarray:
    """
    Project ego-frame LiDAR points onto a BEV occupancy grid.

    Only points within [height_min, height_max] are kept (filters ground
    returns below and aerial noise above).

    Returns:
        grid: (BEV_SIZE, BEV_SIZE) bool array.
              True  = occupied (a LiDAR point fell in this cell).
              False = free or unknown.
    """
    # Height filter
    mask = (pts_ego[:, 2] >= height_min) & (pts_ego[:, 2] <= height_max)
    pts_filtered = pts_ego[mask]

    # Convert metric x,y → pixel indices
    # ego is at (BEV_CENTER, BEV_CENTER); x→col, y→row (y-axis points left)
    col = ((pts_filtered[:, 0] + BEV_RANGE_M) / BEV_RESOLUTION).astype(int)
    row = ((pts_filtered[:, 1] + BEV_RANGE_M) / BEV_RESOLUTION).astype(int)

    # Clip to grid bounds
    valid = (col >= 0) & (col < BEV_SIZE) & (row >= 0) & (row < BEV_SIZE)
    col, row = col[valid], row[valid]

    grid = np.zeros((BEV_SIZE, BEV_SIZE), dtype=bool)
    grid[row, col] = True

    n_occupied = int(grid.sum())
    assert n_occupied > 100, \
        f"Only {n_occupied} occupied cells — likely a coordinate error."
    return grid


def cast_osz_mask(occ_grid: np.ndarray) -> np.ndarray:
    """
    Ray-cast from ego center outward in N_RAYS directions.
    A cell is in the OSZ if it lies BEHIND the first occupied cell
    along its ray (i.e., the ray was blocked before reaching it).

    Returns:
        osz: (BEV_SIZE, BEV_SIZE) float32 in [0, 1].
             1.0 = definitely occluded shadow zone.
             0.0 = visible or free.
    """
    assert occ_grid.shape == (BEV_SIZE, BEV_SIZE), \
        f"Expected ({BEV_SIZE},{BEV_SIZE}), got {occ_grid.shape}"

    osz = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)

    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    dx = np.cos(angles)   # x-direction (col)
    dy = np.sin(angles)   # y-direction (row)

    step_px = RAY_STEP_M / BEV_RESOLUTION
    max_steps = int(MAX_RANGE_M / RAY_STEP_M)

    for ray_idx in range(N_RAYS):
        blocked = False
        for step in range(1, max_steps + 1):
            col = BEV_CENTER + dx[ray_idx] * step * step_px
            row = BEV_CENTER + dy[ray_idx] * step * step_px

            c_int = int(round(col))
            r_int = int(round(row))

            if c_int < 0 or c_int >= BEV_SIZE or r_int < 0 or r_int >= BEV_SIZE:
                break

            if not blocked:
                if occ_grid[r_int, c_int]:
                    blocked = True
            else:
                osz[r_int, c_int] = 1.0

    return osz


def get_osz_for_sample(nusc: NuScenes,
                       sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
    """Full legacy pipeline for one sample: LiDAR → occupancy → OSZ mask."""
    pts_ego  = _lidar_to_ego(nusc, sample_token)
    occ_grid = build_bev_occupancy(pts_ego)
    osz_mask = cast_osz_mask(occ_grid)
    return occ_grid, osz_mask


def bev_coords_to_pixel(x_ego: float, y_ego: float) -> Tuple[int, int]:
    """Convert metric (x, y) in ego frame to BEV pixel (col, row).

    col indexes ego-x (axis 0), row indexes ego-y (axis 1). Row is reversed
    so that imshow with extent=[y_max, y_min, x_min, x_max] and origin='lower'
    places ego-left (+y) on the LEFT.
    """
    col = int((x_ego + BEV_RANGE_M) / BEV_RESOLUTION)
    row = int((BEV_RANGE_M - y_ego) / BEV_RESOLUTION)
    return col, row


def pixel_to_bev_coords(col: int, row: int) -> Tuple[float, float]:
    """Inverse of bev_coords_to_pixel."""
    x = col * BEV_RESOLUTION - BEV_RANGE_M
    y = BEV_RANGE_M - row * BEV_RESOLUTION
    return x, y


if __name__ == '__main__':
    print("This is the legacy, no-longer-active OSZ implementation.")
    print("Run PA_gen_v2/osz_source_viz.py instead for the current pipeline's "
          "single-sample OSZ sanity check (OSZ/modules/ray_casting.py via "
          "osz_source.py).")
