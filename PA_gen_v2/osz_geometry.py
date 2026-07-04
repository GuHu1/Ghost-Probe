"""
osz_geometry.py
---------------
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

    Note on the binary nature:
        The mask is geometrically hard — a cell either is or isn't behind
        an occluder.  Any "soft" boundary effect should come from
        uncertainty in the occupancy grid itself, NOT from post-processing
        like CRF (which belongs to depth estimation, not geometry).
    """
    assert occ_grid.shape == (BEV_SIZE, BEV_SIZE), \
        f"Expected ({BEV_SIZE},{BEV_SIZE}), got {occ_grid.shape}"

    osz = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)

    # Pre-compute unit direction vectors for all rays
    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    dx = np.cos(angles)   # x-direction (col)
    dy = np.sin(angles)   # y-direction (row)

    # Ray step in pixels
    step_px = RAY_STEP_M / BEV_RESOLUTION
    max_steps = int(MAX_RANGE_M / RAY_STEP_M)

    for ray_idx in range(N_RAYS):
        blocked = False
        for step in range(1, max_steps + 1):
            # Current position along ray in pixel coords
            col = BEV_CENTER + dx[ray_idx] * step * step_px
            row = BEV_CENTER + dy[ray_idx] * step * step_px

            c_int = int(round(col))
            r_int = int(round(row))

            # Out-of-bounds → stop this ray
            if c_int < 0 or c_int >= BEV_SIZE or r_int < 0 or r_int >= BEV_SIZE:
                break

            if not blocked:
                if occ_grid[r_int, c_int]:
                    blocked = True   # this cell is the occluder itself
            else:
                # Everything past the first occluder is OSZ
                osz[r_int, c_int] = 1.0

    return osz


def get_osz_for_sample(nusc: NuScenes,
                       sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for one sample: LiDAR → occupancy → OSZ mask.

    Returns:
        occ_grid : (BEV_SIZE, BEV_SIZE) bool   — occupied cells
        osz_mask : (BEV_SIZE, BEV_SIZE) float32 — OSZ cells in [0,1]
    """
    pts_ego  = _lidar_to_ego(nusc, sample_token)
    occ_grid = build_bev_occupancy(pts_ego)
    osz_mask = cast_osz_mask(occ_grid)
    return occ_grid, osz_mask


def bev_coords_to_pixel(x_ego: float, y_ego: float) -> Tuple[int, int]:
    """
    Convert a metric (x, y) position in ego frame to BEV pixel (col, row).
    Useful for projecting bounding box centers onto the grid.
    """
    col = int((x_ego + BEV_RANGE_M) / BEV_RESOLUTION)
    row = int((y_ego + BEV_RANGE_M) / BEV_RESOLUTION)
    return col, row


def pixel_to_bev_coords(col: int, row: int) -> Tuple[float, float]:
    """Inverse of bev_coords_to_pixel."""
    x = col * BEV_RESOLUTION - BEV_RANGE_M
    y = row * BEV_RESOLUTION - BEV_RANGE_M
    return x, y


# ---------------------------------------------------------------------------
# Self-test: run this file directly to visually verify OSZ output.
# Usage:  python osz_geometry.py --dataroot /path/to/nuscenes --version v1.0-mini
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/data/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which sample index to visualize (0-based)')
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Pick a sample
    sample_token = nusc.sample[args.sample_idx]['token']
    print(f"Processing sample {args.sample_idx}: {sample_token}")

    occ_grid, osz_mask = get_osz_for_sample(nusc, sample_token)

    # --- Sanity checks ---
    osz_fraction = osz_mask.mean()
    print(f"OSZ coverage: {osz_fraction*100:.1f}% of BEV grid")
    assert 0.01 < osz_fraction < 0.70, \
        f"OSZ fraction {osz_fraction:.2f} looks wrong — check ray casting."
    print("Sanity checks passed.")

    # --- Visualize ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Sample {args.sample_idx} | {sample_token[:16]}...',
                 fontsize=11)

    axes[0].imshow(occ_grid, cmap='gray', origin='lower')
    axes[0].set_title('BEV Occupancy (LiDAR)')
    axes[0].plot(BEV_CENTER, BEV_CENTER, 'r+', markersize=12, markeredgewidth=2)

    axes[1].imshow(osz_mask, cmap='hot', origin='lower', vmin=0, vmax=1)
    axes[1].set_title('OSZ Mask (ray cast)')
    axes[1].plot(BEV_CENTER, BEV_CENTER, 'b+', markersize=12, markeredgewidth=2)

    # Overlay: occupancy in gray, OSZ in red, free in black
    overlay = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    overlay[occ_grid]        = [0.8, 0.8, 0.8]   # gray  = occupied
    overlay[osz_mask > 0.5]  = [0.9, 0.2, 0.2]   # red   = OSZ
    axes[2].imshow(overlay, origin='lower')
    axes[2].set_title('Overlay (gray=occ, red=OSZ)')
    axes[2].plot(BEV_CENTER, BEV_CENTER, 'g+', markersize=12, markeredgewidth=2)

    legend = [mpatches.Patch(color=[0.8,0.8,0.8], label='Occupied'),
              mpatches.Patch(color=[0.9,0.2,0.2], label='OSZ'),
              mpatches.Patch(color='green', label='Ego')]
    axes[2].legend(handles=legend, loc='upper right', fontsize=8)

    for ax in axes:
        ax.set_xlabel('col (→ ego-x)')
        ax.set_ylabel('row (→ ego-y)')

    plt.tight_layout()
    out_path = '/home/claude/phantom_agent/osz_sample_viz.png'
    plt.savefig(out_path, dpi=120)
    print(f"Saved visualization to {out_path}")
    plt.close()
