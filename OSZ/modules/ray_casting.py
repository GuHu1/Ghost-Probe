"""
Stage 1 + 2: 3D Ray Casting → Height-stratified Voxel Occlusion Annotation
===========================================================================
For each camera:
  1. Unproject depth map pixels → 3D world rays
  2. For each voxel (x,y,z) in height gate [z_min, z_max]:
       project voxel center → camera image plane
       if depth_at_pixel < voxel_distance_from_cam → shadow voxel
  3. Output: V_occ^c  (X, Y, Z) binary voxel grid
"""

import numpy as np
from typing import Tuple


def build_bev_occ_from_voxel_cast(
    cameras: dict,
    caster: "RayCaster3D",
) -> np.ndarray:
    """
    Build a solid BEV obstacle map by per-camera 3D voxel casting.

    Each camera's cast returns OCCLUDER SURFACE voxels (where a physical
    surface actually sits), not shadow voxels. Max-pooling these along the
    height axis and unioning across cameras gives bev_occ — the set of
    true obstacles. Occlusion shadow is computed separately by cast_osz_2d
    over this solid map, so shadows never become new "walls".

    The voxel cast avoids the range-dependent gaps of point-density
    binning: it queries every voxel directly against the measured depth at
    the pixel it projects to, regardless of how far the surface is.

    Returns:
        bev_occ : (nx, ny) bool — union of all cameras' occluder-surface
                  BEV occupancy (obstacles, not shadows)
    """
    nx, ny = caster.nx, caster.ny
    bev_occ = np.zeros((nx, ny), dtype=bool)

    for cam_name, cam_data in cameras.items():
        V_occ = caster.cast(
            depth_map=cam_data['depth_map'],
            intrinsic=cam_data['K'],
            cam2ego=cam_data['T_cam2ego'],
        )
        M_occ = voxel_to_bev_maxpool(V_occ)
        bev_occ |= M_occ

    return bev_occ


def cast_osz_2d(bev_occ: np.ndarray,
                caster: "RayCaster3D",
                substep: float = 0.25) -> np.ndarray:
    """
    Ego-centric 360° 2D ray casting over an ALREADY-SOLID bev_occ grid.

    Because bev_occ now comes from the voxel-cast method (no gaps), the
    ray casting itself can use a small, safe substep without worrying
    about missing thin walls — there are none.

    Args:
        bev_occ : (nx, ny) bool, solid obstacle map (no point-density gaps)
        caster  : provides nx, ny, bev_range, bev_res
        substep : ray step size in BEV CELLS (not metres). 0.25 means the
                  ray advances a quarter-cell per iteration — small enough
                  that it cannot skip over a single occupied cell.

    Returns:
        osz_mask : (nx, ny) bool — cells lying behind the first occluder
                   along their ray from ego.
    """
    nx, ny = caster.nx, caster.ny
    x_min, x_max, y_min, y_max = caster.bev_range

    ego_xi = int(np.floor((0.0 - x_min) / caster.bev_res))
    ego_yi = int(np.floor((0.0 - y_min) / caster.bev_res))

    osz_mask = np.zeros((nx, ny), dtype=bool)
    if not (0 <= ego_xi < nx and 0 <= ego_yi < ny):
        return osz_mask

    # Angular resolution: fine enough that adjacent rays don't leave gaps
    # at the maximum range. At range R cells, angular spacing of dtheta
    # leaves a gap of R*dtheta cells between rays — we want that << 1.
    max_range_cells = max(nx, ny)
    n_angles = int(2 * np.pi * max_range_cells / substep)
    n_angles = max(n_angles, 720)   # floor at 0.5° even for small grids
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    max_steps = int(max_range_cells / substep)

    for ray_idx in range(n_angles):
        dx, dy = cos_a[ray_idx], sin_a[ray_idx]
        x, y = float(ego_xi), float(ego_yi)
        hit = False

        for _ in range(max_steps):
            x += dx * substep
            y += dy * substep
            xi_i = int(round(x))
            yi_i = int(round(y))

            if not (0 <= xi_i < nx and 0 <= yi_i < ny):
                break

            if hit:
                osz_mask[xi_i, yi_i] = True
            elif bev_occ[xi_i, yi_i]:
                hit = True   # this cell itself is the occluder, not shadow

    return osz_mask


def compute_osz_from_ego_raycasting(
    cameras: dict,
    caster: "RayCaster3D",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full OSZ pipeline: 3D voxel cast → solid BEV occupancy → 2D ray casting.

      Step 1 (3D): per-camera voxel cast -> solid per-camera BEV occupancy
                   -> union across cameras.
      Step 2 (2D): ego-centric 360° ray casting over the solid BEV map
                   -> everything behind the first occluder is OSZ.

    Returns:
        osz_mask : (nx, ny) bool — ego-centric occlusion shadow zone
        bev_occ  : (nx, ny) bool — solid BEV obstacle map (height-gated)
    """
    bev_occ  = build_bev_occ_from_voxel_cast(cameras, caster)
    osz_mask = cast_osz_2d(bev_occ, caster)
    return osz_mask, bev_occ


class RayCaster3D:
    """
    Vectorized 3D ray casting for a single camera.
    All coordinates are in the ego-vehicle world frame (nuScenes convention).

    Args:
        bev_range   : (x_min, x_max, y_min, y_max) in metres, ego-centred
        bev_res     : BEV grid resolution in metres/cell  (e.g. 0.2)
        z_min/z_max : height gate for vehicle-body voxels (metres)
        z_res       : voxel height resolution (metres)
        depth_scale : multiplier to convert raw depth map values → metres
    """

    def __init__(
        self,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        bev_res: float = 0.2,
        z_min: float = 0.3,
        z_max: float = 2.2,
        z_res: float = 0.2,
        depth_scale: float = 1.0,
    ):
        self.bev_range = bev_range  # (x_min, x_max, y_min, y_max)
        self.bev_res = bev_res
        self.z_min = z_min
        self.z_max = z_max
        self.z_res = z_res
        self.depth_scale = depth_scale

        # Pre-compute voxel grid dimensions
        self.nx = int((bev_range[1] - bev_range[0]) / bev_res)
        self.ny = int((bev_range[3] - bev_range[2]) / bev_res)
        self.nz = int((z_max - z_min) / z_res)

        # Voxel centre coordinates in world frame [nx, ny, nz, 3]
        xs = np.linspace(bev_range[0] + bev_res / 2, bev_range[1] - bev_res / 2, self.nx)
        ys = np.linspace(bev_range[2] + bev_res / 2, bev_range[3] - bev_res / 2, self.ny)
        zs = np.linspace(z_min + z_res / 2, z_max - z_res / 2, self.nz)

        # shape: (nx*ny*nz, 3)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        self.voxel_centers = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=-1
        ).astype(np.float32)  # (N_vox, 3)

    def cast(
        self,
        depth_map: np.ndarray,         # (H, W)  depth in metres (already scaled)
        intrinsic: np.ndarray,          # (3, 3)  camera intrinsic K
        cam2ego: np.ndarray,            # (4, 4)  camera → ego extrinsic
        max_depth: float = 70.0,
        surface_tolerance: float = None,
    ) -> np.ndarray:
        """
        Returns V_occ: (nx, ny, nz) bool array marking OCCLUDER SURFACES.

        A voxel is occupied only if the measured depth at the pixel it
        projects to is approximately equal to the voxel's own distance from
        the camera (within surface_tolerance). This identifies real physical
        surfaces, not shadows.

        Shadow computation belongs to cast_osz_2d, which traces rays over
        this solid occupancy map. Keeping the two stages separate prevents
        3D shadows from being treated as new 2D walls.

        Args:
            surface_tolerance: voxels within this distance (metres) of the
                measured depth are considered "on the surface". Defaults
                to one BEV cell's diagonal.
        """
        if surface_tolerance is None:
            surface_tolerance = max(self.bev_res, self.z_res) * 1.5

        H, W = depth_map.shape
        K = intrinsic
        T_c2e = cam2ego                    # 4×4
        T_e2c = np.linalg.inv(T_c2e)      # ego → camera

        # ── Transform voxel centres to camera frame ──────────────────────────
        n = self.voxel_centers.shape[0]
        pts_ego_h = np.concatenate(
            [self.voxel_centers, np.ones((n, 1), dtype=np.float32)], axis=1
        )                                  # (N, 4)
        pts_cam_h = (T_e2c @ pts_ego_h.T).T  # (N, 4)
        pts_cam = pts_cam_h[:, :3]            # (N, 3)  in camera frame

        # Keep only points in front of camera (z_cam > 0)
        valid_front = pts_cam[:, 2] > 0.1
        pts_cam_v = pts_cam[valid_front]      # (M, 3)

        # ── Project to image plane ────────────────────────────────────────────
        uvw = (K @ pts_cam_v.T).T            # (M, 3)
        z_cam = uvw[:, 2]
        u = (uvw[:, 0] / z_cam).astype(np.int32)
        v = (uvw[:, 1] / z_cam).astype(np.int32)

        # Keep only projections inside image
        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx_valid = np.where(valid_front)[0][in_image]
        u_valid   = u[in_image]
        v_valid   = v[in_image]
        z_valid   = z_cam[in_image]

        # ── Compare voxel depth to measured surface depth ────────────────────
        d_obs = depth_map[v_valid, u_valid].astype(np.float32)
        # Surface condition: voxel distance ≈ measured depth → a real
        # object surface occupies this voxel (this IS the occluder).
        # NOT "voxel is further than measured depth" (that would mark
        # everything behind any object as occupied, recreating the same
        # shadow-as-wall bug).
        valid_depth = (d_obs > 0) & (d_obs < max_depth)
        on_surface  = np.abs(z_valid - d_obs) <= surface_tolerance
        is_occluder = valid_depth & on_surface

        # ── Build output voxel grid ───────────────────────────────────────────
        V_occ = np.zeros(self.nx * self.ny * self.nz, dtype=bool)
        V_occ[idx_valid[is_occluder]] = True
        return V_occ.reshape(self.nx, self.ny, self.nz)


def voxel_to_bev_maxpool(V_occ: np.ndarray) -> np.ndarray:
    """
    Stage 3: Z-axis max-pool.
    Input : (nx, ny, nz) bool
    Output: (nx, ny) bool  — M_occ^c
    """
    return V_occ.any(axis=2)
