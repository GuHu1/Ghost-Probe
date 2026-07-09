"""
OSZ Pipeline — Main Runner
===========================
Ties all stages together:

  Stage 1+2 : 3D ray casting  →  V_occ^c  (per camera, height-stratified)
  Stage 3   : Z-axis max-pool →  M_occ^c  (per camera BEV mask)
  Stage 4a  : Multi-camera AND → M_OSZ   (raw geometric OSZ)
  Stage 4b  : Drivable filter →  M_OSZ_PA = M_OSZ ∩ drivable area
  Stage 5   : Visualization   →  PNG exports

Run:
  # With real nuScenes data:
  python run_osz_pipeline.py --dataroot /data/sets/nuscenes --version v1.0-mini

  # Without data (synthetic mock):
  python run_osz_pipeline.py --mock
"""

import argparse
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for common/

from modules.ray_casting     import RayCaster3D, voxel_to_bev_maxpool, compute_osz_from_ego_raycasting

# drivable_filter.py depends on shapely + the nuScenes map API. The core OSZ
# pipeline (geometric ray casting) works without them, so guard the import and
# skip the optional semantic filter when they are missing.
try:
    from modules.drivable_filter import build_drivable_mask, filter_osz_by_drivable, MAP_AVAILABLE
    _DRIVABLE_FILTER_AVAILABLE = True
except ImportError:
    build_drivable_mask = filter_osz_by_drivable = None
    MAP_AVAILABLE = False
    _DRIVABLE_FILTER_AVAILABLE = False

from utils.nuscenes_loader   import NuScenesOSZLoader
from visualize.bev_viz       import (plot_camera_osz_comparison,
                                     plot_gt_osz, plot_osz_explained)
from common.bev_config import (
    BEV_RANGE_XYXY   as DEFAULT_BEV_RANGE,
    BEV_RESOLUTION_M as DEFAULT_BEV_RES,
)


# ═══════════════════════════════════════════════════════════════════════════
# BEV depth aggregation helper
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_depth_bev(
    cameras: dict,
    caster: RayCaster3D,
) -> np.ndarray:
    """
    For each BEV cell, store the minimum depth measured across all cameras.
    Returns (nx, ny) float32.
    """
    depth_bev = np.zeros((caster.nx, caster.ny), dtype=np.float32)
    count_bev = np.zeros((caster.nx, caster.ny), dtype=np.int32)

    x_min, x_max, y_min, y_max = caster.bev_range
    res = caster.bev_res

    for cam_name, cam_data in cameras.items():
        depth_map = cam_data['depth_map']  # (H, W)
        K         = cam_data['K']
        T_c2e     = cam_data['T_cam2ego']
        H, W      = depth_map.shape

        T_e2c = np.linalg.inv(T_c2e)

        # Build grid of BEV cell centres (nx, ny, 2) → project to camera
        xs = np.linspace(x_min + res / 2, x_max - res / 2, caster.nx)
        ys = np.linspace(y_min + res / 2, y_max - res / 2, caster.ny)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny)
        zz = np.full_like(xx, 0.8)  # sample at vehicle centroid height

        pts_ego = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones(caster.nx * caster.ny)], axis=1)
        pts_cam = (T_e2c @ pts_ego.T).T[:, :3]

        valid = pts_cam[:, 2] > 0.1
        uvw = (K @ pts_cam[valid].T).T
        z_c = uvw[:, 2]
        u   = (uvw[:, 0] / z_c).astype(np.int32)
        v   = (uvw[:, 1] / z_c).astype(np.int32)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx_all = np.where(valid)[0][in_img]
        d_vals  = depth_map[v[in_img], u[in_img]]

        valid_depth = d_vals > 0
        idx_flat = idx_all[valid_depth]
        xi = idx_flat // caster.ny
        yi = idx_flat  % caster.ny

        np.add.at(depth_bev, (xi, yi), d_vals[valid_depth])
        np.add.at(count_bev, (xi, yi), 1)

    mask = count_bev > 0
    depth_bev[mask] /= count_bev[mask]
    return depth_bev


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OSZ Pipeline Demo on nuScenes")
    parser.add_argument('--dataroot',  type=str, default='/data/sets/nuscenes')
    parser.add_argument('--version',   type=str, default='v1.0-mini')
    parser.add_argument('--mock',      action='store_true',
                        help='Use synthetic mock data (no nuScenes needed)')
    parser.add_argument('--max_samples', type=int, default=3,
                        help='Number of frames to process')
    parser.add_argument('--bev_range', type=float, nargs=4,
                        default=list(DEFAULT_BEV_RANGE),
                        metavar=('X_MIN','X_MAX','Y_MIN','Y_MAX'),
                        help='BEV range in metres (default from common/bev_config.py)')
    parser.add_argument('--bev_res',   type=float, default=DEFAULT_BEV_RES,
                        help='BEV resolution in metres (default from common/bev_config.py)')
    parser.add_argument('--z_min',     type=float, default=0.3)
    parser.add_argument('--z_max',     type=float, default=2.2)
    parser.add_argument('--z_res',     type=float, default=0.3)
    parser.add_argument('--outdir',    type=str, default='./osz_output')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OSZ Pipeline Demo")
    print(f"  BEV range : {args.bev_range}  res={args.bev_res}m")
    print(f"  Height gate: z ∈ [{args.z_min}, {args.z_max}]m")
    print("=" * 60)

    # ── Init modules ─────────────────────────────────────────────────────
    caster = RayCaster3D(
        bev_range  = tuple(args.bev_range),
        bev_res    = args.bev_res,
        z_min      = args.z_min,
        z_max      = args.z_max,
        z_res      = args.z_res,
    )
    print(f"  Voxel grid : {caster.nx} × {caster.ny} × {caster.nz}  "
          f"({caster.nx * caster.ny * caster.nz:,} voxels)")

    loader = NuScenesOSZLoader(
        dataroot    = args.dataroot,
        version     = args.version,
        max_samples = args.max_samples,
    )
    if args.mock:
        loader._use_mock = True

    # ── Per-frame processing ──────────────────────────────────────────────
    all_frames = list(loader)

    for frame_idx, frame in enumerate(all_frames):
        token   = frame['sample_token']
        cameras = frame['cameras']
        print(f"\n[Frame {frame_idx+1}/{len(all_frames)}]  token={token}")
        if not cameras:
            print("  No cameras — skip")
            continue

        # Stage 1+2+3: per-camera voxel shadow → BEV mask
        per_cam_masks = {}
        cam_images = {}
        cam_depths = {}
        t0 = time.time()
        for cam_name, cam_data in cameras.items():
            t_c = time.time()
            V_occ = caster.cast(
                depth_map  = cam_data['depth_map'],
                intrinsic  = cam_data['K'],
                cam2ego    = cam_data['T_cam2ego'],
            )
            M_occ = voxel_to_bev_maxpool(V_occ)
            per_cam_masks[cam_name] = M_occ
            cam_images[cam_name] = cam_data.get('image', np.zeros((cam_data['img_h'], cam_data['img_w'], 3), dtype=np.uint8))
            cam_depths[cam_name] = cam_data['depth_map']
            print(f"  {cam_name:20s}  shadow cells={M_occ.sum():5d}  "
                  f"({time.time()-t_c:.2f}s)")

        print(f"  Ray casting total: {time.time()-t0:.2f}s")

        # Stage 4a: ego-centric 2D BEV ray casting for OSZ
        # (replaces multi-camera AND — produces large continuous shadow zones
        #  behind occluders as seen from the ego vehicle)
        osz_raw, bev_occ = compute_osz_from_ego_raycasting(cameras, caster)
        print(f"  Ego OSZ cells: {osz_raw.sum()}  "
              f"/ {caster.nx * caster.ny} BEV cells")

        # BEV depth map for visualization
        depth_bev = aggregate_depth_bev(cameras, caster)

        # Stage 4b: semantic filter — intersect geometric OSZ with
        # vehicle-plausible area from nuScenes HD map.
        # This is the critical step that resolves the "surrounded by buildings
        # = 70% OSZ" problem: buildings are geometrically occluding, but
        # their shadow is not a valid PA candidate region.
        drivable_mask = None
        osz_pa = osz_raw  # fallback: unfiltered (used when no map available)
        if (not loader._use_mock and _DRIVABLE_FILTER_AVAILABLE
                and MAP_AVAILABLE and hasattr(loader, 'nusc')):
            try:
                drivable_mask = build_drivable_mask(
                    nusc        = loader.nusc,
                    sample_token= token,
                    bev_range   = tuple(args.bev_range),
                    bev_res     = args.bev_res,
                )
                osz_pa = filter_osz_by_drivable(osz_raw, drivable_mask)
                kept_pct = osz_pa.sum() / max(osz_raw.sum(), 1) * 100
                print(f"  PA-relevant OSZ: {osz_pa.sum()} cells "                      f"({kept_pct:.1f}% of raw OSZ retained after "                      f"drivable-area filter)")
            except Exception as e:
                print(f"  [WARN] Drivable filter failed: {e}")
                osz_pa = osz_raw
        else:
            print("  Drivable filter: skipped (mock mode or map unavailable)")

        # ── Visualize ────────────────────────────────────────────────────
        # ── Camera vs BEV comparison (with image & frame token) ────────────
        fig3 = plot_camera_osz_comparison(
            images       = cam_images,
            depth_maps   = cam_depths,
            per_cam_masks= per_cam_masks,
            osz_mask     = osz_raw,
            refined_mask = None,
            depth_bev    = depth_bev,
            bev_occ      = bev_occ,
            osz_pa       = osz_pa,
            bev_range    = tuple(args.bev_range),
            sample_token = token,
            save_path    = str(outdir / f"frame_{frame_idx:04d}_comparison.png"),
        )
        plt.close(fig3)

        # GT bounding box overlay — shows which annotations are phantom candidates
        if drivable_mask is not None and not loader._use_mock and hasattr(loader, 'nusc'):
            try:
                fig5 = plot_gt_osz(
                    osz_pa       = osz_pa,
                    bev_occ      = bev_occ,
                    drivable_mask= drivable_mask,
                    nusc         = loader.nusc,
                    sample_token = token,
                    bev_range    = tuple(args.bev_range),
                    bev_res      = args.bev_res,
                    save_path    = str(outdir / f"frame_{frame_idx:04d}_pa.png"),
                )
                plt.close(fig5)
            except Exception as e:
                print(f"  [WARN] GT viz failed: {e}")

        # OSZ explained: single-panel view showing occluders (orange) + shadow
        # (black) + road (dark grey) — same palette as PA_gen_v2.
        try:
            fig_osz_exp = plot_osz_explained(
                osz_pa        = osz_pa,
                bev_occ       = bev_occ,
                drivable_mask = drivable_mask,
                bev_range     = tuple(args.bev_range),
                sample_token  = token,
                save_path     = str(outdir / f"frame_{frame_idx:04d}_osz.png"),
                draw_lanes    = (drivable_mask is not None and
                                 not loader._use_mock and
                                 hasattr(loader, 'nusc')),
                nusc          = loader.nusc if (not loader._use_mock and
                                                hasattr(loader, 'nusc')) else None,
            )
            plt.close(fig_osz_exp)
        except Exception as e:
            print(f"  [WARN] OSZ explained viz failed: {e}")

        # Save numpy arrays for downstream PA framework integration
        np.save(outdir / f"frame_{frame_idx:04d}_osz_raw.npy",     osz_raw)
        np.save(outdir / f"frame_{frame_idx:04d}_depth_bev.npy",   depth_bev)
        np.save(outdir / f"frame_{frame_idx:04d}_osz_pa.npy",      osz_pa)
        if drivable_mask is not None:
            np.save(outdir / f"frame_{frame_idx:04d}_drivable.npy", drivable_mask)

    print(f"\n✓ Done. Outputs in: {outdir}/")
    return outdir


if __name__ == '__main__':
    main()
