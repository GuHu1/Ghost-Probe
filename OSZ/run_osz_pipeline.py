"""
OSZ Pipeline — Main Runner
===========================
Ties all stages together:

  Stage 1+2 : 3D ray casting  →  V_occ^c  (per camera, height-stratified)
  Stage 3   : Z-axis max-pool →  M_occ^c  (per camera BEV mask)
  Stage 4a  : Multi-camera AND → M_OSZ   (raw geometric OSZ)
  Stage 4b  : passthrough     →  M_OSZ_refined = M_OSZ (CRF removed; see note in code)
  [Optional] Stage 5: CNN refinement training loop (self-supervised via LiDAR GT)

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

# crf_refine.py depends on torch; don't make the whole pipeline fail on CPU-only
# environments just because someone wants to run geometric OSZ without CNN.
try:
    from modules.crf_refine import CRFBoundaryRefiner, HierarchicalOSZLoss, OSZRefineCNN
    _CRF_AVAILABLE = True
except ImportError:
    CRFBoundaryRefiner = HierarchicalOSZLoss = OSZRefineCNN = None
    _CRF_AVAILABLE = False
from utils.nuscenes_loader   import NuScenesOSZLoader
from visualize.bev_viz       import (plot_bev_osz, plot_refinement_comparison,
                                     plot_camera_osz_comparison, plot_pa_osz,
                                     plot_gt_osz)
from common.bev_config import (
    BEV_RANGE_XYXY   as DEFAULT_BEV_RANGE,
    BEV_RESOLUTION_M as DEFAULT_BEV_RES,
)


# ═══════════════════════════════════════════════════════════════════════════
# BEV depth aggregation helper (for CRF pairwise term)
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
# Self-supervised CNN training (one epoch demo)
# ═══════════════════════════════════════════════════════════════════════════

def train_cnn_one_epoch(model, optimizer, loss_fn, dataset_frames, caster, device):
    """
    Self-supervised single epoch:
      - GT = geometric OSZ (pseudo-label from LiDAR voxel-cast)
      - Input = [geometric_mask, depth_bev] (2 channels)
    Demonstrates the training loop structure; not a full training run.
    """
    import torch

    model.train()
    epoch_loss = {}

    for frame_idx, frame in enumerate(dataset_frames):
        cameras = frame['cameras']
        if not cameras:
            continue

        # Forward pass through geometric pipeline to get pseudo-GT
        per_cam_masks = {}
        for cam_name, cam_data in cameras.items():
            V = caster.cast(cam_data['depth_map'], cam_data['K'], cam_data['T_cam2ego'])
            per_cam_masks[cam_name] = voxel_to_bev_maxpool(V)

        osz_raw = np.ones((caster.nx, caster.ny), dtype=bool)
        for m in per_cam_masks.values():
            osz_raw &= m

        depth_bev = aggregate_depth_bev(cameras, caster)
        # GT for the CNN is the geometric OSZ itself (no CRF pre-processing)
        osz_as_float = osz_raw.astype(np.float32)

        # Build tensors  (B=1)
        inp = np.stack([osz_as_float, depth_bev / 70.0], axis=0)
        inp_t  = torch.tensor(inp[None], dtype=torch.float32, device=device)
        gt_t   = torch.tensor(osz_as_float[None, None], dtype=torch.float32, device=device)
        d_t    = torch.tensor(depth_bev[None, None], dtype=torch.float32, device=device)

        pred_logit = model(inp_t)
        losses = loss_fn(pred_logit, gt_t, depth_bev=d_t)

        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        for k, v in losses.items():
            epoch_loss[k] = epoch_loss.get(k, 0.0) + v.item()

        print(f"  [frame {frame_idx+1}] "
              f"total={losses['total'].item():.4f}  "
              f"focal={losses['focal'].item():.4f}  "
              f"lap={losses['laplacian'].item():.4f}  "
              f"depth={losses['depth_constraint'].item():.4f}")

    n = max(frame_idx + 1, 1)
    return {k: v / n for k, v in epoch_loss.items()}


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
    parser.add_argument('--train_cnn', action='store_true',
                        help='Run one self-supervised CNN training epoch')
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

    # CRFBoundaryRefiner removed: the geometric OSZ boundary is already
    # physically correct (determined by exact ray-casting geometry), and
    # applying a Gaussian-blur CRF over it would BLUR the sharp occluder
    # edge into a soft uncertain region — the wrong thing to do here.
    # The CRF in BEVNeXt operates in IMAGE SPACE on DEPTH ESTIMATION
    # (color-similarity -> depth consistency); that is a fundamentally
    # different task from OSZ boundary post-processing.
    # If soft boundary refinement is later needed, use the learned
    # OSZRefineCNN + HierarchicalOSZLoss in crf_refine.py (optional Stage 5).

    loader = NuScenesOSZLoader(
        dataroot    = args.dataroot,
        version     = args.version,
        max_samples = args.max_samples,
    )
    if args.mock:
        loader._use_mock = True

    # Optional CNN
    device = 'cpu'
    cnn_model, optimizer, loss_fn = None, None, None
    if args.train_cnn:
        if not _CRF_AVAILABLE:
            print("  [WARN] crf_refine module not importable (torch/scipy missing?); "
                  "skipping CNN training.")
        else:
            try:
                import torch
                cnn_model = OSZRefineCNN(in_channels=2).to(device)
                optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
                loss_fn   = HierarchicalOSZLoss().to(device)
                print(f"\n  CNN params: {sum(p.numel() for p in cnn_model.parameters()):,}")
            except ImportError:
                print("  [WARN] torch not available; skipping CNN training.")

    # ── Per-frame processing ──────────────────────────────────────────────
    all_frames = list(loader)   # collect for optional CNN training

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

        # BEV depth map for CRF
        depth_bev = aggregate_depth_bev(cameras, caster)

        # Stage 4b: passthrough (CRF removed, see comment above)
        # osz_refined is kept as float32 for downstream compatibility
        # (plot functions and .npy saves expect float, not bool).
        osz_refined = osz_raw.astype(np.float32)

        # Stage 4c: semantic filter — intersect geometric OSZ with
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
        fig1 = plot_bev_osz(
            per_cam_masks = per_cam_masks,
            osz_mask      = osz_raw,
            bev_range     = tuple(args.bev_range),
            refined_mask  = osz_refined,
            depth_bev     = depth_bev,
            bev_occ       = bev_occ,
            title         = f"OSZ — {token}",
            save_path     = str(outdir / f"frame_{frame_idx:04d}_osz.png"),
        )
        plt.close(fig1)

        fig2 = plot_refinement_comparison(
            raw_mask    = osz_raw,
            refined_soft= osz_refined,
            depth_bev   = depth_bev,
            bev_occ     = bev_occ,
            bev_range   = tuple(args.bev_range),
            save_path   = str(outdir / f"frame_{frame_idx:04d}_refine.png"),
        )
        plt.close(fig2)

        # ── Camera vs BEV comparison (with image & frame token) ────────────
        fig3 = plot_camera_osz_comparison(
            images       = cam_images,
            depth_maps   = cam_depths,
            per_cam_masks= per_cam_masks,
            osz_mask     = osz_raw,
            refined_mask = osz_refined,
            depth_bev    = depth_bev,
            bev_occ      = bev_occ,
            bev_range    = tuple(args.bev_range),
            sample_token = token,
            save_path    = str(outdir / f"frame_{frame_idx:04d}_comparison.png"),
        )
        plt.close(fig3)

        # PA-relevant OSZ visualization: the signal PA actually trains on
        if drivable_mask is not None:
            fig4 = plot_pa_osz(
                osz_raw      = osz_raw,
                osz_pa       = osz_pa,
                drivable_mask= drivable_mask,
                bev_occ      = bev_occ,
                bev_range    = tuple(args.bev_range),
                sample_token = token,
                save_path    = str(outdir / f"frame_{frame_idx:04d}_pa_osz.png"),
            )
            plt.close(fig4)

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
                    save_path    = str(outdir / f"frame_{frame_idx:04d}_gt_osz.png"),
                )
                plt.close(fig5)
            except Exception as e:
                print(f"  [WARN] GT viz failed: {e}")

        # Save numpy arrays for downstream PA framework integration
        np.save(outdir / f"frame_{frame_idx:04d}_osz_raw.npy",     osz_raw)
        np.save(outdir / f"frame_{frame_idx:04d}_osz_refined.npy", osz_refined)
        np.save(outdir / f"frame_{frame_idx:04d}_depth_bev.npy",   depth_bev)
        np.save(outdir / f"frame_{frame_idx:04d}_osz_pa.npy",      osz_pa)
        if drivable_mask is not None:
            np.save(outdir / f"frame_{frame_idx:04d}_drivable.npy", drivable_mask)

    # ── Optional CNN training epoch ───────────────────────────────────────
    if args.train_cnn and cnn_model is not None:
        print("\n" + "=" * 60)
        print("  Running self-supervised CNN training epoch...")
        avg_losses = train_cnn_one_epoch(
            cnn_model, optimizer, loss_fn,
            all_frames, caster, device
        )
        print("\n  Epoch avg losses: "
              + "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))

        # Save model weights
        import torch
        torch.save(cnn_model.state_dict(), outdir / 'osz_refine_cnn.pth')
        print(f"  Model saved → {outdir / 'osz_refine_cnn.pth'}")

    print(f"\n✓ Done. Outputs in: {outdir}/")
    return outdir


if __name__ == '__main__':
    main()
