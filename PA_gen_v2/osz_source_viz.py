"""
osz_source_viz.py
------------------
Single-sample OSZ sanity check for the ACTIVE pipeline (osz_source.py ->
OSZ/modules/ray_casting.py). This replaces the old
filter/osz_geometry.py's __main__ block, which visualized the legacy
independent implementation instead.

This is step 2 of run_pipeline.py: "look at one frame before mining the
whole dataset." Two things get checked visually:

  1. Basic plumbing sanity (3-panel: occupancy | OSZ | overlay) — the
     same kind of check the old osz_geometry.py did, just sourced from
     OSZ/ instead.
  2. OSZ-vs-GT agreement: nuScenes ground-truth vehicle boxes for this
     sample are overlaid, color-coded by whether OSZ marks their center
     as occluded. This directly answers "does OSZ line up with GT" for
     one frame at a glance — red box = GT says a vehicle is there, but
     OSZ says that spot is a shadow (worth a closer look: either OSZ is
     over-covering, or the box is genuinely in a blind spot); green box
     = GT vehicle sits in a region OSZ correctly leaves clear.

     Note: OSZ/run_osz_pipeline.py's plot_gt_osz (in
     OSZ/visualize/bev_viz.py) does a related but fuller check — it also
     overlays the drivable-area filter and the PA-relevant OSZ. Use that
     when you want the full picture with a real nuScenes map; use this
     script when you just want a fast, map-independent look at raw
     OSZ-vs-GT agreement for one sample.

Usage:
  python osz_source_viz.py --dataroot /data/sets/nuscenes --version v1.0-mini --sample_idx 5
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
# Reuse OSZ/'s own tested GT-box helpers rather than re-deriving the
# ego-frame box math a second time in filter/.
from OSZ.visualize.bev_viz import get_gt_boxes_ego, _box_corners_ego


def _bev_extent(caster):
    """Same convention as OSZ/visualize/bev_viz.py's _bev_extent."""
    x_min, x_max, y_min, y_max = caster.bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


VEHICLE_CATS = {
    'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
    'vehicle.trailer', 'vehicle.construction',
    'vehicle.emergency.ambulance', 'vehicle.emergency.police',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which sample index to visualize (0-based)')
    parser.add_argument('--outdir', default=str(_REPO_ROOT / 'filter' / 'output'))
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    sample_token = nusc.sample[args.sample_idx]['token']
    print(f"Processing sample {args.sample_idx}: {sample_token}")

    bev_occ, osz_mask = osz_source.get_osz_for_sample(nusc, sample_token)
    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    osz_fraction = osz_mask.mean()
    print(f"OSZ coverage: {osz_fraction*100:.1f}% of BEV grid")

    # ── Ground-truth boxes + occlusion verdict ──────────────────────────
    gt_boxes = get_gt_boxes_ego(nusc, sample_token, caster.bev_range)
    for box in gt_boxes:
        i, j = osz_source.bev_xy_to_ij(box['cx'], box['cy'])
        nx, ny = osz_mask.shape
        box['in_osz'] = bool(0 <= i < nx and 0 <= j < ny and osz_mask[i, j] > 0.5)

    vehicle_boxes = [b for b in gt_boxes if b['category'] in VEHICLE_CATS]
    n_phantom = sum(1 for b in vehicle_boxes if b['in_osz'])
    print(f"GT vehicles in this sample : {len(vehicle_boxes)}")
    print(f"  -> inside OSZ (phantom candidates) : {n_phantom}")
    print(f"  -> outside OSZ (correctly clear)    : {len(vehicle_boxes) - n_phantom}")

    # ── 3-panel basic sanity + 1-panel GT overlay ───────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Sample {args.sample_idx} | {sample_token[:16]}... '
                 f'(via osz_source.py -> OSZ/modules/ray_casting.py)', fontsize=11)

    axes[0].imshow(bev_occ, cmap='gray', origin='lower', extent=extent)
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*ylim)
    axes[0].set_title('BEV Occupancy (OSZ/ voxel-cast)')
    axes[0].plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

    axes[1].imshow(osz_mask, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=1)
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*ylim)
    axes[1].set_title('OSZ Mask (2D ray cast)')
    axes[1].plot(0, 0, 'b+', markersize=12, markeredgewidth=2)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ]        = [0.8, 0.8, 0.8]
    overlay[osz_mask > 0.5] = [0.9, 0.2, 0.2]
    axes[2].imshow(overlay, origin='lower', extent=extent)
    axes[2].set_xlim(*xlim); axes[2].set_ylim(*ylim)
    axes[2].set_title('Overlay (gray=occ, red=OSZ)')
    axes[2].plot(0, 0, 'g+', markersize=12, markeredgewidth=2)
    axes[2].legend(handles=[
        mpatches.Patch(color=[0.8,0.8,0.8], label='Occupied'),
        mpatches.Patch(color=[0.9,0.2,0.2], label='OSZ'),
    ], loc='upper right', fontsize=7)

    axes[3].imshow(overlay, origin='lower', extent=extent, alpha=0.6)
    axes[3].set_xlim(*xlim); axes[3].set_ylim(*ylim)
    for box in vehicle_boxes:
        corners = _box_corners_ego(box['cx'], box['cy'], box['length'],
                                    box['width'], box['yaw'])
        poly_x = list(corners[:, 1]) + [corners[0, 1]]   # ego-y -> horizontal
        poly_y = list(corners[:, 0]) + [corners[0, 0]]   # ego-x -> vertical
        color = '#E74C3C' if box['in_osz'] else '#2ECC71'
        axes[3].plot(poly_x, poly_y, color=color, linewidth=1.8, alpha=0.9)
    axes[3].plot(0, 0, 'w+', markersize=12, markeredgewidth=2)
    axes[3].set_title(f'GT vehicles vs OSZ  '
                       f'({n_phantom} inside / {len(vehicle_boxes)} total)')
    axes[3].legend(handles=[
        plt.Line2D([0],[0], color='#2ECC71', lw=2, label='Vehicle (OSZ correctly clear)'),
        plt.Line2D([0],[0], color='#E74C3C', lw=2, label='Vehicle inside OSZ'),
    ], loc='upper right', fontsize=6)

    for ax in axes:
        ax.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=7)
        ax.set_ylabel('x (m) ↑ forward', fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'osz_sample_viz.png'
    plt.savefig(out_path, dpi=120)
    print(f"Saved visualization to {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
