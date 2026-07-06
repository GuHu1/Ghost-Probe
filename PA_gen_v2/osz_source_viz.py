"""
osz_source_viz.py
------------------
Single-sample OSZ sanity check for the ACTIVE pipeline (osz_source.py ->
OSZ/modules/ray_casting.py). This replaces the old
PA_gen_v2/osz_geometry.py's __main__ block, which visualized the legacy
independent implementation instead.

This is step 2 of run_pipeline.py: "look at one frame before mining the
whole dataset." Three things get checked visually:

  1. Basic plumbing sanity (occupancy | raw OSZ) — the same kind of check
     the old osz_geometry.py did, just sourced from OSZ/ instead.

  2. Drivable-area filtering (raw OSZ | drivable area | PA-relevant OSZ):
     raw geometric OSZ counts the shadow of buildings as occluded even
     though no vehicle could ever be there — in dense urban scenes this
     routinely covers 70-80%+ of the grid, which is EXPECTED (see
     OSZ/modules/drivable_filter.py and osz_source.py's docstrings), not
     a sign anything is broken. PA-relevant OSZ = raw OSZ ∩ drivable area
     is what ghost_vehicle_miner.py actually uses for occlusion
     decisions. This panel set lets you see the before/after on one
     frame instead of just trusting the percentage.

  3. OSZ-vs-GT agreement: nuScenes ground-truth vehicle boxes for this
     sample are overlaid, color-coded by whether PA-relevant OSZ marks
     their center as occluded. Red box = GT says a vehicle is there and
     it sits in a plausible-but-occluded (drivable, shadowed) spot —
     a genuine phantom-vehicle candidate. Green box = GT vehicle sits
     somewhere PA-relevant OSZ correctly leaves clear.

     Note: OSZ/run_osz_pipeline.py's plot_gt_osz (in
     OSZ/visualize/bev_viz.py) does a related but fuller check on the
     same PA-relevant OSZ concept, with more decoration (grid lines,
     phantom-candidate legend, etc). Use that when you want the full
     picture; use this script when you just want a fast single-sample
     check from PA_gen_v2/'s own pipeline.

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
from OSZ.visualize.bev_viz import get_gt_boxes_ego, _box_corners_ego


def _bev_extent(caster):
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
    parser.add_argument('--outdir', default=str(_REPO_ROOT / 'PA_gen_v2' / 'output'))
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    sample_token = nusc.sample[args.sample_idx]['token']
    print(f"Processing sample {args.sample_idx}: {sample_token}")

    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    bev_occ, osz_raw, osz_pa, drivable_mask = \
        osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)

    raw_pct = (osz_raw > 0.5).mean() * 100
    pa_pct  = osz_pa.mean() * 100
    drivable_pct = drivable_mask.mean() * 100
    print(f"Raw OSZ coverage        : {raw_pct:.1f}% of BEV grid")
    print(f"Drivable area coverage  : {drivable_pct:.1f}% of BEV grid")
    print(f"PA-relevant OSZ coverage: {pa_pct:.1f}% of BEV grid  "
          f"(= raw OSZ ∩ drivable area)")
    if not osz_source.drivable_filter_available():
        print("[WARN] drivable-area filtering is UNAVAILABLE in this "
              "environment (shapely not importable) — the numbers above "
              "for 'drivable area' and 'PA-relevant OSZ' are both "
              "unfiltered raw OSZ. Install shapely to get real filtering.")

    gt_boxes = get_gt_boxes_ego(nusc, sample_token, caster.bev_range)
    for box in gt_boxes:
        i, j = osz_source.bev_xy_to_ij(box['cx'], box['cy'])
        nx, ny = osz_pa.shape
        box['in_osz'] = bool(0 <= i < nx and 0 <= j < ny and osz_pa[i, j])

    vehicle_boxes = [b for b in gt_boxes if b['category'] in VEHICLE_CATS]
    n_phantom = sum(1 for b in vehicle_boxes if b['in_osz'])
    print(f"GT vehicles in this sample : {len(vehicle_boxes)}")
    print(f"  -> inside PA-relevant OSZ (phantom candidates) : {n_phantom}")
    print(f"  -> outside PA-relevant OSZ (correctly clear)    : "
          f"{len(vehicle_boxes) - n_phantom}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    fig.suptitle(f'Sample {args.sample_idx} | {sample_token[:16]}... '
                 f'(via osz_source.py -> OSZ/modules/ray_casting.py '
                 f'+ drivable_filter.py)', fontsize=11)
    ax = axes.ravel()

    ax[0].imshow(bev_occ, cmap='gray', origin='lower', extent=extent)
    ax[0].set_xlim(*xlim); ax[0].set_ylim(*ylim)
    ax[0].set_title('BEV Occupancy (OSZ/ voxel-cast)')
    ax[0].plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

    ax[1].imshow(osz_raw, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=1)
    ax[1].set_xlim(*xlim); ax[1].set_ylim(*ylim)
    ax[1].set_title(f'Raw OSZ (2D ray cast) — {raw_pct:.0f}% of grid')
    ax[1].plot(0, 0, 'b+', markersize=12, markeredgewidth=2)

    ax[2].imshow(drivable_mask.astype(float), cmap='Greens', origin='lower',
                 extent=extent, vmin=0, vmax=1)
    ax[2].set_xlim(*xlim); ax[2].set_ylim(*ylim)
    ax[2].set_title(f'Drivable area (nuScenes map) — {drivable_pct:.0f}% of grid')
    ax[2].plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

    ax[3].imshow(osz_pa.astype(float), cmap='hot', origin='lower',
                 extent=extent, vmin=0, vmax=1)
    ax[3].set_xlim(*xlim); ax[3].set_ylim(*ylim)
    ax[3].set_title(f'PA-relevant OSZ (raw ∩ drivable) — {pa_pct:.0f}% of grid')
    ax[3].plot(0, 0, 'b+', markersize=12, markeredgewidth=2)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ]   = [0.8, 0.8, 0.8]
    overlay[osz_pa]    = [0.9, 0.2, 0.2]
    ax[4].imshow(overlay, origin='lower', extent=extent)
    ax[4].set_xlim(*xlim); ax[4].set_ylim(*ylim)
    ax[4].set_title('Overlay (gray=occ, red=PA-relevant OSZ)')
    ax[4].plot(0, 0, 'g+', markersize=12, markeredgewidth=2)
    ax[4].legend(handles=[
        mpatches.Patch(color=[0.8, 0.8, 0.8], label='Occupied'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='PA-relevant OSZ'),
    ], loc='upper right', fontsize=7)

    ax[5].imshow(overlay, origin='lower', extent=extent, alpha=0.6)
    ax[5].set_xlim(*xlim); ax[5].set_ylim(*ylim)
    for box in vehicle_boxes:
        corners = _box_corners_ego(box['cx'], box['cy'], box['length'],
                                    box['width'], box['yaw'])
        poly_x = list(corners[:, 1]) + [corners[0, 1]]
        poly_y = list(corners[:, 0]) + [corners[0, 0]]
        color = '#E74C3C' if box['in_osz'] else '#2ECC71'
        ax[5].plot(poly_x, poly_y, color=color, linewidth=1.8, alpha=0.9)
    ax[5].plot(0, 0, 'w+', markersize=12, markeredgewidth=2)
    ax[5].set_title(f'GT vehicles vs PA-relevant OSZ  '
                     f'({n_phantom} inside / {len(vehicle_boxes)} total)')
    ax[5].legend(handles=[
        plt.Line2D([0],[0], color='#2ECC71', lw=2, label='Vehicle (OSZ correctly clear)'),
        plt.Line2D([0],[0], color='#E74C3C', lw=2, label='Vehicle inside OSZ'),
    ], loc='upper right', fontsize=6)

    for a in ax:
        a.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=7)
        a.set_ylabel('x (m) ↑ forward', fontsize=7)
        a.tick_params(labelsize=6)

    plt.tight_layout()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'osz_sample_viz.png'
    plt.savefig(out_path, dpi=120)
    print(f"Saved visualization to {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
