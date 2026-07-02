"""
visualize_events.py
-------------------
Visualize mined ghost vehicle events — the most important debugging step.

For each event we draw:
  - BEV occupancy grid of frame t (the emergence frame)
  - OSZ mask of frame t
  - The emerged vehicle position (green star)
  - Its lookback positions: red = confirmed in OSZ, blue = confirmed
    visible, gray x = no evidence either way (see filter/trajectory.py)
  - Trajectory line through the confirmed/interpolated positions

Karpathy rule: if you cannot look at the output and immediately see that
it makes geometric sense, the mining logic is wrong. Do NOT proceed to
model training without this visual check.

Changes vs. the original version
---------------------------------
1. OSZ now comes from osz_source.py (OSZ/modules/ray_casting.py), not the
   old filter/osz_geometry.py.
2. Plotting is done entirely in METRIC ego coordinates (imshow with
   extent=, matplotlib axis limits in metres) instead of manually
   computed pixel row/col indices. This removes a whole class of
   pixel-rounding / axis-order bugs — see osz_source.py's module
   docstring for why OSZ's (i,j) convention and the old filter/'s
   (row,col) convention are transposed relative to each other.
3. --dataroot is required; --out_pos/--out_neg/--events default inside
   this repo (filter/output/) instead of a hardcoded /home/claude/... path.
4. was_in_osz entries can now be True / False / None (see
   ghost_vehicle_miner.py) — None is drawn as a gray '?' marker instead
   of being coerced into a color that implies false certainty.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from tqdm import tqdm

import pyquaternion
from nuscenes.nuscenes import NuScenes

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
from ghost_vehicle_miner import _get_ego_pose, _global_to_ego, load_events


def _bev_extent(caster):
    """
    Same convention as OSZ/visualize/bev_viz.py's _bev_extent: forward=UP,
    ego-left=LEFT, no transpose applied to the (nx,ny) array. Kept local
    here (not imported) to avoid a hard dependency of filter/ on OSZ's
    plotting internals — this is 6 lines of pure arithmetic, cheap to
    duplicate, easy to eyeball-verify against the OSZ/ original if ever
    in doubt.
    """
    x_min, x_max, y_min, y_max = caster.bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


def visualize_event(nusc: NuScenes, event: Dict, ax: plt.Axes,
                    title: str = '') -> None:
    """Draw one ghost vehicle event onto ax, in metric ego coordinates."""
    emerge_tok   = event['emerge_sample']
    instance_tok = event['instance_token']
    lb_tokens    = event['lookback_tokens']
    was_in_osz   = event['was_in_osz']   # list of True / False / None

    bev_occ, osz_mask = osz_source.get_osz_for_sample(nusc, emerge_tok)
    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    # bev_occ / osz_mask are (nx, ny) with axis-0=ego-x, axis-1=ego-y —
    # imshow with this extent and NO transpose places axis-0 on the
    # vertical (forward=up) axis, matching OSZ/visualize/bev_viz.py.
    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ]       = [0.55, 0.55, 0.55]
    overlay[osz_mask > 0.5] = [0.80, 0.15, 0.15]
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Ego marker at (0,0) in metric coords — no pixel math needed.
    ax.plot(0, 0, 'w^', markersize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Emerged vehicle position (frame t), already stored in metric ego xy
    ex, ey = event['emerge_bev_xy']
    ax.plot(ey, ex, 'g*', markersize=14,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
            label='Emerged (t)')
    # NOTE: plotted as (ey, ex) not (ex, ey) — matplotlib's x-axis here is
    # ego-y (horizontal) and y-axis is ego-x (forward), matching the
    # imshow extent above. This is the same axis mapping OSZ/'s _draw_ego
    # uses; getting this backwards is exactly the silent-transpose risk
    # documented in osz_source.py.

    ego_t, ego_q_t = _get_ego_pose(nusc, emerge_tok)

    traj_x, traj_y = [], []  # metric (ego-y, ego-x) for the trajectory line

    for i, (lb_tok, in_osz) in enumerate(zip(lb_tokens, was_in_osz)):
        lb_global = _get_vehicle_global_pos(nusc, instance_tok, lb_tok)
        if lb_global is None:
            continue

        ego_lb, ego_q_lb = _get_ego_pose(nusc, lb_tok)
        # Transform into frame-t's ego frame so everything is drawn in one
        # consistent reference frame (frame t may not be lb_tok's own frame)
        pt_ego_in_t = _global_to_ego(lb_global, ego_t, ego_q_t)

        if not osz_source.in_bev_range(pt_ego_in_t[0], pt_ego_in_t[1]):
            continue

        px, py = pt_ego_in_t[1], pt_ego_in_t[0]   # (ego-y, ego-x) for plotting
        traj_x.append(px)
        traj_y.append(py)

        if in_osz is True:
            color, marker = 'red', 'o'
        elif in_osz is False:
            color, marker = 'deepskyblue', 'o'
        else:
            color, marker = '#888888', 'x'   # no evidence either way

        ax.plot(px, py, marker, color=color, markersize=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        ax.text(px + 0.8, py + 0.8, f't-{len(lb_tokens)-i}',
                color='white', fontsize=5,
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    if len(traj_x) >= 2:
        traj_x.append(ey)
        traj_y.append(ex)
        ax.plot(traj_x, traj_y, '-', color='yellow', linewidth=1.2, alpha=0.7)

    n_osz  = sum(1 for v in was_in_osz if v is True)
    n_unk  = sum(1 for v in was_in_osz if v is None)
    if not title:
        title = (f"Ghost event | {n_osz}/{len(lb_tokens)} lb frames in OSZ"
                 + (f" ({n_unk} unknown)" if n_unk else ""))
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=6)
    ax.set_ylabel('x (m) ↑ forward', fontsize=6)
    ax.tick_params(labelsize=5)

    legend_patches = [
        mpatches.Patch(color=[0.55]*3, label='Occupied'),
        mpatches.Patch(color=[0.8, 0.15, 0.15], label='OSZ'),
        plt.Line2D([0],[0], marker='*', color='g', markersize=8,
                   linestyle='none', label='Emerged (t)'),
        plt.Line2D([0],[0], marker='o', color='red', markersize=5,
                   linestyle='none', label='Lookback (confirmed in OSZ)'),
        plt.Line2D([0],[0], marker='o', color='deepskyblue', markersize=5,
                   linestyle='none', label='Lookback (confirmed visible)'),
        plt.Line2D([0],[0], marker='x', color='#888888', markersize=5,
                   linestyle='none', label='Lookback (no evidence)'),
    ]
    ax.legend(handles=legend_patches, fontsize=5, loc='upper right')


def _get_vehicle_global_pos(nusc: NuScenes, instance_token: str,
                            sample_token: str):
    """Global position of an instance in a specific sample, or None."""
    sample = nusc.get('sample', sample_token)
    for at in sample['anns']:
        ann = nusc.get('sample_annotation', at)
        if ann['instance_token'] == instance_token:
            return np.array(ann['translation'], dtype=np.float32)
    return None


def make_event_grid(nusc: NuScenes, events: List[Dict], max_events: int = 12,
                    label_filter: int = 1, out_path: str = None) -> None:
    """Render a grid of ghost vehicle events and save to disk."""
    if label_filter >= 0:
        subset = [e for e in events if e['label'] == label_filter]
    else:
        subset = events
    subset = subset[:max_events]

    if not subset:
        print(f"No events with label={label_filter} found.")
        return

    ncols = 4
    nrows = (len(subset) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    label_str = 'POSITIVE (ghost)' if label_filter == 1 else 'NEGATIVE (visible)'
    fig.suptitle(f'{label_str} events', fontsize=10)

    for i, (ax, event) in enumerate(zip(axes, subset)):
        try:
            visualize_event(nusc, event, ax, title=f"Event {i} | label={event['label']}")
        except Exception as ex:
            ax.set_title(f"Event {i} FAILED: {ex}", fontsize=6)
            ax.axis('off')

    for ax in axes[len(subset):]:
        ax.axis('off')

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Saved event grid ({len(subset)} events) → {out_path}")
    plt.close()


def print_event_stats(events: List[Dict]) -> None:
    positives = [e for e in events if e['label'] == 1]
    negatives = [e for e in events if e['label'] == 0]

    print(f"\n{'─'*50}")
    print(f"Event set statistics")
    print(f"{'─'*50}")
    print(f"Total events   : {len(events)}")
    print(f"Positives      : {len(positives)}")
    print(f"Negatives      : {len(negatives)}")

    if positives:
        osz_counts = [e['n_osz_frames'] for e in positives]
        print(f"\nPositive events — confirmed-OSZ frame counts:")
        print(f"  mean : {np.mean(osz_counts):.2f}")
        print(f"  min  : {np.min(osz_counts)}")
        print(f"  max  : {np.max(osz_counts)}")

        dists = [np.sqrt(e['emerge_bev_xy'][0]**2 + e['emerge_bev_xy'][1]**2)
                 for e in positives]
        print(f"\nEmergence distance from ego (m):")
        print(f"  mean : {np.mean(dists):.1f}")
        print(f"  min  : {np.min(dists):.1f}")
        print(f"  max  : {np.max(dists):.1f}")

    print(f"{'─'*50}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--events',     default=str(_REPO_ROOT / 'filter' /
                                                     'output' / 'ghost_events_mini.json'))
    parser.add_argument('--max_events', type=int, default=12)
    parser.add_argument('--out_pos',    default=str(_REPO_ROOT / 'filter' /
                                                     'output' / 'events_positive.png'))
    parser.add_argument('--out_neg',    default=str(_REPO_ROOT / 'filter' /
                                                     'output' / 'events_negative.png'))
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f"Loading events from {args.events} ...")
    events = load_events(args.events)

    print_event_stats(events)

    print("Rendering positive events ...")
    make_event_grid(nusc, events, max_events=args.max_events,
                    label_filter=1, out_path=args.out_pos)

    print("Rendering negative events (sanity check) ...")
    make_event_grid(nusc, events, max_events=args.max_events,
                    label_filter=0, out_path=args.out_neg)

    print("\nDone. Open the PNGs and verify:")
    print("  POSITIVE: green star should appear at OSZ boundary,")
    print("            red dots should be inside the red OSZ region.")
    print("  NEGATIVE: all dots should be blue (confirmed visible),")
    print("            none should sit inside the red OSZ region.")
