"""
visualize_events.py
-------------------
Visualize mined ghost vehicle events — the most important debugging step.

For each event we draw:
  - BEV occupancy grid of frame t (the emergence frame)
  - OSZ mask of frame t
  - The emerged vehicle position (green star)
  - Its lookback positions (colored dots: red=was in OSZ, blue=was visible)
  - Trajectory arrow showing direction of emergence

Karpathy rule: if you cannot look at the output and immediately see that
it makes geometric sense, the mining logic is wrong.  Do NOT proceed to
model training without this visual check.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

import pyquaternion
from nuscenes.nuscenes import NuScenes

from osz_geometry import (
    get_osz_for_sample,
    bev_coords_to_pixel,
    BEV_SIZE, BEV_CENTER, BEV_RANGE_M,
)
from ghost_vehicle_miner import (
    _get_ego_pose,
    _global_to_ego,
    load_events,
)


def _get_vehicle_global_pos(nusc: NuScenes,
                             instance_token: str,
                             sample_token: str) -> np.ndarray:
    """
    Look up the global position of an instance in a specific sample.
    Returns None if the instance has no annotation in that sample.
    """
    sample = nusc.get('sample', sample_token)
    # Iterate sample_annotations for this sample
    ann_tok = sample['anns']
    for at in ann_tok:
        ann = nusc.get('sample_annotation', at)
        if ann['instance_token'] == instance_token:
            return np.array(ann['translation'], dtype=np.float32)
    return None


def visualize_event(nusc: NuScenes,
                    event: Dict,
                    ax: plt.Axes,
                    title: str = '') -> None:
    """
    Draw one ghost vehicle event onto ax.

    Color legend:
      gray        = BEV occupied cells (LiDAR)
      red (heat)  = OSZ mask at emergence frame
      green star  = vehicle position at frame t (emerged)
      red dots    = vehicle position at lookback frames where it was in OSZ
      blue dots   = vehicle position at lookback frames where it was visible
      white ×     = vehicle not annotated in that lookback frame
    """
    emerge_tok  = event['emerge_sample']
    instance_tok = event['instance_token']
    lb_tokens   = event['lookback_tokens']
    was_in_osz  = event['was_in_osz']

    # --- Compute OSZ of emergence frame ---
    occ_grid, osz_mask = get_osz_for_sample(nusc, emerge_tok)

    # --- Build overlay image ---
    overlay = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    overlay[occ_grid]       = [0.55, 0.55, 0.55]   # gray = occupied
    overlay[osz_mask > 0.5] = [0.80, 0.15, 0.15]   # red  = OSZ

    ax.imshow(overlay, origin='lower')

    # --- Ego marker ---
    ax.plot(BEV_CENTER, BEV_CENTER, 'w^', markersize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # --- Emerged vehicle position (frame t) ---
    ex, ey = event['emerge_bev_xy']
    ec, er = bev_coords_to_pixel(ex, ey)
    ax.plot(ec, er, 'g*', markersize=14,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
            label='Emerged (t)')

    # --- Ego pose at frame t (for coordinate transforms of lookback) ---
    ego_t, ego_q_t = _get_ego_pose(nusc, emerge_tok)

    # --- Lookback positions ---
    # We'll draw a trajectory line through known positions
    traj_cols, traj_rows = [], []

    for i, (lb_tok, in_osz) in enumerate(zip(lb_tokens, was_in_osz)):
        lb_global = _get_vehicle_global_pos(nusc, instance_tok, lb_tok)

        if lb_global is None:
            # Not annotated — mark with small white x at approximate position
            # (can't know where it was, skip trajectory)
            continue

        # Transform lookback global position into frame-t ego coords
        # so everything is in the same BEV reference frame for display
        ego_lb, ego_q_lb = _get_ego_pose(nusc, lb_tok)
        pt_ego_lb = _global_to_ego(lb_global, ego_lb, ego_q_lb)

        # Then convert to frame-t ego (two-step: lb_ego → global → t_ego)
        # Faster: use stored global position directly
        pt_ego_in_t = _global_to_ego(lb_global, ego_t, ego_q_t)

        col_lb, row_lb = bev_coords_to_pixel(pt_ego_in_t[0], pt_ego_in_t[1])

        if 0 <= col_lb < BEV_SIZE and 0 <= row_lb < BEV_SIZE:
            traj_cols.append(col_lb)
            traj_rows.append(row_lb)

            color = 'red' if in_osz else 'deepskyblue'
            ax.plot(col_lb, row_lb, 'o', color=color, markersize=6,
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground='black')])
            # Frame index label
            ax.text(col_lb + 3, row_lb + 3,
                    f't-{len(lb_tokens)-i}',
                    color='white', fontsize=5,
                    path_effects=[pe.withStroke(linewidth=1,
                                                foreground='black')])

    # Draw trajectory line if we have ≥2 points
    if len(traj_cols) >= 2:
        # Add emergence point at the end
        traj_cols.append(ec)
        traj_rows.append(er)
        ax.plot(traj_cols, traj_rows, '-', color='yellow',
                linewidth=1.2, alpha=0.7)

    # --- Labels & legend ---
    n_osz = sum(was_in_osz)
    if not title:
        title = f"Ghost event | {n_osz}/{len(lb_tokens)} lb frames in OSZ"
    ax.set_title(title, fontsize=7)
    ax.set_xlim(0, BEV_SIZE)
    ax.set_ylim(0, BEV_SIZE)
    ax.set_xlabel('col (ego-x →)', fontsize=6)
    ax.set_ylabel('row (ego-y →)', fontsize=6)
    ax.tick_params(labelsize=5)

    legend_patches = [
        mpatches.Patch(color=[0.55]*3, label='Occupied'),
        mpatches.Patch(color=[0.8, 0.15, 0.15], label='OSZ'),
        plt.Line2D([0],[0], marker='*', color='g', markersize=8,
                   linestyle='none', label='Emerged (t)'),
        plt.Line2D([0],[0], marker='o', color='red', markersize=5,
                   linestyle='none', label='Lookback (in OSZ)'),
        plt.Line2D([0],[0], marker='o', color='deepskyblue', markersize=5,
                   linestyle='none', label='Lookback (visible)'),
    ]
    ax.legend(handles=legend_patches, fontsize=5, loc='upper right')


def make_event_grid(nusc: NuScenes,
                    events: List[Dict],
                    max_events: int = 12,
                    label_filter: int = 1,
                    out_path: str = '/home/claude/phantom_agent/events_grid.png'
                    ) -> None:
    """
    Render a grid of ghost vehicle events and save to disk.

    Args:
        label_filter: 1 = show positives, 0 = show negatives, -1 = show both
    """
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
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    label_str = 'POSITIVE (ghost)' if label_filter == 1 else 'NEGATIVE (visible)'
    fig.suptitle(f'{label_str} events — nuScenes mini', fontsize=10)

    for i, (ax, event) in enumerate(zip(axes, subset)):
        try:
            visualize_event(nusc, event, ax,
                            title=f"Event {i} | label={event['label']}")
        except Exception as ex:
            ax.set_title(f"Event {i} FAILED: {ex}", fontsize=6)
            ax.axis('off')

    # Hide unused axes
    for ax in axes[len(subset):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Saved event grid ({len(subset)} events) → {out_path}")
    plt.close()


def print_event_stats(events: List[Dict]) -> None:
    """Print a human-readable summary of the mined event set."""
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
        print(f"\nPositive events — OSZ frame counts:")
        print(f"  mean : {np.mean(osz_counts):.2f}")
        print(f"  min  : {np.min(osz_counts)}")
        print(f"  max  : {np.max(osz_counts)}")

        # Distance of emergence from ego
        dists = [np.sqrt(e['emerge_bev_xy'][0]**2 +
                         e['emerge_bev_xy'][1]**2)
                 for e in positives]
        print(f"\nEmergence distance from ego (m):")
        print(f"  mean : {np.mean(dists):.1f}")
        print(f"  min  : {np.min(dists):.1f}")
        print(f"  max  : {np.max(dists):.1f}")

    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Main: load events JSON → visualize
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',   default='/data/nuscenes')
    parser.add_argument('--version',    default='v1.0-mini')
    parser.add_argument('--events',     default='/home/claude/phantom_agent/'
                                                 'ghost_events_mini.json')
    parser.add_argument('--max_events', type=int, default=12)
    parser.add_argument('--out_pos',    default='/home/claude/phantom_agent/'
                                                 'events_positive.png')
    parser.add_argument('--out_neg',    default='/home/claude/phantom_agent/'
                                                 'events_negative.png')
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f"Loading events from {args.events} ...")
    events = load_events(args.events)

    print_event_stats(events)

    print("Rendering positive events ...")
    make_event_grid(nusc, events,
                    max_events=args.max_events,
                    label_filter=1,
                    out_path=args.out_pos)

    print("Rendering negative events (sanity check) ...")
    make_event_grid(nusc, events,
                    max_events=args.max_events,
                    label_filter=0,
                    out_path=args.out_neg)

    print("\nDone. Open the PNGs and verify:")
    print("  POSITIVE: green star should appear at OSZ boundary,")
    print("            red dots should be inside the red OSZ region.")
    print("  NEGATIVE: all dots should be in the gray/clear area.")
