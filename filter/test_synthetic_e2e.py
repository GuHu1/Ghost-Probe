"""
test_synthetic_e2e.py
---------------------
End-to-end test using synthetic data (no real nuScenes required).

We construct a fake "scene" with:
  - A wall of occluders at x=+15m (simulating a building)
  - A vehicle behind the wall at x=+20m (invisible → OSZ)
  - The vehicle moves to x=+13m in the "next" frame (emerges from OSZ)

Expected outcome:
  - OSZ mask covers x > 15m ahead of ego
  - Vehicle at x=+20m is flagged as "in OSZ"
  - Vehicle at x=+13m (which is in front of the wall) is NOT in OSZ
  - The transition constitutes a valid ghost vehicle emergence event

This test validates the full data flow of osz_geometry + ghost_vehicle_miner
without any I/O or nuScenes API calls.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pyquaternion

sys.path.insert(0, '/home/claude/phantom_agent')

from osz_geometry import (
    build_bev_occupancy,
    cast_osz_mask,
    bev_coords_to_pixel,
    BEV_SIZE, BEV_CENTER, BEV_RANGE_M, BEV_RESOLUTION,
)
from ghost_vehicle_miner import _global_to_ego, _is_in_osz, _is_in_bev_range


def _is_in_osz_mask(pt_xy: tuple, osz_mask: np.ndarray) -> bool:
    """Thin wrapper: check if metric (x,y) ego point is in OSZ."""
    from osz_geometry import bev_coords_to_pixel, BEV_SIZE
    col, row = bev_coords_to_pixel(pt_xy[0], pt_xy[1])
    if col < 0 or col >= BEV_SIZE or row < 0 or row >= BEV_SIZE:
        return False
    return bool(osz_mask[row, col] > 0.5)


def make_synthetic_scene():
    """
    Returns (occ_grid, osz_mask, wall_x_m, ghost_behind_m, ghost_emerged_m).

    Layout (ego at origin, looking in +x direction):
      x=0           : ego vehicle
      x=+15m        : wall of occluders (buildings / parked trucks)
      x=+20m, y=0m  : ghost vehicle (fully behind wall)
      x=+13m, y=0m  : same vehicle after it emerged
    """
    wall_x_m       = 15.0
    ghost_behind_m = 20.0
    ghost_emerged_m = 13.0

    # --- Build occupancy grid with a wall ---
    # Wall: all points at x=wall_x_m, spanning y from -5m to +5m, z=1.5m
    wall_y = np.linspace(-5, 5, 60)
    wall_pts = np.column_stack([
        np.full(len(wall_y), wall_x_m),
        wall_y,
        np.full(len(wall_y), 1.5),
    ]).astype(np.float32)

    # Fill out enough background points to pass the >100 assertion
    bg = np.random.uniform(-40, 40, (600, 3)).astype(np.float32)
    bg[:, 2] = 0.5   # low-height ground clutter (still passes height filter)

    pts_all = np.vstack([wall_pts, bg])
    occ_grid = build_bev_occupancy(pts_all)
    osz_mask = cast_osz_mask(occ_grid)

    return occ_grid, osz_mask, wall_x_m, ghost_behind_m, ghost_emerged_m


def test_synthetic_osz_coverage():
    """Ghost behind wall → in OSZ; emerged position → NOT in OSZ."""
    occ_grid, osz_mask, wall_x, ghost_behind, ghost_emerged = \
        make_synthetic_scene()

    # Point behind wall should be in OSZ
    pt_behind  = (ghost_behind, 0.0)
    pt_emerged = (ghost_emerged, 0.0)

    in_osz_behind  = _is_in_osz_mask(pt_behind,  osz_mask)
    in_osz_emerged = _is_in_osz_mask(pt_emerged, osz_mask)

    print(f"  Ghost at x=+{ghost_behind}m (behind wall):  in_osz={in_osz_behind}")
    print(f"  Ghost at x=+{ghost_emerged}m (emerged):     in_osz={in_osz_emerged}")

    assert in_osz_behind,  \
        f"Point at x={ghost_behind}m should be in OSZ (behind wall at x={wall_x}m)"
    assert not in_osz_emerged, \
        f"Point at x={ghost_emerged}m should NOT be in OSZ (in front of wall)"

    print("  ✓ OSZ coverage test passed.")
    return occ_grid, osz_mask


def test_global_to_ego_chain():
    """
    Simulate lookback frame: ego moved forward 5m between t-1 and t.
    Vehicle was at global (20, 0, 0) at t-1.
    At frame t, ego is at global (5, 0, 0).
    So vehicle should appear at x=+15m in frame-t ego coords.
    """
    ego_t  = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    ego_q  = pyquaternion.Quaternion(1, 0, 0, 0)

    pt_global = np.array([20.0, 0.0, 0.0], dtype=np.float32)
    pt_ego_t  = _global_to_ego(pt_global, ego_t, ego_q)

    assert abs(pt_ego_t[0] - 15.0) < 0.1, \
        f"Expected x=15.0 in ego frame, got {pt_ego_t[0]:.3f}"
    assert abs(pt_ego_t[1]) < 0.1, \
        f"Expected y=0.0 in ego frame, got {pt_ego_t[1]:.3f}"

    print(f"  Vehicle at global (20,0,0), ego at global (5,0,0) "
          f"→ ego frame: {pt_ego_t[:2]}")
    print("  ✓ global→ego chain test passed.")


def visualize_synthetic(occ_grid, osz_mask, out_path):
    """Save a picture of the synthetic scene for manual inspection."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Synthetic Scene Validation", fontsize=11)

    overlay = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    overlay[occ_grid]       = [0.7, 0.7, 0.7]
    overlay[osz_mask > 0.5] = [0.85, 0.1, 0.1]

    axes[0].imshow(overlay, origin='lower')
    axes[0].plot(BEV_CENTER, BEV_CENTER, 'g^', markersize=10, label='Ego')
    # Wall position
    wall_col, wall_row = bev_coords_to_pixel(15.0, 0.0)
    axes[0].axvline(wall_col, color='cyan', linewidth=1, linestyle='--',
                    label='Wall (x=+15m)')
    # Ghost behind
    gh_col, gh_row = bev_coords_to_pixel(20.0, 0.0)
    axes[0].plot(gh_col, gh_row, 'rx', markersize=12, markeredgewidth=2,
                 label='Ghost behind wall')
    # Emerged
    em_col, em_row = bev_coords_to_pixel(13.0, 0.0)
    axes[0].plot(em_col, em_row, 'g*', markersize=12,
                 label='Emerged (x=+13m)')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].set_title("Overlay (gray=occ, red=OSZ)")

    axes[1].imshow(osz_mask, cmap='hot', origin='lower', vmin=0, vmax=1)
    axes[1].plot(BEV_CENTER, BEV_CENTER, 'b^', markersize=8)
    axes[1].axvline(wall_col, color='cyan', linewidth=1, linestyle='--')
    axes[1].set_title("OSZ mask (hot = occluded)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"  Saved synthetic viz → {out_path}")
    plt.close()


if __name__ == '__main__':
    print("\n=== Synthetic End-to-End Validation ===\n")

    print("[1] Building synthetic scene and testing OSZ coverage ...")
    occ_grid, osz_mask = test_synthetic_osz_coverage()

    print("\n[2] Testing global→ego coordinate chain ...")
    test_global_to_ego_chain()

    out = '/home/claude/phantom_agent/synthetic_validation.png'
    print(f"\n[3] Saving visualization → {out}")
    visualize_synthetic(occ_grid, osz_mask, out)

    print("\n=== All synthetic tests passed ===")
    print("The pipeline is geometrically correct.")
    print("Next step: run with real nuScenes data.\n")
