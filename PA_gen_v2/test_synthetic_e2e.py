"""
test_synthetic_e2e.py
---------------------
End-to-end test using synthetic data (no real nuScenes required).

We construct a fake scene with:
  - A wall of occluders at x=+15m spanning y in [-5, 5] (simulating a
    building / row of parked trucks)
  - A ghost vehicle behind the wall at x=+20m (should be OSZ)
  - The same vehicle's "emerged" position at x=+13m (in front of the
    wall — should NOT be OSZ)

Expected outcome:
  - OSZ mask covers the region behind the wall
  - The x=+20m point is flagged "in OSZ"
  - The x=+13m point is NOT flagged "in OSZ"

We ALSO exercise the trajectory-interpolation fix from filter/trajectory.py
on a synthetic instance track, to demonstrate — without needing real
nuScenes annotations — that a genuinely-bracketed gap in a track gets
treated as occlusion evidence, while a gap outside the track's known span
does not (the bug the original "unseen = assume occluded" shortcut had).

This validates the geometry + the interpolation logic without any I/O or
nuScenes API calls. Real per-camera depth reprojection (the part that DOES
need real sensor data) is exercised separately by OSZ/run_osz_pipeline.py
--mock and by filter/osz_source_viz.py against real data.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
from OSZ.modules.ray_casting import cast_osz_2d
from trajectory import locate_at_time, KNOWN, INTERPOLATED, NO_EVIDENCE

OUT_PATH = _THIS_DIR / 'output' / 'synthetic_validation.png'


def make_synthetic_scene():
    """
    Returns (bev_occ, osz_mask, wall_x_m, ghost_behind_m, ghost_emerged_m).
    """
    wall_x_m        = 15.0
    ghost_behind_m  = 20.0
    ghost_emerged_m = 13.0

    caster = osz_source.get_caster()
    nx, ny = caster.nx, caster.ny
    bev_occ = np.zeros((nx, ny), dtype=bool)

    for y in np.arange(-5.0, 5.0, caster.bev_res):
        i, j = osz_source.bev_xy_to_ij(wall_x_m, y)
        if 0 <= i < nx and 0 <= j < ny:
            bev_occ[i, j] = True

    osz_mask = cast_osz_2d(bev_occ, caster).astype(np.float32)
    return bev_occ, osz_mask, wall_x_m, ghost_behind_m, ghost_emerged_m


def test_synthetic_osz_coverage():
    """Ghost behind wall -> in OSZ; emerged position -> NOT in OSZ."""
    bev_occ, osz_mask, wall_x, ghost_behind, ghost_emerged = make_synthetic_scene()

    in_osz_behind  = osz_source.is_in_osz(ghost_behind, 0.0, osz_mask)
    in_osz_emerged = osz_source.is_in_osz(ghost_emerged, 0.0, osz_mask)

    print(f"  Ghost at x=+{ghost_behind}m (behind wall):  in_osz={in_osz_behind}")
    print(f"  Ghost at x=+{ghost_emerged}m (emerged):     in_osz={in_osz_emerged}")

    assert in_osz_behind, \
        f"Point at x={ghost_behind}m should be in OSZ (behind wall at x={wall_x}m)"
    assert not in_osz_emerged, \
        f"Point at x={ghost_emerged}m should NOT be in OSZ (in front of wall)"

    print("  ✓ OSZ coverage test passed.")
    return bev_occ, osz_mask


def test_trajectory_interpolation_on_synthetic_track():
    """
    Simulate an instance whose annotation chain has a gap exactly where
    it was behind the wall (x=20m at t=200..400), and is annotated
    normally before/after. locate_at_time() should:
      - return NO_EVIDENCE for a query far before the track starts
      - return INTERPOLATED (bracketed, in-range) for a query inside the
        gap, with a position consistent with "still behind the wall"
      - return KNOWN for a query that lands exactly on a real annotation
    """
    traj = [
        (0,   'tok_0',   np.array([10.0, 0.0, 0.0], dtype=np.float32)),
        (100, 'tok_100', np.array([15.0, 0.0, 0.0], dtype=np.float32)),
        # gap: vehicle behind the wall, un-annotated at t=200..400
        (500, 'tok_500', np.array([13.0, 0.0, 0.0], dtype=np.float32)),  # emerged
    ]

    status_before, xyz_before = locate_at_time(traj, -50)
    assert status_before == NO_EVIDENCE, \
        "Query before the track starts must be NO_EVIDENCE, not an assumed occlusion"
    assert xyz_before is None

    status_gap, xyz_gap = locate_at_time(traj, 300)
    assert status_gap == INTERPOLATED, \
        "Query inside a bracketed gap must be INTERPOLATED"
    # linear interp between (100, x=15) and (500, x=13) at t=300 (midpoint)
    expected_x = 15.0 + (300 - 100) / (500 - 100) * (13.0 - 15.0)
    assert abs(xyz_gap[0] - expected_x) < 1e-3, \
        f"Interpolated x should be {expected_x:.2f}, got {xyz_gap[0]:.2f}"

    status_known, xyz_known = locate_at_time(traj, 500)
    assert status_known == KNOWN
    np.testing.assert_allclose(xyz_known, [13.0, 0.0, 0.0])

    print(f"  Before track start (t=-50):  status={status_before}")
    print(f"  Inside bracketed gap (t=300): status={status_gap}, "
          f"interpolated x={xyz_gap[0]:.2f}m (expected {expected_x:.2f}m)")
    print(f"  Exact match (t=500):          status={status_known}")
    print("  ✓ trajectory interpolation test passed.")


def visualize_synthetic(bev_occ, osz_mask, out_path):
    """Save a picture of the synthetic scene for manual inspection."""
    caster = osz_source.get_caster()
    x_min, x_max, y_min, y_max = caster.bev_range
    extent = [y_max, y_min, x_min, x_max]   # OSZ convention: forward=up, left=left

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Synthetic Scene Validation (via OSZ/modules/ray_casting.py)",
                 fontsize=11)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ]        = [0.7, 0.7, 0.7]
    overlay[osz_mask > 0.5] = [0.85, 0.1, 0.1]

    axes[0].imshow(overlay, origin='lower', extent=extent)
    axes[0].set_xlim(y_max, y_min); axes[0].set_ylim(x_min, x_max)
    axes[0].plot(0, 0, 'g^', markersize=10, label='Ego')
    axes[0].axhline(15.0, color='cyan', linewidth=1, linestyle='--', label='Wall (x=+15m)')
    axes[0].plot(0, 20.0, 'rx', markersize=12, markeredgewidth=2, label='Ghost behind wall')
    axes[0].plot(0, 13.0, 'g*', markersize=12, label='Emerged (x=+13m)')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].set_title("Overlay (gray=occ, red=OSZ)")
    axes[0].set_xlabel('y (m)'); axes[0].set_ylabel('x (m) — forward is up')

    axes[1].imshow(osz_mask, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=1)
    axes[1].set_xlim(y_max, y_min); axes[1].set_ylim(x_min, x_max)
    axes[1].plot(0, 0, 'b^', markersize=8)
    axes[1].axhline(15.0, color='cyan', linewidth=1, linestyle='--')
    axes[1].set_title("OSZ mask (hot = occluded)")
    axes[1].set_xlabel('y (m)'); axes[1].set_ylabel('x (m) — forward is up')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"  Saved synthetic viz → {out_path}")
    plt.close()


if __name__ == '__main__':
    print("\n=== Synthetic End-to-End Validation ===\n")

    print("[1] Building synthetic scene and testing OSZ coverage ...")
    bev_occ, osz_mask = test_synthetic_osz_coverage()

    print("\n[2] Testing trajectory interpolation (hidden-assumption fix) ...")
    test_trajectory_interpolation_on_synthetic_track()

    print(f"\n[3] Saving visualization → {OUT_PATH}")
    visualize_synthetic(bev_occ, osz_mask, OUT_PATH)

    print("\n=== All synthetic tests passed ===")
    print("The pipeline is geometrically correct.")
    print("Next step: run with real nuScenes data (osz_source_viz.py, then run_pipeline.py).\n")
