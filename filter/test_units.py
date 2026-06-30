"""
test_units.py
-------------
Unit tests for coordinate transforms and OSZ geometry.
Run this FIRST before touching any real nuScenes data.

Karpathy: if your unit tests pass but your output looks wrong,
you have a bug in the test. If your output looks right but tests
fail, you have a bug in the code. Either way, fix it before moving on.
"""

import numpy as np
import sys
import pyquaternion

# ── import modules under test ───────────────────────────────────────────────
sys.path.insert(0, '/home/claude/phantom_agent')
from osz_geometry import (
    build_bev_occupancy,
    cast_osz_mask,
    bev_coords_to_pixel,
    pixel_to_bev_coords,
    BEV_SIZE, BEV_CENTER, BEV_RANGE_M, BEV_RESOLUTION,
)
from ghost_vehicle_miner import _global_to_ego, _is_in_bev_range


def _run(name: str, fn):
    try:
        fn()
        print(f"  ✓  {name}")
    except AssertionError as e:
        print(f"  ✗  {name}: {e}")
        return False
    return True


# ── Test 1: pixel ↔ metric round-trip ───────────────────────────────────────
def test_pixel_roundtrip():
    for x, y in [(0.0, 0.0), (10.5, -7.3), (-30.1, 25.0)]:
        col, row = bev_coords_to_pixel(x, y)
        xr, yr   = pixel_to_bev_coords(col, row)
        # Should be within one pixel's worth of rounding error
        assert abs(xr - x) < BEV_RESOLUTION, \
            f"x roundtrip failed: {x} → col={col} → {xr}"
        assert abs(yr - y) < BEV_RESOLUTION, \
            f"y roundtrip failed: {y} → row={row} → {yr}"


# ── Test 2: ego is at BEV_CENTER ────────────────────────────────────────────
def test_ego_at_center():
    col, row = bev_coords_to_pixel(0.0, 0.0)
    assert col == BEV_CENTER, f"ego col={col}, expected {BEV_CENTER}"
    assert row == BEV_CENTER, f"ego row={row}, expected {BEV_CENTER}"


# ── Test 3: out-of-range points correctly flagged ───────────────────────────
def test_bev_range_check():
    assert _is_in_bev_range(np.array([0.0, 0.0, 0.0]))          # inside
    assert not _is_in_bev_range(np.array([BEV_RANGE_M + 1, 0.0, 0.0]))  # out


# ── Test 4: global→ego transform — identity when ego is at origin ───────────
def test_global_to_ego_identity():
    ego_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(1, 0, 0, 0)   # identity rotation
    pt_global = np.array([5.0, 3.0, 0.0], dtype=np.float32)
    pt_ego    = _global_to_ego(pt_global, ego_t, ego_q)
    np.testing.assert_allclose(pt_ego, pt_global, atol=1e-5,
        err_msg="Identity transform should be no-op")


# ── Test 5: global→ego transform — ego translated ───────────────────────────
def test_global_to_ego_translation():
    ego_t = np.array([10.0, 5.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(1, 0, 0, 0)   # no rotation
    pt_global = np.array([15.0, 8.0, 0.0], dtype=np.float32)
    pt_ego    = _global_to_ego(pt_global, ego_t, ego_q)
    expected  = np.array([5.0, 3.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(pt_ego, expected, atol=1e-5,
        err_msg=f"Expected {expected}, got {pt_ego}")


# ── Test 6: global→ego transform — ego rotated 90° around z ─────────────────
def test_global_to_ego_rotation():
    # Ego is at origin, facing "left" (rotated 90° CCW around z-axis)
    ego_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(axis=[0,0,1], angle=np.pi/2)
    # A point directly ahead in global x-direction
    pt_global = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    pt_ego    = _global_to_ego(pt_global, ego_t, ego_q)
    # After 90° CCW rotation of ego: global +x maps to ego -y
    assert abs(pt_ego[0]) < 0.1, f"ego-x should be ~0, got {pt_ego[0]:.3f}"
    assert pt_ego[1] < -4.0,     f"ego-y should be ~-5, got {pt_ego[1]:.3f}"


# ── Test 7: OSZ mask — single occluder blocks shadow ────────────────────────
def test_osz_single_occluder():
    """
    Place a wall of obstacles at x=+10m (col ≈ BEV_CENTER + 50).
    Everything beyond x=+10m should be in the OSZ.
    Ego at center looks in +x direction.
    """
    grid = np.zeros((BEV_SIZE, BEV_SIZE), dtype=bool)
    # Wall at col = BEV_CENTER + 50 (= x = +10m), spanning all rows
    wall_col = BEV_CENTER + 50
    grid[:, wall_col] = True

    osz = cast_osz_mask(grid)

    # A cell at col = BEV_CENTER + 60 (x=+12m) should be in OSZ
    shadow_col = BEV_CENTER + 60
    shadow_row = BEV_CENTER           # directly ahead

    # Near the wall but behind it
    assert osz[shadow_row, shadow_col] > 0.5, \
        f"Cell at ({shadow_col},{shadow_row}) should be in OSZ but osz={osz[shadow_row,shadow_col]:.2f}"

    # A cell in front of the wall should NOT be in OSZ
    front_col = BEV_CENTER + 30
    assert osz[shadow_row, front_col] < 0.5, \
        f"Cell at ({front_col},{shadow_row}) should be visible but osz={osz[shadow_row,front_col]:.2f}"


# ── Test 8: occupancy grid — basic point projection ─────────────────────────
def test_occupancy_projection():
    # Put a single point at (10m, 0m, 1m) in ego frame
    pts = np.array([[10.0, 0.0, 1.0]], dtype=np.float32)
    # Pad to meet >100 points threshold by adding many ground points
    ground = np.random.uniform(-40, 40, (500, 3)).astype(np.float32)
    ground[:, 2] = 0.5   # just above ground plane
    pts_all = np.vstack([pts, ground])

    grid = build_bev_occupancy(pts_all)
    col, row = bev_coords_to_pixel(10.0, 0.0)
    assert grid[row, col], \
        f"Point at (10m,0m) should appear in grid at ({col},{row})"


# ── Run all tests ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ("pixel ↔ metric round-trip",          test_pixel_roundtrip),
        ("ego at BEV_CENTER",                   test_ego_at_center),
        ("BEV range check",                     test_bev_range_check),
        ("global→ego identity",                 test_global_to_ego_identity),
        ("global→ego translation",              test_global_to_ego_translation),
        ("global→ego rotation 90°",             test_global_to_ego_rotation),
        ("OSZ single occluder",                 test_osz_single_occluder),
        ("occupancy projection",                test_occupancy_projection),
    ]

    print("\n=== Unit Tests ===")
    results = [_run(name, fn) for name, fn in tests]
    passed  = sum(results)
    total   = len(results)

    print(f"\n{'='*30}")
    print(f"  {passed}/{total} passed")
    if passed < total:
        print("  FIX THE FAILURES BEFORE RUNNING THE PIPELINE.")
        sys.exit(1)
    else:
        print("  All clear — proceed to pipeline.")
    print(f"{'='*30}\n")
