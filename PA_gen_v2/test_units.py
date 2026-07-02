"""
test_units.py
-------------
Unit tests for coordinate transforms, OSZ-source plumbing, and the
trajectory-interpolation fix. Run this FIRST before touching any real
nuScenes data.

Karpathy: if your unit tests pass but your output looks wrong, you have a
bug in the test. If your output looks right but tests fail, you have a
bug in the code. Either way, fix it before moving on.

Changes vs. the original version
---------------------------------
- Tests now target osz_source.py (OSZ/modules/ray_casting.py) instead of
  the old filter/osz_geometry.py.
- New: test_osz_source_ij_convention — an intentionally ASYMMETRIC
  synthetic occluder (different extent in x vs y) that would catch a
  transposed-axis bug the way a symmetric wall test cannot (see
  osz_source.py's module docstring for why this axis order is a real
  risk here).
- New: tests for filter/trajectory.py's locate_at_time, covering the
  KNOWN / INTERPOLATED / NO_EVIDENCE cases that replace the old
  "unannotated lookback frame = assume occluded" shortcut.
- sys.path no longer hardcodes /home/claude/phantom_agent; it's computed
  from this file's own location so the test suite works regardless of
  where the repo is checked out.
"""

import sys
from pathlib import Path
import numpy as np
import pyquaternion

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
from ghost_vehicle_miner import _global_to_ego
from trajectory import build_instance_trajectories, locate_at_time, KNOWN, INTERPOLATED, NO_EVIDENCE
from OSZ.modules.ray_casting import cast_osz_2d
from common import bev_config


def _run(name: str, fn):
    try:
        fn()
        print(f"  ✓  {name}")
    except AssertionError as e:
        print(f"  ✗  {name}: {e}")
        return False
    return True


# ── Test: common/bev_config.py internal consistency ─────────────────────
def test_bev_config_consistency():
    nx, ny = osz_source.grid_shape()
    assert nx == ny == bev_config.BEV_NX, \
        f"osz_source's caster grid ({nx},{ny}) doesn't match bev_config.BEV_NX={bev_config.BEV_NX}"
    assert nx > 0


# ── Test: bev_xy_to_ij / ij_to_bev_xy round-trip ─────────────────────────
def test_ij_roundtrip():
    for x, y in [(0.0, 0.0), (10.5, -7.3), (-30.1, 25.0)]:
        i, j = osz_source.bev_xy_to_ij(x, y)
        xr, yr = osz_source.ij_to_bev_xy(i, j)
        assert abs(xr - x) < bev_config.BEV_RESOLUTION_M, \
            f"x roundtrip failed: {x} -> i={i} -> {xr}"
        assert abs(yr - y) < bev_config.BEV_RESOLUTION_M, \
            f"y roundtrip failed: {y} -> j={j} -> {yr}"


# ── Test: ego sits at the grid centre ────────────────────────────────────
def test_ego_at_center():
    i, j = osz_source.bev_xy_to_ij(0.0, 0.0)
    nx, ny = osz_source.grid_shape()
    assert abs(i - nx // 2) <= 1, f"ego i={i}, expected ~{nx // 2}"
    assert abs(j - ny // 2) <= 1, f"ego j={j}, expected ~{ny // 2}"


# ── Test: out-of-range points correctly flagged ──────────────────────────
def test_bev_range_check():
    assert osz_source.in_bev_range(0.0, 0.0)
    assert not osz_source.in_bev_range(bev_config.BEV_EXTENT_M + 1, 0.0)


# ── Test: single occluder blocks a symmetric shadow ──────────────────────
def test_osz_single_occluder_symmetric():
    """
    Wall spanning all of ego-y at x=+10m. Everything beyond x=+10m along
    the +x ray directly ahead of ego should be OSZ; everything in front
    of the wall should not.
    """
    nx, ny = osz_source.grid_shape()
    bev_occ = np.zeros((nx, ny), dtype=bool)
    wall_i, _ = osz_source.bev_xy_to_ij(10.0, 0.0)
    bev_occ[wall_i, :] = True

    caster = osz_source.get_caster()
    osz = cast_osz_2d(bev_occ, caster)

    shadow_i, shadow_j = osz_source.bev_xy_to_ij(12.0, 0.0)
    assert osz[shadow_i, shadow_j], \
        f"Cell at x=12m,y=0m should be in OSZ (behind wall at x=10m)"

    front_i, front_j = osz_source.bev_xy_to_ij(5.0, 0.0)
    assert not osz[front_i, front_j], \
        f"Cell at x=5m,y=0m should be visible (in front of wall)"


# ── Test: (i,j) axis order regression — the transpose bug this repo hit ──
def test_osz_source_ij_convention():
    """
    Build an INTENTIONALLY ASYMMETRIC occluder: a wall that only exists
    for y in [-2, 2] (narrow in y), positioned at a specific x. If the
    (i, j) axis order were ever accidentally swapped anywhere in this
    pipeline, the shadow would extend in the wrong physical direction —
    e.g. narrow-in-y would become narrow-in-x — and this test would fail
    where the single-occluder symmetric test above could not catch it
    (a full-width wall looks the same regardless of axis order).
    """
    nx, ny = osz_source.grid_shape()
    bev_occ = np.zeros((nx, ny), dtype=bool)

    # Wall: fixed x=+8m, spans only y in [-2, 2] (narrow strip in y)
    wall_x = 8.0
    for y in np.arange(-2.0, 2.0, bev_config.BEV_RESOLUTION_M):
        i, j = osz_source.bev_xy_to_ij(wall_x, y)
        bev_occ[i, j] = True

    caster = osz_source.get_caster()
    osz = cast_osz_2d(bev_occ, caster)

    # Directly behind the wall (x=12, y=0) -> should be OSZ.
    i_behind, j_behind = osz_source.bev_xy_to_ij(12.0, 0.0)
    assert osz[i_behind, j_behind], \
        "Point directly behind the narrow wall should be in OSZ"

    # Same x=12m but y=10m, well outside the wall's y-span -> should NOT
    # be OSZ (nothing occludes the ray to this point). If (i,j) were
    # swapped somewhere, this axis (large offset in y) would instead
    # behave like the "behind the wall along x" case and wrongly show
    # up as OSZ, or vice versa for a symmetric offset in x.
    i_clear, j_clear = osz_source.bev_xy_to_ij(12.0, 10.0)
    assert not osz[i_clear, j_clear], \
        ("Point at x=12m,y=10m (outside the wall's narrow y-span) should "
         "be visible. If this fails, check for a swapped (i,j) axis order "
         "somewhere in the pipeline — see osz_source.py's module docstring.")

    # And a point at the SAME y as the wall (y=0) but x=8m -> BEFORE the
    # wall along the ray from ego -> should NOT be OSZ either.
    i_front, j_front = osz_source.bev_xy_to_ij(4.0, 0.0)
    assert not osz[i_front, j_front], \
        "Point in front of the wall (x=4m) should be visible"


# ── Test: global -> ego transform ────────────────────────────────────────
def test_global_to_ego_identity():
    ego_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(1, 0, 0, 0)
    pt_global = np.array([5.0, 3.0, 0.0], dtype=np.float32)
    pt_ego = _global_to_ego(pt_global, ego_t, ego_q)
    np.testing.assert_allclose(pt_ego, pt_global, atol=1e-5)


def test_global_to_ego_translation():
    ego_t = np.array([10.0, 5.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(1, 0, 0, 0)
    pt_global = np.array([15.0, 8.0, 0.0], dtype=np.float32)
    pt_ego = _global_to_ego(pt_global, ego_t, ego_q)
    expected = np.array([5.0, 3.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(pt_ego, expected, atol=1e-5)


def test_global_to_ego_rotation():
    ego_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_q = pyquaternion.Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
    pt_global = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    pt_ego = _global_to_ego(pt_global, ego_t, ego_q)
    assert abs(pt_ego[0]) < 0.1, f"ego-x should be ~0, got {pt_ego[0]:.3f}"
    assert pt_ego[1] < -4.0,     f"ego-y should be ~-5, got {pt_ego[1]:.3f}"


# ── Tests: trajectory.locate_at_time (the hidden-assumption fix) ────────
def _make_traj(entries):
    """entries: list of (timestamp, x). Builds a trajectory list with
    y=z=0 for brevity."""
    return [(t, f'sample_{t}', np.array([x, 0.0, 0.0], dtype=np.float32))
            for t, x in entries]


def test_trajectory_known_exact_match():
    traj = _make_traj([(100, 1.0), (200, 2.0), (300, 3.0)])
    status, xyz = locate_at_time(traj, 200)
    assert status == KNOWN
    np.testing.assert_allclose(xyz, [2.0, 0.0, 0.0])


def test_trajectory_interpolated_midpoint():
    """
    Query timestamp falls exactly halfway between two known frames ->
    interpolated x should be exactly halfway too. This is the case that
    replaces the old "assume occluded" shortcut: track continues on both
    sides, so a missing annotation here really does mean near-zero
    visibility, and we can also sanity-check WHERE it likely was.
    """
    traj = _make_traj([(100, 0.0), (300, 10.0)])
    status, xyz = locate_at_time(traj, 200)
    assert status == INTERPOLATED
    np.testing.assert_allclose(xyz, [5.0, 0.0, 0.0], atol=1e-5)


def test_trajectory_no_evidence_before_track_start():
    """
    Query timestamp is before the instance's first annotation -> the old
    code would have assumed "occluded" here (False positive risk: the car
    might simply not have existed in the scene yet). The fix must return
    NO_EVIDENCE, not True.
    """
    traj = _make_traj([(500, 1.0), (600, 2.0)])
    status, xyz = locate_at_time(traj, 100)
    assert status == NO_EVIDENCE
    assert xyz is None


def test_trajectory_no_evidence_after_track_end():
    traj = _make_traj([(100, 1.0), (200, 2.0)])
    status, xyz = locate_at_time(traj, 999)
    assert status == NO_EVIDENCE
    assert xyz is None


def test_trajectory_no_evidence_empty():
    status, xyz = locate_at_time([], 100)
    assert status == NO_EVIDENCE
    assert xyz is None


def test_trajectory_no_evidence_does_not_equal_occluded():
    """
    Explicit regression test for the actual bug being fixed: NO_EVIDENCE
    is a THIRD state distinct from both KNOWN and INTERPOLATED, and
    callers (ghost_vehicle_miner.py) must map it to None, never to True.
    This test pins down the contract: the three status strings must all
    be distinct, so a future edit cannot silently merge NO_EVIDENCE into
    one of the other two (which is exactly how the original "unseen =
    assume occluded" bug happened — a missing-annotation case got folded
    into the same code path as a confirmed-occluded case).
    """
    assert len({KNOWN, INTERPOLATED, NO_EVIDENCE}) == 3, \
        "KNOWN / INTERPOLATED / NO_EVIDENCE must be three distinct values"

    traj = _make_traj([(500, 1.0), (600, 2.0)])
    status, xyz = locate_at_time(traj, 100)
    assert status == NO_EVIDENCE
    assert xyz is None, "NO_EVIDENCE must carry no position — nothing to interpolate from"


# ── Run all tests ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ("bev_config internal consistency",      test_bev_config_consistency),
        ("(i,j) <-> metric round-trip",          test_ij_roundtrip),
        ("ego at grid centre",                    test_ego_at_center),
        ("BEV range check",                       test_bev_range_check),
        ("OSZ single occluder (symmetric)",       test_osz_single_occluder_symmetric),
        ("OSZ (i,j) axis-order regression",       test_osz_source_ij_convention),
        ("global->ego identity",                  test_global_to_ego_identity),
        ("global->ego translation",               test_global_to_ego_translation),
        ("global->ego rotation 90°",              test_global_to_ego_rotation),
        ("trajectory: known exact match",         test_trajectory_known_exact_match),
        ("trajectory: interpolated midpoint",     test_trajectory_interpolated_midpoint),
        ("trajectory: no evidence (before start)",test_trajectory_no_evidence_before_track_start),
        ("trajectory: no evidence (after end)",   test_trajectory_no_evidence_after_track_end),
        ("trajectory: no evidence (empty track)", test_trajectory_no_evidence_empty),
        ("trajectory: no_evidence != occluded",   test_trajectory_no_evidence_does_not_equal_occluded),
    ]

    print("\n=== Unit Tests ===")
    print(bev_config.describe())
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
