"""
PA_gen_v2/trajectory.py
=====================
Fixes a hidden assumption that used to live in ghost_vehicle_miner.py:

    if lb_xyz_global is None:
        # Vehicle was not annotated in this lookback frame.
        # Treat as "effectively in OSZ" (unseen = possibly hidden)
        was_in_osz_per_frame.append(True)

That line conflated three different situations under one guess:

  (a) the vehicle genuinely has ~0% visibility at that frame. nuScenes
      annotators still box objects down to ~40% visibility (vis=1), so a
      real annotation gap WHILE THE TRACK CONTINUES ON BOTH SIDES usually
      does mean "fully invisible right now" -> real occlusion evidence.
  (b) the vehicle's track hasn't started yet, or already ended, at that
      timestamp. It may not even exist in the scene there. This is NOT
      evidence of occlusion — it's simply no information.
  (c) the vehicle was outside our BEV grid at that moment. Also not
      evidence of occlusion, just outside our sensing range.

Blindly treating all three as "occluded" inflates positive ghost-vehicle
events with false positives — e.g. a car that simply drove out of range
gets mined as "emerged from a shadow" even though nothing occluded it.

The fix implemented here: build each instance's FULL annotation chain
(not just the k-frame lookback window used by the miner), and for any
lookback frame with no direct annotation, ask whether the query timestamp
falls BETWEEN two known chain entries (bracketed -> case a, interpolate a
position and let the caller re-verify it's within BEV range) or outside
the track's known span entirely (case b -> "no evidence", explicitly
distinguishable from both True and False so the caller can exclude it
from both the positive and negative evidence counts instead of guessing).

Every function here is a pure function of its arguments (no nuScenes
network/disk calls inside locate_at_time) so it's cheap to unit test in
isolation — see test_units.py.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Status values returned by locate_at_time()
KNOWN        = 'known'          # exact annotation exists at this timestamp
INTERPOLATED = 'interpolated'   # no annotation, but the track brackets the gap
NO_EVIDENCE  = 'no_evidence'    # timestamp is outside the track's known span

Trajectory = List[Tuple[int, str, np.ndarray]]   # (timestamp, sample_token, xyz_global)


def build_instance_trajectories(nusc, instance_tokens) -> Dict[str, Trajectory]:
    """
    For each instance token, walk its FULL sample_annotation chain
    (first_annotation_token -> 'next' links — the same mechanism
    preprocess/create_pa_labels_mini.py already uses to build per-vehicle
    tracks) and record every frame where it has a real annotation.

    This is independent of any lookback window: the point of building the
    whole track is to see past the k-frame window the miner looks at, so
    we can tell whether a gap in that window is "no annotation because
    genuinely occluded" or "no annotation because the track doesn't reach
    there at all".

    Returns {instance_token: [(timestamp, sample_token, xyz_global), ...]}
    each list sorted ascending by timestamp (guaranteed by construction,
    since 'next' always advances forward in time).
    """
    traj: Dict[str, Trajectory] = {}
    for inst_token in instance_tokens:
        inst = nusc.get('instance', inst_token)
        chain: Trajectory = []
        tok = inst['first_annotation_token']
        while tok:
            ann = nusc.get('sample_annotation', tok)
            sample = nusc.get('sample', ann['sample_token'])
            chain.append((
                sample['timestamp'],
                ann['sample_token'],
                np.array(ann['translation'], dtype=np.float32),
            ))
            tok = ann['next']
        traj[inst_token] = chain
    return traj


def locate_at_time(traj: Trajectory, query_timestamp: int
                   ) -> Tuple[str, Optional[np.ndarray]]:
    """
    Given one instance's trajectory and a query timestamp, return
    (status, global_xyz_or_None):

        KNOWN / INTERPOLATED : global_xyz is a real or linearly
            interpolated 3D position. The caller should still transform
            to ego frame and check in_bev_range() before treating it as
            occlusion evidence — an interpolated position can land
            outside the grid mid-track (e.g. during a wide turn).
        NO_EVIDENCE : global_xyz is None. The caller must NOT count this
            lookback frame as evidence of occlusion OR of visibility —
            there simply isn't any.
    """
    if not traj:
        return NO_EVIDENCE, None

    if query_timestamp < traj[0][0] or query_timestamp > traj[-1][0]:
        return NO_EVIDENCE, None

    for t_k, _, p_k in traj:
        if t_k == query_timestamp:
            return KNOWN, p_k

    for k in range(len(traj) - 1):
        t0, _, p0 = traj[k]
        t1, _, p1 = traj[k + 1]
        if t0 < query_timestamp < t1:
            alpha = (query_timestamp - t0) / (t1 - t0)
            p = p0 + alpha * (p1 - p0)
            return INTERPOLATED, p.astype(np.float32)

    # Bounds check above makes this unreachable; kept as an explicit
    # failure mode rather than silently falling through.
    raise RuntimeError(
        f"locate_at_time: query_timestamp {query_timestamp} is within "
        f"[{traj[0][0]}, {traj[-1][0]}] but no bracketing pair was found "
        f"— trajectory is not sorted ascending, which should be "
        f"impossible given how build_instance_trajectories() builds it."
    )
