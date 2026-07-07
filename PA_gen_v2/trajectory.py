"""
PA_gen_v2/trajectory.py
=====================
Trajectory interpolation for ghost-vehicle lookback analysis.

For a vehicle missing a direct annotation in a lookback frame, we must
 distinguish three cases:

  (a) KNOWN: exact annotation exists at that timestamp.
  (b) INTERPOLATED: the timestamp falls between two known annotations of
      the same track -> linearly interpolate and let the caller verify
      the position is still inside the BEV grid.
  (c) NO_EVIDENCE: timestamp is before track start or after track end ->
      no occlusion evidence either way; the caller must drop this frame
      instead of guessing.

This avoids conflating "genuinely occluded while the track continues",
"track hasn't started / already ended", and "object is out of sensing
range", which would otherwise inflate ghost-vehicle positives with false
events (e.g. a car that simply drove out of range).

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
    PA_gen_v1/create_pa_labels_mini.py already uses to build per-vehicle
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
