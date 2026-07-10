"""
ghost_vehicle_miner.py
----------------------
Mine "ghost vehicle emergence events" from nuScenes tracking annotations.

Definition of a ghost vehicle emergence event:
    At timestamp t, a tracked vehicle V appears (its 3D bbox center) inside
    the VISIBLE region near the OSZ boundary.  For the previous k frames
    (t-k ... t-1), V was either:
      (a) not detected at all, OR
      (b) its last known position was inside the OSZ of those frames.
    This means V "emerged from the shadow" at frame t — it was a phantom
    before t, and became visible at t.

Output per event:
    {
        'scene_token':    str,
        'emerge_sample':  str,          # token of frame t
        'instance_token': str,          # nuScenes instance id of V
        'emerge_bev_xy':  (float,float),# BEV ego coords at frame t
        'lookback_tokens': [str, ...],  # sample tokens t-k .. t-1
        'was_in_osz':     [bool|None, ...],  # per lookback frame verdict —
                                              # True=confirmed occluded,
                                              # False=confirmed visible,
                                              # None=no evidence either way
                                              # (see PA_gen_v2/trajectory.py)
        'n_osz_frames':      int,       # count of True
        'n_evidence_frames': int,       # count of True or False (not None)
        'label':          int,          # 1 = positive ghost event
    }

Negative samples are also generated: frames where a vehicle is visible in
the current frame AND has a CONFIRMED-visible verdict in every one of the
k lookback frames (no unknowns, no OSZ involvement).

OSZ is computed by OSZ/modules/ray_casting.py via osz_source.py. Occlusion
decisions use PA-relevant OSZ (raw OSZ intersected with the nuScenes
drivable area) so that building shadows do not count as phantom-vehicle
candidate regions.

Karpathy notes:
  - Print counts at every stage. Never trust a silent loop.
  - Assert coordinate transforms at known test cases.
  - Build lookup tables up front; don't query nuScenes in the inner loop.
"""

import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

import pyquaternion
from nuscenes.nuscenes import NuScenes

_THIS_DIR   = Path(__file__).resolve().parent          # PA_gen_v2/ itself
_REPO_ROOT  = _THIS_DIR.parent                          # repo root, for common/, OSZ/
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
from trajectory import build_instance_trajectories, locate_at_time, NO_EVIDENCE


# ---------------------------------------------------------------------------
# nuScenes category filter — we only care about wheeled vehicles
# ---------------------------------------------------------------------------
VEHICLE_CATEGORIES = {
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.motorcycle',
    'vehicle.bicycle',
    'vehicle.trailer',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
}

LOOKBACK_FRAMES   = 4    # k: how many past frames to inspect
MIN_OSZ_OVERLAP   = 1    # at least this many CONFIRMED-occluded lookback
                          # frames are required for a positive event


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def _build_instance_category_map(nusc: NuScenes) -> Dict[str, str]:
    """Returns {instance_token: category_name} for all instances."""
    mapping = {}
    for instance in nusc.instance:
        ann = nusc.get('sample_annotation', instance['first_annotation_token'])
        mapping[instance['token']] = ann['category_name']
    print(f"  [index] Built category map for {len(mapping)} instances.")
    return mapping


def _build_sample_annotations_map(
        nusc: NuScenes,
        instance_category_map: Dict[str, str]
) -> Dict[str, List[Dict]]:
    """
    Returns {sample_token: [annotation_dict, ...]} for VEHICLE annotations
    only. Positions are stored in GLOBAL frame; converted to ego frame
    per-sample when needed.
    """
    mapping: Dict[str, List[Dict]] = {}

    for ann in tqdm(nusc.sample_annotation,
                    desc='  [index] Indexing annotations', leave=False):
        cat = instance_category_map.get(ann['instance_token'], '')
        if cat not in VEHICLE_CATEGORIES:
            continue
        tok = ann['sample_token']
        if tok not in mapping:
            mapping[tok] = []
        mapping[tok].append({
            'instance_token':    ann['instance_token'],
            'translation_global': np.array(ann['translation'], dtype=np.float32),
            'rotation_global':    np.array(ann['rotation'], dtype=np.float32),
            'size':               ann['size'],
        })

    total_anns = sum(len(v) for v in mapping.values())
    print(f"  [index] {total_anns} vehicle annotations across "
          f"{len(mapping)} samples.")
    return mapping


def _global_to_ego(translation_global: np.ndarray,
                   ego_translation: np.ndarray,
                   ego_rotation_q: pyquaternion.Quaternion) -> np.ndarray:
    """Transform a 3D point from global frame to ego vehicle frame."""
    delta = translation_global - ego_translation
    pt_ego = ego_rotation_q.inverse.rotate(delta)
    return pt_ego.astype(np.float32)


def _global_heading_to_ego(rotation_global: np.ndarray,
                           ego_rotation_q: pyquaternion.Quaternion) -> float:
    """Transform a vehicle's global rotation to its yaw in ego frame."""
    global_q = pyquaternion.Quaternion(rotation_global.astype(np.float64))
    rel_q = ego_rotation_q.inverse * global_q
    return float(rel_q.yaw_pitch_roll[0])


def _get_ego_pose(nusc: NuScenes, sample_token: str
                  ) -> Tuple[np.ndarray, pyquaternion.Quaternion]:
    """Returns (translation, rotation_quaternion) of ego for a sample."""
    sample     = nusc.get('sample', sample_token)
    lidar_tok  = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_tok)
    ep         = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    t = np.array(ep['translation'], dtype=np.float32)
    q = pyquaternion.Quaternion(ep['rotation'])
    return t, q


def _get_lookback_samples(nusc: NuScenes, sample_token: str, k: int) -> List[str]:
    """
    Return the k samples BEFORE sample_token in the same scene, ordered
    oldest-first. Returns fewer than k items near the start of a scene.
    """
    tokens = []
    current = nusc.get('sample', sample_token)
    for _ in range(k):
        prev_tok = current['prev']
        if prev_tok == '':
            break
        tokens.append(prev_tok)
        current = nusc.get('sample', prev_tok)
    tokens.reverse()
    return tokens


# ---------------------------------------------------------------------------
# Core mining logic
# ---------------------------------------------------------------------------

def mine_ghost_events(
        nusc: NuScenes,
        scene_tokens: Optional[List[str]] = None,
        lookback_k: int = LOOKBACK_FRAMES,
        min_osz_overlap: int = MIN_OSZ_OVERLAP,
        neg_ratio: float = 3.0,      # negative : positive ratio (for balance)
        verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main mining function. Iterates over all samples in given scenes
    (or all scenes if None), identifies ghost vehicle emergence events.
    """
    if scene_tokens is None:
        scene_tokens = [s['token'] for s in nusc.scene]

    print(f"\n{'='*60}")
    print(f"Mining ghost events across {len(scene_tokens)} scenes")
    print(f"  lookback_k={lookback_k}, min_osz_overlap={min_osz_overlap}")
    grid_nx, grid_ny = osz_source.grid_shape()
    print(f"  OSZ source: OSZ/modules/ray_casting.py via osz_source.py "
          f"({grid_nx}x{grid_ny} grid)")
    print(f"  Occlusion decisions use PA-relevant OSZ (raw OSZ ∩ drivable "
          f"area) — {'drivable-area filtering ENABLED' if osz_source.drivable_filter_available() else 'drivable-area filtering UNAVAILABLE (shapely missing) — falling back to raw OSZ, expect much higher occlusion rates from building shadows'}")
    print(f"{'='*60}")

    # --- Build indices once ---
    inst_cat_map = _build_instance_category_map(nusc)
    ann_map      = _build_sample_annotations_map(nusc, inst_cat_map)

    # --- Build full trajectories for every vehicle instance up front ---
    # (needed to distinguish "genuinely occluded" gaps from "no evidence"
    #  gaps — see PA_gen_v2/trajectory.py)
    vehicle_instance_tokens = {
        tok for tok, cat in inst_cat_map.items() if cat in VEHICLE_CATEGORIES
    }
    print(f"  [index] Building full trajectories for "
          f"{len(vehicle_instance_tokens)} vehicle instances...")
    traj_map = build_instance_trajectories(nusc, vehicle_instance_tokens)

    positive_events: List[Dict] = []
    negative_events: List[Dict] = []
    n_dropped_ambiguous = 0   # events with unknown frames but no confirmed
                              # occlusion — neither a clean positive nor a
                              # clean negative, so we don't guess

    # --- Scene loop ---
    for scene_token in tqdm(scene_tokens, desc='Scenes'):
        scene = nusc.get('scene', scene_token)
        osz_source.clear_cache()   # bound memory; lookback never crosses
                                    # scene boundaries anyway (sample['prev']
                                    # is '' at scene start)

        sample_tokens_in_scene = []
        tok = scene['first_sample_token']
        while tok != '':
            sample_tokens_in_scene.append(tok)
            tok = nusc.get('sample', tok)['next']

        if verbose:
            tqdm.write(f"  Scene '{scene['name']}': "
                       f"{len(sample_tokens_in_scene)} samples")

        # --- Sample loop ---
        for sample_token in sample_tokens_in_scene:

            try:
                bev_occ_t, osz_raw_t, osz_pa_t, drivable_t = \
                    osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)
            except Exception as e:
                tqdm.write(f"  [WARN] OSZ failed for {sample_token}: {e}")
                continue

            ego_t, ego_q_t = _get_ego_pose(nusc, sample_token)

            anns_t = ann_map.get(sample_token, [])
            if not anns_t:
                continue

            lookback_tokens = _get_lookback_samples(nusc, sample_token, lookback_k)
            if len(lookback_tokens) < lookback_k:
                continue   # not enough history — skip (near start of scene)

            # Pre-fetch OSZ + ego pose for lookback frames (osz_source's
            # own cache means this is cheap even with the overlap across
            # consecutive samples' lookback windows).
            lookback_data = []
            valid_lookback = True
            for lb_tok in lookback_tokens:
                try:
                    bev_occ_lb, _, osz_pa_lb, _ = osz_source.get_pa_relevant_osz_for_sample(nusc, lb_tok)
                    ego_lb, ego_q_lb = _get_ego_pose(nusc, lb_tok)
                    # Degenerate frames (e.g. near-ego artifact short-circuiting
                    # the whole grid) are structurally uninformative — treat as
                    # NO_EVIDENCE rather than confirmed-occluded.
                    is_degenerate_lb = osz_source.is_frame_degenerate(nusc, lb_tok)
                    lookback_data.append((osz_pa_lb, bev_occ_lb, ego_lb, ego_q_lb, is_degenerate_lb))
                except Exception as e:
                    tqdm.write(f"  [WARN] OSZ failed for lookback {lb_tok}: {e}")
                    valid_lookback = False
                    break
            if not valid_lookback:
                continue

            # {instance_token: [annotation_dict or None per lookback frame]}
            inst_lb_presence: Dict[str, List[Optional[Dict]]] = {}
            for lb_idx, lb_tok in enumerate(lookback_tokens):
                for ann in ann_map.get(lb_tok, []):
                    itok = ann['instance_token']
                    if itok not in inst_lb_presence:
                        inst_lb_presence[itok] = [None] * len(lookback_tokens)
                    inst_lb_presence[itok][lb_idx] = ann

            # --- Instance loop at frame t ---
            for ann_t in anns_t:
                itok = ann_t['instance_token']

                pt_ego_t = _global_to_ego(ann_t['translation_global'], ego_t, ego_q_t)
                if not osz_source.in_bev_range(pt_ego_t[0], pt_ego_t[1]):
                    continue

                # Full box check: the vehicle is a valid PA candidate ONLY if
                # it is NOT the occluder itself. If any corner of its footprint
                # peeks outside OSZ, it's partially visible → skip.
                heading_t = _global_heading_to_ego(
                    ann_t['rotation_global'], ego_q_t)
                if osz_source.is_box_occluded_not_occluder(
                    pt_ego_t[0], pt_ego_t[1], heading_t,
                    ann_t['size'], osz_pa_t, bev_occ_t)[0]:
                    continue   # still fully in OSZ, no LiDAR hit — not emergent

                # --- Analyze lookback history of this instance ---
                lb_positions = inst_lb_presence.get(itok, [None] * lookback_k)
                traj = traj_map.get(itok, [])

                was_in_osz_per_frame: List[Optional[bool]] = []
                for lb_idx, lb_ann in enumerate(lb_positions):
                    osz_pa_lb, bev_occ_lb, ego_lb, ego_q_lb, is_degenerate_lb = lookback_data[lb_idx]
                    lb_tok = lookback_tokens[lb_idx]

                    if is_degenerate_lb:
                        was_in_osz_per_frame.append(None)
                        continue

                    if lb_ann is not None:
                        # Direct annotation — full box check to exclude occluders.
                        pt_ego_lb = _global_to_ego(
                            lb_ann['translation_global'], ego_lb, ego_q_lb)
                        if not osz_source.in_bev_range(pt_ego_lb[0], pt_ego_lb[1]):
                            was_in_osz_per_frame.append(None)
                            continue
                        heading_lb = _global_heading_to_ego(
                            lb_ann['rotation_global'], ego_q_lb)
                        was_in_osz_per_frame.append(
                            osz_source.is_box_occluded_not_occluder(
                                pt_ego_lb[0], pt_ego_lb[1],
                                heading_lb, lb_ann['size'],
                                osz_pa_lb, bev_occ_lb)[0])
                        continue

                    # No direct annotation: trajectory interpolation.
                    # Cannot do a full box check (no heading / size here),
                    # fall back to centre-point check.
                    lb_timestamp = nusc.get('sample', lb_tok)['timestamp']
                    status, xyz_interp = locate_at_time(traj, lb_timestamp)

                    if status == NO_EVIDENCE:
                        was_in_osz_per_frame.append(None)
                        continue

                    pt_ego_interp = _global_to_ego(xyz_interp, ego_lb, ego_q_lb)
                    if not osz_source.in_bev_range(pt_ego_interp[0], pt_ego_interp[1]):
                        was_in_osz_per_frame.append(None)
                        continue

                    # Track brackets the gap and the interpolated position
                    # is within our grid -> genuine occlusion evidence.
                    was_in_osz_per_frame.append(True)

                n_osz_frames = sum(1 for v in was_in_osz_per_frame if v is True)
                n_evidence   = sum(1 for v in was_in_osz_per_frame if v is not None)

                event_base = {
                    'scene_token':       scene_token,
                    'emerge_sample':     sample_token,
                    'instance_token':    itok,
                    'emerge_bev_xy':     (float(pt_ego_t[0]), float(pt_ego_t[1])),
                    'lookback_tokens':   lookback_tokens,
                    'was_in_osz':        was_in_osz_per_frame,
                    'n_osz_frames':      n_osz_frames,
                    'n_evidence_frames': n_evidence,
                }

                if n_osz_frames >= min_osz_overlap:
                    # POSITIVE: confirmed occluded in >=1 lookback frame,
                    # now emerged.
                    positive_events.append({**event_base, 'label': 1})
                elif n_evidence == lookback_k and n_osz_frames == 0:
                    # NEGATIVE: every lookback frame had a definitive
                    # answer, and every one of them says "visible".
                    negative_events.append({**event_base, 'label': 0})
                else:
                    # Ambiguous: some frames have no evidence and none are
                    # confirmed-occluded either. Not a clean positive, not
                    # a clean negative — drop rather than guess.
                    n_dropped_ambiguous += 1

    # --- Balance negative set ---
    n_pos = len(positive_events)
    n_neg_target = int(n_pos * neg_ratio)
    if len(negative_events) > n_neg_target:
        rng = np.random.default_rng(seed=42)
        neg_idx = rng.choice(len(negative_events), n_neg_target, replace=False)
        negative_events = [negative_events[i] for i in neg_idx]

    all_events = positive_events + negative_events

    print(f"\n{'='*60}")
    print("Mining complete.")
    print(f"  Positive events (ghost emergence): {len(positive_events)}")
    print(f"  Negative events (always visible):  {len(negative_events)}")
    print(f"  Dropped (ambiguous, no evidence):  {n_dropped_ambiguous}")
    print(f"  Total events:                      {len(all_events)}")
    print(f"{'='*60}\n")

    if len(positive_events) == 0:
        print("\n  ⚠  WARNING: No positive ghost emergence events found.")
        print("  This may be expected if:")
        print("    - The dataset is very small (e.g. v1.0-mini with few scenes)")
        print("    - OSZ coverage is insufficient (check --min_osz)")
        print("    - lookback_k is too large for the scene lengths")
        print("  Consider reducing --min_osz or --lookback to increase yield.\n")
    for e in all_events[:5]:
        assert 'emerge_sample'   in e
        assert 'emerge_bev_xy'   in e
        assert len(e['was_in_osz']) == lookback_k
        assert e['label'] in (0, 1)
    print("Output structure assertions passed.")

    return all_events


def save_events(events: List[Dict], path: str) -> None:
    """Serialize events to JSON. Converts numpy types for JSON compat."""
    def _convert(obj):
        if obj is None:                     return None
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, bool):           return obj
        if isinstance(obj, (list, tuple)):  return [_convert(x) for x in obj]
        return obj

    clean = [{k: _convert(v) for k, v in e.items()} for e in events]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Saved {len(clean)} events → {path}")


def load_events(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Self-test: run on nuScenes mini, print stats, save JSON
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help='Path to nuScenes dataset root '
                             '(e.g. /data/sets/nuscenes)')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--out',      default=str(_REPO_ROOT / 'PA_gen_v2' /
                                                   'output' / 'ghost_events_mini.json'))
    parser.add_argument('--lookback', type=int, default=LOOKBACK_FRAMES)
    parser.add_argument('--min_osz',  type=int, default=MIN_OSZ_OVERLAP)
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    events = mine_ghost_events(
        nusc,
        lookback_k=args.lookback,
        min_osz_overlap=args.min_osz,
        verbose=True,
    )

    save_events(events, args.out)

    from collections import Counter
    scene_counts = Counter(e['scene_token'] for e in events if e['label'] == 1)
    print("\nPositive events per scene:")
    for scene_tok, cnt in scene_counts.most_common():
        scene = nusc.get('scene', scene_tok)
        print(f"  {scene['name']}: {cnt}")
