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
        'was_in_osz':     [bool, ...],  # per lookback frame: was V in OSZ?
        'label':          int,          # 1 = positive ghost event
    }

Negative samples are also generated: frames where a vehicle is visible
in the current frame AND was visible in all previous k frames (no OSZ
involvement).

Karpathy notes:
  - Print counts at every stage. Never trust a silent loop.
  - Assert coordinate transforms at known test cases.
  - Build lookup tables up front; don't query nuScenes in the inner loop.
"""

import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

import pyquaternion
from nuscenes.nuscenes import NuScenes

from osz_geometry import (
    get_osz_for_sample,
    bev_coords_to_pixel,
    BEV_SIZE, BEV_RANGE_M,
)


# ---------------------------------------------------------------------------
# nuScenes category filter — we only care about wheeled vehicles
# ---------------------------------------------------------------------------
VEHICLE_CATEGORIES = {
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
}

LOOKBACK_FRAMES   = 4    # k: how many past frames to inspect
MIN_OSZ_OVERLAP   = 1    # at least this many lookback frames must be in OSZ
                          # for a positive event (avoids false positives from
                          # vehicles that just temporarily went off-sensor)


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def _build_instance_category_map(nusc: NuScenes) -> Dict[str, str]:
    """
    Returns {instance_token: category_name} for all instances.
    Built once, queried in the inner loop.
    """
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
    Returns {sample_token: [annotation_dict, ...]} for VEHICLE annotations only.
    annotation_dict keys: instance_token, translation (ego frame), size, rotation.

    Coordinates: nuScenes annotations are in GLOBAL frame. We'll convert to
    ego frame per sample when needed, so here we store them in global frame
    and tag them with the ego pose token for later conversion.
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
            'instance_token':  ann['instance_token'],
            'translation_global': np.array(ann['translation'], dtype=np.float32),
            'size':            ann['size'],
        })

    total_anns = sum(len(v) for v in mapping.values())
    print(f"  [index] {total_anns} vehicle annotations across "
          f"{len(mapping)} samples.")
    return mapping


def _global_to_ego(translation_global: np.ndarray,
                   ego_translation: np.ndarray,
                   ego_rotation_q: pyquaternion.Quaternion) -> np.ndarray:
    """
    Transform a 3D point from global frame to ego vehicle frame.

    ego_translation: (3,) position of ego in global frame
    ego_rotation_q : rotation of ego in global frame (as Quaternion)
    """
    delta = translation_global - ego_translation
    # ego_rotation_q rotates global→ego: apply inverse rotation
    pt_ego = ego_rotation_q.inverse.rotate(delta)
    return pt_ego.astype(np.float32)


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


def _is_in_osz(pt_ego: np.ndarray, osz_mask: np.ndarray) -> bool:
    """
    Check whether a 3D ego-frame point falls inside the OSZ mask.
    Uses only the x-y plane (ignores z).
    """
    col, row = bev_coords_to_pixel(pt_ego[0], pt_ego[1])
    if col < 0 or col >= BEV_SIZE or row < 0 or row >= BEV_SIZE:
        return False
    return bool(osz_mask[row, col] > 0.5)


def _is_in_bev_range(pt_ego: np.ndarray) -> bool:
    """Check that a point is within our BEV grid bounds."""
    return (abs(pt_ego[0]) < BEV_RANGE_M and abs(pt_ego[1]) < BEV_RANGE_M)


# ---------------------------------------------------------------------------
# Core mining logic
# ---------------------------------------------------------------------------

def _get_lookback_samples(nusc: NuScenes,
                          sample_token: str,
                           k: int) -> List[str]:
    """
    Return the k samples BEFORE sample_token in the same scene,
    ordered oldest-first.  Returns fewer than k items if we're near
    the beginning of a scene.
    """
    tokens = []
    current = nusc.get('sample', sample_token)
    for _ in range(k):
        prev_tok = current['prev']
        if prev_tok == '':
            break
        tokens.append(prev_tok)
        current = nusc.get('sample', prev_tok)
    tokens.reverse()  # oldest first
    return tokens


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

    Returns list of event dicts (see module docstring for schema).
    """
    if scene_tokens is None:
        scene_tokens = [s['token'] for s in nusc.scene]

    print(f"\n{'='*60}")
    print(f"Mining ghost events across {len(scene_tokens)} scenes")
    print(f"  lookback_k={lookback_k}, min_osz_overlap={min_osz_overlap}")
    print(f"{'='*60}")

    # --- Build indices once ---
    inst_cat_map = _build_instance_category_map(nusc)
    ann_map      = _build_sample_annotations_map(nusc, inst_cat_map)

    positive_events: List[Dict] = []
    negative_events: List[Dict] = []

    # --- Scene loop ---
    for scene_token in tqdm(scene_tokens, desc='Scenes'):
        scene = nusc.get('scene', scene_token)

        # Collect all sample tokens in this scene (chronological order)
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

            # Compute OSZ for current frame t
            try:
                occ_grid, osz_mask_t = get_osz_for_sample(nusc, sample_token)
            except Exception as e:
                tqdm.write(f"  [WARN] OSZ failed for {sample_token}: {e}")
                continue

            # Ego pose at frame t
            ego_t, ego_q_t = _get_ego_pose(nusc, sample_token)

            # Get vehicle annotations in current frame
            anns_t = ann_map.get(sample_token, [])
            if not anns_t:
                continue

            # Get lookback sample tokens (t-k ... t-1)
            lookback_tokens = _get_lookback_samples(
                nusc, sample_token, lookback_k)
            if len(lookback_tokens) < lookback_k:
                # Not enough history — skip (near start of scene)
                continue

            # Pre-compute OSZ masks and ego poses for lookback frames
            lookback_data = []   # list of (osz_mask, ego_t, ego_q)
            valid_lookback = True
            for lb_tok in lookback_tokens:
                try:
                    _, osz_lb = get_osz_for_sample(nusc, lb_tok)
                    ego_lb, ego_q_lb = _get_ego_pose(nusc, lb_tok)
                    lookback_data.append((osz_lb, ego_lb, ego_q_lb))
                except Exception as e:
                    tqdm.write(f"  [WARN] OSZ failed for lookback "
                               f"{lb_tok}: {e}")
                    valid_lookback = False
                    break
            if not valid_lookback:
                continue

            # Build instance → last-known-global-position map for lookback
            # {instance_token: [global_xyz per lookback frame or None]}
            inst_lb_presence: Dict[str, List[Optional[np.ndarray]]] = {}
            for lb_tok in lookback_tokens:
                for ann in ann_map.get(lb_tok, []):
                    itok = ann['instance_token']
                    if itok not in inst_lb_presence:
                        inst_lb_presence[itok] = [None] * len(lookback_tokens)
                    lb_idx = lookback_tokens.index(lb_tok)
                    inst_lb_presence[itok][lb_idx] = \
                        ann['translation_global']

            # --- Instance loop at frame t ---
            for ann_t in anns_t:
                itok = ann_t['instance_token']

                # Convert current position to ego frame of frame t
                pt_ego_t = _global_to_ego(
                    ann_t['translation_global'], ego_t, ego_q_t)

                if not _is_in_bev_range(pt_ego_t):
                    continue

                # Is V visible (not in OSZ) at frame t?
                if _is_in_osz(pt_ego_t, osz_mask_t):
                    # Vehicle is STILL in OSZ at t — not an emergence
                    continue

                # --- Analyze lookback history of this instance ---
                lb_positions = inst_lb_presence.get(itok, [None] * lookback_k)

                was_in_osz_per_frame: List[bool] = []
                for lb_idx, (lb_xyz_global) in enumerate(lb_positions):
                    osz_lb, ego_lb, ego_q_lb = lookback_data[lb_idx]

                    if lb_xyz_global is None:
                        # Vehicle was not annotated in this lookback frame.
                        # Treat as "effectively in OSZ" (unseen = possibly hidden)
                        was_in_osz_per_frame.append(True)
                    else:
                        pt_ego_lb = _global_to_ego(lb_xyz_global,
                                                   ego_lb, ego_q_lb)
                        in_osz = _is_in_osz(pt_ego_lb, osz_lb)
                        was_in_osz_per_frame.append(in_osz)

                n_osz_frames = sum(was_in_osz_per_frame)

                event_base = {
                    'scene_token':     scene_token,
                    'emerge_sample':   sample_token,
                    'instance_token':  itok,
                    'emerge_bev_xy':   (float(pt_ego_t[0]),
                                        float(pt_ego_t[1])),
                    'lookback_tokens': lookback_tokens,
                    'was_in_osz':      was_in_osz_per_frame,
                    'n_osz_frames':    n_osz_frames,
                }

                if n_osz_frames >= min_osz_overlap:
                    # ✅ POSITIVE: vehicle was hidden in OSZ, now emerged
                    positive_events.append({**event_base, 'label': 1})
                else:
                    # Candidate negative: visible throughout
                    if n_osz_frames == 0:
                        negative_events.append({**event_base, 'label': 0})

    # --- Balance negative set ---
    n_pos = len(positive_events)
    n_neg_target = int(n_pos * neg_ratio)
    if len(negative_events) > n_neg_target:
        rng = np.random.default_rng(seed=42)
        neg_idx = rng.choice(len(negative_events), n_neg_target, replace=False)
        negative_events = [negative_events[i] for i in neg_idx]

    all_events = positive_events + negative_events

    print(f"\n{'='*60}")
    print(f"Mining complete.")
    print(f"  Positive events (ghost emergence): {len(positive_events)}")
    print(f"  Negative events (always visible):  {len(negative_events)}")
    print(f"  Total events:                      {len(all_events)}")
    print(f"{'='*60}\n")

    # Sanity checks on output
    assert len(positive_events) > 0, \
        "No positive events found! Check OSZ coverage or lookback logic."
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
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, (list, tuple)):  return [_convert(x) for x in obj]
        return obj

    clean = [{k: _convert(v) for k, v in e.items()} for e in events]
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
    parser.add_argument('--dataroot', default='/data/nuscenes')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--out',      default='/home/claude/phantom_agent/'
                                               'ghost_events_mini.json')
    parser.add_argument('--lookback', type=int, default=LOOKBACK_FRAMES)
    parser.add_argument('--min_osz',  type=int, default=MIN_OSZ_OVERLAP)
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    events = mine_ghost_events(
        nusc,
        lookback_k=args.lookback,
        min_osz_overlap=args.min_osz,
        verbose=True,
    )

    save_events(events, args.out)

    # Quick per-scene breakdown
    from collections import Counter
    scene_counts = Counter(e['scene_token'] for e in events if e['label'] == 1)
    print("\nPositive events per scene:")
    for scene_tok, cnt in scene_counts.most_common():
        scene = nusc.get('scene', scene_tok)
        print(f"  {scene['name']}: {cnt}")
