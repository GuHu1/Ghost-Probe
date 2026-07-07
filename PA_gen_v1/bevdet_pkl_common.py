"""
bevdet_pkl_common.py
====================
Shared logic for generating BEVDet-format .pkl info files from nuScenes.
Used by:
  - create_bevdet_pkl_mini.py  (v1.0-mini)
  - create_bevdet_pkl_full.py  (v1.0-trainval)

This module centralises the `fill_info` and `main` logic so the two split
scripts only need to supply their version, dataroot/output paths, and the
appropriate train/val scene-name lists.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import List

from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


# ═══════════════════════════════════════════════════════════════════════════
#  Common config
# ═══════════════════════════════════════════════════════════════════════════
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
LIDAR = 'LIDAR_TOP'
N_SWEEPS = 9    # matches BEVDet default

NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

CAT2CLS = {
    'vehicle.car':                              'car',
    'vehicle.truck':                            'truck',
    'vehicle.construction':                     'construction_vehicle',
    'vehicle.bus.bendy':                        'bus',
    'vehicle.bus.rigid':                        'bus',
    'vehicle.trailer':                          'trailer',
    'movable_object.barrier':                   'barrier',
    'vehicle.motorcycle':                       'motorcycle',
    'vehicle.bicycle':                          'bicycle',
    'human.pedestrian.adult':                   'pedestrian',
    'human.pedestrian.child':                   'pedestrian',
    'human.pedestrian.construction_worker':     'pedestrian',
    'human.pedestrian.police_officer':          'pedestrian',
    'movable_object.trafficcone':               'traffic_cone',
}


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════
def get_lidar_velocity(nusc, ann_token):
    """Global-frame velocity (vx, vy) transformed to LiDAR frame."""
    ann = nusc.get('sample_annotation', ann_token)
    p, n = ann['prev'], ann['next']
    if not (p and n):
        return np.array([np.nan, np.nan])
    ap = nusc.get('sample_annotation', p)
    an = nusc.get('sample_annotation', n)
    t0 = nusc.get('sample', ap['sample_token'])['timestamp']
    t1 = nusc.get('sample', an['sample_token'])['timestamp']
    dt = (t1 - t0) * 1e-6
    if dt <= 0 or dt > 1.5:
        return np.array([np.nan, np.nan])
    vel_g = (np.array(an['translation']) - np.array(ap['translation'])) / dt
    return vel_g[:2]   # BEVDet stores global (vx, vy)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-sample info builder
# ═══════════════════════════════════════════════════════════════════════════
def fill_info(nusc, sample):
    """
    Build a single sample's BEVDet info dict.
    Format matches BEVDet's nuscenes_converter.py _fill_trainval_infos.
    """
    # ── LiDAR reference frame ──────────────────────────────────
    lidar_sd = nusc.get('sample_data', sample['data'][LIDAR])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    lidar_path = str(Path(lidar_sd['filename']))   # relative path

    e2g_r = np.array(lidar_pose['rotation'])
    e2g_t = np.array(lidar_pose['translation'])
    l2e_r = np.array(lidar_cs['rotation'])
    l2e_t = np.array(lidar_cs['translation'])

    # ── Sweeps ─────────────────────────────────────────────────
    sweeps = []
    tok_prev = lidar_sd['prev']
    while tok_prev and len(sweeps) < N_SWEEPS:
        sw = nusc.get('sample_data', tok_prev)
        sw_cs = nusc.get('calibrated_sensor', sw['calibrated_sensor_token'])
        sw_ep = nusc.get('ego_pose', sw['ego_pose_token'])
        sweeps.append({
            'data_path': str(Path(sw['filename'])),
            'type': LIDAR,
            'sample_data_token': sw['token'],
            'sensor2ego_translation': sw_cs['translation'],
            'sensor2ego_rotation': sw_cs['rotation'],
            'ego2global_translation': sw_ep['translation'],
            'ego2global_rotation': sw_ep['rotation'],
            'sensor2lidar_rotation':
                (Quaternion(lidar_cs['rotation']).inverse *
                 Quaternion(sw_cs['rotation'])).rotation_matrix,
            'sensor2lidar_translation':
                (Quaternion(lidar_cs['rotation']).inverse.rotate(
                    np.array(sw_ep['translation']) - np.array(lidar_pose['translation'])
                ) + Quaternion(lidar_cs['rotation']).inverse.rotate(
                    np.array(sw_cs['translation'])
                ) - np.array(lidar_cs['translation'])),
            'timestamp': sw['timestamp'],
        })
        tok_prev = sw['prev']

    # ── Cameras ────────────────────────────────────────────────
    cams = {}
    for cam in CAMERAS:
        cam_sd = nusc.get('sample_data', sample['data'][cam])
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])

        R_cam2ego = Quaternion(cam_cs['rotation']).rotation_matrix
        R_l2ego = Quaternion(lidar_cs['rotation']).rotation_matrix
        R_s2l = R_l2ego.T @ R_cam2ego
        t_cam_ego = np.array(cam_cs['translation'])
        t_lid_ego = np.array(lidar_cs['translation'])
        t_s2l = R_l2ego.T @ (t_cam_ego - t_lid_ego)

        cams[cam] = {
            'data_path': str(Path(cam_sd['filename'])),
            'type': cam,
            'sample_data_token': cam_sd['token'],
            'sensor2ego_translation': cam_cs['translation'],
            'sensor2ego_rotation': cam_cs['rotation'],
            'ego2global_translation': cam_pose['translation'],
            'ego2global_rotation': cam_pose['rotation'],
            'sensor2lidar_rotation': R_s2l,
            'sensor2lidar_translation': t_s2l,
            'cam_intrinsic': np.array(cam_cs['camera_intrinsic']),
            'timestamp': cam_sd['timestamp'],
        }

    # ── GT boxes ───────────────────────────────────────────────
    R_e2g = Quaternion(e2g_r).rotation_matrix
    R_l2e = Quaternion(l2e_r).rotation_matrix

    gt_boxes, gt_names, gt_vels, gt_pts, valid_flag = [], [], [], [], []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        inst = nusc.get('instance', ann['instance_token'])
        cat_name = nusc.get('category', inst['category_token'])['name']
        cls_name = CAT2CLS.get(cat_name, None)
        valid = cls_name is not None

        pos_g = np.array(ann['translation'])
        pos_e = R_e2g.T @ (pos_g - e2g_t)
        pos_l = R_l2e.T @ (pos_e - l2e_t)

        wlh = np.array(ann['size'])
        dims = np.array([wlh[1], wlh[0], wlh[2]])

        q_g = Quaternion(ann['rotation'])
        q_l2e = Quaternion(l2e_r)
        q_e2g = Quaternion(e2g_r)
        yaw = (q_l2e.inverse * q_e2g.inverse * q_g).yaw_pitch_roll[0]

        vel_g = get_lidar_velocity(nusc, ann_tok)

        gt_boxes.append([
            pos_l[0], pos_l[1], pos_l[2],
            dims[0], dims[1], dims[2],
            yaw,
            vel_g[0] if not np.isnan(vel_g[0]) else 0.,
            vel_g[1] if not np.isnan(vel_g[1]) else 0.,
        ])
        gt_names.append(cls_name if cls_name else cat_name)
        gt_vels.append(vel_g if not np.isnan(vel_g).any() else np.array([0., 0.]))
        gt_pts.append(ann['num_lidar_pts'])
        valid_flag.append(valid and ann['num_lidar_pts'] > 0)

    gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 9) if gt_boxes else np.zeros((0, 9), np.float32)

    return {
        'lidar_path': lidar_path,
        'token': sample['token'],
        'sweeps': sweeps,
        'cams': cams,
        'lidar2ego_translation': lidar_cs['translation'],
        'lidar2ego_rotation': lidar_cs['rotation'],
        'ego2global_translation': lidar_pose['translation'],
        'ego2global_rotation': lidar_pose['rotation'],
        'timestamp': sample['timestamp'],
        'gt_boxes': gt_boxes,
        'gt_names': np.array(gt_names),
        'gt_velocity': np.array(gt_vels, dtype=np.float32).reshape(-1, 2) if gt_vels else np.zeros((0, 2), np.float32),
        'num_lidar_pts': np.array(gt_pts, dtype=np.int32),
        'valid_flag': np.array(valid_flag, dtype=bool),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main(version: str, dataroot: str, out_dir: str,
         train_scenes: list, val_scenes: list,
         train_pkl_name: str, val_pkl_name: str):
    """
    Build BEVDet .pkl files for a given nuScenes version.

    Args:
        version:        nuScenes version string, e.g. 'v1.0-mini'.
        dataroot:       Path to the nuScenes root (should contain the meta data
                        and sensor files).
        out_dir:        Where to write the output .pkl files.
        train_scenes:   List of scene names in the training split.
        val_scenes:     List of scene names in the validation split.
        train_pkl_name: Output filename for the train split.
        val_pkl_name:   Output filename for the val split.
    """
    print(f"Loading NuScenes {version} from {dataroot}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    scene2split = {}
    for sc_name in train_scenes:
        scene2split[sc_name] = 'train'
    for sc_name in val_scenes:
        scene2split[sc_name] = 'val'

    train_infos, val_infos = [], []
    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        split = scene2split.get(scene['name'], None)
        if split is None:
            continue
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            info = fill_info(nusc, sample)
            if split == 'train':
                train_infos.append(info)
            else:
                val_infos.append(info)
            tok = sample['next']

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pkl = out / train_pkl_name
    val_pkl = out / val_pkl_name

    with open(train_pkl, 'wb') as f:
        pickle.dump({'infos': train_infos, 'metadata': {'version': version}}, f)
    with open(val_pkl, 'wb') as f:
        pickle.dump({'infos': val_infos, 'metadata': {'version': version}}, f)

    print("\nDone:")
    print(f"  train: {len(train_infos)} frames → {train_pkl}")
    print(f"  val:   {len(val_infos)} frames → {val_pkl}")
    print("\nCorresponding fields in BEVNeXt config:")
    print(f"  data_root = '{dataroot}/'")
    print(f"  ann_file  = '{train_pkl}'  # train")
    print(f"  ann_file  = '{val_pkl}'    # val")


def parse_and_run(
    version: str,
    dataroot: str,
    out_dir: str,
    train_scenes: List[str],
    val_scenes: List[str],
    train_pkl_name: str,
    val_pkl_name: str,
):
    """
    Thin CLI wrapper around `main`.

    The calling script (mini/full) supplies its split-specific defaults;
    the user can override --dataroot and --outdir on the command line
    without editing the file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=dataroot,
                        help='nuScenes root (should contain the metadata tables '
                             'and sensor files, or a `full/` subset)')
    parser.add_argument('--out_dir',  type=str, default=out_dir,
                        help='Where to write the output .pkl files')
    args = parser.parse_args()

    main(
        version=version,
        dataroot=args.dataroot,
        out_dir=args.out_dir,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        train_pkl_name=train_pkl_name,
        val_pkl_name=val_pkl_name,
    )
