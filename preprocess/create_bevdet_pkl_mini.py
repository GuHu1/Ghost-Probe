#!/usr/bin/env python3
"""
create_bevdet_pkl_mini.py  —  v1.0-mini 版
==========================================================
不依赖 BEVNeXt 内部的 create_data_bevdet.py，
直接生成 BEVDet 格式的 pkl，可被 NuScenesDataset 直接加载。

运行（在任意目录，不需要进入 BEVNeXt）：
  python create_bevdet_pkl_mini.py

输出：
  /data/jhc/pa_nuscenes/full/
    bevdet-nuscenes-mini-train.pkl
    bevdet-nuscenes-mini-val.pkl
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val    # ← mini 专用

# ═══════════════════════════════════════════════════════════════
#  配置（与 create_pa_labels_mini.py 保持一致）
# ═══════════════════════════════════════════════════════════════
DATAROOT = "/data/jhc/pa_nuscenes/full"   # ← 指向已复制好的 full/ 目录
OUT_DIR  = "/data/jhc/pa_nuscenes/full"
VERSION  = "v1.0-mini"

CAMERAS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
LIDAR   = 'LIDAR_TOP'
N_SWEEPS = 9    # 与 BEVDet 默认一致

# nuScenes 检测类别（与 BEVDet 官方一致）
NUSCENES_CLASSES = [
    'car','truck','construction_vehicle','bus','trailer',
    'barrier','motorcycle','bicycle','pedestrian','traffic_cone'
]
# nuScenes category_name → BEVDet class_name
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

# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def quat2mat(q_wxyz):
    return Quaternion(q_wxyz).rotation_matrix

def get_lidar_velocity(nusc, ann_token):
    """全局系速度 (vx,vy)，变换到 LiDAR 系"""
    ann = nusc.get('sample_annotation', ann_token)
    p, n = ann['prev'], ann['next']
    if not (p and n):
        return np.array([np.nan, np.nan])
    ap=nusc.get('sample_annotation',p); an=nusc.get('sample_annotation',n)
    t0=nusc.get('sample',ap['sample_token'])['timestamp']
    t1=nusc.get('sample',an['sample_token'])['timestamp']
    dt=(t1-t0)*1e-6
    if dt<=0 or dt>1.5: return np.array([np.nan,np.nan])
    vel_g=(np.array(an['translation'])-np.array(ap['translation']))/dt  # (3,) global
    return vel_g[:2]   # BEVDet 存全局速度 (vx,vy)

# ═══════════════════════════════════════════════════════════════
#  单帧 info 构建
# ═══════════════════════════════════════════════════════════════

def fill_info(nusc, sample):
    """
    构建单个 sample 的 BEVDet info dict。
    格式与 BEVDet nuscenes_converter.py 的 _fill_trainval_infos 完全一致。
    """
    # ── LiDAR 参考帧 ───────────────────────────────────────────
    lidar_sd    = nusc.get('sample_data', sample['data'][LIDAR])
    lidar_cs    = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    lidar_path  = str(Path(lidar_sd['filename']))   # 相对路径

    # Global → Ego → LiDAR
    e2g_r = np.array(lidar_pose['rotation'])       # ego2global quaternion
    e2g_t = np.array(lidar_pose['translation'])
    l2e_r = np.array(lidar_cs['rotation'])         # lidar2ego quaternion
    l2e_t = np.array(lidar_cs['translation'])

    # ── Sweeps ─────────────────────────────────────────────────
    sweeps = []
    tok_prev = lidar_sd['prev']
    while tok_prev and len(sweeps) < N_SWEEPS:
        sw     = nusc.get('sample_data', tok_prev)
        sw_cs  = nusc.get('calibrated_sensor', sw['calibrated_sensor_token'])
        sw_ep  = nusc.get('ego_pose', sw['ego_pose_token'])
        sweeps.append({
            'data_path':              str(Path(sw['filename'])),
            'type':                   LIDAR,
            'sample_data_token':      sw['token'],
            'sensor2ego_translation': sw_cs['translation'],
            'sensor2ego_rotation':    sw_cs['rotation'],
            'ego2global_translation': sw_ep['translation'],
            'ego2global_rotation':    sw_ep['rotation'],
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

    # ── 相机 ────────────────────────────────────────────────────
    cams = {}
    for cam in CAMERAS:
        cam_sd   = nusc.get('sample_data', sample['data'][cam])
        cam_cs   = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])

        # sensor2lidar（BEVDet 需要）
        # R_s2l = R_l2e^{-1} * R_e2cam^{-1} * R_cam
        # 简化：同一时刻 ego 姿态假设与 lidar ego 相同（标准做法）
        R_cam2ego = Quaternion(cam_cs['rotation']).rotation_matrix
        R_l2ego   = Quaternion(lidar_cs['rotation']).rotation_matrix
        R_s2l     = R_l2ego.T @ R_cam2ego
        t_cam_ego = np.array(cam_cs['translation'])
        t_lid_ego = np.array(lidar_cs['translation'])
        t_s2l     = R_l2ego.T @ (t_cam_ego - t_lid_ego)

        cams[cam] = {
            'data_path':              str(Path(cam_sd['filename'])),
            'type':                   cam,
            'sample_data_token':      cam_sd['token'],
            'sensor2ego_translation': cam_cs['translation'],
            'sensor2ego_rotation':    cam_cs['rotation'],
            'ego2global_translation': cam_pose['translation'],
            'ego2global_rotation':    cam_pose['rotation'],
            'sensor2lidar_rotation':  R_s2l,
            'sensor2lidar_translation': t_s2l,
            'cam_intrinsic':          np.array(cam_cs['camera_intrinsic']),
            'timestamp':              cam_sd['timestamp'],
        }

    # ── GT boxes ────────────────────────────────────────────────
    ann_tokens  = sample['anns']
    gt_boxes, gt_names, gt_vels, gt_pts, valid_flag = [], [], [], [], []

    # Global→Ego→LiDAR 变换矩阵
    R_e2g = Quaternion(e2g_r).rotation_matrix
    R_l2e = Quaternion(l2e_r).rotation_matrix

    for ann_tok in ann_tokens:
        ann      = nusc.get('sample_annotation', ann_tok)
        inst     = nusc.get('instance', ann['instance_token'])
        cat_name = nusc.get('category', inst['category_token'])['name']
        cls_name = CAT2CLS.get(cat_name, None)
        valid    = cls_name is not None

        # 位置：global → LiDAR
        pos_g = np.array(ann['translation'])
        pos_e = R_e2g.T @ (pos_g - e2g_t)
        pos_l = R_l2e.T @ (pos_e - l2e_t)

        # 尺寸：nuScenes [w,l,h] → BEVDet [l,w,h]
        wlh   = np.array(ann['size'])  # [w,l,h]
        dims  = np.array([wlh[1], wlh[0], wlh[2]])  # [l,w,h]

        # Yaw：global → LiDAR
        q_g   = Quaternion(ann['rotation'])
        q_l2e = Quaternion(l2e_r)
        q_e2g = Quaternion(e2g_r)
        yaw   = (q_l2e.inverse * q_e2g.inverse * q_g).yaw_pitch_roll[0]

        # 速度（全局系）
        vel_g = get_lidar_velocity(nusc, ann_tok)

        gt_boxes.append([pos_l[0],pos_l[1],pos_l[2],
                         dims[0],dims[1],dims[2],
                         yaw,
                         vel_g[0] if not np.isnan(vel_g[0]) else 0.,
                         vel_g[1] if not np.isnan(vel_g[1]) else 0.])
        gt_names.append(cls_name if cls_name else cat_name)
        gt_vels.append(vel_g if not np.isnan(vel_g).any() else np.array([0.,0.]))
        gt_pts.append(ann['num_lidar_pts'])
        valid_flag.append(valid and ann['num_lidar_pts']>0)

    gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1,9) if gt_boxes else np.zeros((0,9),np.float32)
    gt_names = np.array(gt_names)
    gt_vels  = np.array(gt_vels, dtype=np.float32).reshape(-1,2) if gt_vels else np.zeros((0,2),np.float32)
    gt_pts   = np.array(gt_pts, dtype=np.int32)
    valid_flag = np.array(valid_flag, dtype=bool)

    return {
        'lidar_path':            lidar_path,
        'token':                 sample['token'],
        'sweeps':                sweeps,
        'cams':                  cams,
        'lidar2ego_translation': lidar_cs['translation'],
        'lidar2ego_rotation':    lidar_cs['rotation'],
        'ego2global_translation': lidar_pose['translation'],
        'ego2global_rotation':    lidar_pose['rotation'],
        'timestamp':             sample['timestamp'],
        'gt_boxes':              gt_boxes,
        'gt_names':              gt_names,
        'gt_velocity':           gt_vels,
        'num_lidar_pts':         gt_pts,
        'valid_flag':            valid_flag,
    }


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Loading NuScenes {VERSION} from {DATAROOT}...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    # mini_train / mini_val 是 scene_name 列表，需转为 sample_token 集合
    scene2split = {}
    for sc_name in mini_train: scene2split[sc_name] = 'train'
    for sc_name in mini_val:   scene2split[sc_name] = 'val'

    train_infos, val_infos = [], []
    for scene in tqdm(nusc.scene, desc="处理 scenes"):
        split = scene2split.get(scene['name'], None)
        if split is None:
            continue   # 不在 mini split 中（理论上不会发生）
        # 遍历该 scene 所有 sample
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            info   = fill_info(nusc, sample)
            if split == 'train': train_infos.append(info)
            else:                val_infos.append(info)
            tok = sample['next']

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    train_pkl = out / 'bevdet-nuscenes-mini-train.pkl'
    val_pkl   = out / 'bevdet-nuscenes-mini-val.pkl'

    with open(train_pkl,'wb') as f:
        pickle.dump({'infos': train_infos, 'metadata': {'version': VERSION}}, f)
    with open(val_pkl,'wb') as f:
        pickle.dump({'infos': val_infos,   'metadata': {'version': VERSION}}, f)

    print(f"\n生成完成：")
    print(f"  train: {len(train_infos)} 帧  → {train_pkl}")
    print(f"  val:   {len(val_infos)} 帧  → {val_pkl}")
    print(f"\nBEVNeXt config 中对应字段：")
    try:
        ann_file_display = train_pkl.relative_to(Path.cwd())
    except ValueError:
        ann_file_display = train_pkl
    print(f"  ann_file = '{ann_file_display}'")


if __name__ == '__main__':
    main()
