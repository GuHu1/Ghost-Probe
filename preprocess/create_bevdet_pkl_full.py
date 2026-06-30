#!/usr/bin/env python3
"""
create_bevdet_pkl_full.py  —  v1.0-trainval 版
==========================================================
与 create_bevdet_pkl_mini.py 逻辑完全一致，
仅修改了以下配置：
  VERSION  = "v1.0-trainval"
  DATAROOT = "/data/jhc/pa_nuscenes/full"    ← 指向 trainval 的 full/ 目录
  导入的 split: train, val（而非 mini_train, mini_val）

输出：
  /data/jhc/pa_nuscenes/full/
    bevdet-nuscenes-trainval-train.pkl   (~28130 帧)
    bevdet-nuscenes-trainval-val.pkl     (~6019 帧)
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train, val    # ← trainval 专用

# ═══════════════════════════════════════════════════════════════
#  ★ 配置区（trainval 版）
# ═══════════════════════════════════════════════════════════════
DATAROOT = "/data/jhc/pa_nuscenes/full"   # ← full/ 目录（已含完整数据集）
OUT_DIR  = "/data/jhc/pa_nuscenes/full"
VERSION  = "v1.0-trainval"                # ← 唯一区别

CAMERAS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
LIDAR   = 'LIDAR_TOP'
N_SWEEPS = 9

NUSCENES_CLASSES = [
    'car','truck','construction_vehicle','bus','trailer',
    'barrier','motorcycle','bicycle','pedestrian','traffic_cone'
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

# ═══════════════════════════════════════════════════════════════
#  工具（与 mini 版完全相同）
# ═══════════════════════════════════════════════════════════════

def get_lidar_velocity(nusc, ann_token):
    ann=nusc.get('sample_annotation',ann_token); p,n=ann['prev'],ann['next']
    if not(p and n): return np.array([np.nan,np.nan])
    ap=nusc.get('sample_annotation',p); an=nusc.get('sample_annotation',n)
    t0=nusc.get('sample',ap['sample_token'])['timestamp']
    t1=nusc.get('sample',an['sample_token'])['timestamp']
    dt=(t1-t0)*1e-6
    if dt<=0 or dt>1.5: return np.array([np.nan,np.nan])
    vel_g=(np.array(an['translation'])-np.array(ap['translation']))/dt
    return vel_g[:2]

def fill_info(nusc, sample):
    lidar_sd   =nusc.get('sample_data',sample['data'][LIDAR])
    lidar_cs   =nusc.get('calibrated_sensor',lidar_sd['calibrated_sensor_token'])
    lidar_pose =nusc.get('ego_pose',lidar_sd['ego_pose_token'])
    lidar_path =str(Path(lidar_sd['filename']))
    e2g_r=np.array(lidar_pose['rotation']); e2g_t=np.array(lidar_pose['translation'])
    l2e_r=np.array(lidar_cs['rotation']);   l2e_t=np.array(lidar_cs['translation'])

    sweeps=[]; tok_prev=lidar_sd['prev']
    while tok_prev and len(sweeps)<N_SWEEPS:
        sw=nusc.get('sample_data',tok_prev); sw_cs=nusc.get('calibrated_sensor',sw['calibrated_sensor_token'])
        sw_ep=nusc.get('ego_pose',sw['ego_pose_token'])
        sweeps.append({'data_path':str(Path(sw['filename'])),'type':LIDAR,'sample_data_token':sw['token'],
                       'sensor2ego_translation':sw_cs['translation'],'sensor2ego_rotation':sw_cs['rotation'],
                       'ego2global_translation':sw_ep['translation'],'ego2global_rotation':sw_ep['rotation'],
                       'sensor2lidar_rotation':(Quaternion(lidar_cs['rotation']).inverse*Quaternion(sw_cs['rotation'])).rotation_matrix,
                       'sensor2lidar_translation':(Quaternion(lidar_cs['rotation']).inverse.rotate(
                           np.array(sw_ep['translation'])-np.array(lidar_pose['translation']))+
                           Quaternion(lidar_cs['rotation']).inverse.rotate(np.array(sw_cs['translation']))-np.array(lidar_cs['translation'])),
                       'timestamp':sw['timestamp']}); tok_prev=sw['prev']

    cams={}
    for cam in CAMERAS:
        cam_sd=nusc.get('sample_data',sample['data'][cam]); cam_cs=nusc.get('calibrated_sensor',cam_sd['calibrated_sensor_token'])
        cam_pose=nusc.get('ego_pose',cam_sd['ego_pose_token'])
        R_cam2ego=Quaternion(cam_cs['rotation']).rotation_matrix; R_l2ego=Quaternion(lidar_cs['rotation']).rotation_matrix
        R_s2l=R_l2ego.T@R_cam2ego; t_s2l=R_l2ego.T@(np.array(cam_cs['translation'])-np.array(lidar_cs['translation']))
        cams[cam]={'data_path':str(Path(cam_sd['filename'])),'type':cam,'sample_data_token':cam_sd['token'],
                   'sensor2ego_translation':cam_cs['translation'],'sensor2ego_rotation':cam_cs['rotation'],
                   'ego2global_translation':cam_pose['translation'],'ego2global_rotation':cam_pose['rotation'],
                   'sensor2lidar_rotation':R_s2l,'sensor2lidar_translation':t_s2l,
                   'cam_intrinsic':np.array(cam_cs['camera_intrinsic']),'timestamp':cam_sd['timestamp']}

    R_e2g=Quaternion(e2g_r).rotation_matrix; R_l2e=Quaternion(l2e_r).rotation_matrix
    gt_boxes,gt_names,gt_vels,gt_pts,valid_flag=[],[],[],[],[]
    for ann_tok in sample['anns']:
        ann=nusc.get('sample_annotation',ann_tok); cat_name=nusc.get('category',ann['category_token'])['name']
        cls_name=CAT2CLS.get(cat_name,None); valid=cls_name is not None
        pos_g=np.array(ann['translation']); pos_e=R_e2g.T@(pos_g-e2g_t); pos_l=R_l2e.T@(pos_e-l2e_t)
        wlh=np.array(ann['size']); dims=np.array([wlh[1],wlh[0],wlh[2]])
        q_g=Quaternion(ann['rotation']); yaw=(Quaternion(l2e_r).inverse*Quaternion(e2g_r).inverse*q_g).yaw_pitch_roll[0]
        vel_g=get_lidar_velocity(nusc,ann_tok)
        gt_boxes.append([pos_l[0],pos_l[1],pos_l[2],dims[0],dims[1],dims[2],yaw,
                         vel_g[0] if not np.isnan(vel_g[0]) else 0.,vel_g[1] if not np.isnan(vel_g[1]) else 0.])
        gt_names.append(cls_name if cls_name else cat_name)
        gt_vels.append(vel_g if not np.isnan(vel_g).any() else np.array([0.,0.]))
        gt_pts.append(ann['num_lidar_pts']); valid_flag.append(valid and ann['num_lidar_pts']>0)

    gt_boxes=np.array(gt_boxes,dtype=np.float32).reshape(-1,9) if gt_boxes else np.zeros((0,9),np.float32)
    return {'lidar_path':lidar_path,'token':sample['token'],'sweeps':sweeps,'cams':cams,
            'lidar2ego_translation':lidar_cs['translation'],'lidar2ego_rotation':lidar_cs['rotation'],
            'ego2global_translation':lidar_pose['translation'],'ego2global_rotation':lidar_pose['rotation'],
            'timestamp':sample['timestamp'],'gt_boxes':gt_boxes,'gt_names':np.array(gt_names),
            'gt_velocity':np.array(gt_vels,dtype=np.float32).reshape(-1,2) if gt_vels else np.zeros((0,2),np.float32),
            'num_lidar_pts':np.array(gt_pts,dtype=np.int32),'valid_flag':np.array(valid_flag,dtype=bool)}

# ═══════════════════════════════════════════════════════════════
#  主程序（与 mini 版逻辑相同，split 来源不同）
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Loading NuScenes {VERSION} from {DATAROOT} ...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    # trainval split 是 scene_name 列表
    scene2split = {}
    for sc_name in train: scene2split[sc_name] = 'train'
    for sc_name in val:   scene2split[sc_name] = 'val'

    train_infos, val_infos = [], []
    for scene in tqdm(nusc.scene, desc="处理 scenes"):
        split = scene2split.get(scene['name'], None)
        if split is None: continue
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            info   = fill_info(nusc, sample)
            if split=='train': train_infos.append(info)
            else:              val_infos.append(info)
            tok = sample['next']

    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    train_pkl = out / 'bevdet-nuscenes-trainval-train.pkl'
    val_pkl   = out / 'bevdet-nuscenes-trainval-val.pkl'
    with open(train_pkl,'wb') as f: pickle.dump({'infos':train_infos,'metadata':{'version':VERSION}},f)
    with open(val_pkl,'wb') as f:   pickle.dump({'infos':val_infos,  'metadata':{'version':VERSION}},f)

    print(f"\n生成完成：")
    print(f"  train: {len(train_infos)} 帧  → {train_pkl}")
    print(f"  val:   {len(val_infos)} 帧  → {val_pkl}")
    print(f"\nBEVNeXt config 对应字段：")
    print(f"  data_root = '{DATAROOT}/'")
    print(f"  ann_file  = '{DATAROOT}/bevdet-nuscenes-trainval-train.pkl'  # train")
    print(f"  ann_file  = '{DATAROOT}/bevdet-nuscenes-trainval-val.pkl'    # val")

if __name__ == '__main__':
    main()
