#!/usr/bin/env python3
"""
pa_visible.py
-------------
Visualize strongly-occluded Phantom-Agent targets directly from nuScenes
annotations (visibility_token <= 1) without temporal tracking.

For each sample we:
  - find PA-category annotations with visibility <= 1,
  - project their 3D boxes onto CAM_FRONT / left / right,
  - overlay visibility level, category, and current velocity.

Output (under OUTDIR_BASE):
    occluded_vis/          visualization images
    occluded_labels.pkl    {sample_token: [label_dict]}
    occluded_tokens.txt
    stats.txt
"""

import pickle, shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib
matplotlib.use('Agg')  # headless-safe backend; must be set before pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as _fm

# Font setup for CJK labels (falls back to DejaVu Sans if unavailable)
_CJK_FONT = 'Noto Sans CJK JP'
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
if _CJK_FONT in _available_fonts:
    matplotlib.rcParams['font.family'] = _CJK_FONT
    matplotlib.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    print(f"[WARN] {_CJK_FONT} not found, using DejaVu Sans (ASCII only)")
matplotlib.rcParams['axes.unicode_minus'] = False
from PIL import Image

# Default paths — override with --dataroot / --outdir_base on the CLI.
DATAROOT    = "/data/sets/nuscenes"
OUTDIR_BASE = "./output/pa_visible"
VERSION     = "v1.0-mini"

CAMERAS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
LIDAR   = 'LIDAR_TOP'

VIS_OCCLUDED = 1   # include annotations with visibility <= this value

PA_CATEGORY_MAP = {
    'vehicle.car':          (0,'Vehicle'),
    'vehicle.truck':        (0,'Vehicle'),
    'vehicle.bus.bendy':    (0,'Vehicle'),
    'vehicle.bus.rigid':    (0,'Vehicle'),
    'vehicle.construction': (0,'Vehicle'),
    'vehicle.trailer':      (0,'Vehicle'),
    'human.pedestrian.adult':       (1,'Pedestrian'),
    'human.pedestrian.child':       (1,'Pedestrian'),
    'human.pedestrian.construction_worker': (1,'Pedestrian'),
    'human.pedestrian.police_officer':      (1,'Pedestrian'),
    'vehicle.bicycle':      (2,'Cyclist'),
    'vehicle.motorcycle':   (2,'Cyclist'),
}
PA_DIST_THRESHOLD = 100.0   # only consider PA within this radius of ego
PA_SIZE_PRIOR = {           # fixed size priors [l, w, h] in metres
    0: [4.5, 1.8, 1.5],
    1: [0.5, 0.5, 1.7],
    2: [1.8, 0.6, 1.5],
}
PA_COLOR = {0:'#FF5555', 1:'#55FF55', 2:'#5599FF'}
PA_NAME  = {0:'Vehicle',  1:'Pedestrian', 2:'Cyclist'}

# ── Utilities ────────────────────────────────────────────────────────

def make_tf(trans: list, rot: list, inv: bool = False) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation + quaternion rotation.

    Args:
        trans: [x, y, z] translation vector in metres.
        rot:   [w, x, y, z] quaternion rotation.
        inv:   If True, return the inverse transform instead.

    Returns:
        (4, 4) float64 array.
    """
    m = np.eye(4); m[:3,:3]=R; m[:3,3]=np.array(trans)
    return np.linalg.inv(m) if inv else m

def get_velocity(nusc: NuScenes, ann_token: str) -> np.ndarray:
    """Estimate velocity via central/backward/forward finite difference.

    Prefers central difference when both prev and next annotations exist.
    Returns [nan, nan, nan] if fewer than 2 annotations or time gap > 1.5s.

    Returns:
        (3,) float64 — global-frame velocity vector [vx, vy, vz] in m/s.
    """
    ann = nusc.get('sample_annotation', ann_token)
    p, n = ann['prev'], ann['next']
    if p and n:
        ap = nusc.get('sample_annotation', p)
        an = nusc.get('sample_annotation', n)
        t0 = nusc.get('sample', ap['sample_token'])['timestamp']
        t1 = nusc.get('sample', an['sample_token'])['timestamp']
        d0, d1 = np.array(ap['translation']), np.array(an['translation'])
    elif n:
        an = nusc.get('sample_annotation', n)
        t0 = nusc.get('sample', ann['sample_token'])['timestamp']
        t1 = nusc.get('sample', an['sample_token'])['timestamp']
        d0, d1 = np.array(ann['translation']), np.array(an['translation'])
    elif p:
        ap = nusc.get('sample_annotation', p)
        t0 = nusc.get('sample', ap['sample_token'])['timestamp']
        t1 = nusc.get('sample', ann['sample_token'])['timestamp']
        d0, d1 = np.array(ap['translation']), np.array(ann['translation'])
    else:
        return np.full(3, np.nan)
    dt = (t1 - t0) * 1e-6
    return (d1 - d0) / dt if 0 < dt <= 1.5 else np.full(3, np.nan)


def g2l(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """Build the global-to-LiDAR 4x4 transform for a given sample.

    Returns:
        (4, 4) float64 homogeneous transform matrix.
    """
    sam = nusc.get('sample', sample_token)
    sd  = nusc.get('sample_data', sam['data'][LIDAR])
    cs  = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep  = nusc.get('ego_pose', sd['ego_pose_token'])
    return np.linalg.inv(
        make_tf(ep['translation'], ep['rotation']) @
        make_tf(cs['translation'], cs['rotation'])
    )

def _project_vel_arrow(
    pos_global: list, vel_global: list, ep: dict, cs: dict,
    K: np.ndarray, W: int, H: int, scale: float = 1.5
) -> Optional[Tuple[float, float, float, float]]:
    """Project a global-frame velocity vector onto the camera image plane.

    Returns:
        (x0, y0, x1, y1) pixel coords of arrow base and tip, or None if
        the arrow is too short, behind camera, or off-screen.
    """
    def g2cam(p):
        p = np.array(p, dtype=float)
        p -= np.array(ep['translation'])
        p  = Quaternion(ep['rotation']).inverse.rotate(p)
        p -= np.array(cs['translation'])
        p  = Quaternion(cs['rotation']).inverse.rotate(p)
        return p
    p0c = g2cam(pos_global)
    p1c = g2cam(np.array(pos_global) + np.array(vel_global) * scale)
    if p0c[2] <= 0.1 or p1c[2] <= 0.1: return None
    def proj(p): uv = K @ p; return uv[0]/uv[2], uv[1]/uv[2]
    x0, y0 = proj(p0c); x1, y1 = proj(p1c)
    if np.hypot(x1-x0, y1-y0) < 3: return None
    if not (-100 <= x0 <= W+100 and -100 <= y0 <= H+100): return None
    return x0, y0, x1, y1

# ── Core: collect occluded-frame annotations ─────────────────────────

def collect_occluded_labels(nusc):
    """
    Collect occluded PA-category annotations for all samples.

    For each sample, find annotations with visibility <= VIS_OCCLUDED,
    directly from nuScenes current-frame annotations. No temporal tracking.
    """
    labels = {}   # {sample_token: [label_dict, ...]}

    for sample in tqdm(nusc.sample, desc="Scanning samples"):
        st = sample['token']
        frame_labels = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            vis = int(ann['visibility_token'])

            if vis > VIS_OCCLUDED:
                continue   # not strongly occluded

            # category filter
            inst = nusc.get('instance', ann['instance_token'])
            cat  = nusc.get('category', inst['category_token'])['name']
            if cat not in PA_CATEGORY_MAP:
                continue   # drop traffic_cone / barrier

            pa_type, pa_type_str = PA_CATEGORY_MAP[cat]
            vel = get_velocity(nusc, ann_token)

            label = {
                'ann_token':   ann_token,
                'instance_token': ann['instance_token'],
                'category':    cat,
                'pa_type':     pa_type,
                'pa_type_str': pa_type_str,
                # current-frame annotation from nuScenes
                'translation': list(ann['translation']),
                'rotation':    list(ann['rotation']),
                'size':        list(ann['size']),
                'visibility':  vis,
                # current-frame velocity (finite-difference estimate)
                'velocity':    (vel.tolist() if not np.isnan(vel).any()
                                else [np.nan, np.nan, np.nan]),
                'num_lidar_pts': ann['num_lidar_pts'],
            }
            frame_labels.append(label)

        if frame_labels:
            labels[st] = frame_labels

    return labels   # {sample_token: [label_dict]}

# ── Visualization ────────────────────────────────────────────────────

def _clip_line(x1, y1, x2, y2, W, H):
    """
    Cohen-Sutherland line clip against [0, W] × [0, H].
    Returns clipped endpoints, or None if the line is entirely outside.
    """
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < 0:   c |= LEFT
        elif x > W: c |= RIGHT
        if y < 0:   c |= TOP       # image y points downward
        elif y > H: c |= BOTTOM
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    while True:
        if not (c1 | c2):       # fully inside
            return x1, y1, x2, y2
        if c1 & c2:             # fully outside on the same side
            return None
        # pick the endpoint that lies outside
        c_out = c1 if c1 else c2
        if c_out & BOTTOM:
            x = x1 + (x2-x1)*(H-y1)/(y2-y1) if y2!=y1 else x1
            y = float(H)
        elif c_out & TOP:
            x = x1 + (x2-x1)*(0-y1)/(y2-y1) if y2!=y1 else x1
            y = 0.0
        elif c_out & RIGHT:
            y = y1 + (y2-y1)*(W-x1)/(x2-x1) if x2!=x1 else y1
            x = float(W)
        else:  # LEFT
            y = y1 + (y2-y1)*(0-x1)/(x2-x1) if x2!=x1 else y1
            x = 0.0
        if c_out == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)


def visualize_occluded_frame_multicam(nusc, sample_token, frame_labels,
                                       save_path, dataroot: str = DATAROOT):
    """
    Multi-camera visualization: CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT.

      - distance filter (PA_DIST_THRESHOLD)
      - fixed size priors instead of raw annotation sizes
      - thin boxes (lw=0.7) to reduce clutter
      - yellow velocity arrows
      - visibility = current-frame value
    """
    sample   = nusc.get('sample', sample_token)
    cam_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    # ego-distance filter
    G2L = g2l(nusc, sample_token)
    def dist_ego(t): return float(np.linalg.norm((G2L @ np.append(t,1.))[:2]))
    near_labels = [l for l in frame_labels if dist_ego(l['translation']) <= PA_DIST_THRESHOLD]

    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    fig.patch.set_facecolor('#0f0f1e')

    for ax_i, cam_ch in enumerate(cam_list):
        ax = axes[ax_i]
        cam_sd   = nusc.get('sample_data', sample['data'][cam_ch])
        img_path = Path(dataroot) / cam_sd['filename']
        img = (Image.open(img_path).convert('RGB')
               if img_path.exists()
               else Image.new('RGB', (1600, 900), (20, 20, 30)))
        W, H = img.width, img.height
        ax.imshow(img, extent=[0, W, H, 0])

        cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        K  = np.array(cs['camera_intrinsic'])

        drawn = 0
        for lbl in near_labels:
            pa_type = lbl['pa_type']
            color   = PA_COLOR[pa_type]

            # fixed size prior (nuScenes Box expects [w, l, h])
            prior_lwh = PA_SIZE_PRIOR[pa_type]
            prior_wlh = [prior_lwh[1], prior_lwh[0], prior_lwh[2]]

            box = Box(lbl['translation'], prior_wlh, Quaternion(lbl['rotation']))
            box.translate(-np.array(ep['translation']))
            box.rotate(Quaternion(ep['rotation']).inverse)
            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)

            corners = box.corners()
            if np.any(corners[2, :] <= 0.1):
                continue

            pts = view_points(corners, K, normalize=True)
            x2d, y2d = pts[0, :], pts[1, :]
            margin = 50
            if (np.all(x2d < -margin) or np.all(x2d > W+margin) or
                    np.all(y2d < -margin) or np.all(y2d > H+margin)):
                continue

            # thin wireframe (lw=0.7)
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for s, e in edges:
                cl = _clip_line(float(x2d[s]),float(y2d[s]),float(x2d[e]),float(y2d[e]),W,H)
                if cl is not None:
                    ax.plot([cl[0],cl[2]], [cl[1],cl[3]],
                            color=color, lw=0.7, alpha=0.75, zorder=3)

            in_view = (x2d>=0)&(x2d<=W)&(y2d>=0)&(y2d<=H)
            if not in_view.any():
                continue
            cx = float(np.mean(x2d[in_view]))
            cy = float(np.clip(np.min(y2d[in_view])-18, 5, H-30))

            # compact label text
            vel = lbl['velocity']
            vx, vy = float(vel[0]), float(vel[1])
            spd = np.hypot(vx, vy) if not np.isnan(vx) else float('nan')
            spd_str = f"{spd:.1f}" if not np.isnan(spd) else "nan"
            ax.text(cx, cy,
                    f"{PA_NAME[pa_type]} vis={lbl['visibility']} V={spd_str}m/s",
                    color='white', fontsize=6.5, ha='center',
                    bbox=dict(fc=color, alpha=0.65, pad=1.2, ec='none', boxstyle='round'),
                    zorder=5)

            # yellow velocity arrow
            if not np.isnan(vx) and np.hypot(vx,vy) > 0.1:
                arr = _project_vel_arrow(lbl['translation'],[vx,vy,0.],ep,cs,K,W,H,scale=1.5)
                if arr is not None:
                    ax.annotate('', xy=(arr[2],arr[3]), xytext=(arr[0],arr[1]),
                                arrowprops=dict(arrowstyle='->', color='#FFD700',
                                               lw=1.0, mutation_scale=9), zorder=6)
            drawn += 1

        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f'{cam_ch}  |  within {PA_DIST_THRESHOLD:.0f}m: {drawn} PAs  '
                     f'thin=size prior  yellow=V',
                     color='#aaaaff', fontsize=9)
        ax.axis('off')

    legend_elems = [
        Line2D([0],[0], color=PA_COLOR[t], lw=2, label=f'{PA_NAME[t]} vis<=1')
        for t in [0,1,2]
    ] + [Line2D([0],[0], color='#FFD700', lw=2, label='current velocity V')]
    fig.legend(handles=legend_elems, loc='lower center',
               facecolor='#1a1a2e', labelcolor='white',
               fontsize=9, ncol=4, framealpha=0.9)

    n_total = len(near_labels)
    n_veh = sum(1 for l in near_labels if l['pa_type']==0)
    n_ped = sum(1 for l in near_labels if l['pa_type']==1)
    n_cyc = sum(1 for l in near_labels if l['pa_type']==2)
    fig.suptitle(
        f"Token: {sample_token[:20]}   "
        f"occluded targets within {PA_DIST_THRESHOLD:.0f}m: {n_total}  "
        f"(V={n_veh} P={n_ped} C={n_cyc})\n"
        f"3D box = size prior  vis = current frame  yellow arrow = velocity",
        color='white', fontsize=9.5
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=110, facecolor='#0f0f1e', bbox_inches='tight')
    plt.close()


def visualize_bev_occluded(nusc, sample_token, frame_labels, save_path):
    """
    BEV overview: show all occluded targets in the current frame (ego frame).
    """
    sample = nusc.get('sample', sample_token)
    sd     = nusc.get('sample_data', sample['data'][LIDAR])
    cs     = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep     = nusc.get('ego_pose', sd['ego_pose_token'])

    # global -> ego rotation only
    R_g2e = Quaternion(ep['rotation']).rotation_matrix.T
    T_ego = np.array(ep['translation'])
    R_l2e = Quaternion(cs['rotation']).rotation_matrix

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor('#0f0f1e')
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.axhline(0,color='white',lw=0.5,alpha=0.3)
    ax.axvline(0,color='white',lw=0.5,alpha=0.3)

    for lbl in frame_labels:
        color = PA_COLOR[lbl['pa_type']]
        # global -> ego
        pos_g  = np.array(lbl['translation'])
        pos_e  = R_g2e @ (pos_g - T_ego)
        # ego -> LiDAR (approximately ego for BEV top-down)
        pos_l  = R_l2e.T @ (pos_e - np.array(cs['translation']))
        x, y   = pos_l[0], pos_l[1]

        # rotated BEV rectangle
        q_g   = Quaternion(lbl['rotation'])
        q_e2l = Quaternion(cs['rotation']).inverse
        q_g2e = Quaternion(ep['rotation']).inverse
        yaw   = (q_e2l * q_g2e * q_g).yaw_pitch_roll[0]
        l_sz, w_sz = lbl['size'][1], lbl['size'][0]
        corners = np.array([
            [-l_sz/2,-w_sz/2],[l_sz/2,-w_sz/2],
            [l_sz/2, w_sz/2], [-l_sz/2, w_sz/2]
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R2 = np.array([[c,-s],[s,c]])
        corners = (R2 @ corners.T).T + np.array([x, y])
        poly = plt.Polygon(corners, fill=False, edgecolor=color, lw=2, alpha=0.9)
        ax.add_patch(poly)
        ax.fill(*zip(*corners), color=color, alpha=0.2)
        ax.annotate(
            f"vis={lbl['visibility']}\n{PA_NAME[lbl['pa_type']]}",
            (x, y), textcoords='offset points', xytext=(4,4),
            color='white', fontsize=6.5,
            bbox=dict(fc=color, alpha=0.6, pad=1, ec='none')
        )

    ax.plot(0, 0, 'w^', ms=14, zorder=8, label='Ego')
    ax.legend(handles=[
        Line2D([0],[0], color=PA_COLOR[t], lw=3, label=f'{PA_NAME[t]} vis<=1')
        for t in [0,1,2]] + [
        Line2D([0],[0], marker='^', color='white', ms=10, label='Ego', ls='none')
    ], fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper right')
    ax.set_xlabel('X (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.set_title(f'BEV — occluded PA target locations\n'
                 f'Token: {sample_token[:20]}', color='#aaaaff', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, facecolor='#0f0f1e', bbox_inches='tight')
    plt.close()

# ── File copy ────────────────────────────────────────────────────────

def copy_files(
    nusc: NuScenes, token: str, dataroot: str = DATAROOT,
    outdir_base: str = OUTDIR_BASE
) -> None:
    """Copy camera images for one sample to the output tree, preserving layout."""
    sam  = nusc.get('sample', token)
    for ch in CAMERAS:
        sd  = nusc.get('sample_data', sam['data'][ch])
        src = Path(dataroot)/sd['filename']; dst = base/sd['filename']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and src.exists(): shutil.copy2(src, dst)
    sd  = nusc.get('sample_data', sam['data'][LIDAR])
    src = Path(dataroot)/sd['filename']; dst = base/sd['filename']
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and src.exists(): shutil.copy2(src, dst)

# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run occluded PA target visualization: collect visibility<=1 annotations,
    generate multi-camera and BEV overview images, save labels as pkl."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',    type=str, default=DATAROOT,
                        help='nuScenes dataset root')
    parser.add_argument('--outdir_base', type=str, default=OUTDIR_BASE,
                        help='Base output directory')
    parser.add_argument('--version',     type=str, default=VERSION,
                        help='nuScenes version')
    parser.add_argument('--vis_n',       type=int, default=10,
                        help='可视化帧数（-1=全部）')
    parser.add_argument('--bev',         action='store_true',
                        help='同时生成 BEV 俯视图')
    parser.add_argument('--no_copy',     action='store_true',
                        help='不复制传感器文件')
    args = parser.parse_args()

    dataroot = args.dataroot
    outdir_base = args.outdir_base
    version = args.version

    out = Path(outdir_base)
    vis_dir = out/'occluded_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"NuScenes {version} loaded from {dataroot}: "
          f"{len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    print("\n[Version B] 扫描所有帧中的遮挡 PA 目标标注...")
    labels = collect_occluded_labels(nusc)
    print(f"  含遮挡 PA 目标的帧: {len(labels)}")
    total = sum(len(v) for v in labels.values())
    type_c = {0:0,1:0,2:0}
    vis_c  = {1:0,2:0,3:0,4:0}
    for ls in labels.values():
        for l in ls:
            type_c[l['pa_type']] += 1
            vis_c[l['visibility']] = vis_c.get(l['visibility'],0)+1
    print(f"  PA 实例总数: {total}")
    print(f"  Vehicle={type_c[0]}  Pedestrian={type_c[1]}  Cyclist={type_c[2]}")

    # save labels
    with open(out/'occluded_labels.pkl','wb') as f:
        pickle.dump(labels, f)
    with open(out/'occluded_tokens.txt','w') as f:
        f.writelines(t+'\n' for t in labels)

    # copy sensor files
    if not args.no_copy:
        for token in tqdm(labels, desc="Copy files"):
            copy_files(nusc, token, dataroot=dataroot, outdir_base=outdir_base)

    # visualization
    tokens = sorted(labels.keys(),
                    key=lambda t: len(labels[t]), reverse=True)
    if args.vis_n != -1:
        tokens = tokens[:args.vis_n]

    print(f"\n可视化 {len(tokens)} 帧...")
    for tok in tqdm(tokens, desc="Visualizing"):
        try:
            visualize_occluded_frame_multicam(
                nusc, tok, labels[tok],
                str(vis_dir/f'{tok}.png'),
                dataroot=dataroot,
            )
            if args.bev:
                visualize_bev_occluded(
                    nusc, tok, labels[tok],
                    str(vis_dir/f'{tok}_bev.png')
                )
        except Exception as ex:
            print(f"\n  [WARN] {tok[:12]}: {ex}")

    # statistics report
    report = f"""
{'='*50}
  Version B — 遮挡标注直接可视化统计
{'='*50}
  无需时序追踪，直接读取 NuScenes 标注

  含遮挡 PA 目标的帧: {len(labels)}
  PA 实例总数: {total}
    Vehicle:    {type_c[0]}
    Pedestrian: {type_c[1]}
    Cyclist:    {type_c[2]}
  Visibility 分布（仅统计 vis<=1 部分）:
    vis=1: {vis_c.get(1,0)}

  pkl 字段（每个 label）:
    translation  ← NuScenes 标注员估计的当前帧 3D 位置
    rotation     ← 当前帧朝向
    size         ← 当前帧尺寸
    visibility   ← 当前帧真实 visibility（1=<40%可见）
    velocity     ← 当前帧速度估计（差分，仅供参考）
    num_lidar_pts← 框内 LiDAR 点数（=0 说明纯靠视觉估计）
{'='*50}"""
    print(report)
    with open(out/'stats.txt','w') as f: f.write(report)
    print(f"\n可视化已保存至 {vis_dir}/")

if __name__ == '__main__':
    main()
