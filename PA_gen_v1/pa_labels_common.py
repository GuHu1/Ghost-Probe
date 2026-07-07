"""
pa_labels_common.py
===================
Shared PA (Phantom Agent) label generation logic used by both
`create_pa_labels_mini.py` and `create_pa_labels_full.py`.

The only differences between the mini and trainval pipelines are the
nuScenes version, dataroot, and output directory. All data processing,
visualization, and file-copying code lives here to avoid maintaining two
near-identical 400+ line files.

BEV grid parameters are read from `common/bev_config.py` — the single
source of truth for the whole repository.
"""

import sys
import pickle
import shutil
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.font_manager as _fm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for common/
from common.bev_config import (
    BEV_RANGE_X0Y0X1Y1 as BEV_RANGE,
    BEV_RESOLUTION_M as BEV_RES,
)


# ── Font ───────────────────────────────────────────────────────────────────
_CJK = 'Noto Sans CJK JP'
if _CJK in {f.name for f in _fm.fontManager.ttflist}:
    matplotlib.rcParams['font.family'] = _CJK
    matplotlib.rcParams['font.sans-serif'] = [_CJK, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════════════
#  Common config
# ═══════════════════════════════════════════════════════════════════════════
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
LIDAR = 'LIDAR_TOP'
N_SWEEPS = 9          # matches BEVDet default

VIS_OCCLUDED = 1
VIS_EMERGED = 4
BACK_SECONDS = 5.0
LAMBDA = 0.5

# BEV grid from common/bev_config.py — single source of truth for the repo
H_BEV = int((BEV_RANGE[3] - BEV_RANGE[1]) / BEV_RES)
W_BEV = int((BEV_RANGE[2] - BEV_RANGE[0]) / BEV_RES)

PA_DIST_THRESHOLD = 50.0

PA_CATEGORY_MAP = {
    'vehicle.car':                          (0, 'Vehicle'),
    'vehicle.truck':                        (0, 'Vehicle'),
    'vehicle.bus.bendy':                    (0, 'Vehicle'),
    'vehicle.bus.rigid':                    (0, 'Vehicle'),
    'vehicle.construction':                 (0, 'Vehicle'),
    'vehicle.trailer':                      (0, 'Vehicle'),
    'human.pedestrian.adult':               (1, 'Pedestrian'),
    'human.pedestrian.child':               (1, 'Pedestrian'),
    'human.pedestrian.construction_worker': (1, 'Pedestrian'),
    'human.pedestrian.police_officer':      (1, 'Pedestrian'),
    'vehicle.bicycle':                      (2, 'Cyclist'),
    'vehicle.motorcycle':                   (2, 'Cyclist'),
}
PA_SIZE_PRIOR = {0: [4.5, 1.8, 1.5], 1: [0.5, 0.5, 1.7], 2: [1.8, 0.6, 1.5]}
PA_COLOR = {0: '#FF5555', 1: '#55FF55', 2: '#5599FF'}
PA_NAME = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════
def make_tf(trans, rot, inv=False):
    R = Quaternion(rot).rotation_matrix
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = np.array(trans)
    return np.linalg.inv(m) if inv else m


def g2l(nusc, sample_token):
    sam = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sam['data'][LIDAR])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    return np.linalg.inv(
        make_tf(ep['translation'], ep['rotation']) @
        make_tf(cs['translation'], cs['rotation'])
    )


def get_velocity(nusc, ann_token):
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
        d0, d1 = np.array(ap['translation']), np.array(an['translation'])
    else:
        return np.full(3, np.nan)
    dt = (t1 - t0) * 1e-6
    return (d1 - d0) / dt if 0 < dt <= 1.5 else np.full(3, np.nan)


def _clip_line(x1, y1, x2, y2, W, H):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < 0:
            c |= LEFT
        elif x > W:
            c |= RIGHT
        if y < 0:
            c |= TOP
        elif y > H:
            c |= BOTTOM
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    while True:
        if not (c1 | c2):
            return x1, y1, x2, y2
        if c1 & c2:
            return None
        c_out = c1 if c1 else c2
        if c_out & BOTTOM:
            x = x1 + (x2 - x1) * (H - y1) / (y2 - y1) if y2 != y1 else x1
            y = float(H)
        elif c_out & TOP:
            x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1) if y2 != y1 else x1
            y = 0.
        elif c_out & RIGHT:
            y = y1 + (y2 - y1) * (W - x1) / (x2 - x1) if x2 != x1 else y1
            x = float(W)
        else:
            y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1) if x2 != x1 else y1
            x = 0.
        if c_out == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)


def _project_vel_arrow(pos_g, vel_g, ep, cs, K, W, H, scale=1.5):
    def g2cam(p):
        p = np.array(p, dtype=float)
        p -= np.array(ep['translation'])
        p = Quaternion(ep['rotation']).inverse.rotate(p)
        p -= np.array(cs['translation'])
        p = Quaternion(cs['rotation']).inverse.rotate(p)
        return p

    p0c = g2cam(pos_g)
    p1c = g2cam(np.array(pos_g) + np.array(vel_g) * scale)
    if p0c[2] <= 0.1 or p1c[2] <= 0.1:
        return None

    def proj(p):
        uv = K @ p
        return uv[0] / uv[2], uv[1] / uv[2]

    x0, y0 = proj(p0c)
    x1, y1 = proj(p1c)
    if np.hypot(x1 - x0, y1 - y0) < 3:
        return None
    if not (-100 <= x0 <= W + 100 and -100 <= y0 <= H + 100):
        return None
    return x0, y0, x1, y1


# ═══════════════════════════════════════════════════════════════════════════
#  Label core
# ═══════════════════════════════════════════════════════════════════════════
def _make_neg_label(ann_token, inst, cat, pa_type, pa_type_str, ann):
    return {
        'instance_token': inst['token'],
        'category': cat,
        'pa_type': pa_type,
        'pa_type_str': pa_type_str,
        'is_positive': False,
        'ann_token_current': ann_token,
        'sample_token_current': ann['sample_token'],
        'visibility_current': int(ann['visibility_token']),
        'translation_current': list(ann['translation']),
        'rotation_current': list(ann['rotation']),
        'size_current': list(ann['size']),
        'num_lidar_pts': ann['num_lidar_pts'],
        'ann_token_emerge': None,
        'sample_token_emerge': None,
        'translation_emerge': [np.nan, np.nan, np.nan],
        'velocity_emerge': [np.nan, np.nan, np.nan],
        'k': None,
        'time_to_emerge': None,
        'sample_weight': 1.0,
        'exist_prob_gt': 0.0,
    }


def build_all_pa_labels(nusc):
    all_occ_anns, pos_labels = {}, {}
    for inst in tqdm(nusc.instance, desc="Scanning instances"):
        cat = nusc.get('category', inst['category_token'])['name']
        if cat not in PA_CATEGORY_MAP:
            continue
        pa_type, pa_type_str = PA_CATEGORY_MAP[cat]

        chain, tok = [], inst['first_annotation_token']
        while tok:
            ann = nusc.get('sample_annotation', tok)
            sample = nusc.get('sample', ann['sample_token'])
            chain.append({
                'ann_token': tok,
                'sample_token': ann['sample_token'],
                'timestamp': sample['timestamp'],
                'visibility': int(ann['visibility_token']),
                'translation': list(ann['translation']),
                'rotation': list(ann['rotation']),
                'size': list(ann['size']),
                'num_lidar_pts': ann['num_lidar_pts'],
            })
            tok = ann['next']

        for e in chain:
            if e['visibility'] <= VIS_OCCLUDED:
                all_occ_anns.setdefault(e['sample_token'], []).append(
                    (e['ann_token'], inst, cat, pa_type, pa_type_str))

        if len(chain) < 2:
            continue

        for i in range(1, len(chain)):
            cur, prev = chain[i], chain[i - 1]
            if cur['visibility'] < VIS_EMERGED:
                continue
            if prev['visibility'] > VIS_OCCLUDED:
                continue

            emerge_vel = get_velocity(nusc, cur['ann_token'])
            vel_list = (emerge_vel.tolist() if not np.isnan(emerge_vel).any()
                        else [np.nan, np.nan, np.nan])

            for j in range(i - 1, -1, -1):
                past = chain[j]
                dt = (cur['timestamp'] - past['timestamp']) * 1e-6
                if dt > BACK_SECONDS:
                    break
                if past['visibility'] > VIS_OCCLUDED:
                    continue

                k = i - j
                pos_labels.setdefault(past['sample_token'], []).append({
                    'instance_token': inst['token'],
                    'category': cat,
                    'pa_type': pa_type,
                    'pa_type_str': pa_type_str,
                    'is_positive': True,
                    'ann_token_current': past['ann_token'],
                    'sample_token_current': past['sample_token'],
                    'visibility_current': past['visibility'],
                    'translation_current': past['translation'],
                    'rotation_current': past['rotation'],
                    'size_current': past['size'],
                    'num_lidar_pts': past['num_lidar_pts'],
                    'ann_token_emerge': cur['ann_token'],
                    'sample_token_emerge': cur['sample_token'],
                    'translation_emerge': cur['translation'],
                    'velocity_emerge': vel_list,
                    'k': k,
                    'time_to_emerge': round(dt, 3),
                    'sample_weight': float(np.exp(-LAMBDA * (k - 1))),
                    'exist_prob_gt': 1.0,
                })

    neg_labels = {}
    for st, anns_info in all_occ_anns.items():
        if st in pos_labels:
            continue
        frame_neg = []
        for ann_token, inst, cat, pa_type, pa_type_str in anns_info:
            ann = nusc.get('sample_annotation', ann_token)
            frame_neg.append(_make_neg_label(ann_token, inst, cat, pa_type, pa_type_str, ann))
        if frame_neg:
            neg_labels[st] = frame_neg

    return pos_labels, neg_labels


# ═══════════════════════════════════════════════════════════════════════════
#  BEV GT
# ═══════════════════════════════════════════════════════════════════════════
def _gaussian_radius(h, w, min_overlap=0.7):
    def r(a, b, c):
        d = b * b - 4 * a * c
        return (b - np.sqrt(max(d, 0))) / (2 * a) if a else 0

    r1 = r(1, h + w, h * w * (1 - min_overlap) / (1 + min_overlap))
    r2 = r(4, 2 * (h + w), (1 - min_overlap) * h * w)
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * h * w
    r3 = (-b3 + np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))) / (2 * a3)
    return max(min(r1, r2, r3), 2.0)


def _draw_gaussian(hm, cr, cc, radius):
    H, W = hm.shape
    r = int(radius)
    s = (2 * radius + 1) / 6
    r0, r1 = max(0, cr - r), min(H, cr + r + 1)
    c0, c1 = max(0, cc - r), min(W, cc + r + 1)
    if r0 >= r1 or c0 >= c1:
        return hm
    rr, cc_ = np.meshgrid(np.arange(r0, r1) - cr, np.arange(c0, c1) - cc, indexing='ij')
    hm[r0:r1, c0:c1] = np.maximum(
        hm[r0:r1, c0:c1],
        np.exp(-(rr ** 2 + cc_ ** 2) / (2 * s ** 2))
    )
    return hm


def compute_bev_gt(nusc, sample_token, pa_labels):
    x0, y0, x1, y1 = BEV_RANGE
    hm = np.zeros((H_BEV, W_BEV), np.float32)
    vm = np.zeros((2, H_BEV, W_BEV), np.float32)
    wm = np.ones((H_BEV, W_BEV), np.float32)
    G2L = g2l(nusc, sample_token)
    for lbl in pa_labels:
        if not lbl['is_positive']:
            continue
        pos_l = (G2L @ np.append(lbl['translation_emerge'], 1.))[:3]
        px, py = pos_l[0], pos_l[1]
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        col = int(np.clip((px - x0) / BEV_RES, 0, W_BEV - 1))
        row = int(np.clip((py - y0) / BEV_RES, 0, H_BEV - 1))
        sz = PA_SIZE_PRIOR[lbl['pa_type']]
        rad = _gaussian_radius(sz[0] / BEV_RES, sz[1] / BEV_RES)
        hm = _draw_gaussian(hm, row, col, rad)
        vel = np.array(lbl['velocity_emerge'])
        if not np.isnan(vel).any():
            vl = (G2L[:3, :3] @ vel)[:2]
            vm[0, row, col] = float(vl[0])
            vm[1, row, col] = float(vl[1])
        w = lbl['sample_weight']
        ri = int(rad)
        wm[max(0, row - ri):min(H_BEV, row + ri + 1),
           max(0, col - ri):min(W_BEV, col + ri + 1)] = np.maximum(
            wm[max(0, row - ri):min(H_BEV, row + ri + 1),
               max(0, col - ri):min(W_BEV, col + ri + 1)], w)
    return hm, vm, wm, (hm > 1e-3).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════
def _draw_3d_box_on_ax(ax, lbl, ep, cs, K, W, H, color, lw=0.7):
    prior_lwh = PA_SIZE_PRIOR[lbl['pa_type']]
    prior_wlh = [prior_lwh[1], prior_lwh[0], prior_lwh[2]]
    box = Box(lbl['translation_current'], prior_wlh, Quaternion(lbl['rotation_current']))
    box.translate(-np.array(ep['translation']))
    box.rotate(Quaternion(ep['rotation']).inverse)
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)
    corners = box.corners()
    if np.any(corners[2, :] <= 0.1):
        return False
    pts = view_points(corners, K, normalize=True)
    x2d, y2d = pts[0, :], pts[1, :]
    margin = 50
    if (np.all(x2d < -margin) or np.all(x2d > W + margin) or
            np.all(y2d < -margin) or np.all(y2d > H + margin)):
        return False
    for s, e in [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        cl = _clip_line(float(x2d[s]), float(y2d[s]), float(x2d[e]), float(y2d[e]), W, H)
        if cl:
            ax.plot([cl[0], cl[2]], [cl[1], cl[3]], color=color, lw=lw, alpha=0.8, zorder=3)
    in_view = (x2d >= 0) & (x2d <= W) & (y2d >= 0) & (y2d <= H)
    if not in_view.any():
        return False
    cx = float(np.mean(x2d[in_view]))
    cy = float(np.clip(np.min(y2d[in_view]) - 20, 5, H - 35))
    vel = lbl['velocity_emerge']
    vx, vy = float(vel[0]), float(vel[1])
    spd = np.hypot(vx, vy) if not np.isnan(vx) else float('nan')
    spd_str = f"{spd:.1f}m/s" if not np.isnan(spd) else "—"
    pos_str = "POS" if lbl['is_positive'] else "NEG"
    k_str = f" k={lbl['k']}" if lbl['k'] is not None else ""
    ax.text(cx, cy,
            f"[{pos_str}] {PA_NAME[lbl['pa_type']]}  "
            f"vis={lbl['visibility_current']}  V_emg={spd_str}{k_str}",
            color='white', fontsize=6.5, ha='center', fontweight='bold',
            bbox=dict(fc=color, alpha=0.30, pad=0, ec='none', boxstyle='round,pad=0'), zorder=5)
    return True, cx, cy, x2d, y2d


def visualize_multicam(nusc, sample_token, frame_labels, save_path, dataroot):
    """Three-camera visualization: 3D boxes + pos/neg labels + V_emerge arrows."""
    sample = nusc.get('sample', sample_token)
    G2L = g2l(nusc, sample_token)
    near = [l for l in frame_labels
            if float(np.linalg.norm((G2L @ np.append(l['translation_current'], 1.))[:2])) <= PA_DIST_THRESHOLD]

    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    fig.patch.set_facecolor('#0f0f1e')
    for ax_i, cam_ch in enumerate(['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']):
        ax = axes[ax_i]
        cam_sd = nusc.get('sample_data', sample['data'][cam_ch])
        img_p = Path(dataroot) / cam_sd['filename']
        img = (Image.open(img_p).convert('RGB') if img_p.exists()
               else Image.new('RGB', (1600, 900), (20, 20, 30)))
        W, H = img.width, img.height
        ax.imshow(img, extent=[0, W, H, 0])
        cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        K = np.array(cs['camera_intrinsic'])
        drawn = 0
        for lbl in near:
            color = PA_COLOR[lbl['pa_type']]
            result = _draw_3d_box_on_ax(ax, lbl, ep, cs, K, W, H, color)
            if not result:
                continue
            drawn += 1
            vel = lbl['velocity_emerge']
            vx, vy = float(vel[0]), float(vel[1])
            if not np.isnan(vx) and np.hypot(vx, vy) > 0.1:
                arr = _project_vel_arrow(lbl['translation_current'], [vx, vy, 0.], ep, cs, K, W, H)
                if arr:
                    ax.annotate('', xy=(arr[2], arr[3]), xytext=(arr[0], arr[1]),
                                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.0, mutation_scale=9), zorder=6)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f'{cam_ch}  |  PA目标: {drawn}  细线=先验尺寸  黄箭头=V_emerge',
                     color='#aaaaff', fontsize=9)
        ax.axis('off')

    n_pos = sum(1 for l in near if l['is_positive'])
    n_neg = len(near) - n_pos
    fig.legend(handles=[
        Line2D([0], [0], color=PA_COLOR[t], lw=2, label=PA_NAME[t]) for t in [0, 1, 2]
    ] + [Line2D([0], [0], color='#FFD700', lw=2, label='V_emerge'),
         mpatches.Patch(color='#aaffaa', alpha=0.7, label=f'正样本 {n_pos}'),
         mpatches.Patch(color='#ff9999', alpha=0.7, label=f'负样本 {n_neg}')],
        loc='lower center', facecolor='#1a1a2e', labelcolor='white', fontsize=9, ncol=6, framealpha=0.9)
    fig.suptitle(f"Token:{sample_token[:20]}  PA:{len(near)}(正={n_pos} 负={n_neg})\n"
                 f"vis=当前遮挡状态  V_emerge=冲出盲区瞬时速度", color='white', fontsize=9.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=110, facecolor='#0f0f1e', bbox_inches='tight')
    plt.close()


def visualize_bev(nusc, sample_token, frame_labels, frame_data, save_path):
    """BEV heatmap base layer + vector overlay (rotated boxes, stars, links, V_emerge arrows)."""
    sample = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sample['data'][LIDAR])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    G2L = g2l(nusc, sample_token)

    R_g2e = Quaternion(ep['rotation']).rotation_matrix.T
    T_ego = np.array(ep['translation'])
    R_l2e = Quaternion(cs['rotation']).rotation_matrix

    def g2lidar_xy(pos):
        pe = R_g2e @ (np.array(pos) - T_ego)
        pl = R_l2e.T @ (pe - np.array(cs['translation']))
        return pl[0], pl[1]

    hm = frame_data.get('heatmap')
    vm = frame_data.get('velocity_map')
    if hm is None:
        hm, vm, _, _ = compute_bev_gt(nusc, sample_token, frame_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_facecolor('#0f0f1e')

    extent = [BEV_RANGE[0], BEV_RANGE[2], BEV_RANGE[1], BEV_RANGE[3]]
    ax.imshow(hm, extent=extent, origin='lower', cmap='inferno', vmin=0, vmax=1, alpha=0.6)
    xs = np.linspace(BEV_RANGE[0], BEV_RANGE[2], W_BEV)
    ys = np.linspace(BEV_RANGE[1], BEV_RANGE[3], H_BEV)
    XX, YY = np.meshgrid(xs, ys)
    mask = hm > 0.3
    if mask.any() and vm is not None:
        ax.quiver(XX[mask], YY[mask], vm[0][mask], vm[1][mask],
                  color='#FFD700', scale=60, width=0.004, alpha=0.85, headwidth=4, headlength=5, zorder=5)

    in_bev = lambda p: (BEV_RANGE[0] <= p[0] <= BEV_RANGE[2] and
                        BEV_RANGE[1] <= p[1] <= BEV_RANGE[3])

    for lbl in frame_labels:
        if float(np.linalg.norm((G2L @ np.append(lbl['translation_current'], 1.))[:2])) > PA_DIST_THRESHOLD:
            continue
        color = PA_COLOR[lbl['pa_type']]
        xc, yc = g2lidar_xy(lbl['translation_current'])

        q_g = Quaternion(lbl['rotation_current'])
        yaw = (Quaternion(cs['rotation']).inverse * Quaternion(ep['rotation']).inverse * q_g).yaw_pitch_roll[0]
        sz = lbl['size_current']
        l_sz, w_sz = sz[1], sz[0]
        corners_local = np.array([
            [-l_sz / 2, -w_sz / 2], [l_sz / 2, -w_sz / 2],
            [l_sz / 2, w_sz / 2], [-l_sz / 2, w_sz / 2]
        ])
        c_, s_ = np.cos(yaw), np.sin(yaw)
        R2 = np.array([[c_, -s_], [s_, c_]])
        corners_bev = (R2 @ corners_local.T).T + np.array([xc, yc])
        ax.add_patch(plt.Polygon(corners_bev, fill=True, facecolor=color,
                                 edgecolor=color, alpha=0.25, lw=1.5, zorder=3))
        ax.plot(xc, yc, 'o', color=color, ms=7, mec='white', mew=0.6, alpha=0.9, zorder=5)

        vel = lbl['velocity_emerge']
        vx, vy = float(vel[0]), float(vel[1])
        spd = np.hypot(vx, vy) if not np.isnan(vx) else float('nan')
        spd_str = f"{spd:.1f}m/s" if not np.isnan(spd) else "—"
        pos_str = "POS" if lbl['is_positive'] else "NEG"
        k_str = f" k={lbl['k']}" if lbl['k'] is not None else ""
        ax.annotate(f"[{pos_str}] vis={lbl['visibility_current']}\nV_emg={spd_str}{k_str}",
                    (xc, yc), textcoords='offset points', xytext=(5, 5), color='white', fontsize=6,
                    bbox=dict(fc=color, alpha=0.28, pad=0, ec='none', boxstyle='round,pad=0'), zorder=6)

        if lbl['is_positive']:
            te = lbl['translation_emerge']
            if not any(np.isnan(te)):
                xe, ye = g2lidar_xy(te)
                if in_bev((xe, ye)):
                    ax.plot(xe, ye, '*', color=color, ms=11, mec='white', mew=0.5, zorder=6)
                if in_bev((xc, yc)) and in_bev((xe, ye)):
                    ax.annotate('', xy=(xe, ye), xytext=(xc, yc),
                                arrowprops=dict(arrowstyle='->', color=color, lw=0.9, alpha=0.6, linestyle='dashed'), zorder=4)
                if not np.isnan(vx):
                    vel_l = (G2L[:3, :3] @ np.array([vx, vy, 0.]))[:2]
                    ax.annotate('', xy=(xe + vel_l[0] * 1.5, ye + vel_l[1] * 1.5), xytext=(xe, ye),
                                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.1, mutation_scale=9, alpha=0.95), zorder=7)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(PA_DIST_THRESHOLD * np.cos(theta), PA_DIST_THRESHOLD * np.sin(theta),
            color='white', lw=0.6, ls='--', alpha=0.35)
    ax.plot(0, 0, 'w^', ms=12, zorder=8)
    ax.axhline(0, color='white', lw=0.4, alpha=0.2)
    ax.axvline(0, color='white', lw=0.4, alpha=0.2)
    ax.set_xlim(BEV_RANGE[0], BEV_RANGE[2])
    ax.set_ylim(BEV_RANGE[1], BEV_RANGE[3])
    ax.tick_params(colors='white', labelsize=7)
    ax.set_xlabel('X(m)', color='white')
    ax.set_ylabel('Y(m)', color='white')
    ax.set_title(f'BEV  ●遮挡位置  ★露头GT  黄箭头=V_emerge\n'
                 f'底层=训练热力图  Token:{sample_token[:20]}', color='#aaaaff', fontsize=8)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=PA_COLOR[t], ms=7, label=PA_NAME[t], ls='none') for t in [0, 1, 2]
    ] + [Line2D([0], [0], marker='^', color='white', ms=9, label='Ego', ls='none'),
         Line2D([0], [0], marker='*', color='white', ms=9, label='露头位置', ls='none'),
         Line2D([0], [0], color='#FFD700', lw=2, label='V_emerge')],
        fontsize=7, facecolor='#1a1a2e', labelcolor='white', loc='upper right', framealpha=0.85)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, facecolor='#0f0f1e', bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  File copy
# ═══════════════════════════════════════════════════════════════════════════
def copy_sensor_files(nusc, sample_token, base, dataroot):
    sam = nusc.get('sample', sample_token)
    for ch in list(CAMERAS) + [LIDAR]:
        sd = nusc.get('sample_data', sam['data'][ch])
        src = Path(dataroot) / sd['filename']
        dst = base / sd['filename']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
    # sweep LiDAR
    sd_lidar = nusc.get('sample_data', sam['data'][LIDAR])
    tok_prev = sd_lidar['prev']
    count = 0
    while tok_prev and count < N_SWEEPS:
        sd_sw = nusc.get('sample_data', tok_prev)
        src = Path(dataroot) / sd_sw['filename']
        dst = base / sd_sw['filename']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
        tok_prev = sd_sw['prev']
        count += 1


def copy_maps_and_meta(dataroot, base, version):
    for sub in ['maps', version]:
        src = Path(dataroot) / sub
        dst = base / sub
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main(version: str, dataroot: str, outdir_base: str,
         no_copy: bool = False, vis_n: int = 5):
    """Run the PA label generation pipeline for a given nuScenes split."""
    out = Path(outdir_base)
    for d in [out / 'full', out / 'positive', out / 'negative', out / 'preview']:
        d.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"NuScenes {version}: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
    print(f"BEV grid: {H_BEV}x{W_BEV} @ {BEV_RES}m/px  (from common/bev_config.py)")

    print("\n[1/5] 标注...")
    pos_labels, neg_labels = build_all_pa_labels(nusc)
    print(f"  正样本帧: {len(pos_labels)}  负样本帧: {len(neg_labels)}")

    print("\n[2/5] 计算 BEV GT...")
    pos_full, neg_full = {}, {}
    for st, labels in tqdm(pos_labels.items(), desc="pos BEV"):
        hm, vm, wm, ex = compute_bev_gt(nusc, st, labels)
        pos_full[st] = {
            'token': st,
            'pa_instances': labels,
            'heatmap': hm,
            'velocity_map': vm,
            'weight_map': wm,
            'exist_map': ex,
            'n_pos': sum(1 for l in labels if l['is_positive']),
            'has_pa': True,
        }
    for st, labels in neg_labels.items():
        neg_full[st] = {
            'token': st,
            'pa_instances': labels,
            'heatmap': np.zeros((H_BEV, W_BEV), np.float32),
            'velocity_map': np.zeros((2, H_BEV, W_BEV), np.float32),
            'weight_map': np.ones((H_BEV, W_BEV), np.float32),
            'exist_map': np.zeros((H_BEV, W_BEV), np.float32),
            'n_pos': 0,
            'has_pa': True,
        }

    # full = all nuScenes samples
    all_full = {}
    for sample in nusc.sample:
        st = sample['token']
        if st in pos_full:
            all_full[st] = pos_full[st]
        elif st in neg_full:
            all_full[st] = neg_full[st]
        else:
            all_full[st] = {
                'token': st,
                'pa_instances': [],
                'heatmap': np.zeros((H_BEV, W_BEV), np.float32),
                'velocity_map': np.zeros((2, H_BEV, W_BEV), np.float32),
                'weight_map': np.ones((H_BEV, W_BEV), np.float32),
                'exist_map': np.zeros((H_BEV, W_BEV), np.float32),
                'n_pos': 0,
                'has_pa': False,
            }

    print(f"  full 总帧: {len(all_full)} (正={sum(1 for v in all_full.values() if v['n_pos'] > 0)}"
          f" 负={sum(1 for v in all_full.values() if v['has_pa'] and v['n_pos'] == 0)}"
          f" 无PA={sum(1 for v in all_full.values() if not v['has_pa'])})")

    print("\n[3/5] 保存 pkl...")
    with open(out / 'full' / 'phantom_labels_full.pkl', 'wb') as f:
        pickle.dump(all_full, f)
    with open(out / 'positive' / 'phantom_labels_positive.pkl', 'wb') as f:
        pickle.dump(pos_full, f)
    with open(out / 'negative' / 'phantom_labels_negative.pkl', 'wb') as f:
        pickle.dump(neg_full, f)
    for name, d in [('full', all_full), ('positive', pos_full), ('negative', neg_full)]:
        with open(out / f'tokens_{name}.txt', 'w') as f:
            f.writelines(t + '\n' for t in d)
    print("  pkl 已保存")

    print("\n[4/5] 复制文件...")
    if not no_copy:
        for st in tqdm(all_full, desc="Copying sensor files"):
            try:
                copy_sensor_files(nusc, st, out / 'full', dataroot)
            except Exception as ex:
                print(f"  [WARN] {st[:8]}: {ex}")
        copy_maps_and_meta(dataroot, out / 'full', version)
        print("  文件复制完成")
    else:
        print("  跳过（--no_copy）")

    print("\n[5/5] 可视化...")
    vis_dir = out / 'preview'
    vis_n = vis_n if vis_n != -1 else len(pos_full)
    best = sorted(pos_full, key=lambda t: min(l['k'] for l in pos_full[t]['pa_instances'] if l['is_positive']))[:vis_n]
    for i, tok in enumerate(best):
        fd = pos_full[tok]
        try:
            visualize_multicam(nusc, tok, fd['pa_instances'],
                               str(vis_dir / f'{i:03d}_{tok[:8]}_cam.png'), dataroot)
            visualize_bev(nusc, tok, fd['pa_instances'], fd,
                          str(vis_dir / f'{i:03d}_{tok[:8]}_bev.png'))
        except Exception as ex:
            print(f"  [WARN] {tok[:8]}: {ex}")
    print(f"  预览图已保存至 {vis_dir}/")

    # statistics
    type_c = {0: 0, 1: 0, 2: 0}
    k_dist = {}
    for labels in pos_labels.values():
        for l in labels:
            type_c[l['pa_type']] += 1
            k_dist[l['k']] = k_dist.get(l['k'], 0) + 1
    report = f"""
{'='*55}
  {version} PA 标注统计
{'='*55}
  全部帧: {len(all_full)}
    正样本帧: {len(pos_full)}   负样本帧: {len(neg_full)}
    无PA帧:   {len(all_full) - len(pos_full) - len(neg_full)}

  PA实例(正): Vehicle={type_c[0]} Ped={type_c[1]} Cyclist={type_c[2]}
  k分布: {dict(sorted(k_dist.items()))}
  BACK={BACK_SECONDS}s  λ={LAMBDA}  BEV={H_BEV}x{W_BEV}@{BEV_RES}m/px

  输出:
    full/phantom_labels_full.pkl    ← 训练用（所有帧）
    positive/phantom_labels_positive.pkl
    negative/phantom_labels_negative.pkl
    preview/*_cam.png  *_bev.png
{'='*55}"""
    print(report)
    with open(out / 'stats.txt', 'w') as f:
        f.write(report)


def parse_and_run(
    version: str,
    dataroot: str,
    outdir_base: str,
):
    """
    Thin CLI wrapper around `main`.

    The calling script supplies split-specific defaults for version/dataroot/
    outdir_base; the user can override any of them on the command line without
    editing the file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',    type=str, default=version,
                        help='nuScenes version (e.g. v1.0-mini)')
    parser.add_argument('--dataroot',   type=str, default=dataroot,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--outdir_base', type=str, default=outdir_base,
                        help='Base output directory')
    parser.add_argument('--no_copy',    action='store_true')
    parser.add_argument('--vis_n',      type=int, default=5,
                        help='Number of positive samples to visualize (-1=all)')
    args = parser.parse_args()
    main(args.version, args.dataroot, args.outdir_base,
         no_copy=args.no_copy, vis_n=args.vis_n)
