"""
Visualization Utilities for OSZ Demo
=====================================

BEV coordinate system (applies to ALL imshow / contour calls in this file)
---------------------------------------------------------------------------
Ego frame:   x = forward,  y = left,  z = up  (nuScenes convention)
BEV array:   shape (nx, ny), indexing='ij'
               array[i, j]  where  i ↔ ego-x (axis-0),  j ↔ ego-y (axis-1)

imshow convention used throughout:
    imshow(data,                           # NO transpose
           origin='lower',
           extent=[y_max, y_min, x_min, x_max])
                    ^     ^               ← y-axis INVERTED so ego-left appears LEFT
    set_xlim(y_max, y_min)               ← match extent (left=ego-left=+y)
    set_ylim(x_min, x_max)               ← forward=UP (+x at top)

Why no transpose?
    With origin='lower':
        data[row=i, col=j] is placed at axes (x=col_coord, y=row_coord).
        i (ego-x = forward) maps to the vertical axis → forward=UP ✓
        j (ego-y = left)    maps to the horizontal axis

Why y-axis inverted in extent (y_max on left, y_min on right)?
    ego-y is positive LEFTWARD.  We want image-LEFT = ego-LEFT = +y.
    extent left edge = y_max (the largest positive-y = most-left in ego) ✓
    extent right edge = y_min (negative-y = ego-right) ✓

_draw_ego arrow:
    axes x = ego-y,  axes y = ego-x.
    Arrow from (0, size) → (0, size*1.5):  x stays 0 (centerline), y increases (forward). ✓
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
# NOTE: we deliberately do NOT call matplotlib.use('Agg') at import time.
# savefig() works under ANY backend — it does not need Agg. Removing the
# forced Agg lets callers that want plt.show() (e.g. interactive browsing)
# import this module without their window silently breaking. Callers that
# only save to files (e.g. run_osz_pipeline.py) set Agg themselves before
# importing this module so there is no behaviour change for them.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Dict, Optional

# Allow importing common/ when this file is imported directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from common.bev_config import BEV_RESOLUTION_M as _DEFAULT_BEV_RES
    from common.bev_config import bev_extent as _bev_extent
except Exception:
    _DEFAULT_BEV_RES = 0.2
    # fallback _bev_extent if common/ is unavailable
    def _bev_extent(bev_range):
        x_min, x_max, y_min, y_max = bev_range
        return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


CAMERA_COLORS = {
    'CAM_FRONT':       '#4ECDC4',
    'CAM_FRONT_LEFT':  '#45B7D1',
    'CAM_FRONT_RIGHT': '#96CEB4',
    'CAM_BACK':        '#FFEAA7',
    'CAM_BACK_LEFT':   '#DDA0DD',
    'CAM_BACK_RIGHT':  '#F0E68C',
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: unified BEV imshow / contour wrappers
# ─────────────────────────────────────────────────────────────────────────────


def _bev_imshow(ax: plt.Axes, data: np.ndarray, bev_range: tuple,
                cmap=None, vmin=None, vmax=None,
                alpha: float = 1.0, interpolation: str = 'nearest'):
    """
    Unified BEV imshow: forward=UP, ego-left=LEFT.

    data must be (nx, ny) with indexing='ij'
        axis-0 = ego-x (forward), axis-1 = ego-y (left)
    No transpose is applied here — the caller must NOT pre-transpose.
    """
    extent, xlim, ylim = _bev_extent(bev_range)
    ax.imshow(
        data,                   # NO .T — rows=ego-x (forward), cols=ego-y (left)
        origin='lower',         # row 0 at bottom = x_min (behind ego)
        extent=extent,          # [y_max, y_min, x_min, x_max]
        cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=alpha, interpolation=interpolation,
    )
    ax.set_xlim(*xlim)          # left=y_max=ego-left, right=y_min=ego-right
    ax.set_ylim(*ylim)          # bottom=x_min, top=x_max (forward=UP)
    ax.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=7)
    ax.set_ylabel('x (m) ↑ forward', fontsize=7)


def _bev_contour(ax: plt.Axes, data: np.ndarray, bev_range: tuple, **kwargs):
    """Contour overlay consistent with _bev_imshow convention."""
    extent, _, _ = _bev_extent(bev_range)
    ax.contour(data, origin='lower', extent=extent, **kwargs)


def _draw_ego(ax: plt.Axes, size: float = 2.0, color: str = '#1976d2'):
    """Draw ego vehicle as a small rectangle + forward arrow.

    In BEV axes: x-axis = ego-y, y-axis = ego-x.
    Arrow goes from (ax_x=0, ax_y=size) → (ax_x=0, ax_y=size*1.5):
        ax_x stays 0 → ego-y=0 (centerline) ✓
        ax_y increases → ego-x increases → FORWARD ✓
    """
    rect = plt.Rectangle((-size / 2, -size), size, size * 2,
                          linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.annotate('', xy=(0, size * 1.5), xytext=(0, size),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# ─────────────────────────────────────────────────────────────────────────────
# Public plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_bev_osz(
    per_cam_masks: Dict[str, np.ndarray],   # { cam_name: (nx, ny) bool }
    osz_mask: np.ndarray,                    # (nx, ny) bool — final OSZ
    bev_range: tuple = (-50, 50, -50, 50),
    refined_mask: Optional[np.ndarray] = None,  # (nx, ny) float [0,1]
    depth_bev: Optional[np.ndarray] = None,
    bev_occ: Optional[np.ndarray] = None,    # (nx, ny) bool — BEV obstacles
    title: str = "BEV Occlusion Shadow Zone",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel BEV visualization.
    Panel layout:
      [cam1] [cam2] [cam3]
      [cam4] [cam5] [OSZ + refined]
      [depth BEV (optional)]
    """
    n_cam = len(per_cam_masks)
    n_cols = 3
    n_rows = 2 + (1 if depth_bev is not None else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)

    shadow_cmap = ListedColormap(['none', '#d32f2f'])

    # ── Per-camera panels ──────────────────────────────────────────────────
    cam_names = list(per_cam_masks.keys())
    for idx, cam_name in enumerate(cam_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        mask = per_cam_masks[cam_name]
        color = CAMERA_COLORS.get(cam_name, '#888888')

        cmap_cam = ListedColormap(['none', color])
        ax.imshow(mask, origin='lower', extent=extent,
                  cmap=cmap_cam, vmin=0, vmax=1, interpolation='nearest', alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(cam_name.replace('CAM_', ''), fontsize=9, pad=3)
        ax.set_xlabel('y (m)', fontsize=7)
        ax.set_ylabel('x (m)', fontsize=7)
        ax.tick_params(labelsize=6)
        _draw_ego(ax)

    # ── OSZ summary panel ─────────────────────────────────────────────────
    osz_ax = axes[n_rows - (2 if depth_bev is None else 3)][n_cols - 1]

    for cam_name, mask in per_cam_masks.items():
        color = CAMERA_COLORS.get(cam_name, '#aaaaaa')
        cm = ListedColormap(['none', color])
        osz_ax.imshow(mask, origin='lower', extent=extent,
                      cmap=cm, vmin=0, vmax=1, interpolation='nearest', alpha=0.2)

    if bev_occ is not None:
        osz_ax.imshow(bev_occ, origin='lower', extent=extent,
                      cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.4)

    osz_ax.imshow(osz_mask, origin='lower', extent=extent,
                  cmap=shadow_cmap, vmin=0, vmax=1, interpolation='nearest', alpha=0.9)

    if refined_mask is not None:
        _bev_contour(osz_ax, refined_mask, bev_range,
                     levels=[0.5], colors=['#FF8800'], linewidths=1.2)

    osz_ax.set_xlim(*xlim)
    osz_ax.set_ylim(*ylim)
    osz_ax.set_title("OSZ (ego ray casting)", fontsize=9, pad=3)
    osz_ax.set_xlabel('y (m)', fontsize=7)
    osz_ax.set_ylabel('x (m)', fontsize=7)
    osz_ax.tick_params(labelsize=6)
    _draw_ego(osz_ax)

    patches = [mpatches.Patch(color='#d32f2f', label='OSZ')]
    if refined_mask is not None:
        patches.append(mpatches.Patch(color='#FF8800', label='Refined boundary'))
    osz_ax.legend(handles=patches, fontsize=6, loc='upper right')

    # ── Depth BEV panel ───────────────────────────────────────────────────
    if depth_bev is not None:
        d_ax = axes[-1][0]
        im = d_ax.imshow(depth_bev, origin='lower', extent=extent,
                         cmap='plasma', vmin=0, vmax=70, interpolation='bilinear')
        plt.colorbar(im, ax=d_ax, fraction=0.046, pad=0.04, label='depth (m)')
        d_ax.set_xlim(*xlim)
        d_ax.set_ylim(*ylim)
        d_ax.set_title("BEV depth", fontsize=9, pad=3)
        d_ax.set_xlabel('y (m)', fontsize=7)
        d_ax.set_ylabel('x (m)', fontsize=7)
        d_ax.tick_params(labelsize=6)
        _draw_ego(d_ax)

    # Hide unused axes
    all_axes = axes.ravel()
    used = n_cam + 1 + (1 if depth_bev is not None else 0)
    for ax in all_axes[used:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [saved] {save_path}")

    return fig


def plot_refinement_comparison(
    raw_mask: np.ndarray,
    refined_soft: np.ndarray,
    depth_bev: np.ndarray,
    bev_occ: Optional[np.ndarray] = None,
    bev_range: tuple = (-50, 50, -50, 50),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Three-panel: raw OSZ | CRF-refined | depth BEV with boundary overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    extent, xlim, ylim = _bev_extent(bev_range)

    # ── Raw OSZ ───────────────────────────────────────────────────────────
    if bev_occ is not None:
        axes[0].imshow(bev_occ, origin='lower', extent=extent,
                       cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.4)
    axes[0].imshow(raw_mask, origin='lower', extent=extent,
                   cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*ylim)
    axes[0].set_title("Geometric OSZ (raw)", fontsize=10)
    _draw_ego(axes[0])

    # ── Refined ───────────────────────────────────────────────────────────
    if bev_occ is not None:
        axes[1].imshow(bev_occ, origin='lower', extent=extent,
                       cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.4)
    axes[1].imshow(refined_soft, origin='lower', extent=extent,
                   cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
    _bev_contour(axes[1], refined_soft, bev_range,
                 levels=[0.5], colors=['#FF8800'], linewidths=1.5)
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*ylim)
    axes[1].set_title("CRF-refined OSZ", fontsize=10)
    _draw_ego(axes[1])

    # ── Depth + boundary ──────────────────────────────────────────────────
    im = axes[2].imshow(depth_bev, origin='lower', extent=extent,
                        cmap='plasma', vmin=0, vmax=60, interpolation='bilinear')
    if bev_occ is not None:
        axes[2].imshow(bev_occ, origin='lower', extent=extent,
                       cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.25)
    _bev_contour(axes[2], refined_soft, bev_range,
                 levels=[0.5], colors=['white'], linewidths=1.5)
    _bev_contour(axes[2], raw_mask.astype(float), bev_range,
                 levels=[0.5], colors=['#d32f2f'], linewidths=0.8,
                 linestyles='dashed')
    plt.colorbar(im, ax=axes[2], fraction=0.046, label='depth (m)')
    axes[2].set_xlim(*xlim); axes[2].set_ylim(*ylim)
    axes[2].set_title("Depth BEV + boundaries", fontsize=10)
    _draw_ego(axes[2])

    for ax in axes:
        ax.set_xlabel('y (m)', fontsize=8)
        ax.set_ylabel('x (m)', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("OSZ Boundary Refinement Comparison", fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [saved] {save_path}")

    return fig


def plot_camera_osz_comparison(
    images: Dict[str, np.ndarray],
    depth_maps: Dict[str, np.ndarray],
    per_cam_masks: Dict[str, np.ndarray],
    osz_mask: np.ndarray,
    refined_mask: Optional[np.ndarray],
    depth_bev: np.ndarray,
    bev_occ: Optional[np.ndarray] = None,
    osz_pa: Optional[np.ndarray] = None,
    bev_range: tuple = (-50, 50, -50, 50),
    sample_token: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    One row per camera: image → dense depth → camera-view BEV shadow mask.
    Final row shows fused results: OSZ Raw → PA-relevant OSZ → BEV Depth.
    The main title includes the frame token for easy identification.
    """
    cam_names = list(images.keys())
    n_cam = len(cam_names)
    n_rows = n_cam + 1
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"OSZ Comparison — Frame: {sample_token}", fontsize=14,
                 fontweight='bold', y=0.98)

    extent, xlim, ylim = _bev_extent(bev_range)
    shadow_cmap = ListedColormap(['none', '#d32f2f'])

    for i, cam_name in enumerate(cam_names):
        ax_img   = axes[i][0]
        ax_depth = axes[i][1]
        ax_bev   = axes[i][2]

        # image
        ax_img.imshow(images[cam_name])
        ax_img.set_title(f"{cam_name} — Image", fontsize=10)
        ax_img.axis('off')

        # dense depth map
        dmap = depth_maps[cam_name]
        im_d = ax_depth.imshow(dmap, cmap='plasma', vmin=0, vmax=70)
        ax_depth.set_title(f"{cam_name} — Dense Depth", fontsize=10)
        ax_depth.axis('off')
        plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)

        # per-camera BEV shadow mask
        mask = per_cam_masks[cam_name]
        color = CAMERA_COLORS.get(cam_name, '#888888')
        cmap_cam = ListedColormap(['none', color])
        if bev_occ is not None:
            ax_bev.imshow(bev_occ, origin='lower', extent=extent,
                          cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.3)
        ax_bev.imshow(mask, origin='lower', extent=extent,
                      cmap=cmap_cam, vmin=0, vmax=1, interpolation='nearest', alpha=0.85)
        ax_bev.set_xlim(*xlim); ax_bev.set_ylim(*ylim)
        ax_bev.set_title(f"{cam_name} — Shadow BEV", fontsize=10)
        ax_bev.set_xlabel('y (m)', fontsize=7)
        ax_bev.set_ylabel('x (m)', fontsize=7)
        ax_bev.tick_params(labelsize=6)
        _draw_ego(ax_bev)

    # ── Last row: fused results ─────────────────────────────────────────
    ax_osz = axes[-1][0]
    if bev_occ is not None:
        ax_osz.imshow(bev_occ, origin='lower', extent=extent,
                      cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.4)
    ax_osz.imshow(osz_mask, origin='lower', extent=extent,
                  cmap=shadow_cmap, vmin=0, vmax=1, interpolation='nearest')
    ax_osz.set_xlim(*xlim); ax_osz.set_ylim(*ylim)
    ax_osz.set_title("OSZ Raw (ego ray casting)", fontsize=10)
    ax_osz.set_xlabel('y (m)', fontsize=7)
    ax_osz.set_ylabel('x (m)', fontsize=7)
    _draw_ego(ax_osz)

    ax_ref = axes[-1][1]
    if osz_pa is not None:
        ax_ref.imshow(osz_pa, origin='lower', extent=extent,
                      cmap=shadow_cmap, vmin=0, vmax=1, interpolation='nearest')
    elif refined_mask is not None:
        ax_ref.imshow(refined_mask, origin='lower', extent=extent,
                      cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
        _bev_contour(ax_ref, refined_mask, bev_range,
                     levels=[0.5], colors=['#FF8800'], linewidths=1.5)
    ax_ref.set_xlim(*xlim); ax_ref.set_ylim(*ylim)
    ax_ref.set_title("PA-relevant OSZ", fontsize=10)
    ax_ref.set_xlabel('y (m)', fontsize=7)
    ax_ref.set_ylabel('x (m)', fontsize=7)
    _draw_ego(ax_ref)

    ax_dbev = axes[-1][2]
    im_db = ax_dbev.imshow(depth_bev, origin='lower', extent=extent,
                           cmap='plasma', vmin=0, vmax=70, interpolation='bilinear')
    if bev_occ is not None:
        ax_dbev.imshow(bev_occ, origin='lower', extent=extent,
                       cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.25)
    ax_dbev.set_xlim(*xlim); ax_dbev.set_ylim(*ylim)
    ax_dbev.set_title("BEV Depth", fontsize=10)
    ax_dbev.set_xlabel('y (m)', fontsize=7)
    ax_dbev.set_ylabel('x (m)', fontsize=7)
    _draw_ego(ax_dbev)
    plt.colorbar(im_db, ax=ax_dbev, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [saved] {save_path}")

    return fig


def get_gt_boxes_ego(nusc, sample_token: str, bev_range: tuple) -> list:
    """
    Fetch all sample_annotations for this frame and return them as a list
    of dicts in EGO-CENTRIC coordinates, filtered to those within bev_range.

    Each dict:
        cx, cy      : box center in ego frame (metres)
        length, width: box dimensions (metres)
        yaw         : heading in ego frame (radians, 0=forward=ego-x)
        category    : nuScenes category name
        in_osz      : bool (filled in by the caller)
        token       : annotation token
    """
    import pyquaternion

    sample   = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep       = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_t    = np.array(ep['translation'], dtype=np.float64)
    ego_q    = pyquaternion.Quaternion(ep['rotation'])

    x_min, x_max, y_min, y_max = bev_range
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        delta  = np.array(ann['translation']) - ego_t
        pt_ego = ego_q.inverse.rotate(delta)

        cx, cy = float(pt_ego[0]), float(pt_ego[1])
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            continue

        box_q     = pyquaternion.Quaternion(ann['rotation'])
        box_q_ego = ego_q.inverse * box_q
        yaw_ego   = box_q_ego.yaw_pitch_roll[0]

        boxes.append({
            'cx':       cx,
            'cy':       cy,
            'length':   ann['size'][1],
            'width':    ann['size'][0],
            'yaw':      yaw_ego,
            'category': ann['category_name'],
            'token':    ann_token,
            'in_osz':   False,
        })
    return boxes


def _box_corners_ego(cx, cy, length, width, yaw):
    """Return (4,2) array of box corners in ego frame (x=fwd, y=left)."""
    cos_h, sin_h = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    local = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]])
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (R @ local.T).T + np.array([cx, cy])


def plot_gt_osz(
    osz_pa:        np.ndarray,
    bev_occ:       np.ndarray,
    drivable_mask: np.ndarray,
    nusc,
    sample_token:  str,
    bev_range:     tuple = (-50, 50, -50, 50),
    bev_res:       float = _DEFAULT_BEV_RES,
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Single-panel BEV GT overlay.
    Axes convention: horizontal = ego-y (left=+y), vertical = ego-x (up=forward).
    Box corners: plot(ego-y, ego-x) = plot(corners[:,1], corners[:,0]).
    """
    VEHICLE_CATS = {
        'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy',
        'vehicle.bus.rigid', 'vehicle.motorcycle', 'vehicle.bicycle',
        'vehicle.trailer', 'vehicle.construction',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police',
    }
    PERSON_CATS = {
        'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.wheelchair', 'human.pedestrian.stroller',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.construction_worker',
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)
    nx, ny = osz_pa.shape

    # ── background layers: same palette as PA_gen_v2 / plot_osz_explained ─
    overlay = np.zeros((nx, ny, 3), dtype=np.float32)
    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE['grass']
        overlay[drivable_mask] = _OSZ_PALETTE['road']
    else:
        overlay[:] = _OSZ_PALETTE['grass']
        overlay[~bev_occ] = _OSZ_PALETTE['road']

    # Orange obstacles first, then black OSZ shadow on top
    overlay[bev_occ] = _OSZ_PALETTE['obstacle']
    overlay[osz_pa] = _OSZ_PALETTE['osz']

    ax.imshow(overlay, origin='lower', extent=extent)

    # ── GT boxes ─────────────────────────────────────────────────────────
    boxes = get_gt_boxes_ego(nusc, sample_token, bev_range)

    for box in boxes:
        xi = int(np.rint((box['cx'] - x_min) / bev_res))
        yi = int(np.rint((y_max - box['cy']) / bev_res))
        if 0 <= xi < osz_pa.shape[0] and 0 <= yi < osz_pa.shape[1]:
            box['in_osz'] = bool(osz_pa[xi, yi])

    n_phantom = sum(1 for b in boxes
                    if b['in_osz'] and b['category'] in VEHICLE_CATS | PERSON_CATS)

    for box in boxes:
        corners = _box_corners_ego(
            box['cx'], box['cy'], box['length'], box['width'], box['yaw'])
        poly_ax_x = list(corners[:, 1]) + [corners[0, 1]]   # ego-y → horizontal
        poly_ax_y = list(corners[:, 0]) + [corners[0, 0]]   # ego-x → vertical

        cat = box['category']
        if cat in VEHICLE_CATS:
            if box['in_osz']:
                # phantom vehicle: vivid red with black outline so it pops on black OSZ
                base_color = '#ff0000'
                edge_color = '#ffffff'
                lw = 2.2
            else:
                base_color = '#ff9800'
                edge_color = '#e65100'
                lw = 2.0
        elif cat in PERSON_CATS:
            if box['in_osz']:
                base_color = '#ff0000'
                edge_color = '#ffffff'
                lw = 1.8
            else:
                base_color = '#7b1fa2'
                edge_color = '#4a148c'
                lw = 1.5
        else:
            base_color = '#AAAAAA'
            edge_color = '#666666'
            lw = 1.0

        ax.plot(poly_ax_x, poly_ax_y, color=edge_color, linewidth=lw + 0.6)
        ax.plot(poly_ax_x, poly_ax_y, color=base_color, linewidth=lw, alpha=0.95)

        # Forward arrow
        cos_h, sin_h = np.cos(box['yaw']), np.sin(box['yaw'])
        front_len = box['length'] * 0.4
        ax.annotate('',
            xy=(box['cy'] + sin_h * front_len,
                box['cx'] + cos_h * front_len),
            xytext=(box['cy'], box['cx']),
            arrowprops=dict(arrowstyle='->', color=base_color,
                            lw=lw * 0.8, mutation_scale=8),
        )

    # ── ego ───────────────────────────────────────────────────────────────
    _draw_ego(ax, size=2.5)

    # ── grid ─────────────────────────────────────────────────────────────
    for d in range(-40, 50, 10):
        ax.axhline(d, color='#dddddd', lw=0.4, alpha=0.8)
        ax.axvline(d, color='#dddddd', lw=0.4, alpha=0.8)
    ax.axhline(0, color='#999999', lw=0.8)
    ax.axvline(0, color='#999999', lw=0.8)

    # ── legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=_OSZ_PALETTE['road'], label='Road'),
        mpatches.Patch(facecolor=_OSZ_PALETTE['grass'], label='Non-drivable ground'),
        mpatches.Patch(facecolor=_OSZ_PALETTE['obstacle'], label='Occluder'),
        mpatches.Patch(facecolor=_OSZ_PALETTE['osz'],
                       label=f'PA-relevant OSZ ({osz_pa.sum()} cells)'),
        plt.Line2D([0],[0], color='#ff9800', lw=2, label='Vehicle (visible)'),
        plt.Line2D([0],[0], color='#7b1fa2', lw=2, label='Person (visible)'),
        plt.Line2D([0],[0], color='#ff0000', lw=2,
                   label=f'Vehicle/Person in OSZ ({n_phantom}) — phantom candidate'),
        mpatches.Patch(facecolor='#AAAAAA', label='Other object'),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc='upper right', framealpha=0.9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('y (m)  ← ego-left | ego-right →', fontsize=9,
                  color=_OSZ_PALETTE['text'])
    ax.set_ylabel('x (m)  ↑ forward', fontsize=9,
                  color=_OSZ_PALETTE['text'])
    ax.tick_params(labelsize=8, colors=_OSZ_PALETTE['text'])
    ax.set_title(
        f'BEV GT + PA-relevant OSZ  |  {sample_token[:16]}...\n'
        f'{len(boxes)} annotations  |  {n_phantom} phantom candidates',
        fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"  [saved] {save_path}")

    return fig


# ════════════════════════════════════════════════════════════════════
# Unified single-panel: "what is causing the shadow?"
#   Dark-grey road, green grass, orange occluders, black OSZ, blue ego.
#   Matches PA_gen_v2/visualize_events.py's PALETTE so the two tools
#   look visually consistent.
# ════════════════════════════════════════════════════════════════════

_OSZ_PALETTE = {
    'road':      (0.29, 0.29, 0.29),   # #4a4a4a dark grey road
    'grass':     (0.30, 0.69, 0.31),   # #4caf50 darker green
    'obstacle':  (1.00, 0.60, 0.00),   # #ff9800 orange occluders
    'osz':       (0.00, 0.00, 0.00),   # #000000 black shadow
    'ego':       (0.10, 0.46, 0.82),   # #1976d2 blue
    'lane':      (1.00, 1.00, 1.00),   # #ffffff white lane lines
    'bg':        (1.00, 1.00, 1.00),   # #ffffff white
    'text':      (0.13, 0.13, 0.13),   # #222222
    'text_mid':  (0.33, 0.33, 0.33),   # #555555
}


def plot_osz_explained(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask = None,
    bev_range: tuple = (-50, 50, -50, 50),
    sample_token: str = "",
    title_extra: str = "",
    save_path = None,
    draw_lanes: bool = False,
    nusc = None,
) -> plt.Figure:
    """Single-panel BEV showing exactly what casts the shadow and where it falls.

    Layers (bottom to top):
      - Grass green:       non-drivable ground
      - Dark grey:         drivable area / road surface
      - Orange:            voxel-cast occluders — the things CAUSING the shadow
      - Black:             PA-relevant OSZ — the shadow itself
      - White lane lines:  HD map lane boundaries (optional)
      - Blue triangle:     ego vehicle

    OSZ should radiate outward from orange occluders across the dark-grey road.
    OSZ on grass (green) or without a nearby occluder suggests a bug.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)
    nx, ny = osz_pa.shape

    # Build RGB overlay — same convention as PA_gen_v2/visualize_events.py
    overlay = np.zeros((nx, ny, 3), dtype=np.float32)

    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE['grass']
        overlay[drivable_mask] = _OSZ_PALETTE['road']
    else:
        overlay[:] = _OSZ_PALETTE['grass']
        overlay[~bev_occ] = _OSZ_PALETTE['road']

    # Orange occluders — Black OSZ shadow
    overlay[bev_occ] = _OSZ_PALETTE['obstacle']
    overlay[osz_pa] = _OSZ_PALETTE['osz']

    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _draw_ego(ax, size=2.5)

    # Lane lines from HD map
    if draw_lanes and nusc is not None:
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            import pyquaternion
            sample = nusc.get('sample', sample_token)
            log = nusc.get('log', sample['log_token'])
            location = log.get('location', '')
            lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            ego_t = np.array(ep['translation'], dtype=np.float64)
            ego_q = pyquaternion.Quaternion(ep['rotation'])
            if location:
                nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
                recs = nusc_map.get_records_in_radius(
                    float(ego_t[0]), float(ego_t[1]), 55.0, ['lane', 'road_segment'])
                for layer in ('lane', 'road_segment'):
                    for tok in recs.get(layer, []):
                        rec = nusc_map.get(layer, tok)
                        poly_rec = nusc_map.get('polygon', rec['polygon_token'])
                        nodes = [nusc_map.get('node', nt)
                                 for nt in poly_rec['exterior_node_tokens']]
                        pts = []
                        for nd in nodes:
                            gpos = np.array([nd['x'], nd['y'], 0.0], dtype=np.float32)
                            delta = gpos - ego_t
                            epos = ego_q.inverse.rotate(delta)
                            pts.append((epos[1], epos[0]))
                        if len(pts) >= 2:
                            xs, ys = zip(*pts)
                            ax.plot(xs, ys, '-', color=_OSZ_PALETTE['lane'],
                                    linewidth=0.4, alpha=0.5, zorder=1)
        except Exception:
            pass

    # Grid lines
    for d in range(-40, 50, 10):
        ax.axhline(d, color='#666666', lw=0.3, alpha=0.35)
        ax.axvline(d, color='#666666', lw=0.3, alpha=0.35)
    ax.axhline(0, color='#888888', lw=0.6)
    ax.axvline(0, color='#888888', lw=0.6)

    # Title
    n_osz = int(osz_pa.sum())
    n_occ = int(bev_occ.sum())
    pct_osz = n_osz / max(osz_pa.size, 1) * 100
    pct_occ = n_occ / max(bev_occ.size, 1) * 100
    title = (f"OSZ explained | {pct_osz:.1f}% OSZ ({n_osz} cells) "
             f"| {pct_occ:.1f}% occluders ({n_occ} cells)")
    if sample_token:
        title = f"{title}\n{sample_token[:24]}..."
    if title_extra:
        title = f"{title}  {title_extra}"
    ax.set_title(title, fontsize=10, fontweight='bold', color=_OSZ_PALETTE['text'])

    ax.set_xlabel('y (m)  \u2190 ego-left | ego-right \u2192', fontsize=8,
                  color=_OSZ_PALETTE['text_mid'])
    ax.set_ylabel('x (m)  \u2191 forward', fontsize=8,
                  color=_OSZ_PALETTE['text_mid'])
    ax.tick_params(labelsize=7)

    # Legend
    handles = [
        mpatches.Patch(color=_OSZ_PALETTE['road'], label='Road (drivable)'),
        mpatches.Patch(color=_OSZ_PALETTE['grass'], label='Non-drivable ground'),
        mpatches.Patch(color=_OSZ_PALETTE['obstacle'],
                       label=f'Occluders ({n_occ} cells) — cause the shadow'),
        mpatches.Patch(color=_OSZ_PALETTE['osz'],
                       label=f'OSZ ({n_osz} cells) — the shadow'),
        plt.Line2D([0], [0], marker='^', color=_OSZ_PALETTE['ego'],
                   markersize=8, linestyle='none', label='Ego'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"  [saved] {save_path}")

    return fig
