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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Dict, Optional, List
from pathlib import Path


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

def _bev_extent(bev_range):
    """Return (extent, xlim, ylim) for imshow/contour calls.

    extent = [y_max, y_min, x_min, x_max]
      – horizontal axis = ego-y, inverted so ego-LEFT is on the LEFT
      – vertical   axis = ego-x, forward is UP
    """
    x_min, x_max, y_min, y_max = bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


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


def _draw_ego(ax: plt.Axes, size: float = 2.0):
    """Draw ego vehicle as a small rectangle + forward arrow.

    In BEV axes: x-axis = ego-y, y-axis = ego-x.
    Arrow goes from (ax_x=0, ax_y=size) → (ax_x=0, ax_y=size*1.5):
        ax_x stays 0 → ego-y=0 (centerline) ✓
        ax_y increases → ego-x increases → FORWARD ✓
    """
    rect = plt.Rectangle((-size / 2, -size), size, size * 2,
                          linewidth=1, edgecolor='white', facecolor='white', alpha=0.7)
    ax.add_patch(rect)
    ax.annotate('', xy=(0, size * 1.5), xytext=(0, size),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))


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

    shadow_cmap = ListedColormap(['none', '#E24B4A'])

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

    patches = [mpatches.Patch(color='#E24B4A', label='OSZ')]
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
                 levels=[0.5], colors=['#E24B4A'], linewidths=0.8,
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
    bev_range: tuple = (-50, 50, -50, 50),
    sample_token: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    每行展示一个相机的：原图 → 稠密深度图 → 该相机视角的 BEV 阴影掩码。
    最后一行展示融合结果：OSZ Raw → OSZ Refined → BEV Depth。
    大标题带上 frame token，方便识别当前帧。
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
    shadow_cmap = ListedColormap(['none', '#E24B4A'])

    for i, cam_name in enumerate(cam_names):
        ax_img   = axes[i][0]
        ax_depth = axes[i][1]
        ax_bev   = axes[i][2]

        # 原图
        ax_img.imshow(images[cam_name])
        ax_img.set_title(f"{cam_name} — Image", fontsize=10)
        ax_img.axis('off')

        # 稠密深度图
        dmap = depth_maps[cam_name]
        im_d = ax_depth.imshow(dmap, cmap='plasma', vmin=0, vmax=70)
        ax_depth.set_title(f"{cam_name} — Dense Depth", fontsize=10)
        ax_depth.axis('off')
        plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)

        # 该相机的 BEV 阴影掩码
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

    # ── 最后一行：融合结果 ────────────────────────────────────────────────
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
    if bev_occ is not None:
        ax_ref.imshow(bev_occ, origin='lower', extent=extent,
                      cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.4)
    if refined_mask is not None:
        ax_ref.imshow(refined_mask, origin='lower', extent=extent,
                      cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
        _bev_contour(ax_ref, refined_mask, bev_range,
                     levels=[0.5], colors=['#FF8800'], linewidths=1.5)
    ax_ref.set_xlim(*xlim); ax_ref.set_ylim(*ylim)
    ax_ref.set_title("OSZ Refined (CRF)", fontsize=10)
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


def plot_pa_osz(
    osz_raw:       np.ndarray,
    osz_pa:        np.ndarray,
    drivable_mask: np.ndarray,
    bev_occ:       Optional[np.ndarray],
    bev_range:     tuple = (-50, 50, -50, 50),
    sample_token:  str = "",
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel comparison showing the effect of the drivable-area filter:
      Panel 1: Drivable area (green) overlaid on raw OSZ (red)
      Panel 2: PA-relevant OSZ = raw OSZ ∩ drivable area (orange)
      Panel 3: Side-by-side cell counts as a bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)

    shadow_cmap   = ListedColormap(['none', '#E24B4A'])
    drivable_cmap = ListedColormap(['none', '#2ECC71'])
    pa_cmap       = ListedColormap(['none', '#FF8C00'])

    # ── Panel 1: raw OSZ + drivable overlay ──────────────────────────────
    ax = axes[0]
    if bev_occ is not None:
        ax.imshow(bev_occ, origin='lower', extent=extent,
                  cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.35)
    ax.imshow(osz_raw, origin='lower', extent=extent,
              cmap=shadow_cmap, vmin=0, vmax=1, interpolation='nearest', alpha=0.7)
    ax.imshow(drivable_mask, origin='lower', extent=extent,
              cmap=drivable_cmap, vmin=0, vmax=1, interpolation='nearest', alpha=0.45)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title("Raw OSZ (red) + Drivable area (green)", fontsize=10)
    ax.set_xlabel('y (m)', fontsize=8); ax.set_ylabel('x (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    _draw_ego(ax)
    patches = [
        mpatches.Patch(color='#E24B4A', label=f'Raw OSZ ({osz_raw.sum()} cells)'),
        mpatches.Patch(color='#2ECC71', alpha=0.7,
                       label=f'Drivable ({drivable_mask.sum()} cells)'),
    ]
    ax.legend(handles=patches, fontsize=7, loc='upper right')

    # ── Panel 2: PA-relevant OSZ ─────────────────────────────────────────
    ax = axes[1]
    if bev_occ is not None:
        ax.imshow(bev_occ, origin='lower', extent=extent,
                  cmap='Greys', vmin=0, vmax=1, interpolation='nearest', alpha=0.35)
    ax.imshow(drivable_mask, origin='lower', extent=extent,
              cmap=drivable_cmap, vmin=0, vmax=1, interpolation='nearest', alpha=0.25)
    ax.imshow(osz_pa, origin='lower', extent=extent,
              cmap=pa_cmap, vmin=0, vmax=1, interpolation='nearest', alpha=0.9)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    kept_pct = osz_pa.sum() / max(osz_raw.sum(), 1) * 100
    ax.set_title(f"PA-relevant OSZ (orange)\n"
                 f"{osz_pa.sum()} cells = {kept_pct:.0f}% of raw OSZ retained",
                 fontsize=10)
    ax.set_xlabel('y (m)', fontsize=8); ax.set_ylabel('x (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    _draw_ego(ax)

    # ── Panel 3: bar chart summary ────────────────────────────────────────
    ax = axes[2]
    total_cells = osz_raw.size
    labels = ['Total BEV\ngrid', 'Raw OSZ\n(geometric)', 'Drivable\narea',
              'PA-relevant\nOSZ']
    values = [total_cells, osz_raw.sum(), drivable_mask.sum(), osz_pa.sum()]
    colors = ['#95A5A6', '#E24B4A', '#2ECC71', '#FF8C00']
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        pct = val / total_cells * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_cells * 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel('BEV cells', fontsize=9)
    ax.set_title("Coverage breakdown", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, total_cells * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    fig.suptitle(f"PA-relevant OSZ filter — {sample_token[:16]}...",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

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
    bev_res:       float = 0.4,
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

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)

    # ── background layers ─────────────────────────────────────────────────
    drivable_cmap = ListedColormap(['#1a1a1a', '#1a4a2a'])
    _bev_imshow(ax, drivable_mask.astype(np.float32), bev_range,
                cmap=drivable_cmap, vmin=0, vmax=1, alpha=0.6)

    pa_cmap = ListedColormap(['none', '#FF8C00'])
    _bev_imshow(ax, osz_pa.astype(np.float32), bev_range,
                cmap=pa_cmap, vmin=0, vmax=1, alpha=0.5)

    if bev_occ is not None:
        occ_cmap = ListedColormap(['none', '#888888'])
        _bev_imshow(ax, bev_occ.astype(np.float32), bev_range,
                    cmap=occ_cmap, vmin=0, vmax=1, alpha=0.6)

    # ── GT boxes ─────────────────────────────────────────────────────────
    boxes = get_gt_boxes_ego(nusc, sample_token, bev_range)

    for box in boxes:
        xi = int((box['cx'] - x_min) / bev_res)
        yi = int((box['cy'] - y_min) / bev_res)
        if 0 <= xi < osz_pa.shape[0] and 0 <= yi < osz_pa.shape[1]:
            box['in_osz'] = bool(osz_pa[xi, yi])

    n_phantom = sum(1 for b in boxes
                    if b['in_osz'] and b['category'] in VEHICLE_CATS | PERSON_CATS)

    for box in boxes:
        corners = _box_corners_ego(
            box['cx'], box['cy'], box['length'], box['width'], box['yaw'])
        # In BEV axes: horizontal = ego-y, vertical = ego-x
        poly_ax_x = list(corners[:, 1]) + [corners[0, 1]]   # ego-y → horizontal
        poly_ax_y = list(corners[:, 0]) + [corners[0, 0]]   # ego-x → vertical

        cat = box['category']
        if cat in VEHICLE_CATS:
            base_color = '#E74C3C' if box['in_osz'] else '#2ECC71'
            lw = 2.0
        elif cat in PERSON_CATS:
            base_color = '#E74C3C' if box['in_osz'] else '#3498DB'
            lw = 1.5
        else:
            base_color = '#AAAAAA'
            lw = 1.0

        ax.plot(poly_ax_x, poly_ax_y, color=base_color, linewidth=lw, alpha=0.9)

        # Forward tick: from box centre → front face centre, in BEV axes
        cos_h, sin_h = np.cos(box['yaw']), np.sin(box['yaw'])
        front_len = box['length'] * 0.4
        # ego centre: (cx, cy),  ego front: (cx+cos_h*front_len, cy+sin_h*front_len)
        ax.annotate('',
            xy=(box['cy'] + sin_h * front_len,   # ax_x = ego-y
                box['cx'] + cos_h * front_len),  # ax_y = ego-x
            xytext=(box['cy'], box['cx']),
            arrowprops=dict(arrowstyle='->', color=base_color,
                            lw=lw * 0.8, mutation_scale=8),
        )

    # ── ego ───────────────────────────────────────────────────────────────
    _draw_ego(ax, size=2.5)

    # ── grid ─────────────────────────────────────────────────────────────
    for d in range(-40, 50, 10):
        ax.axhline(d, color='#333333', lw=0.4, alpha=0.5)
        ax.axvline(d, color='#333333', lw=0.4, alpha=0.5)
    ax.axhline(0, color='#555555', lw=0.8)
    ax.axvline(0, color='#555555', lw=0.8)

    # ── legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor='#1a4a2a', label='Drivable area'),
        mpatches.Patch(facecolor='#FF8C00', alpha=0.7,
                       label=f'PA-relevant OSZ ({osz_pa.sum()} cells)'),
        mpatches.Patch(facecolor='#888888', label='Occluder surface'),
        plt.Line2D([0],[0], color='#2ECC71', lw=2, label='Vehicle (visible)'),
        plt.Line2D([0],[0], color='#3498DB', lw=2, label='Person (visible)'),
        plt.Line2D([0],[0], color='#E74C3C', lw=2,
                   label=f'Vehicle/Person in OSZ ({n_phantom}) — phantom candidate'),
        mpatches.Patch(facecolor='#AAAAAA', label='Other object'),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc='upper right', framealpha=0.85)

    ax.set_xlim(*xlim)   # left=y_max=ego-left, right=y_min=ego-right
    ax.set_ylim(*ylim)   # bottom=x_min, top=x_max (forward=UP)
    ax.set_xlabel('y (m)  ← ego-left | ego-right →', fontsize=9)
    ax.set_ylabel('x (m)  ↑ forward', fontsize=9)
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
