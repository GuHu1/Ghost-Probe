"""
visualize_events.py
-------------------
Visualize mined ghost vehicle events — the most important debugging step.

Three modes (pick with CLI flag):

  1. OFFLINE EXPORT (no flag): render a grid of events to PNG via
     make_event_grid(). Headless, works anywhere (Agg or any backend;
     savefig does not need a GUI). Fastest for batch QA.

  2. INTERACTIVE BROWSER (--browse): three sub-modes share the same
     renderer (HeadlessEventBrowser._render) so the visuals are identical:
       a. GUI (default if tkinter/Qt available): n/p/r/q keyboard
          navigation in a matplotlib window
       b. Headless (--headless): terminal stdin n/p/j/k/numbers/r/q,
          each event saved to output/browser/event_XXXX.png so you can
          flip through with any image viewer
       c. Web gallery (--web): render all events to output/web/ and
          build a browser-viewable index.html. ←/→/j/k keys. Most
          recommended for reviewing many events; relies on the OSZ
          disk cache for fast second-run load.

     All three render this 2x3 layout (each panel uses that frame's OWN
     ego frame, ego at origin, showing the frame's OWN PA-relevant OSZ):

        +---------+---------+---------+
        |  t-4    |  t-3    |  t-2    |
        +---------+---------+---------+
        |  t-1    |   t     |  info   |
        +---------+---------+---------+

     Per-panel layers (zorder low->high):
       - drivable area (dark green)
       - voxel-cast obstacles (gray)
       - PA-relevant OSZ (red)
       - HD map lane boundaries (thin blue, from NuScenesMap)
       - other scene vehicles (cyan BEV boxes with heading)
       - pedestrians (yellow dots)
       - the tracked vehicle (verdict-coloured box + white heading arrow)
       - ego marker (white triangle) at frame's own origin

     Why "own ego frame" per panel: each frame's OSZ is computed in
     that frame's ego coordinates by osz_source (ego sits at the
     origin every time). Plotting each panel in its own ego frame
     therefore shows the TRUE per-frame OSZ geometry — you can watch
     OSZ grow/shrink/rotate as ego moves and occluders shift.
     Collapsing all frames into frame-t's ego frame (what
     visualize_event() does for the offline grid) would warp every
     other frame's OSZ and hide exactly the motion this browser
     exists to reveal.

All plotting is in metric ego coordinates to avoid pixel-index rounding
and axis-order mistakes.

Karpathy rule: if you cannot look at the output and immediately see that
it makes geometric sense, the mining logic is wrong. Do NOT proceed to
model training without this visual check.
"""

import sys
import argparse
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
# NOTE: we deliberately do NOT call matplotlib.use('Agg') at import time.
# The original code did, which silently broke plt.show() for the interactive
# browser. savefig() works under ANY backend (it does not need Agg), so
# removing the forced Agg keeps offline PNG export working while letting the
# browser open a real window. _ensure_gui_backend() below switches to a GUI
# backend on demand when the browser launches.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from nuscenes.nuscenes import NuScenes
import pyquaternion

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osz_source
from ghost_vehicle_miner import _get_ego_pose, _global_to_ego, load_events, LOOKBACK_FRAMES
from trajectory import build_instance_trajectories, locate_at_time, NO_EVIDENCE


# ─────────────────────────────────────────────────────────────────────
# Verdict colour map — shared between offline + interactive views so the
# two are visually consistent. Keys mirror the was_in_osz contract:
#   True  = confirmed occluded (in OSZ)
#   False = confirmed visible
#   None  = no evidence either way
#   'emerged' = frame t (the emergence frame)
# ─────────────────────────────────────────────────────────────────────
VERDICT_STYLE = {
    True:      ('#dc2626', 'o'),   # confirmed in OSZ  (red)
    False:     ('#2563eb', 'o'),   # confirmed visible (blue)
    None:      ('#6b7280', 'x'),   # no evidence (gray)
    'emerged': ('#16a34a', '*'),   # emerged at frame t (green)
}


# ─────────────────────────────────────────────────────────────────────
# Unified light-theme colour palette (academic-style BEV figure)
# ─────────────────────────────────────────────────────────────────────
# White background, light-grey road, black occlusion shadow, orange/blue
# vehicles — matches the reference style the user provided.
PALETTE = {
    # BEV base layers
    'bg':              '#f8f8f7',   # figure background (off-white)
    'panel_bg':        '#ffffff',   # BEV panel background
    'offroad':         '#f0f0ea',   # non-drivable ground (very light beige)
    'road':            '#e6e6e6',   # drivable area / road surface
    'obstacle':        '#7f7f7f',   # voxel-cast obstacles (walls, vehicles)
    'osz':             '#1a1a1a',   # PA-relevant OSZ (black shadow)
    'lane':            '#5a5a5a',   # HD-map lane lines

    # Vehicles / agents
    'ego':             '#2563eb',   # blue ego marker
    'other_vehicle':   '#f97316',   # orange other vehicles
    'other_vehicle_edge': '#b45309',
    'pedestrian':      '#333333',   # dark grey pedestrian dot
    'tracked_arrow':   '#111111',   # heading arrow for tracked vehicle

    # Text / UI
    'text_dark':       '#222222',
    'text_mid':        '#555555',
    'text_light':      '#888888',
    'spine':           '#cccccc',
    'error':           '#dc2626',

    # Info panel (light theme)
    'info_bg':         '#f8f9fa',
    'info_separator':  '#dddddd',
    'info_title':      '#111111',
    'info_key':        '#2563eb',   # keyboard hint blue
}


def _hex_to_rgb(h: str):
    """Convert '#rrggbb' to (r,g,b) in [0,1]."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))



def _bev_extent(caster):
    """
    Same convention as OSZ/visualize/bev_viz.py's _bev_extent: forward=UP,
    ego-left=LEFT, no transpose applied to the (nx,ny) array. Kept local
    here (not imported) to avoid a hard dependency of PA_gen_v2/ on OSZ's
    plotting internals — this is 6 lines of pure arithmetic, cheap to
    duplicate, easy to eyeball-verify against the OSZ/ original if ever
    in doubt.
    """
    x_min, x_max, y_min, y_max = caster.bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


# ════════════════════════════════════════════════════════════════════
# OFFLINE: single-event drawer (collapses all lookback into frame-t ego)
# ════════════════════════════════════════════════════════════════════

def visualize_event(nusc: NuScenes, event: Dict, ax: plt.Axes,
                    title: str = '') -> None:
    """Draw one ghost vehicle event onto ax, in metric ego coordinates.

    All lookback positions are transformed into frame-t's ego frame so the
    whole trajectory reads as one continuous line. Use the interactive
    EventBrowser instead if you need to see each frame's OWN OSZ.
    """
    emerge_tok   = event['emerge_sample']
    instance_tok = event['instance_token']
    lb_tokens    = event['lookback_tokens']
    was_in_osz   = event['was_in_osz']   # list of True / False / None

    bev_occ, osz_raw, osz_pa, drivable_mask = \
        osz_source.get_pa_relevant_osz_for_sample(nusc, emerge_tok)
    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    # bev_occ / osz_pa are (nx, ny) with axis-0=ego-x, axis-1=ego-y —
    # imshow with this extent and NO transpose places axis-0 on the
    # vertical (forward=up) axis, matching OSZ/visualize/bev_viz.py.
    # We show PA-relevant OSZ (raw OSZ ∩ drivable area) here, not raw
    # OSZ, because that's what ghost_vehicle_miner.py actually used to
    # make every was_in_osz decision below — showing raw OSZ here would
    # make correct mining decisions look wrong (raw OSZ is usually much
    # larger, including building shadows the miner never counted).
    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    # Light base: white background
    overlay[:] = _hex_to_rgb(PALETTE['panel_bg'])
    # Road surface (drivable area)
    overlay[drivable_mask] = _hex_to_rgb(PALETTE['road'])
    # Solid obstacles / voxel-cast surfaces
    overlay[bev_occ] = _hex_to_rgb(PALETTE['obstacle'])
    # PA-relevant OSZ (black shadow on drivable area)
    overlay[osz_pa] = _hex_to_rgb(PALETTE['osz'])
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor(PALETTE['panel_bg'])

    # Ego marker at (0,0) in metric coords — blue triangle.
    ax.plot(0, 0, '^', color=PALETTE['ego'], markersize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Emerged vehicle position (frame t), already stored in metric ego xy
    ex, ey = event['emerge_bev_xy']
    ax.plot(ey, ex, '*', color=PALETTE['tracked_arrow'], markersize=14,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
            label='Emerged (t)')
    # NOTE: plotted as (ey, ex) not (ex, ey) — matplotlib's x-axis here is
    # ego-y (horizontal) and y-axis is ego-x (forward), matching the
    # imshow extent above. This is the same axis mapping OSZ/'s _draw_ego
    # uses; getting this backwards is exactly the silent-transpose risk
    # documented in osz_source.py.

    ego_t, ego_q_t = _get_ego_pose(nusc, emerge_tok)

    traj_x, traj_y = [], []  # metric (ego-y, ego-x) for the trajectory line

    for i, (lb_tok, in_osz) in enumerate(zip(lb_tokens, was_in_osz)):
        lb_global = _get_vehicle_global_pos(nusc, instance_tok, lb_tok)
        if lb_global is None:
            continue

        ego_lb, ego_q_lb = _get_ego_pose(nusc, lb_tok)
        # Transform into frame-t's ego frame so everything is drawn in one
        # consistent reference frame (frame t may not be lb_tok's own frame)
        pt_ego_in_t = _global_to_ego(lb_global, ego_t, ego_q_t)

        if not osz_source.in_bev_range(pt_ego_in_t[0], pt_ego_in_t[1]):
            continue

        px, py = pt_ego_in_t[1], pt_ego_in_t[0]   # (ego-y, ego-x) for plotting
        traj_x.append(px)
        traj_y.append(py)

        color, marker = VERDICT_STYLE.get(in_osz, VERDICT_STYLE[None])
        ax.plot(px, py, marker, color=color, markersize=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        ax.text(px + 0.8, py + 0.8, f't-{len(lb_tokens)-i}',
                color=PALETTE['text_dark'], fontsize=5,
                path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    if len(traj_x) >= 2:
        traj_x.append(ey)
        traj_y.append(ex)
        ax.plot(traj_x, traj_y, '-', color=PALETTE['text_mid'], linewidth=1.2, alpha=0.7)

    n_osz  = sum(1 for v in was_in_osz if v is True)
    n_unk  = sum(1 for v in was_in_osz if v is None)
    if not title:
        title = (f"Ghost event | {n_osz}/{len(lb_tokens)} lb frames in OSZ"
                 + (f" ({n_unk} unknown)" if n_unk else ""))
    ax.set_title(title, fontsize=7, color=PALETTE['text_dark'])
    ax.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=6, color=PALETTE['text_mid'])
    ax.set_ylabel('x (m) ↑ forward', fontsize=6, color=PALETTE['text_mid'])
    ax.tick_params(labelsize=5, colors=PALETTE['text_dark'])
    for s in ax.spines.values():
        s.set_color(PALETTE['spine'])

    legend_patches = [
        mpatches.Patch(color=_hex_to_rgb(PALETTE['obstacle']), label='Occupied'),
        mpatches.Patch(color=_hex_to_rgb(PALETTE['osz']), label='PA-relevant OSZ'),
        plt.Line2D([0],[0], marker='*', color=PALETTE['tracked_arrow'], markersize=8,
                   linestyle='none', label='Emerged (t)'),
        plt.Line2D([0],[0], marker='o', color=VERDICT_STYLE[True][0], markersize=5,
                   linestyle='none', label='Lookback (confirmed in OSZ)'),
        plt.Line2D([0],[0], marker='o', color=VERDICT_STYLE[False][0], markersize=5,
                   linestyle='none', label='Lookback (confirmed visible)'),
        plt.Line2D([0],[0], marker='x', color=VERDICT_STYLE[None][0], markersize=5,
                   linestyle='none', label='Lookback (no evidence)'),
    ]
    ax.legend(handles=legend_patches, fontsize=5, loc='upper right', framealpha=0.9)


def _get_vehicle_global_pos(nusc: NuScenes, instance_token: str,
                            sample_token: str):
    """Global position of an instance in a specific sample, or None."""
    sample = nusc.get('sample', sample_token)
    for at in sample['anns']:
        ann = nusc.get('sample_annotation', at)
        if ann['instance_token'] == instance_token:
            return np.array(ann['translation'], dtype=np.float32)
    return None


# ════════════════════════════════════════════════════════════════════
# INTERACTIVE EVENT BROWSER
# ════════════════════════════════════════════════════════════════════

def _ensure_gui_backend() -> str:
    """
    Switch to a GUI-capable matplotlib backend if the active one is
    non-interactive (Agg / inline). savefig() works under any backend, but
    plt.show() only opens a window under a GUI backend — the interactive
    EventBrowser needs that. TkAgg is preferred because tkinter ships with
    CPython on Windows/macOS (no extra install); Qt fallbacks follow.

    Returns the now-active backend name. Raises RuntimeError if no GUI
    backend is importable.
    """
    cur = matplotlib.get_backend().lower()
    # Already interactive enough for show()
    if 'agg' not in cur and 'inline' not in cur and 'nbagg' not in cur:
        return matplotlib.get_backend()

    candidates = []
    # ── Native windows FIRST — these open a real OS window locally.
    # TkAgg is preferred: tkinter ships with CPython on Windows/macOS and
    # needs no extra install.
    try:
        import tkinter  # noqa: F401  (probe only)
        candidates.append('TkAgg')
    except Exception:
        pass
    for mod, be in (('PyQt5', 'Qt5Agg'), ('PyQt6', 'QtAgg'),
                    ('PySide6', 'QtAgg'), ('PySide2', 'Qt5Agg')):
        try:
            __import__(mod)
            candidates.append(be)
        except Exception:
            pass
    # ── WebAgg LAST — serves interactive plots over HTTP (open in browser).
    # Only useful when no native window is possible (e.g. headless remote
    # without display). Picked last so local users get a real window.
    try:
        import tornado  # noqa: F401  (webagg dependency)
        candidates.append('WebAgg')
    except Exception:
        pass

    for be in candidates:
        try:
            matplotlib.use(be, force=True)
            # No reload of pyplot: use(force=True) retargets the default
            # backend, and pyplot picks it up when it next creates a figure
            # manager. Reloading pyplot would detach the module object
            # already bound to our top-level `plt` name.
            return matplotlib.get_backend()
        except Exception:
            continue

    raise RuntimeError(
        "No interactive matplotlib backend available. Install tkinter "
        "(usually bundled with Python) or PyQt5/PyQt6 to use the event "
        "browser. For headless PNG export, run without --browse."
    )


def _vehicle_pos_in_frame_ego(nusc: NuScenes, instance_token: str,
                              sample_token: str, traj) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Vehicle (x_ego, y_ego) in sample_token's OWN ego frame.

    Tries a direct annotation first; if absent, falls back to trajectory
    interpolation (trajectory.locate_at_time). This MIRRORS
    ghost_vehicle_miner's lookback logic exactly, so the position drawn
    here agrees with how the was_in_osz verdict was actually produced —
    the QA view cannot disagree with the miner by construction.

    Returns (pos_xy_or_None, status) with status in:
        'known'         — exact annotation
        'interpolated'  — bracketed by the track, linearly interpolated
        'no_evidence'   — outside the track's known span (no position)
    """
    ego, ego_q = _get_ego_pose(nusc, sample_token)

    gpos = _get_vehicle_global_pos(nusc, instance_token, sample_token)
    if gpos is not None:
        pt = _global_to_ego(gpos, ego, ego_q)
        return (float(pt[0]), float(pt[1])), 'known'

    if traj:
        ts = nusc.get('sample', sample_token)['timestamp']
        status, ginterp = locate_at_time(traj, ts)
        if status == NO_EVIDENCE or ginterp is None:
            return None, 'no_evidence'
        pt = _global_to_ego(ginterp, ego, ego_q)
        return (float(pt[0]), float(pt[1])), status

    return None, 'no_evidence'


# ════════════════════════════════════════════════════════════════════
# BEV GROUND-TRUTH OVERLAY — all annotated objects + HD-map lanes
# ════════════════════════════════════════════════════════════════════
# These functions let you see the FULL scene (every vehicle, pedestrian,
# the road network) on top of the OSZ — so you can verify at a glance
# whether the ghost vehicle was genuinely occluded and whether the OSZ
# geometry makes sense given the surrounding traffic and infrastructure.

_map_cache: Dict[str, object] = {}


def _bev_box_corners_ego(x_ego, y_ego, heading, w, l):
    """4 corners of a BEV box in ego frame (x=forward, y=left).

    heading=0 → box faces +x (forward).  Positive heading = left turn
    (counterclockwise, toward +y).  w=width (lateral), l=length (forward).
    Returns (4, 2) array of (x_ego, y_ego).
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    local = np.array([[ l/2,  w/2],
                      [ l/2, -w/2],
                      [-l/2, -w/2],
                      [-l/2,  w/2]])
    corners = local @ R.T
    corners[:, 0] += x_ego
    corners[:, 1] += y_ego
    return corners


def _get_scene_annotations_ego(nusc: NuScenes, sample_token: str) -> List[Dict]:
    """All annotated objects in this sample, transformed to ego frame.

    Each item: {x, y, heading, w, l, category, instance}.
    x=forward, y=left, heading in radians (0=forward, +=left).
    """
    sample = nusc.get('sample', sample_token)
    ego_t, ego_q = _get_ego_pose(nusc, sample_token)

    anns = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        gpos = np.array(ann['translation'], dtype=np.float32)
        epos = _global_to_ego(gpos, ego_t, ego_q)

        ann_q = pyquaternion.Quaternion(ann['rotation'])
        rel_q = ego_q.inverse * ann_q          # ann orientation in ego frame
        heading = float(rel_q.yaw_pitch_roll[0])

        w, l, h = ann['size']                  # nuScenes: [width, length, height]
        anns.append({
            'x': float(epos[0]),
            'y': float(epos[1]),
            'heading': heading,
            'w': float(w), 'l': float(l),
            'category': ann['category_name'],
            'instance': ann['instance_token'],
        })
    return anns


def _draw_annotation_boxes(ax, anns: List[Dict],
                           tracked_instance: str = None,
                           tracked_verdict=None) -> None:
    """Draw all annotation BEV boxes on ax (axes uses ego-y, ego-x)."""
    for a in anns:
        if not osz_source.in_bev_range(a['x'], a['y']):
            continue

        is_tracked = (tracked_instance is not None and
                      a['instance'] == tracked_instance)
        cat = a['category']

        if cat.startswith('human'):
            # Pedestrians: small dark dots, no box (too small to matter at BEV scale)
            ax.plot(a['y'], a['x'], 'o', color=PALETTE['pedestrian'],
                    markersize=2.5, alpha=0.8)
            continue

        corners = _bev_box_corners_ego(a['x'], a['y'], a['heading'],
                                       a['w'], a['l'])
        # Swap to plot coords: matplotlib x-axis=ego-y, y-axis=ego-x
        plot_pts = corners[:, [1, 0]]

        if is_tracked:
            vcolor, _ = VERDICT_STYLE.get(tracked_verdict, VERDICT_STYLE[None])
            poly = mpatches.Polygon(plot_pts, closed=True,
                                    facecolor=vcolor, edgecolor='black',
                                    alpha=0.35, linewidth=1.5, zorder=5)
            ax.add_patch(poly)
            # Heading arrow for the tracked vehicle
            dx = np.cos(a['heading']) * 2.5
            dy = np.sin(a['heading']) * 2.5
            ax.annotate('', xy=(a['y'] + dy, a['x'] + dx),
                        xytext=(a['y'], a['x']),
                        arrowprops=dict(arrowstyle='->', color=PALETTE['tracked_arrow'],
                                        lw=1.5), zorder=6)
        elif cat.startswith('vehicle'):
            poly = mpatches.Polygon(plot_pts, closed=True,
                                    facecolor=PALETTE['other_vehicle'], edgecolor=PALETTE['other_vehicle_edge'],
                                    alpha=0.25, linewidth=0.6, zorder=3)
            ax.add_patch(poly)
        else:
            # Barriers, cones, construction objects, etc.
            poly = mpatches.Polygon(plot_pts, closed=True,
                                    facecolor='#9ca3af', edgecolor='#4b5563',
                                    alpha=0.18, linewidth=0.4, zorder=3)
            ax.add_patch(poly)


def _draw_map_overlay(ax, nusc: NuScenes, sample_token: str) -> None:
    """Draw lane boundaries from the nuScenes HD map (if available).

    Uses the node-based polygon access (no shapely dependency) and
    transforms global-map coordinates to ego frame.  Silently skips
    if the map expansion module isn't installed or the location has
    no map data.
    """
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
    except ImportError:
        return

    try:
        sample = nusc.get('sample', sample_token)
        log = nusc.get('log', sample['log_token'])
        location = log['location']
        if not location:
            return

        if location not in _map_cache:
            _map_cache[location] = NuScenesMap(
                dataroot=nusc.dataroot, map_name=location)
        nusc_map = _map_cache[location]

        ego_t, ego_q = _get_ego_pose(nusc, sample_token)
        gx, gy, _ = ego_t

        # Lanes within 55 m of ego — covers the full BEV extent (±50 m)
        records = nusc_map.get_records_in_radius(
            float(gx), float(gy), 55.0, ['lane', 'road_segment'])

        for layer in ('lane', 'road_segment'):
            for tok in records.get(layer, []):
                rec = nusc_map.get(layer, tok)
                poly_rec = nusc_map.get('polygon', rec['polygon_token'])
                nodes = [nusc_map.get('node', nt)
                         for nt in poly_rec['exterior_node_tokens']]
                pts = []
                for nd in nodes:
                    gpos = np.array([nd['x'], nd['y'], 0.0], dtype=np.float32)
                    epos = _global_to_ego(gpos, ego_t, ego_q)
                    pts.append((epos[1], epos[0]))   # (ego-y, ego-x)
                if len(pts) >= 2:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, '-', color=PALETTE['lane'],
                            linewidth=0.4, alpha=0.5, zorder=1)
    except Exception:
        pass    # map not available / wrong location / etc — skip silently


def _draw_frame_own_ego(ax, nusc, sample_token, instance_token,
                        verdict, frame_label, traj) -> Optional[float]:
    """
    Draw ONE frame's BEV in that frame's OWN ego coordinate system:
      - PA-relevant OSZ (red) over occupancy (gray) — this frame's own
      - ego marker at the origin (white triangle)
      - the tracked vehicle's position in this frame's ego frame, coloured
        by verdict (red=in OSZ, blue=visible, gray=unknown, green=emerged)

    Returns that frame's PA-relevant OSZ coverage (% of grid), or None if
    OSZ computation failed for this frame (caller will show an error note).

    Why "own ego frame" per panel: each frame's OSZ is computed in that
    frame's ego coordinates by osz_source (ego sits at the origin every
    time). Plotting each panel in its own ego frame therefore shows the
    TRUE per-frame OSZ geometry — you can watch OSZ grow/shrink/rotate as
    ego moves and occluders shift. Collapsing all frames into frame-t's
    ego frame (what visualize_event() does) would warp every other frame's
    OSZ and hide exactly the motion this browser exists to reveal.
    """
    ax.clear()
    ax.set_facecolor(PALETTE['panel_bg'])

    try:
        bev_occ, osz_raw, osz_pa, drivable_mask = \
            osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)
    except Exception as ex:
        ax.axis('off')
        ax.set_facecolor(PALETTE['panel_bg'])
        ax.text(0.5, 0.5, f"OSZ failed\n{ex}",
                transform=ax.transAxes, ha='center', va='center',
                color=PALETTE['error'], fontsize=8)
        ax.set_title(f'{frame_label}  (OSZ ERROR)', fontsize=8, color=PALETTE['error'])
        return None

    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[:] = _hex_to_rgb(PALETTE['panel_bg'])  # white base
    overlay[drivable_mask] = _hex_to_rgb(PALETTE['road'])  # light grey road
    overlay[bev_occ] = _hex_to_rgb(PALETTE['obstacle'])  # grey obstacles
    overlay[osz_pa] = _hex_to_rgb(PALETTE['osz'])  # black OSZ shadow
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5, colors=PALETTE['text_dark'])
    for s in ax.spines.values():
        s.set_color(PALETTE['spine'])

    # HD-map lane boundaries (thin grey lines) — adds road context.
    _draw_map_overlay(ax, nusc, sample_token)

    # Ego at origin of THIS frame's ego frame — blue triangle.
    ax.plot(0, 0, '^', color=PALETTE['ego'], markersize=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    # Heading tick: ego always faces +x (forward=up) in its own frame.
    ax.annotate('', xy=(0, 3.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=PALETTE['text_dark'], lw=1.0))

    # All scene annotations (vehicles=orange boxes, peds=dark dots,
    # tracked vehicle=verdict-coloured box + heading arrow).
    _anns = _get_scene_annotations_ego(nusc, sample_token)
    _draw_annotation_boxes(ax, _anns, tracked_instance=instance_token,
                           tracked_verdict=verdict)

    coverage_pct = float(osz_pa.mean()) * 100.0

    # Vehicle position in this frame's own ego frame. This function only
    # handles LOOKBACK frames (verdict in {True, False, None}); frame t
    # ('emerged') is drawn by _draw_frame_own_ego_emerged, which uses the
    # event's stored emerge_bev_xy verbatim instead of re-deriving it —
    # keeping these two paths separate guarantees the green star matches
    # the miner's recorded emergence point exactly.
    pos, status = _vehicle_pos_in_frame_ego(
        nusc, instance_token, sample_token, traj)

    title_bits = [frame_label]
    if pos is not None:
        x_ego, y_ego = pos
        if osz_source.in_bev_range(x_ego, y_ego):
            # plot (ego-y, ego-x): x-axis=ego-y, y-axis=ego-x (forward=up)
            color, marker = VERDICT_STYLE.get(verdict, VERDICT_STYLE[None])
            ms = 14 if verdict == 'emerged' else 7
            ax.plot(y_ego, x_ego, marker, color=color, markersize=ms,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
            ax.text(y_ego + 0.8, x_ego + 0.8, frame_label,
                    color=PALETTE['text_dark'], fontsize=6,
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])
            if verdict is True:
                title_bits.append('IN OSZ')
            elif verdict is False:
                title_bits.append('visible')
        else:
            title_bits.append('out of grid')
    else:
        title_bits.append('no evidence' if status == 'no_evidence' else 'no pos')

    title_bits.append(f'OSZ {coverage_pct:.1f}%')
    ax.set_title('  |  '.join(title_bits), fontsize=7, color=PALETTE['text_dark'])
    ax.set_xlabel('y (m) ←left | right→', fontsize=6, color=PALETTE['text_mid'])
    ax.set_ylabel('x (m) ↑ forward', fontsize=6, color=PALETTE['text_mid'])

    return coverage_pct


class EventBrowser:
    """
    Interactive 2x3 event browser for Ghost-Probe QA.

      [ t-4 ][ t-3 ][ t-2 ]
      [ t-1 ][  t  ][ info]

    Each BEV panel shows that frame's OWN PA-relevant OSZ + ego + the
    tracked vehicle's position (in that frame's ego frame), coloured by
    the miner's was_in_osz verdict. The info panel summarises the event.

    Keys (focus the figure window first):
        n / →   next event
        p / ←   previous event
        r       redraw current event (refresh)
        q / Esc quit
    """

    def __init__(self, nusc: NuScenes, events: List[Dict],
                 start_idx: int = 0, label_filter: int = 1):
        self.nusc = nusc
        # Browser operates on a filtered subset (positives by default —
        # those are the ones worth QA-ing; flip to -1 for all).
        if label_filter >= 0:
            self.events = [e for e in events if e['label'] == label_filter]
        else:
            self.events = list(events)
        self.label_filter = label_filter
        if not self.events:
            raise ValueError(
                f"No events with label={label_filter} to browse. "
                f"Pass --label_filter -1 for all.")
        self.idx = max(0, min(start_idx, len(self.events) - 1))
        self.fig = None
        self.axes = None
        # Per-instance trajectory cache (built lazily per event). Single-
        # instance, so cheap; avoids re-walking the annotation chain when
        # interpolating un-annotated lookback frames.
        self._traj_cache: Dict[str, list] = {}

    # ── trajectory for one instance (cached) ──────────────────────────
    def _traj_for(self, instance_token: str):
        if instance_token not in self._traj_cache:
            traj = build_instance_trajectories(
                self.nusc, {instance_token}).get(instance_token, [])
            self._traj_cache[instance_token] = traj
        return self._traj_cache[instance_token]

    # ── render one event across all 6 panels ──────────────────────────
    def _render(self) -> None:
        event = self.events[self.idx]
        instance_tok = event['instance_token']
        emerge_tok   = event['emerge_sample']
        lb_tokens    = event['lookback_tokens']
        was_in_osz   = event['was_in_osz']
        traj         = self._traj_for(instance_tok)

        # 5 frames oldest→newest: lookback tokens then frame t.
        # lookback_tokens is already ordered oldest-first (t-k..t-1).
        # Pad to 4 so the grid layout is stable even if a scene boundary
        # truncated the lookback (miner normally skips those, but be safe).
        padded_lb = (list(lb_tokens) + [None] * 4)[:4]
        padded_verdict = (list(was_in_osz) + [None] * 4)[:4]

        frames = [
            # (sample_token, verdict, frame_label)
            (padded_lb[0], padded_verdict[0], 't-4'),
            (padded_lb[1], padded_verdict[1], 't-3'),
            (padded_lb[2], padded_verdict[2], 't-2'),
            (padded_lb[3], padded_verdict[3], 't-1'),
            (emerge_tok,   'emerged',        't'),
        ]

        coverages: List[Optional[float]] = []
        # axes layout: row0 = [0,1,2], row1 = [3,4,5(info)]
        be_axes = [self.axes[0][0], self.axes[0][1], self.axes[0][2],
                   self.axes[1][0], self.axes[1][1]]
        info_ax = self.axes[1][2]

        for ax, (tok, verdict, label) in zip(be_axes, frames):
            if tok is None:
                ax.clear()
                ax.axis('off')
                ax.text(0.5, 0.5, f'{label}\n(no data)',
                        transform=ax.transAxes, ha='center', va='center',
                        color=PALETTE['text_light'], fontsize=9)
                coverages.append(None)
                continue

            # Frame t ('emerged') goes through a dedicated drawer that uses
            # the event's stored emerge_bev_xy verbatim (already in frame-t's
            # ego frame) — this guarantees the green star matches the miner's
            # recorded emergence point exactly, with no re-derivation drift.
            # Lookback frames go through _draw_frame_own_ego, which derives
            # the vehicle position per-frame (direct annotation → trajectory
            # interpolation), mirroring the miner's own lookback logic.
            if verdict == 'emerged':
                ex, ey = event['emerge_bev_xy']
                pos_known = (float(ex), float(ey))
                cov = _draw_frame_own_ego_emerged(
                    ax, self.nusc, tok, pos_known, label,
                    instance_token=instance_tok)
            else:
                cov = _draw_frame_own_ego(
                    ax, self.nusc, tok, instance_tok, verdict, label, traj)
            coverages.append(cov)

        _draw_info_panel(info_ax, self.nusc, event, self.idx,
                         len(self.events), self.label_filter,
                         coverages, frames)

        label_str = 'POSITIVE (ghost)' if event['label'] == 1 else 'NEGATIVE (visible)'
        self.fig.suptitle(
            f'Ghost-Probe Event Browser  —  [{self.idx+1}/{len(self.events)}]  '
            f'{label_str}   (n=next  p=prev  r=redraw  q=quit)',
            fontsize=11, color='#222')
        self.fig.canvas.draw_idle()

    # ── key handler ───────────────────────────────────────────────────
    def _on_key(self, event) -> None:
        k = (event.key or '').lower()
        if k in ('n', 'right'):
            if self.idx < len(self.events) - 1:
                self.idx += 1
                self._render()
        elif k in ('p', 'left'):
            if self.idx > 0:
                self.idx -= 1
                self._render()
        elif k == 'r':
            self._render()
        elif k in ('q', 'escape'):
            plt.close(self.fig)

    # ── entry ─────────────────────────────────────────────────────────
    def launch(self) -> None:
        be = _ensure_gui_backend()
        # Use the top-level `plt` (already imported at module load). After
        # use(force=True) it targets the new GUI backend on first figure.

        if be.lower().startswith('webagg'):
            print('\n  ⓘ WebAgg backend: an interactive window is served '
                  'over HTTP.')
            print('    Open this in your browser:  http://localhost:8988/')
            print('    (terminal stays busy while the server runs; '
                  'Ctrl-C to stop)\n')
        else:
            print('\n  ⓘ GUI backend active. A window should now be open.')
            print('    Click it, then use:  n=next  p=prev  r=redraw  '
                  'q=quit\n')

        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 11))
        self.fig.canvas.manager.set_window_title('Ghost-Probe Event Browser')
        self.fig.patch.set_facecolor(PALETTE['bg'])
        # Tighter spacing; the info panel needs room for text.
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04,
                                 wspace=0.18, hspace=0.22)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._render()
        plt.show()


# ════════════════════════════════════════════════════════════════════
# HEADLESS: terminal-based browser (zero GUI dependencies)
# ════════════════════════════════════════════════════════════════════

class HeadlessEventBrowser:
    """
    Terminal-driven event browser that works on ANY server — no tkinter,
    no Qt, no display required.  Uses Agg backend to render each event
    to a temporary PNG, then reads navigation commands from stdin.

    Usage is identical to EventBrowser (n/p/j/k/q/numbers), but output
    goes to a file instead of a GUI window.
    """

    def __init__(self, nusc: NuScenes, events: List[Dict],
                 start_idx: int = 0, label_filter: int = 1):
        self.nusc = nusc
        if label_filter >= 0:
            self.events = [e for e in events if e['label'] == label_filter]
        else:
            self.events = list(events)
        self.idx = max(0, min(start_idx, len(self.events) - 1))
        self.label_filter = label_filter
        # Per-event PNGs in a dedicated folder (so you can flip through them
        # in any image viewer instead of watching a single overwritten file).
        self.out_dir = str(_REPO_ROOT / 'PA_gen_v2' / 'output' / 'browser')
        self.out_path = None  # set per render

    # ── render (same logic as EventBrowser, but saves to file) ────────
    def _render(self) -> None:
        """Render current event to a per-event PNG in self.out_dir."""
        matplotlib.use('Agg')
        import matplotlib.pyplot as _hp  # headless pyplot

        event = self.events[self.idx]
        instance_tok = event['instance_token']
        traj = build_instance_trajectories(self.nusc, {instance_tok}).get(instance_tok)

        lookback = event.get('lookback_tokens', [])
        was_in_osz = event.get('was_in_osz', [])
        frames = []          # [(sample_token_or_None, verdict, label_str)]
        for i in range(LOOKBACK_FRAMES):  # t-4 .. t-1
            if i < len(lookback):
                tok = lookback[i]
                verdict = was_in_osz[i] if i < len(was_in_osz) else None
                frames.append((tok, verdict, f't-{LOOKBACK_FRAMES - i - 1}'))
            else:
                frames.append((None, None, f't-{LOOKBACK_FRAMES - i - 1}'))
        # Emergence frame t.
        frames.append((event['emerge_sample'], 'emerged', 't'))

        fig, axes = _hp.subplots(2, 3, figsize=(18, 11))
        fig.patch.set_facecolor(PALETTE['bg'])
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04,
                            wspace=0.18, hspace=0.22)

        coverages = []
        for ax, (tok, verdict, label) in zip(axes.flat[:5], frames):
            if verdict == 'emerged':
                ex, ey = event['emerge_bev_xy']
                cov = _draw_frame_own_ego_emerged(
                    ax, self.nusc, tok, (float(ex), float(ey)), label,
                    instance_token=instance_tok)
            elif tok is not None:
                cov = _draw_frame_own_ego(
                    ax, self.nusc, tok, instance_tok, verdict, label, traj)
            else:
                ax.clear(); ax.set_facecolor(PALETTE['panel_bg']); ax.axis('off')
                ax.set_title(f'{label}  (no data)', fontsize=8, color=PALETTE['text_light'])
                cov = None
            coverages.append(cov)

        _draw_info_panel(axes[1, 2], self.nusc, event, self.idx,
                         len(self.events), self.label_filter,
                         coverages, frames)

        fig.suptitle(f'Ghost-Probe Event Browser  [{self.idx+1} / '
                     f'{len(self.events)}]   (headless mode)',
                     fontsize=11, color=PALETTE['text_dark'], fontweight='bold')
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.out_path = str(Path(self.out_dir) /
                            f'event_{self.idx:04d}.png')
        fig.savefig(self.out_path, dpi=130, bbox_inches='tight')
        _hp.close(fig)

    # ── terminal UI ───────────────────────────────────────────────────
    def _print_status(self) -> None:
        ev = self.events[self.idx]
        scene = self.nusc.get('scene', ev['scene_token'])
        lbl = 'GHOST' if ev['label'] == 1 else 'VISIBLE'
        print(f'\n  ┌─ Event {self.idx+1}/{len(self.events)} ─'
              f' {lbl} ─ {scene["name"]}')
        print(f'  │ instance={ev["instance_token"][:16]}…')
        ex, ey = ev['emerge_bev_xy']
        d = (ex*ex + ey*ey) ** 0.5
        print(f' │ emerge_dist={d:.1f}m  osz_frames={ev["n_osz_frames"]}')
        print(f' │ saved → {self.out_path}')
        print(f' │ (folder: {self.out_dir})')
        print(f' └─')

    # ── entry ─────────────────────────────────────────────────────────
    def launch(self) -> None:
        print('\n╔══════════════════════════════════════════════════╗')
        print('║  Ghost-Probe Event Browser  (HEADLESS MODE)       ║')
        print('║  No GUI backend found; using terminal + PNG.      ║')
        print('╠══════════════════════════════════════════════════╣')
        print('║  n / →     next event                            ║')
        print('║  p / ←     previous event                        ║')
        print('║  j / +10   jump forward 10                       ║')
        print('║  k / -10   jump backward 10                      ║')
        print('║  <number>  go to event index                      ║')
        print('║  r         redraw current                         ║')
        print('║  q         quit                                   ║')
        print('╚══════════════════════════════════════════════════╝')
        print('\n  Each event is saved as a separate PNG:')
        print(f'    {self.out_dir}/event_0000.png, event_0001.png, ...')
        print('  Open the folder in any image viewer to flip through')
        print('  (most viewers let you scroll with ←/→ once focused).\n')

        self._render()
        self._print_status()

        while True:
            try:
                cmd = input('  Command [n/p/r/q/<index>]> ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Bye.")
                break

            if cmd in ('q', 'quit'):
                print("  Bye.")
                break
            elif cmd in ('n', '', '→', 'right'):
                if self.idx < len(self.events) - 1:
                    self.idx += 1; self._render(); self._print_status()
                else:
                    print('  Already at last event.')
            elif cmd in ('p', '←', 'left'):
                if self.idx > 0:
                    self.idx -= 1; self._render(); self._print_status()
                else:
                    print('  Already at first event.')
            elif cmd in ('j', '+'):
                ni = min(self.idx + 10, len(self.events) - 1)
                if ni != self.idx:
                    self.idx = ni; self._render(); self._print_status()
            elif cmd in ('k', '-'):
                ni = max(self.idx - 10, 0)
                if ni != self.idx:
                    self.idx = ni; self._render(); self._print_status()
            elif cmd.isdigit():
                target = int(cmd)
                if 0 <= target < len(self.events):
                    self.idx = target; self._render(); self._print_status()
                else:
                    print(f'  Index out of range [0..{len(self.events)-1}]')
            elif cmd == 'r':
                self._render(); self._print_status()
            else:
                print('  Unknown command. n=next p=prev r=redraw q=quit')


# ── frame-t drawer (uses the stored emerge position verbatim) ─────────
def _draw_frame_own_ego_emerged(ax, nusc, sample_token,
                                emerge_xy: Tuple[float, float],
                                frame_label: str,
                                instance_token: str = None) -> Optional[float]:
    """
    Frame-t drawer: identical to _draw_frame_own_ego but the vehicle
    position comes from the event's stored emerge_bev_xy (already in
    frame-t's ego frame) rather than a re-derivation through the
    annotation table. Guarantees the green star matches the miner's
    recorded emergence point to the last decimal.
    """
    ax.clear()
    ax.set_facecolor(PALETTE['panel_bg'])

    try:
        bev_occ, osz_raw, osz_pa, drivable_mask = \
            osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)
    except Exception as ex:
        ax.axis('off')
        ax.set_facecolor(PALETTE['panel_bg'])
        ax.text(0.5, 0.5, f"OSZ failed\n{ex}",
                transform=ax.transAxes, ha='center', va='center',
                color=PALETTE['error'], fontsize=8)
        ax.set_title(f'{frame_label}  (OSZ ERROR)', fontsize=8, color=PALETTE['error'])
        return None

    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[:] = _hex_to_rgb(PALETTE['panel_bg'])
    overlay[drivable_mask] = _hex_to_rgb(PALETTE['road'])
    overlay[bev_occ] = _hex_to_rgb(PALETTE['obstacle'])
    overlay[osz_pa] = _hex_to_rgb(PALETTE['osz'])
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5, colors=PALETTE['text_dark'])
    for s in ax.spines.values():
        s.set_color(PALETTE['spine'])

    _draw_map_overlay(ax, nusc, sample_token)

    ax.plot(0, 0, '^', color=PALETTE['ego'], markersize=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    ax.annotate('', xy=(0, 3.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=PALETTE['text_dark'], lw=1.0))

    _anns = _get_scene_annotations_ego(nusc, sample_token)
    _draw_annotation_boxes(ax, _anns, tracked_instance=instance_token,
                           tracked_verdict='emerged')

    coverage_pct = float(osz_pa.mean()) * 100.0

    x_ego, y_ego = emerge_xy
    if osz_source.in_bev_range(x_ego, y_ego):
        ax.plot(y_ego, x_ego, '*', color=PALETTE['tracked_arrow'], markersize=15,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        ax.text(y_ego + 0.8, x_ego + 0.8, frame_label,
                color=PALETTE['text_dark'], fontsize=6,
                path_effects=[pe.withStroke(linewidth=1, foreground='white')])
    ax.set_title(f'{frame_label}  |  EMERGED  |  OSZ {coverage_pct:.1f}%',
                 fontsize=7, color=PALETTE['text_dark'])
    ax.set_xlabel('y (m) ←left | right→', fontsize=6, color=PALETTE['text_mid'])
    ax.set_ylabel('x (m) ↑ forward', fontsize=6, color=PALETTE['text_mid'])
    return coverage_pct


def _draw_info_panel(ax, nusc, event, idx, total, label_filter,
                     coverages, frames) -> None:
    """
    Right-hand info panel: event metadata + per-frame verdict table +
    per-frame OSZ coverage + key reminder. Pure text, no BEV.
    """
    ax.clear()
    ax.axis('off')
    ax.set_facecolor(PALETTE['info_bg'])

    scene = nusc.get('scene', event['scene_token'])
    label_str = 'POSITIVE (ghost)' if event['label'] == 1 else 'NEGATIVE (visible)'
    label_color = '#dc2626' if event['label'] == 1 else '#2563eb'

    ex, ey = event['emerge_bev_xy']
    emerge_dist = float(np.sqrt(ex*ex + ey*ey))

    n_osz  = sum(1 for v in event['was_in_osz'] if v is True)
    n_unk  = sum(1 for v in event['was_in_osz'] if v is None)
    n_ev   = event.get('n_evidence_frames',
                       sum(1 for v in event['was_in_osz'] if v is not None))

    # Build the verdict table rows: t-4 .. t
    # frames = [(tok, verdict, label), ...] for t-4,t-3,t-2,t-1,t
    verdict_rows = []
    for (tok, verdict, label), cov in zip(frames, coverages):
        if tok is None:
            verdict_rows.append((label, '—', '—', PALETTE['text_light']))
            continue
        if verdict == 'emerged':
            text, col = 'EMERGED', VERDICT_STYLE['emerged'][0]
        elif verdict is True:
            text, col = 'IN OSZ', VERDICT_STYLE[True][0]
        elif verdict is False:
            text, col = 'visible', VERDICT_STYLE[False][0]
        else:
            text, col = 'unknown', VERDICT_STYLE[None][0]
        cov_s = f'{cov:.1f}%' if cov is not None else '—'
        verdict_rows.append((label, text, cov_s, col))

    lines = []  # (text, color, weight)
    def add(text, color=PALETTE['text_dark'], weight='normal'):
        lines.append((text, color, weight))

    add(f'EVENT  [{idx+1} / {total}]', PALETTE['info_title'], 'bold')
    add(f'(showing label={label_filter} subset)', PALETTE['text_light'])
    add('─' * 40, PALETTE['info_separator'])
    add(f'Label         : {label_str}', label_color, 'bold')
    add(f'Scene         : {scene["name"]}', PALETTE['text_dark'])
    add(f'Instance      : {event["instance_token"][:18]}...', PALETTE['text_mid'])
    add(f'Emerge sample : {event["emerge_sample"][:18]}...', PALETTE['text_mid'])
    add(f'Emerge dist   : {emerge_dist:.1f} m', PALETTE['text_dark'])
    add('─' * 40, PALETTE['info_separator'])
    add('Lookback verdicts (oldest → newest):', PALETTE['text_dark'], 'bold')
    add(f'  {"frame":<6} {"verdict":<10} {"OSZ cov":<8}', PALETTE['text_light'])
    for label, text, cov_s, col in verdict_rows:
        add(f'  {label:<6} {text:<10} {cov_s:<8}', col)
    add('─' * 40, PALETTE['info_separator'])
    add(f'OSZ overlap     : {n_osz} / {len(event["was_in_osz"])} lookback frames',
        VERDICT_STYLE[True][0] if n_osz else PALETTE['text_dark'])
    add(f'Evidence frames : {n_ev} / {len(event["was_in_osz"])} '
        f'(unknown={n_unk})', PALETTE['text_dark'])
    add('─' * 40, PALETTE['info_separator'])
    add('legend:', PALETTE['text_dark'], 'bold')
    add('  ■ black   = PA-relevant OSZ (shadow)', '#1a1a1a')
    add('  ■ gray    = obstacles (walls, vehicles)', PALETTE['obstacle'])
    add('  ■ lgrey   = drivable area (road)', PALETTE['road'])
    add('  ■ orange  = other vehicles', PALETTE['other_vehicle'])
    add('  ■ dk dot  = pedestrians', PALETTE['pedestrian'])
    add('  ■ grey ln = lane boundaries', PALETTE['lane'])
    add('  ★ colored = tracked ghost vehicle + heading', VERDICT_STYLE['emerged'][0])
    add('─' * 40, PALETTE['info_separator'])
    add('read the grid:', PALETTE['text_dark'], 'bold')
    add('  • car stayed in OSZ?  red dots across t-4..t-1', PALETTE['text_light'])
    add('  • when did it come out?  red→green transition', PALETTE['text_light'])
    add('  • OSZ moving?  compare red shape across frames', PALETTE['text_light'])
    add('  • ego turning?  OSZ shape rotates between frames', PALETTE['text_light'])
    add('  • other cars explain OSZ?  see orange boxes', PALETTE['text_light'])
    add('  • road makes sense?  grey surface + lane lines', PALETTE['text_light'])
    add('─' * 40, PALETTE['info_separator'])
    add('keys:  n=next   p=prev   r=redraw   q=quit', PALETTE['info_key'], 'bold')

    # Render lines top-to-bottom.
    y0 = 0.97
    dy = 0.0285
    y = y0
    for text, color, weight in lines:
        ax.text(0.03, y, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8.5, family='monospace',
                color=color, weight=weight)
        y -= dy


def launch_browser(nusc: NuScenes, events: List[Dict],
                   start_idx: int = 0, label_filter: int = 1,
                   force_headless: bool = False) -> None:
    """Launch interactive EventBrowser; falls back to headless if no GUI.

    force_headless=True skips GUI detection entirely and uses the
    terminal + PNG mode (handy on headless servers or when no display
    is available). Auto-fallback still triggers if GUI launch fails.
    """
    if force_headless:
        hb = HeadlessEventBrowser(nusc, events, start_idx=start_idx,
                                  label_filter=label_filter)
        hb.launch()
        return
    try:
        browser = EventBrowser(nusc, events, start_idx=start_idx,
                               label_filter=label_filter)
        browser.launch()
    except RuntimeError as exc:
        # No GUI backend available (headless server / missing tkinter/Qt).
        # Automatically fall back to terminal-based headless mode.
        print(f'\n  ⚠  {exc}')
        print('  → Falling back to HEADLESS (terminal + PNG) mode.\n')
        hb = HeadlessEventBrowser(nusc, events, start_idx=start_idx,
                                  label_filter=label_filter)
        hb.launch()


# ════════════════════════════════════════════════════════════════════
# OFFLINE: grid export + stats (unchanged)
# ════════════════════════════════════════════════════════════════════

def make_event_grid(nusc: NuScenes, events: List[Dict], max_events: int = 12,
                    label_filter: int = 1, out_path: str = None) -> None:
    """Render a grid of ghost vehicle events and save to disk."""
    if label_filter >= 0:
        subset = [e for e in events if e['label'] == label_filter]
    else:
        subset = events
    subset = subset[:max_events]

    if not subset:
        print(f"No events with label={label_filter} found.")
        return

    ncols = 4
    nrows = (len(subset) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    label_str = 'POSITIVE (ghost)' if label_filter == 1 else 'NEGATIVE (visible)'
    fig.suptitle(f'{label_str} events', fontsize=10)

    for i, (ax, event) in enumerate(zip(axes, subset)):
        try:
            visualize_event(nusc, event, ax, title=f"Event {i} | label={event['label']}")
        except Exception as ex:
            ax.set_title(f"Event {i} FAILED: {ex}", fontsize=6)
            ax.axis('off')

    for ax in axes[len(subset):]:
        ax.axis('off')

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Saved event grid ({len(subset)} events) → {out_path}")
    plt.close()


def print_event_stats(events: List[Dict]) -> None:
    positives = [e for e in events if e['label'] == 1]
    negatives = [e for e in events if e['label'] == 0]

    print(f"\n{'─'*50}")
    print("Event set statistics")
    print(f"{'─'*50}")
    print(f"Total events   : {len(events)}")
    print(f"Positives      : {len(positives)}")
    print(f"Negatives      : {len(negatives)}")

    if positives:
        osz_counts = [e['n_osz_frames'] for e in positives]
        print("\nPositive events — confirmed-OSZ frame counts:")
        print(f"  mean : {np.mean(osz_counts):.2f}")
        print(f"  min  : {np.min(osz_counts)}")
        print(f"  max  : {np.max(osz_counts)}")

        dists = [np.sqrt(e['emerge_bev_xy'][0]**2 + e['emerge_bev_xy'][1]**2)
                 for e in positives]
        print("\nEmergence distance from ego (m):")
        print(f"  mean : {np.mean(dists):.1f}")
        print(f"  min  : {np.min(dists):.1f}")
        print(f"  max  : {np.max(dists):.1f}")

    print(f"{'─'*50}\n")


# ════════════════════════════════════════════════════════════════════
# WEB GALLERY: pre-render every event to PNG + an index.html you open
# in a browser.  Fastest, most responsive review path — no matplotlib
# GUI at all (which the user found slow / frequently unresponsive).
# Keyboard in the browser: ←/→ step, j/k ±10, or type a number + Go.
# ════════════════════════════════════════════════════════════════════

_GALLERY_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost-Probe Event Gallery</title>
<style>
  html,body{margin:0;height:100%;background:#11141a;color:#ddd;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  body{display:flex;flex-direction:column}
  .bar{display:flex;gap:12px;align-items:center;padding:8px 14px;
    background:#1a1d22;border-bottom:1px solid #333;font-size:14px;flex-wrap:wrap}
  #counter{font-weight:bold;color:#fff;min-width:64px}
  .hint{color:#8a8f99;font-size:12px}
  .cap{color:#aab;font-size:12px;flex:1;min-width:200px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  button{background:#2a2f3a;color:#ddd;border:1px solid #444;border-radius:4px;
    padding:4px 12px;cursor:pointer}
  button:hover{background:#353b48}
  input{width:74px;background:#0d0f13;color:#ddd;border:1px solid #444;
    border-radius:4px;padding:3px 6px}
  #wrap{flex:1;display:flex;align-items:center;justify-content:center;
    overflow:auto;padding:10px}
  img{max-width:100%;max-height:100%;box-shadow:0 0 24px #000}
</style>
</head>
<body>
  <div class="bar">
    <span id="counter">1 / __N__</span>
    <button onclick="go(-1)">&#8592; Prev</button>
    <button onclick="go(1)">Next &#8594;</button>
    <span class="hint">&#8593;/&#8595; or j/k = &plusmn;10</span>
    <span>goto <input id="jump" type="number" min="1" value="1">
      <button onclick="jumpTo()">Go</button></span>
    <span id="cap" class="cap"></span>
  </div>
  <div id="wrap"><img id="view" src="" alt="event"></div>
<script>
const data = __DATA__;
let i = 0;
function show(){
  document.getElementById('view').src = data[i].f;
  document.getElementById('counter').textContent = (i+1)+' / '+data.length;
  document.getElementById('cap').textContent = data[i].c;
  document.getElementById('jump').value = i+1;
  document.title = 'Event '+(i+1)+' / '+data.length;
}
function go(d){ i=Math.max(0,Math.min(data.length-1,i+d)); show(); }
function jumpTo(){
  const v=parseInt(document.getElementById('jump').value,10);
  if(v>=1 && v<=data.length){ i=v-1; show(); }
}
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowRight'||e.key==='ArrowDown') go(1);
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp') go(-1);
  else if(e.key==='j') go(10);
  else if(e.key==='k') go(-10);
  else if(e.key==='q') window.close();
});
show();
</script>
</body>
</html>
"""


def _write_gallery_html(out_dir: str, names: List[str],
                        captions: List[str]) -> None:
    """Write index.html with an embedded JSON array of {file, caption}."""
    items = [{"f": n, "c": c} for n, c in zip(names, captions)]
    data_js = json.dumps(items, ensure_ascii=False)
    html = _GALLERY_HTML.replace("__DATA__", data_js).replace(
        "__N__", str(len(names)))
    Path(out_dir, "index.html").write_text(html, encoding="utf-8")


def build_web_gallery(nusc: NuScenes, events: List[Dict],
                      label_filter: int = 1, max_events: int = 80,
                      out_dir: str = None) -> None:
    """Render every event to a PNG and build a browser-viewable gallery.

    Reuses HeadlessEventBrowser's per-event renderer (identical visuals to
    the GUI/headless views), so the QA picture is exactly the same — just
    served as static images in a fast HTML page instead of a matplotlib
    window.  Open the resulting index.html in any browser.
    """
    out_dir = out_dir or str(_REPO_ROOT / 'PA_gen_v2' / 'output' / 'web')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Reuse the headless renderer; point its output dir at the gallery dir.
    hb = HeadlessEventBrowser(nusc, events, start_idx=0,
                              label_filter=label_filter)
    hb.out_dir = out_dir                   # ← redirect PNGs into the gallery dir
    # Cap how many events we render (galleries with hundreds of PNGs are
    # still fine, but 80 keeps it snappy by default).
    if max_events and max_events > 0:
        hb.events = hb.events[:max_events]
    if not hb.events:
        print("No events to render for the gallery.")
        return

    names, captions, total = [], [], len(hb.events)
    print(f"  Rendering {total} events to {out_dir} ...")
    print(f"  (first run: ~5-15s/event for 3D ray-casting; cached after)")
    import time as _t
    t0 = _t.time()
    for i in range(total):
        hb.idx = i
        hb._render()                       # → event_{i:04d}.png in out_dir
        ev = hb.events[i]
        scene = nusc.get('scene', ev['scene_token'])
        ex, ey = ev['emerge_bev_xy']
        d = float((ex * ex + ey * ey) ** 0.5)
        lbl = 'GHOST' if ev['label'] == 1 else 'VISIBLE'
        cap = (f"{scene['name']} | {lbl} | dist {d:.1f} m | "
               f"osz {ev['n_osz_frames']} | {ev['instance_token'][:12]}")
        names.append(Path(hb.out_path).name)
        captions.append(cap)
        elapsed = _t.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1)
        print(f"  [{i + 1:3d}/{total}] {Path(hb.out_path).name}  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    _write_gallery_html(out_dir, names, captions)
    html = str(Path(out_dir) / 'index.html')
    print(f"\nGallery ready: {total} events → {html}")
    print("Opening in your default browser...")
    try:
        webbrowser.open(html)
    except Exception:
        print("  (could not auto-open; open the file manually)")

    print("\nHow to view:")
    print(f"  • Local : open  {html}")
    print(f"  • Server: scp -r {out_dir} <you>@<laptop>:/tmp/  then open index.html")
    print(f"           or run  python -m http.server --directory {out_dir} 8080")
    print(f"           and visit http://<server>:8080/")
    print("  • Keys  : ←/→ or ↑/↓ flip; j/k jump ±10; type a number + Go.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version',  default='v1.0-mini')
    parser.add_argument('--events',     default=str(_REPO_ROOT / 'PA_gen_v2' /
                                                     'output' / 'ghost_events_mini.json'))
    parser.add_argument('--max_events', type=int, default=12,
                        help='(offline export only) events per PNG')
    parser.add_argument('--out_pos',    default=str(_REPO_ROOT / 'PA_gen_v2' /
                                                     'output' / 'events_positive.png'))
    parser.add_argument('--out_neg',    default=str(_REPO_ROOT / 'PA_gen_v2' /
                                                     'output' / 'events_negative.png'))
    parser.add_argument('--browse', action='store_true',
                        help='Launch the interactive EventBrowser instead of '
                             'exporting PNGs.')
    parser.add_argument('--headless', action='store_true',
                        help='With --browse: force terminal + PNG mode '
                             '(no GUI window). Useful if no display is '
                             'available or WebAgg is awkward.')
    parser.add_argument('--web', action='store_true',
                        help='Render every event to a static PNG and build a '
                             'browser-viewable gallery (index.html). No '
                             'matplotlib window — open the HTML in any browser '
                             'and use arrow keys / j / k to flip through.')
    parser.add_argument('--web_max', type=int, default=20,
                        help='(with --web) max events to render into the '
                             'gallery. Set 0 for all. Default 20 keeps the '
                             'first run under ~5 min; cached OSZ makes '
                             'subsequent runs near-instant.')
    parser.add_argument('--label_filter', type=int, default=1,
                        help='Browser subset: 1=positives (default), '
                             '0=negatives, -1=all.')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Browser: event index to start at (within the '
                             'filtered subset).')
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f"Loading events from {args.events} ...")
    events = load_events(args.events)

    print_event_stats(events)

    if args.web:
        print(f"\nBuilding web gallery "
              f"(label_filter={args.label_filter}, "
              f"max={args.web_max or 'all'})...")
        build_web_gallery(nusc, events, label_filter=args.label_filter,
                          max_events=args.web_max)
    elif args.browse:
        print(f"\nLaunching EventBrowser "
              f"(label_filter={args.label_filter}, "
              f"{len([e for e in events if args.label_filter < 0 or e['label'] == args.label_filter])} events)...")
        if args.headless:
            print("Mode: HEADLESS (terminal + PNG, no GUI window)")
        else:
            print("GUI mode:   n=next  p=prev  r=redraw  q=quit   (click window)")
            print("Headless:   --headless  for terminal + PNG mode")
        launch_browser(nusc, events, start_idx=args.start_idx,
                       label_filter=args.label_filter,
                       force_headless=args.headless)
    else:
        print("Rendering positive events ...")
        make_event_grid(nusc, events, max_events=args.max_events,
                        label_filter=1, out_path=args.out_pos)

        print("Rendering negative events (sanity check) ...")
        make_event_grid(nusc, events, max_events=args.max_events,
                        label_filter=0, out_path=args.out_neg)

        print("\nDone. Open the PNGs and verify:")
        print("  POSITIVE: green star should appear at OSZ boundary,")
        print("            red dots should be inside the red OSZ region.")
        print("  NEGATIVE: all dots should be blue (confirmed visible),")
        print("            none should sit inside the red OSZ region.")
        print("\nTip: add --browse for the interactive EventBrowser.")
