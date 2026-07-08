"""
visualize_events.py
-------------------
Visualize mined ghost vehicle events — the most important debugging step.

Two modes:

  1. OFFLINE EXPORT (the original mode): render a grid of events to PNG via
     make_event_grid(). Headless, works anywhere (Agg or any backend;
     savefig does not need a GUI).

  2. INTERACTIVE EVENT BROWSER (new): an interactive 2x3 window that lets
     you walk the event set with n/p/r/q and, crucially, shows each frame
     in ITS OWN ego frame — so you can see at a glance:
        - whether the car stayed inside OSZ the whole time,
        - when it came out,
        - whether OSZ moved between frames,
        - whether ego turned (OSZ shape rotates relative to ego).

     Layout:
        +---------+---------+---------+
        |  t-4    |  t-3    |  t-2    |
        +---------+---------+---------+
        |  t-1    |   t     |  info   |
        +---------+---------+---------+

     Each BEV panel draws that frame's OWN PA-relevant OSZ (raw OSZ ∩
     drivable area — exactly what ghost_vehicle_miner.py used for every
     was_in_osz decision), the ego marker at the origin, and the tracked
     vehicle's position in that frame's ego coordinates. This is different
     from visualize_event() (offline), which collapses every lookback
     position into frame-t's ego frame — fine for a static summary, useless
     for watching OSZ evolve over time.

All plotting is in metric ego coordinates to avoid pixel-index rounding
and axis-order mistakes.

Karpathy rule: if you cannot look at the output and immediately see that
it makes geometric sense, the mining logic is wrong. Do NOT proceed to
model training without this visual check.
"""

import sys
import argparse
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
    True:      ('red',         'o'),   # confirmed in OSZ
    False:     ('deepskyblue', 'o'),   # confirmed visible
    None:      ('#888888',     'x'),   # no evidence
    'emerged': ('lime',        '*'),   # emerged at frame t
}


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
    overlay[bev_occ]  = [0.55, 0.55, 0.55]
    overlay[osz_pa]   = [0.80, 0.15, 0.15]
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Ego marker at (0,0) in metric coords — no pixel math needed.
    ax.plot(0, 0, 'w^', markersize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Emerged vehicle position (frame t), already stored in metric ego xy
    ex, ey = event['emerge_bev_xy']
    ax.plot(ey, ex, 'g*', markersize=14,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
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
                color='white', fontsize=5,
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    if len(traj_x) >= 2:
        traj_x.append(ey)
        traj_y.append(ex)
        ax.plot(traj_x, traj_y, '-', color='yellow', linewidth=1.2, alpha=0.7)

    n_osz  = sum(1 for v in was_in_osz if v is True)
    n_unk  = sum(1 for v in was_in_osz if v is None)
    if not title:
        title = (f"Ghost event | {n_osz}/{len(lb_tokens)} lb frames in OSZ"
                 + (f" ({n_unk} unknown)" if n_unk else ""))
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('y (m) ← ego-left | ego-right →', fontsize=6)
    ax.set_ylabel('x (m) ↑ forward', fontsize=6)
    ax.tick_params(labelsize=5)

    legend_patches = [
        mpatches.Patch(color=[0.55]*3, label='Occupied'),
        mpatches.Patch(color=[0.8, 0.15, 0.15], label='PA-relevant OSZ'),
        plt.Line2D([0],[0], marker='*', color='g', markersize=8,
                   linestyle='none', label='Emerged (t)'),
        plt.Line2D([0],[0], marker='o', color='red', markersize=5,
                   linestyle='none', label='Lookback (confirmed in OSZ)'),
        plt.Line2D([0],[0], marker='o', color='deepskyblue', markersize=5,
                   linestyle='none', label='Lookback (confirmed visible)'),
        plt.Line2D([0],[0], marker='x', color='#888888', markersize=5,
                   linestyle='none', label='Lookback (no evidence)'),
    ]
    ax.legend(handles=legend_patches, fontsize=5, loc='upper right')


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
    # ── WebAgg: serves interactive plots in browser (ideal for remote servers)
    try:
        import tornado  # noqa: F401  (webagg dependency)
        candidates.append('WebAgg')
    except Exception:
        pass
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
    ax.set_facecolor('#101418')

    try:
        bev_occ, osz_raw, osz_pa, drivable_mask = \
            osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)
    except Exception as ex:
        ax.axis('off')
        ax.text(0.5, 0.5, f"OSZ failed\n{ex}",
                transform=ax.transAxes, ha='center', va='center',
                color='#ff8080', fontsize=8)
        ax.set_title(f'{frame_label}  (OSZ ERROR)', fontsize=8, color='#ff8080')
        return None

    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ] = [0.30, 0.30, 0.32]
    overlay[osz_pa]  = [0.80, 0.18, 0.18]
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5, colors='#cccccc')
    for s in ax.spines.values():
        s.set_color('#444')

    # Ego at origin of THIS frame's ego frame.
    ax.plot(0, 0, 'w^', markersize=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    # Heading tick: ego always faces +x (forward=up) in its own frame.
    ax.annotate('', xy=(0, 3.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.0))

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
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
            ax.text(y_ego + 0.8, x_ego + 0.8, frame_label,
                    color='white', fontsize=6,
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')])
            if verdict is True:
                title_bits.append('IN OSZ')
            elif verdict is False:
                title_bits.append('visible')
        else:
            title_bits.append('out of grid')
    else:
        title_bits.append('no evidence' if status == 'no_evidence' else 'no pos')

    title_bits.append(f'OSZ {coverage_pct:.1f}%')
    ax.set_title('  |  '.join(title_bits), fontsize=7, color='#e8e8e8')
    ax.set_xlabel('y (m) ←left | right→', fontsize=6, color='#aaaaaa')
    ax.set_ylabel('x (m) ↑ forward', fontsize=6, color='#aaaaaa')

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
                        color='#888', fontsize=9)
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
                    ax, self.nusc, tok, pos_known, label)
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
        _ensure_gui_backend()
        # Use the top-level `plt` (already imported at module load). After
        # use(force=True) it targets the new GUI backend on first figure.

        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 11))
        self.fig.canvas.manager.set_window_title('Ghost-Probe Event Browser')
        self.fig.patch.set_facecolor('#1a1d22')
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
        # Output path for the current-frame PNG.
        self.out_path = str(_REPO_ROOT / 'PA_gen_v2' / 'output' /
                            '_browser_current.png')

    # ── render (same logic as EventBrowser, but saves to file) ────────
    def _render(self) -> None:
        """Render current event to self.out_path (overwrites each time)."""
        matplotlib.use('Agg')
        import matplotlib.pyplot as _hp  # headless pyplot

        event = self.events[self.idx]
        instance_tok = event['instance_token']
        traj = build_instance_trajectories(self.nusc).get(instance_tok)

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
        fig.patch.set_facecolor('#1a1d22')
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04,
                            wspace=0.18, hspace=0.22)

        coverages = []
        for ax, (tok, verdict, label) in zip(axes.flat[:5], frames):
            if verdict == 'emerged':
                ex, ey = event['emerge_bev_xy']
                cov = _draw_frame_own_ego_emerged(
                    ax, self.nusc, tok, (float(ex), float(ey)), label)
            elif tok is not None:
                cov = _draw_frame_own_ego(
                    ax, self.nusc, tok, instance_tok, verdict, label, traj)
            else:
                ax.clear(); ax.set_facecolor('#101418'); ax.axis('off')
                ax.set_title(f'{label}  (no data)', fontsize=8, color='#888')
                cov = None
            coverages.append(cov)

        _draw_info_panel(axes[1, 2], self.nusc, event, self.idx,
                         len(self.events), self.label_filter,
                         coverages, frames)

        fig.suptitle(f'Ghost-Probe Event Browser  [{self.idx+1} / '
                     f'{len(self.events)}]   (headless mode)',
                     fontsize=11, color='#e8e8e8', fontweight='bold')
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
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
        print('\n  (open the PNG in another terminal/image viewer)')
        print("  Tip: use 'tail -f' on a log or a viewer that")
        print("  auto-reloads to see updates without reopening.\n")

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
                                frame_label: str) -> Optional[float]:
    """
    Frame-t drawer: identical to _draw_frame_own_ego but the vehicle
    position comes from the event's stored emerge_bev_xy (already in
    frame-t's ego frame) rather than a re-derivation through the
    annotation table. Guarantees the green star matches the miner's
    recorded emergence point to the last decimal.
    """
    ax.clear()
    ax.set_facecolor('#101418')

    try:
        bev_occ, osz_raw, osz_pa, drivable_mask = \
            osz_source.get_pa_relevant_osz_for_sample(nusc, sample_token)
    except Exception as ex:
        ax.axis('off')
        ax.text(0.5, 0.5, f"OSZ failed\n{ex}",
                transform=ax.transAxes, ha='center', va='center',
                color='#ff8080', fontsize=8)
        ax.set_title(f'{frame_label}  (OSZ ERROR)', fontsize=8, color='#ff8080')
        return None

    caster = osz_source.get_caster()
    extent, xlim, ylim = _bev_extent(caster)

    overlay = np.zeros((*bev_occ.shape, 3), dtype=np.float32)
    overlay[bev_occ] = [0.30, 0.30, 0.32]
    overlay[osz_pa]  = [0.80, 0.18, 0.18]
    ax.imshow(overlay, origin='lower', extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5, colors='#cccccc')
    for s in ax.spines.values():
        s.set_color('#444')

    ax.plot(0, 0, 'w^', markersize=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    ax.annotate('', xy=(0, 3.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.0))

    coverage_pct = float(osz_pa.mean()) * 100.0

    x_ego, y_ego = emerge_xy
    if osz_source.in_bev_range(x_ego, y_ego):
        ax.plot(y_ego, x_ego, '*', color='lime', markersize=15,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        ax.text(y_ego + 0.8, x_ego + 0.8, frame_label,
                color='white', fontsize=6,
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])
    ax.set_title(f'{frame_label}  |  EMERGED  |  OSZ {coverage_pct:.1f}%',
                 fontsize=7, color='#e8e8e8')
    ax.set_xlabel('y (m) ←left | right→', fontsize=6, color='#aaaaaa')
    ax.set_ylabel('x (m) ↑ forward', fontsize=6, color='#aaaaaa')
    return coverage_pct


def _draw_info_panel(ax, nusc, event, idx, total, label_filter,
                     coverages, frames) -> None:
    """
    Right-hand info panel: event metadata + per-frame verdict table +
    per-frame OSZ coverage + key reminder. Pure text, no BEV.
    """
    ax.clear()
    ax.axis('off')
    ax.set_facecolor('#1a1d22')

    scene = nusc.get('scene', event['scene_token'])
    label_str = 'POSITIVE (ghost)' if event['label'] == 1 else 'NEGATIVE (visible)'
    label_color = '#ff5d5d' if event['label'] == 1 else '#5db7ff'

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
            verdict_rows.append((label, '—', '—', '#888'))
            continue
        if verdict == 'emerged':
            text, col = 'EMERGED', '#9be86b'
        elif verdict is True:
            text, col = 'IN OSZ', '#ff6b6b'
        elif verdict is False:
            text, col = 'visible', '#5db7ff'
        else:
            text, col = 'unknown', '#aaaaaa'
        cov_s = f'{cov:.1f}%' if cov is not None else '—'
        verdict_rows.append((label, text, cov_s, col))

    lines = []  # (text, color, weight)
    def add(text, color='#e8e8e8', weight='normal'):
        lines.append((text, color, weight))

    add(f'EVENT  [{idx+1} / {total}]', '#ffffff', 'bold')
    add(f'(showing label={label_filter} subset)', '#888')
    add('─' * 40, '#444')
    add(f'Label         : {label_str}', label_color, 'bold')
    add(f'Scene         : {scene["name"]}', '#e8e8e8')
    add(f'Instance      : {event["instance_token"][:18]}...', '#bdbdbd')
    add(f'Emerge sample : {event["emerge_sample"][:18]}...', '#bdbdbd')
    add(f'Emerge dist   : {emerge_dist:.1f} m', '#e8e8e8')
    add('─' * 40, '#444')
    add('Lookback verdicts (oldest → newest):', '#cccccc', 'bold')
    add(f'  {"frame":<6} {"verdict":<10} {"OSZ cov":<8}', '#888')
    for label, text, cov_s, col in verdict_rows:
        add(f'  {label:<6} {text:<10} {cov_s:<8}', col)
    add('─' * 40, '#444')
    add(f'OSZ overlap     : {n_osz} / {len(event["was_in_osz"])} lookback frames',
        '#ff6b6b' if n_osz else '#e8e8e8')
    add(f'Evidence frames : {n_ev} / {len(event["was_in_osz"])} '
        f'(unknown={n_unk})', '#e8e8e8')
    add('─' * 40, '#444')
    add('read the grid:', '#cccccc', 'bold')
    add('  • car stayed in OSZ?  red dots across t-4..t-1', '#888')
    add('  • when did it come out?  red→green transition', '#888')
    add('  • OSZ moving?  compare red shape across frames', '#888')
    add('  • ego turning?  OSZ shape rotates between frames', '#888')
    add('─' * 40, '#444')
    add('keys:  n=next   p=prev   r=redraw   q=quit', '#5db7ff', 'bold')

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
                   start_idx: int = 0, label_filter: int = 1) -> None:
    """Launch interactive EventBrowser; falls back to headless if no GUI."""
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

    if args.browse:
        print(f"\nLaunching EventBrowser "
              f"(label_filter={args.label_filter}, "
              f"{len([e for e in events if args.label_filter < 0 or e['label'] == args.label_filter])} events)...")
        print("GUI mode:   n=next  p=prev  r=redraw  q=quit   (click figure)")
        print("Headless:   same keys + j/k/+/-/number    (terminal input)")
        launch_browser(nusc, events, start_idx=args.start_idx,
                       label_filter=args.label_filter)
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
