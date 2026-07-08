"""
PA_gen_v2/osz_source.py
=====================
Bridge between PA_gen_v2/ (ghost-vehicle mining) and OSZ/'s geometric
occlusion-shadow-zone computation.

This module is the single entry point for OSZ inside PA_gen_v2/. It wraps
OSZ/modules/ray_casting.py and the optional drivable-area filter, caches
per-frame results, and exposes coordinate helpers that use OSZ/'s (i, j)
axis convention:

    axis-0 = ego-x (forward),  axis-1 = ego-y (left)
    mask[i, j]  where  i = ego-x index,  j = ego-y index

Swapping these indices silently transposes the OSZ mask because the grid
is square; use only bev_xy_to_ij() / ij_to_bev_xy() below to avoid that
mistake.

Main functions:
  - get_osz_for_sample()            : raw geometric OSZ (for visualization)
  - get_pa_relevant_osz_for_sample() : raw OSZ ∩ drivable area (use this for
                                      vehicle-occlusion decisions)
  - get_drivable_mask_for_sample()  : nuScenes HD-map drivable area

Raw geometric OSZ can cover 70-80%+ of the BEV grid in dense urban scenes
because it counts building shadows; that is expected, not a bug. Phantom-
vehicle mining must use PA-relevant OSZ.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.bev_config import BEV_RANGE_XYXY, BEV_RESOLUTION_M
from OSZ.modules.ray_casting import (
    RayCaster3D,
    build_bev_occ_from_voxel_cast,
    cast_osz_2d,
)
from OSZ.utils.nuscenes_loader import NuScenesOSZLoader, NUSCENES_CAMERAS

# OSZ/modules/drivable_filter.py imports shapely unconditionally at its
# top level (only the nuScenes-map half of that file guards its own
# import). shapely is a real dependency of the drivable-area filter, but
# it is NOT a dependency of the rest of osz_source.py (raw OSZ works fine
# without it) — so we guard this import here rather than let a missing
# shapely install take down every function in this file. If it's
# missing, get_pa_relevant_osz_for_sample() below falls back to raw OSZ
# with an unmistakable one-time warning, instead of silently doing
# nothing or crashing deep inside a mining run.
try:
    from OSZ.modules.drivable_filter import (
        build_drivable_mask,
        filter_osz_by_drivable,
        MAP_AVAILABLE as _MAP_MODULE_AVAILABLE,
    )
    _DRIVABLE_FILTER_IMPORTABLE = True
except ImportError as _e:
    _DRIVABLE_FILTER_IMPORTABLE = False
    _MAP_MODULE_AVAILABLE = False
    print(f"[osz_source] [WARN] OSZ.modules.drivable_filter not importable "
          f"({_e}); drivable-area filtering is disabled, "
          f"get_pa_relevant_osz_for_sample() will return raw OSZ unfiltered. "
          f"Install shapely to enable it.")


# Height gate for the 3D voxel-cast stage. Not part of the shared BEV grid
# knob (common/bev_config.py) since it's a different axis (vertical, not
# the ground-plane cell size) — kept here as an explicit, visible constant
# rather than a buried default, matching OSZ/run_osz_pipeline.py's CLI
# defaults so PA_gen_v2/ and OSZ/ agree on what counts as "vehicle body".
Z_MIN = 0.3
Z_MAX = 2.2
Z_RES = 0.3


# ─────────────────────────────────────────────────────────────────────
# Shared RayCaster3D instance
# ─────────────────────────────────────────────────────────────────────
# Rebuilding RayCaster3D per-frame would recompute its voxel-centre grid
# (nx*ny*nz points) for nothing — the grid geometry never changes at
# runtime. One instance, built lazily on first use.
_caster: Optional[RayCaster3D] = None


def get_caster() -> RayCaster3D:
    global _caster
    if _caster is None:
        _caster = RayCaster3D(
            bev_range=BEV_RANGE_XYXY,
            bev_res=BEV_RESOLUTION_M,
            z_min=Z_MIN,
            z_max=Z_MAX,
            z_res=Z_RES,
        )
        print(f"[osz_source] RayCaster3D grid: {_caster.nx}x{_caster.ny}x{_caster.nz} "
              f"@ {BEV_RESOLUTION_M}m/cell (range={BEV_RANGE_XYXY})")
    return _caster


# ─────────────────────────────────────────────────────────────────────
# Per-loader cache: one NuScenesOSZLoader per (dataroot, version) so we
# don't re-open the nuScenes tables on every single-sample call.
# ─────────────────────────────────────────────────────────────────────
_loaders: Dict[Tuple[str, str], NuScenesOSZLoader] = {}


def _get_loader(nusc) -> NuScenesOSZLoader:
    """
    PA_gen_v2/'s call sites already hold a `nusc` (NuScenes) object
    (built once in each script's main()). Rather than have osz_source.py
    build its own second NuScenes instance from a dataroot string
    (doubling load time and memory), we wrap the caller's existing `nusc`
    in a lightweight NuScenesOSZLoader shim that reuses it directly.
    """
    key = id(nusc)
    if key not in _loaders:
        loader = NuScenesOSZLoader.__new__(NuScenesOSZLoader)
        loader.nusc = nusc
        loader.cameras = NUSCENES_CAMERAS
        loader.img_h = 900
        loader.img_w = 1600
        loader._use_mock = False
        _loaders[key] = loader
    return _loaders[key]


# ─────────────────────────────────────────────────────────────────────
# Per-scene OSZ cache (raw geometric) + PA-relevant (drivable-filtered) cache
# ─────────────────────────────────────────────────────────────────────
# mine_ghost_events() calls get_osz_for_sample() / get_pa_relevant_osz_for_
# sample() many times for overlapping lookback windows (frame t's lookback
# is t-1..t-4, frame t+1's lookback is t-3..t, etc.) — without a cache,
# every overlapping frame's (expensive: camera depth reprojection + KDTree
# densification + voxel cast, PLUS drivable-mask rasterisation) OSZ gets
# recomputed up to lookback_k times. Cleared per scene by the caller so
# memory doesn't grow across a full dataset run.
_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_drivable_cache: Dict[str, np.ndarray] = {}
_pa_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

_warned_map_unavailable = False   # print the "no map" warning once, not per-sample


# ─────────────────────────────────────────────────────────────────────
# Disk cache — persists computed OSZ results so visualization / re-mining
# doesn't recompute the expensive 3D voxel cast + 2D ray casting every run.
# One .npz per sample_token, under a config-keyed subdirectory so changing
# BEV_EXTENT_M / BEV_RESOLUTION_M / Z gates automatically invalidates.
# ─────────────────────────────────────────────────────────────────────
import hashlib as _hashlib

def _disk_cache_dir() -> Path:
    """Config-keyed cache dir; changes when BEV grid or Z-gate changes."""
    cfg = f"{BEV_RANGE_XYXY}_{BEV_RESOLUTION_M}_{Z_MIN}_{Z_MAX}_{Z_RES}"
    h = _hashlib.md5(cfg.encode()).hexdigest()[:8]
    return _REPO_ROOT / 'PA_gen_v2' / 'output' / 'osz_cache' / h


def _disk_load(sample_token: str):
    """Load cached OSZ from disk; returns None on miss."""
    p = _disk_cache_dir() / f"{sample_token}.npz"
    if not p.exists():
        return None
    try:
        d = np.load(p, allow_pickle=False)
        return (d['bev_occ'], d['osz_raw'], d['osz_pa'], d['drivable_mask'])
    except Exception:
        return None


def _disk_save(sample_token: str, bev_occ, osz_raw, osz_pa, drivable_mask) -> None:
    """Persist OSZ result to disk for future runs."""
    d = _disk_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sample_token}.npz"
    np.savez(p, bev_occ=bev_occ, osz_raw=osz_raw,
             osz_pa=osz_pa, drivable_mask=drivable_mask)


def clear_cache() -> None:
    """Clears every cache this module keeps (raw OSZ, drivable mask, PA OSZ)."""
    _cache.clear()
    _drivable_cache.clear()
    _pa_cache.clear()


def get_osz_for_sample(nusc, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns RAW geometric OSZ — everything behind an occluder, including
    the far side of buildings a vehicle could never occupy. For vehicle-
    occlusion decisions (ghost-vehicle mining), use
    get_pa_relevant_osz_for_sample() instead; this function stays useful
    for visualizing the before/after effect of the drivable-area filter.

    Returns:
        bev_occ  : (nx, ny) bool    — OSZ/'s solid occluder-surface
                                       occupancy (build_bev_occ_from_voxel_cast)
        osz_mask : (nx, ny) float32 — occlusion shadow zone, values in {0,1}

    Both arrays use OSZ's (i, j) = (ego-x index, ego-y index) convention.
    Index them as bev_occ[i, j] / osz_mask[i, j].
    """
    if sample_token in _cache:
        return _cache[sample_token]

    loader = _get_loader(nusc)
    frame = loader.build_frame_for_token(sample_token)
    cams = frame['cameras']
    if not cams:
        raise RuntimeError(
            f"No camera data for sample {sample_token}. Check that "
            f"--dataroot points at a real nuScenes root containing "
            f"samples/sweeps for this version, not just the metadata "
            f"tables."
        )

    caster = get_caster()
    bev_occ = build_bev_occ_from_voxel_cast(cams, caster)
    osz_mask = cast_osz_2d(bev_occ, caster).astype(np.float32)

    _cache[sample_token] = (bev_occ, osz_mask)
    return bev_occ, osz_mask


def get_drivable_mask_for_sample(nusc, sample_token: str) -> np.ndarray:
    """
    (nx, ny) bool — True where a vehicle could physically be (drivable
    area / carpark, per the nuScenes HD map, dilated by ~1.5m to cover
    the road edge / parking margin). Ego-centric, same (i,j) convention
    as everything else in this module.

    Falls back to an all-True mask (no filtering) if shapely or the
    nuScenes map API aren't importable, or if this sample's log has no
    map data — printed ONCE, not once per sample, since that would drown
    out everything else during a full mining run.
    """
    global _warned_map_unavailable

    if sample_token in _drivable_cache:
        return _drivable_cache[sample_token]

    caster = get_caster()
    nx, ny = caster.nx, caster.ny

    if not _DRIVABLE_FILTER_IMPORTABLE:
        mask = np.ones((nx, ny), dtype=bool)
    else:
        try:
            mask = build_drivable_mask(
                nusc=nusc,
                sample_token=sample_token,
                bev_range=caster.bev_range,
                bev_res=caster.bev_res,
            )
        except Exception as e:
            if not _warned_map_unavailable:
                print(f"[osz_source] [WARN] build_drivable_mask failed "
                      f"({e}); falling back to no filtering for this and "
                      f"any further samples with the same issue.")
                _warned_map_unavailable = True
            mask = np.ones((nx, ny), dtype=bool)

    _drivable_cache[sample_token] = mask
    return mask


def get_pa_relevant_osz_for_sample(
        nusc, sample_token: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw OSZ intersected with the drivable area — the "PA-relevant OSZ"
    that OSZ/run_osz_pipeline.py computes at Stage 4c. Use THIS for any
    vehicle-occlusion decision (ghost_vehicle_miner.py does). Raw OSZ
    alone routinely covers 70-80%+ of the grid in dense urban scenes
    because it also counts the shadow of buildings, which no vehicle can
    ever occupy — see this module's docstring and
    OSZ/modules/drivable_filter.py for why that's expected, and why this
    function exists.

    Returns:
        bev_occ       : (nx, ny) bool    — same as get_osz_for_sample()
        osz_raw       : (nx, ny) float32 — same as get_osz_for_sample()
        osz_pa        : (nx, ny) bool    — osz_raw ∩ drivable_mask
        drivable_mask : (nx, ny) bool    — from get_drivable_mask_for_sample()
    """
    # 1) in-memory cache (fastest, per-session)
    if sample_token in _pa_cache:
        return _pa_cache[sample_token]

    # 2) disk cache (fast, cross-session — avoids recomputing the expensive
    #    3D voxel cast + 2D ray casting on every visualization / re-mining run)
    disk = _disk_load(sample_token)
    if disk is not None:
        bev_occ, osz_raw, osz_pa, drivable_mask = disk
        result = (bev_occ, osz_raw, osz_pa, drivable_mask)
        _pa_cache[sample_token] = result
        return result

    # 3) compute from scratch (slow path)
    bev_occ, osz_raw = get_osz_for_sample(nusc, sample_token)
    drivable_mask = get_drivable_mask_for_sample(nusc, sample_token)

    if _DRIVABLE_FILTER_IMPORTABLE:
        osz_pa = filter_osz_by_drivable(osz_raw > 0.5, drivable_mask)
    else:
        osz_pa = (osz_raw > 0.5) & drivable_mask   # drivable_mask is all-True here anyway

    result = (bev_occ, osz_raw, osz_pa, drivable_mask)
    _pa_cache[sample_token] = result
    _disk_save(sample_token, bev_occ, osz_raw, osz_pa, drivable_mask)
    return result


# ─────────────────────────────────────────────────────────────────────
# Coordinate helpers — explicit (i, j), never (col, row) or (row, col)
# ─────────────────────────────────────────────────────────────────────

def bev_xy_to_ij(x_ego: float, y_ego: float) -> Tuple[int, int]:
    """
    Metric ego coords (x=forward, y=left) -> BEV array indices (i, j)
    matching OSZ/'s (nx, ny) indexing='ij' EXACTLY:
        i = ego-x index (axis 0),  j = ego-y index (axis 1)
    Index arrays as mask[i, j]. Do NOT swap to mask[j, i].
    """
    caster = get_caster()
    x_min, x_max, y_min, y_max = caster.bev_range
    i = int((x_ego - x_min) / caster.bev_res)
    j = int((y_ego - y_min) / caster.bev_res)
    return i, j


def ij_to_bev_xy(i: int, j: int) -> Tuple[float, float]:
    """Inverse of bev_xy_to_ij (returns the cell-centre metric coords)."""
    caster = get_caster()
    x_min, x_max, y_min, y_max = caster.bev_range
    x = x_min + (i + 0.5) * caster.bev_res
    y = y_min + (j + 0.5) * caster.bev_res
    return x, y


def in_bev_range(x_ego: float, y_ego: float) -> bool:
    caster = get_caster()
    x_min, x_max, y_min, y_max = caster.bev_range
    return x_min <= x_ego <= x_max and y_min <= y_ego <= y_max


def grid_shape() -> Tuple[int, int]:
    caster = get_caster()
    return caster.nx, caster.ny


def drivable_filter_available() -> bool:
    """
    True if OSZ/modules/drivable_filter.py imported successfully (shapely
    is installed). Callers should use this instead of reaching into the
    private _DRIVABLE_FILTER_IMPORTABLE flag directly, so this module can
    freely rename/restructure that internal state later.
    """
    return _DRIVABLE_FILTER_IMPORTABLE


def is_in_osz(x_ego: float, y_ego: float, osz_mask: np.ndarray) -> bool:
    """
    Check whether a metric ego-frame point falls inside an OSZ mask
    returned by get_osz_for_sample(). Returns False for points outside
    the BEV grid (can't claim occlusion for something we don't cover).
    """
    if not in_bev_range(x_ego, y_ego):
        return False
    i, j = bev_xy_to_ij(x_ego, y_ego)
    nx, ny = osz_mask.shape
    if not (0 <= i < nx and 0 <= j < ny):
        return False
    return bool(osz_mask[i, j] > 0.5)
