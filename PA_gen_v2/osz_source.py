"""
filter/osz_source.py
=====================
Bridges the ghost-vehicle mining pipeline (this folder) to OSZ/'s
occlusion-shadow-zone computation, instead of filter/ maintaining a
second, independent implementation.

Why this file exists
---------------------
filter/ used to have its own osz_geometry.py: raw LiDAR points binned
directly into a BEV grid (a simple height-clip, no camera reprojection),
then a 2D ray cast over that grid. OSZ/modules/ray_casting.py does this
more carefully — per-camera 3D voxel casting against the actual MEASURED
depth to find true occluder SURFACES (see that module's docstring for why
naive point-density binning creates range-dependent gaps), height-gated
to the vehicle-body band, and only THEN a 2D ego-centric ray cast over
the resulting solid occupancy map.

The user's instruction: OSZ/ is meant to be the shared source of truth
that other modules consume. So filter/ now calls into it directly.
osz_geometry.py is kept in this folder for reference only, renamed
osz_geometry_legacy.py, and nothing in the active pipeline imports it —
see that file's docstring.

get_osz_for_sample() below is a drop-in replacement for the old
filter/osz_geometry.get_osz_for_sample: same 2-tuple return, same
conceptual meaning, so ghost_vehicle_miner.py / visualize_events.py only
needed an import-line change, not a rewrite.

Coordinate convention — READ THIS BEFORE TOUCHING INDICES
------------------------------------------------------------
OSZ/ arrays are shape (nx, ny) with indexing='ij':
    axis-0 = ego-x (forward),  axis-1 = ego-y (left)
    i.e.  mask[i, j]  where  i = ego-x index,  j = ego-y index

filter/'s OLD osz_geometry.py used the opposite, image-style convention:
    mask[row, col]  where  row = ego-y index,  col = ego-x index

Because the grid is square, swapping these two conventions by accident
does NOT crash — it silently mirrors/transposes the OSZ mask along the
diagonal. That is exactly the kind of bug that "looks like it's working"
until someone overlays it on GT boxes and the shadow is rotated 90° from
where it should be.

To make this impossible to get wrong by habit, this module does NOT
expose a col/row-style helper. It exposes bev_xy_to_ij() / ij_to_bev_xy()
using OSZ's own (i, j) naming, and every docstring below says explicitly
"index as mask[i, j]". test_units.py has a regression test
(test_osz_source_ij_convention) that builds an intentionally asymmetric
synthetic occluder and would fail loudly if this axis order were ever
flipped.
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


# Height gate for the 3D voxel-cast stage. Not part of the shared BEV grid
# knob (common/bev_config.py) since it's a different axis (vertical, not
# the ground-plane cell size) — kept here as an explicit, visible constant
# rather than a buried default, matching OSZ/run_osz_pipeline.py's CLI
# defaults so filter/ and OSZ/ agree on what counts as "vehicle body".
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
    filter/'s call sites already hold a `nusc` (NuScenes) object (built
    once in each script's main()). Rather than have osz_source.py build
    its own second NuScenes instance from a dataroot string (doubling
    load time and memory), we wrap the caller's existing `nusc` in a
    lightweight NuScenesOSZLoader shim that reuses it directly.
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
# Per-scene OSZ cache
# ─────────────────────────────────────────────────────────────────────
# mine_ghost_events() calls get_osz_for_sample() many times for
# overlapping lookback windows (frame t's lookback is t-1..t-4, frame
# t+1's lookback is t-3..t, etc.) — without a cache, every overlapping
# frame's (expensive: camera depth reprojection + KDTree densification +
# voxel cast) OSZ gets recomputed up to lookback_k times. Cleared per
# scene by the caller so memory doesn't grow across a full dataset run.
_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def clear_cache() -> None:
    _cache.clear()


def get_osz_for_sample(nusc, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for the old filter/osz_geometry.get_osz_for_sample.

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
