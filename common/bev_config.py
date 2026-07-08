"""
common/bev_config.py
=====================
The ONE place that defines the BEV grid shared by OSZ/, PA_gen_v1/, and
PA_gen_v2/.

Karpathy rule: don't rely on three files independently hard-coding "0.2"
(or "0.4", or "0.5" — which is exactly what had happened here) and hoping
nobody edits only two of them. There is exactly one knob below. Change it
here; every module that imports from this file picks it up automatically.

Historically each consumer used a different tuple ORDER for its BEV range:
    OSZ/       : (x_min, x_max, y_min, y_max)
    PA_gen_v1/ : (x0, y0, x1, y1)          [= x_min, y_min, x_max, y_max]
    PA_gen_v2/ : a single symmetric extent + a square pixel grid
Rather than force every file to adopt a common order (risking a silent
x/y transposition bug somewhere), this module derives the SAME underlying
grid in each of those shapes from the same two source numbers below.
Each consumer imports the flavor it already expects.

To change the BEV cell size for the WHOLE project:
    edit BEV_RESOLUTION_M below, nothing else.

Note on OSZ disk cache invalidation (PA_gen_v2/osz_source.py):
    The on-disk .npz cache at `output/osz_cache/{config_hash}/` is keyed
    by an md5 of (BEV_RANGE_XYXY, BEV_RESOLUTION_M, Z_MIN, Z_MAX, Z_RES).
    Changing any of those here automatically switches the cache to a
    new directory — old cached results are not deleted but will be
    ignored. To reclaim disk space, manually remove
    `PA_gen_v2/output/osz_cache/<old_hash>/`.
"""

# ─────────────────────────────────────────────────────────────────────
# THE TWO NUMBERS THAT DEFINE THE PROJECT'S ENTIRE BEV GRID
# ─────────────────────────────────────────────────────────────────────
BEV_EXTENT_M     = 50.0   # ego-centred half-extent (metres), same on x & y
BEV_RESOLUTION_M = 0.2    # metres per BEV cell  <-- CHANGE THIS ONE NUMBER

# ─────────────────────────────────────────────────────────────────────
# Everything below is DERIVED — do not hand-edit these.
# ─────────────────────────────────────────────────────────────────────

# OSZ/ convention: (x_min, x_max, y_min, y_max)  — used by
# OSZ/run_osz_pipeline.py, OSZ/modules/ray_casting.py (RayCaster3D),
# OSZ/modules/drivable_filter.py
BEV_RANGE_XYXY = (-BEV_EXTENT_M, BEV_EXTENT_M, -BEV_EXTENT_M, BEV_EXTENT_M)

# PA_gen_v1/ convention: (x0, y0, x1, y1)  — used by
# PA_gen_v1/create_pa_labels_mini.py, create_pa_labels_full.py
BEV_RANGE_X0Y0X1Y1 = (-BEV_EXTENT_M, -BEV_EXTENT_M, BEV_EXTENT_M, BEV_EXTENT_M)

# PA_gen_v2/ convention: single symmetric extent + square pixel grid,
# ego always sits at pixel (BEV_CENTER, BEV_CENTER)
BEV_RANGE_M = BEV_EXTENT_M

# Grid dimensions — identical everywhere, since every flavor above derives
# from the same extent/resolution. Asserted equal at import time so a
# future edit that breaks the invariant fails loudly instead of silently
# producing mismatched grids again.
_nx_osz = int(round((BEV_RANGE_XYXY[1] - BEV_RANGE_XYXY[0]) / BEV_RESOLUTION_M))
_ny_osz = int(round((BEV_RANGE_XYXY[3] - BEV_RANGE_XYXY[2]) / BEV_RESOLUTION_M))
_nx_pre = int(round((BEV_RANGE_X0Y0X1Y1[2] - BEV_RANGE_X0Y0X1Y1[0]) / BEV_RESOLUTION_M))
_ny_pre = int(round((BEV_RANGE_X0Y0X1Y1[3] - BEV_RANGE_X0Y0X1Y1[1]) / BEV_RESOLUTION_M))
assert _nx_osz == _ny_osz == _nx_pre == _ny_pre, (
    "BEV grid must come out square and identical across conventions — "
    f"got OSZ=({_nx_osz},{_ny_osz}) PA_gen_v1=({_nx_pre},{_ny_pre}). "
    "This should be impossible unless BEV_EXTENT_M/BEV_RESOLUTION_M were "
    "hand-edited inconsistently above."
)

BEV_NX     = _nx_osz          # cells along ego-x (forward)
BEV_NY     = _ny_osz          # cells along ego-y (left)
BEV_SIZE   = BEV_NX           # PA_gen_v2/'s name for the same quantity (square grid)
BEV_CENTER = BEV_SIZE // 2    # pixel index of ego, PA_gen_v2/'s convention


def describe() -> str:
    return (f"BEV grid: {BEV_NX}x{BEV_NY} cells @ {BEV_RESOLUTION_M}m/cell "
            f"(±{BEV_EXTENT_M}m range, {BEV_NX*BEV_NY:,} cells total)")


# ─────────────────────────────────────────────────────────────────────
# Shared coordinate helpers (PA_gen_v2/'s pixel<->metric convention).
# Previously duplicated inside PA_gen_v2/osz_geometry_legacy.py; centralised here
# so there is exactly one implementation to trust.
# ─────────────────────────────────────────────────────────────────────

def bev_coords_to_pixel(x_ego: float, y_ego: float):
    """metric (x, y) in ego frame -> BEV pixel (col, row). Ego at BEV_CENTER."""
    col = int((x_ego + BEV_RANGE_M) / BEV_RESOLUTION_M)
    row = int((y_ego + BEV_RANGE_M) / BEV_RESOLUTION_M)
    return col, row


def pixel_to_bev_coords(col: int, row: int):
    """Inverse of bev_coords_to_pixel."""
    x = col * BEV_RESOLUTION_M - BEV_RANGE_M
    y = row * BEV_RESOLUTION_M - BEV_RANGE_M
    return x, y


if __name__ == '__main__':
    print(describe())
    print(f"  BEV_RANGE_XYXY      = {BEV_RANGE_XYXY}   (OSZ/)")
    print(f"  BEV_RANGE_X0Y0X1Y1  = {BEV_RANGE_X0Y0X1Y1}   (PA_gen_v1/)")
    print(f"  BEV_RANGE_M         = {BEV_RANGE_M}, BEV_SIZE={BEV_SIZE}, "
          f"BEV_CENTER={BEV_CENTER}   (PA_gen_v2/)")
