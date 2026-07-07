"""
modules/drivable_filter.py
--------------------------
Stage 4c: intersect geometric OSZ with vehicle-plausible area from the
nuScenes HD map, producing PA-relevant OSZ.

Key design decision — how to handle coordinate frames:
    nuScenes get_map_geom(patch_angle=0) returns polygon in LOCAL coords
    centred at ego, but still aligned to GLOBAL north-up axes.
    Our BEV grid (RayCaster3D) uses EGO-CENTRIC coords:
        axis-0 = ego forward (ego +x)
        axis-1 = ego left   (ego +y)
    The rotation from global→ego is a 2D rotation by -ego_yaw around
    the origin. We apply this rotation DIRECTLY to the polygon vertices
    (exact, no image interpolation) and THEN rasterize in ego frame.
    This avoids all image-rotation artifacts and axis-direction confusion
    that came from rotating the rasterized image with scipy.ndimage.rotate.

No cv2 used anywhere — PIL.ImageDraw handles all rasterization.
"""

import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from shapely import affinity

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
    import pyquaternion
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


DRIVABLE_LAYERS = ['drivable_area', 'carpark_area']
EXCLUDE_LAYERS  = ['walkway', 'ped_crossing']
DEFAULT_DILATION_M = 1.5

_map_cache = {}


def get_nusc_map(dataroot: str, map_name: str) -> "NuScenesMap":
    key = (dataroot, map_name)
    if key not in _map_cache:
        _map_cache[key] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return _map_cache[key]


def _get_map_name(nusc, scene_token: str) -> str:
    scene = nusc.get('scene', scene_token)
    return nusc.get('log', scene['log_token'])['location']


def _get_ego_pose(nusc, sample_token: str) -> Tuple[np.ndarray, float]:
    """Returns (ego_translation_global [3], ego_yaw_rad)."""
    sample   = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep       = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    t   = np.array(ep['translation'], dtype=np.float64)
    q   = pyquaternion.Quaternion(ep['rotation'])
    return t, q.yaw_pitch_roll[0]


def _rotate_polygon_to_ego(geom, ego_yaw_rad: float):
    """
    Rotate a Shapely geometry from global-aligned local coords to ego-centric.

    Input coords are already centred at ego (patch-centred, as returned by
    get_map_geom), but still aligned to global north-up axes.
    Ego-centric means: x = ego-forward, y = ego-left.

    The rotation is: ego_frame = R(-ego_yaw) @ global_frame
    where R(-ego_yaw) rotates by -ego_yaw (clockwise by ego_yaw).

    Using Shapely affinity.rotate: positive angle = CCW.
    We need to rotate by -ego_yaw degrees (CCW = -ego_yaw_deg means CW).
    """
    angle_deg = -np.degrees(ego_yaw_rad)
    return affinity.rotate(geom, angle_deg, origin=(0, 0), use_radians=False)


def _rasterize_polygons_pil(
    geometries:  list,
    bev_range:   Tuple[float, float, float, float],  # (x_min,x_max,y_min,y_max)
    canvas_size: Tuple[int, int],                     # (nx, ny) = (W, H) in PIL
    fill_value:  int = 1,
) -> np.ndarray:
    """
    Rasterize Shapely geometries into a (nx, ny) BEV mask using PIL.
    No cv2 dependency.

    Coordinate mapping (ego-centric):
        metric x ∈ [x_min, x_max]  →  pixel col ∈ [0, nx)   (axis-0)
        metric y ∈ [y_min, y_max]  →  pixel row ∈ [0, ny)   (axis-1)

    Returns (nx, ny) array with indexing='ij' matching RayCaster3D.
    """
    x_min, x_max, y_min, y_max = bev_range
    nx, ny = canvas_size
    scale_x = nx / (x_max - x_min)
    scale_y = ny / (y_max - y_min)

    # PIL uses (W, H) = (nx, ny)
    img  = Image.new('L', (nx, ny), 0)
    draw = ImageDraw.Draw(img)

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        # normalise to flat list of Polygons
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = list(geom.geoms)
        elif geom.geom_type == 'GeometryCollection':
            polys = []
            for g in geom.geoms:
                if g.geom_type == 'Polygon':
                    polys.append(g)
                elif g.geom_type == 'MultiPolygon':
                    polys.extend(list(g.geoms))
        else:
            continue

        for poly in polys:
            if poly.is_empty:
                continue

            def to_pix(coords):
                """
                Convert metric (x,y) ego coords to PIL pixel (col, row).
                col = (x - x_min) * scale_x  [maps ego-x to image column]
                row = (y - y_min) * scale_y  [maps ego-y to image row]
                """
                return [
                    ((x - x_min) * scale_x, (y - y_min) * scale_y)
                    for x, y in coords
                ]

            ext = to_pix(poly.exterior.coords)
            if len(ext) >= 3:
                draw.polygon(ext, fill=fill_value)
            for interior in poly.interiors:
                hole = to_pix(interior.coords)
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    # PIL image is (W=nx, H=ny); np.array gives (H=ny, W=nx)
    # transpose to get (nx, ny) with indexing='ij'
    return np.array(img, dtype=np.uint8).T


def build_drivable_mask(
    nusc,
    sample_token: str,
    bev_range:    Tuple[float, float, float, float],
    bev_res:      float,
    dilation_m:   float = DEFAULT_DILATION_M,
) -> np.ndarray:
    """
    Build a (nx, ny) bool mask in EGO-CENTRIC BEV frame.
    True = vehicle could physically be here (drivable/carpark, not walkway).

    Never calls cv2. Falls back to all-True when map unavailable.
    """
    nx = int((bev_range[1] - bev_range[0]) / bev_res)
    ny = int((bev_range[3] - bev_range[2]) / bev_res)

    if not MAP_AVAILABLE:
        return np.ones((nx, ny), dtype=bool)

    sample      = nusc.get('sample', sample_token)
    map_name    = _get_map_name(nusc, sample['scene_token'])
    nusc_map    = get_nusc_map(nusc.dataroot, map_name)
    ego_t, ego_yaw = _get_ego_pose(nusc, sample_token)

    # patch_box in GLOBAL metres — nuScenes API uses (x,y,H,W)
    half_x = (bev_range[1] - bev_range[0]) / 2.0
    half_y = (bev_range[3] - bev_range[2]) / 2.0
    patch_box = (float(ego_t[0]), float(ego_t[1]), half_x * 2, half_y * 2)

    # get_map_geom(patch_angle=0): polygons returned in global-aligned LOCAL
    # coords centred at ego — i.e. (x - ego_t[0], y - ego_t[1]) in global frame
    include_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=DRIVABLE_LAYERS)
    exclude_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=EXCLUDE_LAYERS)

    all_include, all_exclude = [], []
    for _, geom_list in include_geoms:
        all_include.extend(geom_list)
    for _, geom_list in exclude_geoms:
        all_exclude.extend(geom_list)

    # Rotate polygons from global-aligned local → ego-centric
    # (rotate each polygon by -ego_yaw around origin)
    all_include_ego = [_rotate_polygon_to_ego(g, ego_yaw) for g in all_include]
    all_exclude_ego = [_rotate_polygon_to_ego(g, ego_yaw) for g in all_exclude]

    # Rasterize in ego-centric BEV frame
    canvas_size = (nx, ny)
    include_mask = _rasterize_polygons_pil(
        all_include_ego, bev_range, canvas_size)
    exclude_mask = _rasterize_polygons_pil(
        all_exclude_ego, bev_range, canvas_size)

    ego_mask = (include_mask > 0) & ~(exclude_mask > 0)

    if dilation_m > 0:
        dilation_px = max(1, int(round(dilation_m / bev_res)))
        ego_mask = binary_dilation(ego_mask, iterations=dilation_px)

    return ego_mask.astype(bool)


def filter_osz_by_drivable(
    osz_mask:      np.ndarray,
    drivable_mask: np.ndarray,
) -> np.ndarray:
    """PA-relevant OSZ = geometric OSZ ∩ drivable_area."""
    assert osz_mask.shape == drivable_mask.shape, (
        f"Shape mismatch: osz_mask {osz_mask.shape} vs "
        f"drivable_mask {drivable_mask.shape}. "
        f"Both must use the same bev_range and bev_res."
    )
    return osz_mask.astype(bool) & drivable_mask
