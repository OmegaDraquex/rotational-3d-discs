"""
High-Resolution, 100% Manifold (0 Open Edges, 0 Non-Manifold Edges) 3D STL Mesh Generator
for Circular Lithophanes & Perforated Hole Discs.

Fixes both issues seen in Bambu Studio:
  1. Zero Open / Non-Manifold Edges:
     - Uses a continuous vertex-grid triangulation for Lithophane mode where adjacent cells
       share exact vertex coordinates and vertical transitions -> 0 open edges!
     - Resolves diagonal "checkerboard" kiss-points in multi-part AMS slot regions and Holes mode
       so two cells never share a single vertical edge across 4 faces -> 0 non-manifold edges!
  2. High Resolution & Smooth Circular Perimeter:
     - Upsamples thickness & slot maps to high mesh resolution (default 320x320, ~0.25mm grid
       suitable for 0.2mm and 0.4mm nozzles).
     - Radially projects boundary vertices onto the exact circle radius R = diameter / 2 so
       the outer rim is a smooth cylinder instead of jagged staircase blocks!
"""

import io
import math
import struct
import zipfile
from typing import List, Optional
import numpy as np
from scipy.ndimage import zoom, gaussian_filter, median_filter
from backend.preprocessor import create_circular_mask


def apply_rim_notches(
    outer_circle: np.ndarray,
    notch_angles_deg: List[float],
    notch_radius_ratio: float = 0.042,
) -> np.ndarray:
    """
    Cuts clean inward semicircular thumb/visual alignment notches into `outer_circle`
    at each clockwise angle in `notch_angles_deg` (where 0° is 12 o'clock top center,
    +90° is 3 o'clock right, etc.).
    Because the notch center sits right on the outer circle perimeter and has radius
    `notch_radius_ratio * N` (~3.3mm wide on an 80mm disc), it stays strictly inside the
    outer structural rim without touching the active image area!
    """
    n = outer_circle.shape[0]
    out = outer_circle.copy()
    center = (n - 1) / 2.0
    disc_r = n / 2.0
    notch_r_sq = (n * notch_radius_ratio) ** 2
    yy, xx = np.ogrid[:n, :n]

    for deg in notch_angles_deg:
        rad = math.radians(deg)
        # 0° is top center: x = center + sin(rad)*R, y = center - cos(rad)*R
        nx = center + math.sin(rad) * disc_r
        ny = center - math.cos(rad) * disc_r
        in_notch = ((xx - nx) ** 2 + (yy - ny) ** 2) <= notch_r_sq
        out[in_notch] = False

    return out


def _triangles_to_binary_stl(triangles: np.ndarray, header_text: str = "Rotational3DDisc") -> bytes:
    """
    Converts an (M, 3, 3) float32 array of triangle vertices into standard Binary STL bytes.
    """
    if triangles.shape[0] == 0:
        triangles = np.zeros((1, 3, 3), dtype=np.float32)

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    Valid = (norms[:, 0] > 1e-11)
    if np.any(Valid):
        triangles = triangles[Valid]
        normals = normals[Valid] / norms[Valid]
    else:
        normals = np.zeros_like(normals)

    num_tris = triangles.shape[0]
    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])

    records = np.zeros(num_tris, dtype=stl_dtype)
    records["normal"] = normals.astype(np.float32)
    records["v0"] = triangles[:, 0, :].astype(np.float32)
    records["v1"] = triangles[:, 1, :].astype(np.float32)
    records["v2"] = triangles[:, 2, :].astype(np.float32)
    records["attr"] = 0

    header_bytes = header_text.encode("ascii", errors="ignore")[:80].ljust(80, b"\x00")
    buf = io.BytesIO()
    buf.write(header_bytes)
    buf.write(struct.pack("<I", num_tris))
    buf.write(records.tobytes())
    return buf.getvalue()


def _fix_checkerboard_manifold(mask: np.ndarray) -> np.ndarray:
    """
    Eliminates 2x2 diagonal checkerboard patterns:
        1 0      or      0 1
        0 1              1 0
    where two solid cells touch ONLY at a single vertical corner line (which causes
    Bambu Studio's 'non-manifold edge' error because 4 vertical walls meet at 1 vertical edge!).
    By filling the bridge cell on diagonal kiss points, every vertical edge belongs to
    strictly 2 faces (100% 2-manifold).
    """
    m = mask.copy()
    for _ in range(3):
        c00 = m[:-1, :-1]
        c01 = m[:-1, 1:]
        c10 = m[1:, :-1]
        c11 = m[1:, 1:]

        diag1 = c00 & c11 & (~c01) & (~c10)
        diag2 = c01 & c10 & (~c00) & (~c11)
        if not (np.any(diag1) or np.any(diag2)):
            break
        m[:-1, 1:][diag1] = True
        m[:-1, :-1][diag2] = True
    return m


def _build_smooth_vertex_coords(
    n: int,
    diameter_mm: float,
    solid_mask: np.ndarray,
    chamfer_mm: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates vertex coordinate grids VX, VY in millimeters:
      - `vx_top, vy_top`: smoothly projected onto the exact circle radius R = diameter / 2
        so the outer rim is a sleek, perfectly circular cylinder without blocky steps.
      - `vx_bot, vy_bot`: bottom perimeter is inset by `chamfer_mm` (default 0.6mm) to create
        a clean 45° bottom bevel/chamfer undercut on Layer 1. This allows a print scraper blade
        or fingernail to easily catch under the edge and lift the disc off Smooth PEI plates
        without bending or damaging the disc!
    """
    half_d = diameter_mm / 2.0
    lin = np.linspace(-half_d, half_d, n + 1, dtype=np.float32)
    vx, vy = np.meshgrid(lin, -lin)  # vy goes from +half_d at row 0 to -half_d at row n

    # Find which of the 4 cells surrounding each vertex (r, c) are inside the disc
    m_pad = np.pad(solid_mask, pad_width=1, mode="constant", constant_values=False)
    c_nw = m_pad[:-1, :-1]
    c_ne = m_pad[:-1, 1:]
    c_sw = m_pad[1:, :-1]
    c_se = m_pad[1:, 1:]
    count_adj = c_nw.astype(np.int32) + c_ne.astype(np.int32) + c_sw.astype(np.int32) + c_se.astype(np.int32)

    # Outer perimeter vertices have 1, 2, or 3 adjacent active cells AND lie near the outer rim
    rad = np.sqrt(vx ** 2 + vy ** 2)
    is_outer_boundary = (count_adj > 0) & (count_adj < 4) & (rad > half_d * 0.88)
    safe_rad = np.where(rad < 1e-6, 1.0, rad)

    # Top boundary vertices projected onto exact circle radius half_d
    vx_top = vx.copy()
    vy_top = vy.copy()
    vx_top[is_outer_boundary] = vx[is_outer_boundary] * (half_d / safe_rad[is_outer_boundary])
    vy_top[is_outer_boundary] = vy[is_outer_boundary] * (half_d / safe_rad[is_outer_boundary])

    # Recompute radius with projected top outer boundary
    rad_top = np.sqrt(vx_top ** 2 + vy_top ** 2)
    safe_rad_top = np.where(rad_top < 1e-6, 1.0, rad_top)

    # Monotonic radial taper for the bottom contact layer (Z = 0) to create a smooth, easy-peel rim chamfer
    # smoothly contracting the outer perimeter by chamfer_mm without any mesh folding
    w_zone = 2.5  # mm transition width near outer edge
    r0 = max(0.0, half_d - w_zone)
    scale_factor = (w_zone - chamfer_mm) / w_zone

    in_taper_zone = rad_top > r0
    rad_bot = rad_top.copy()
    rad_bot[in_taper_zone] = r0 + (rad_top[in_taper_zone] - r0) * scale_factor

    bot_ratio = rad_bot / safe_rad_top
    vx_bot = vx_top * bot_ratio
    vy_bot = vy_top * bot_ratio

    return vx_top, vy_top, vx_bot, vy_bot


def _build_watertight_region_mesh(
    solid_mask: np.ndarray,
    z_top_cell: np.ndarray,
    vx_top: np.ndarray,
    vy_top: np.ndarray,
    smooth_top_surface: bool = True,
    z_bottom_cell: Optional[np.ndarray] = None,
    vx_bot: Optional[np.ndarray] = None,
    vy_bot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Builds a 100% watertight, 0-open-edge, 0-non-manifold-edge triangle mesh for the 2D boolean
    region `solid_mask` (n, n):
      - Top surface corner heights `VZ_top` (n+1, n+1) are computed as the average of adjacent
        active cells in `solid_mask`. Because adjacent cells share exact corner vertices `(vx, vy, vz)`,
        there are ZERO internal vertical step walls and ZERO open edges!
      - Bottom surface supports either flat bed contact (`z_bottom_cell=None`, Z=0 with easy-peel chamfer)
        or elevated base support for layered multi-color glazes (`z_bottom_cell=Z_base(x,y)`).
      - Perimeter boundary walls connect top vertices to bottom vertices strictly where `solid_mask`
        transitions from True to False, closing all borders with 0 open edges!
    """
    mask = _fix_checkerboard_manifold(solid_mask)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    n = mask.shape[0]

    if vx_bot is None:
        vx_bot = vx_top
    if vy_bot is None:
        vy_bot = vy_top

    # Compute corner vertex height VZ_top (n+1, n+1) averaged ONLY over active cells in `mask`
    m_pad = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    z_pad = np.pad(np.where(mask, z_top_cell, 0.0), pad_width=1, mode="constant", constant_values=0.0)

    c_nw = m_pad[:-1, :-1].astype(np.float32)
    c_ne = m_pad[:-1, 1:].astype(np.float32)
    c_sw = m_pad[1:, :-1].astype(np.float32)
    c_se = m_pad[1:, 1:].astype(np.float32)

    z_nw = z_pad[:-1, :-1]
    z_ne = z_pad[:-1, 1:]
    z_sw = z_pad[1:, :-1]
    z_se = z_pad[1:, 1:]

    denom = np.maximum(1.0, c_nw + c_ne + c_sw + c_se)
    vz_top = (z_nw + z_ne + z_sw + z_se) / denom

    # Compute corner vertex height VZ_bottom (n+1, n+1)
    if z_bottom_cell is not None:
        zb_pad = np.pad(np.where(mask, z_bottom_cell, 0.0), pad_width=1, mode="constant", constant_values=0.0)
        zb_nw = zb_pad[:-1, :-1]
        zb_ne = zb_pad[:-1, 1:]
        zb_sw = zb_pad[1:, :-1]
        zb_se = zb_pad[1:, 1:]
        vz_bottom = (zb_nw + zb_ne + zb_sw + zb_se) / denom
    else:
        vz_bottom = np.zeros_like(vz_top)

    # For vertices touching the bed (Z < 0.05mm), use the easy-peel chamfered coordinates (vx_bot, vy_bot).
    # For elevated surfaces (Z >= 0.05mm), use full radius coordinates (vx_top, vy_top).
    is_bed_contact = vz_bottom < 0.05
    xb_grid = np.where(is_bed_contact, vx_bot, vx_top)
    yb_grid = np.where(is_bed_contact, vy_bot, vy_top)

    # Extract 4 corner vertices for each active cell (ys, xs)
    # NW: (y, x), NE: (y, x+1), SE: (y+1, x+1), SW: (y+1, x)
    # Top vertices:
    x00_t, y00_t, z00_t = vx_top[ys + 1, xs], vy_top[ys + 1, xs], vz_top[ys + 1, xs]
    x10_t, y10_t, z10_t = vx_top[ys + 1, xs + 1], vy_top[ys + 1, xs + 1], vz_top[ys + 1, xs + 1]
    x11_t, y11_t, z11_t = vx_top[ys, xs + 1], vy_top[ys, xs + 1], vz_top[ys, xs + 1]
    x01_t, y01_t, z01_t = vx_top[ys, xs], vy_top[ys, xs], vz_top[ys, xs]

    # Bottom vertices:
    x00_b, y00_b, z00_b = xb_grid[ys + 1, xs], yb_grid[ys + 1, xs], vz_bottom[ys + 1, xs]
    x10_b, y10_b, z10_b = xb_grid[ys + 1, xs + 1], yb_grid[ys + 1, xs + 1], vz_bottom[ys + 1, xs + 1]
    x11_b, y11_b, z11_b = xb_grid[ys, xs + 1], yb_grid[ys, xs + 1], vz_bottom[ys, xs + 1]
    x01_b, y01_b, z01_b = xb_grid[ys, xs], yb_grid[ys, xs], vz_bottom[ys, xs]

    p00_t = np.stack([x00_t, y00_t, z00_t], axis=-1)
    p10_t = np.stack([x10_t, y10_t, z10_t], axis=-1)
    p11_t = np.stack([x11_t, y11_t, z11_t], axis=-1)
    p01_t = np.stack([x01_t, y01_t, z01_t], axis=-1)

    p00_b = np.stack([x00_b, y00_b, z00_b], axis=-1)
    p10_b = np.stack([x10_b, y10_b, z10_b], axis=-1)
    p11_b = np.stack([x11_b, y11_b, z11_b], axis=-1)
    p01_b = np.stack([x01_b, y01_b, z01_b], axis=-1)

    tris = []

    # 1. Top surface triangles (CCW normal +Z): (p00 -> p10 -> p11) and (p00 -> p11 -> p01)
    tris.append(np.stack([p00_t, p10_t, p11_t], axis=1))
    tris.append(np.stack([p00_t, p11_t, p01_t], axis=1))

    # 2. Bottom surface triangles (CCW normal -Z): (p00 -> p11 -> p10) and (p00 -> p01 -> p11)
    tris.append(np.stack([p00_b, p11_b, p10_b], axis=1))
    tris.append(np.stack([p00_b, p01_b, p11_b], axis=1))

    # 3. Boundary vertical / sloped walls ONLY where neighbor cell is False!
    has_north = m_pad[ys, xs + 1]      # neighbor at (y-1, x)
    has_south = m_pad[ys + 2, xs + 1]  # neighbor at (y+1, x)
    has_west = m_pad[ys + 1, xs]       # neighbor at (y, x-1)
    has_east = m_pad[ys + 1, xs + 2]   # neighbor at (y, x+1)

    def _emit_boundary_wall(cond_no_neighbor, vt_start, vt_end, vb_start, vb_end):
        idx = np.nonzero(cond_no_neighbor)[0]
        if len(idx) == 0:
            return
        ts, te = vt_start[idx], vt_end[idx]
        bs, be = vb_start[idx], vb_end[idx]
        # Outward normal when viewed from outside: bs -> be -> te and bs -> te -> ts
        tris.append(np.stack([bs, be, te], axis=1))
        tris.append(np.stack([bs, te, ts], axis=1))

    # North boundary (+Y): top edge goes p11 -> p01
    _emit_boundary_wall(~has_north, p11_t, p01_t, p11_b, p01_b)
    # South boundary (-Y): top edge goes p00 -> p10
    _emit_boundary_wall(~has_south, p00_t, p10_t, p00_b, p10_b)
    # East boundary (+X): top edge goes p10 -> p11
    _emit_boundary_wall(~has_east, p10_t, p11_t, p10_b, p11_b)
    # West boundary (-X): top edge goes p01 -> p00
    _emit_boundary_wall(~has_west, p01_t, p00_t, p01_b, p00_b)

    return np.concatenate(tris, axis=0)


def _upsample_disc_maps(
    thickness_map_mm: np.ndarray,
    slot_map: np.ndarray = None,
    mode: str = "lithophane",
    target_res: int = 280,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsamples the solver grid (e.g. 200x200) to a fine 3D mesh grid (`target_res` x `target_res`,
    default 280x280 -> ~0.28mm resolution on an 80mm disc) and applies anti-aliasing / median
    smoothing so lithophane surfaces and color regions are smooth and high-resolution instead of blocky!
    """
    n_in = thickness_map_mm.shape[0]
    if mode in ("holes", "halftone"):
        # In Holes and Halftone Photo modes, keep crisp aperture boundaries without cubic blurring
        scale_factor = max(1, target_res // n_in)
        if scale_factor > 1:
            z_up = np.repeat(np.repeat(thickness_map_mm, scale_factor, axis=0), scale_factor, axis=1)
            s_up = (
                np.repeat(np.repeat(slot_map, scale_factor, axis=0), scale_factor, axis=1)
                if slot_map is not None
                else None
            )
        else:
            z_up = thickness_map_mm.copy()
            s_up = slot_map.copy() if slot_map is not None else None
        return z_up, s_up

    # In Lithophane / Ultra-Thin modes: smooth cubic upsampling of the heightmap
    zoom_ratio = float(target_res) / float(n_in)
    z_up = zoom(thickness_map_mm.astype(np.float32), zoom_ratio, order=3)
    z_up = gaussian_filter(z_up, sigma=0.75)

    s_up = None
    if slot_map is not None:
        cleaned_slots = median_filter(slot_map.astype(np.int32), size=3)
        s_up = zoom(cleaned_slots, zoom_ratio, order=0)
        s_up = median_filter(s_up, size=3)

    return z_up, s_up


def generate_disc_stl(
    thickness_map_mm: np.ndarray,
    diameter_mm: float = 80.0,
    max_thickness_mm: float = 2.0,
    mode: str = "lithophane",
    notch_angles_deg: Optional[List[float]] = None,
) -> bytes:
    """
    Generates a high-resolution, 100% manifold (0 open edges, 0 non-manifold edges) binary STL
    of a circular disc of `diameter_mm`, including functional inward alignment notches on the outer rim
    and an easy-peel 0.6mm chamfer undercut on the bottom bed perimeter:
      - Disc 1 has ONE notch at 0° (top 12 o'clock).
      - Disc 2 has TWO notches: one at 0° (Image 1 alignment) and one at +angle_deg (Image 2 alignment).
      - Base skin: Continuous 0.24mm White PLA floor across the entire disc prevents tearing/shredding!
    """
    z_up, _ = _upsample_disc_maps(thickness_map_mm, None, mode=mode, target_res=280)
    n = z_up.shape[0]

    raw_circle = create_circular_mask(n, margin_ratio=0.0)
    active_circle = create_circular_mask(n, margin_ratio=0.055)
    outer_circle = (
        apply_rim_notches(raw_circle, notch_angles_deg, notch_radius_ratio=0.038)
        if notch_angles_deg
        else raw_circle
    )
    rim_mask = outer_circle & (~active_circle)

    # Flush rim thickness matches the disc's active max height so stacked discs have ZERO air gap!
    active_max = float(np.max(thickness_map_mm)) if np.max(thickness_map_mm) > 0.2 else max_thickness_mm
    base_floor = 0.24  # Continuous 0.24mm (2-layer) White PLA base skin across entire circle

    z_up[rim_mask] = active_max
    z_up[~outer_circle] = 0.0

    # Ensure continuous 0.24mm floor across outer_circle so Layer 1 & 2 are 100% solid
    z_up[outer_circle] = np.maximum(base_floor, z_up[outer_circle])
    solid_mask = outer_circle.copy()

    vx_top, vy_top, vx_bot, vy_bot = _build_smooth_vertex_coords(n, diameter_mm, raw_circle, chamfer_mm=0.6)
    triangles = _build_watertight_region_mesh(
        solid_mask,
        z_up,
        vx_top,
        vy_top,
        smooth_top_surface=(mode not in ("holes", "halftone")),
        vx_bot=vx_bot,
        vy_bot=vy_bot,
    )
    return _triangles_to_binary_stl(triangles, header_text=f"RotationalDisc_{mode}")


def generate_slot_stl(
    thickness_map_mm: np.ndarray,
    slot_map: np.ndarray,
    target_slot: int,
    diameter_mm: float = 80.0,
    max_thickness_mm: float = 2.0,
    mode: str = "lithophane",
    pre_upsampled: tuple = None,
    notch_angles_deg: Optional[List[float]] = None,
) -> bytes:
    """
    Generates a 100% watertight, 0-open-edge, 0-non-manifold-edge binary STL mesh for a single
    Bambu AMS filament slot `target_slot` (1..K), implementing the Layered Color Glaze architecture:
      - Slot 1 (Jade White PLA):
        Acts as the continuous structural base & optical diffuser. Spans the entire disc from Z=0.0
        up to Z_base(x, y), guaranteeing 100% solid first 2 layers (0 switches on Layer 1) and
        smooth bed adhesion with the easy-peel rim chamfer.
      - Slots 2..K (Colored Filaments):
        Print ONLY as a thin top color glaze (0.24mm thick, exactly 2 layers at 0.12mm) elevated
        directly on top of Slot 1's white pedestal. This ensures colored filaments act as vibrant
        translucent filters without ever becoming opaque, light-blocking slabs!
    """
    base_floor = 0.24       # Continuous White PLA base skin (mm)
    glaze_thickness = 0.24  # Glaze thickness (2 layers at 0.12mm)
    active_max = float(np.max(thickness_map_mm)) if np.max(thickness_map_mm) > 0.2 else max_thickness_mm

    if pre_upsampled is not None:
        z_up, s_up, vx_top, vy_top, vx_bot, vy_bot, outer_circle, rim_mask = pre_upsampled
    else:
        z_up, s_up = _upsample_disc_maps(thickness_map_mm, slot_map, mode=mode, target_res=260)
        n = z_up.shape[0]
        raw_circle = create_circular_mask(n, margin_ratio=0.0)
        active_circle = create_circular_mask(n, margin_ratio=0.055)
        outer_circle = (
            apply_rim_notches(raw_circle, notch_angles_deg, notch_radius_ratio=0.038)
            if notch_angles_deg
            else raw_circle
        )
        rim_mask = outer_circle & (~active_circle)
        z_up[rim_mask] = active_max
        s_up[rim_mask] = 1
        z_up[outer_circle] = np.maximum(base_floor, z_up[outer_circle])
        vx_top, vy_top, vx_bot, vy_bot = _build_smooth_vertex_coords(n, diameter_mm, raw_circle, chamfer_mm=0.6)

    if target_slot == 1:
        # Slot 1 is the structural Jade White base & relief across the ENTIRE disc
        slot_mask = outer_circle.copy()
        # Bottom is on the build plate (Z = 0.0) with easy-peel chamfer
        z_bottom = np.zeros_like(z_up)
        # Top: where slot == 1 or rim, it extends to full height z_up;
        # where colored glazes sit (s_up >= 2), Slot 1 rises to support the glaze pedestal
        z_top = np.where(
            (s_up == 1) | rim_mask,
            z_up,
            np.maximum(base_floor, z_up - glaze_thickness),
        )
        z_top[rim_mask] = active_max
        triangles = _build_watertight_region_mesh(
            slot_mask,
            z_top,
            vx_top,
            vy_top,
            smooth_top_surface=(mode not in ("holes", "halftone")),
            z_bottom_cell=z_bottom,
            vx_bot=vx_bot,
            vy_bot=vy_bot,
        )
        return _triangles_to_binary_stl(triangles, header_text="AMS_Slot_1_WhiteBase")

    else:
        # Slots 2..K are thin color glazes elevated on top of the white base pedestal
        slot_mask = (s_up == target_slot) & (z_up > 0.05) & outer_circle
        if not np.any(slot_mask):
            return _triangles_to_binary_stl(np.zeros((0, 3, 3), dtype=np.float32), header_text=f"AMS_Slot_{target_slot}")

        # Bottom sits directly on top of Slot 1's pedestal
        z_bottom = np.maximum(base_floor, z_up - glaze_thickness)
        # Top is the outer surface of the disc
        z_top = np.maximum(z_bottom + 0.12, z_up)

        # Elevated surface: no bed contact, so bottom vertices use vx_top, vy_top
        triangles = _build_watertight_region_mesh(
            slot_mask,
            z_top,
            vx_top,
            vy_top,
            smooth_top_surface=(mode not in ("holes", "halftone")),
            z_bottom_cell=z_bottom,
            vx_bot=vx_top,
            vy_bot=vy_top,
        )
        return _triangles_to_binary_stl(triangles, header_text=f"AMS_Slot_{target_slot}_Glaze")


def package_both_stls_zip(
    disc1_mm: np.ndarray,
    disc2_mm: np.ndarray,
    diameter_mm: float = 80.0,
    max_thickness_mm: float = 2.0,
    mode: str = "lithophane",
    angle_deg: float = 90.0,
    color_mode: str = "monochrome",
    slot1_map: np.ndarray = None,
    slot2_map: np.ndarray = None,
    palette_info: list = None,
) -> bytes:
    """
    Generates high-resolution, 100% manifold `disc_1` and `disc_2` STL files with functional
    inward alignment notches:
      - Disc 1: ONE notch at 0° (top 12 o'clock)
      - Disc 2: TWO notches at 0° (Image 1) and +angle_deg (Image 2)
    """
    notches_disc1 = [0.0]
    notches_disc2 = [0.0, float(angle_deg)]

    stl1_bytes = generate_disc_stl(
        disc1_mm, diameter_mm, max_thickness_mm, mode, notch_angles_deg=notches_disc1
    )
    stl2_bytes = generate_disc_stl(
        disc2_mm, diameter_mm, max_thickness_mm, mode, notch_angles_deg=notches_disc2
    )
    active_max1 = float(np.max(disc1_mm)) if np.max(disc1_mm) > 0.2 else max_thickness_mm
    active_max2 = float(np.max(disc2_mm)) if np.max(disc2_mm) > 0.2 else max_thickness_mm

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"disc_1_{mode}.stl", stl1_bytes)
        zf.writestr(f"disc_2_{mode}.stl", stl2_bytes)

        ams_section = ""
        if color_mode == "multicolor" and slot1_map is not None and slot2_map is not None and palette_info:
            target_res = 260
            z1_up, s1_up = _upsample_disc_maps(disc1_mm, slot1_map, mode=mode, target_res=target_res)
            z2_up, s2_up = _upsample_disc_maps(disc2_mm, slot2_map, mode=mode, target_res=target_res)
            n = z1_up.shape[0]
            raw_circle = create_circular_mask(n, margin_ratio=0.0)
            active_circle = create_circular_mask(n, margin_ratio=0.055)

            outer1 = apply_rim_notches(raw_circle, notches_disc1, notch_radius_ratio=0.038)
            outer2 = apply_rim_notches(raw_circle, notches_disc2, notch_radius_ratio=0.038)
            rim1 = outer1 & (~active_circle)
            rim2 = outer2 & (~active_circle)

            z1_up[rim1] = active_max1
            s1_up[rim1] = 1
            z1_up[outer1] = np.maximum(0.24, z1_up[outer1])
            z1_up[~outer1] = 0.0
            s1_up[~outer1] = 0

            z2_up[rim2] = active_max2
            s2_up[rim2] = 1
            z2_up[outer2] = np.maximum(0.24, z2_up[outer2])
            z2_up[~outer2] = 0.0
            s2_up[~outer2] = 0

            vx_top, vy_top, vx_bot, vy_bot = _build_smooth_vertex_coords(n, diameter_mm, raw_circle, chamfer_mm=0.6)
            pre1 = (z1_up, s1_up, vx_top, vy_top, vx_bot, vy_bot, outer1, rim1)
            pre2 = (z2_up, s2_up, vx_top, vy_top, vx_bot, vy_bot, outer2, rim2)

            ams_lines = ["\nBambu Lab AMS Multi-Color Filament Slot Mapping:", "-" * 50]
            for p_item in palette_info:
                s_idx = int(p_item["slot"])
                s_hex = p_item["hex"]
                s_name = p_item["name"].replace(" ", "_")
                ams_lines.append(f"  AMS Slot {s_idx}: {p_item['name']} ({s_hex})")

                d1_slot_stl = generate_slot_stl(
                    disc1_mm, slot1_map, s_idx, diameter_mm, max_thickness_mm, mode=mode, pre_upsampled=pre1
                )
                d2_slot_stl = generate_slot_stl(
                    disc2_mm, slot2_map, s_idx, diameter_mm, max_thickness_mm, mode=mode, pre_upsampled=pre2
                )
                zf.writestr(
                    f"bambu_ams_disc_1/disc_1_slot_{s_idx}_{s_name}_{s_hex.lstrip('#')}.stl",
                    d1_slot_stl,
                )
                zf.writestr(
                    f"bambu_ams_disc_2/disc_2_slot_{s_idx}_{s_name}_{s_hex.lstrip('#')}.stl",
                    d2_slot_stl,
                )
            ams_lines.append("\nHow to Import into Bambu Studio:")
            ams_lines.append("  1. Select all files inside 'bambu_ams_disc_1/' and drag them into Bambu Studio.")
            ams_lines.append("  2. Click 'Yes' when asked: 'Load these files as a single object with multiple parts?'")
            ams_lines.append("  3. Assign each part (slot_1 .. slot_K) to its matching AMS Filament Slot above.")
            ams_lines.append("  4. Repeat for 'bambu_ams_disc_2/'.")
            ams_lines.append("  Note: Slot 1 (Jade White) forms the continuous bottom skin across the whole plate,")
            ams_lines.append("        so Layer 1 prints in 100% White with ZERO filament swaps on the bed!")
            ams_section = "\n".join(ams_lines) + "\n"

        readme_content = f"""Rotational Dual-Image 3D Discs
==============================
Mode:            {mode.upper()} ({color_mode.upper()})
Diameter:        {diameter_mm:.1f} mm
Max Thickness:   {max_thickness_mm:.2f} mm
Reveal Angle:    0 deg (Image 1)  -->  {angle_deg:+.0f} deg (Image 2)
{ams_section}
Recommended 3D Printer Slicer Settings:
---------------------------------------
- Layer Height:  0.12 mm or 0.16 mm
- Infill:        100% Solid (Rectilinear or Concentric, do NOT use sparse infill)
- Perimeters:    2 walls
- Print Bed:     Print flat on the build plate (Z = 0 down)

How to Use:
-----------
1. Stack Disc 1 directly in front of Disc 2 and hold them up to a light source or window.
2. At 0 degrees alignment, Image 1 appears.
3. Rotate Disc 1 by {angle_deg:+.0f} degrees to reveal Image 2 right-side up!
"""
        zf.writestr("PRINT_INSTRUCTIONS.txt", readme_content)

    return zip_buf.getvalue()
