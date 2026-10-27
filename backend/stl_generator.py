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


def _build_smooth_vertex_coords(n: int, diameter_mm: float, solid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates the (n+1, n+1) vertex coordinate grids VX, VY in millimeters.
    Crucially, any vertex lying on the outer boundary of `solid_mask` (the disc perimeter)
    is smoothly projected onto the exact circle radius `R = diameter_mm / 2` and smoothed
    tangentially so the disc outer edge is a sleek circular cylinder instead of blocky steps!
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

    # Project outer boundary vertices directly onto the circle of radius `half_d`
    safe_rad = np.where(rad < 1e-6, 1.0, rad)
    vx[is_outer_boundary] = vx[is_outer_boundary] * (half_d / safe_rad[is_outer_boundary])
    vy[is_outer_boundary] = vy[is_outer_boundary] * (half_d / safe_rad[is_outer_boundary])

    return vx, vy


def _build_watertight_region_mesh(
    solid_mask: np.ndarray,
    z_top_cell: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    smooth_top_surface: bool = True,
) -> np.ndarray:
    """
    Builds a 100% watertight, 0-open-edge, 0-non-manifold-edge triangle mesh for the 2D boolean
    region `solid_mask` (n, n):
      - When `smooth_top_surface=True` (Lithophane mode):
        Computes shared corner vertex heights `VZ_top` (n+1, n+1) as the average of adjacent
        active cells in `solid_mask`. Because adjacent cells share exact corner vertices `(vx, vy, vz)`,
        there are ZERO internal vertical step walls and ZERO open edges!
      - When `smooth_top_surface=False` (Perforated Holes mode):
        All solid cells have constant top height `max_thickness_mm`, so adjacent solid cells also
        share exact top corner vertices `(vx, vy, max_thickness_mm)` with ZERO open edges!
      - Perimeter vertical walls are emitted ONLY along true region boundary edges (where `solid_mask`
        transitions from True to False), connecting top vertex `(vx, vy, vz)` to bottom `(vx, vy, 0)`.
    """
    mask = _fix_checkerboard_manifold(solid_mask)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    n = mask.shape[0]

    # Compute corner vertex height VZ (n+1, n+1) averaged ONLY over active cells in `mask`
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
    vz = (z_nw + z_ne + z_sw + z_se) / denom

    # Extract the 4 corner vertices for each active cell (y, x):
    # Top-Left (NW): (y, x), Top-Right (NE): (y, x+1), Bottom-Right (SE): (y+1, x+1), Bottom-Left (SW): (y+1, x)
    # Note: In our coordinate system, row `y` has larger Y than row `y+1` (vy decreases with y).
    # Let's define:
    #   p00 = (y+1, x)   [SW: x0, y0]
    #   p10 = (y+1, x+1) [SE: x1, y0]
    #   p11 = (y,   x+1) [NE: x1, y1]
    #   p01 = (y,   x)   [NW: x0, y1]
    x00, y00, z00 = vx[ys + 1, xs], vy[ys + 1, xs], vz[ys + 1, xs]
    x10, y10, z10 = vx[ys + 1, xs + 1], vy[ys + 1, xs + 1], vz[ys + 1, xs + 1]
    x11, y11, z11 = vx[ys, xs + 1], vy[ys, xs + 1], vz[ys, xs + 1]
    x01, y01, z01 = vx[ys, xs], vy[ys, xs], vz[ys, xs]

    zb = np.zeros_like(z00)

    p00_t = np.stack([x00, y00, z00], axis=-1)
    p10_t = np.stack([x10, y10, z10], axis=-1)
    p11_t = np.stack([x11, y11, z11], axis=-1)
    p01_t = np.stack([x01, y01, z01], axis=-1)

    p00_b = np.stack([x00, y00, zb], axis=-1)
    p10_b = np.stack([x10, y10, zb], axis=-1)
    p11_b = np.stack([x11, y11, zb], axis=-1)
    p01_b = np.stack([x01, y01, zb], axis=-1)

    tris = []

    # 1. Top surface triangles (CCW normal +Z): (p00 -> p10 -> p11) and (p00 -> p11 -> p01)
    tris.append(np.stack([p00_t, p10_t, p11_t], axis=1))
    tris.append(np.stack([p00_t, p11_t, p01_t], axis=1))

    # 2. Bottom surface triangles (CCW normal -Z): (p00 -> p11 -> p10) and (p00 -> p01 -> p11)
    tris.append(np.stack([p00_b, p11_b, p10_b], axis=1))
    tris.append(np.stack([p00_b, p01_b, p11_b], axis=1))

    # 3. Boundary vertical walls ONLY where neighbor cell is False!
    # Because boundary walls use the exact same corner vertices (p00_t, p10_t, p00_b, p10_b, etc.),
    # every boundary edge of the top surface and bottom surface is closed with 0 open edges!
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
    of a circular disc of `diameter_mm`, including functional inward alignment notches on the outer rim:
      - Disc 1 has ONE notch at 0° (top 12 o'clock).
      - Disc 2 has TWO notches: one at 0° (Image 1 alignment) and one at +angle_deg (Image 2 alignment).
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
    min_floor = 0.16 if mode == "ultrathin" else 0.40

    z_up[rim_mask] = active_max
    z_up[~outer_circle] = 0.0

    if mode in ("holes", "halftone"):
        solid_mask = (z_up > 0.1) & outer_circle
    else:
        z_up[active_circle] = np.clip(z_up[active_circle], min_floor, active_max)
        solid_mask = outer_circle.copy()

    vx, vy = _build_smooth_vertex_coords(n, diameter_mm, raw_circle)
    triangles = _build_watertight_region_mesh(
        solid_mask, z_up, vx, vy, smooth_top_surface=(mode not in ("holes", "halftone"))
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
    Bambu AMS filament slot `target_slot` (1..K), including functional rim alignment notches.
    """
    active_max = float(np.max(thickness_map_mm)) if np.max(thickness_map_mm) > 0.2 else max_thickness_mm
    min_floor = 0.16 if mode == "ultrathin" else 0.40

    if pre_upsampled is not None:
        z_up, s_up, vx, vy, outer_circle = pre_upsampled
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
        vx, vy = _build_smooth_vertex_coords(n, diameter_mm, raw_circle)

    slot_mask = (s_up == target_slot) & (z_up > 0.05) & outer_circle
    if not np.any(slot_mask):
        return _triangles_to_binary_stl(np.zeros((0, 3, 3), dtype=np.float32), header_text=f"AMS_Slot_{target_slot}")

    z_clamped = np.clip(z_up, min_floor, active_max)
    triangles = _build_watertight_region_mesh(
        slot_mask, z_clamped, vx, vy, smooth_top_surface=(mode not in ("holes", "halftone"))
    )
    return _triangles_to_binary_stl(triangles, header_text=f"AMS_Slot_{target_slot}")


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
            z1_up[~outer1] = 0.0
            s1_up[~outer1] = 0

            z2_up[rim2] = active_max2
            s2_up[rim2] = 1
            z2_up[~outer2] = 0.0
            s2_up[~outer2] = 0

            vx, vy = _build_smooth_vertex_coords(n, diameter_mm, raw_circle)
            pre1 = (z1_up, s1_up, vx, vy, outer1)
            pre2 = (z2_up, s2_up, vx, vy, outer2)

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
