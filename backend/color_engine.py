"""
Bambu Lab AMS Multi-Color Engine & Optimal Filament Palette Calculator.

Implements:
  1. Configurable K-Filament Palette (2 <= K <= 8 slots):
     - Standard CMYW Preset (White, Cyan, Magenta, Yellow + additional primaries if K > 4)
     - Auto-Optimize Colors for K Slots: K-Means / Spectral Decomposition in Optical Density Space
       to find the K filament colors that best reconstruct Image 1 and Image 2!
     - Custom User-Selected Hex Colors for any of the K slots.
  2. Multi-Color Rotational Solver (Lithophane & Holes modes):
     - Solves coupled Beer-Lambert spectral equations in RGB density space across Disc 1 and Disc 2,
       then projects each pixel's spectral density onto the active K-filament palette to determine
       both the physical thickness heightmap AND the discrete AMS filament slot assignment (1..K).
  3. Palette-Aware Chromatic Clarity Score & Angle Optimizer:
     - Evaluates 0-100% Clarity taking into account both rotational cross-talk AND how accurately
       the active K-color filament palette can reproduce the colors of both images at angle theta.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from backend.preprocessor import (
    create_circular_mask,
    rotate_color_array,
    rotate_image_array,
    apply_photo_anti_diffusion_boost,
)
from backend.engines import validate_angle


# Standard named filament reference library for human-friendly labels
FILAMENT_NAME_LIBRARY = [
    ("#FFFFFF", "Jade White"),
    ("#00AEEF", "Cyan"),
    ("#EC008C", "Magenta"),
    ("#FFF200", "Yellow"),
    ("#E61919", "Scarlet Red"),
    ("#1A56DB", "Cobalt Blue"),
    ("#10B981", "Emerald Green"),
    ("#F97316", "Tangerine Orange"),
    ("#7E22CE", "Royal Purple"),
    ("#14B8A6", "Teal"),
    ("#F59E0B", "Warm Gold"),
    ("#1E293B", "Charcoal Black"),
]


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    """Converts '#RRGGBB' to float32 array [R, G, B] in [0.0, 1.0]."""
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)


def rgb01_to_hex(rgb: np.ndarray) -> str:
    """Converts float32 [R, G, B] in [0, 1] to '#RRGGBB'."""
    c = np.clip(np.round(rgb * 255.0), 0, 255).astype(int)
    return f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def closest_filament_name(hex_str: str) -> str:
    """Finds a descriptive filament name for a given hex color."""
    rgb = hex_to_rgb01(hex_str)
    best_name = "Custom PLA"
    best_dist = 1e9
    for ref_hex, name in FILAMENT_NAME_LIBRARY:
        ref_rgb = hex_to_rgb01(ref_hex)
        dist = float(np.linalg.norm(rgb - ref_rgb))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def get_default_cmyw_palette(num_colors: int = 4) -> List[Dict[str, Any]]:
    """
    Returns the default K-slot Bambu AMS palette (2 <= K <= 8).
    Slot 1 is always Jade White (#FFFFFF) diffuser base, followed by Cyan, Magenta, Yellow, etc.
    """
    k = int(np.clip(num_colors, 2, 8))
    default_order = [
        ("#FFFFFF", "Slot 1: Jade White (Base)"),
        ("#00AEEF", "Slot 2: Cyan"),
        ("#EC008C", "Slot 3: Magenta"),
        ("#FFF200", "Slot 4: Yellow"),
        ("#E61919", "Slot 5: Scarlet Red"),
        ("#1A56DB", "Slot 6: Cobalt Blue"),
        ("#10B981", "Slot 7: Emerald Green"),
        ("#1E293B", "Slot 8: Charcoal Black"),
    ]
    palette = []
    for idx in range(k):
        hx, label = default_order[idx]
        palette.append({
            "slot": idx + 1,
            "hex": hx,
            "name": closest_filament_name(hx),
            "label": label,
        })
    return palette


def optimize_palette_for_images(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    num_colors: int = 4,
) -> List[Dict[str, Any]]:
    """
    Calculates the optimal K filament colors (for 2 <= K <= 8) tailored to the chromatic
    distribution of `img1_rgb` and `img2_rgb`.
    - Slot 1 is kept as Translucent/Jade White (#FFFFFF) so bright highlights can pass light cleanly.
    - Slots 2..K are found by clustering the active non-white pixels of both images in
      subtractive optical density space and boosting saturation so thin filament layers can
      achieve strong chromatic filtering!
    """
    k = int(np.clip(num_colors, 2, 8))
    mask = create_circular_mask(img1_rgb.shape[0], margin_ratio=0.06)

    # Gather all interior RGB pixels from both images
    p1 = img1_rgb[mask].reshape(-1, 3)
    p2 = img2_rgb[mask].reshape(-1, 3)
    all_pixels = np.concatenate([p1, p2], axis=0)

    # Convert RGB [0, 1] into subtractive absorption vector A = 1.0 - RGB
    absorption = 1.0 - all_pixels
    # Filter to pixels that actually have noticeable color/darkness (norm > 0.12)
    norms = np.linalg.norm(absorption, axis=1)
    colored = absorption[norms > 0.12]

    if len(colored) < 20:
        return get_default_cmyw_palette(k)

    num_active = k - 1  # Slot 1 is White (#FFFFFF)
    # Normalize absorption direction vectors so clustering captures distinct HUES
    unit_abs = colored / (np.linalg.norm(colored, axis=1, keepdims=True) + 1e-6)

    # Deterministic K-means++ initialization in spectral absorption direction space
    rng = np.random.default_rng(seed=101)
    centers = np.zeros((num_active, 3), dtype=np.float32)

    # Pick first center as the pixel with highest saturation/absorption
    sat = np.max(colored, axis=1) - np.min(colored, axis=1)
    first_idx = int(np.argmax(sat + 0.3 * norms[norms > 0.12]))
    centers[0] = unit_abs[first_idx]

    for c_idx in range(1, num_active):
        # Compute squared distance to nearest existing center
        dists = np.min(
            [np.sum((unit_abs - centers[j]) ** 2, axis=1) for j in range(c_idx)],
            axis=0,
        )
        # Weight by saturation so vivid primary colors are favored as filament bases
        weights = dists * (0.2 + sat)
        best_next = int(np.argmax(weights))
        centers[c_idx] = unit_abs[best_next]

    # Run 18 iterations of Lloyd's K-means weighted by pixel absorption magnitude
    sample_weights = norms[norms > 0.12] + 0.5 * sat
    for _ in range(18):
        # Assign each pixel to closest center
        d_mat = np.stack(
            [np.sum((unit_abs - centers[j]) ** 2, axis=1) for j in range(num_active)],
            axis=1,
        )
        labels = np.argmin(d_mat, axis=1)
        for j in range(num_active):
            sel = labels == j
            if np.any(sel):
                w_sel = sample_weights[sel, None]
                mean_vec = np.sum(unit_abs[sel] * w_sel, axis=0) / (np.sum(w_sel) + 1e-6)
                norm_v = np.linalg.norm(mean_vec)
                if norm_v > 1e-6:
                    centers[j] = mean_vec / norm_v

    # Convert each unit absorption direction back into a rich, printable translucent filament color!
    # A pure filament has high saturation: scale max absorption component to ~0.92
    palette = [
        {
            "slot": 1,
            "hex": "#FFFFFF",
            "name": "Jade White",
            "label": "Slot 1: Jade White (Base)",
        }
    ]

    for j in range(num_active):
        abs_dir = centers[j]
        max_comp = float(np.max(abs_dir)) + 1e-6
        scaled_abs = np.clip((abs_dir / max_comp) * 0.90, 0.0, 0.94)
        # Filament RGB transmittance at full thickness
        fil_rgb = np.clip(1.0 - scaled_abs, 0.04, 1.0)
        hx = rgb01_to_hex(fil_rgb)
        fname = closest_filament_name(hx)
        palette.append({
            "slot": j + 2,
            "hex": hx,
            "name": fname,
            "label": f"Slot {j + 2}: {fname} ({hx})",
        })

    return palette


def _project_density_to_palette(
    disc_density_rgb: np.ndarray,
    palette_hex_list: List[str],
    t_min: float = 0.45,
    t_max: float = 2.0,
    mode: str = "lithophane",
    is_back_disc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a desired per-pixel spectral optical density `disc_density_rgb` (H, W, 3) in [0, 0.58]
    and a list of K filament hex colors:
      1. Computes each filament's spectral absorption vector A_k = 1.0 - RGB_k.
      2. For each pixel (y, x), assigns the filament slot k in {1..K} whose chromatic absorption
         profile best produces `disc_density_rgb[y, x]`, and computes the optimal physical
         thickness `thickness_mm[y, x]` in [t_min, t_max] (or 0 vs t_max in Holes/Halftone mode).
      3. Applies Asymmetric Back-Disc Contrast Boost (`is_back_disc=True`) so Disc 2 punches
         through Disc 1's scattering layer, and keeps the outer rim flush for zero air gap!
    """
    h, w, _ = disc_density_rgb.shape
    k_count = len(palette_hex_list)
    outer_circle = create_circular_mask(h, margin_ratio=0.0)
    active_circle = create_circular_mask(h, margin_ratio=0.05)
    rim_mask = outer_circle & (~active_circle)

    # Build filament absorption matrix (K, 3)
    # Note: Even pure white filament (#FFFFFF) has a small baseline neutral scattering density
    # (0.22 across R, G, B) so thicker white plastic still attenuates brightness!
    fil_rgb = np.stack([hex_to_rgb01(hx) for hx in palette_hex_list], axis=0)  # (K, 3)
    # Effective absorption per normalized thickness unit u in [0, 0.58]:
    # White (#FFFFFF) absorbs [0.35, 0.35, 0.35] per unit thickness; colored filaments absorb
    # strongly in their complementary channels up to 1.15 per unit thickness.
    fil_abs = 0.32 + 0.95 * (1.0 - fil_rgb)  # shape (K, 3)

    flat_d = disc_density_rgb.reshape(-1, 3)  # (N, 3)
    num_pix = flat_d.shape[0]

    best_slot = np.ones(num_pix, dtype=np.int32)  # 1-indexed slot (1..K)
    best_u = np.zeros(num_pix, dtype=np.float32)
    best_err = np.full(num_pix, 1e9, dtype=np.float32)

    # Spatially smooth the chromatic component of disc_density_rgb slightly before slot assignment
    # so AMS filament color zones form clean, coherent regions without single-pixel confetti noise!
    smooth_d = np.stack([gaussian_filter(disc_density_rgb[:, :, c], sigma=0.85) for c in range(3)], axis=-1)
    flat_smooth_d = smooth_d.reshape(-1, 3)

    # Evaluate each filament slot k = 0 .. K-1
    for k_idx in range(k_count):
        ak = fil_abs[k_idx]  # (3,)
        ak_norm_sq = float(np.dot(ak, ak)) + 1e-6
        # Use smoothed density for slot selection, and exact density for physical height u_opt
        u_smooth = np.clip(np.dot(flat_smooth_d, ak) / ak_norm_sq, 0.0, 0.58)
        rec_d = u_smooth[:, None] * ak[None, :]
        diff = flat_smooth_d - rec_d
        mean_diff = np.mean(diff, axis=1, keepdims=True)
        chroma_diff = diff - mean_diff
        err = np.sum(mean_diff ** 2, axis=1) + 2.5 * np.sum(chroma_diff ** 2, axis=1)

        better = err < best_err
        best_err[better] = err[better]
        best_slot[better] = k_idx + 1

    # Clean isolated 1-pixel speckles in the 2D slot map via 3x3 median filter
    slot_map = median_filter(best_slot.reshape(h, w), size=3)

    # Compute exact optimal thickness u_map for the final assigned slot at each pixel
    chosen_abs = fil_abs[slot_map - 1]  # (H, W, 3)
    ak_norm_sq_map = np.sum(chosen_abs ** 2, axis=-1) + 1e-6
    u_map = np.clip(np.sum(disc_density_rgb * chosen_abs, axis=-1) / ak_norm_sq_map, 0.0, 0.58)

    if mode == "holes":
        # In Holes mode, pixels with low density (u < 0.22) become open through-holes (0.0 mm),
        # while higher-density pixels become solid colored blocks of thickness t_max!
        is_hole = (u_map < 0.21) & active_circle
        thickness_mm = np.where(is_hole, 0.0, t_max).astype(np.float32)
        u_eff = np.where(is_hole, 0.0, 0.42).astype(np.float32)
        slot_map[is_hole] = 0  # 0 indicates open hole
        rim_thickness = t_max
    elif mode == "halftone":
        # Halftone Photo Mode: variable-aperture micro-screen with connected grid backbone
        yy, xx = np.mgrid[:h, :w]
        phase = 1.0 if is_back_disc else 0.0
        wx = np.cos(2.0 * math.pi * (xx - w / 2.0 + phase) / 4.0)
        wy = np.cos(2.0 * math.pi * (yy - h / 2.0 + phase) / 4.0)
        screen = 0.50 + 0.38 * (0.5 * (wx + wy))
        strut_grid = ((xx % 4 == 0) | (yy % 4 == 0)) & active_circle
        op = np.clip((u_map / 0.54) * (1.12 if is_back_disc else 1.0), 0.0, 1.0)
        is_solid = ((op >= screen) | strut_grid) & active_circle
        is_hole = (~is_solid) & active_circle
        thickness_mm = np.where(is_solid, np.clip(0.45 + op * (t_max - 0.45), 0.45, t_max), 0.0).astype(np.float32)
        u_eff = np.where(is_hole, 0.0, u_map * (1.15 if is_back_disc else 1.0)).astype(np.float32)
        slot_map[is_hole] = 0
        rim_thickness = max(1.2, t_max)
    else:
        # Asymmetric Back-to-Front Optical Compensation:
        # Back disc gets steeper physical relief so its contrast punches through Front disc's PLA scattering;
        # Front disc stays thinner to minimize blurring the rays coming from Back disc!
        if mode == "ultrathin":
            eff_tmax = max(t_max, 1.35) if is_back_disc else max(t_min + 0.28, t_max * 0.68)
        else:
            eff_tmax = t_max if is_back_disc else max(t_min + 0.45, t_max * 0.78)
        scale = (eff_tmax - t_min) / 0.58
        thickness_mm = np.clip(t_min + u_map * scale, t_min, eff_tmax).astype(np.float32)
        u_eff = u_map * (1.15 if is_back_disc else 0.92)
        # Keep outer rim flush with eff_tmax so the two discs' active surfaces sit in direct contact!
        rim_thickness = eff_tmax

    thickness_mm[rim_mask] = rim_thickness
    slot_map[rim_mask] = 1
    thickness_mm[~outer_circle] = 0.0
    slot_map[~outer_circle] = 0

    quantized_density_rgb = u_eff[:, :, None] * chosen_abs
    quantized_density_rgb[~active_circle] = 0.85
    quantized_density_rgb[~outer_circle] = 1.0

    # Single-disc backlit RGB transmission in [0, 1] for the 2D viewer
    disc_trans_rgb = np.clip(1.0 - 0.82 * quantized_density_rgb, 0.06, 1.0).astype(np.float32)
    disc_trans_rgb[~outer_circle] = 0.0

    return thickness_mm, slot_map, disc_trans_rgb


def solve_multicolor_pair(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    angle_deg: float,
    palette_hex_list: List[str],
    t_min: float = 0.45,
    t_max: float = 2.0,
    mode: str = "lithophane",
    num_iters: int = 28,
) -> Dict[str, Any]:
    """
    Solves for two multi-color Bambu AMS discs (Disc 1 & Disc 2) using the provided
    K-color filament palette `palette_hex_list`.

    When `mode == 'ultrathin'`:
      - Pre-sharpens both photos with local contrast enhancement (`apply_photo_anti_diffusion_boost`)
        and solves onto an ultra-thin [0.16 mm, 0.80 mm] membrane with a rigid 1.60 mm outer ring!
    """
    size = img1_rgb.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.05)
    outer_circle = create_circular_mask(size, margin_ratio=0.0)
    is_ultra = mode == "ultrathin"

    work1 = apply_photo_anti_diffusion_boost(img1_rgb, mask) if is_ultra else img1_rgb
    work2 = apply_photo_anti_diffusion_boost(img2_rgb, mask) if is_ultra else img2_rgb

    gamma = 0.78 if is_ultra else 0.85
    d1_norm = np.clip(1.0 - (work1 ** gamma), 0.0, 1.0)
    d2_norm = np.clip(1.0 - (work2 ** gamma), 0.0, 1.0)

    d_floor = 0.02 if is_ultra else 0.05
    d_span = 0.95 if is_ultra else 0.88
    target_d1 = d_floor + d_span * d1_norm
    target_d2 = d_floor + d_span * d2_norm

    # Pre-rotate target_d2 by +angle_deg
    target_d2_inv_rot = rotate_color_array(target_d2, angle_deg, fill_val=0.0)

    u1 = np.full((size, size, 3), 0.22, dtype=np.float32)
    u2 = np.full((size, size, 3), 0.22, dtype=np.float32)

    for _ in range(num_iters):
        u1_rot = rotate_color_array(u1, -angle_deg, fill_val=0.0)
        est_u2_1 = target_d1 - u1
        est_u2_2 = target_d2 - u1_rot
        u2 = np.clip(0.5 * (est_u2_1 + est_u2_2), 0.0, 0.56)
        u2[~mask] = 0.0

        u2_inv_rot = rotate_color_array(u2, angle_deg, fill_val=0.0)
        est_u1_1 = target_d1 - u2
        est_u1_2 = target_d2_inv_rot - u2_inv_rot
        u1 = np.clip(0.5 * (est_u1_1 + est_u1_2), 0.0, 0.56)
        u1[~mask] = 0.0

    # Residual contrast boost
    u1_rot = rotate_color_array(u1, -angle_deg, fill_val=0.0)
    res1 = target_d1 - (u1 + u2)
    res2 = target_d2 - (u1_rot + u2)
    u2 = np.clip(u2 + 0.35 * (res1 + res2), 0.0, 0.58)

    u2_inv_rot = rotate_color_array(u2, angle_deg, fill_val=0.0)
    res1_b = target_d1 - (u1 + u2)
    res2_b = target_d2_inv_rot - (u1 + u2_inv_rot)
    u1 = np.clip(u1 + 0.35 * (res1_b + res2_b), 0.0, 0.58)

    u1[~mask] = 0.0
    u2[~mask] = 0.0

    # Project continuous RGB densities onto the K chosen AMS filament colors
    disc1_mm, slot1_map, d1_trans_rgb = _project_density_to_palette(
        u1, palette_hex_list, t_min=t_min, t_max=t_max, mode=mode, is_back_disc=False
    )
    disc2_mm, slot2_map, d2_trans_rgb = _project_density_to_palette(
        u2, palette_hex_list, t_min=t_min, t_max=t_max, mode=mode, is_back_disc=True
    )

    # Simulated stacked RGB transmission at 0° and θ°
    d1_rot_rgb = rotate_color_array(d1_trans_rgb, -angle_deg, fill_val=0.0)
    reveal_0_rgb = np.clip((d1_trans_rgb * d2_trans_rgb) * 1.25, 0.0, 1.0)
    reveal_theta_rgb = np.clip((d1_rot_rgb * d2_trans_rgb) * 1.25, 0.0, 1.0)
    reveal_0_rgb[~outer_circle] = 0.0
    reveal_theta_rgb[~outer_circle] = 0.0

    return {
        "disc1_mm": disc1_mm,
        "disc2_mm": disc2_mm,
        "slot1_map": slot1_map,
        "slot2_map": slot2_map,
        "d1_trans_rgb": d1_trans_rgb,
        "d2_trans_rgb": d2_trans_rgb,
        "reveal_0_rgb": reveal_0_rgb,
        "reveal_theta_rgb": reveal_theta_rgb,
    }


def compute_color_angle_clarity_score(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    angle_deg: float,
    palette_hex_list: List[str],
    mode: str = "lithophane",
) -> float:
    """
    Computes a 0-100% Color & Palette-Aware Clarity Score for candidate rotation angle `angle_deg`
    with respect to the active K-color filament palette `palette_hex_list`.
    """
    norm_angle = validate_angle(angle_deg)
    abs_angle = abs(norm_angle)
    size = img1_rgb.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.06)

    # Fast 8-iteration multi-color solve with palette projection
    res = solve_multicolor_pair(
        img1_rgb,
        img2_rgb,
        angle_deg=norm_angle,
        palette_hex_list=palette_hex_list,
        mode=mode,
        num_iters=8,
    )

    rec1 = res["reveal_0_rgb"][mask]       # (N, 3)
    rec2 = res["reveal_theta_rgb"][mask]   # (N, 3)
    tgt1 = img1_rgb[mask]                  # (N, 3)
    tgt2 = img2_rgb[mask]                  # (N, 3)

    # 1. Per-channel RGB correlation (measures both luminance and chromatic fidelity)
    corrs = []
    for c in range(3):
        if np.std(rec1[:, c]) > 1e-4 and np.std(tgt1[:, c]) > 1e-4:
            corrs.append(float(np.corrcoef(tgt1[:, c], rec1[:, c])[0, 1]))
        if np.std(rec2[:, c]) > 1e-4 and np.std(tgt2[:, c]) > 1e-4:
            corrs.append(float(np.corrcoef(tgt2[:, c], rec2[:, c])[0, 1]))
    rgb_fidelity = max(0.0, float(np.mean(corrs))) if corrs else 0.5

    # 2. Chromatic Delta-E (color gamut accuracy of the chosen K-color palette at this angle)
    # Normalize mean brightness so we measure true color/contrast error
    rmse_rgb = float(np.sqrt(0.5 * (np.mean((tgt1 - rec1) ** 2) + np.mean((tgt2 - rec2) ** 2))))
    gamut_accuracy = max(0.0, 1.0 - 1.35 * rmse_rgb)

    # 3. Rotational self-ghosting & physical arc displacement
    eff_angle = min(abs_angle, 360.0 - abs_angle)
    arc_bonus = math.sin(math.radians(eff_angle * 0.75))
    dist_to_90 = min(abs(eff_angle - 90.0), abs(eff_angle - 180.0))
    grid_sharpness = math.exp(-((dist_to_90 / 38.0) ** 2))

    raw_score = (
        44.0 * rgb_fidelity
        + 26.0 * gamut_accuracy
        + 16.0 * arc_bonus
        + 14.0 * grid_sharpness
    )
    return float(np.clip(round(raw_score, 1), 12.0, 99.0))


def compare_color_angles(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    palette_hex_list: List[str],
    custom_angle: float = 90.0,
    mode: str = "lithophane",
) -> Dict[str, Any]:
    """
    Evaluates Color & Palette-Aware Clarity Scores across candidate angles (|theta| >= 45°),
    returning `default_90`, `auto_best`, `custom`, and `worst`.
    """
    eval_size = 48
    img1_small = np.asarray(
        Image.fromarray(np.clip(img1_rgb * 255.0, 0, 255).astype(np.uint8)).resize(
            (eval_size, eval_size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    img2_small = np.asarray(
        Image.fromarray(np.clip(img2_rgb * 255.0, 0, 255).astype(np.uint8)).resize(
            (eval_size, eval_size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    valid_custom = validate_angle(custom_angle)
    candidates: List[float] = [
        45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0,
        -45.0, -60.0, -75.0, -90.0, -105.0, -120.0, -135.0, -150.0, -165.0,
    ]
    if valid_custom not in candidates:
        candidates.append(valid_custom)

    scores_map: Dict[float, float] = {}
    for ang in candidates:
        scores_map[ang] = compute_color_angle_clarity_score(
            img1_small, img2_small, ang, palette_hex_list=palette_hex_list, mode=mode
        )

    best_ang = max(scores_map.keys(), key=lambda a: scores_map[a])
    worst_ang = min(scores_map.keys(), key=lambda a: scores_map[a])

    return {
        "default_90": {
            "label": "90° (Default)",
            "angle": 90.0,
            "score": scores_map[90.0],
            "description": f"Orthogonal 90° turn with {len(palette_hex_list)}-color AMS palette.",
        },
        "auto_best": {
            "label": "Auto (Optimal)",
            "angle": float(best_ang),
            "score": scores_map[best_ang],
            "description": f"Best color separation & lowest chromatic cross-talk at {best_ang:+.0f}°.",
        },
        "custom": {
            "label": f"Custom ({valid_custom:+.0f}°)",
            "angle": float(valid_custom),
            "score": scores_map[valid_custom],
            "description": "Your currently selected rotation offset.",
        },
        "worst": {
            "label": f"Worst ({worst_ang:+.0f}°)",
            "angle": float(worst_ang),
            "score": scores_map[worst_ang],
            "description": "Highest chromatic conflict & subpixel aliasing.",
        },
    }
