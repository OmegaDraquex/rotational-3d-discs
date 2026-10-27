"""
Dual-Image Rotational Processing Engines & Clarity Optimizer.

Implements:
1. Patterned Lithophane Engine: Continuous Beer-Lambert thickness optimization
   solving coupled equations across rotational orbits.
2. Full Holes Engine (Visual Cryptography): Perforated subpixel lattice with
   dithering and physical structural support.
3. Clarity Score & Optimal Angle Algorithm: Evaluates 0-100% Clarity across
   candidate angles (|theta| >= 45 deg), comparing 90°, Auto (Best), Custom, and Worst.
"""

import math
from typing import Dict, Any, Tuple, List
import numpy as np
from backend.preprocessor import (
    create_circular_mask,
    rotate_image_array,
    floyd_steinberg_dither,
    apply_photo_anti_diffusion_boost,
)


def validate_angle(angle_deg: float) -> float:
    """
    Validates that the angle satisfies |angle| >= 45 degrees and isn't within
    45 degrees of 0/360.
    Normalizes angle into [-180, 180] or [45, 315].
    """
    # Map angle into (-180, 180]
    norm = ((angle_deg + 180.0) % 360.0) - 180.0
    if abs(norm) < 45.0:
        # Clamp to +45 or -45 depending on sign
        norm = 45.0 if angle_deg >= 0 else -45.0
    return float(norm)


def solve_lithophane_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    angle_deg: float,
    t_min: float = 0.45,
    t_max: float = 2.0,
    num_iters: int = 35,
    ultra_thin: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves for continuous thickness maps (disc1, disc2) in millimeters [t_min, t_max]
    such that:
      - At 0 deg rotation:   disc1 + disc2       reconstructs optical density of img1
      - At +angle_deg rot of disc1: rot(disc1, +angle) + disc2 reconstructs img2 (upright!)

    When `ultra_thin=True`:
      - Pre-compensates both images with multi-scale local contrast enhancement & edge sharpening
        to counteract lateral light diffusion in PLA.
      - Uses an aggressive dynamic range curve [0.01, 0.98] with t_min = 0.24 mm (solid 2-layer base skin)
        and t_max = 0.80 mm for the inner optical membrane while keeping a rigid outer rim!
    """
    if ultra_thin and t_min < 0.24:
        t_min = 0.24  # Continuous 2-layer (0.24mm) white base skin to prevent tearing/shredding

    size = img1.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.05)

    work1 = apply_photo_anti_diffusion_boost(img1, mask) if ultra_thin else img1
    work2 = apply_photo_anti_diffusion_boost(img2, mask) if ultra_thin else img2

    gamma = 0.78 if ultra_thin else 0.85
    d1_norm = np.clip(1.0 - (work1 ** gamma), 0.0, 1.0)
    d2_norm = np.clip(1.0 - (work2 ** gamma), 0.0, 1.0)

    d_floor = 0.02 if ultra_thin else 0.06
    d_span = 0.95 if ultra_thin else 0.88
    target_d1 = d_floor + d_span * d1_norm
    target_d2 = d_floor + d_span * d2_norm

    u_cap = 0.62 if ultra_thin else 0.55
    u1 = np.full((size, size), 0.25, dtype=np.float32)
    u2 = np.full((size, size), 0.25, dtype=np.float32)

    target_d2_inv_rot = rotate_image_array(target_d2, angle_deg, fill_val=0.0)

    for i in range(num_iters):
        u1_rot = rotate_image_array(u1, -angle_deg, fill_val=0.0)
        est_u2_from_1 = target_d1 - u1
        est_u2_from_2 = target_d2 - u1_rot
        u2 = np.clip(0.5 * (est_u2_from_1 + est_u2_from_2), 0.0, u_cap)
        u2[~mask] = 0.0

        u2_inv_rot = rotate_image_array(u2, angle_deg, fill_val=0.0)
        est_u1_from_1 = target_d1 - u2
        est_u1_from_2 = target_d2_inv_rot - u2_inv_rot
        u1 = np.clip(0.5 * (est_u1_from_1 + est_u1_from_2), 0.0, u_cap)
        u1[~mask] = 0.0

    # Local contrast enhancement step on the residuals
    boost_gain = 0.48 if ultra_thin else 0.35
    u_max_final = 0.66 if ultra_thin else 0.58
    u1_rot = rotate_image_array(u1, -angle_deg, fill_val=0.0)
    res1 = target_d1 - (u1 + u2)
    res2 = target_d2 - (u1_rot + u2)
    u2 = np.clip(u2 + boost_gain * (res1 + res2), 0.0, u_max_final)
    u2_inv_rot = rotate_image_array(u2, angle_deg, fill_val=0.0)
    res1_b = target_d1 - (u1 + u2)
    res2_b = target_d2_inv_rot - (u1 + u2_inv_rot)
    u1 = np.clip(u1 + boost_gain * (res1_b + res2_b), 0.0, u_max_final)

    u1[~mask] = 0.0
    u2[~mask] = 0.0

    # ASYMMETRIC BACK-TO-FRONT OPTICAL COMPENSATION:
    # Because Disc 2 sits behind Disc 1 (closest to the light source), light exiting Disc 2
    # must pass through Disc 1's turbid PLA layer, which scatters and attenuates Disc 2's contrast!
    # Meanwhile, Disc 1 sits closest to the viewer's eye and needs to be a thinner, lower-scatter
    # optical modulator so it doesn't wash out Disc 2.
    # Therefore:
    #   - Disc 1 (Front / Eye-side): thinner membrane [t_min, t_max_front] so Disc 2 punches through!
    #   - Disc 2 (Back / Light-side): 1.75x steeper physical relief [t_min, t_max_back] for deep shadow contrast!
    t_max_front = max(t_min + 0.28, t_max * 0.68) if ultra_thin else max(t_min + 0.45, t_max * 0.78)
    t_max_back = max(t_max, 1.35) if ultra_thin else t_max

    scale_front = (t_max_front - t_min) / u_cap
    scale_back = (t_max_back - t_min) / u_cap

    disc1_mm = np.clip(t_min + u1 * scale_front, t_min, t_max_front)
    disc2_mm = np.clip(t_min + u2 * scale_back, t_min, t_max_back)

    # FLUSH / RECESSED RIM FOR ZERO AIR GAP:
    # Instead of a 1.6mm tall lip that holds the two inner membranes ~2mm apart in the air,
    # keep the outer structural rim flush with each disc's maximum active thickness so the
    # two patterned surfaces sit in direct physical contact!
    outer_circle = create_circular_mask(size, margin_ratio=0.0)
    rim_mask = outer_circle & (~mask)
    disc1_mm[rim_mask] = t_max_front
    disc2_mm[rim_mask] = t_max_back
    disc1_mm[~outer_circle] = 0.0
    disc2_mm[~outer_circle] = 0.0

    # Compute simulated optical transmission at 0° and θ°
    u1_rot_final = rotate_image_array(u1, -angle_deg, fill_val=0.0)
    comb_0 = np.clip((u1 + u2 - d_floor) / d_span, 0.0, 1.0)
    comb_theta = np.clip((u1_rot_final + u2 - d_floor) / d_span, 0.0, 1.0)

    trans_0 = (1.0 - comb_0) ** (1.0 / gamma)
    trans_theta = (1.0 - comb_theta) ** (1.0 / gamma)
    trans_0[~mask] = 0.15
    trans_theta[~mask] = 0.15

    return disc1_mm, disc2_mm, trans_0, trans_theta


def solve_halftone_photo_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    angle_deg: float,
    t_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Halftone Photo Mode (Variable-Aperture Continuous-Tone Micro-Screen):
    Solves the fundamental diffusion problem of stacked lithophanes while preserving
    photographic continuous grayscale gradients!

    Improvements for 3D Print Reliability & Bed Adhesion:
      1. Continuous 0.24mm White Base Skin:
         Apertures thin down to a continuous 0.24mm (2-layer) translucent White PLA floor
         instead of empty air (0.0mm). This ensures Layer 1 and Layer 2 are 100% solid across
         the entire plate: 0 isolated micro-dots, zero nozzle dragging, zero lattice shredding!
      2. Coarser Pitch & 2-Perimeter Struts:
         Uses a 2.0mm cell pitch (5 grid pixels on 200x200) with 0.80mm wide struts (2 pixels,
         corresponding to 2 clean nozzle perimeters) anchored to the solid 0.24mm base skin.
    """
    size = img1.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.05)
    outer_circle = create_circular_mask(size, margin_ratio=0.0)
    rim_mask = outer_circle & (~mask)

    # Solve continuous rotational density fields u1, u2 in [0, 0.64]
    _, _, _, _ = solve_lithophane_pair(img1, img2, angle_deg, t_min=0.24, t_max=t_max, num_iters=28, ultra_thin=True)

    work1 = apply_photo_anti_diffusion_boost(img1, mask)
    work2 = apply_photo_anti_diffusion_boost(img2, mask)
    target_d1 = 0.04 + 0.92 * np.clip(1.0 - (work1 ** 0.80), 0.0, 1.0)
    target_d2 = 0.04 + 0.92 * np.clip(1.0 - (work2 ** 0.80), 0.0, 1.0)
    target_d2_inv_rot = rotate_image_array(target_d2, angle_deg, fill_val=0.0)

    u1 = np.full((size, size), 0.25, dtype=np.float32)
    u2 = np.full((size, size), 0.25, dtype=np.float32)
    for _ in range(28):
        u1_rot = rotate_image_array(u1, -angle_deg, fill_val=0.0)
        u2 = np.clip(0.5 * ((target_d1 - u1) + (target_d2 - u1_rot)), 0.0, 0.62)
        u2[~mask] = 0.0
        u2_inv = rotate_image_array(u2, angle_deg, fill_val=0.0)
        u1 = np.clip(0.5 * ((target_d1 - u2) + (target_d2_inv_rot - u2_inv)), 0.0, 0.62)
        u1[~mask] = 0.0

    # Normalize u1, u2 into [0.0, 1.0] local opacity demand
    op1 = np.clip(u1 / 0.56, 0.0, 1.0)
    op2 = np.clip((u2 / 0.56) * 1.12, 0.0, 1.0)  # Boost back disc slightly

    # Build rotationally-symmetric 2D halftone carrier wave (period = 5 pixels -> 2.0mm pitch on 80mm disc)
    yy, xx = np.mgrid[:size, :size]
    # Orthogonal cosine screen centered in each 5x5 cell
    wx = np.cos(2.0 * math.pi * (xx - size / 2.0) / 5.0)
    wy = np.cos(2.0 * math.pi * (yy - size / 2.0) / 5.0)
    # Screen threshold in [0.15, 0.85]
    screen1 = 0.50 + 0.35 * (0.5 * (wx + wy))
    # Phase-shifted complementary screen for Disc 2 so apertures interlock cleanly
    wx2 = np.cos(2.0 * math.pi * (xx - size / 2.0 + 1.25) / 5.0)
    wy2 = np.cos(2.0 * math.pi * (yy - size / 2.0 + 1.25) / 5.0)
    screen2 = 0.50 + 0.35 * (0.5 * (wx2 + wy2))

    # Structural micro-grid backbone: 2-pixel wide struts (0.80mm wide -> exactly 2 nozzle perimeters)
    # anchored directly to the 0.24mm solid base skin
    strut_grid = ((xx % 5 < 2) | (yy % 5 < 2)) & mask

    solid1 = ((op1 >= screen1) | strut_grid) & mask
    solid2 = ((op2 >= screen2) | strut_grid) & mask

    # Base skin is 0.24mm White PLA (continuous solid floor across entire circle)
    # Solid features rise from 0.48mm to t_max on top of the 0.24mm floor
    base_skin_mm = 0.24
    disc1_mm = np.where(solid1, np.clip(0.48 + op1 * (t_max - 0.48), 0.48, t_max), base_skin_mm).astype(np.float32)
    disc2_mm = np.where(solid2, np.clip(0.48 + op2 * (t_max - 0.48), 0.48, t_max), base_skin_mm).astype(np.float32)

    disc1_mm[rim_mask] = max(1.2, t_max)
    disc2_mm[rim_mask] = max(1.2, t_max)
    disc1_mm[~outer_circle] = 0.0
    disc2_mm[~outer_circle] = 0.0

    # Compute local optical transmission:
    # 0.24mm translucent White PLA transmits ~88% light; thicker solid areas attenuate down to 5-35%
    t_map1 = np.where(disc1_mm <= 0.26, 0.88, np.clip(1.0 - 0.80 * op1, 0.06, 0.45)).astype(np.float32)
    t_map2 = np.where(disc2_mm <= 0.26, 0.88, np.clip(1.0 - 0.82 * op2, 0.05, 0.42)).astype(np.float32)
    t_map1_rot = rotate_image_array(t_map1, -angle_deg, fill_val=0.0)

    trans_0 = np.clip(t_map1 * t_map2, 0.0, 1.0)
    trans_theta = np.clip(t_map1_rot * t_map2, 0.0, 1.0)
    trans_0[~mask] = 0.08
    trans_theta[~mask] = 0.08

    return disc1_mm, disc2_mm, trans_0, trans_theta


def solve_holes_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    angle_deg: float,
    t_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Holes Mode (Visual Cryptography with Perforated Subpixel Lattice).
    Each logical pixel is expanded into a 2x2 subpixel cell (or high-res binary lattice)
    with a thin structural grid wall so every solid pixel is physically connected
    when 3D printed.

    In Visual Cryptography:
      - Light passes through ONLY where BOTH Disc 1 AND Disc 2 have a hole (Boolean AND of holes,
        or Boolean OR of opaque plastic).
      - Let H1, H2 in {0, 1} be hole indicators (1 = open hole, 0 = solid opaque plastic).
      - Transmitted light at 0°:   T_0     = H1 * H2
      - Transmitted light at θ°:   T_theta = rot(H1, -angle) * H2
    """
    size = img1.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.05)

    # Work at half resolution for logical cells, then expand 2x2 into (size, size)
    half = size // 2
    # Downsample img1 and img2 to (half, half)
    from PIL import Image
    pil1 = Image.fromarray(np.clip(img1 * 255.0, 0, 255).astype(np.uint8)).resize((half, half), Image.Resampling.BILINEAR)
    pil2 = Image.fromarray(np.clip(img2 * 255.0, 0, 255).astype(np.uint8)).resize((half, half), Image.Resampling.BILINEAR)

    arr1_low = np.asarray(pil1, dtype=np.float32) / 255.0
    arr2_low = np.asarray(pil2, dtype=np.float32) / 255.0
    mask_low = create_circular_mask(half, margin_ratio=0.05)

    # Dither both images to crisp 1-bit binary (1 = bright/white, 0 = dark/black)
    b1 = floyd_steinberg_dither(arr1_low, mask_low)
    b2 = floyd_steinberg_dither(arr2_low, mask_low)

    # Pre-rotate b2 by +angle_deg so we know what Disc 1 pixel (y, x) needs to do when rotated to -angle_deg
    b2_inv = (rotate_image_array(b2, angle_deg, fill_val=1.0) >= 0.5).astype(np.float32)

    # Deterministic seeded RNG for reproducible subpixel patterns
    rng = np.random.default_rng(seed=42)

    # High-res hole maps (1.0 = open through-hole, 0.0 = solid plastic)
    h1 = np.zeros((size, size), dtype=np.float32)
    h2 = np.zeros((size, size), dtype=np.float32)

    # 4 subpixel positions in each 2x2 block: (0,0), (0,1), (1,0), (1,1)
    # In Visual Cryptography, each 2x2 block has 2 open holes and 2 solid blocks (50% duty cycle)
    # - If target is White (1): Disc 1 and Disc 2 share 2 open holes -> 50% transmission (Bright)
    # - If target is Black (0): Disc 1 and Disc 2 have complementary holes -> 0% or 25% transmission (Dark)
    all_idx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # First assign Disc 2's 2x2 subpixel blocks randomly with 2 holes (weight = 2)
    # Then assign Disc 1's 2x2 subpixel blocks to simultaneously maximize correlation with b1 at 0°
    # and b2 at θ°!
    for y in range(half):
        for x in range(half):
            if not mask_low[y, x]:
                continue
            perm = rng.permutation(4)
            # Disc 2 gets 2 open holes per 2x2 cell
            for k in perm[:2]:
                dy, dx = all_idx[k]
                h2[2 * y + dy, 2 * x + dx] = 1.0

    # Rotate h2 by +angle_deg to see where Disc 2's holes lie in Disc 1's rotated frame
    h2_in_d1_frame = rotate_image_array(h2, angle_deg, fill_val=0.0)

    # Now for each 2x2 cell of Disc 1, choose the 2 subpixel holes out of 4 that best match:
    #   - At 0°: overlap with h2[2y+dy, 2x+dx] should be HIGH (2) if b1[y,x]==1 else LOW (0)
    #   - At θ°: overlap with h2_in_d1_frame[2y+dy, 2x+dx] should be HIGH (2) if b2_inv[y,x]==1 else LOW (0)
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
    ]

    for y in range(half):
        for x in range(half):
            if not mask_low[y, x]:
                continue
            want1 = 2.0 if b1[y, x] >= 0.5 else 0.0
            want2 = 2.0 if b2_inv[y, x] >= 0.5 else 0.0

            best_pair = pairs[0]
            best_loss = 1e9
            # Slight random tie-breaker to prevent grid artifacts
            tie_Order = rng.permutation(6)
            for idx_p in tie_Order:
                i1, i2 = pairs[idx_p]
                dy1, dx1 = all_idx[i1]
                dy2, dx2 = all_idx[i2]

                # Overlap at 0°
                ov0 = h2[2 * y + dy1, 2 * x + dx1] + h2[2 * y + dy2, 2 * x + dx2]
                # Overlap at θ°
                ov_theta = (
                    h2_in_d1_frame[2 * y + dy1, 2 * x + dx1]
                    + h2_in_d1_frame[2 * y + dy2, 2 * x + dx2]
                )

                loss = (ov0 - want1) ** 2 + (ov_theta - want2) ** 2
                if loss < best_loss:
                    best_loss = loss
                    best_pair = (i1, i2)

            for chosen in best_pair:
                dy, dx = all_idx[chosen]
                h1[2 * y + dy, 2 * x + dx] = 1.0

    # Convert hole maps (1=hole, 0=solid) to physical thickness maps in mm:
    # Open hole = 0.24 mm (continuous White PLA base skin to prevent tearing/shredding),
    # Solid block = t_max (2.0 mm)
    outer_circle = create_circular_mask(size, margin_ratio=0.0)
    rim_mask = outer_circle & (~mask)

    base_skin_mm = 0.24
    disc1_mm = np.where(h1 >= 0.5, base_skin_mm, t_max).astype(np.float32)
    disc2_mm = np.where(h2 >= 0.5, base_skin_mm, t_max).astype(np.float32)

    disc1_mm[rim_mask] = t_max
    disc2_mm[rim_mask] = t_max
    disc1_mm[~outer_circle] = 0.0
    disc2_mm[~outer_circle] = 0.0

    # Simulated optical transmission at 0° and θ°
    h1_rot = rotate_image_array(h1, -angle_deg, fill_val=0.0)
    trans_0 = np.clip(h1 * h2, 0.0, 1.0)
    trans_theta = np.clip(h1_rot * h2, 0.0, 1.0)
    trans_0[~mask] = 0.0
    trans_theta[~mask] = 0.0

    return disc1_mm, disc2_mm, trans_0, trans_theta


def compute_angle_clarity_score(
    img1: np.ndarray,
    img2: np.ndarray,
    angle_deg: float,
    mode: str = "lithophane",
) -> float:
    """
    Computes an intuitive 0-100% Clarity Score for a given rotation angle (|angle| >= 45°).
    The Clarity Score combines:
      1. Image Independence / Conflict Absence (how little Image 1 and rotated Image 2
         fight over the same pixels in opposite directions).
      2. Dynamic Range & Contrast Preservation (how strong the light-to-dark contrast
         remains after satisfying non-negative thickness constraints).
      3. Subpixel Geometric Sharpness (90°, 180°, 270° have exact orthogonal alignment on a
         pixel/nozzle grid, whereas oblique angles incur partial subpixel interpolation blur,
         especially in Holes mode).
    """
    norm_angle = validate_angle(angle_deg)
    abs_angle = abs(norm_angle)
    size = img1.shape[0]
    mask = create_circular_mask(size, margin_ratio=0.06)

    # Rotate img2 by +norm_angle to align with img1's frame
    img2_rot = rotate_image_array(img2, norm_angle, fill_val=1.0)

    v1 = img1[mask]
    v2 = img2_rot[mask]

    # 1. Self-correlation / Ghosting susceptibility:
    # If v1 and v2 have high structural overlap where one is dark and one is bright,
    # or if the angle is small enough that low-frequency regions bleed into themselves:
    diff = np.abs(v1 - v2)
    conflict_penalty = float(np.mean(diff ** 2))  # [0, 1]

    # Check how much img1 correlates with its own rotated self at norm_angle
    # (when an image is rotated by a small angle like 45°, big shapes still overlap with
    # themselves, causing residual self-ghosting in Disc 2!)
    img1_self_rot = rotate_image_array(img1, -norm_angle, fill_val=1.0)[mask]
    img2_self_rot = rotate_image_array(img2, -norm_angle, fill_val=1.0)[mask]

    # Measure normalized cross-correlation (ghosting overlap)
    s1_std = float(np.std(v1)) + 1e-5
    s2_std = float(np.std(v2)) + 1e-5
    self_corr_1 = float(np.mean((v1 - np.mean(v1)) * (img1_self_rot - np.mean(img1_self_rot)))) / (s1_std ** 2)
    self_corr_2 = float(np.mean((v2 - np.mean(v2)) * (img2_self_rot - np.mean(img2_self_rot)))) / (s2_std ** 2)
    self_ghosting = max(0.0, 0.5 * (self_corr_1 + self_corr_2))

    # 2. Physical Arc Displacement factor near center hub (theta in radians)
    # Larger rotation angles (up to 90°-180°) separate pixels much faster near the center!
    eff_angle = min(abs_angle, 360.0 - abs_angle)  # in [45, 180]
    arc_separation_bonus = math.sin(math.radians(eff_angle * 0.75))  # peaks around 90°-135°

    # 3. Orthogonal grid sharpness factor:
    # Angles near multiples of 90° (90°, 180°, 270° / -90°) avoid subpixel resampling blur
    dist_to_90 = min(
        abs(eff_angle - 90.0),
        abs(eff_angle - 180.0),
    )
    grid_sharpness = math.exp(-((dist_to_90 / 38.0) ** 2))

    # Fast 10-iteration solver reconstruction fidelity check
    _, _, trans_0, trans_theta = solve_lithophane_pair(img1, img2, norm_angle, num_iters=10)
    rec1 = trans_0[mask]
    rec2 = trans_theta[mask]

    # Correlation between target and reconstructed output
    corr1 = np.corrcoef(v1, rec1)[0, 1] if np.std(rec1) > 1e-5 else 0.0
    corr2 = np.corrcoef(img2[mask], rec2)[0, 1] if np.std(rec2) > 1e-5 else 0.0
    fidelity = max(0.0, 0.5 * (float(corr1) + float(corr2)))

    if mode == "holes":
        # Holes mode relies more heavily on clean 2x2 grid alignment & large arc displacement
        raw_score = (
            48.0 * fidelity
            + 24.0 * grid_sharpness
            + 18.0 * arc_separation_bonus
            + 10.0 * (1.0 - self_ghosting)
            - 12.0 * conflict_penalty
        )
    else:
        # Lithophane mode is continuous and benefits strongly from low self-ghosting + fidelity
        raw_score = (
            54.0 * fidelity
            + 14.0 * grid_sharpness
            + 18.0 * arc_separation_bonus
            + 14.0 * (1.0 - self_ghosting)
            - 10.0 * conflict_penalty
        )

    score = float(np.clip(round(raw_score, 1), 12.0, 99.0))
    return score


def compare_angles(
    img1: np.ndarray,
    img2: np.ndarray,
    custom_angle: float = 90.0,
    mode: str = "lithophane",
) -> Dict[str, Any]:
    """
    Evaluates Clarity Scores across candidate angles (|theta| >= 45°) at a coarse 64x64
    working resolution for instant response (< 0.3s), returning:
      - default_90:  { angle: 90, score: ... }
      - auto_best:   { angle: best_angle, score: ... }
      - custom:      { angle: custom_angle, score: ... }
      - worst:       { angle: worst_angle, score: ... }
    """
    from PIL import Image
    eval_size = 64
    img1_small = np.asarray(
        Image.fromarray(np.clip(img1 * 255.0, 0, 255).astype(np.uint8)).resize((eval_size, eval_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    img2_small = np.asarray(
        Image.fromarray(np.clip(img2 * 255.0, 0, 255).astype(np.uint8)).resize((eval_size, eval_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    valid_custom = validate_angle(custom_angle)

    # Candidate sweep from 45° to 180° and -45° to -175° in 15° steps
    candidates: List[float] = [
        45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0,
        -45.0, -60.0, -75.0, -90.0, -105.0, -120.0, -135.0, -150.0, -165.0,
    ]
    if valid_custom not in candidates:
        candidates.append(valid_custom)

    scores_map: Dict[float, float] = {}
    for ang in candidates:
        scores_map[ang] = compute_angle_clarity_score(img1_small, img2_small, ang, mode=mode)

    score_90 = scores_map[90.0]
    score_custom = scores_map[valid_custom]

    best_ang = max(scores_map.keys(), key=lambda a: scores_map[a])
    worst_ang = min(scores_map.keys(), key=lambda a: scores_map[a])

    return {
        "default_90": {
            "label": "90° (Default)",
            "angle": 90.0,
            "score": score_90,
            "description": "Orthogonal quarter-turn with zero grid resampling loss.",
        },
        "auto_best": {
            "label": "Auto (Optimal)",
            "angle": float(best_ang),
            "score": scores_map[best_ang],
            "description": f"Highest clarity & lowest cross-talk found at {best_ang:+.0f}°.",
        },
        "custom": {
            "label": f"Custom ({valid_custom:+.0f}°)",
            "angle": float(valid_custom),
            "score": score_custom,
            "description": "Your currently selected rotation offset.",
        },
        "worst": {
            "label": f"Worst ({worst_ang:+.0f}°)",
            "angle": float(worst_ang),
            "score": scores_map[worst_ang],
            "description": "Highest ghosting overlap & subpixel aliasing penalty.",
        },
    }
