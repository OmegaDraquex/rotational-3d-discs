"""
Image Preprocessor & Sample Generator for Rotational Dual-Image 3D Generator.
Handles square padding, scaling, circular masking, contrast normalization,
and generating built-in sample images for instant testing.
"""

import io
import math
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter


def create_circular_mask(size: int, margin_ratio: float = 0.04) -> np.ndarray:
    """
    Returns a boolean mask of shape (size, size) where True indicates pixels
    inside the printable active circle (excluding the outer structural rim).
    """
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    max_radius = (size / 2.0) * (1.0 - margin_ratio)
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    return dist_sq <= (max_radius ** 2)


def apply_photo_anti_diffusion_boost(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies multi-scale local contrast enhancement (unsharp mask at fine & medium scales)
    plus a steep sigmoidal S-curve contrast stretch to pre-compensate for lateral
    subsurface light scattering in translucent 3D-printed PLA.
    Works on both 2D grayscale (H, W) and 3D RGB (H, W, 3) arrays in [0.0, 1.0].
    """
    out = arr.copy()
    if out.ndim == 2:
        fine_blur = gaussian_filter(out, sigma=1.2)
        med_blur = gaussian_filter(out, sigma=4.5)
        # Boost fine edges and medium-scale local features (eyes, mouth, contours)
        enhanced = out + 0.55 * (out - fine_blur) + 0.35 * (out - med_blur)
        enhanced = np.clip(enhanced, 0.0, 1.0)
        # S-curve contrast stretch around 0.5 to push highlights to 1.0 and shadows to 0.0
        centered = (enhanced[mask] - 0.5) * 2.0  # [-1, 1]
        s_curve = 0.5 + 0.5 * np.tanh(1.45 * centered) / math.tanh(1.45)
        out[mask] = np.clip(s_curve, 0.0, 1.0)
        out[~mask] = 1.0
    else:
        for c in range(3):
            ch = out[:, :, c]
            fine_blur = gaussian_filter(ch, sigma=1.2)
            med_blur = gaussian_filter(ch, sigma=4.5)
            enhanced = ch + 0.50 * (ch - fine_blur) + 0.30 * (ch - med_blur)
            enhanced = np.clip(enhanced, 0.0, 1.0)
            centered = (enhanced[mask] - 0.5) * 2.0
            s_curve = 0.5 + 0.5 * np.tanh(1.40 * centered) / math.tanh(1.40)
            ch_out = ch.copy()
            ch_out[mask] = np.clip(s_curve, 0.0, 1.0)
            ch_out[~mask] = 1.0
            out[:, :, c] = ch_out
    return out


def preprocess_image(
    img_bytes: bytes,
    target_size: int = 120,
    pad_color: int = 255,
    contrast_pct: float = 0.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Takes raw image bytes, converts to grayscale, pads to a square with blank (white)
    space, scales to (target_size, target_size) inscribed within the active circular
    area, applies optional inversion and contrast adjustment (-100% to +100%),
    and normalizes brightness into float32 [0.0, 1.0] where:
    - 1.0 = bright / white (maximum light transmission)
    - 0.0 = dark / black (minimum light transmission)
    """
    img = Image.open(io.BytesIO(img_bytes))

    # Handle RGBA transparency by compositing onto white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        bg = Image.new("RGBA", alpha_img.size, (pad_color, pad_color, pad_color, 255))
        composited = Image.alpha_composite(bg, alpha_img)
        gray = composited.convert("L")
    else:
        gray = img.convert("L")

    # Apply invert to the source image before padding so padding always stays blank white (1.0)
    if invert:
        gray = ImageOps.invert(gray)

    # Pad to square first so aspect ratio is preserved
    w, h = gray.size
    max_dim = max(w, h)
    square_img = Image.new("L", (max_dim, max_dim), color=pad_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square_img.paste(gray, (offset_x, offset_y))

    # Scale so the square content fits comfortably inside the circular disc
    # We use ~82% of diameter so corners of square images don't get clipped aggressively
    inner_size = max(16, int(round(target_size * 0.82)))
    resized = square_img.resize((inner_size, inner_size), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (target_size, target_size), color=pad_color)
    paste_pos = (target_size - inner_size) // 2
    canvas.paste(resized, (paste_pos, paste_pos))

    # Auto-contrast stretch inside the circle to ensure strong dynamic range
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mask = create_circular_mask(target_size, margin_ratio=0.05)

    inside_vals = arr[mask]
    p_low = np.percentile(inside_vals, 2)
    p_high = np.percentile(inside_vals, 98)
    if p_high - p_low > 0.05:
        arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)

    # Apply independent user contrast adjustment (-100% to +100%)
    c = float(np.clip(contrast_pct, -100.0, 100.0))
    if abs(c) > 1e-3:
        # Standard photographic contrast factor mapping [-100, +100] -> [0.15, 3.8]
        factor = (100.0 + c) / (100.0 - 0.68 * c) if c >= 0 else (100.0 + 0.85 * c) / 100.0
        arr[mask] = np.clip(0.5 + factor * (arr[mask] - 0.5), 0.0, 1.0)

    # Outside the active circle, set to 1.0 (blank space)
    arr[~mask] = 1.0
    return arr


def rotate_image_array(arr: np.ndarray, angle_deg: float, fill_val: float = 1.0) -> np.ndarray:
    """
    Rotates a 2D float32 array in [0, 1] counter-clockwise by angle_deg around its center.
    Uses PIL BICUBIC resampling for clean rotation at any arbitrary angle.
    """
    # Normalize angle
    angle_mod = angle_deg % 360.0
    if abs(angle_mod) < 1e-4:
        return arr.copy()
    if abs(angle_mod - 90.0) < 1e-4:
        return np.rot90(arr, k=1).copy()
    if abs(angle_mod - 180.0) < 1e-4:
        return np.rot90(arr, k=2).copy()
    if abs(angle_mod - 270.0) < 1e-4:
        return np.rot90(arr, k=3).copy()

    uint8_arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_arr, mode="L")
    fill_int = int(round(fill_val * 255.0))
    rotated_pil = pil_img.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill_int,
    )
    out = np.asarray(rotated_pil, dtype=np.float32) / 255.0
    mask = create_circular_mask(arr.shape[0], margin_ratio=0.05)
    out[~mask] = fill_val
    return out


def floyd_steinberg_dither(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies Floyd-Steinberg error-diffusion dithering to a float32 [0, 1] image.
    Returns a binary array where 1.0 = white (open hole) and 0.0 = black (solid block).
    """
    h, w = arr.shape
    work = arr.copy()
    # Enhance midtone contrast slightly for crisper dithering
    work = np.clip((work - 0.5) * 1.15 + 0.5, 0.0, 1.0)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                work[y, x] = 1.0
                continue
            old_pixel = work[y, x]
            new_pixel = 1.0 if old_pixel >= 0.5 else 0.0
            work[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < w and mask[y, x + 1]:
                work[y, x + 1] += quant_error * (7.0 / 16.0)
            if y + 1 < h:
                if x - 1 >= 0 and mask[y + 1, x - 1]:
                    work[y + 1, x - 1] += quant_error * (3.0 / 16.0)
                if mask[y + 1, x]:
                    work[y + 1, x] += quant_error * (5.0 / 16.0)
                if x + 1 < w and mask[y + 1, x + 1]:
                    work[y + 1, x + 1] += quant_error * (1.0 / 16.0)

    return (work >= 0.5).astype(np.float32)


def preprocess_color_image(
    img_bytes: bytes,
    target_size: int = 120,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
    contrast_pct: float = 0.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Takes raw image bytes, converts to RGB, pads to a square with blank (white)
    space, scales to (target_size, target_size) inscribed within the active circular
    area, applies optional inversion and contrast adjustment (-100% to +100%),
    and normalizes RGB values into float32 (target_size, target_size, 3) in [0.0, 1.0].
    """
    img = Image.open(io.BytesIO(img_bytes))

    # Handle RGBA transparency by compositing onto white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        bg = Image.new("RGBA", alpha_img.size, (*pad_color, 255))
        composited = Image.alpha_composite(bg, alpha_img)
        rgb = composited.convert("RGB")
    else:
        rgb = img.convert("RGB")

    if invert:
        rgb = ImageOps.invert(rgb)

    w, h = rgb.size
    max_dim = max(w, h)
    square_img = Image.new("RGB", (max_dim, max_dim), color=pad_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square_img.paste(rgb, (offset_x, offset_y))

    inner_size = max(16, int(round(target_size * 0.82)))
    resized = square_img.resize((inner_size, inner_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), color=pad_color)
    paste_pos = (target_size - inner_size) // 2
    canvas.paste(resized, (paste_pos, paste_pos))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mask = create_circular_mask(target_size, margin_ratio=0.05)

    # Gentle auto-levels inside the circle across all channels
    inside_vals = arr[mask]
    p_low = np.percentile(inside_vals, 2)
    p_high = np.percentile(inside_vals, 98)
    if p_high - p_low > 0.05:
        arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)

    c = float(np.clip(contrast_pct, -100.0, 100.0))
    if abs(c) > 1e-3:
        factor = (100.0 + c) / (100.0 - 0.68 * c) if c >= 0 else (100.0 + 0.85 * c) / 100.0
        arr[mask] = np.clip(0.5 + factor * (arr[mask] - 0.5), 0.0, 1.0)

    arr[~mask] = 1.0
    return arr


def rotate_color_array(arr: np.ndarray, angle_deg: float, fill_val: float = 1.0) -> np.ndarray:
    """
    Rotates a 3D float32 RGB array (H, W, 3) in [0, 1] counter-clockwise by angle_deg.
    """
    if arr.ndim == 2:
        return rotate_image_array(arr, angle_deg, fill_val)

    angle_mod = angle_deg % 360.0
    if abs(angle_mod) < 1e-4:
        return arr.copy()
    if abs(angle_mod - 90.0) < 1e-4:
        return np.rot90(arr, k=1, axes=(0, 1)).copy()
    if abs(angle_mod - 180.0) < 1e-4:
        return np.rot90(arr, k=2, axes=(0, 1)).copy()
    if abs(angle_mod - 270.0) < 1e-4:
        return np.rot90(arr, k=3, axes=(0, 1)).copy()

    uint8_arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_arr, mode="RGB")
    fill_int = int(round(fill_val * 255.0))
    rotated_pil = pil_img.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(fill_int, fill_int, fill_int),
    )
    out = np.asarray(rotated_pil, dtype=np.float32) / 255.0
    mask = create_circular_mask(arr.shape[0], margin_ratio=0.05)
    out[~mask] = fill_val
    return out


def generate_sample_images(size: int = 256) -> Tuple[bytes, bytes]:
    """
    Generates two distinct, visually rich FULL-COLOR sample PNG images:
      - Sample 1: Vibrant Golden/Crimson Sunburst over an Azure sky ring (with smiling face)
      - Sample 2: Deep Indigo/Violet Crescent Moon & Golden/Cyan Stars
    Works seamlessly in both Monochrome mode (via luminance conversion) and Multi-Color AMS mode!
    """
    # Sample 1: Golden/Crimson Sunburst
    img1 = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    cx, cy = size // 2, size // 2

    # Soft sky blue background circle so multi-color mode has rich cyan/blue contrast too
    bg_r = int(size * 0.45)
    draw1.ellipse((cx - bg_r, cy - bg_r, cx + bg_r, cy + bg_r), fill=(215, 240, 255))

    # Draw alternating crimson & vibrant orange rays
    num_rays = 12
    for i in range(num_rays):
        angle = i * (2 * math.pi / num_rays)
        r_outer = size * 0.44 if i % 2 == 0 else size * 0.36
        angle_left = angle - 0.14
        angle_right = angle + 0.14
        p1 = (cx + math.cos(angle) * r_outer, cy + math.sin(angle) * r_outer)
        p2 = (cx + math.cos(angle_left) * (size * 0.21), cy + math.sin(angle_left) * (size * 0.21))
        p3 = (cx + math.cos(angle_right) * (size * 0.21), cy + math.sin(angle_right) * (size * 0.21))
        ray_col = (210, 35, 25) if i % 2 == 0 else (245, 130, 15)
        draw1.polygon([p1, p2, p3], fill=ray_col)

    # Central sun disc with radial warm gradient (deep scarlet rim -> bright gold center)
    sun_r = int(size * 0.23)
    for r in range(sun_r, 0, -1):
        t = 1.0 - (r / sun_r)
        cr = int(200 + 50 * t)
        cg = int(45 + 155 * t)
        cb = int(15 + 25 * t)
        draw1.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(cr, cg, cb))

    # Eyes & smile on the sun
    eye_dx = int(size * 0.07)
    eye_y = int(cy - size * 0.04)
    eye_r = int(size * 0.025)
    draw1.ellipse((cx - eye_dx - eye_r, eye_y - eye_r, cx - eye_dx + eye_r, eye_y + eye_r), fill=(255, 252, 235))
    draw1.ellipse((cx + eye_dx - eye_r, eye_y - eye_r, cx + eye_dx + eye_r, eye_y + eye_r), fill=(255, 252, 235))
    smile_r = int(size * 0.09)
    draw1.arc((cx - smile_r, cy - smile_r // 2, cx + smile_r, cy + smile_r), start=20, end=160, fill=(255, 252, 235), width=5)

    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")

    # Sample 2: Royal Indigo/Magenta Crescent Moon & Cyan/Gold Stars
    img2 = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)

    # Soft lavender-mint night glow inside the circle
    draw2.ellipse((cx - bg_r, cy - bg_r, cx + bg_r, cy + bg_r), fill=(242, 236, 255))

    moon_r = int(size * 0.34)
    for r in range(moon_r, 0, -1):
        t = 1.0 - (r / moon_r)
        cr = int(35 + 85 * t)
        cg = int(18 + 30 * t)
        cb = int(125 + 95 * t)
        draw2.ellipse(
            (cx - moon_r - int(size * 0.03) + (moon_r - r), cy - r, cx + moon_r - int(size * 0.03) - (moon_r - r), cy + r),
            fill=(cr, cg, cb),
        )

    # Cutout circle to form crescent
    cut_r = int(size * 0.27)
    cut_cx = cx + int(size * 0.11)
    cut_cy = cy - int(size * 0.04)
    draw2.ellipse((cut_cx - cut_r, cut_cy - cut_r, cut_cx + cut_r, cut_cy + cut_r), fill=(242, 236, 255))

    # Draw 3 four-pointed stars in vivid Magenta, Teal-Cyan, and Deep Gold
    stars = [
        (cx + int(size * 0.18), cy - int(size * 0.16), int(size * 0.095), (215, 20, 125)),
        (cx + int(size * 0.25), cy + int(size * 0.06), int(size * 0.075), (0, 150, 185)),
        (cx + int(size * 0.08), cy + int(size * 0.18), int(size * 0.068), (230, 145, 0)),
    ]
    for sx, sy, sr, scol in stars:
        pts = [
            (sx, sy - sr),
            (sx + sr * 0.3, sy - sr * 0.3),
            (sx + sr, sy),
            (sx + sr * 0.3, sy + sr * 0.3),
            (sx, sy + sr),
            (sx - sr * 0.3, sy + sr * 0.3),
            (sx - sr, sy),
            (sx - sr * 0.3, sy - sr * 0.3),
        ]
        draw2.polygon(pts, fill=scol)

    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")

    return buf1.getvalue(), buf2.getvalue()
