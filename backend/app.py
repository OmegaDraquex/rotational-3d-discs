"""
FastAPI Backend for the Rotational Dual-Image 3D Generator (Monochrome & Bambu AMS Multi-Color).

Endpoints:
  - GET  /                     : Serves the Web UI (frontend/index.html)
  - GET  /api/samples          : Returns base64 previews of built-in sample Image 1 & Image 2
  - POST /api/optimize-palette : Calculates the optimal K-color Bambu AMS palette (2..8 colors)
                                 for the current pair of images
  - POST /api/clarity          : Calculates Clarity Score (0-100%) and 4-way comparison
                                 (90° vs Auto vs Custom vs Worst) for either Monochrome or Multi-Color
  - POST /api/generate         : Runs the selected engine (Lithophane/Holes + Monochrome/Multi-Color)
                                 and returns Disc 1 & Disc 2 backlit transmission maps for the 2D viewer
  - POST /api/download-stls    : Generates and downloads the ZIP bundle containing disc_1 & disc_2 STLs
                                 (including per-slot Bambu AMS multi-part STLs when in Multi-Color mode)
"""

import base64
import io
import json
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.preprocessor import (
    preprocess_image,
    preprocess_color_image,
    generate_sample_images,
    create_circular_mask,
)
from backend.engines import (
    validate_angle,
    solve_lithophane_pair,
    solve_halftone_photo_pair,
    solve_holes_pair,
    compare_angles,
)
from backend.color_engine import (
    get_default_cmyw_palette,
    optimize_palette_for_images,
    solve_multicolor_pair,
    compare_color_angles,
    closest_filament_name,
)
from backend.stl_generator import package_both_stls_zip

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Rotational Dual-Image 3D Generator")

# Cache for most recently generated discs so "Download Disc STLs" is instantaneous
_LAST_GENERATED_CACHE = {
    "disc1_mm": None,
    "disc2_mm": None,
    "slot1_map": None,
    "slot2_map": None,
    "palette_info": None,
    "color_mode": "monochrome",
    "mode": "lithophane",
    "angle_deg": 90.0,
    "diameter_mm": 80.0,
    "thickness_mm": 2.0,
}


def _array_to_base64_png(arr_0_to_1: np.ndarray, size: Optional[int] = None) -> str:
    """Converts a 2D (H,W) or 3D (H,W,3) float32 [0, 1] array into a data:image/png;base64 string."""
    uint8 = np.clip(np.round(arr_0_to_1 * 255.0), 0, 255).astype(np.uint8)
    mode_str = "RGB" if uint8.ndim == 3 else "L"
    pil_img = Image.fromarray(uint8, mode=mode_str)
    if size and size != arr_0_to_1.shape[0]:
        pil_img = pil_img.resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


async def _read_image_bytes(
    image1: Optional[UploadFile],
    image2: Optional[UploadFile],
    use_samples: bool,
) -> tuple[bytes, bytes]:
    if use_samples or (image1 is None and image2 is None):
        return generate_sample_images(size=256)
    b1_default, b2_default = generate_sample_images(size=256)
    b1 = await image1.read() if image1 is not None else b1_default
    b2 = await image2.read() if image2 is not None else b2_default
    return b1, b2


def _parse_palette_json(palette_json: str, num_colors: int) -> List[str]:
    """Parses user-provided JSON list of hex colors or falls back to default CMYW."""
    k = int(np.clip(num_colors, 2, 8))
    try:
        parsed = json.loads(palette_json)
        if isinstance(parsed, list) and len(parsed) >= 2:
            hexes = [str(x).strip() for x in parsed[:k]]
            while len(hexes) < k:
                hexes.append("#FFFFFF")
            return hexes
    except Exception:
        pass
    default_pal = get_default_cmyw_palette(k)
    return [item["hex"] for item in default_pal]


@app.get("/api/samples")
async def get_samples():
    """Returns built-in full-color sample images + default 4-color CMYW palette."""
    b1, b2 = generate_sample_images(size=256)
    arr1_rgb = preprocess_color_image(b1, target_size=140)
    arr2_rgb = preprocess_color_image(b2, target_size=140)
    return {
        "sample1_b64": _array_to_base64_png(arr1_rgb, size=200),
        "sample2_b64": _array_to_base64_png(arr2_rgb, size=200),
        "default_palette": get_default_cmyw_palette(4),
    }


@app.post("/api/optimize-palette")
async def api_optimize_palette(
    num_colors: int = Form(4),
    palette_preset: str = Form("auto"),  # "auto" | "cmyw"
    use_samples: bool = Form(False),
    contrast1: float = Form(0.0),
    invert1: bool = Form(False),
    contrast2: float = Form(0.0),
    invert2: bool = Form(False),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    """
    Calculates and returns the optimal K-color Bambu AMS filament palette (2 <= K <= 8)
    for the current two images (or returns the standard CMYW K-color preset).
    """
    k = int(np.clip(num_colors, 2, 8))
    if palette_preset == "cmyw":
        pal = get_default_cmyw_palette(k)
        return {"num_colors": k, "palette": pal}

    b1, b2 = await _read_image_bytes(image1, image2, use_samples)
    img1_rgb = preprocess_color_image(b1, target_size=96, contrast_pct=contrast1, invert=invert1)
    img2_rgb = preprocess_color_image(b2, target_size=96, contrast_pct=contrast2, invert=invert2)

    pal = optimize_palette_for_images(img1_rgb, img2_rgb, num_colors=k)
    return {"num_colors": k, "palette": pal}


@app.post("/api/clarity")
async def api_calculate_clarity(
    mode: str = Form("lithophane"),
    color_mode: str = Form("monochrome"),  # "monochrome" | "multicolor"
    num_colors: int = Form(4),
    palette_json: str = Form("[]"),
    angle: float = Form(90.0),
    auto_mode: bool = Form(False),
    use_samples: bool = Form(False),
    contrast1: float = Form(0.0),
    invert1: bool = Form(False),
    contrast2: float = Form(0.0),
    invert2: bool = Form(False),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    """
    Calculates the Clarity Score (0-100%) and full comparison across:
      90° vs Auto (Optimal) vs Custom vs Worst.
    Works for both Monochrome and Multi-Color (Bambu AMS) with respect to the active K-color palette!
    """
    valid_angle = validate_angle(angle)
    b1, b2 = await _read_image_bytes(image1, image2, use_samples)

    if color_mode == "multicolor":
        img1_rgb = preprocess_color_image(b1, target_size=72, contrast_pct=contrast1, invert=invert1)
        img2_rgb = preprocess_color_image(b2, target_size=72, contrast_pct=contrast2, invert=invert2)
        palette_hex_list = _parse_palette_json(palette_json, num_colors)
        comparison = compare_color_angles(
            img1_rgb, img2_rgb, palette_hex_list=palette_hex_list, custom_angle=valid_angle, mode=mode
        )
    else:
        arr1 = preprocess_image(b1, target_size=96, contrast_pct=contrast1, invert=invert1)
        arr2 = preprocess_image(b2, target_size=96, contrast_pct=contrast2, invert=invert2)
        comparison = compare_angles(arr1, arr2, custom_angle=valid_angle, mode=mode)

    if auto_mode:
        selected_angle = comparison["auto_best"]["angle"]
        selected_score = comparison["auto_best"]["score"]
    else:
        selected_angle = valid_angle
        selected_score = comparison["custom"]["score"]

    return {
        "selected_angle": selected_angle,
        "selected_score": selected_score,
        "comparison": comparison,
    }


@app.post("/api/generate")
async def api_generate_preview(
    mode: str = Form("lithophane"),
    color_mode: str = Form("monochrome"),
    num_colors: int = Form(4),
    palette_json: str = Form("[]"),
    angle: float = Form(90.0),
    auto_mode: bool = Form(False),
    diameter_mm: float = Form(80.0),
    thickness_mm: float = Form(2.0),
    use_samples: bool = Form(False),
    contrast1: float = Form(0.0),
    invert1: bool = Form(False),
    contrast2: float = Form(0.0),
    invert2: bool = Form(False),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    """
    Generates the two discs (Disc 1 & Disc 2) for the chosen mode, filament color mode,
    active K-color palette, and rotation angle.
    """
    grid_res = 200
    b1, b2 = await _read_image_bytes(image1, image2, use_samples)
    valid_angle = validate_angle(angle)

    is_ultra = mode == "ultrathin"
    t_min = 0.16 if is_ultra else 0.45
    t_max = max(0.45, float(thickness_mm)) if is_ultra else max(1.0, float(thickness_mm))

    if color_mode == "multicolor":
        img1_rgb = preprocess_color_image(b1, target_size=grid_res, contrast_pct=contrast1, invert=invert1)
        img2_rgb = preprocess_color_image(b2, target_size=grid_res, contrast_pct=contrast2, invert=invert2)
        palette_hex_list = _parse_palette_json(palette_json, num_colors)

        comparison = compare_color_angles(
            img1_rgb, img2_rgb, palette_hex_list=palette_hex_list, custom_angle=valid_angle, mode=mode
        )
        effective_angle = comparison["auto_best"]["angle"] if auto_mode else valid_angle
        clarity_score = comparison["auto_best"]["score"] if auto_mode else comparison["custom"]["score"]

        mc_res = solve_multicolor_pair(
            img1_rgb,
            img2_rgb,
            angle_deg=effective_angle,
            palette_hex_list=palette_hex_list,
            t_min=t_min,
            t_max=t_max,
            mode=mode,
        )

        palette_info = [
            {
                "slot": idx + 1,
                "hex": hx,
                "name": closest_filament_name(hx),
            }
            for idx, hx in enumerate(palette_hex_list)
        ]

        _LAST_GENERATED_CACHE["disc1_mm"] = mc_res["disc1_mm"]
        _LAST_GENERATED_CACHE["disc2_mm"] = mc_res["disc2_mm"]
        _LAST_GENERATED_CACHE["slot1_map"] = mc_res["slot1_map"]
        _LAST_GENERATED_CACHE["slot2_map"] = mc_res["slot2_map"]
        _LAST_GENERATED_CACHE["palette_info"] = palette_info
        _LAST_GENERATED_CACHE["color_mode"] = "multicolor"
        _LAST_GENERATED_CACHE["mode"] = mode
        _LAST_GENERATED_CACHE["angle_deg"] = effective_angle
        _LAST_GENERATED_CACHE["diameter_mm"] = float(diameter_mm)
        _LAST_GENERATED_CACHE["thickness_mm"] = float(t_max)

        return {
            "mode": mode,
            "color_mode": "multicolor",
            "palette_info": palette_info,
            "effective_angle": effective_angle,
            "clarity_score": clarity_score,
            "comparison": comparison,
            "grid_res": grid_res,
            "disc1_png": _array_to_base64_png(mc_res["d1_trans_rgb"], size=256),
            "disc2_png": _array_to_base64_png(mc_res["d2_trans_rgb"], size=256),
            "reveal_0_png": _array_to_base64_png(mc_res["reveal_0_rgb"], size=256),
            "reveal_theta_png": _array_to_base64_png(mc_res["reveal_theta_rgb"], size=256),
        }

    # Monochrome Mode
    arr1 = preprocess_image(b1, target_size=grid_res, contrast_pct=contrast1, invert=invert1)
    arr2 = preprocess_image(b2, target_size=grid_res, contrast_pct=contrast2, invert=invert2)
    comparison = compare_angles(arr1, arr2, custom_angle=valid_angle, mode=mode)

    effective_angle = comparison["auto_best"]["angle"] if auto_mode else valid_angle
    clarity_score = comparison["auto_best"]["score"] if auto_mode else comparison["custom"]["score"]

    outer_mask = create_circular_mask(grid_res, margin_ratio=0.0)
    active_mask = create_circular_mask(grid_res, margin_ratio=0.05)

    if mode == "holes":
        disc1_mm, disc2_mm, trans_0, trans_theta = solve_holes_pair(
            arr1, arr2, angle_deg=effective_angle, t_max=t_max
        )
        d1_trans = np.where(disc1_mm < 0.1, 1.0, 0.06).astype(np.float32)
        d2_trans = np.where(disc2_mm < 0.1, 1.0, 0.06).astype(np.float32)
        d1_trans[outer_mask & (~active_mask)] = 0.05
        d2_trans[outer_mask & (~active_mask)] = 0.05
        d1_trans[~outer_mask] = 0.0
        d2_trans[~outer_mask] = 0.0
    elif mode == "halftone":
        disc1_mm, disc2_mm, trans_0, trans_theta = solve_halftone_photo_pair(
            arr1, arr2, angle_deg=effective_angle, t_max=t_max
        )
        # Open apertures pass 1.0 light; variable-height struts/blocks pass modulated light
        d1_trans = np.where(
            disc1_mm < 0.05,
            1.0,
            np.clip(1.0 - 0.65 * (disc1_mm / (t_max + 1e-6)), 0.08, 0.55),
        ).astype(np.float32)
        d2_trans = np.where(
            disc2_mm < 0.05,
            1.0,
            np.clip(1.0 - 0.68 * (disc2_mm / (t_max + 1e-6)), 0.08, 0.52),
        ).astype(np.float32)
        d1_trans[outer_mask & (~active_mask)] = 0.06
        d2_trans[outer_mask & (~active_mask)] = 0.06
        d1_trans[~outer_mask] = 0.0
        d2_trans[~outer_mask] = 0.0
    else:
        disc1_mm, disc2_mm, trans_0, trans_theta = solve_lithophane_pair(
            arr1, arr2, angle_deg=effective_angle, t_min=t_min, t_max=t_max, ultra_thin=is_ultra
        )
        u1 = np.clip((disc1_mm - t_min) / (t_max - t_min + 1e-6), 0.0, 1.0)
        u2 = np.clip((disc2_mm - t_min) / (t_max - t_min + 1e-6), 0.0, 1.0)
        d1_trans = np.clip(1.0 - 0.55 * u1, 0.12, 1.0)
        d2_trans = np.clip(1.0 - 0.55 * u2, 0.12, 1.0)
        d1_trans[outer_mask & (~active_mask)] = 0.10
        d2_trans[outer_mask & (~active_mask)] = 0.10
        d1_trans[~outer_mask] = 0.0
        d2_trans[~outer_mask] = 0.0

    _LAST_GENERATED_CACHE["disc1_mm"] = disc1_mm
    _LAST_GENERATED_CACHE["disc2_mm"] = disc2_mm
    _LAST_GENERATED_CACHE["slot1_map"] = None
    _LAST_GENERATED_CACHE["slot2_map"] = None
    _LAST_GENERATED_CACHE["palette_info"] = None
    _LAST_GENERATED_CACHE["color_mode"] = "monochrome"
    _LAST_GENERATED_CACHE["mode"] = mode
    _LAST_GENERATED_CACHE["angle_deg"] = effective_angle
    _LAST_GENERATED_CACHE["diameter_mm"] = float(diameter_mm)
    _LAST_GENERATED_CACHE["thickness_mm"] = float(t_max)

    return {
        "mode": mode,
        "color_mode": "monochrome",
        "palette_info": None,
        "effective_angle": effective_angle,
        "clarity_score": clarity_score,
        "comparison": comparison,
        "grid_res": grid_res,
        "disc1_png": _array_to_base64_png(d1_trans, size=256),
        "disc2_png": _array_to_base64_png(d2_trans, size=256),
        "reveal_0_png": _array_to_base64_png(trans_0, size=256),
        "reveal_theta_png": _array_to_base64_png(trans_theta, size=256),
    }


@app.post("/api/download-stls")
async def api_download_stls(
    diameter_mm: float = Form(80.0),
    thickness_mm: float = Form(2.0),
):
    """
    Packages and downloads `disc_1` and `disc_2` STLs in a single ZIP archive
    (including Bambu AMS per-slot STL assemblies if Multi-Color mode was used).
    """
    if _LAST_GENERATED_CACHE["disc1_mm"] is None or _LAST_GENERATED_CACHE["disc2_mm"] is None:
        raise HTTPException(status_code=400, detail="Please click 'Generate Preview' first.")

    zip_bytes = package_both_stls_zip(
        disc1_mm=_LAST_GENERATED_CACHE["disc1_mm"],
        disc2_mm=_LAST_GENERATED_CACHE["disc2_mm"],
        diameter_mm=float(diameter_mm),
        max_thickness_mm=float(thickness_mm),
        mode=_LAST_GENERATED_CACHE["mode"],
        angle_deg=_LAST_GENERATED_CACHE["angle_deg"],
        color_mode=_LAST_GENERATED_CACHE["color_mode"],
        slot1_map=_LAST_GENERATED_CACHE["slot1_map"],
        slot2_map=_LAST_GENERATED_CACHE["slot2_map"],
        palette_info=_LAST_GENERATED_CACHE["palette_info"],
    )
    mode_name = _LAST_GENERATED_CACHE["mode"]
    cmode = _LAST_GENERATED_CACHE["color_mode"]
    ang_int = int(round(_LAST_GENERATED_CACHE["angle_deg"]))
    filename = f"rotational_discs_{cmode}_{mode_name}_{ang_int}deg_stls.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend loading...</h1>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
