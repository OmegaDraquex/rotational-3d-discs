"""
Automated Unit & Integration Tests for the Rotational Dual-Image 3D Generator.
Verifies:
  1. Angle validation (|angle| >= 45°).
  2. Clarity Score & 4-way Comparison (90° vs Auto vs Custom vs Worst).
  3. Lithophane Engine optical reconstruction at 0° and θ°.
  4. Full Holes Engine optical reconstruction at 0° and θ°.
  5. Binary STL mesh generation & ZIP packaging.
  6. FastAPI endpoints (/api/samples, /api/clarity, /api/generate, /api/download-stls).
"""

import struct
import zipfile
import io
import numpy as np
from fastapi.testclient import TestClient

from backend.preprocessor import (
    generate_sample_images,
    preprocess_image,
    preprocess_color_image,
    create_circular_mask,
)
from backend.engines import (
    validate_angle,
    solve_lithophane_pair,
    solve_holes_pair,
    compare_angles,
)
from backend.color_engine import (
    optimize_palette_for_images,
    solve_multicolor_pair,
    compare_color_angles,
)
from backend.stl_generator import generate_disc_stl, generate_slot_stl, package_both_stls_zip
from backend.app import app


def verify_stl_is_100_percent_manifold(stl_bytes: bytes, label: str = "mesh"):
    """
    Parses a binary STL and verifies strict 2-manifold topology:
      - 0 open edges (every directed half-edge (u -> v) has an exact opposite (v -> u),
        meaning every undirected edge is shared by an even number of faces, specifically 2!)
      - 0 non-manifold edges (every undirected edge is shared by strictly 2 triangles, never 1, 3, or 4!)
    """
    num_tris = struct.unpack("<I", stl_bytes[80:84])[0]
    if num_tris <= 1:
        return  # Empty placeholder slot
    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    recs = np.frombuffer(stl_bytes, dtype=stl_dtype, offset=84, count=num_tris)
    # Quantize vertex coordinates to 1e-4 mm to build exact topological edge table
    v0 = np.round(recs["v0"] * 10000).astype(np.int64)
    v1 = np.round(recs["v1"] * 10000).astype(np.int64)
    v2 = np.round(recs["v2"] * 10000).astype(np.int64)

    from collections import Counter
    undirected_counts = Counter()
    directed_counts = Counter()

    for i in range(num_tris):
        pts = [tuple(v0[i]), tuple(v1[i]), tuple(v2[i])]
        if pts[0] == pts[1] or pts[1] == pts[2] or pts[2] == pts[0]:
            continue
        for a, b in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            directed_counts[(a, b)] += 1
            undir = (a, b) if a < b else (b, a)
            undirected_counts[undir] += 1

    open_edges = sum(1 for e, c in undirected_counts.items() if c == 1)
    non_manifold_edges = sum(
        1 for (a, b), c in undirected_counts.items()
        if c != 2 or directed_counts[(a, b)] != 1 or directed_counts[(b, a)] != 1
    )
    assert open_edges == 0, f"[{label}] Found {open_edges} open edges!"
    assert non_manifold_edges == 0, f"[{label}] Found {non_manifold_edges} non-manifold edges!"
    print(f"[PASS] Manifold Check ({label}): {num_tris} triangles, 0 open edges, 0 non-manifold edges!")


def test_angle_validation():
    assert validate_angle(90.0) == 90.0
    assert validate_angle(-90.0) == -90.0
    assert validate_angle(10.0) == 45.0
    assert validate_angle(-20.0) == -45.0
    assert validate_angle(45.0) == 45.0
    print("[PASS] Angle validation (|theta| >= 45°)")


def test_engines_and_clarity():
    b1, b2 = generate_sample_images(size=128)
    img1 = preprocess_image(b1, target_size=64)
    img2 = preprocess_image(b2, target_size=64)
    mask = create_circular_mask(64, margin_ratio=0.06)

    # 1. Test Clarity Comparison
    comp = compare_angles(img1, img2, custom_angle=75.0, mode="lithophane")
    assert "default_90" in comp
    assert "auto_best" in comp
    assert "custom" in comp
    assert "worst" in comp
    assert comp["auto_best"]["score"] >= comp["default_90"]["score"]
    assert comp["auto_best"]["score"] >= comp["custom"]["score"]
    assert comp["custom"]["score"] >= comp["worst"]["score"]
    print(
        f"[PASS] Clarity Comparison -> Auto ({comp['auto_best']['angle']}°: {comp['auto_best']['score']}%) "
        f">= 90° ({comp['default_90']['score']}%) >= Custom ({comp['custom']['score']}%) "
        f">= Worst ({comp['worst']['angle']}°: {comp['worst']['score']}%)"
    )

    # 2. Test Lithophane Engine
    d1_mm, d2_mm, t0, t_theta = solve_lithophane_pair(img1, img2, angle_deg=90.0)
    corr_0 = np.corrcoef(img1[mask], t0[mask])[0, 1]
    corr_theta = np.corrcoef(img2[mask], t_theta[mask])[0, 1]
    assert corr_0 > 0.75, f"Expected high correlation at 0°, got {corr_0}"
    assert corr_theta > 0.75, f"Expected high correlation at 90°, got {corr_theta}"
    print(f"[PASS] Lithophane Engine correlation -> 0°: {corr_0:.3f}, 90°: {corr_theta:.3f}")

    # 2b. Test Ultra-Thin Photo Lithophane Mode (Asymmetric Back-Disc Boost + Flush Zero-Gap Rim)
    u_d1, u_d2, ut0, ut_theta = solve_lithophane_pair(
        img1, img2, angle_deg=90.0, t_min=0.16, t_max=0.80, ultra_thin=True
    )
    assert float(np.min(u_d1[mask])) < 0.28, f"Expected ultra-thin highlights near 0.16mm, got {np.min(u_d1[mask])}"
    assert float(np.max(u_d2[mask])) > float(np.max(u_d1[mask])) + 0.35, "Expected Back Disc (Disc 2) to have steeper contrast relief than Front Disc (Disc 1)!"
    stl_ultra = generate_disc_stl(u_d1, diameter_mm=80.0, max_thickness_mm=0.80, mode="ultrathin", notch_angles_deg=[0.0])
    verify_stl_is_100_percent_manifold(stl_ultra, label="disc_1_ultrathin_1notch")
    stl_ultra2 = generate_disc_stl(u_d2, diameter_mm=80.0, max_thickness_mm=0.80, mode="ultrathin", notch_angles_deg=[0.0, 90.0])
    verify_stl_is_100_percent_manifold(stl_ultra2, label="disc_2_ultrathin_2notches")
    print("[PASS] Ultra-Thin Litho Mode -> Asymmetric Back-Boost + Flush Rim + 1-Notch/2-Notch Manifold STLs!")

    # 2c. Test Halftone Photo Mode (Variable-Aperture Micro-Screen)
    from backend.engines import solve_halftone_photo_pair
    ht_d1, ht_d2, htt0, htt_theta = solve_halftone_photo_pair(img1, img2, angle_deg=90.0, t_max=1.0)
    stl_ht1 = generate_disc_stl(ht_d1, diameter_mm=80.0, max_thickness_mm=1.0, mode="halftone", notch_angles_deg=[0.0])
    verify_stl_is_100_percent_manifold(stl_ht1, label="disc_1_halftone_1notch")
    print("[PASS] Halftone Photo Mode -> Variable-Aperture Micro-Screen + 100% Manifold STL!")
    h1_mm, h2_mm, ht0, ht_theta = solve_holes_pair(img1, img2, angle_deg=90.0)
    # Check mean transmission in bright vs dark areas of img1 and img2
    bright1 = ht0[(img1 > 0.6) & mask].mean()
    dark1 = ht0[(img1 < 0.4) & mask].mean()
    bright2 = ht_theta[(img2 > 0.6) & mask].mean()
    dark2 = ht_theta[(img2 < 0.4) & mask].mean()
    assert bright1 > dark1 + 0.15
    assert bright2 > dark2 + 0.15
    print(
        f"[PASS] Holes Engine contrast -> Image 1 (bright={bright1:.2f} vs dark={dark1:.2f}), "
        f"Image 2 (bright={bright2:.2f} vs dark={dark2:.2f})"
    )

    # 4. Test Binary STL Generation & ZIP Bundle + Strict 0-Open-Edge / 0-Non-Manifold-Edge Check
    stl_bytes = generate_disc_stl(d1_mm, diameter_mm=80.0, max_thickness_mm=2.0, mode="lithophane")
    num_tris = struct.unpack("<I", stl_bytes[80:84])[0]
    assert len(stl_bytes) == 84 + num_tris * 50
    assert num_tris > 1000
    verify_stl_is_100_percent_manifold(stl_bytes, label="disc_1_lithophane")

    zip_bytes = package_both_stls_zip(d1_mm, d2_mm, 80.0, 2.0, "lithophane", 90.0)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
        assert "disc_1_lithophane.stl" in names
        assert "disc_2_lithophane.stl" in names
    print(f"[PASS] STL Generator -> {num_tris} triangles per disc, valid binary STL & ZIP archive")


def test_api_endpoints():
    client = TestClient(app)

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "Rotational Disc Studio" in r_root.text

    r_samples = client.get("/api/samples")
    assert r_samples.status_code == 200
    assert "sample1_b64" in r_samples.json()

    r_clarity = client.post("/api/clarity", data={"mode": "lithophane", "angle": 90, "auto_mode": True, "use_samples": True})
    assert r_clarity.status_code == 200
    c_json = r_clarity.json()
    assert "selected_score" in c_json
    assert "comparison" in c_json

    r_gen = client.post(
        "/api/generate",
        data={
            "mode": "lithophane",
            "angle": 90,
            "auto_mode": False,
            "use_samples": True,
            "contrast1": 35.0,
            "invert1": False,
            "contrast2": -25.0,
            "invert2": True,
        },
    )
    assert r_gen.status_code == 200
    g_json = r_gen.json()
    assert g_json["disc1_png"].startswith("data:image/png;base64,")
    assert g_json["disc2_png"].startswith("data:image/png;base64,")

    r_dl = client.post("/api/download-stls", data={"diameter_mm": 80.0, "thickness_mm": 2.0})
    assert r_dl.status_code == 200
    assert r_dl.headers["content-type"] == "application/zip"

    # Test Bambu AMS Multi-Color endpoints (Palette Optimization, Clarity, Generate, Multi-Part STL Download)
    r_pal = client.post(
        "/api/optimize-palette",
        data={"num_colors": 5, "palette_preset": "auto", "use_samples": True},
    )
    assert r_pal.status_code == 200
    pal_json = r_pal.json()
    assert pal_json["num_colors"] == 5
    assert len(pal_json["palette"]) == 5
    print(f"[PASS] Multi-Color Palette Optimizer (5 slots): {[p['name'] + ' ' + p['hex'] for p in pal_json['palette']]}")

    custom_hexes = '["#FFFFFF", "#00AEEF", "#EC008C", "#FFF200", "#E61919"]'
    r_c_ams = client.post(
        "/api/clarity",
        data={
            "mode": "lithophane",
            "color_mode": "multicolor",
            "num_colors": 5,
            "palette_json": custom_hexes,
            "angle": 90,
            "auto_mode": True,
            "use_samples": True,
        },
    )
    assert r_c_ams.status_code == 200
    assert "selected_score" in r_c_ams.json()

    r_gen_ams = client.post(
        "/api/generate",
        data={
            "mode": "lithophane",
            "color_mode": "multicolor",
            "num_colors": 5,
            "palette_json": custom_hexes,
            "angle": 90,
            "auto_mode": False,
            "use_samples": True,
        },
    )
    assert r_gen_ams.status_code == 200
    assert r_gen_ams.json()["color_mode"] == "multicolor"

    r_dl_ams = client.post("/api/download-stls", data={"diameter_mm": 80.0, "thickness_mm": 2.0})
    assert r_dl_ams.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r_dl_ams.content), "r") as zf:
        z_names = zf.namelist()
        ams_slot_files = [n for n in z_names if "bambu_ams_disc_1/" in n]
        assert len(ams_slot_files) == 5, f"Expected 5 AMS slot STL files for Disc 1, got {ams_slot_files}"
        for sf in ams_slot_files:
            verify_stl_is_100_percent_manifold(zf.read(sf), label=sf)
    print(f"[PASS] Multi-Color Bambu AMS STL Export -> {len(ams_slot_files)} slot STLs per disc verified!")
    print("[PASS] FastAPI REST Endpoints (/ , /api/samples, /api/optimize-palette, /api/clarity, /api/generate, /api/download-stls)")


if __name__ == "__main__":
    test_angle_validation()
    test_engines_and_clarity()
    test_api_endpoints()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
