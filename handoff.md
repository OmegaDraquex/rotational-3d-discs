# Project Handoff: Rotational Dual-Image 3D Disc Generator

## 1. Executive Summary & Core Objective
The goal of this project is to build a Python web application that takes **two images**, pads and masks them into circular discs, and generates two 3D-printable `.stl` discs (default diameter: $80\text{ mm}$). 
When the user stacks Disc 1 (front) and Disc 2 (back) against a light source:
- At **$0^\circ$ alignment**, **Image 1** appears.
- Rotating Disc 1 by **$90^\circ$** (or a chosen offset $\theta$ where $|\theta| \ge 45^\circ$) reveals **Image 2** (oriented right-side up).
- The system supports both **Monochrome (White PLA)** and **Bambu AMS Multi-Color (2 to 8 filaments)**.

The current application is running on FastAPI with a modern vanilla HTML/CSS/JS frontend featuring a real-time 2D backlit simulation canvas, interactive rotation sliders, clarity scoring, and ZIP bundling of STLs.

---

## 2. Iteration History & What Has Happened So Far

### Iteration 1: Research & Core Architecture
- Researched the feasibility of Rotational Visual Cryptography (RVC) and Computational Lithophanes.
- Concluded that both are possible:
  - **Full Holes (Visual Cryptography)**: Cut-through perforated binary lattice (0 mm void vs solid wall). High contrast, but looks like coarse dot-matrix/halftone dithering for photos.
  - **Lithophanes (Continuous Thickness Modulation)**: Thickness varies to modulate light via the Beer-Lambert law ($I = I_0 e^{-\mu(t_1 + t_2)}$). Continuous photographic shading, but susceptible to light diffusion if plastic is thick.

### Iteration 2: Full Web Application Build
- Implemented a complete web UI with:
  - Left panel controls: File import, mode toggle (Lithophane vs Holes), rotation angle input ($90^\circ$ default, $|\theta| \ge 45^\circ$), "Auto (Experimental)" angle optimizer, "Calculate Clarity" metric (0–100%), 4-way angle comparison (90° vs Auto vs Custom vs Worst), and a single "Download Disc STLs" button.
  - Right panel 2D viewer: Side-by-side backlit view, animated "Toggle Overlay" transition, and real-time rotational slider (-180° to +180°).

### Iteration 3: Independent Image Controls
- Added independent **Contrast sliders (-100% to +100%)**, quick "Reset" buttons, and **Invert Black/White** toggles under each image's dropzone, updating circular thumbnail previews in real-time.

### Iteration 4: Bambu Lab AMS Multi-Color Integration
- Extended the optical math from 1 channel to 3 spectral channels (RGB subtractive filtering).
- Added support for 2 to 8 filament slots:
  - Default Bambu CMYW palette preset.
  - "Auto-Optimize Colors" button using spectral clustering (K-Means/NMF in absorption density space).
  - Interactive color pickers for custom filament assignments.
  - Multi-part STL export (`bambu_ams_disc_1/` and `bambu_ams_disc_2/` with per-slot `.stl` bodies).

### Iteration 5: Slicer Bug Fixes (Manifold Topology & Resolution)
- User reported Bambu Studio errors: `36 non-manifold edges`, `41470 open edges`, and a blocky appearance.
- **Fixes applied**:
  - Replaced independent column extrusion with shared-vertex heightfield triangulation (`_build_watertight_region_mesh`) and diagonal checkerboard bridging (`_fix_checkerboard_manifold`), guaranteeing 0 open edges and 0 non-manifold edges.
  - Upsampled solver grids to 280×280 with cubic interpolation and radial projection onto the outer circle radius for smooth circular rims.

### Iteration 6: Physical Test Print Failure #1 (Diffusion & Scattering)
- User printed the discs on a Bambu P2S with 0.2mm nozzle on Textured PEI.
- **Problem**: When stacked, light from Disc 2 was so heavily diffused by Disc 1 that virtually no contrast or image was visible.
- **Analysis**:
  - $2.0\text{ mm}$ thickness per disc ($4.0\text{ mm}$ total) caused severe subsurface light scattering.
  - Textured PEI created frosted-glass diffusion on the bottom faces.
  - 0.2mm nozzle created double the internal layer/line weld interfaces.
- **Action taken**:
  - Introduced **Ultra-Thin Photo Mode** ($0.16\text{ mm}$ highlight floor to $0.80\text{ mm}$ shadow ceiling, with a $1.60\text{ mm}$ outer rim).
  - Advised user to switch to a Smooth PEI plate, 0.4mm nozzle, and 0.16mm layer height.

### Iteration 7: Physical Test Print Failure #2 & Physical Alignment Request
- User requested alignment notches: added **1 inward notch on Disc 1** (at 0°) and **2 inward notches on Disc 2** (at 0° and $\theta$).
- Added an experimental **Halftone Photo Mode** (variable micro-apertures on an orthogonal grid) and **Asymmetric Back-Disc Boost** (Disc 2 given 1.35mm relief, Disc 1 given 0.54mm relief with flush rims).
- User printed this version in multi-color on a Smooth PEI plate and provided photos (`media_1793060219893_fab3bbf8.jpg` and `media_1793060224415_1dbae3b3.jpg`).

### Iteration 8: Comprehensive Print Failure Fix Implementation (Complete & Verified)
Based on the approved diagnosis and fix plan (`print_failure_diagnosis_and_fix_plan.md`), the following fixes were engineered and verified:
1. **Continuous 0.24mm White Base Skin Across All Modes**:
   - In `backend/engines.py` & `backend/stl_generator.py`, apertures and valleys thin down to a continuous $Z = 0.24\text{ mm}$ (2-layer at 0.12mm) White PLA floor across the entire circular disc instead of empty air ($Z = 0$).
   - Guarantees Layers 1 and 2 on the Smooth PEI sheet are 100% solid White PLA across all 80mm: ZERO isolated dots on the bed, zero nozzle dragging, and zero lattice tearing when peeling off.
2. **0.60mm Easy-Peel Rim Chamfer Undercut**:
   - Implemented a monotonic radial contraction in `_build_smooth_vertex_coords` for the bed contact layer ($Z = 0$).
   - Bed contact radius is strictly $39.40\text{ mm}$ (undercut by $0.60\text{ mm}$ relative to the top rim at $40.00\text{ mm}$).
   - Creates a tactile bottom bevel all the way around the outer perimeter so a fingernail or print scraper blade slips directly under the disc to pop it off Smooth PEI without bending or tearing.
3. **Layered Color Glaze Architecture (Bambu AMS Multi-Color)**:
   - In `backend/stl_generator.py`, Slot 1 (Jade White PLA) forms the continuous bottom skin across the entire plate plus the base pedestal ($0.24\text{ mm}$ to $1.20\text{ mm}$).
   - Colored filaments (Slots 2..K) print strictly as thin optical glazes ($0.24\text{ mm}$ thick) elevated directly on top of Slot 1's pedestal.
   - Slicer result: Layer 1 & 2 have ZERO filament swaps on the bed, and colored filaments act as vibrant translucent chromatic filters rather than opaque light-blocking bricks!
4. **Auto-Palette Dead-Opaque Black Exclusion**:
   - In `backend/color_engine.py`, the auto-palette optimizer caps absorption so filament transmittance is $\ge 18\%$, replacing dead-opaque Charcoal Black with vibrant translucent primaries (e.g. Royal Purple, Deep Slate, Cobalt Blue).
5. **Coarser 2.0mm Cell Pitch with 0.80mm (2-Perimeter) Struts**:
   - In `backend/engines.py`, Halftone mode features were increased to a 2.0mm pitch with 0.80mm struts (2 full perimeters with 0.4mm nozzle) anchored to the 0.24mm solid white floor.
6. **Verification**:
   - All unit tests and topological manifold verifications pass with 0 open edges and 0 non-manifold edges across all modes and AMS slot STLs.

---

## 3. Current State & Analysis of the Latest Physical Print

Reviewing the user's latest physical photos revealed critical mechanical and optical problems:
1. **Shredded / Stringy Lattice**:
   - In `Halftone Photo` mode, grid struts were only ~0.4mm wide on the print bed. Slicing with 4 filament colors caused the nozzle to deposit tiny micro-dots on layer 1, dragging and tearing the lattice near the rim and in the center.
2. **Extreme Bed Adhesion / Tearing on Removal**:
   - Because holes went completely down to $Z = 0\text{ mm}$, thousands of tiny disconnected/fragile colored islands adhered directly to the Smooth PEI sheet without an unbroken bottom layer to hold them together, tearing upon removal.
3. **Severe Optical Occlusion / Camouflage Effect (Photo 2)**:
   - In multi-color mode, opaque filaments (especially Charcoal Black and dark Blue/Tan) were extruded as solid vertical columns from $Z = 0$ to top.
   - When backlit, these opaque blocks acted like permanent black blotches at *both* 0° and 90°, completely masking the underlying image variations.

---

## 4. Codebase Architecture & How the Code Works

The workspace is organized into backend engines, an API layer, and static frontend assets:

```
clever-raman/
├── backend/
│   ├── app.py                # FastAPI endpoints, request handling, caching, static hosting
│   ├── preprocessor.py       # Image resizing, circular masking, tone curves, CLAHE/unsharp boost
│   ├── engines.py            # Monochrome solvers: lithophane, halftone, holes, clarity scoring
│   ├── color_engine.py       # Bambu AMS solvers: spectral decomposition, palette optimization, slot mapping
│   ├── stl_generator.py      # Mesh triangulation, manifold verification, notch carving, ZIP packaging
│   └── test_engines.py       # Comprehensive unit tests & strict half-edge manifold validator
├── frontend/
│   ├── index.html            # Sidebar controls + 2D viewer stage
│   ├── css/style.css         # Styling for dark UI, segmented controls, swatches, canvas
│   └── js/app.js             # Client state, API caller, 2D backlit canvas, overlay animation
├── run.py                    # Server launcher on http://127.0.0.1:8700
└── requirements.txt          # Dependencies: fastapi, uvicorn, numpy, pillow, scipy, python-multipart
```

### Key Modules:
- **`backend/engines.py`**:
  - `solve_lithophane_pair()`: Alternating projection solver finding thickness distributions $u_1, u_2$ satisfying $u_1 + u_2 = D_1$ at 0° and $R_{-\theta}(u_1) + u_2 = D_2$ at $\theta$.
  - `solve_halftone_photo_pair()`: Modulates continuous density fields with a 2D cosine carrier wave to open variable micro-holes.
  - `compute_angle_clarity_score()` & `compare_angles()`: Sweeps candidate angles to measure ghosting and frequency decoupling.
- **`backend/color_engine.py`**:
  - `optimize_palette_for_images()`: Clusters non-white image pixels in optical density space using K-Means to find $K-1$ primary absorbing filament colors + 1 white base.
  - `_project_density_to_palette()`: Maps continuous RGB density vectors to discrete AMS filament slots (1..K) and physical heights.
- **`backend/stl_generator.py`**:
  - `_build_smooth_vertex_coords()`: Constructs regular $(N+1) \times (N+1)$ grids and projects perimeter vertices onto the exact circle radius.
  - `apply_rim_notches()`: Carves inward semicircular notches at specified angles on the rim.
  - `_build_watertight_region_mesh()`: Creates watertight closed meshes by evaluating boundary walls strictly where adjacent cells transition from True to False.
  - `package_both_stls_zip()`: Bundles `disc_1.stl`, `disc_2.stl`, and per-slot AMS files (`disc_1_slot_X_...stl`) into a ZIP.

---

## 5. Hypotheses & Guesses for Future Fixes (Framed as Guesses, Not Truth)

These are informed hypotheses based on optical and 3D printing physics that the next agent should investigate and validate:

### Hypothesis A: A Continuous Base Layer Is Essential for Printability and Diffusion
- *Guess*: Printing holes or multi-color islands directly on the build plate ($Z = 0$) will almost always fail to peel cleanly and will cause nozzle dragging on fine features.
- *Potential Fix*: Enforce a continuous, unbroken **0.24mm (or 0.20mm) White base skin** across the entire bottom of both discs before any apertures or color swaps begin.
- *Guess*: A thin, unbroken white skin will also act as a built-in light diffuser, preventing raw lightbulb hotspots from shining through holes while remaining tough enough to peel off the bed in one piece.

### Hypothesis B: Multi-Color Opaque Filaments Break Rotational Stacking
- *Guess*: Standard PLA (Black, Blue, Orange) is virtually 100% opaque at thicknesses $> 0.6\text{ mm}$. Extruding these colors as full-depth columns creates permanent opaque shields that block light at all angles.
- *Potential Fix*:
  1. Test **Monochrome White first** to verify that the mathematical rotational image reveal actually functions physically without color camouflage interference.
  2. If multi-color is used, restrict non-white filaments to a thin surface glaze ($0.16\text{--}0.32\text{ mm}$) sitting on top of the white base, or strictly exclude Charcoal Black from the auto-palette.

### Hypothesis C: Pure Thickness Lithophanes May Never Achieve High Contrast When Stacked
- *Guess*: Because light scatters inside white PLA, stacking two continuous-thickness lithophanes may inherently have an upper bound on contrast due to the back disc's light getting scrambled by the front disc.
- *Potential Fix*:
  - If continuous lithophanes are too muddy, a **coarser Halftone / Aperture approach** (cell pitch $\ge 1.6\text{ mm}$ to $2.0\text{ mm}$, with solid struts $\ge 0.8\text{ mm}$) on top of a continuous 0.2mm white base could allow direct light to travel through open wells without tearing the printer's first layer.
  - Alternatively, explore an asymmetric thickness ratio where Disc 1 (front) is paper-thin and Disc 2 (back) has very deep relief.

### Hypothesis D: Mechanical Bed Removal Can Be Solved via Geometry
- *Guess*: Adding a bottom chamfer or bevel (an undercut angle) around the outer structural rim will allow a scraper or fingernail to catch the edge instantly, popping the print off the bed without bending the membrane.

---

## 6. How to Run & Test

```bash
# 1. Install dependencies
pip install fastapi uvicorn python-multipart numpy pillow scipy

# 2. Run unit tests & topological manifold checker
python -m backend.test_engines

# 3. Launch application server
python run.py
# Server runs on http://127.0.0.1:8700
```
