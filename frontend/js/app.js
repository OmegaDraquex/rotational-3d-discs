/**
 * Rotational Dual-Image 3D Disc Studio - Frontend Controller & 2D Backlit Optical Renderer
 */

const state = {
  file1: null,
  file2: null,
  useSamples: true,
  contrast1: 0,             // -100 to +100
  invert1: false,
  contrast2: 0,             // -100 to +100
  invert2: false,
  baseSrc1: null,           // Raw data URL for live thumbnail contrast/invert preview
  baseSrc2: null,
  mode: "halftone",         // "halftone" | "ultrathin" | "lithophane" | "holes"
  colorMode: "monochrome",  // "monochrome" | "multicolor"
  numColors: 4,             // 2..8 filaments
  palette: [
    { slot: 1, hex: "#FFFFFF", name: "Jade White" },
    { slot: 2, hex: "#00AEEF", name: "Cyan" },
    { slot: 3, hex: "#EC008C", name: "Magenta" },
    { slot: 4, hex: "#FFF200", name: "Yellow" },
    { slot: 5, hex: "#E61919", name: "Scarlet Red" },
    { slot: 6, hex: "#1A56DB", name: "Cobalt Blue" },
    { slot: 7, hex: "#10B981", name: "Emerald Green" },
    { slot: 8, hex: "#1E293B", name: "Charcoal Black" },
  ],
  angle: 90,                // Must satisfy |angle| >= 45
  autoMode: false,
  effectiveAngle: 90,
  clarityScore: null,
  comparisonData: null,

  // Generated Disc images (HTMLImageElement loaded from base64 PNGs)
  disc1Img: null,
  disc2Img: null,
  hasGenerated: false,

  // Viewer state
  overlayMode: false,       // false = Side-by-Side, true = Stacked Overlay
  overlayProgress: 0.0,     // 0.0 (Side-by-Side) -> 1.0 (Stacked) animated
  animatingOverlay: false,
  panel1RotationDeg: 0.0,   // Controlled by bottom slider in Overlay Mode
  backlightIntensity: 1.05,
};

// DOM References
const dropzone1 = document.getElementById("dropzone-1");
const dropzone2 = document.getElementById("dropzone-2");
const fileInput1 = document.getElementById("file-input-1");
const fileInput2 = document.getElementById("file-input-2");
const thumb1 = document.getElementById("thumb-1");
const thumb2 = document.getElementById("thumb-2");
const filename1 = document.getElementById("filename-1");
const filename2 = document.getElementById("filename-2");
const btnLoadSamples = document.getElementById("btn-load-samples");

const contrastSlider1 = document.getElementById("contrast-slider-1");
const contrastVal1 = document.getElementById("contrast-val-1");
const invertToggle1 = document.getElementById("invert-toggle-1");
const btnReset1 = document.getElementById("btn-reset-1");

const contrastSlider2 = document.getElementById("contrast-slider-2");
const contrastVal2 = document.getElementById("contrast-val-2");
const invertToggle2 = document.getElementById("invert-toggle-2");
const btnReset2 = document.getElementById("btn-reset-2");

const modeHalftoneBtn = document.getElementById("mode-halftone");
const modeUltrathinBtn = document.getElementById("mode-ultrathin");
const modeLithophaneBtn = document.getElementById("mode-lithophane");
const modeHolesBtn = document.getElementById("mode-holes");
const advSummaryLabel = document.getElementById("adv-summary-label");
const colorModeMonoBtn = document.getElementById("colormode-mono");
const colorModeAmsBtn = document.getElementById("colormode-ams");
const amsPalettePanel = document.getElementById("ams-palette-panel");
const numColorsSelect = document.getElementById("num-colors-select");
const btnAutoPalette = document.getElementById("btn-auto-palette");
const btnCmywPreset = document.getElementById("btn-cmyw-preset");
const amsSlotsContainer = document.getElementById("ams-slots-container");

const angleInput = document.getElementById("angle-input");
const autoAngleToggle = document.getElementById("auto-angle-toggle");
const angleValidationMsg = document.getElementById("angle-validation-msg");

const btnCalcClarity = document.getElementById("btn-calc-clarity");
const btnCompareClarity = document.getElementById("btn-compare-clarity");
const clarityBanner = document.getElementById("clarity-banner");
const clarityBannerLabel = document.getElementById("clarity-banner-label");
const clarityBannerScore = document.getElementById("clarity-banner-score");
const clarityBannerBar = document.getElementById("clarity-banner-bar");

const comparisonCard = document.getElementById("comparison-card");
const btnCloseComp = document.getElementById("btn-close-comp");
const compListContainer = document.getElementById("comp-list-container");

const btnGeneratePreview = document.getElementById("btn-generate-preview");
const generateBtnText = document.getElementById("generate-btn-text");

const diameterInput = document.getElementById("diameter-input");
const thicknessInput = document.getElementById("thickness-input");
const btnDownloadStls = document.getElementById("btn-download-stls");
const downloadBtnText = document.getElementById("download-btn-text");
const stlStatusNote = document.getElementById("stl-status-note");

// Viewer DOM
const viewerModeTitle = document.getElementById("viewer-mode-title");
const backlightSlider = document.getElementById("backlight-slider");
const btnToggleOverlay = document.getElementById("btn-toggle-overlay");
const overlayBtnLabel = document.getElementById("overlay-btn-label");
const opticalCanvas = document.getElementById("optical-canvas");
const pillStateInfo = document.getElementById("pill-state-info");

const overlaySliderBar = document.getElementById("overlay-slider-bar");
const panel1RotationSlider = document.getElementById("panel1-rotation-slider");
const currentRotReadout = document.getElementById("current-rot-readout");
const revealStateBadge = document.getElementById("reveal-state-badge");
const btnSnap0 = document.getElementById("btn-snap-0");
const btnSnapTarget = document.getElementById("btn-snap-target");
const sliderTargetTick = document.getElementById("slider-target-tick");

// Offscreen compositing canvases for physical optical transmission rendering
const offCanvas1 = document.createElement("canvas");
const offCanvas2 = document.createElement("canvas");
offCanvas1.width = 256;
offCanvas1.height = 256;
offCanvas2.width = 256;
offCanvas2.height = 256;

// ==================== INITIALIZATION ====================
window.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  renderAmsSlotsUI();
  updateSliderBarState();
  await loadSampleThumbnails();
  // Automatically generate initial preview so the right viewer is immediately alive
  await generatePreview();
});

async function loadSampleThumbnails() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    state.baseSrc1 = data.sample1_b64;
    state.baseSrc2 = data.sample2_b64;
    await updateLiveThumbnail(1);
    await updateLiveThumbnail(2);
  } catch (err) {
    console.error("Failed to load sample thumbnails:", err);
  }
}

function setupEventListeners() {
  // 1. File Import
  dropzone1.addEventListener("click", () => fileInput1.click());
  dropzone2.addEventListener("click", () => fileInput2.click());

  fileInput1.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      setCustomFile(1, e.target.files[0]);
    }
  });
  fileInput2.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      setCustomFile(2, e.target.files[0]);
    }
  });

  // Drag & drop support
  [dropzone1, dropzone2].forEach((zone, idx) => {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.style.borderColor = "#f59e0b";
    });
    zone.addEventListener("dragleave", () => {
      zone.style.borderColor = "";
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.style.borderColor = "";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        setCustomFile(idx + 1, e.dataTransfer.files[0]);
      }
    });
  });

  btnLoadSamples.addEventListener("click", async () => {
    state.file1 = null;
    state.file2 = null;
    state.useSamples = true;
    filename1.textContent = "Built-in Sample (Sun)";
    filename2.textContent = "Built-in Sample (Moon)";
    resetImageAdjustment(1, false);
    resetImageAdjustment(2, false);
    await loadSampleThumbnails();
    await generatePreview();
  });

  // Independent Contrast & Invert Listeners for Image 1 & Image 2
  contrastSlider1.addEventListener("input", () => {
    state.contrast1 = parseFloat(contrastSlider1.value) || 0;
    const sign = state.contrast1 > 0 ? "+" : "";
    contrastVal1.textContent = `${sign}${state.contrast1}%`;
    updateLiveThumbnail(1);
  });
  invertToggle1.addEventListener("change", () => {
    state.invert1 = invertToggle1.checked;
    updateLiveThumbnail(1);
  });
  btnReset1.addEventListener("click", () => resetImageAdjustment(1, true));

  contrastSlider2.addEventListener("input", () => {
    state.contrast2 = parseFloat(contrastSlider2.value) || 0;
    const sign = state.contrast2 > 0 ? "+" : "";
    contrastVal2.textContent = `${sign}${state.contrast2}%`;
    updateLiveThumbnail(2);
  });
  invertToggle2.addEventListener("change", () => {
    state.invert2 = invertToggle2.checked;
    updateLiveThumbnail(2);
  });
  btnReset2.addEventListener("click", () => resetImageAdjustment(2, true));

  // 2. Mode Toggle (Halftone Photo vs Ultra-Thin Litho vs Thick Litho vs Full Holes)
  [modeHalftoneBtn, modeUltrathinBtn, modeLithophaneBtn, modeHolesBtn].forEach((btn) => {
    btn.addEventListener("click", () => {
      modeHalftoneBtn.classList.remove("active");
      modeUltrathinBtn.classList.remove("active");
      modeLithophaneBtn.classList.remove("active");
      modeHolesBtn.classList.remove("active");
      btn.classList.add("active");
      state.mode = btn.dataset.mode;

      if (state.mode === "ultrathin") {
        thicknessInput.value = "0.8";
        advSummaryLabel.textContent = "Advanced Print Settings (80mm, Asymmetric Back-Boost + Flush Rim)";
      } else if (state.mode === "halftone") {
        thicknessInput.value = "1.0";
        advSummaryLabel.textContent = "Advanced Print Settings (80mm, 1.0mm Halftone Micro-Screen)";
      } else {
        thicknessInput.value = "2.0";
        advSummaryLabel.textContent = "Advanced Print Settings (80mm, 2.0mm)";
      }
    });
  });

  // 2b. Filament Color Mode Toggle (Monochrome vs Multi-Color Bambu AMS)
  [colorModeMonoBtn, colorModeAmsBtn].forEach((btn) => {
    btn.addEventListener("click", async () => {
      colorModeMonoBtn.classList.remove("active");
      colorModeAmsBtn.classList.remove("active");
      btn.classList.add("active");
      state.colorMode = btn.dataset.colormode;
      amsPalettePanel.classList.toggle("hidden", state.colorMode !== "multicolor");
      await updateLiveThumbnail(1);
      await updateLiveThumbnail(2);
    });
  });

  // 2c. Number of Filament Colors (2..8) & Palette Optimizer / Custom Pickers
  numColorsSelect.addEventListener("change", () => {
    state.numColors = parseInt(numColorsSelect.value, 10) || 4;
    renderAmsSlotsUI();
  });

  btnAutoPalette.addEventListener("click", async () => {
    await fetchOptimizedPalette("auto");
  });

  btnCmywPreset.addEventListener("click", async () => {
    await fetchOptimizedPalette("cmyw");
  });

  // 3. Rotation Angle & Auto Toggle
  angleInput.addEventListener("change", () => {
    validateAndSetAngleInput();
  });

  autoAngleToggle.addEventListener("change", async () => {
    state.autoMode = autoAngleToggle.checked;
    angleInput.disabled = state.autoMode;
    if (state.autoMode) {
      angleValidationMsg.textContent = "Auto (Experimental) active: calculating optimal angle for maximum clarity...";
      angleValidationMsg.classList.remove("error");
      await calculateClarity(false);
    } else {
      angleValidationMsg.textContent = "Default is 90°. Any angle with magnitude ≥ 45° is supported.";
    }
  });

  // 4. Calculate Clarity & Compare Buttons
  btnCalcClarity.addEventListener("click", () => calculateClarity(false));
  btnCompareClarity.addEventListener("click", () => calculateClarity(true));
  btnCloseComp.addEventListener("click", () => comparisonCard.classList.add("hidden"));

  // 5. Generate Preview Button
  btnGeneratePreview.addEventListener("click", () => generatePreview());

  // 6. Download Disc STLs Button
  btnDownloadStls.addEventListener("click", () => downloadStls());

  // 7. Right 2D Viewer Controls
  backlightSlider.addEventListener("input", () => {
    state.backlightIntensity = parseFloat(backlightSlider.value) / 100.0;
    requestAnimationFrame(renderViewerCanvas);
  });

  btnToggleOverlay.addEventListener("click", () => {
    toggleOverlayMode(!state.overlayMode);
  });

  panel1RotationSlider.addEventListener("input", () => {
    state.panel1RotationDeg = parseFloat(panel1RotationSlider.value);
    updateRotationReadout();
    requestAnimationFrame(renderViewerCanvas);
  });

  btnSnap0.addEventListener("click", () => animateRotationTo(0));
  btnSnapTarget.addEventListener("click", () => animateRotationTo(state.effectiveAngle));
}

function setCustomFile(slot, file) {
  state.useSamples = false;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    if (slot === 1) {
      state.file1 = file;
      state.baseSrc1 = ev.target.result;
      filename1.textContent = file.name;
      await updateLiveThumbnail(1);
    } else {
      state.file2 = file;
      state.baseSrc2 = ev.target.result;
      filename2.textContent = file.name;
      await updateLiveThumbnail(2);
    }
  };
  reader.readAsDataURL(file);
}

function resetImageAdjustment(slot, updateThumb = true) {
  if (slot === 1) {
    state.contrast1 = 0;
    state.invert1 = false;
    contrastSlider1.value = 0;
    contrastVal1.textContent = "0%";
    invertToggle1.checked = false;
  } else {
    state.contrast2 = 0;
    state.invert2 = false;
    contrastSlider2.value = 0;
    contrastVal2.textContent = "0%";
    invertToggle2.checked = false;
  }
  if (updateThumb) {
    updateLiveThumbnail(slot);
  }
}

function renderAmsSlotsUI() {
  amsSlotsContainer.innerHTML = "";
  const k = state.numColors;

  for (let i = 0; i < k; i++) {
    const slotObj = state.palette[i] || { slot: i + 1, hex: "#FFFFFF", name: "Custom PLA" };
    const chip = document.createElement("label");
    chip.className = "ams-slot-chip";
    chip.title = `Click to customize AMS Slot ${i + 1} filament color`;

    chip.innerHTML = `
      <input type="color" class="ams-color-picker" value="${slotObj.hex}" data-slotidx="${i}" />
      <div class="ams-slot-meta">
        <span class="ams-slot-num">AMS Slot ${i + 1}${i === 0 ? " (Base)" : ""}</span>
        <span class="ams-slot-hex" id="ams-slot-label-${i}">${slotObj.name} (${slotObj.hex.toUpperCase()})</span>
      </div>
    `;

    const colorInput = chip.querySelector(".ams-color-picker");
    colorInput.addEventListener("input", (ev) => {
      const newHex = ev.target.value.toUpperCase();
      state.palette[i].hex = newHex;
      state.palette[i].name = "Custom";
      const lbl = document.getElementById(`ams-slot-label-${i}`);
      if (lbl) lbl.textContent = `Custom (${newHex})`;
    });

    amsSlotsContainer.appendChild(chip);
  }
}

async function fetchOptimizedPalette(presetType = "auto") {
  btnAutoPalette.disabled = true;
  btnCmywPreset.disabled = true;
  const origText = btnAutoPalette.textContent;
  if (presetType === "auto") {
    btnAutoPalette.textContent = "Optimizing Colors...";
  }

  try {
    const fd = new FormData();
    fd.append("num_colors", state.numColors);
    fd.append("palette_preset", presetType);
    fd.append("use_samples", state.useSamples);
    fd.append("contrast1", state.contrast1);
    fd.append("invert1", state.invert1);
    fd.append("contrast2", state.contrast2);
    fd.append("invert2", state.invert2);
    if (state.file1) fd.append("image1", state.file1);
    if (state.file2) fd.append("image2", state.file2);

    const res = await fetch("/api/optimize-palette", { method: "POST", body: fd });
    const data = await res.json();
    if (data.palette && Array.isArray(data.palette)) {
      data.palette.forEach((p, idx) => {
        state.palette[idx] = {
          slot: p.slot,
          hex: p.hex.toUpperCase(),
          name: p.name || "Filament",
        };
      });
      renderAmsSlotsUI();
    }
  } catch (err) {
    console.error("Failed to optimize palette:", err);
  } finally {
    btnAutoPalette.disabled = false;
    btnCmywPreset.disabled = false;
    btnAutoPalette.textContent = origText;
  }
}

async function updateLiveThumbnail(slot) {
  const baseSrc = slot === 1 ? state.baseSrc1 : state.baseSrc2;
  const targetThumb = slot === 1 ? thumb1 : thumb2;
  const c = slot === 1 ? state.contrast1 : state.contrast2;
  const inv = slot === 1 ? state.invert1 : state.invert2;

  if (!baseSrc) return;

  try {
    const img = await loadImageAsync(baseSrc);
    const size = 140;
    const tCanvas = document.createElement("canvas");
    tCanvas.width = size;
    tCanvas.height = size;
    const tCtx = tCanvas.getContext("2d");

    // Fill white background
    tCtx.fillStyle = "#ffffff";
    tCtx.fillRect(0, 0, size, size);

    // Fit square-padded image inside circle
    const maxDim = Math.max(img.width, img.height);
    const scale = (size * 0.86) / maxDim;
    const drawW = img.width * scale;
    const drawH = img.height * scale;
    const dx = (size - drawW) / 2;
    const dy = (size - drawH) / 2;
    tCtx.drawImage(img, dx, dy, drawW, drawH);

    const imgData = tCtx.getImageData(0, 0, size, size);
    const d = imgData.data;

    const factor =
      c >= 0 ? (100.0 + c) / (100.0 - 0.68 * c) : (100.0 + 0.85 * c) / 100.0;

    const isColor = state.colorMode === "multicolor";

    for (let i = 0; i < d.length; i += 4) {
      if (isColor) {
        for (let ch = 0; ch < 3; ch++) {
          let v = d[i + ch] / 255.0;
          if (inv) v = 1.0 - v;
          v = Math.min(1.0, Math.max(0.0, 0.5 + factor * (v - 0.5)));
          d[i + ch] = Math.round(v * 255);
        }
      } else {
        // Luminance
        let lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255.0;
        if (inv) {
          lum = 1.0 - lum;
        }
        lum = Math.min(1.0, Math.max(0.0, 0.5 + factor * (lum - 0.5)));
        const v = Math.round(lum * 255);
        d[i] = v;
        d[i + 1] = v;
        d[i + 2] = v;
      }
    }

    tCtx.putImageData(imgData, 0, 0);
    targetThumb.src = tCanvas.toDataURL("image/png");
  } catch (err) {
    console.error("Thumbnail update failed:", err);
  }
}

function validateAndSetAngleInput() {
  let raw = parseFloat(angleInput.value);
  if (isNaN(raw)) raw = 90;

  // Enforce |angle| >= 45
  if (Math.abs(raw) < 45) {
    const clamped = raw >= 0 ? 45 : -45;
    angleInput.value = clamped;
    state.angle = clamped;
    angleValidationMsg.textContent = `Angle must be ≥ |±45°|. Clamped to ${clamped}°.`;
    angleValidationMsg.classList.add("error");
  } else if (Math.abs(raw) > 315) {
    const clamped = raw >= 0 ? 315 : -315;
    angleInput.value = clamped;
    state.angle = clamped;
    angleValidationMsg.textContent = `Angle must stay ≥ 45° away from full 360° turn. Clamped to ${clamped}°.`;
    angleValidationMsg.classList.add("error");
  } else {
    state.angle = raw;
    angleValidationMsg.textContent = `Valid rotation offset (${raw > 0 ? "+" : ""}${raw}°).`;
    angleValidationMsg.classList.remove("error");
  }
}

function buildFormData() {
  validateAndSetAngleInput();
  const activeHexes = state.palette.slice(0, state.numColors).map((p) => p.hex);
  const fd = new FormData();
  fd.append("mode", state.mode);
  fd.append("color_mode", state.colorMode);
  fd.append("num_colors", state.numColors);
  fd.append("palette_json", JSON.stringify(activeHexes));
  fd.append("angle", state.angle);
  fd.append("auto_mode", state.autoMode);
  fd.append("use_samples", state.useSamples);
  fd.append("contrast1", state.contrast1);
  fd.append("invert1", state.invert1);
  fd.append("contrast2", state.contrast2);
  fd.append("invert2", state.invert2);
  fd.append("diameter_mm", parseFloat(diameterInput.value) || 80.0);
  fd.append("thickness_mm", parseFloat(thicknessInput.value) || 2.0);
  if (state.file1) fd.append("image1", state.file1);
  if (state.file2) fd.append("image2", state.file2);
  return fd;
}

// ==================== CLARITY & COMPARISON ====================
async function calculateClarity(showComparisonCard = false) {
  btnCalcClarity.disabled = true;
  btnCompareClarity.disabled = true;
  const origText = btnCalcClarity.innerHTML;
  btnCalcClarity.innerHTML = "<span>Calculating...</span>";

  try {
    const fd = buildFormData();
    const res = await fetch("/api/clarity", { method: "POST", body: fd });
    const data = await res.json();

    state.comparisonData = data.comparison;
    state.clarityScore = data.selected_score;

    if (state.autoMode) {
      state.angle = data.selected_angle;
      angleInput.value = data.selected_angle;
      angleValidationMsg.textContent = `Auto selected optimal angle: ${data.selected_angle > 0 ? "+" : ""}${data.selected_angle}°`;
    }

    renderClarityBanner(data.selected_angle, data.selected_score);
    if (showComparisonCard) {
      renderComparisonList(data.comparison);
      comparisonCard.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Clarity calculation error:", err);
  } finally {
    btnCalcClarity.disabled = false;
    btnCompareClarity.disabled = false;
    btnCalcClarity.innerHTML = origText;
  }
}

function renderClarityBanner(angleDeg, score) {
  clarityBanner.classList.remove("hidden");
  const sign = angleDeg > 0 ? "+" : "";
  const modeLabel =
    state.mode === "holes"
      ? "Holes"
      : state.mode === "halftone"
      ? "Halftone Photo"
      : state.mode === "ultrathin"
      ? "Ultra-Thin Litho"
      : "Lithophane";
  clarityBannerLabel.textContent = `Clarity Score at ${sign}${angleDeg}° (${modeLabel})`;
  clarityBannerScore.textContent = `${score.toFixed(1)}%`;
  clarityBannerBar.style.width = `${Math.min(100, Math.max(5, score))}%`;
}

function renderComparisonList(comp) {
  compListContainer.innerHTML = "";
  const order = ["default_90", "auto_best", "custom", "worst"];

  order.forEach((key) => {
    const item = comp[key];
    if (!item) return;

    const color =
      key === "worst"
        ? "#f43f5e"
        : key === "auto_best"
        ? "#10b981"
        : "#38bdf8";

    const row = document.createElement("div");
    row.className = "comp-item";
    row.innerHTML = `
      <div class="comp-item-top">
        <span class="comp-item-name">${item.label}</span>
        <div class="comp-item-right">
          <span class="comp-score-pill" style="color: ${color}">${item.score.toFixed(1)}%</span>
          <button type="button" class="btn-apply-angle" data-angle="${item.angle}">Apply ${item.angle > 0 ? "+" : ""}${item.angle}°</button>
        </div>
      </div>
      <div class="clarity-bar-bg">
        <div class="clarity-bar-fill" style="width: ${item.score}%; background: ${color}"></div>
      </div>
      <span style="font-size: 10px; color: #94a3b8;">${item.description}</span>
    `;

    const applyBtn = row.querySelector(".btn-apply-angle");
    applyBtn.addEventListener("click", () => {
      autoAngleToggle.checked = key === "auto_best";
      state.autoMode = autoAngleToggle.checked;
      angleInput.disabled = state.autoMode;
      angleInput.value = item.angle;
      state.angle = item.angle;
      renderClarityBanner(item.angle, item.score);
    });

    compListContainer.appendChild(row);
  });
}

// ==================== GENERATE PREVIEW ====================
async function generatePreview() {
  btnGeneratePreview.disabled = true;
  generateBtnText.textContent = "Generating Optical Discs...";

  try {
    const fd = buildFormData();
    const res = await fetch("/api/generate", { method: "POST", body: fd });
    const data = await res.json();

    state.effectiveAngle = data.effective_angle;
    state.clarityScore = data.clarity_score;
    state.comparisonData = data.comparison;

    if (state.autoMode) {
      angleInput.value = data.effective_angle;
      state.angle = data.effective_angle;
    }

    renderClarityBanner(data.effective_angle, data.clarity_score);
    if (!comparisonCard.classList.contains("hidden")) {
      renderComparisonList(data.comparison);
    }

    // Load Disc 1 and Disc 2 images for the 2D Backlit Canvas
    const [img1, img2] = await Promise.all([
      loadImageAsync(data.disc1_png),
      loadImageAsync(data.disc2_png),
    ]);

    state.disc1Img = img1;
    state.disc2Img = img2;
    state.hasGenerated = true;

    // Enable STL download button
    btnDownloadStls.disabled = false;
    stlStatusNote.textContent = `Ready: Disc 1 & Disc 2 (${state.mode}, ${state.effectiveAngle > 0 ? "+" : ""}${state.effectiveAngle}° reveal)`;

    // Update target angle button & tick on the bottom slider
    updateTargetAngleUI(state.effectiveAngle);

    // Render the 2D viewer
    requestAnimationFrame(renderViewerCanvas);
  } catch (err) {
    console.error("Failed to generate preview:", err);
  } finally {
    btnGeneratePreview.disabled = false;
    generateBtnText.textContent = "Generate Preview";
  }
}

function loadImageAsync(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function updateTargetAngleUI(targetDeg) {
  // Normalize targetDeg into [-180, 180] for the slider
  let sliderTarget = ((targetDeg + 180) % 360) - 180;
  if (sliderTarget <= -180) sliderTarget += 360;
  const sign = sliderTarget > 0 ? "+" : "";
  btnSnapTarget.textContent = `${sign}${sliderTarget}° (Image 2)`;

  const pct = ((sliderTarget + 180) / 360) * 100;
  sliderTargetTick.style.left = `${pct}%`;
  updateRotationReadout();
}

// ==================== OVERLAY ANIMATION & ROTATION SLIDER ====================
function toggleOverlayMode(targetOverlayState) {
  state.overlayMode = targetOverlayState;
  btnToggleOverlay.classList.toggle("active", state.overlayMode);
  overlayBtnLabel.textContent = state.overlayMode
    ? "Exit Overlay (Side-by-Side)"
    : "Toggle Overlay";

  viewerModeTitle.textContent = state.overlayMode
    ? "Stacked Optical Overlay — Rotate Panel 1 Below to Reveal Both Images"
    : "Side-by-Side Backlit View (Disc 1 & Disc 2)";

  updateSliderBarState();

  // Animate `state.overlayProgress` from current value to 1.0 (Overlay) or 0.0 (Side-by-Side)
  const startVal = state.overlayProgress;
  const endVal = state.overlayMode ? 1.0 : 0.0;
  const durationMs = 520;
  const startTime = performance.now();

  function step(now) {
    const t = Math.min(1.0, (now - startTime) / durationMs);
    // Smooth cubic ease-in-out
    const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    state.overlayProgress = startVal + (endVal - startVal) * eased;
    renderViewerCanvas();
    if (t < 1.0) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

function updateSliderBarState() {
  if (state.overlayMode) {
    overlaySliderBar.classList.remove("disabled-bar");
    pillStateInfo.textContent =
      "Panel 1 stacked in front of Panel 2 — Drag the rotation slider below to transform Image 1 into Image 2";
  } else {
    overlaySliderBar.classList.add("disabled-bar");
    pillStateInfo.textContent =
      "Showing the two 3D-printed white discs side-by-side with warm artificial backlight";
  }
}

function animateRotationTo(targetDeg) {
  let normTarget = ((targetDeg + 180) % 360) - 180;
  if (normTarget <= -180) normTarget += 360;

  const startDeg = state.panel1RotationDeg;
  const durationMs = 420;
  const startTime = performance.now();

  function step(now) {
    const t = Math.min(1.0, (now - startTime) / durationMs);
    const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    state.panel1RotationDeg = Math.round(startDeg + (normTarget - startDeg) * eased);
    panel1RotationSlider.value = state.panel1RotationDeg;
    updateRotationReadout();
    renderViewerCanvas();
    if (t < 1.0) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

function updateRotationReadout() {
  const deg = state.panel1RotationDeg;
  const sign = deg > 0 ? "+" : "";
  currentRotReadout.textContent = `${sign}${deg}°`;

  let normTarget = ((state.effectiveAngle + 180) % 360) - 180;
  if (normTarget <= -180) normTarget += 360;

  const dist0 = Math.abs(deg);
  const distTarget = Math.min(
    Math.abs(deg - normTarget),
    Math.abs(deg - normTarget + 360),
    Math.abs(deg - normTarget - 360)
  );

  btnSnap0.classList.toggle("active", dist0 <= 4);
  btnSnapTarget.classList.toggle("active", distTarget <= 4);

  if (dist0 <= 6) {
    revealStateBadge.textContent = "✦ Aligned: Revealing Image 1";
    revealStateBadge.style.color = "#6ee7b7";
    revealStateBadge.style.borderColor = "rgba(16, 185, 129, 0.5)";
  } else if (distTarget <= 6) {
    revealStateBadge.textContent = "✦ Aligned: Revealing Image 2 (Upright)";
    revealStateBadge.style.color = "#38bdf8";
    revealStateBadge.style.borderColor = "rgba(56, 189, 248, 0.5)";
  } else {
    revealStateBadge.textContent = "In-Between Angle (Optical Transition)";
    revealStateBadge.style.color = "#cbd5e1";
    revealStateBadge.style.borderColor = "rgba(148, 163, 184, 0.35)";
  }
}

// ==================== 2D BACKLIT OPTICAL RENDERER ====================
function renderViewerCanvas() {
  const ctx = opticalCanvas.getContext("2d");
  const W = opticalCanvas.width;
  const H = opticalCanvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!state.disc1Img || !state.disc2Img) return;

  const p = state.overlayProgress; // 0 = Side-by-Side, 1 = Stacked at center
  const centerStageX = W / 2;
  const centerStageY = H / 2 - 8;

  // Disc radius grows slightly when stacked in the center for a detailed view
  const radiusSide = 185;
  const radiusStacked = 225;
  const R = radiusSide + (radiusStacked - radiusSide) * p;

  // Positions of Disc 1 (left -> center) and Disc 2 (right -> center)
  const x1_side = W * 0.27;
  const x2_side = W * 0.73;

  const x1 = x1_side + (centerStageX - x1_side) * p;
  const x2 = x2_side + (centerStageX - x2_side) * p;
  const y1 = centerStageY;
  const y2 = centerStageY;

  // If we are in Side-by-Side or mid-animation (p < 0.98), draw Disc 2 and Disc 1 translating
  if (p < 0.98) {
    // Draw artificial light halos behind both discs
    drawBacklightHalo(ctx, x2, y2, R);
    drawBacklightHalo(ctx, x1, y1, R);

    // Draw Disc 2 first (underneath): 2 notches at 0° and +effectiveAngle
    drawSingleBacklitDisc(
      ctx,
      state.disc2Img,
      x2,
      y2,
      R,
      0,
      "DISC 2 (Back Plate • 2 Notches)",
      1.0 - p * 0.5,
      [0, state.effectiveAngle]
    );

    // Draw Disc 1 sliding over in front of Disc 2: 1 notch at 0°
    const rotRad = (state.panel1RotationDeg * Math.PI) / 180.0;
    drawSingleBacklitDisc(
      ctx,
      state.disc1Img,
      x1,
      y1,
      R,
      rotRad,
      "DISC 1 (Front Plate • 1 Notch)",
      1.0,
      [0]
    );
  }

  // When stacked (or nearly stacked p >= 0.65), blend in the true multiplicative optical stack
  if (p >= 0.65) {
    const stackAlpha = Math.min(1.0, (p - 0.65) / 0.35);
    ctx.save();
    ctx.globalAlpha = stackAlpha;
    drawBacklightHalo(ctx, centerStageX, centerStageY, R * 1.06);
    drawStackedOpticalDiscs(ctx, centerStageX, centerStageY, R, state.panel1RotationDeg);
    ctx.restore();
  }
}

function drawBacklightHalo(ctx, cx, cy, radius) {
  const intensity = state.backlightIntensity;
  const grad = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.32);
  grad.addColorStop(0, `rgba(255, 244, 214, ${0.42 * intensity})`);
  grad.addColorStop(0.72, `rgba(245, 158, 11, ${0.18 * intensity})`);
  grad.addColorStop(1, "rgba(12, 14, 20, 0)");

  ctx.save();
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 1.32, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawSingleBacklitDisc(ctx, img, cx, cy, radius, rotRad, label, labelAlpha, notchAnglesDeg = []) {
  const N = 256;
  const ctx1 = offCanvas1.getContext("2d");
  ctx1.clearRect(0, 0, N, N);
  ctx1.save();
  ctx1.translate(N / 2, N / 2);
  ctx1.rotate(rotRad);
  ctx1.drawImage(img, -N / 2, -N / 2, N, N);
  ctx1.restore();

  const imgData = ctx1.getImageData(0, 0, N, N);
  const data = imgData.data;
  const intensity = state.backlightIntensity;
  const isColor = state.colorMode === "multicolor";

  for (let i = 0; i < data.length; i += 4) {
    if (isColor) {
      // Multi-Color Bambu AMS backlit PLA rendering (preserve per-channel RGB transmission!)
      const tr = (data[i] / 255.0) * intensity;
      const tg = (data[i + 1] / 255.0) * intensity;
      const tb = (data[i + 2] / 255.0) * intensity;
      data[i] = Math.min(255, Math.round(22 + 233 * Math.pow(tr, 0.88)));
      data[i + 1] = Math.min(255, Math.round(20 + 232 * Math.pow(tg, 0.88)));
      data[i + 2] = Math.min(255, Math.round(18 + 230 * Math.pow(tb, 0.88)));
    } else {
      const t = (data[i] / 255.0) * intensity;
      // Warm ivory / incandescent backlight palette
      data[i] = Math.min(255, Math.round(38 + 225 * Math.pow(t, 0.88)));     // R
      data[i + 1] = Math.min(255, Math.round(34 + 214 * Math.pow(t, 0.92))); // G
      data[i + 2] = Math.min(255, Math.round(28 + 188 * Math.pow(t, 1.02))); // B
    }
  }
  ctx1.putImageData(imgData, 0, 0);

  // Draw clipped circle onto main canvas
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(offCanvas1, cx - radius, cy - radius, radius * 2, radius * 2);

  // Outer 3D plastic rim bevel highlight
  ctx.strokeStyle = "rgba(255, 250, 235, 0.35)";
  ctx.lineWidth = 4;
  ctx.stroke();

  // Draw inward semicircular alignment notches cut into the outer rim
  notchAnglesDeg.forEach((deg, idx) => {
    const aRad = (deg * Math.PI) / 180.0 + rotRad - Math.PI / 2;
    const nx = cx + Math.cos(aRad) * radius;
    const ny = cy + Math.sin(aRad) * radius;
    ctx.fillStyle = "#0b0e15";
    ctx.beginPath();
    ctx.arc(nx, ny, 8.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = idx === 0 ? "#f59e0b" : "#38bdf8";
    ctx.lineWidth = 2.2;
    ctx.stroke();
  });

  ctx.restore();

  // Label below disc
  if (labelAlpha > 0.05) {
    ctx.save();
    ctx.globalAlpha = labelAlpha;
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "600 13.5px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, cx, cy + radius + 28);
    ctx.restore();
  }
}

/**
 * Physically simulates light passing through BOTH stacked discs (Disc 2 stationary at 0°,
 * Disc 1 rotated by `rotDeg`).
 * - In Multi-Color Mode: Subtractive Beer-Lambert RGB filtering (per-channel product of transmittances).
 * - In Monochrome Lithophane Mode: Optical densities add via Beer-Lambert law.
 * - In Monochrome Holes Mode: Light passes through only where both Disc 1 and Disc 2 have open holes.
 */
function drawStackedOpticalDiscs(ctx, cx, cy, radius, rotDeg) {
  const N = 256;
  const ctx1 = offCanvas1.getContext("2d");
  const ctx2 = offCanvas2.getContext("2d");

  const rotRad = (rotDeg * Math.PI) / 180.0;

  // Draw rotated Disc 1 into offCanvas1
  ctx1.clearRect(0, 0, N, N);
  ctx1.save();
  ctx1.translate(N / 2, N / 2);
  ctx1.rotate(rotRad);
  ctx1.drawImage(state.disc1Img, -N / 2, -N / 2, N, N);
  ctx1.restore();

  // Draw stationary Disc 2 into offCanvas2
  ctx2.clearRect(0, 0, N, N);
  ctx2.drawImage(state.disc2Img, 0, 0, N, N);

  const d1 = ctx1.getImageData(0, 0, N, N).data;
  const imgData2 = ctx2.getImageData(0, 0, N, N);
  const d2 = imgData2.data;
  const intensity = state.backlightIntensity;
  const isColor = state.colorMode === "multicolor";

  for (let i = 0; i < d2.length; i += 4) {
    if (isColor) {
      // Subtractive per-channel RGB Beer-Lambert multiplication across rotated Disc 1 & Disc 2
      for (let ch = 0; ch < 3; ch++) {
        const t1_c = d1[i + ch] / 255.0;
        const t2_c = d2[i + ch] / 255.0;
        const comb_c = Math.min(1.0, t1_c * t2_c * 1.25 * intensity);
        d2[i + ch] = Math.min(255, Math.round(16 + 239 * Math.pow(comb_c, 0.88)));
      }
      d2[i + 3] = 255;
      continue;
    }

    // Normalized single-disc transmission in [0, 1]
    const t1 = d1[i] / 255.0;
    const t2 = d2[i] / 255.0;

    let combinedT;
    if (state.mode === "holes" || state.mode === "halftone") {
      // Direct product of aperture transmissions + subtle warm plastic glow
      combinedT = t1 * t2;
      if (combinedT > 0.22) {
        combinedT = Math.min(1.0, combinedT * 1.18);
      }
    } else {
      // Beer-Lambert Law: recover normalized density u1, u2 from single-disc transmission
      const u1 = Math.max(0.0, (1.0 - t1) / 0.55);
      const u2 = Math.max(0.0, (1.0 - t2) / 0.55);
      const combDensity = Math.min(1.0, Math.max(0.0, (u1 + u2 - 0.06) / 0.88));
      combinedT = Math.pow(1.0 - combDensity, 1.12);
    }

    const lit = Math.min(1.0, combinedT * intensity);

    // Warm realistic backlit PLA shading
    d2[i] = Math.min(255, Math.round(24 + 231 * Math.pow(lit, 0.85)));     // R
    d2[i + 1] = Math.min(255, Math.round(20 + 220 * Math.pow(lit, 0.92))); // G
    d2[i + 2] = Math.min(255, Math.round(16 + 195 * Math.pow(lit, 1.04))); // B
    d2[i + 3] = 255;
  }

  ctx2.putImageData(imgData2, 0, 0);

  // Render stacked disc onto main canvas
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(offCanvas2, cx - radius, cy - radius, radius * 2, radius * 2);

  // Draw Disc 2's TWO stationary inward notches (0° for Image 1, +effectiveAngle for Image 2)
  [0, state.effectiveAngle].forEach((deg, idx) => {
    const aRad = (deg * Math.PI) / 180.0 - Math.PI / 2;
    const nx2 = cx + Math.cos(aRad) * radius;
    const ny2 = cy + Math.sin(aRad) * radius;
    ctx.fillStyle = "#0b0e15";
    ctx.beginPath();
    ctx.arc(nx2, ny2, 9.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = idx === 0 ? "#10b981" : "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  });

  // Draw Disc 1's ONE rotating inward notch (rotates with `rotRad`)
  const notchAngle = rotRad - Math.PI / 2;
  const nx = cx + Math.cos(notchAngle) * radius;
  const ny = cy + Math.sin(notchAngle) * radius;
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(nx, ny, 6.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(255, 248, 225, 0.45)";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.restore();

  // Caption under the stacked discs
  ctx.save();
  ctx.fillStyle = "#f8fafc";
  ctx.font = "700 14px Inter, sans-serif";
  ctx.textAlign = "center";
  const sign = rotDeg > 0 ? "+" : "";
  ctx.fillText(
    `STACKED OVERLAY (Panel 1 at ${sign}${rotDeg}° over Panel 2)`,
    cx,
    cy + radius + 30
  );
  ctx.restore();
}

// ==================== DOWNLOAD STLS ====================
async function downloadStls() {
  if (!state.hasGenerated) return;
  btnDownloadStls.disabled = true;
  const origLabel = downloadBtnText.textContent;
  downloadBtnText.textContent = "Building 3D STL Meshes...";

  try {
    const fd = new FormData();
    fd.append("diameter_mm", parseFloat(diameterInput.value) || 80.0);
    fd.append("thickness_mm", parseFloat(thicknessInput.value) || 2.0);

    const res = await fetch("/api/download-stls", { method: "POST", body: fd });
    if (!res.ok) throw new Error("STL export failed");

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rotational_discs_${state.mode}_${Math.round(state.effectiveAngle)}deg_stls.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Download error:", err);
  } finally {
    btnDownloadStls.disabled = false;
    downloadBtnText.textContent = origLabel;
  }
}
