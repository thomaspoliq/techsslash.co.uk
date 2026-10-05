/**
 * AI Hairstyle Assistant — Frontend Script
 * -----------------------------------------
 * Responsibilities:
 *   1. File upload + drag-and-drop
 *   2. Image preview
 *   3. POST to /upload-image → render face shape + recommendations
 *   4. Try-on: POST to /try-hairstyle → show result image
 *   5. Live camera: getUserMedia → capture → POST to /capture-frame
 *   6. Error handling / toast messages
 */

"use strict";

/* ── Config ──────────────────────────────────────────────────── */
const API_BASE = "http://localhost:8000";

/* ── State ───────────────────────────────────────────────────── */
let currentFile      = null;   // the File object the user uploaded
let currentFileBlob  = null;   // same but kept for try-on re-use
let hairColor        = { r: 220, g: 170, b: 80 };  // default: Blonde
let cameraStream     = null;

/* ── DOM refs ────────────────────────────────────────────────── */
const fileInput        = document.getElementById("file-input");
const dropZone         = document.getElementById("drop-zone");
const browseBtn        = document.getElementById("browse-btn");
const previewWrap      = document.getElementById("preview-wrap");
const previewImg       = document.getElementById("preview-img");
const previewScan      = document.getElementById("preview-scan");
const analyseBtn       = document.getElementById("analyse-btn");
const changePhotoBtn   = document.getElementById("change-photo-btn");
const colourPicker     = document.getElementById("colour-picker");
const swatches         = document.querySelectorAll(".swatch");

const heroUploadBtn    = document.getElementById("hero-upload-btn");
const heroCameraBtn    = document.getElementById("hero-camera-btn");
const navCameraBtn     = document.getElementById("nav-camera-btn");

const cameraSection    = document.getElementById("camera-section");
const cameraVideo      = document.getElementById("camera-video");
const cameraCanvas     = document.getElementById("camera-canvas");
const captureBtn       = document.getElementById("capture-btn");
const closeCameraBtn   = document.getElementById("close-camera-btn");

const processingOverlay = document.getElementById("processing-overlay");
const processingLabel   = document.getElementById("processing-label");

const resultsSection   = document.getElementById("results-section");
const faceShapeName    = document.getElementById("face-shape-name");
const measurementsGrid = document.getElementById("measurements-grid");
const recsSubtitle     = document.getElementById("recs-sub");
const cardsGrid        = document.getElementById("cards-grid");

const tryonResult      = document.getElementById("tryon-result");
const tryonOriginalImg = document.getElementById("tryon-original-img");
const tryonResultImg   = document.getElementById("tryon-result-img");
const tryonDownloadBtn = document.getElementById("tryon-download-btn");

const toast            = document.getElementById("toast");
const toastMsg         = document.getElementById("toast-msg");

document.getElementById("footer-year").textContent = new Date().getFullYear();

/* ── Hairstyle art icons (emoji stand-ins rendered in cards) ── */
const STYLE_ICONS = {
  "Classic Fade":         "✂️",
  "Textured Quiff":       "💈",
  "Side Part":            "🪮",
  "Buzz Cut":             "⚡",
  "High Fade with Volume":"🔝",
  "Angular Fringe":       "📐",
  "Pompadour":            "🎸",
  "Mohawk Fade":          "⚔️",
  "Textured Crop":        "🌿",
  "Layered Waves":        "🌊",
  "Undercut":             "🗡️",
  "Messy Quiff":          "🍃",
  "Side Swept Bangs":     "💫",
  "Chin-Length Bob":      "🎭",
  "Low Fade":             "🎯",
  "Curtain Bangs":        "🪟",
  "Soft Fringe":          "🌸",
  "Full Volume Top":      "🎆",
  "Shaggy Layers":        "🌀",
  "Side Part with Volume":"📊",
  "Curly Afro":           "☁️",
  "Blunt Fringe":         "📏",
  "Layered Medium Cut":   "🍂",
};

/* ================================================================
   UPLOAD SYSTEM
   ================================================================ */

heroUploadBtn.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click",     () => fileInput.click());
changePhotoBtn.addEventListener("click", resetToUpload);

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

/* Drag and drop */
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener("click", (e) => {
  // Avoid double-trigger when Browse button inside is clicked
  if (e.target !== browseBtn) fileInput.click();
});

/**
 * Validate and preview a selected image file.
 */
function handleFile(file) {
  // Validate type
  const allowed = ["image/jpeg", "image/png", "image/webp", "image/gif"];
  if (!allowed.includes(file.type)) {
    showToast("Please upload a JPEG, PNG or WebP image.");
    return;
  }
  // Validate size (10 MB)
  if (file.size > 10 * 1024 * 1024) {
    showToast("Image is too large. Maximum size is 10 MB.");
    return;
  }

  currentFile     = file;
  currentFileBlob = file;

  const url = URL.createObjectURL(file);
  previewImg.src = url;

  // Show preview, hide drop zone
  dropZone.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  colourPicker.classList.remove("hidden");

  // Scroll down smoothly
  previewWrap.scrollIntoView({ behavior: "smooth", block: "center" });

  // Reset previous results
  resultsSection.classList.add("hidden");
  tryonResult.classList.add("hidden");
}

function resetToUpload() {
  currentFile = null;
  previewImg.src = "";
  fileInput.value = "";
  dropZone.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  colourPicker.classList.add("hidden");
  resultsSection.classList.add("hidden");
  tryonResult.classList.add("hidden");
}

/* ================================================================
   COLOUR PICKER
   ================================================================ */

swatches.forEach((swatch) => {
  swatch.addEventListener("click", () => {
    swatches.forEach((s) => s.classList.remove("active"));
    swatch.classList.add("active");
    hairColor = {
      r: parseInt(swatch.dataset.r),
      g: parseInt(swatch.dataset.g),
      b: parseInt(swatch.dataset.b),
    };
  });
});

/* ================================================================
   FACE ANALYSIS (POST /upload-image)
   ================================================================ */

analyseBtn.addEventListener("click", async () => {
  if (!currentFile) {
    showToast("Please select a photo first.");
    return;
  }
  await runFaceAnalysis(currentFile);
});

/**
 * Send image to backend, receive face shape + recommendations.
 * @param {File|Blob} imageSource
 */
async function runFaceAnalysis(imageSource) {
  showProcessing("Detecting face geometry…");

  // Animate scan line on preview
  previewScan.classList.add("running");

  const formData = new FormData();
  formData.append("file", imageSource, "photo.jpg");

  let data;
  try {
    const response = await fetch(`${API_BASE}/upload-image`, {
      method: "POST",
      body: formData,
    });
    data = await response.json();
  } catch (err) {
    hideProcessing();
    showToast("Could not reach the server. Make sure the backend is running on port 8000.");
    console.error(err);
    return;
  } finally {
    previewScan.classList.remove("running");
  }

  hideProcessing();

  if (data.status === "error" || !data.face_shape) {
    showToast(data.message || "No face detected. Please try a clearer photo.");
    return;
  }

  renderResults(data);
}

/* ================================================================
   RENDER RESULTS
   ================================================================ */

function renderResults(data) {
  const { face_shape, measurements, recommendations } = data;

  // Face shape
  faceShapeName.textContent = face_shape;

  // Measurements
  measurementsGrid.innerHTML = "";
  const mLabels = {
    face_length:     "Face Length",
    forehead_width:  "Forehead",
    jaw_width:       "Jaw",
    cheekbone_width: "Cheekbones",
  };
  Object.entries(measurements || {}).forEach(([key, val]) => {
    const item = document.createElement("div");
    item.className = "measurement-item";
    item.innerHTML = `
      <div class="measurement-value">${Math.round(val)}px</div>
      <div class="measurement-label">${mLabels[key] || key}</div>
    `;
    measurementsGrid.appendChild(item);
  });

  // Subtitle
  recsSubtitle.textContent = `${recommendations.length} styles matched to your ${face_shape} face shape.`;

  // Cards
  cardsGrid.innerHTML = "";
  recommendations.forEach((style, idx) => {
    const card = buildStyleCard(style, idx);
    cardsGrid.appendChild(card);
  });

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

/**
 * Build a hairstyle recommendation card element.
 */
function buildStyleCard(style, idx) {
  const card = document.createElement("div");
  card.className = "style-card";
  card.style.animationDelay = `${idx * 0.08}s`;

  const icon  = STYLE_ICONS[style.name] || "✂️";
  const tags  = (style.tags || []).map((t) => `<span class="tag">${t}</span>`).join("");

  card.innerHTML = `
    <div class="style-card-art">${icon}</div>
    <div class="style-card-body">
      <div class="style-card-name">${style.name}</div>
      <p class="style-card-desc">${style.description}</p>
      <div class="style-card-tags">${tags}</div>
      <button class="try-btn" data-name="${style.name}">Try This Style →</button>
    </div>
  `;

  card.querySelector(".try-btn").addEventListener("click", () => {
    handleTryOn(style.name, card.querySelector(".try-btn"));
  });

  return card;
}

/* ================================================================
   TRY-ON (POST /try-hairstyle)
   ================================================================ */

async function handleTryOn(hairstyleName, btn) {
  if (!currentFileBlob) {
    showToast("Upload a photo first to try on styles.");
    return;
  }

  // Disable all try buttons while request is in flight
  document.querySelectorAll(".try-btn").forEach((b) => (b.disabled = true));
  btn.textContent = "Generating…";

  showProcessing(`Placing "${hairstyleName}" on your photo…`);

  const formData = new FormData();
  formData.append("file", currentFileBlob, "photo.jpg");
  formData.append("hairstyle_name", hairstyleName);
  formData.append("hair_color_r", hairColor.r);
  formData.append("hair_color_g", hairColor.g);
  formData.append("hair_color_b", hairColor.b);

  let data;
  try {
    const response = await fetch(`${API_BASE}/try-hairstyle`, {
      method: "POST",
      body: formData,
    });
    data = await response.json();
  } catch (err) {
    hideProcessing();
    showToast("Try-on request failed. Check the server.");
    console.error(err);
    document.querySelectorAll(".try-btn").forEach((b) => (b.disabled = false));
    btn.textContent = "Try This Style →";
    return;
  }

  hideProcessing();
  document.querySelectorAll(".try-btn").forEach((b) => (b.disabled = false));
  btn.textContent = "Try This Style →";

  if (data.status === "error") {
    showToast(data.message || "Try-on failed.");
    return;
  }

  // Show result
  tryonOriginalImg.src = previewImg.src;
  tryonResultImg.src   = `${API_BASE}${data.image_url}`;
  tryonDownloadBtn.href = `${API_BASE}${data.image_url}`;
  tryonResult.classList.remove("hidden");
  tryonResult.scrollIntoView({ behavior: "smooth" });
}

/* ================================================================
   LIVE CAMERA
   ================================================================ */

heroCameraBtn.addEventListener("click",  openCamera);
navCameraBtn.addEventListener("click",   (e) => { e.preventDefault(); openCamera(); });
closeCameraBtn.addEventListener("click", closeCamera);
captureBtn.addEventListener("click",     captureAndAnalyse);

async function openCamera() {
  // Scroll camera section into view first
  cameraSection.classList.remove("hidden");
  cameraSection.scrollIntoView({ behavior: "smooth" });

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
  } catch (err) {
    showToast("Camera access denied. Please allow camera permissions and try again.");
    cameraSection.classList.add("hidden");
    console.error(err);
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraSection.classList.add("hidden");
}

async function captureAndAnalyse() {
  if (!cameraStream) return;

  // Draw current video frame to canvas
  const vw = cameraVideo.videoWidth;
  const vh = cameraVideo.videoHeight;
  cameraCanvas.width  = vw;
  cameraCanvas.height = vh;

  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0, vw, vh);

  // Get base64 data URL
  const dataUrl = cameraCanvas.toDataURL("image/jpeg", 0.92);

  closeCamera();
  showProcessing("Analysing captured frame…");

  let data;
  try {
    const response = await fetch(`${API_BASE}/capture-frame`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image_data: dataUrl }),
    });
    data = await response.json();
  } catch (err) {
    hideProcessing();
    showToast("Frame analysis failed. Make sure the backend is running.");
    console.error(err);
    return;
  }

  hideProcessing();

  if (data.status === "error") {
    showToast(data.message || "No face detected in the captured frame.");
    return;
  }

  // Convert the base64 frame to a File so try-on works too
  const blob = dataURLToBlob(dataUrl);
  currentFileBlob = blob;
  previewImg.src  = dataUrl;
  previewWrap.classList.remove("hidden");
  dropZone.classList.add("hidden");
  colourPicker.classList.remove("hidden");

  renderResults(data);
}

/** Convert a data URL to a Blob for FormData use. */
function dataURLToBlob(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime             = header.match(/:(.*?);/)[1];
  const binary           = atob(base64);
  const bytes            = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

/* ================================================================
   UI HELPERS
   ================================================================ */

const PROCESSING_MESSAGES = [
  "Mapping face landmarks…",
  "Calculating proportions…",
  "Matching hairstyle database…",
  "Almost done…",
];

let processingTimer = null;

function showProcessing(firstMsg) {
  processingLabel.textContent = firstMsg || PROCESSING_MESSAGES[0];
  processingOverlay.classList.remove("hidden");

  let step = 0;
  processingTimer = setInterval(() => {
    step = (step + 1) % PROCESSING_MESSAGES.length;
    processingLabel.textContent = PROCESSING_MESSAGES[step];
  }, 1400);
}

function hideProcessing() {
  clearInterval(processingTimer);
  processingOverlay.classList.add("hidden");
}

let toastTimer = null;

function showToast(msg) {
  toastMsg.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 5000);
}
