"""
AI Hairstyle Assistant — FastAPI Backend
-----------------------------------------
Endpoints:
  POST /upload-image    → Detect face shape, return recommendations
  POST /try-hairstyle   → Overlay a hairstyle illustration on the uploaded image
  POST /capture-frame   → Same as upload-image but accepts base64 webcam frames
"""

import io
import os
import json
import uuid
import base64
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hairstyle-assistant")

BASE_DIR   = Path(__file__).parent
UPLOADS    = BASE_DIR / "uploads"
OUTPUT     = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

MAX_FILE_MB   = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Hairstyle Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated try-on images
app.mount("/output", StaticFiles(directory=str(OUTPUT)), name="output")

# ---------------------------------------------------------------------------
# Hairstyle database
# ---------------------------------------------------------------------------

def load_hairstyle_db() -> dict:
    db_path = MODELS_DIR / "hairstyles.json"
    with open(db_path, "r") as f:
        return json.load(f)

HAIRSTYLE_DB = load_hairstyle_db()

# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------

mp_face_mesh      = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Key landmark indices used for measurements
# See https://mediapipe.dev/solutions/face_mesh for the landmark map
FOREHEAD_IDX    = 10     # top of forehead
CHIN_IDX        = 152    # bottom of chin
LEFT_TEMPLE_IDX = 234    # left temple / outer eye corner
RIGHT_TEMPLE_IDX= 454    # right temple
LEFT_JAW_IDX    = 172    # left jaw
RIGHT_JAW_IDX   = 397    # right jaw
LEFT_CHEEK_IDX  = 234
RIGHT_CHEEK_IDX = 454
LEFT_EYE_IDX    = 33
RIGHT_EYE_IDX   = 263
NOSE_TIP_IDX    = 4

def _lm(landmarks, idx, w, h):
    """Return pixel coordinates for a landmark index."""
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def _dist(a, b):
    """Euclidean distance between two (x,y) points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def detect_face_shape(image_bgr: np.ndarray) -> dict:
    """
    Run MediaPipe Face Mesh on a BGR image, extract key measurements,
    and classify the face shape.

    Returns a dict with face_shape, confidence, and raw measurements.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None  # caller handles "no face" case

    landmarks = results.multi_face_landmarks[0].landmark

    # --- Key points --------------------------------------------------------
    forehead  = _lm(landmarks, FOREHEAD_IDX, w, h)
    chin      = _lm(landmarks, CHIN_IDX, w, h)
    l_temple  = _lm(landmarks, LEFT_TEMPLE_IDX, w, h)
    r_temple  = _lm(landmarks, RIGHT_TEMPLE_IDX, w, h)
    l_jaw     = _lm(landmarks, LEFT_JAW_IDX, w, h)
    r_jaw     = _lm(landmarks, RIGHT_JAW_IDX, w, h)
    l_cheek   = _lm(landmarks, LEFT_CHEEK_IDX, w, h)
    r_cheek   = _lm(landmarks, RIGHT_CHEEK_IDX, w, h)

    # --- Measurements ------------------------------------------------------
    face_length      = _dist(forehead, chin)
    forehead_width   = _dist(l_temple, r_temple)
    jaw_width        = _dist(l_jaw, r_jaw)
    cheekbone_width  = _dist(l_cheek, r_cheek)

    # Avoid division by zero
    if face_length == 0 or cheekbone_width == 0:
        return None

    length_to_cheek  = face_length / cheekbone_width
    jaw_to_cheek     = jaw_width   / cheekbone_width
    fore_to_cheek    = forehead_width / cheekbone_width

    # --- Classification heuristics -----------------------------------------
    # These thresholds are empirically tuned for typical selfie proportions.
    face_shape = "Oval"  # default fallback

    if length_to_cheek >= 1.5:
        face_shape = "Long"
    elif jaw_to_cheek >= 0.9 and fore_to_cheek >= 0.9:
        face_shape = "Square"
    elif jaw_to_cheek <= 0.75 and fore_to_cheek >= 0.85:
        face_shape = "Heart"
    elif jaw_to_cheek <= 0.75 and fore_to_cheek <= 0.75:
        face_shape = "Diamond"
    elif length_to_cheek <= 1.1:
        face_shape = "Round"
    else:
        face_shape = "Oval"

    return {
        "face_shape": face_shape,
        "measurements": {
            "face_length":      round(face_length, 1),
            "forehead_width":   round(forehead_width, 1),
            "jaw_width":        round(jaw_width, 1),
            "cheekbone_width":  round(cheekbone_width, 1),
        },
        "ratios": {
            "length_to_cheek": round(length_to_cheek, 3),
            "jaw_to_cheek":    round(jaw_to_cheek, 3),
            "fore_to_cheek":   round(fore_to_cheek, 3),
        },
        "landmarks_used": {
            "forehead":  forehead,
            "chin":      chin,
            "l_temple":  l_temple,
            "r_temple":  r_temple,
            "l_jaw":     l_jaw,
            "r_jaw":     r_jaw,
        },
    }


def get_face_bounding_box(image_bgr: np.ndarray) -> Optional[dict]:
    """
    Use MediaPipe Face Detection (faster, less detailed) to get a
    bounding box around the face for overlay positioning.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as detector:
        results = detector.process(image_rgb)

    if not results.detections:
        return None

    det   = results.detections[0]
    bbox  = det.location_data.relative_bounding_box
    x     = int(bbox.xmin * w)
    y     = int(bbox.ymin * h)
    bw    = int(bbox.width * w)
    bh    = int(bbox.height * h)

    return {"x": x, "y": y, "w": bw, "h": bh, "img_w": w, "img_h": h}


# ---------------------------------------------------------------------------
# Hairstyle overlay renderer (uses Pillow + procedural SVG-like drawing)
# ---------------------------------------------------------------------------

# Maps overlay keys to a drawing function.
# Real production apps would load transparent PNGs here;
# we generate stylised vector-like illustrations with Pillow.

def _draw_overlay(draw: ImageDraw.ImageDraw, bbox: dict, style: str, color=(40, 30, 20)):
    """
    Draw a hairstyle overlay onto a Pillow ImageDraw canvas.
    bbox: {x, y, w, h} face bounding box in pixels
    style: overlay key from hairstyles.json
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = x + w // 2       # face centre x
    top = y               # top of face bounding box
    pad = int(w * 0.1)    # small padding

    c = tuple(color) + (210,)   # RGBA with transparency

    # Each branch draws a simplified hairstyle silhouette.
    # The shapes are intentionally stylised / iconic rather than photo-realistic.

    if style in ("fade", "buzz", "low_fade"):
        # Very short — just a thin cap
        draw.ellipse([cx - w//2 - pad, top - int(h*0.08), cx + w//2 + pad, top + int(h*0.12)], fill=c)

    elif style in ("high_fade", "mohawk"):
        # Tall narrow strip on top
        strip_w = w // 4
        draw.rectangle([cx - strip_w, top - int(h*0.35), cx + strip_w, top + int(h*0.05)], fill=c)

    elif style in ("quiff", "messy_quiff", "pompadour", "full_volume"):
        # Swept-up volume blob
        draw.ellipse([cx - w//2, top - int(h*0.45), cx + w//2, top + int(h*0.1)], fill=c)
        # Extra swept section
        draw.ellipse([cx - int(w*0.3), top - int(h*0.55), cx + int(w*0.55), top + int(h*0.05)], fill=c)

    elif style in ("side_part", "side_volume", "undercut"):
        # Offset volume cap
        draw.ellipse([cx - int(w*0.6), top - int(h*0.35), cx + int(w*0.4), top + int(h*0.1)], fill=c)

    elif style in ("fringe", "soft_fringe", "blunt_fringe", "curtain", "side_swept"):
        # Cap + fringe strip across forehead
        draw.ellipse([cx - w//2 - pad, top - int(h*0.25), cx + w//2 + pad, top + int(h*0.05)], fill=c)
        fringe_y = top + int(h * 0.1)
        draw.rectangle([cx - w//2 - pad, top, cx + w//2 + pad, fringe_y], fill=c)

    elif style in ("bob", "layered_medium"):
        # Cap + ear-length falls on the sides
        draw.ellipse([cx - w//2 - pad, top - int(h*0.2), cx + w//2 + pad, top + int(h*0.05)], fill=c)
        fall_h = int(h * 0.45)
        draw.rectangle([cx - w//2 - pad, top, cx - w//2 + pad*2, top + fall_h], fill=c)
        draw.rectangle([cx + w//2 - pad*2, top, cx + w//2 + pad, top + fall_h], fill=c)

    elif style in ("waves", "shaggy", "layered_medium"):
        # Wavy medium length — cap + side falls with wavy bottom
        draw.ellipse([cx - w//2 - pad, top - int(h*0.25), cx + w//2 + pad, top + int(h*0.05)], fill=c)
        fall_h = int(h * 0.6)
        for side, sign in [(-1, -1), (1, 1)]:
            ox = cx + sign * (w//2 + pad//2)
            for i in range(0, fall_h, 12):
                wave_x = ox + sign * int(8 * math.sin(i / 10))
                draw.ellipse([wave_x - 10, top + i, wave_x + 10, top + i + 14], fill=c)

    elif style == "afro":
        # Large circular puff
        draw.ellipse(
            [cx - int(w*0.75), top - int(h*0.7), cx + int(w*0.75), top + int(h*0.25)],
            fill=c
        )

    elif style == "crop":
        # Flat top textured crop
        draw.rectangle([cx - w//2 - pad, top - int(h*0.15), cx + w//2 + pad, top + int(h*0.05)], fill=c)

    else:
        # Generic fallback cap
        draw.ellipse([cx - w//2 - pad, top - int(h*0.3), cx + w//2 + pad, top + int(h*0.1)], fill=c)


def apply_hairstyle_overlay(
    image_bgr: np.ndarray,
    hairstyle_name: str,
    output_path: Path,
    hair_color=(40, 30, 20),
) -> str:
    """
    Composite a hairstyle illustration onto the user's photo.
    Returns the filename of the saved output image.
    """
    bbox = get_face_bounding_box(image_bgr)
    if bbox is None:
        raise ValueError("No face detected for try-on overlay.")

    # Find the overlay key for this hairstyle name
    overlay_key = None
    for shape_list in HAIRSTYLE_DB.values():
        for style in shape_list:
            if style["name"].lower() == hairstyle_name.lower():
                overlay_key = style.get("overlay", "generic")
                break
        if overlay_key:
            break

    if not overlay_key:
        overlay_key = "generic"

    # Convert to PIL RGBA
    pil_img    = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay    = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw       = ImageDraw.Draw(overlay)

    _draw_overlay(draw, bbox, overlay_key, color=hair_color)

    # Soften overlay edges slightly
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=6))

    # Composite
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")

    fname = f"tryon_{uuid.uuid4().hex[:8]}.jpg"
    result.save(output_path / fname, quality=92)
    return fname


# ---------------------------------------------------------------------------
# Image validation helpers
# ---------------------------------------------------------------------------

def validate_image(file: UploadFile) -> bytes:
    """Read, size-check, and return raw bytes of an uploaded image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG or WebP.",
        )
    data = file.file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_FILE_MB} MB limit.",
        )
    return data


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes → OpenCV BGR array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid photo.")
    return img


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "AI Hairstyle Assistant is running"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept a photo, detect the face shape, return hairstyle recommendations.
    """
    data = validate_image(file)
    img_bgr = bytes_to_bgr(data)

    # Save upload
    fname = f"{uuid.uuid4().hex}.jpg"
    saved = UPLOADS / fname
    cv2.imwrite(str(saved), img_bgr)
    logger.info(f"Saved upload: {fname}")

    # Detect face shape
    result = detect_face_shape(img_bgr)
    if result is None:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "No face detected. Please upload a clear, well-lit photo where your face is visible.",
            },
        )

    face_shape = result["face_shape"]
    recommendations = HAIRSTYLE_DB.get(face_shape, [])

    return {
        "status":          "success",
        "face_shape":      face_shape,
        "measurements":    result["measurements"],
        "recommendations": recommendations,
        "uploaded_file":   fname,
    }


@app.post("/try-hairstyle")
async def try_hairstyle(
    file:           UploadFile = File(...),
    hairstyle_name: str        = Form(...),
    hair_color_r:   int        = Form(40),
    hair_color_g:   int        = Form(30),
    hair_color_b:   int        = Form(20),
):
    """
    Composite a hairstyle illustration onto the uploaded photo.
    Returns the URL of the generated image.
    """
    data    = validate_image(file)
    img_bgr = bytes_to_bgr(data)

    color = (
        max(0, min(255, hair_color_r)),
        max(0, min(255, hair_color_g)),
        max(0, min(255, hair_color_b)),
    )

    try:
        out_fname = apply_hairstyle_overlay(img_bgr, hairstyle_name, OUTPUT, hair_color=color)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)},
        )

    return {
        "status":    "success",
        "image_url": f"/output/{out_fname}",
        "filename":  out_fname,
    }


@app.post("/capture-frame")
async def capture_frame(payload: dict):
    """
    Accept a base64-encoded webcam frame, detect face shape,
    return recommendations — same response shape as /upload-image.
    """
    b64 = payload.get("image_data", "")
    if not b64:
        raise HTTPException(status_code=400, detail="No image_data field in request body.")

    # Strip data-URL prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    img_bgr = bytes_to_bgr(raw)

    result = detect_face_shape(img_bgr)
    if result is None:
        return JSONResponse(
            status_code=422,
            content={
                "status":  "error",
                "message": "No face detected in camera frame. Centre your face and try again.",
            },
        )

    face_shape = result["face_shape"]
    return {
        "status":          "success",
        "face_shape":      face_shape,
        "measurements":    result["measurements"],
        "recommendations": HAIRSTYLE_DB.get(face_shape, []),
    }
