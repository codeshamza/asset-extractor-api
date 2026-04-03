"""
Visual Asset Extractor API
FastAPI backend using Grounding DINO for text-prompted object detection
+ flood-fill background removal + 4x Lanczos upscale.
"""
import os
import io
import base64
import tempfile
import zipfile
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Visual Asset Extractor API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Netlify frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ────────────────────────────────────────────
print("Loading Grounding DINO Tiny...", flush=True)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

gd_processor = AutoProcessor.from_pretrained(MODEL_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
gd_model.eval()

model_size_mb = sum(p.numel() * p.element_size() for p in gd_model.parameters()) / 1e6
print(f"Grounding DINO loaded on {device} ({model_size_mb:.0f} MB)", flush=True)

# ── Detection Concepts ───────────────────────────────────────
# Two-pass: parent (composite) + child (individual) elements
# No text, no logo — video-editing focused
PARENT_CONCEPTS = [
    "chart", "diagram", "graph", "table", "illustration",
    "infographic", "figure", "photo", "picture",
]
CHILD_CONCEPTS = [
    "icon", "symbol", "arrow", "bar", "person",
    "object", "button", "badge", "circle",
]
# Grounding DINO uses a single text prompt with "." separator
ALL_CONCEPTS_TEXT = " . ".join(PARENT_CONCEPTS + CHILD_CONCEPTS) + " ."

# ── Utility Functions ────────────────────────────────────────

def box_iou(b1, b2):
    """IoU between two boxes [x0, y0, x1, y1]."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def is_notebooklm_logo(box, img_w, img_h):
    """Filter small detections in bottom-right corner (NotebookLM watermark)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw < 80 and bh < 80:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if cx > img_w * 0.85 and cy > img_h * 0.85:
            return True
    return False


def remove_color_bg(crop_rgb: np.ndarray, bg_color=(255, 255, 255), tolerance=45) -> np.ndarray:
    """Remove background by flood-filling from edges with strong anti-aliasing.
    
    Improvements:
    - Auto-detects actual BG color from border pixels if specified BG doesn't match
    - Higher tolerance for gradient backgrounds  
    - Multi-pass Gaussian feathering for smooth edges
    - Morphological refinement to push transition inward
    """
    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = crop_rgb
        rgba[:, :, 3] = 255
        return rgba

    # Auto-detect BG from border pixels (more reliable than user input)
    border_pixels = []
    border_pixels.extend(crop_rgb[0, :].tolist())        # top row
    border_pixels.extend(crop_rgb[h-1, :].tolist())      # bottom row
    border_pixels.extend(crop_rgb[:, 0].tolist())         # left col
    border_pixels.extend(crop_rgb[:, w-1].tolist())       # right col
    border_arr = np.array(border_pixels, dtype=np.float32)
    detected_bg = np.median(border_arr, axis=0).astype(np.float32)
    
    # Use auto-detected BG if it's close to the specified BG, otherwise use detected
    specified_bg = np.array(bg_color, dtype=np.float32)
    bg_diff = np.sqrt(np.sum((detected_bg - specified_bg) ** 2))
    if bg_diff > 80:
        # Detected BG is very different from specified — use detected
        bg = detected_bg
        print(f"    BG auto-detect: specified={bg_color} detected={tuple(detected_bg.astype(int))} — using detected", flush=True)
    else:
        bg = specified_bg

    # Color distance from BG
    diff = np.sqrt(np.sum((crop_rgb.astype(np.float32) - bg) ** 2, axis=2))
    color_match = (diff < tolerance).astype(np.uint8) * 255

    # Flood fill from border pixels to find CONNECTED background
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bg_connected = np.zeros((h, w), dtype=np.uint8)

    # Collect border seeds (sample every 2 px for speed)
    border_seeds = set()
    for x in range(0, w, 2):
        if color_match[0, x]: border_seeds.add((x, 0))
        if color_match[h - 1, x]: border_seeds.add((x, h - 1))
    for y in range(0, h, 2):
        if color_match[y, 0]: border_seeds.add((0, y))
        if color_match[y, w - 1]: border_seeds.add((w - 1, y))

    for sx, sy in border_seeds:
        if bg_connected[sy, sx] == 0 and color_match[sy, sx]:
            flood_mask[:] = 0
            cv2.floodFill(
                color_match.copy(), flood_mask, (sx, sy), 128,
                loDiff=0, upDiff=0,
                flags=cv2.FLOODFILL_MASK_ONLY | (8 << 8),
            )
            bg_connected |= flood_mask[1:-1, 1:-1]

    # Raw alpha: 255 for foreground, 0 for background
    alpha_raw = np.where(bg_connected > 0, np.uint8(0), np.uint8(255))

    # ── Strong Anti-Aliasing ──
    # Step 1: Morphological erosion on BG to push edge inward by 1-2px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bg_eroded = cv2.erode(bg_connected, kernel, iterations=1)
    
    # Step 2: Create soft transition zone
    # Edge zone = pixels that were BG but NOT in eroded BG (the 1-2px border)
    alpha_f = alpha_raw.astype(np.float32)
    
    # Step 3: Multi-pass Gaussian blur for smooth feathering
    alpha_blur = cv2.GaussianBlur(alpha_f, (7, 7), sigmaX=1.5)
    alpha_blur2 = cv2.GaussianBlur(alpha_f, (5, 5), sigmaX=1.0)
    
    # Combine: use hard alpha for clear interior, blurred for edges
    interior = alpha_raw == 255
    exterior = bg_eroded > 0
    # Edge zone: pixels near the boundary
    edge_zone = ~interior & ~exterior
    
    alpha_final = np.zeros_like(alpha_f)
    alpha_final[interior] = 255.0
    alpha_final[exterior] = 0.0
    # Smooth blend in the edge zone
    alpha_final[edge_zone] = (alpha_blur[edge_zone] * 0.6 + alpha_blur2[edge_zone] * 0.4)
    
    # Extra pass: smooth the entire alpha to remove any remaining steps
    alpha_final = cv2.GaussianBlur(alpha_final, (3, 3), sigmaX=0.7)
    # Re-clamp interior to prevent bleeding
    alpha_final[alpha_raw == 255] = np.maximum(alpha_final[alpha_raw == 255], 250.0)
    alpha_final[bg_eroded > 0] = np.minimum(alpha_final[bg_eroded > 0], 5.0)
    
    alpha = alpha_final.clip(0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = alpha
    return rgba


def upscale_4x(rgba: np.ndarray) -> np.ndarray:
    """Smart upscale with Lanczos + unsharp masking. Caps output at 2048px max dimension."""
    h, w = rgba.shape[:2]
    max_dim = max(h, w)

    # Choose scale factor to keep output under 2048px
    if max_dim * 4 <= 2048:
        scale = 4
    elif max_dim * 2 <= 2048:
        scale = 2
    else:
        scale = max(1, 2048 / max_dim)

    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= w and new_h <= h:
        return rgba  # Already large enough

    upscaled = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask on RGB only
    rgb = upscaled[:, :, :3]
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.0)
    rgb_sharp = cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)
    upscaled[:, :, :3] = rgb_sharp
    return upscaled


def rgba_to_base64_png(rgba: np.ndarray) -> str:
    """Convert RGBA numpy array to base64 PNG string."""
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def detect_and_extract(image_rgb: np.ndarray, bg_color=(255, 255, 255), tolerance=30):
    """Run Grounding DINO detection → flood-fill BG removal → upscale.
    
    Returns list of base64 PNG strings.
    """
    h, w = image_rgb.shape[:2]
    img_area = h * w
    pil_img = Image.fromarray(image_rgb)

    # Run Grounding DINO
    inputs = gd_processor(images=pil_img, text=ALL_CONCEPTS_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    # Post-process: get boxes and scores above threshold
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.20,
        text_threshold=0.20,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()  # [N, 4] as x0,y0,x1,y1
    scores = results["scores"].cpu().numpy()
    labels = results.get("labels", results.get("text_labels", [""] * len(boxes)))

    print(f"  Grounding DINO: {len(boxes)} raw detections", flush=True)

    # Filter detections
    kept_boxes = []
    kept_scores = []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        bw, bh = x1 - x0, y1 - y0
        box_area = bw * bh
        score = float(scores[i])

        if score < 0.20:
            continue
        if box_area < 500 or bw < 20 or bh < 20:
            continue
        if box_area > img_area * 0.90:
            continue
        if is_notebooklm_logo([x0, y0, x1, y1], w, h):
            print(f"    [{i}] SKIP NotebookLM logo", flush=True)
            continue

        # Add padding (8%)
        pad_x = max(8, int(bw * 0.08))
        pad_y = max(8, int(bh * 0.08))
        bx0 = max(0, x0 - pad_x)
        by0 = max(0, y0 - pad_y)
        bx1 = min(w, x1 + pad_x)
        by1 = min(h, y1 + pad_y)

        kept_boxes.append([bx0, by0, bx1, by1])
        kept_scores.append(score)
        print(f"    [{i}] KEPT: {labels[i]} score={score:.3f} box=[{bx0},{by0},{bx1},{by1}]", flush=True)

    if not kept_boxes:
        return []

    # Deduplicate by box IoU
    order = sorted(range(len(kept_boxes)), key=lambda i: kept_scores[i], reverse=True)
    keep = []
    for i in order:
        dup = False
        for ki in keep:
            if box_iou(kept_boxes[i], kept_boxes[ki]) > 0.5:
                dup = True
                break
        if not dup:
            keep.append(i)

    print(f"  After dedup: {len(keep)} unique assets", flush=True)

    # Crop → flood-fill BG removal → upscale → base64
    results_b64 = []
    for idx, ki in enumerate(keep):
        bx0, by0, bx1, by1 = kept_boxes[ki]
        crop_rgb = image_rgb[by0:by1, bx0:bx1]

        rgba = remove_color_bg(crop_rgb, bg_color=bg_color, tolerance=tolerance)
        rgba = upscale_4x(rgba)

        b64 = rgba_to_base64_png(rgba)
        results_b64.append(b64)
        print(f"    asset[{idx}] done ({rgba.shape[1]}x{rgba.shape[0]})", flush=True)

    return results_b64


# ── API Endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    bg_color: str = Form("#FFFFFF"),
    tolerance: int = Form(30),
):
    """Extract visual assets from a single image."""
    try:
        # Parse bg color
        bg_hex = bg_color.lstrip("#")
        try:
            bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            bg_rgb = (255, 255, 255)

        # Read image
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)

        print(f">>> /extract: {img_np.shape[1]}x{img_np.shape[0]}, bg={bg_rgb}", flush=True)

        assets = detect_and_extract(img_np, bg_color=bg_rgb, tolerance=tolerance)

        print(f">>> Returning {len(assets)} assets", flush=True)
        return JSONResponse({"assets": assets, "count": len(assets)})

    except Exception as e:
        print(f">>> ERROR in /extract: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-pdf")
async def extract_pdf(
    pdf: UploadFile = File(...),
    bg_color: str = Form("#FFFFFF"),
    tolerance: int = Form(30),
):
    """Extract visual assets from every page of a PDF."""
    try:
        import fitz

        bg_hex = bg_color.lstrip("#")
        try:
            bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            bg_rgb = (255, 255, 255)

        contents = await pdf.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        total_pages = len(doc)
        print(f">>> /extract-pdf: {total_pages} pages, bg={bg_rgb}", flush=True)

        all_assets = []
        for page_num in range(total_pages):
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)  # 144 DPI
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img_rgb = img_array[:, :, :3].copy()
            else:
                img_rgb = img_array.copy()

            page_assets = detect_and_extract(img_rgb, bg_color=bg_rgb, tolerance=tolerance)
            all_assets.extend(page_assets)
            print(f"  Page {page_num + 1}/{total_pages}: {len(page_assets)} assets", flush=True)

        doc.close()
        print(f">>> PDF complete: {len(all_assets)} total assets", flush=True)
        return JSONResponse({"assets": all_assets, "count": len(all_assets)})

    except Exception as e:
        print(f">>> ERROR in /extract-pdf: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download-zip")
async def download_zip(
    image: UploadFile = File(...),
    bg_color: str = Form("#FFFFFF"),
    tolerance: int = Form(30),
):
    """Extract assets and return as a ZIP file."""
    try:
        bg_hex = bg_color.lstrip("#")
        try:
            bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            bg_rgb = (255, 255, 255)

        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)

        assets_b64 = detect_and_extract(img_np, bg_color=bg_rgb, tolerance=tolerance)

        # Build ZIP in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, b64 in enumerate(assets_b64):
                png_bytes = base64.b64decode(b64)
                zf.writestr(f"asset_{i+1:04d}.png", png_bytes)

        zip_buf.seek(0)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=extracted_assets.zip"},
        )

    except Exception as e:
        print(f">>> ERROR in /download-zip: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
