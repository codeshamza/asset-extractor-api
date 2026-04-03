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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Visual Asset Extractor API", version="3.1")

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

print("Loading Meta SAM (Segment Anything)...", flush=True)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_model.eval()

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


def trim_transparent(rgba: np.ndarray) -> np.ndarray:
    """Trim fully transparent borders from an RGBA image so the object perfectly fits the rect."""
    alpha = rgba[:, :, 3]
    y_non_zero, x_non_zero = np.nonzero(alpha)
    if len(y_non_zero) == 0:
        return rgba  # Edge case: totally empty mask
    
    top, bottom = np.min(y_non_zero), np.max(y_non_zero)
    left, right = np.min(x_non_zero), np.max(x_non_zero)
    return rgba[top:bottom + 1, left:right + 1]


def upscale_crisp(rgba: np.ndarray) -> np.ndarray:
    """High quality Lanczos upscale tailored to make small UI assets look extremely crisp."""
    h, w = rgba.shape[:2]
    max_dim = max(h, w)

    # Calculate optimal resolution multiplier (ensure output is ~800px-1200px where possible)
    if max_dim >= 1200:
        scale = 1.0
    elif max_dim * 4 <= 1200:
        scale = 4.0
    elif max_dim * 3 <= 1200:
        scale = 3.0
    elif max_dim * 2 <= 1200:
        scale = 2.0
    else:
        scale = 1200.0 / max_dim

    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale <= 1.0:
        return rgba

    # Perform High-Quality Resize
    upscaled = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    upscaled[:, :, 3] = np.clip(upscaled[:, :, 3], 0, 255)

    # Moderate unsharp masking on RGB channels (keeps anti-aliased alpha channel smooth)
    rgb = upscaled[:, :, :3]
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=0.8)
    rgb_sharp = cv2.addWeighted(rgb, 1.4, blurred, -0.4, 0)
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

        # Add slight padding for SAM prompting
        pad_x = max(4, int(bw * 0.05))
        pad_y = max(4, int(bh * 0.05))
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

    # Run SAM → mask smoothing → crop → trim → scale crisp → base64
    results_b64 = []
    
    # Process all image inputs for SAM once
    print("Preparing SAM image...", flush=True)
    pil_image = Image.fromarray(image_rgb)
    
    for idx, ki in enumerate(keep):
        bx0, by0, bx1, by1 = kept_boxes[ki]

        # Use transformers SAM with mathematically perfect sizing
        sam_inputs = sam_processor(pil_image, input_boxes=[[[bx0, by0, bx1, by1]]], return_tensors="pt").to(device)
        
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)
            
        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get the best mask out of the 3 alternatives SAM returns
        scores = sam_outputs.iou_scores[0][0]
        best_mask_idx = torch.argmax(scores).item()
        mask_binary = masks[0][0][best_mask_idx].numpy()
        mask = mask_binary.astype(np.float32)
            
        # Severe anti-aliasing (smooth the binary mask)
        mask_blur = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.8)
        alpha = (mask_blur * 255).clip(0, 255).astype(np.uint8)
        
        # Apply to full image
        rgba_full = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_full[:, :, :3] = image_rgb
        rgba_full[:, :, 3] = alpha
        
        # Crop to bounding box with safe margin
        margin = 10
        cy0 = max(0, by0 - margin)
        cy1 = min(h, by1 + margin)
        cx0 = max(0, bx0 - margin)
        cx1 = min(w, bx1 + margin)
        
        rgba_crop = rgba_full[cy0:cy1, cx0:cx1]

        rgba = trim_transparent(rgba_crop)
        rgba = upscale_crisp(rgba)

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
