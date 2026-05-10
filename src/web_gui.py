__author__ = "Aditya Kanagalekar"

import base64
import csv
import logging
import os
from io import BytesIO

import cv2
import json
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from pvt_analyzer import PVTAnalyzer
from vision_engine import detect_spherical_reactor, detect_meniscus, calculate_volume

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "../templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "../static"),
)

os.makedirs("results", exist_ok=True)
os.makedirs("upload", exist_ok=True)

state = {
    "images": {},       # image_id -> BGR numpy array
    "paths": {},        # image_id -> path to temp file on disk
    "rois": {},         # image_id -> (x1, y1, x2, y2)
    "calib_top": {},    # image_id -> (x, y)
    "calib_bottom": {}, # image_id -> (x, y)
    "mm_distance": {},  # image_id -> float
    "metadata": {},     # image_id -> dict
    "results": {},      # image_id -> dict
    "base_id": None     # ID of the base image
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_metadata(data):
    """Extract metadata from request JSON."""
    return {
        "date_time": data.get("date_time", ""),
        "timestamp": data.get("timestamp", ""),
        "sample": data.get("sample", ""),
        "solvent": data.get("solvent", ""),
        "system_pressure": data.get("system_pressure", ""),
        "system_temperature": data.get("system_temperature", ""),
    }

def create_metadata_header(metadata, height_mm, volume_ml, width):
    """Create a styled, professional header image with metadata text."""
    header_width = max(width, 900)
    height = 200
    
    # Dark gray background
    header = np.ones((height, header_width, 3), dtype=np.uint8) * 30
    
    # Bottom accent line (Cyan)
    cv2.line(header, (0, height-2), (header_width, height-2), (255, 200, 0), 4)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(header, "PVT SWELLING ANALYSIS REPORT", (30, 40), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Meta layout columns
    col1_x = 30
    col2_x = 350
    col3_x = 600
    
    y_start = 90
    y_step = 35
    
    # Column 1
    cv2.putText(header, f"Sample: {metadata.get('sample', 'N/A')}", (col1_x, y_start), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, f"Solvent: {metadata.get('solvent', 'N/A')}", (col1_x, y_start + y_step), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, f"Date: {metadata.get('date_time', 'N/A')}", (col1_x, y_start + 2*y_step), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Column 2
    cv2.putText(header, f"Timestamp: {metadata.get('timestamp', '0')}h", (col2_x, y_start), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, f"Pressure: {metadata.get('system_pressure', 'N/A')}", (col2_x, y_start + y_step), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, f"Temp: {metadata.get('system_temperature', 'N/A')}", (col2_x, y_start + 2*y_step), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Column 3 (Results Highlight Box)
    cv2.rectangle(header, (col3_x - 15, y_start - 30), (header_width - 20, height - 20), (45, 45, 45), -1)
    cv2.rectangle(header, (col3_x - 15, y_start - 30), (header_width - 20, height - 20), (100, 100, 100), 1)
    
    cv2.putText(header, "MEASUREMENTS", (col3_x, y_start - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    h_str = f"Height : {height_mm:.2f} mm" if height_mm is not None else "Height : ---"
    v_str = f"Volume : {volume_ml:.2f} ml" if volume_ml is not None else "Volume : ---"
    
    cv2.putText(header, h_str, (col3_x, y_start + y_step), font, 0.8, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(header, v_str, (col3_x, y_start + 2*y_step + 5), font, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
    
    return header, header_width

def image_to_base64(img_bgr: np.ndarray) -> str:
    """Convert a BGR numpy array to a base64-encoded PNG string."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def append_result_to_csv(image_id, metadata, result):
    """Appends the result to a central CSV file for easy validation."""
    csv_path = os.path.join("results", "summary_report.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Image ID", "Date", "Timestamp", "Sample", "Solvent", 
                "System Pressure", "System Temperature", 
                "Meniscus Y (px)", "Height (mm)", "Volume (ml)", "Swelling Factor"
            ])
            
        writer.writerow([
            image_id,
            metadata.get("date_time", ""),
            metadata.get("timestamp", ""),
            metadata.get("sample", ""),
            metadata.get("solvent", ""),
            metadata.get("system_pressure", ""),
            metadata.get("system_temperature", ""),
            result.get("meniscus_y", ""),
            result.get("height_mm", ""),
            result.get("volume_ml", ""),
            result.get("swelling_factor", "") if result.get("swelling_factor") is not None else ""
        ])


def detect_meniscus_smart(img_bgr, roi_rect, calib_top_y, calib_bot_y):
    """Sight-glass optimised meniscus detection. Constrains search to the
    calibration band, then looks for the dark->bright transition
    (gas above, liquid below)."""

    # 1. Crop to roi_rect if provided
    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        offset_y = y1
        roi = img_bgr[y1:y2, x1:x2]
    else:
        offset_y = 0
        roi = img_bgr.copy()

    roi_h = roi.shape[0]

    # 2. Restrict vertical search to calibration band, clamped to [0, roi_h)
    search_top = max(0, min(roi_h, calib_top_y - offset_y))
    search_bot = max(0, min(roi_h, calib_bot_y - offset_y))

    # Add a 5% margin to exclude the static top/bottom edges of the glass window
    # This prevents the algorithm from mistaking the massive contrast of the metal casing for a meniscus
    margin = int(0.05 * (search_bot - search_top))
    if search_bot - search_top > 40:
        search_top += margin
        search_bot -= margin

    # 3. Fallback to full crop if range is degenerate
    if search_top >= search_bot:
        search_top = 0
        search_bot = roi_h

    # 4. Extract search band
    band = roi[search_top:search_bot, :]
    band_h = band.shape[0]

    # 5. Grayscale -> bilateralFilter -> CLAHE
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # 6. Absolute Sobel-Y to find strong horizontal edges (liquid/gas interface)
    # Using absolute instead of positive-only captures both dark->bright and bright->dark transitions
    enhanced_f64 = enhanced.astype(np.float64)
    sob = cv2.Sobel(enhanced_f64, cv2.CV_64F, 0, 1, ksize=5)
    pos = np.absolute(sob)
    row_pos = pos.sum(axis=1)

    # 7. Smooth row_pos with a Gaussian kernel (odd size)
    ks = max(7, band_h // 20)
    if ks % 2 == 0:
        ks += 1
    arr = row_pos.reshape(1, -1).astype(np.float32)
    row_pos_smoothed = cv2.GaussianBlur(arr, (ks, 1), 0).flatten()

    # 8. Peak gradient row within the band
    peak_grad = int(np.argmax(row_pos_smoothed))

    # 9. Otsu bright-region top
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_3x3 = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_3x3)
    row_mean = thresh.mean(axis=1).astype(np.float32)
    arr2 = row_mean.reshape(1, -1)
    row_mean_smoothed = cv2.GaussianBlur(arr2, (ks, 1), 0).flatten()
    max_bri = float(row_mean_smoothed.max())
    if max_bri > 0:
        bright_rows = np.where(row_mean_smoothed > 0.25 * max_bri)[0]
        peak_otsu = int(bright_rows[0]) if len(bright_rows) > 0 else peak_grad
    else:
        peak_otsu = peak_grad

    # 10. Combine: average if close, else trust gradient peak
    if abs(peak_grad - peak_otsu) < 0.05 * band_h:
        peak_combined = int(round((peak_grad + peak_otsu) / 2.0))
    else:
        peak_combined = peak_grad

    # 11. Map back to original image coordinates
    meniscus_y = offset_y + search_top + peak_combined

    # 12. Return
    return int(meniscus_y)


def annotate_image(img_bgr, meniscus_y, roi_rect, calib_top, calib_bottom, height_mm, diagnostics=None):
    """Draw annotations on a copy of img_bgr and return the annotated copy."""
    out = img_bgr.copy()
    h_img, w_img = out.shape[:2]

    # ROI rectangle (green)
    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calibration dots
    if calib_top is not None:
        cv2.circle(out, tuple(calib_top), 2, (0, 255, 255), -1)  # yellow (BGR)
    if calib_bottom is not None:
        cv2.circle(out, tuple(calib_bottom), 2, (255, 255, 0), -1)  # cyan (BGR)

    # Orange dashed line between calibration points
    if calib_top is not None and calib_bottom is not None:
        x1c, y1c = calib_top
        x2c, y2c = calib_bottom
        dx = x2c - x1c
        dy = y2c - y1c
        length = float(np.sqrt(dx * dx + dy * dy))
        if length > 0:
            ux, uy = dx / length, dy / length
            t = 0.0
            while t < length:
                p1 = (int(x1c + ux * t), int(y1c + uy * t))
                t_end = min(t + 10.0, length)
                p2 = (int(x1c + ux * t_end), int(y1c + uy * t_end))
                cv2.line(out, p1, p2, (0, 165, 255), 2)  # orange in BGR
                t += 15.0  # 10 px drawn + 5 px gap

    # Meniscus line — thin and precise
    y_int = int(round(meniscus_y))
    cv2.line(out, (0, y_int), (w_img, y_int), (0, 0, 255), 1, cv2.LINE_AA)
    
    # Optional indicator if ROI exists
    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        center_x = (x1 + x2) // 2
        cv2.circle(out, (center_x, y_int), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw the spatial polynomial fitted curve (Shape of meniscus)
    if diagnostics and "spatial_coeffs" in diagnostics and diagnostics["spatial_coeffs"]:
        s_coeffs = diagnostics["spatial_coeffs"]
        if len(s_coeffs) == 3:
            A, B, C = s_coeffs
            if roi_rect is not None:
                x_start, _, x_end, _ = roi_rect
            else:
                x_start, x_end = 0, w_img
            
            poly_pts = []
            for x_val in range(int(x_start), int(x_end)):
                y_val = int(round(A * x_val**2 + B * x_val + C))
                # Only draw if it's somewhat close to the meniscus to avoid drawing wild extrapolations
                if abs(y_val - meniscus_y) < 200:
                    poly_pts.append([x_val, y_val])
            
            if poly_pts:
                poly_pts = np.array([poly_pts], dtype=np.int32)
                cv2.polylines(out, poly_pts, False, (255, 200, 0), 2, cv2.LINE_AA) # Light blue/Cyan curve

    # Draw contour if available
    if diagnostics and "contour" in diagnostics and diagnostics["contour"]:
        pts = diagnostics["contour"]
        if len(pts) > 0:
            for pt in pts:
                px, py = int(pt["x"]), int(pt["y"])
                cv2.circle(out, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)

    # Authorship Watermark
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Analysis by Aditya Kanagalekar"
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    padding = 10
    text_x = w_img - text_size[0] - padding
    text_y = h_img - padding
    
    # Draw subtle background rectangle for readability
    cv2.rectangle(out, (text_x - 5, text_y - text_size[1] - 5), (w_img - 5, h_img - 5), (30, 30, 30), -1)
    # Draw text
    cv2.putText(out, text, (text_x, text_y), font, font_scale, (150, 150, 150), thickness, cv2.LINE_AA)

    return out


def make_analyzer(image_id: str) -> PVTAnalyzer:
    """Create PVTAnalyzer, apply per-image ROI and calibration from state."""
    an = PVTAnalyzer()

    roi = state["rois"].get(image_id)
    if roi is not None:
        x1, y1, x2, y2 = roi
        an.set_roi(y1, y2, x1, x2)

    ct = state["calib_top"].get(image_id)
    cb = state["calib_bottom"].get(image_id)
    mm = state["mm_distance"].get(image_id, 92.0)
    if ct is not None and cb is not None:
        tx, ty = ct
        bx, by = cb
        dy = abs(by - ty)
        if dy > 0:
            an.scale_mm_per_pixel = mm / dy
            an.calib_lines = (min(ty, by), max(ty, by))

    return an


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("auto_swelling.html")


@app.route("/api/load-image", methods=["POST"])
def api_load_image():
    image_id = request.form.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        safe_name = file.filename.replace(" ", "_")
        temp_path = os.path.join("upload", f"{image_id}_{safe_name}")
        file.save(temp_path)

        img = cv2.imread(temp_path)
        if img is None:
            return jsonify(
                {"error": "Could not decode image — unsupported format?"}
            ), 400

        state["images"][image_id] = img
        state["paths"][image_id] = temp_path

        h, w = img.shape[:2]
        return jsonify(
            {
                "success": True,
                "image_id": image_id,
                "width": w,
                "height": h,
                "image_data": image_to_base64(img),
            }
        )
    except Exception as exc:
        logging.error("load-image error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/set-roi", methods=["POST"])
def api_set_roi():
    data = request.get_json() or {}

    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400

    if data.get("clear"):
        state["rois"].pop(image_id, None)
        logging.info("ROI cleared for %s", image_id)
        return jsonify({"success": True, "image_id": image_id, "roi": None})

    try:
        x1 = int(data["x1"])
        y1 = int(data["y1"])
        x2 = int(data["x2"])
        y2 = int(data["y2"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide x1, y1, x2, y2 as integers"}), 400

    state["rois"][image_id] = (x1, y1, x2, y2)
    logging.info("ROI for %s set to %s", image_id, state["rois"][image_id])
    return jsonify({"success": True, "image_id": image_id, "roi": state["rois"][image_id]})


@app.route("/api/set-calib-top", methods=["POST"])
def api_set_calib_top():
    data = request.get_json() or {}
    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400
    try:
        x, y = int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide x and y as integers"}), 400

    state["calib_top"][image_id] = (x, y)
    logging.info("Calib TOP [%s] set to %s", image_id, state["calib_top"][image_id])
    return jsonify(
        {"success": True, "image_id": image_id, "calib_top": list(state["calib_top"][image_id])}
    )


@app.route("/api/set-calib-bottom", methods=["POST"])
def api_set_calib_bottom():
    data = request.get_json() or {}
    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400
    try:
        x, y = int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide x and y as integers"}), 400

    state["calib_bottom"][image_id] = (x, y)
    if "mm" in data:
        try:
            state["mm_distance"][image_id] = float(data["mm"])
        except (TypeError, ValueError):
            pass

    logging.info(
        "Calib BOTTOM [%s] set to %s  (mm_distance=%.1f)",
        image_id,
        state["calib_bottom"][image_id],
        state["mm_distance"].get(image_id, 92.0),
    )
    return jsonify({"success": True, "image_id": image_id})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json() or {}
    image_id = data.get("image_id")
    
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400
    if image_id not in state["images"]:
        return jsonify({"error": f"Image {image_id} not loaded"}), 400

    geometry = data.get("geometry", "tubular")

    if geometry == "tubular":
        ct = state["calib_top"].get(image_id)
        cb = state["calib_bottom"].get(image_id)
        if ct is None or cb is None:
            return jsonify(
                {"error": f"Set both TOP and BOTTOM calibration points for {image_id} first"}
            ), 400
        if abs(cb[1] - ct[1]) == 0:
            return jsonify(
                {"error": f"Top and bottom calibration points for {image_id} are at the same Y position"}
            ), 400
    else:
        ct = None
        cb = None

    method = data.get("method", "literature")
    
    # Store metadata per image
    state["metadata"][image_id] = extract_metadata(data)
    
    try:
        slope = float(data.get("slope", 1.0))
        intercept = float(data.get("intercept", 0.0))
        vmax = float(data.get("vmax", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "slope, intercept and vmax must be numbers"}), 400

    try:
        img = state["images"][image_id]
        img_path = state["paths"][image_id]
        roi_slot = state["rois"].get(image_id)

        if geometry == "spherical":
            try:
                physical_diameter = float(data.get("physical_diameter", 50.0))
            except (TypeError, ValueError):
                physical_diameter = 50.0

            sphere_res = detect_spherical_reactor(img_path, is_base64=False)
            if not sphere_res["success"]:
                raise Exception("Could not detect spherical reactor boundary.")
            
            cx, cy, r_px = sphere_res["cx"], sphere_res["cy"], sphere_res["r_px"]

            men_res = detect_meniscus(img_path, cx, cy, r_px, is_base64=False)
            if not men_res["success"]:
                y_meniscus = cy + r_px
            else:
                y_meniscus = men_res["y_meniscus"]

            vol_res = calculate_volume(r_px, cy, y_meniscus, physical_diameter)

            meniscus_y = float(y_meniscus)
            height_mm = vol_res["h_mm"]
            volume_ml = vol_res["volume_ml"]
            diagnostics = {"detected_y": meniscus_y, "subpixel_y": meniscus_y, "cx": cx, "cy": cy, "r_px": r_px}
            
            # Pre-draw the spherical boundary so annotate_image captures it
            img_to_annotate = img.copy()
            cv2.circle(img_to_annotate, (cx, cy), r_px, (0, 255, 0), 2)
            cv2.circle(img_to_annotate, (cx, cy), 2, (0, 0, 255), 3)

        else:
            mm = state["mm_distance"].get(image_id, 92.0)
            calib_top_y = ct[1]
            calib_bot_y = cb[1]
            dy = abs(calib_bot_y - calib_top_y)
            scale = mm / dy
            valid_lo = min(calib_top_y, calib_bot_y)
            valid_hi = max(calib_top_y, calib_bot_y)

            # a. Run detection based on selected method
            method = data.get("method", "smart")
            if method == "smart":
                meniscus_y = float(detect_meniscus_smart(img, roi_slot, calib_top_y, calib_bot_y))
            else:
                # Fallback to legacy PVTAnalyzer methods
                an = make_analyzer(image_id)
                if method == "spanning":
                    y, _ = an.find_meniscus_spanning(img)
                elif method == "advanced":
                    y, _ = an.find_meniscus_advanced(img)
                elif method == "basic":
                    y, _ = an.find_meniscus_basic(img)
                else: # literature
                    y, _ = an.find_meniscus_literature(img)
                
                if y is None:
                    # If legacy fails, use smart as safe fallback
                    meniscus_y = float(detect_meniscus_smart(img, roi_slot, calib_top_y, calib_bot_y))
                else:
                    meniscus_y = float(y)

            # b. PVT analyzer edge shape/contour extraction
            diagnostics = {"detected_y": meniscus_y, "subpixel_y": meniscus_y}
            try:
                an = make_analyzer(image_id)
                # Find the true contour near the accurately detected Y height (allow fragmented edges down to 10% width)
                contour_orig = an.find_meniscus_contour(img, meniscus_y, span_pct=0.10)
                
                if contour_orig is not None and len(contour_orig) >= 3:
                    pts = []
                    xs = []
                    ys = []
                    for pt in contour_orig:
                        x_val = float(pt[0][0])
                        y_val = float(pt[0][1])
                        pts.append({"x": x_val, "y": y_val})
                        xs.append(x_val)
                        ys.append(y_val)
                    diagnostics["contour"] = pts
                    try:
                        unique_xs, indices = np.unique(xs, return_index=True)
                        unique_ys = np.array(ys)[indices]
                        if len(unique_xs) >= 3:
                            spatial_coeffs = np.polyfit(unique_xs, unique_ys, 2).tolist()
                            diagnostics["spatial_coeffs"] = spatial_coeffs
                    except Exception as e:
                        logging.warning(f"Spatial polynomial fit failed: {e}")
            except Exception as pvt_exc:
                logging.warning("PVTAnalyzer contour extraction failed for %s: %s", image_id, pvt_exc)

            # d-g. Measurements
            height_px = max(0, calib_bot_y - meniscus_y)
            height_mm = height_px * scale
            v_measured = slope * height_mm + intercept
            volume_ml = 14.81 - v_measured
            
            img_to_annotate = img

        # h. Annotate
        annotated = annotate_image(
            img_to_annotate,
            meniscus_y,
            roi_slot,
            ct,
            cb,
            height_mm,
            diagnostics=diagnostics
        )

        # Output ROI crop and header
        roi_filename = None
        if roi_slot is not None:
            x1, y1, x2, y2 = roi_slot
            h, w = annotated.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if y2 > y1 and x2 > x1:
                roi_crop = annotated[y1:y2, x1:x2]
                
                header_roi, header_width = create_metadata_header(state["metadata"][image_id], height_mm, volume_ml, roi_crop.shape[1])
                
                # Pad ROI if it is narrower than header
                pad_left = (header_width - roi_crop.shape[1]) // 2
                pad_right = header_width - roi_crop.shape[1] - pad_left
                if pad_left > 0 or pad_right > 0:
                    roi_padded = cv2.copyMakeBorder(roi_crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    roi_padded = roi_crop

                final_img = np.vstack([header_roi, roi_padded])
                
                # Generate clean filename
                sample_name = state["metadata"][image_id].get("sample", "Sample").strip().replace(" ", "_")
                time_val = state["metadata"][image_id].get("timestamp", "0").strip()
                clean_name = f"Report_{sample_name}_t{time_val}_{image_id}.png"
                clean_name = "".join([c for c in clean_name if c.isalnum() or c in "._-"])
                
                roi_filename = clean_name
                cv2.imwrite(os.path.join("results", roi_filename), final_img)
        else:
            # If no ROI, use the full annotated image
            header_full, _ = create_metadata_header(state["metadata"][image_id], height_mm, volume_ml, annotated.shape[1])
            final_img = np.vstack([header_full, annotated])
            
            sample_name = state["metadata"][image_id].get("sample", "Sample").strip().replace(" ", "_")
            time_val = state["metadata"][image_id].get("timestamp", "0").strip()
            clean_name = f"Report_{sample_name}_t{time_val}_{image_id}_FULL.png"
            clean_name = "".join([c for c in clean_name if c.isalnum() or c in "._-"])
            
            roi_filename = clean_name
            cv2.imwrite(os.path.join("results", roi_filename), final_img)

        with open(os.path.join("results", f"{image_id}_metadata.json"), "w") as f:
            json.dump(state["metadata"][image_id], f, indent=2)

        # Swelling factor calculation vs base
        swelling_factor = None
        base_id = state.get("base_id")
        if base_id and base_id in state["results"]:
            base_vol = state["results"][base_id].get("volume_ml")
            if base_vol and base_vol != 0:
                swelling_factor = round(volume_ml / base_vol, 6)

        # Store result (Note: We return the unpadded annotated image for the UI viewer)
        result_data = {
            "meniscus_y": meniscus_y,
            "height_mm": round(height_mm, 4),
            "volume_ml": round(volume_ml, 4),
            "roi_filename": roi_filename,
            "swelling_factor": swelling_factor,
            "diagnostics": diagnostics,
            "roi": roi_slot,
            "calib_top": ct,
            "calib_bottom": cb
        }
        
        with open(os.path.join("results", f"{image_id}_measurements.json"), "w") as f:
            json.dump(result_data, f, indent=2)

        state["results"][image_id] = result_data
        
        # Append to central CSV report
        append_result_to_csv(image_id, state["metadata"][image_id], result_data)

        # Add image data for frontend
        state["results"][image_id]["image_data"] = image_to_base64(annotated)

    except Exception as exc:
        logging.error("Detection error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "success": True,
            "image_id": image_id,
            "result": state["results"][image_id]
        }
    )

@app.route("/api/set-meniscus", methods=["POST"])
def api_set_meniscus():
    data = request.get_json() or {}
    image_id = data.get("image_id")

    if not image_id:
        return jsonify({"error": "image_id is required"}), 400

    state["metadata"][image_id] = extract_metadata(data)

    try:
        y_raw = data["y"]
        if not isinstance(y_raw, (int, float)):
            raise ValueError("y must be a number")
        meniscus_y = int(y_raw)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid y: {exc}"}), 400

    img = state["images"].get(image_id)
    if img is None:
        return jsonify({"error": f"{image_id} image not loaded"}), 400

    ct = state["calib_top"].get(image_id)
    cb = state["calib_bottom"].get(image_id)
    if ct is None or cb is None:
        return jsonify(
            {"error": f"Set both calibration points for {image_id} first"}
        ), 400

    calib_top_y = ct[1]
    calib_bot_y = cb[1]
    dy = abs(calib_bot_y - calib_top_y)
    if dy == 0:
        return jsonify({"error": "Calibration points are at the same Y position"}), 400

    mm = state["mm_distance"].get(image_id, 92.0)
    scale = mm / dy
    try:
        slope = float(data.get("slope", 1.0))
        intercept = float(data.get("intercept", 0.0))
    except (TypeError, ValueError):
        slope, intercept = 1.0, 0.0

    height_px = max(0, calib_bot_y - meniscus_y)
    height_mm = height_px * scale
    v_measured = slope * height_mm + intercept
    volume_ml = 14.81 - v_measured

    diagnostics = {"detected_y": meniscus_y, "subpixel_y": meniscus_y}
    try:
        an = make_analyzer(image_id)
        # Find the true contour near the manually provided Y height (allow fragmented edges down to 10% width)
        contour_orig = an.find_meniscus_contour(img, meniscus_y, span_pct=0.10)
        
        if contour_orig is not None and len(contour_orig) >= 3:
            pts = []
            xs = []
            ys = []
            for pt in contour_orig:
                x_val = float(pt[0][0])
                y_val = float(pt[0][1])
                pts.append({"x": x_val, "y": y_val})
                xs.append(x_val)
                ys.append(y_val)
            diagnostics["contour"] = pts
            try:
                unique_xs, indices = np.unique(xs, return_index=True)
                unique_ys = np.array(ys)[indices]
                if len(unique_xs) >= 3:
                    spatial_coeffs = np.polyfit(unique_xs, unique_ys, 2).tolist()
                    diagnostics["spatial_coeffs"] = spatial_coeffs
            except Exception as e:
                logging.warning(f"Spatial polynomial fit failed: {e}")
    except Exception as pvt_exc:
        logging.warning("PVTAnalyzer manual contour extraction failed for %s: %s", image_id, pvt_exc)

    roi_slot = state["rois"].get(image_id)
    annotated = annotate_image(
        img,
        meniscus_y,
        roi_slot,
        ct,
        cb,
        height_mm,
        diagnostics=diagnostics
    )

    roi_filename = None
    if roi_slot is not None:
        x1, y1, x2, y2 = roi_slot
        h, w = annotated.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if y2 > y1 and x2 > x1:
            roi_crop = annotated[y1:y2, x1:x2]
            
            header_roi, header_width = create_metadata_header(state["metadata"][image_id], height_mm, volume_ml, roi_crop.shape[1])
            
            # Pad ROI if it is narrower than header
            pad_left = (header_width - roi_crop.shape[1]) // 2
            pad_right = header_width - roi_crop.shape[1] - pad_left
            if pad_left > 0 or pad_right > 0:
                roi_padded = cv2.copyMakeBorder(roi_crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                roi_padded = roi_crop

            final_img = np.vstack([header_roi, roi_padded])
            
            sample_name = state["metadata"][image_id].get("sample", "Sample").strip().replace(" ", "_")
            time_val = state["metadata"][image_id].get("timestamp", "0").strip()
            clean_name = f"Report_{sample_name}_t{time_val}_{image_id}_corrected.png"
            clean_name = "".join([c for c in clean_name if c.isalnum() or c in "._-"])
            
            roi_filename = clean_name
            cv2.imwrite(os.path.join("results", roi_filename), final_img)
    else:
        # If no ROI, use the full annotated image
        header_full, _ = create_metadata_header(state["metadata"][image_id], height_mm, volume_ml, annotated.shape[1])
        final_img = np.vstack([header_full, annotated])
        
        sample_name = state["metadata"][image_id].get("sample", "Sample").strip().replace(" ", "_")
        time_val = state["metadata"][image_id].get("timestamp", "0").strip()
        clean_name = f"Report_{sample_name}_t{time_val}_{image_id}_corrected_FULL.png"
        clean_name = "".join([c for c in clean_name if c.isalnum() or c in "._-"])
        
        roi_filename = clean_name
        cv2.imwrite(os.path.join("results", roi_filename), final_img)

    with open(os.path.join("results", f"{image_id}_metadata.json"), "w") as f:
        json.dump(state["metadata"][image_id], f, indent=2)

    # Calculate swelling factor if base exists
    swelling_factor = None
    base_id = state.get("base_id")
    if base_id and base_id in state["results"]:
        base_vol = state["results"][base_id].get("volume_ml")
        if base_vol and base_vol != 0:
            swelling_factor = round(volume_ml / base_vol, 6)

    result_data = {
        "meniscus_y": meniscus_y,
        "height_mm": round(height_mm, 4),
        "volume_ml": round(volume_ml, 4),
        "roi_filename": roi_filename,
        "swelling_factor": swelling_factor,
        "diagnostics": {},
        "roi": roi_slot,
        "calib_top": ct,
        "calib_bottom": cb
    }

    with open(os.path.join("results", f"{image_id}_measurements.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    state["results"][image_id] = result_data
    
    # Append to central CSV report
    append_result_to_csv(image_id, state["metadata"][image_id], result_data)

    state["results"][image_id]["image_data"] = image_to_base64(annotated)

    return jsonify(
        {
            "success": True,
            "image_id": image_id,
            "result": state["results"][image_id]
        }
    )

@app.route("/api/set-base-image", methods=["POST"])
def api_set_base_image():
    data = request.get_json() or {}
    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id is required"}), 400
    
    if image_id not in state["images"]:
        return jsonify({"error": f"Image {image_id} not loaded"}), 400
        
    state["base_id"] = image_id
    logging.info("Base image set to %s", image_id)
    return jsonify({"success": True, "base_id": image_id})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all state back to defaults."""
    state["images"].clear()
    state["paths"].clear()
    state["rois"].clear()
    state["calib_top"].clear()
    state["calib_bottom"].clear()
    state["mm_distance"].clear()
    state["metadata"].clear()
    state["results"].clear()
    state["base_id"] = None
    logging.info("State reset")
    return jsonify({"success": True})


@app.route("/api/result-image/<filename>")
def api_result_image(filename):
    """Serve saved images from the results folder."""
    import os
    from flask import send_file
    path = os.path.abspath(os.path.join("results", filename))
    if os.path.exists(path):
        return send_file(path)
    return "Not Found", 404


@app.route("/api/saved-results", methods=["GET"])
def api_saved_results():
    """List all saved results."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return jsonify({"success": True, "results": []})
    
    saved = []
    for f in os.listdir(results_dir):
        if f.endswith("_measurements.json"):
            image_id = f.replace("_measurements.json", "")
            meta_file = os.path.join(results_dir, f"{image_id}_metadata.json")
            
            meta_data = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r") as mf:
                        meta_data = json.load(mf)
                except Exception:
                    pass
            
            # Read measurement summary
            try:
                with open(os.path.join(results_dir, f), "r") as meas_f:
                    meas_data = json.load(meas_f)
                    
                saved.append({
                    "image_id": image_id,
                    "date": meta_data.get("date_time", "Unknown"),
                    "sample": meta_data.get("sample", "Unknown"),
                    "volume": meas_data.get("volume_ml", 0),
                    "height": meas_data.get("height_mm", 0)
                })
            except Exception:
                pass
                
    # Sort by date/filename descending
    saved.sort(key=lambda x: x["image_id"], reverse=True)
    return jsonify({"success": True, "results": saved})


@app.route("/api/load-result/<image_id>", methods=["GET"])
def api_load_result(image_id):
    """Load a specific saved result."""
    results_dir = "results"
    meas_file = os.path.join(results_dir, f"{image_id}_measurements.json")
    meta_file = os.path.join(results_dir, f"{image_id}_metadata.json")
    
    if not os.path.exists(meas_file):
        return jsonify({"error": "Result not found"}), 404
        
    try:
        with open(meas_file, "r") as f:
            meas_data = json.load(f)
            
        meta_data = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
                
        # Attempt to load the original image if available (optional for history viewing)
        # But we definitely need to return the measurements and metadata
        return jsonify({
            "success": True,
            "image_id": image_id,
            "measurements": meas_data,
            "metadata": meta_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("               PVT Auto-Swelling Web GUI")
    print("           Developed by: Aditya Kanagalekar")
    print("=" * 55)
    print("  Access at: http://localhost:5006")
    print("=" * 55)
    app.run(debug=False, host="localhost", port=5006, threaded=True)
