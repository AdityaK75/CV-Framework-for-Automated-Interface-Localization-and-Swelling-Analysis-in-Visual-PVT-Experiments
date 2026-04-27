"""
PVT Meniscus Web GUI - Manual Control Version
Flask-based web interface with complete manual control
Access at: http://localhost:5001
"""

import base64
import glob
import json
import logging
import os
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from pvt_analyzer import PVTAnalyzer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "../templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "../static"),
)
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "upload")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# Global state
state = {
    "current_image": None,
    "current_image_path": None,
    "image_paths": [],
    "current_index": 0,
    "roi_rect": None,
    "calib_top": None,
    "calib_bottom": None,
    "meniscus_point": None,
    "mm_distance": 92.0,
}

analyzer = PVTAnalyzer()


def image_to_base64(img_bgr):
    """Convert CV2 BGR image to base64 PNG"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def load_image_at_index(index):
    """Load image at given index"""
    if not state["image_paths"] or index < 0 or index >= len(state["image_paths"]):
        return False

    path = state["image_paths"][index]
    img = cv2.imread(path)
    if img is None:
        return False

    state["current_image"] = img
    state["current_image_path"] = path
    state["current_index"] = index
    return True


@app.route("/")
def index():
    """Main page"""
    return render_template("manual_control.html")


@app.route("/api/load-image", methods=["POST"])
def api_load_image():
    """Load single image file"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    try:
        # Save temporarily
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(temp_path)

        # Load image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        state["current_image"] = img
        state["current_image_path"] = temp_path
        state["image_paths"] = [temp_path]
        state["current_index"] = 0
        state["roi_rect"] = None
        state["calib_top"] = None
        state["calib_bottom"] = None
        state["meniscus_point"] = None

        h, w = img.shape[:2]
        img_b64 = image_to_base64(img)

        return jsonify(
            {
                "success": True,
                "width": w,
                "height": h,
                "image_data": img_b64,
                "filename": file.filename,
            }
        )
    except Exception as e:
        logging.error(f"Load image error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-folder", methods=["POST"])
def api_load_folder():
    """Load all images from folder (simulated - user selects images)"""
    data = request.get_json() or {}
    # For web, we'll use multiple file uploads instead
    return jsonify({"info": "Use multiple file upload or specify folder path"})


@app.route("/api/get-current-image", methods=["GET"])
def api_get_current_image():
    """Get current image as base64"""
    if state["current_image"] is None:
        return jsonify({"error": "No image loaded"}), 400

    h, w = state["current_image"].shape[:2]
    img_b64 = image_to_base64(state["current_image"])

    status = {
        "width": w,
        "height": h,
        "roi_rect": state["roi_rect"],
        "calib_top": state["calib_top"],
        "calib_bottom": state["calib_bottom"],
        "meniscus_point": state["meniscus_point"],
        "mm_distance": state["mm_distance"],
        "index": state["current_index"],
        "total": len(state["image_paths"]),
        "filename": os.path.basename(state["current_image_path"])
        if state["current_image_path"]
        else "unknown",
    }

    return jsonify({"image_data": img_b64, "status": status})


@app.route("/api/set-roi", methods=["POST"])
def api_set_roi():
    """Set ROI rectangle"""
    data = request.get_json() or {}
    x1 = data.get("x1")
    y1 = data.get("y1")
    x2 = data.get("x2")
    y2 = data.get("y2")

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return jsonify({"error": "Missing coordinates"}), 400

    state["roi_rect"] = (int(x1), int(y1), int(x2), int(y2))
    logging.info(f"ROI set: {state['roi_rect']}")

    return jsonify({"success": True, "roi": state["roi_rect"]})


@app.route("/api/set-calib-top", methods=["POST"])
def api_set_calib_top():
    """Set top calibration point"""
    data = request.get_json() or {}
    x = data.get("x")
    y = data.get("y")

    if x is None or y is None:
        return jsonify({"error": "Missing coordinates"}), 400

    state["calib_top"] = (int(x), int(y))
    logging.info(f"Calibration TOP set: {state['calib_top']}")

    return jsonify({"success": True, "calib_top": state["calib_top"]})


@app.route("/api/set-calib-bottom", methods=["POST"])
def api_set_calib_bottom():
    """Set bottom calibration point"""
    data = request.get_json() or {}
    x = data.get("x")
    y = data.get("y")
    mm = data.get("mm")

    if x is None or y is None:
        return jsonify({"error": "Missing coordinates"}), 400

    state["calib_bottom"] = (int(x), int(y))
    if mm is not None:
        state["mm_distance"] = float(mm)

    logging.info(
        f"Calibration BOTTOM set: {state['calib_bottom']}, MM: {state['mm_distance']}"
    )

    return jsonify(
        {
            "success": True,
            "calib_bottom": state["calib_bottom"],
            "mm_distance": state["mm_distance"],
        }
    )


@app.route("/api/set-meniscus", methods=["POST"])
def api_set_meniscus():
    """Set meniscus point"""
    data = request.get_json() or {}
    x = data.get("x")
    y = data.get("y")

    if x is None or y is None:
        return jsonify({"error": "Missing coordinates"}), 400

    state["meniscus_point"] = (int(x), int(y))
    logging.info(f"Meniscus point set: {state['meniscus_point']}")

    return jsonify({"success": True, "meniscus_point": state["meniscus_point"]})


@app.route("/api/set-meniscus-middle", methods=["POST"])
def api_set_meniscus_middle():
    """Alias for set-meniscus, used by the HTML template"""
    data = request.get_json() or {}
    x = data.get("x")
    y = data.get("y")

    if x is None or y is None:
        return jsonify({"error": "Missing coordinates"}), 400

    state["meniscus_point"] = (int(x), int(y))
    logging.info(f"Meniscus middle set: {state['meniscus_point']}")

    return jsonify({"success": True, "meniscus_point": state["meniscus_point"]})


@app.route("/api/run-detection", methods=["POST"])
def api_run_detection():
    """Run detection on meniscus point"""
    if state["current_image"] is None:
        return jsonify({"error": "No image loaded"}), 400

    if state["meniscus_point"] is None:
        return jsonify({"error": "Meniscus point not set"}), 400

    try:
        req = request.get_json() or {}
        method = req.get("method", "literature")
        slope = float(req.get("slope", 1.0))
        intercept = float(req.get("intercept", 0.0))
        mx, my = state["meniscus_point"]

        # Create annotated image
        annotated = state["current_image"].copy()
        h, w = annotated.shape[:2]

        # Draw meniscus line
        cv2.line(annotated, (0, int(my)), (w - 1, int(my)), (0, 165, 255), 3)

        # Draw ROI if set
        if state["roi_rect"]:
            x1, y1, x2, y2 = state["roi_rect"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw calibration points
        if state["calib_top"]:
            tx, ty = state["calib_top"]
            cv2.circle(annotated, (tx, ty), 5, (0, 255, 255), -1)
        if state["calib_bottom"]:
            bx, by = state["calib_bottom"]
            cv2.circle(annotated, (bx, by), 5, (255, 255, 0), -1)

        # Calculate height if calibrated
        height_mm = None
        volume = None
        if state["calib_top"] and state["calib_bottom"]:
            tx, ty = state["calib_top"]
            bx, by = state["calib_bottom"]
            height_px = max(0, by - int(my))
            scale_mm_px = state["mm_distance"] / abs(by - ty)
            height_mm = height_px * scale_mm_px
            # Calculate volume
            volume = slope * height_mm + intercept if height_mm is not None else None
            # Add text
            text = f"Height: {height_mm:.2f} mm"
            cv2.putText(
                annotated,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Save result
        if state["current_image_path"]:
            base = os.path.splitext(os.path.basename(state["current_image_path"]))[0]
        else:
            base = "result"

        out_path = os.path.join("results", f"{base}_manual_{method}.png")
        cv2.imwrite(out_path, annotated)

        img_b64 = image_to_base64(annotated)

        result = {
            "success": True,
            "meniscus_y": int(my),
            "height_mm": height_mm,
            "output_path": out_path,
            "image_data": img_b64,
            "volume": volume,
        }

        logging.info(
            f"Detection complete: {out_path}, height={height_mm}mm, volume={volume}"
        )
        return jsonify(result)

    except Exception as e:
        logging.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/save-setup", methods=["POST"])
def api_save_setup():
    """Save current setup to JSON"""
    try:
        setup = {
            "roi_rect": state["roi_rect"],
            "calib_top": state["calib_top"],
            "calib_bottom": state["calib_bottom"],
            "meniscus_point": state["meniscus_point"],
            "mm_distance": state["mm_distance"],
        }

        filename = f"setup_{state['current_index']}.json"
        path = os.path.join("results", filename)

        with open(path, "w") as f:
            json.dump(setup, f, indent=2)

        logging.info(f"Setup saved: {path}")
        return jsonify({"success": True, "path": path})

    except Exception as e:
        logging.error(f"Save setup error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-setup", methods=["POST"])
def api_load_setup():
    """Load setup from JSON"""
    data = request.get_json() or {}
    path = data.get("path")

    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 400

    try:
        with open(path, "r") as f:
            setup = json.load(f)

        state["roi_rect"] = (
            tuple(setup.get("roi_rect")) if setup.get("roi_rect") else None
        )
        state["calib_top"] = (
            tuple(setup.get("calib_top")) if setup.get("calib_top") else None
        )
        state["calib_bottom"] = (
            tuple(setup.get("calib_bottom")) if setup.get("calib_bottom") else None
        )
        state["meniscus_point"] = (
            tuple(setup.get("meniscus_point")) if setup.get("meniscus_point") else None
        )
        state["mm_distance"] = setup.get("mm_distance", 92.0)

        logging.info(f"Setup loaded: {path}")
        return jsonify(
            {
                "success": True,
                "roi_rect": state["roi_rect"],
                "calib_top": state["calib_top"],
                "calib_bottom": state["calib_bottom"],
                "meniscus_point": state["meniscus_point"],
                "mm_distance": state["mm_distance"],
            }
        )

    except Exception as e:
        logging.error(f"Load setup error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all selections"""
    state["roi_rect"] = None
    state["calib_top"] = None
    state["calib_bottom"] = None
    state["meniscus_point"] = None
    state["mm_distance"] = 92.0

    logging.info("Reset all selections")
    return jsonify({"success": True})


if __name__ == "__main__":
    logging.info("Starting Manual Control Web GUI on http://localhost:5003")
    app.run(debug=False, host="localhost", port=5003, threaded=True)
