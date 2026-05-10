import cv2
import numpy as np
import base64
import math

def encode_image_base64(img):
    """Encodes an OpenCV image to base64 for frontend rendering."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def decode_image_base64(data_uri):
    """Decodes a base64 image from frontend to OpenCV image."""
    encoded_data = data_uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def preprocess_image(img):
    """Applies Bilateral filtering and morphological operations to suppress heavy glare."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to smooth image while keeping edges sharp (good for glare reduction)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Morphological opening to remove small glare spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    
    return morph

def detect_spherical_reactor(image_path_or_b64, is_base64=False):
    """
    Automatically detects the outer glass edge of the spherical reactor.
    Extracts the center coordinates (cx, cy) and pixel radius (R_px).
    """
    if is_base64:
        img = decode_image_base64(image_path_or_b64)
    else:
        img = cv2.imread(image_path_or_b64)

    if img is None:
        raise ValueError("Could not load image.")

    processed = preprocess_image(img)
    
    # Use Hough Circles to detect the spherical boundary
    # Parameters may need tuning depending on image scale/resolution
    circles = cv2.HoughCircles(
        processed, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=50,   # higher threshold for Canny edge detector
        param2=30,   # accumulator threshold (lower = more circles)
        minRadius=50,
        maxRadius=800
    )

    result_img = img.copy()
    cx, cy, r_px = None, None, None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the largest or strongest circle detected
        best_circle = circles[0, 0]
        cx, cy, r_px = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
        
        # Draw the outer circle on diagnostic image
        cv2.circle(result_img, (cx, cy), r_px, (0, 255, 0), 2)
        # Draw the center
        cv2.circle(result_img, (cx, cy), 2, (0, 0, 255), 3)
    
    encoded_img = encode_image_base64(result_img)

    return {
        "success": circles is not None,
        "cx": cx,
        "cy": cy,
        "r_px": r_px,
        "diagnostic_image": f"data:image/png;base64,{encoded_img}"
    }

def detect_meniscus(image_path_or_b64, cx, cy, r_px, is_base64=False):
    """
    Localized edge detection around vertical center line (x = cx) to find meniscus y_meniscus.
    """
    if is_base64:
        img = decode_image_base64(image_path_or_b64)
    else:
        img = cv2.imread(image_path_or_b64)

    if img is None:
        raise ValueError("Could not load image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define an ROI (Region of Interest) around the vertical center to avoid glass wall reflections
    # We take a vertical strip of width 0.4 * R_px centered at cx.
    roi_width = int(0.4 * r_px)
    x_start = max(0, cx - roi_width // 2)
    x_end = min(img.shape[1], cx + roi_width // 2)
    
    # Y bounds limited to the sphere's bounds
    y_start = max(0, cy - r_px + int(0.1*r_px)) # offset to avoid top glass edge
    y_end = min(img.shape[0], cy + r_px - int(0.1*r_px)) # offset to avoid bottom glass edge

    roi = gray[y_start:y_end, x_start:x_end]
    
    # Edge detection using Sobel (focusing on horizontal edges)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_y = np.absolute(sobel_y)
    
    # Average across the columns (x-axis) to find the strongest horizontal line
    row_averages = np.mean(abs_sobel_y, axis=1)
    
    # Find the y coordinate of the maximum gradient
    if len(row_averages) > 0:
        local_y_max = np.argmax(row_averages)
        y_meniscus = y_start + local_y_max
        success = True
    else:
        y_meniscus = cy
        success = False

    result_img = img.copy()
    if success:
        # Draw meniscus line within the sphere
        line_length = int(math.sqrt(max(0, r_px**2 - (y_meniscus - cy)**2)))
        x1 = cx - line_length
        x2 = cx + line_length
        cv2.line(result_img, (x1, y_meniscus), (x2, y_meniscus), (255, 0, 0), 2)
        
        # Draw ROI box for diagnostic purposes
        cv2.rectangle(result_img, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)

    encoded_img = encode_image_base64(result_img)

    return {
        "success": success,
        "y_meniscus": int(y_meniscus),
        "diagnostic_image": f"data:image/png;base64,{encoded_img}"
    }

def calculate_volume(r_px, cy, y_meniscus, physical_diameter_mm, ref_volume=None):
    """
    Calculates the liquid volume using the spherical cap volume formula.
    V(h) = (π * h^2 * (3R - h)) / 3
    """
    # Scaling factor
    actual_radius_mm = physical_diameter_mm / 2.0
    px_to_mm_ratio = physical_diameter_mm / (2.0 * r_px) if r_px > 0 else 0

    # Calculate physical height of the liquid
    # Bottom of the sphere is at cy + r_px
    bottom_y_px = cy + r_px
    h_px = bottom_y_px - y_meniscus
    h_mm = h_px * px_to_mm_ratio
    
    # Clamp h_mm between 0 and 2*R_mm
    h_mm = max(0, min(h_mm, physical_diameter_mm))

    # Spherical cap formula
    # V(h) = (1/3) * pi * h^2 * (3R - h)
    v_mm3 = (math.pi * (h_mm ** 2) * (3 * actual_radius_mm - h_mm)) / 3.0
    
    # Convert mm^3 to mL (cm^3)
    v_ml = v_mm3 / 1000.0

    sf = None
    if ref_volume and ref_volume > 0:
        sf = v_ml / ref_volume

    return {
        "h_px": h_px,
        "h_mm": round(h_mm, 4),
        "px_to_mm_ratio": round(px_to_mm_ratio, 6),
        "volume_ml": round(v_ml, 4),
        "swelling_factor": round(sf, 4) if sf else None
    }
