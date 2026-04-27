# PVT Swelling Analyzer Architecture

This repository contains a robust, computer-vision driven web application for analyzing PVT swelling test images. The application is designed to accurately detect and trace glowing fluid meniscus boundaries under heavy glare, allowing for precise fluid volume and relative swelling factor measurements over time.

The application architecture is strictly divided into three core files that handle the frontend, backend routing, and mathematical image processing separately.

---

## 1. The Core Engine: `src/pvt_analyzer.py`

This file is the **Scientific Computer Vision Engine**. It is completely decoupled from the web layer and only processes raw pixel data using OpenCV and Numpy.

### Key Responsibilities:
- **Sub-pixel Edge Tracing:** Contains the `find_meniscus_contour` function, which completely abandons standard Canny edge detection (which fails on glowing/fragmented fluids) in favor of a custom **Column-wise Absolute Gradient Search**. 
- **Robustness:** It scans every vertical slice of the fluid column, identifies the exact physical edge by looking for maximum absolute contrast, applies a median filter to remove noise spikes, and generates a mathematically contiguous curve representing the physical shape of the meniscus.
- **Polynomial Fitting:** Returns the traced points so the web GUI can fit a quadratic curve (`y = Ax² + Bx + C`) to perfectly visualize the shape.

---

## 2. The Application Server: `src/web_gui.py`

This file is the **Flask Backend & API Router**. It acts as the bridge between the frontend user interface and the backend mathematical engine.

### Key Responsibilities:
- **State Management:** Keeps track of all loaded images, calibration points, user-defined Regions of Interest (ROIs), and session measurements in-memory, while simultaneously synchronizing them to local `.json` files in the `/results/` directory for persistence.
- **Smart Primary Detection:** Houses the `detect_meniscus_smart` function, which mathematically projects the horizontal pixel intensities strictly within the user's calibration bounds to find the exact Y-height of the meniscus, heavily penalizing static window borders.
- **Image Annotation:** Takes the raw data from `pvt_analyzer.py` and draws the final, highly-precise overlays (the red measurement line, the dotted magenta physical contour, and the solid cyan quadratic fit) before sending the annotated image back to the frontend.
- **Deployment Ready:** Easily deployable using Gunicorn (`gunicorn src.web_gui:app`).

---

## 3. The User Interface: `templates/auto_swelling.html`

This file is the **Modern Frontend Application**. Built with pure HTML/CSS/JS, it implements a highly polished, professional dark-mode user experience without relying on heavy frontend frameworks.

### Key Responsibilities:
- **Interactive Canvas:** Handles all mouse click events to allow the user to draw green ROI rectangles, drop 2px yellow/cyan calibration dots, or manually override the meniscus height.
- **Dynamic Workflows:** Manages the visual state of the application—toggling between "Select ROI" and "Calibration" modes, updating the results sidebar dynamically via AJAX requests, and managing the image thumbnail gallery at the bottom.
- **Responsive Layout:** Ensures the scientific imagery and sidebars scale perfectly regardless of the monitor size, keeping diagnostic data cleanly separated from the image canvas.

---

## 🚀 Running the Application Locally

**1. Install Requirements**
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the Application**
Launch the Flask development server:
```bash
python src/web_gui.py
```
*The app will be available at `http://localhost:5006`*

---

## 📖 How to Use the Application

The UI is designed to be highly interactive but requires a specific workflow to guarantee scientific accuracy:

### Step 1: Load Image & Set Base
1. Upload your series of PVT images using the **Upload** button.
2. Select the first image in your sequence from the bottom gallery.
3. Click **"Set Base Image"**. This locks in the baseline volume so all subsequent images can have their swelling factor calculated relative to this frame.

### Step 2: Define Region of Interest (ROI)
1. Click the **"Select ROI"** button to activate drawing mode.
2. Click and drag your mouse over the canvas to draw a green rectangle perfectly encompassing the transparent sight-glass window.
3. Keep the bounding box relatively tight to the edges of the fluid column.

### Step 3: Set Calibration Points
1. Click **"Set Calib TOP"** and click on the physical top interior edge of the sight-glass. A 2px yellow dot will appear.
2. Click **"Set Calib BOTTOM"** and click on the physical bottom interior edge of the sight-glass. A 2px cyan dot will appear.
3. Ensure the **Distance (mm)** field accurately reflects the physical distance between these two dots on your specific PVT cell.

### Step 4: Run Automatic Detection
1. Click **"Run Detection"**. 
2. The algorithm will automatically find the exact meniscus height using horizontal projection profiles.
3. It will extract the physical shape of the fluid and draw a **Dotted Magenta Contour** over the true edge.
4. It will fit a **Solid Cyan Quadratic Curve** perfectly to the contour.

### 🛠️ Manual Meniscus Fallback (When Auto Fails)
In rare cases where extreme glare completely blinds the algorithm, the automatic detection may fail or give an improper height.

If this happens, you can physically override the detection:
1. Click the **"Set Meniscus Manual"** button.
2. Click directly on the image canvas exactly where the meniscus line is located.
3. The system will bypass the automatic height detection, jump straight to the exact Y-coordinate you clicked, and run the absolute gradient edge-tracing algorithm locally at that spot to generate the cyan curve and measurement diagnostics.
