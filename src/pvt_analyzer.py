__author__ = "Aditya Kanagalekar"

import cv2
import matplotlib  # ADD THIS
import numpy as np

matplotlib.use(
    "Agg"
)  # ADD THIS (Forces Matplotlib to work in the background without UI)
import csv
import glob
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt


class PVTAnalyzer:
    def __init__(self, standard_height=800):
        # Future GUI & Physical Parameters
        self.temperature = None
        self.pressure = None
        self.time_elapsed = 0.0

        # Optical Corrections
        self.parallax_error_pct = 0.0
        self.refraction_index = 1.333  # Distilled water

        # Image Processing Parameters
        self.standard_height = standard_height
        self.roi_coords = None  # (y1, y2, x1, x2)
        self.manual_mask = (
            None  # optional polygon mask (same size as image) to restrict analysis
        )
        self.scale_mm_per_pixel = None  # physical scale (mm per pixel)
        self.calib_lines = None  # (top_y, bottom_y) in original image coords

    def set_roi(self, y1, y2, x1, x2):
        self.roi_coords = (y1, y2, x1, x2)

    def preprocess_image(self, img):
        """Applies intelligent filters to highlight the meniscus."""
        # 1. Apply ROI Crop (manual polygon mask takes precedence)
        h, w = img.shape[:2]
        if self.manual_mask is not None:
            # compute bounding box from mask
            ys, xs = np.where(self.manual_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                # crop and apply mask
                roi = img[y1 : y2 + 1, x1 : x2 + 1]
                mask_crop = self.manual_mask[y1 : y2 + 1, x1 : x2 + 1]
                roi = cv2.bitwise_and(roi, roi, mask=mask_crop)
                # update roi_coords to bounding box for downstream visualization
                self.roi_coords = (y1, y2 + 1, x1, x2 + 1)
            else:
                roi = img
        elif self.roi_coords:
            y1, y2, x1, x2 = self.roi_coords
            # Validate ROI bounds
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            if y2 <= y1 or x2 <= x1:
                # Invalid ROI, fallback to full image
                roi = img
            else:
                roi = img[y1:y2, x1:x2]
        else:
            roi = img

        # 2. Grayscale & Smoothing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)

        return roi, enhanced

    def set_manual_outline(self, image, points):
        """Set a manual polygonal outline. `points` is a list of (x,y) tuples in image coordinates.
        This creates a mask and updates `manual_mask` and `roi_coords` accordingly.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty polygon provided")
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        self.manual_mask = mask
        self.roi_coords = (y1, y2 + 1, x1, x2 + 1)

    def clear_manual_outline(self):
        self.manual_mask = None

    # ==========================================
    # LITERATURE-BACKED MENISCUS DETECTION
    # (Sensors-21-02676, Sensors-24-07699, SSRN-4399451)
    # ==========================================

    def preprocess_literature_method(self, img):
        """
        Literature-based preprocessing combining Bilateral Filtering, Otsu's Dynamic Thresholding,
        and Morphological Closing to isolate the meniscus band.

        Based on: Sensors-21-02676 (Capacity Measures), Sensors-24-07699, SSRN-4399451

        Args:
            img: Input BGR image

        Returns:
            roi: Cropped region of interest (BGR)
            binary_img: Binary thresholded image with morphological closing applied
            gray_roi: Grayscale ROI for gradient calculations
        """
        h, w = img.shape[:2]

        # Step 1: Apply ROI Crop (manual polygon mask takes precedence)
        if self.manual_mask is not None:
            ys, xs = np.where(self.manual_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                roi = img[y1 : y2 + 1, x1 : x2 + 1]
                mask_crop = self.manual_mask[y1 : y2 + 1, x1 : x2 + 1]
                roi = cv2.bitwise_and(roi, roi, mask=mask_crop)
                self.roi_coords = (y1, y2 + 1, x1, x2 + 1)
            else:
                roi = img
        elif self.roi_coords:
            y1, y2, x1, x2 = self.roi_coords
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            if y2 <= y1 or x2 <= x1:
                roi = img
            else:
                roi = img[y1:y2, x1:x2]
        else:
            roi = img

        # Step 2: Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 3: Bilateral Filtering (preserves edges, removes noise)
        # Parameters tuned for meniscus detection: diameter=9, sigmaColor=75, sigmaSpace=75
        bilateral = cv2.bilateralFilter(gray_roi, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 4: Canny Edge Detection
        # Canny is far more robust than Otsu thresholding for finding translucent fluid interfaces
        edges = cv2.Canny(bilateral, 30, 100)

        # Step 5: Morphological Closing with rectangular kernel
        # Bridges horizontal gaps in the meniscus edge caused by glare on glass
        # Kernel: width=25 (horizontal), height=3 (minimal vertical distortion)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        binary_img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

        logging.debug("Literature preprocessing: Otsu + morphological closing applied")

        return roi, binary_img, gray_roi

    def _find_spanning_contour(self, binary_img, roi_width, span_pct=0.85):
        """
        Find the meniscus contour that spans across the fluid column.

        Literature criterion: A valid meniscus contour MUST have a bounding box width
        of at least 80-85% of the ROI width to be considered a spanning meniscus.

        Args:
            binary_img: Binary image from morphological closing
            roi_width: Width of the ROI (in pixels)
            span_pct: Minimum spanning percentage (default 0.85 = 85%)

        Returns:
            y_coord: Raw pixel Y-coordinate of the meniscus (or None if not found)
            contour: The spanning contour object (or None)
        """
        # Find all contours in the binary image
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            logging.warning("No contours found in binary image")
            return None, None

        roi_height = binary_img.shape[0]
        # Ignore the top and bottom 5% of the ROI to avoid selecting the static glass window borders
        margin_y = int(0.05 * roi_height)

        # Filter contours: select only those that span >= span_pct of roi_width
        spanning_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Skip contours that are too close to the top or bottom edges of the ROI
            if y < margin_y or y > roi_height - margin_y:
                continue
                
            span_ratio = w / roi_width if roi_width > 0 else 0
            if span_ratio >= span_pct:
                spanning_contours.append((cnt, y, w, h, span_ratio))

        if len(spanning_contours) == 0:
            logging.warning(
                f"No spanning contours found (threshold: {span_pct * 100:.0f}% of width)"
            )
            return None, None

        # Sort by Y-coordinate ascending (top-most first) instead of contour area.
        # The true meniscus is at the top of the fluid column.
        # A large area might just be the fluid column itself, which would have its y-coord
        # at the top, but we want to be robust against bright spots at the bottom.
        # So we sort by 'y' (index 1 in the tuple).
        spanning_contours.sort(key=lambda x: x[1])
        best_cnt, best_y, best_w, best_h, best_ratio = spanning_contours[0]

        logging.debug(
            f"Selected spanning contour: Y={best_y}, span_ratio={best_ratio:.2f}"
        )

        return best_y, best_cnt

    def _calculate_subpixel_y(self, gradient_img, peak_y, window_size=5):
        """
        Calculate sub-pixel Y-coordinate using parabola fitting on gradient intensities.

        Literature basis: Sensors-21-02676 emphasizes pixel-level accuracy is insufficient
        for volumetric swelling calculations. Sub-pixel fitting via parabola provides
        floating-point precision crucial for accurate PVT measurements.

        Args:
            gradient_img: Vertical gradient image (e.g., Sobel in Y direction)
            peak_y: Raw pixel Y-coordinate from contour detection
            window_size: Number of pixels above/below peak_y to sample (default 5)

        Returns:
            subpixel_y: Floating-point Y-coordinate with sub-pixel precision
        """
        h, w = gradient_img.shape

        # Clamp window to image bounds
        y_start = max(0, peak_y - window_size)
        y_end = min(h, peak_y + window_size + 1)

        # Extract gradient intensities in the window
        gradient_window = gradient_img[y_start:y_end, :]

        # Sum gradient intensities across columns to get 1D profile
        gradient_profile = np.sum(gradient_window, axis=1)

        # Create x-axis for fitting (relative to y_start)
        x_data = np.arange(len(gradient_profile))

        # Default values for fallback
        coeffs = None
        profile_list = gradient_profile.tolist() if gradient_profile is not None else []

        # Fit a 2nd-degree polynomial (parabola)
        # polyfit returns coefficients [a, b, c] for ax^2 + bx + c
        try:
            coeffs = np.polyfit(x_data, gradient_profile, 2)
            a, b, c = coeffs

            # Vertex of parabola is at x = -b / (2a)
            # This gives us the sub-pixel peak location
            if abs(a) < 1e-6:
                # Parabola is too flat, fall back to peak_y
                subpixel_offset = 0
            else:
                subpixel_offset = -b / (2 * a)

            # Convert back to original image coordinates
            subpixel_y = float(y_start + subpixel_offset)

            logging.debug(
                f"Sub-pixel calculation: peak_y={peak_y}, subpixel_y={subpixel_y:.2f}, offset={subpixel_offset:.3f}"
            )

            return subpixel_y, coeffs.tolist(), profile_list
        except Exception as e:
            logging.warning(
                f"Sub-pixel fitting failed: {e}. Falling back to peak_y={peak_y}"
            )
            return float(peak_y), coeffs, profile_list

    def find_meniscus_literature(self, image_path, visualize=False):
        """
        Master function implementing the complete literature-backed meniscus detection pipeline.

        Combines: Otsu thresholding, morphological closing, spanning contour detection,
        and sub-pixel parabola fitting for highly accurate meniscus localization.

        Reference papers:
        - Sensors-21-02676: Capacity Measures, sub-pixel fitting requirements
        - Sensors-24-07699: Advanced morphological techniques
        - SSRN-4399451: Dynamic thresholding for varying lighting conditions

        Args:
            image_path: Path to the input image
            visualize: If True, generate diagnostic plots (saved to results/ with Agg backend)

        Returns:
            meniscus_y_original: Floating-point Y-coordinate in original image space
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        try:
            # Step 1: Apply literature preprocessing
            roi, binary_img, gray_roi = self.preprocess_literature_method(img)
            roi_h, roi_w = roi.shape[:2]

            # Step 2: Find spanning contour
            # Progressively lower the span percentage requirement if the meniscus is broken
            peak_y_roi = None
            spanning_contour = None
            for span_thresh in [0.85, 0.60, 0.40, 0.25]:
                peak_y_roi, spanning_contour = self._find_spanning_contour(
                    binary_img, roi_w, span_pct=span_thresh
                )
                if peak_y_roi is not None:
                    break

            if peak_y_roi is None:
                raise ValueError("Could not find any spanning meniscus contour even at 25% width")

            # Step 3: Calculate vertical gradient for sub-pixel refinement
            # Use Sobel operator to find intensity gradients
            grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            grad_y = np.absolute(grad_y)

            # Step 4: Sub-pixel parabola fitting
            subpixel_y_roi, coeffs, profile = self._calculate_subpixel_y(
                grad_y, int(peak_y_roi), window_size=5
            )

            # Step 5: Translate back to original image coordinates
            if self.roi_coords:
                y1_offset = self.roi_coords[0]
            else:
                y1_offset = 0

            meniscus_y_original = y1_offset + peak_y_roi
            subpixel_y_original = y1_offset + subpixel_y_roi

            logging.info(
                f"Literature method: detected meniscus at Y={subpixel_y_original:.2f} (sub-pixel)"
            )

            # Contour formatting and spatial polynomial fit
            contour_pts = []
            spatial_coeffs = None
            if spanning_contour is not None and len(spanning_contour) > 2:
                xs = []
                ys = []
                for pt in spanning_contour:
                    x_val = float(pt[0][0] + (self.roi_coords[2] if self.roi_coords else 0))
                    y_val = float(pt[0][1] + y1_offset)
                    contour_pts.append({"x": x_val, "y": y_val})
                    xs.append(x_val)
                    ys.append(y_val)
                
                # Fit spatial polynomial y = Ax^2 + Bx + C
                try:
                    # Filter unique xs to prevent issues with vertical edges
                    unique_xs, indices = np.unique(xs, return_index=True)
                    unique_ys = np.array(ys)[indices]
                    if len(unique_xs) >= 3:
                        spatial_coeffs = np.polyfit(unique_xs, unique_ys, 2).tolist()
                except Exception as e:
                    logging.warning(f"Spatial polynomial fit failed: {e}")

            diagnostics = {
                "detected_y": float(meniscus_y_original),
                "subpixel_y": float(subpixel_y_original),
                "coeffs": coeffs,
                "spatial_coeffs": spatial_coeffs,
                "profile": profile,
                "contour": contour_pts
            }

            # Optional visualization (using Agg backend for macOS compatibility)
            if visualize:
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                    # Binary image with contour
                    axes[0, 0].imshow(binary_img, cmap="gray")
                    if spanning_contour is not None:
                        cv2.drawContours(binary_img, [spanning_contour], -1, 128, 2)
                        axes[0, 0].axhline(
                            y=peak_y_roi,
                            color="r",
                            linestyle="--",
                            label=f"Raw peak Y={peak_y_roi:.0f}",
                        )
                        axes[0, 0].axhline(
                            y=subpixel_y_roi,
                            color="g",
                            linestyle="-",
                            linewidth=2,
                            label=f"Sub-pixel Y={subpixel_y_roi:.2f}",
                        )
                    axes[0, 0].set_title("Binary Image + Spanning Contour")
                    axes[0, 0].legend()

                    # Gradient image
                    axes[0, 1].imshow(cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB))
                    axes[0, 1].axhline(
                        y=subpixel_y_roi, color="g", linestyle="-", linewidth=2
                    )
                    axes[0, 1].set_title("Grayscale ROI with Meniscus")

                    # Gradient profile
                    grad_profile = np.sum(grad_y, axis=1)
                    axes[1, 0].plot(grad_profile)
                    axes[1, 0].axvline(
                        x=peak_y_roi, color="r", linestyle="--", label="Raw peak"
                    )
                    axes[1, 0].axvline(
                        x=subpixel_y_roi,
                        color="g",
                        linestyle="-",
                        linewidth=2,
                        label="Sub-pixel",
                    )
                    axes[1, 0].set_title("Vertical Gradient Profile")
                    axes[1, 0].set_xlabel("Row (Y)")
                    axes[1, 0].set_ylabel("Gradient Sum")
                    axes[1, 0].legend()

                    # ROI with detected meniscus
                    roi_display = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    axes[1, 1].imshow(roi_display)
                    axes[1, 1].axhline(
                        y=subpixel_y_roi,
                        color="g",
                        linestyle="-",
                        linewidth=2,
                        label=f"Detected: {subpixel_y_roi:.2f}",
                    )
                    axes[1, 1].set_title("ROI with Detected Meniscus")
                    axes[1, 1].legend()

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            os.getcwd(), "results", "meniscus_literature_detection.png"
                        ),
                        dpi=150,
                    )
                    plt.close()
                    logging.info(
                        "Visualization saved to results/meniscus_literature_detection.png"
                    )
                except Exception as e:
                    logging.warning(f"Visualization failed: {e}")

            return diagnostics

        except Exception as e:
            logging.error(f"Literature method failed: {e}")
            raise

    def find_meniscus_edge(self, image_path, visualize=False):
        """
        Locates the exact pixel height of the fluid meniscus.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        # Get the pre-processed image
        roi, preprocessed = self.preprocess_image(img)

        # ---------------------------------------------------------
        # STEP 3: Edge Detection (Horizontal Projection Profile)
        # ---------------------------------------------------------

        # A. Sobel Filter (Vertical Gradients)
        # We only care about horizontal lines, so we compute the derivative in the Y direction (dy=1, dx=0)
        sobel_y = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)

        # Take the absolute value to capture both dark-to-light and light-to-dark transitions
        abs_sobel_y = np.absolute(sobel_y)
        sobel_8u = np.uint8(abs_sobel_y)

        # B. Horizontal Projection Profile
        # Sum the gradient intensities across each row
        row_sums = np.sum(sobel_8u, axis=1)

        # C. Find the Peak
        # The row with the maximum sum is the primary horizontal edge (the meniscus)
        meniscus_y_roi = np.argmax(row_sums)

        # If an ROI was used, translate that Y coordinate back to the original image space
        meniscus_y_original = meniscus_y_roi + (
            self.roi_coords[0] if self.roi_coords else 0
        )

        if visualize:
            self._visualize_edge_detection(roi, sobel_8u, row_sums, meniscus_y_roi)

        return meniscus_y_original

    def find_meniscus_edge_advanced(self, image_path, visualize=False):
        """More robust detection:
        - Preprocess as usual
        - Compute Canny edges + morphological closing
        - Use HoughLinesP to detect horizontal line segments and pick candidate Y
        - If Hough fails, use smoothed horizontal projection and subpixel parabola fit
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        roi, preprocessed = self.preprocess_image(img)

        # 1. Canny edges
        blur = cv2.GaussianBlur(preprocessed, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. Morphological close to join horizontal edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 3. Hough line detection for horizontal segments
        lines = cv2.HoughLinesP(
            closed,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )
        candidate_ys = []
        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = l
                dy = y2 - y1
                dx = x2 - x1
                angle = np.arctan2(dy, dx)
                # keep near-horizontal lines (angle near 0)
                if abs(angle) < np.deg2rad(10):
                    candidate_ys.append((y1 + y2) // 2)

        if candidate_ys:
            # use median of candidate Ys for robustness
            meniscus_y_roi = int(np.median(candidate_ys))
        else:
            # fallback: smoothed horizontal projection profile with subpixel refinement
            sobel_y = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)
            abs_sobel_y = np.absolute(sobel_y)
            sobel_8u = np.uint8(abs_sobel_y)
            row_sums = np.sum(sobel_8u, axis=1)

            # smooth 1D profile with gaussian-like kernel
            kernel_size = max(9, preprocessed.shape[0] // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gauss_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 6).flatten()
            smoothed = np.convolve(
                row_sums, gauss_kernel / gauss_kernel.sum(), mode="same"
            )

            peak = int(np.argmax(smoothed))
            # refine with quadratic fit around peak
            left = max(0, peak - 2)
            right = min(len(smoothed) - 1, peak + 2)
            xs = np.arange(left, right + 1)
            ys = smoothed[left : right + 1]
            if len(xs) >= 3:
                coeffs = np.polyfit(xs, ys, 2)  # a*x^2 + b*x + c
                a, b, c = coeffs
                if a != 0:
                    vertex = -b / (2 * a)
                    meniscus_y_roi = int(np.clip(vertex, 0, preprocessed.shape[0] - 1))
                else:
                    meniscus_y_roi = peak
            else:
                meniscus_y_roi = peak

        # translate back to original image coordinates
        meniscus_y_original = meniscus_y_roi + (
            self.roi_coords[0] if self.roi_coords else 0
        )

        if visualize:
            # recreate sobel for visualization
            sobel_viz = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)
            sobel_viz = np.uint8(np.absolute(sobel_viz))
            self._visualize_edge_detection(
                roi, sobel_viz, np.sum(sobel_viz, axis=1), meniscus_y_roi
            )

        return meniscus_y_original

    def auto_crop_to_column(self, image_path, margin_ratio=0.05, visualize=False):
        """Automatically find the vertical narrow column containing the fluid and return ROI coords (y1,y2,x1,x2).
        margin_ratio: expand detected column width by this fraction on each side.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use vertical edges to find container boundaries
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        col_strength = np.sum(np.absolute(sobel_x), axis=0)

        h, w = img.shape[:2]

        # Try several percentile thresholds to find a reasonably narrow column
        found = False
        left = right = None
        for p in (99, 97, 95, 90, 85):
            thresh = np.percentile(col_strength, p)
            mask = col_strength >= thresh
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            l = int(indices[0])
            r = int(indices[-1])
            width = r - l
            # prefer segments that are not almost the full image width
            if width < 0.6 * w:
                left, right = l, r
                found = True
                break

        if not found:
            # fallback to the strongest local peak (narrow window around the peak)
            peak = int(np.argmax(col_strength))
            half_width = max(10, w // 40)
            x1 = max(0, peak - half_width)
            x2 = min(w, peak + half_width)
            return 0, h, x1, x2

        # expand selected region slightly
        width = right - left
        expand = int(max(2, width * margin_ratio))
        x1 = max(0, left - expand)
        x2 = min(w, right + expand)
        y1 = 0
        y2 = h

        if visualize:
            viz = img.copy()
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Auto Crop", cv2.resize(viz, (800, 800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return y1, y2, x1, x2

    def detect_tube_roi(self, image_path, visualize=False):
        """Detect the central test-tube-like region using contour analysis.
        Returns (y1,y2,x1,x2). Raises if nothing suitable is found.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive threshold to capture the tube body against background
        th = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
        )

        # Morphological cleaning: close then open to form contiguous shapes
        ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ker_close)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ker_open)

        # Find contours
        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Evaluate contours: prefer tall, central contours
        candidates = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if ww <= 0 or hh <= 0:
                continue
            aspect = hh / float(ww)
            cx = x + ww / 2
            # require tall and reasonably large area
            if aspect > 2.0 and hh > 0.4 * h and area > 0.01 * h * w:
                # prefer contours near center
                dist_center = abs(cx - w / 2)
                candidates.append((dist_center, area, (y, y + hh, x, x + ww)))

        if not candidates:
            # fallback: try looser constraints (allow smaller height)
            for c in contours:
                x, y, ww, hh = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if ww <= 0 or hh <= 0:
                    continue
                aspect = hh / float(ww)
                cx = x + ww / 2
                if aspect > 1.5 and hh > 0.25 * h and area > 0.005 * h * w:
                    dist_center = abs(cx - w / 2)
                    candidates.append((dist_center, area, (y, y + hh, x, x + ww)))

        if not candidates:
            raise RuntimeError("Could not detect tube-like contour")

        # choose candidate closest to center and largest area
        candidates.sort(key=lambda t: (t[0], -t[1]))
        _, _, (y1, y2, x1, x2) = candidates[0]

        # expand slightly
        margin_x = int((x2 - x1) * 0.08)
        x1 = max(0, x1 - margin_x)
        x2 = min(w, x2 + margin_x)
        y1 = max(0, y1 - int((y2 - y1) * 0.03))
        y2 = min(h, y2 + int((y2 - y1) * 0.03))

        if visualize:
            viz = img.copy()
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Detected Tube ROI", cv2.resize(viz, (800, 800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return int(y1), int(y2), int(x1), int(x2)

    def interactive_polygon_selector(self, image, display_size=800):
        """Allow the user to draw a polygon on a scaled preview. Returns list of (x,y) in original image coords or None if cancelled."""
        orig_h, orig_w = image.shape[:2]
        scale = display_size / max(orig_h, orig_w)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        disp = cv2.resize(image, (disp_w, disp_h)).copy()
        clone = disp.copy()

        points = []

        def mouse_cb(event, x, y, flags, param):
            nonlocal points, disp
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(disp, points[-2], points[-1], (0, 255, 0), 2)
                cv2.circle(disp, (x, y), 3, (0, 0, 255), -1)

        win = "Draw polygon: left-click to add points, c=close, r=reset, q=cancel"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                # close polygon if >=3 points
                if len(points) >= 3:
                    # draw closing line
                    cv2.line(disp, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow(win, disp)
                    cv2.waitKey(200)
                    break
            elif key == ord("r"):
                points = []
                disp = clone.copy()
            elif key == ord("q"):
                points = None
                break

        cv2.destroyWindow(win)
        if not points:
            return None

        # scale points back to original image coordinates
        scaled = []
        inv_scale = 1.0 / scale
        for x, y in points:
            ox = int(x * inv_scale)
            oy = int(y * inv_scale)
            scaled.append((ox, oy))
        return scaled

    def interactive_set_scale(self, image, mm_between=92, display_size=800):
        """Let user click two points (top and bottom) on a scaled preview to set the physical
        distance between them to `mm_between` millimeters. Stores `scale_mm_per_pixel` and `calib_lines`.
        Returns (top_y, bottom_y, scale_mm_per_pixel).
        """
        orig_h, orig_w = image.shape[:2]
        scale = display_size / max(orig_h, orig_w)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        disp = cv2.resize(image, (disp_w, disp_h)).copy()
        clone = disp.copy()

        points = []

        def mouse_cb(event, x, y, flags, param):
            nonlocal points, disp
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(disp, points[-2], points[-1], (0, 255, 0), 2)

        win = f"Select TOP then BOTTOM (physical distance = {mm_between} mm). q=cancel"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                points = None
                break
            if len(points) >= 2:
                break

        cv2.destroyWindow(win)
        if not points or len(points) < 2:
            raise RuntimeError("Scale selection cancelled or insufficient points")

        # scale back to original coordinates
        inv_scale = 1.0 / scale
        top_x, top_y = points[0]
        bot_x, bot_y = points[1]
        top_y_orig = int(top_y * inv_scale)
        bot_y_orig = int(bot_y * inv_scale)

        pixel_dist = abs(bot_y_orig - top_y_orig)
        if pixel_dist <= 0:
            raise RuntimeError("Invalid pixel distance for calibration")

        scale_mm_per_pixel = float(mm_between) / float(pixel_dist)
        self.scale_mm_per_pixel = scale_mm_per_pixel
        self.calib_lines = (min(top_y_orig, bot_y_orig), max(top_y_orig, bot_y_orig))

        return (self.calib_lines[0], self.calib_lines[1], self.scale_mm_per_pixel)

    def detect_tube_roi_spanning(
        self, image_path, visualize=False, wall_ksize=5, buffer_pct=0.10
    ):
        """Identify the inner glass walls by summing vertical Sobel responses and crop strictly
        between them (with a small inward buffer). Returns (y1,y2,x1,x2).
        This implements the two-step 'tube rectangle' part of the spanning-contour approach.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sobel X to emphasize vertical edges (glass walls)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=wall_ksize)
        abs_sobel = np.absolute(sobel_x)
        col_sums = np.sum(abs_sobel, axis=0)

        # Smooth the 1D column strength profile to reduce spurious local peaks
        kernel_size = max(11, w // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gauss = cv2.getGaussianKernel(kernel_size, kernel_size / 3.0).flatten()
        smooth = np.convolve(col_sums, gauss / gauss.sum(), mode="same")

        # find local maxima in smoothed signal
        peaks = []
        for i in range(1, w - 1):
            if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
                peaks.append((int(i), float(smooth[i])))

        if not peaks:
            # fallback to contour detector if no clear peaks
            logging.debug(
                "No local peaks in vertical-edge profile; falling back to contour-based tube detector"
            )
            return self.detect_tube_roi(image_path, visualize=visualize)

        # sort peaks by strength (descending)
        peaks.sort(key=lambda t: t[1], reverse=True)

        # Consider top-N peaks and choose a pair with sufficient separation
        min_sep = max(
            8, int(w * 0.12)
        )  # require at least 12% of image width between walls
        chosen = None
        top_n = min(len(peaks), 8)
        for i in range(top_n):
            for j in range(i + 1, top_n):
                left_idx = peaks[i][0]
                right_idx = peaks[j][0]
                if right_idx <= left_idx:
                    left_idx, right_idx = right_idx, left_idx
                sep = right_idx - left_idx
                if sep >= min_sep and sep < 0.9 * w:
                    chosen = (left_idx, right_idx)
                    break
            if chosen:
                break

        if chosen is None:
            # try global peaks approach
            sorted_idx = np.argsort(smooth)[-2:]
            sorted_idx = np.sort(sorted_idx)
            left_idx, right_idx = int(sorted_idx[0]), int(sorted_idx[1])
            if right_idx - left_idx < min_sep:
                logging.warning(
                    "Peaks too close or ambiguous (sep=%d). Falling back to contour-based tube detector",
                    right_idx - left_idx,
                )
                return self.detect_tube_roi(image_path, visualize=visualize)
        else:
            left_idx, right_idx = chosen

        # inward buffer to avoid glass edges
        buf = int((right_idx - left_idx) * buffer_pct)
        x1 = max(0, left_idx + buf)
        x2 = min(w, right_idx - buf)
        y1 = 0
        y2 = h

        if x2 <= x1 + 5:
            logging.warning(
                "Computed tube crop too small after buffering; falling back to contour-based detector"
            )
            return self.detect_tube_roi(image_path, visualize=visualize)

        if visualize:
            viz = img.copy()
            cv2.line(viz, (x1, 0), (x1, h), (0, 255, 0), 2)
            cv2.line(viz, (x2, 0), (x2, h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            plt.title("Detected inner tube walls (refined)")
            plt.show()

        return int(y1), int(y2), int(x1), int(x2)

    def find_meniscus_by_spanning(
        self, image_path, visualize=False, canny_lo=30, canny_hi=100, span_pct=0.85
    ):
        """After cropping between inner walls, find a contour that spans the width
        of the cropped region and return its vertical center in original image coords.
        Returns meniscus_y (int) or raises if none found.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        # Step 1: detect tube rectangle
        y1, y2, x1, x2 = self.detect_tube_roi_spanning(image_path, visualize=visualize)
        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            raise RuntimeError("Empty crop when attempting spanning contour detection")

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)

        edges = cv2.Canny(enhanced, canny_lo, canny_hi)

        # Close small horizontal breaks so the meniscus becomes a contiguous contour
        kernel = np.ones((3, 15), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        roi_width = cropped.shape[1]
        spanning = []
        for cnt in contours:
            x, y, w, hcnt = cv2.boundingRect(cnt)
            if w >= span_pct * roi_width:
                spanning.append((cnt, y + hcnt // 2))

        if not spanning:
            raise RuntimeError("No spanning contour found in cropped tube region")

        # pick the contour with the largest bounding-box width (most spanning)
        spanning.sort(key=lambda t: cv2.boundingRect(t[0])[2], reverse=True)
        best_cnt, center_y = spanning[0]

        # convert center_y back to original image coordinates
        meniscus_y_orig = int(center_y + y1)

        if visualize:
            viz = cropped.copy()
            cv2.drawContours(viz, [best_cnt], -1, (0, 0, 255), 3)
            plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            plt.title(
                f"Spanning contour (center Y in original image = {meniscus_y_orig})"
            )
            plt.show()

        return meniscus_y_orig

    def find_meniscus_contour(
        self, img, meniscus_y, span_pct=0.10, visualize=False, canny_lo=30, canny_hi=100
    ):
        """Given an already-drawn meniscus Y (original-image coords), find the exact shape
        by scanning for the peak vertical gradient in each column near the meniscus_y.
        Returns the contour in original-image coordinates (ndarray), or None if not found.
        """
        
        h_img, w_img = img.shape[:2]
        if self.roi_coords:
            y1, y2, x1, x2 = self.roi_coords
            crop = img[y1:y2, x1:x2].copy()
            y_offset = y1
        else:
            crop = img.copy()
            x1 = 0
            y_offset = 0

        if crop.size == 0:
            return None

        # 1. Grayscale and smooth
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Vertical gradient (absolute) to find horizontal edges
        sob = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=5)
        abs_sob = np.abs(sob)
        
        # 3. Mask out everything except a [-100, +100] band around rel_y
        rel_y = int(meniscus_y - y_offset)
        band_radius = 100
        y_min = max(0, rel_y - band_radius)
        y_max = min(crop.shape[0], rel_y + band_radius)
        
        mask = np.zeros_like(abs_sob)
        mask[y_min:y_max, :] = 1.0
        abs_sob_masked = abs_sob * mask
        
        # 4. Find peak gradient in each column
        pts = []
        for x_col in range(crop.shape[1]):
            col_data = abs_sob_masked[:, x_col]
            y_peak = int(np.argmax(col_data))
            # Require at least some minimal gradient to avoid empty noise columns
            if col_data[y_peak] > 10.0:
                pts.append([x_col, y_peak])
                
        if len(pts) < crop.shape[1] * span_pct:
            return None
            
        # 5. Median filter the Y coordinates to remove isolated noise spikes
        pts = np.array(pts)
        xs = pts[:, 0]
        ys = pts[:, 1]
        
        # Simple median filter using numpy
        window_size = 15
        pad = window_size // 2
        ys_padded = np.pad(ys, (pad, pad), mode='edge')
        ys_smooth = np.zeros_like(ys)
        for i in range(len(ys)):
            ys_smooth[i] = np.median(ys_padded[i:i+window_size])
        
        # 6. Format as OpenCV contour shape (N, 1, 2)
        best_cnt = np.zeros((len(xs), 1, 2), dtype=np.int32)
        best_cnt[:, 0, 0] = xs
        best_cnt[:, 0, 1] = ys_smooth
        
        # Offset to original-image coords
        best_cnt_orig = best_cnt.copy()
        best_cnt_orig[:, 0, 0] = best_cnt_orig[:, 0, 0] + x1
        best_cnt_orig[:, 0, 1] = best_cnt_orig[:, 0, 1] + y_offset

        return best_cnt_orig

    def process_and_annotate(
        self,
        image_path,
        out_path=None,
        method="advanced",
        visualize=False,
        auto_crop=False,
        verbose=False,
        extract_contour=False,
    ):
        """Process one image, detect meniscus, optionally extract the meniscus contour,
        create annotated output, and return result dict.
        out_path: full path for annotated image. If None, saves beside input as <base>_detected.png
        """
        if verbose:
            logging.info(
                f"Processing {image_path} (method={method}, auto_crop={auto_crop})"
            )

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        # Require a manual ROI unless auto_crop is explicitly enabled
        if not auto_crop and not self.roi_coords:
            raise ValueError(
                f"No ROI set for {image_path} and auto_crop is disabled. Provide a manual ROI or enable auto-crop."
            )

        # optionally auto-crop to column if enabled and no manual ROI exists
        if auto_crop and not self.roi_coords:
            try:
                y1, y2, x1, x2 = self.auto_crop_to_column(image_path)
                self.set_roi(y1, y2, x1, x2)
            except Exception as ce:
                logging.warning("Auto-crop failed, using full image: %s", ce)

        # choose method
        if extract_contour:
            # Prefer the spanning-contour approach when extracting a contour
            try:
                y = self.find_meniscus_by_spanning(image_path, visualize=visualize)
            except Exception:
                # fallback to advanced then basic
                try:
                    y = self.find_meniscus_edge_advanced(
                        image_path, visualize=visualize
                    )
                except Exception:
                    y = self.find_meniscus_edge(image_path, visualize=visualize)
        else:
            if method == "literature":
                # Literature-backed method: Otsu + morphological + spanning + sub-pixel
                try:
                    y = self.find_meniscus_literature(image_path, visualize=visualize)
                except Exception as e:
                    logging.warning(
                        f"Literature method failed: {e}. Falling back to spanning."
                    )
                    try:
                        y = self.find_meniscus_by_spanning(
                            image_path, visualize=visualize
                        )
                    except Exception:
                        y = self.find_meniscus_edge(image_path, visualize=visualize)
            elif method == "spanning":
                try:
                    y = self.find_meniscus_by_spanning(image_path, visualize=visualize)
                except Exception:
                    y = self.find_meniscus_edge(image_path, visualize=visualize)
            elif method == "advanced":
                y = self.find_meniscus_edge_advanced(image_path, visualize=visualize)
            else:
                y = self.find_meniscus_edge(image_path, visualize=visualize)

        # Return diagnostics natively if returned from literature method, else construct simple dict
        diagnostics = {}
        if isinstance(y, dict):
            diagnostics = y
            y_float = diagnostics.get("subpixel_y", diagnostics.get("detected_y"))
        else:
            y_float = float(y)
            diagnostics = {"detected_y": y_float, "subpixel_y": y_float}

        # prepare annotated image
        annotated = img.copy()
        # draw meniscus line (ensure y is integer for cv2 operations)
        y_int = int(y_float)
        cv2.line(annotated, (0, y_int), (annotated.shape[1] - 1, y_int), (0, 0, 255), 3)
        # draw ROI
        if self.roi_coords:
            y1, y2, x1, x2 = self.roi_coords
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if out_path:
            cv2.imwrite(out_path, annotated)
        else:
            base, ext = os.path.splitext(image_path)
            cv2.imwrite(f"{base}_detected{ext}", annotated)

        result = {
            "detected_y": y_float,
            "method": method,
            "annotated_image_path": out_path or f"{base}_detected{ext}",
            "diagnostics": diagnostics
        }

        contour_info = None
        height_mm = None
        if extract_contour:
            try:
                cnt = self.find_meniscus_contour(img, y, visualize=visualize)
                if cnt is not None:
                    cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 2)
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    contour_info = {
                        "bbox": (int(bx), int(by), int(bw), int(bh)),
                        "points": int(cnt.shape[0]),
                    }
            except Exception as e:
                logging.warning(
                    "Failed to extract meniscus contour for %s: %s", image_path, e
                )

        # Calculate height in mm if calibration available
        if self.scale_mm_per_pixel is not None and self.calib_lines is not None:
            top_y, bottom_y = self.calib_lines
            # compute vertical distance from bottom to meniscus (positive when meniscus below top)
            # height of fluid measured from bottom upwards
            height_pixels = max(0, bottom_y - int(y))
            height_mm = float(height_pixels) * float(self.scale_mm_per_pixel)
            # annotate text on image
            text = f"Height: {height_mm:.2f} mm"
            cv2.putText(
                annotated,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            # draw calibration lines
            cv2.line(
                annotated, (0, top_y), (annotated.shape[1] - 1, top_y), (0, 255, 255), 2
            )
            cv2.line(
                annotated,
                (0, bottom_y),
                (annotated.shape[1] - 1, bottom_y),
                (0, 255, 255),
                2,
            )

        if out_path is None:
            out_dir = os.path.dirname(image_path) or "."
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_dir, f"{base}_detected.png")

        cv2.imwrite(out_path, annotated)
        if verbose:
            logging.info(f"Saved annotated image to {out_path}")

        res = {
            "image": image_path,
            "detected_y": int(y),
            "roi": self.roi_coords,
            "out_path": out_path,
        }
        if contour_info is not None:
            res["contour_bbox"] = contour_info["bbox"]
            res["contour_points"] = contour_info["points"]
        if height_mm is not None:
            res["height_mm"] = float(height_mm)
        return res

    def _visualize_edge_detection(self, roi, sobel, row_sums, meniscus_y):
        """Internal helper to visualize the math behind the detection."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # Draw a red line on the ROI where the meniscus was detected
        roi_display = roi.copy()
        cv2.line(
            roi_display,
            (0, meniscus_y),
            (roi_display.shape[1], meniscus_y),
            (0, 0, 255),
            2,
        )

        axes[0].imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Detected Meniscus (Y = {meniscus_y})")

        axes[1].imshow(sobel, cmap="gray")
        axes[1].set_title("Sobel Vertical Gradients")

        # Plot the row sums alongside the Y-axis to show the "peak"
        y_coords = np.arange(len(row_sums))
        axes[2].plot(row_sums, y_coords)
        axes[2].set_ylim(len(row_sums), 0)  # Invert Y axis to match image coordinates
        axes[2].set_title("Horizontal Projection Profile")
        axes[2].set_xlabel("Gradient Sum")
        axes[2].set_ylabel("Pixel Row")

        plt.tight_layout()
        plt.show()


# ==========================================
# Execution Block with CLI + headless fallback
# ==========================================
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="PVT meniscus analyzer")
    parser.add_argument(
        "--image",
        "-i",
        default="imgdata/water/img1.png",
        help="Path to the input image",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without interactive ROI selection"
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("y1", "y2", "x1", "x2"),
        help="Provide ROI coordinates directly (y1 y2 x1 x2) for headless mode",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show visualization (plots/windows)"
    )
    parser.add_argument(
        "--method",
        choices=["basic", "advanced"],
        default="advanced",
        help="Detection method to use",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Interactive GUI: choose ROI and optionally calibrate scale in mm",
    )
    parser.add_argument(
        "--calibrate-mm",
        type=float,
        default=92.0,
        help="Physical distance in mm between top and bottom calibration points",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output directory for annotated images and CSV",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If image path is a directory or --batch, process all images in the folder",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    # NOTE: Auto-cropping has been disabled to enforce manual ROI selection
    parser.add_argument(
        "--extract-contour",
        action="store_true",
        help="Extract the meniscus contour near the detected meniscus and overlay it",
    )
    args = parser.parse_args()

    analyzer = PVTAnalyzer()
    image_path = args.image

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        raise SystemExit(1)

    results = []

    # Determine processing list (single file or directory)
    targets = []
    if os.path.isdir(image_path) or args.batch:
        # process all common image types in the directory
        search_dir = (
            image_path if os.path.isdir(image_path) else os.path.dirname(image_path)
        )
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
        for pat in patterns:
            targets.extend(sorted(glob.glob(os.path.join(search_dir, pat))))
    else:
        targets = [image_path]

    # filter out previously-generated annotated files to avoid double-processing
    targets = [t for t in targets if not t.endswith("_detected.png")]

    out_dir = args.out or os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.headless:
        # Headless mode: require a provided manual ROI. Auto-cropping is disabled.
        if args.roi:
            y1, y2, x1, x2 = args.roi
            analyzer.set_roi(y1, y2, x1, x2)
            print(f"Using provided ROI: {args.roi}")
        else:
            logging.error(
                "Headless run without --roi: manual ROI is required. Use interactive mode or provide --roi y1 y2 x1 x2"
            )
            for target in targets:
                logging.error("Skipping %s because no ROI was provided.", target)
            csv_path = os.path.join(out_dir, "results_summary.csv")
            with open(csv_path, "w", newline="") as csvfile:
                fieldnames = [
                    "image",
                    "detected_y",
                    "roi",
                    "out_path",
                    "contour_bbox",
                    "contour_points",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            logging.info("Wrote empty summary CSV to %s", csv_path)
            raise SystemExit(1)

        for target in targets:
            try:
                base_out = os.path.join(
                    out_dir,
                    os.path.splitext(os.path.basename(target))[0] + "_detected.png",
                )
                # Respect manual ROI and use the basic projection method
                res = analyzer.process_and_annotate(
                    target,
                    out_path=base_out,
                    method="basic",
                    visualize=args.visualize,
                    auto_crop=False,
                    verbose=args.verbose,
                    extract_contour=args.extract_contour,
                )
                results.append(res)
                logging.info("Processed %s -> %s", target, res["out_path"])
            except Exception as e:
                logging.error("Failed to process %s: %s", target, e)
    else:
        # Interactive mode: let the user choose rectangle or polygon outline
        img_for_roi = cv2.imread(image_path)
        if img_for_roi is None:
            print("Image could not be loaded for interactive ROI selection")
            raise SystemExit(1)

        display_size = 800
        print("INSTRUCTIONS:")
        print(" - Choose ROI method:")
        print("   [r] Rectangle (draw a tall narrow box)")
        print("   [p] Polygon (click to add polygon points; press 'c' to close)")
        print("   [a] Auto center narrow ROI fallback")

        choice = input("Select ROI mode (r/p/a) [r]: ").strip().lower() or "r"

        if choice == "p":
            pts = analyzer.interactive_polygon_selector(
                img_for_roi, display_size=display_size
            )
            if pts:
                try:
                    analyzer.set_manual_outline(img_for_roi, pts)
                    print("Manual polygon outline set.")
                except Exception as e:
                    print("Failed to set polygon outline:", e)
                    print("Falling back to auto center narrow ROI.")
                    h, w = img_for_roi.shape[:2]
                    center_x = w // 2
                    half_width = max(5, w // 40)
                    x1 = max(0, center_x - half_width)
                    x2 = min(w, center_x + half_width)
                    analyzer.set_roi(0, h, x1, x2)
            else:
                print(
                    "Polygon selection cancelled — falling back to auto center narrow ROI."
                )
                h, w = img_for_roi.shape[:2]
                center_x = w // 2
                half_width = max(5, w // 40)
                x1 = max(0, center_x - half_width)
                x2 = min(w, center_x + half_width)
                analyzer.set_roi(0, h, x1, x2)

        elif choice == "r":
            display_img = cv2.resize(img_for_roi, (display_size, display_size))
            roi_box = cv2.selectROI(
                "Draw Narrow ROI (Press Enter to confirm)",
                display_img,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyAllWindows()
            x_start, y_start, width, height = roi_box
            # If user cancels or selects an empty ROI (width or height == 0), fallback to a center narrow ROI
            if width == 0 or height == 0:
                print(
                    "No ROI selected or selection cancelled — falling back to auto center narrow ROI."
                )
                h, w = img_for_roi.shape[:2]
                center_x = w // 2
                half_width = max(5, w // 40)
                x1 = max(0, center_x - half_width)
                x2 = min(w, center_x + half_width)
                analyzer.set_roi(0, h, x1, x2)
            else:
                scale_y = img_for_roi.shape[0] / display_size
                scale_x = img_for_roi.shape[1] / display_size
                y1 = int(y_start * scale_y)
                y2 = int((y_start + height) * scale_y)
                x1 = int(x_start * scale_x)
                x2 = int((x_start + width) * scale_x)
                analyzer.set_roi(y1, y2, x1, x2)

        else:
            # auto center narrow ROI
            h, w = img_for_roi.shape[:2]
            center_x = w // 2
            half_width = max(5, w // 40)
            x1 = max(0, center_x - half_width)
            x2 = min(w, center_x + half_width)
            analyzer.set_roi(0, h, x1, x2)
            print(f"Auto ROI set to center narrow column: x1={x1}, x2={x2}")

        # If GUI mode requested, allow calibration of physical scale
        if args.gui:
            try:
                resp = (
                    input(
                        f"Calibrate physical distance (click TOP then BOTTOM) for {args.calibrate_mm} mm? (y/N): "
                    )
                    .strip()
                    .lower()
                )
            except Exception:
                resp = "n"
            if resp == "y":
                try:
                    top_y, bottom_y, scale = analyzer.interactive_set_scale(
                        img_for_roi,
                        mm_between=args.calibrate_mm,
                        display_size=display_size,
                    )
                    print(
                        f"Calibration set: top={top_y}, bottom={bottom_y}, scale={scale:.6f} mm/px"
                    )
                except Exception as e:
                    print("Calibration failed:", e)

        for target in targets:
            try:
                base_out = os.path.join(
                    out_dir,
                    os.path.splitext(os.path.basename(target))[0] + "_detected.png",
                )
                # Respect manual ROI and use the basic projection method when processing interactively
                res = analyzer.process_and_annotate(
                    target,
                    out_path=base_out,
                    method="basic",
                    visualize=args.visualize,
                    auto_crop=False,
                    verbose=args.verbose,
                    extract_contour=args.extract_contour,
                )
                results.append(res)
                logging.info("Processed %s -> %s", target, res["out_path"])
            except Exception as e:
                logging.error("Failed to process %s: %s", target, e)

    # write CSV summary
    csv_path = os.path.join(out_dir, "results_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "image",
            "detected_y",
            "roi",
            "out_path",
            "contour_bbox",
            "contour_points",
            "height_mm",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logging.info("Wrote summary CSV to %s", csv_path)

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

    return out
