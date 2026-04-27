import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png' 
OUTPUT_PATH = 'results/contour_selection_stages.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}.")
        return

    # 1. Pipeline stages
    raw_img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Get contours for Panel A
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Use Gradient for Panel B (to find the actual interface line)
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobel_y)
    
    # Find peak row in the middle 50% of the image
    h, w = abs_sobel.shape
    roi_top, roi_bottom = int(h*0.3), int(h*0.7)
    column_sum = np.sum(abs_sobel[roi_top:roi_bottom, :], axis=1)
    peak_y = np.argmax(column_sum) + roi_top

    # --- Visualizations ---
    vis_a = raw_img.copy()
    cv2.drawContours(vis_a, contours, -1, (0, 0, 255), 2)

    vis_b = raw_img.copy()
    # Draw all contours faint
    cv2.drawContours(vis_b, contours, -1, (100, 100, 150), 1)
    # Draw the detected meniscus line in Cyan
    cv2.line(vis_b, (int(w*0.1), peak_y), (int(w*0.9), peak_y), (255, 255, 0), 4)
    # Add a marker for clarity
    cv2.circle(vis_b, (int(w/2), peak_y), 8, (255, 255, 0), -1)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].imshow(cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB))
    axes[0].set_title('(a) Extracted Contours\n(Unfiltered)', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB))
    axes[1].set_title('(b) Selected Spanning Interface\n(Meniscus Detection)', fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
