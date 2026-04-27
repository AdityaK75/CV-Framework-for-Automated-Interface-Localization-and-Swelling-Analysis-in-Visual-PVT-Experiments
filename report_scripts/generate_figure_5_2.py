import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png' 
OUTPUT_PATH = 'results/binary_morphology_stages.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}.")
        return

    # Load and preprocess
    raw_img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction before thresholding
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 1. Otsu's Thresholding
    _, thresholded = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphological Closing (Horizontally oriented kernel)
    # 25px wide, 3px high to bridge horizontal gaps without vertical distortion
    kernel = np.ones((3, 25), np.uint8)
    morphology = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(thresholded, cmap='gray')
    axes[0].set_title('(a) Thresholded Image\n(Otsu\'s Method)', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    axes[1].imshow(morphology, cmap='gray')
    axes[1].set_title('(b) After Morphology\n(Horizontal Closing)', fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
