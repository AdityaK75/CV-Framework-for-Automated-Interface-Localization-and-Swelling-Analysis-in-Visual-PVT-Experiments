import cv2
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update this path if your image is located somewhere else
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png' 
OUTPUT_PATH = 'results/real_preprocessing_stages.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}. Please make sure it is in the same folder.")
        return

    # Load the actual image
    raw_img = cv2.imread(IMAGE_PATH)
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    # If the image is a full sight glass, we can optionally crop it to a specific ROI. 
    # (Uncomment and adjust the numbers below if you want to crop it instead of using the full image)
    # h, w = gray.shape
    # roi = gray[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)] 
    roi = gray  # We will use the whole image as the "ROI" for this figure

    # 2. Apply Bilateral Filter (Stronger smoothing to show noise reduction)
    filtered_roi = cv2.bilateralFilter(roi, d=15, sigmaColor=100, sigmaSpace=100)

    # 3. Apply CLAHE (Much stronger contrast stretching to show enhancement)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    enhanced_roi = clahe.apply(filtered_roi)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(roi, cmap='gray')
    axes[0].set_title('(a) Raw ROI', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    axes[1].imshow(filtered_roi, cmap='gray')
    axes[1].set_title('(b) Filtered ROI', fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    axes[2].imshow(enhanced_roi, cmap='gray')
    axes[2].set_title('(c) Enhanced ROI\n(CLAHE)', fontsize=14, fontweight='bold', pad=10)
    axes[2].axis('off')

    plt.tight_layout()
    
    # Ensure results folder exists
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()