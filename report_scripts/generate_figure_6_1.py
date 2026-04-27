import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png' 
OUTPUT_PATH = 'results/calibration_geometry_figure.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}.")
        return

    # 1. Load and rotate
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_horiz = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_horiz.shape[:2]
    
    # Define points (normalized 0-1)
    top_x = 0.85
    bottom_x = 0.15
    meniscus_x = 0.45
    mid_y = 0.5
    
    # Coordinates in pixels
    px_top = top_x * w
    px_bottom = bottom_x * w
    px_meniscus = meniscus_x * w
    px_mid_y = mid_y * h

    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Display image
    ax.imshow(img_horiz)
    
    # 1. Draw points
    ax.scatter([px_top], [px_mid_y], color='red', s=250, edgecolors='white', linewidth=2, zorder=5)
    ax.scatter([px_bottom], [px_mid_y], color='blue', s=250, edgecolors='white', linewidth=2, zorder=5)
    ax.scatter([px_meniscus], [px_mid_y], color='cyan', s=250, edgecolors='white', linewidth=2, zorder=5)
    
    # 2. Vertical projection lines — short, just below image
    proj_top = h + 20
    proj_bot = h + 80
    ax.vlines([px_top, px_bottom, px_meniscus], h, proj_bot, colors='gray', linestyles='dashed', alpha=0.6)
    
    # 3. Baseline
    ax.hlines(proj_bot, px_bottom, px_top, colors='black', linewidth=3)
    
    # 4. Height arrow (bottom to meniscus)
    arr_y = proj_bot + 40
    ax.annotate('', xy=(px_meniscus, arr_y), xytext=(px_bottom, arr_y),
                arrowprops=dict(arrowstyle='<->', color='forestgreen', lw=4))
    
    # 5. Labels — tight to image
    label_y = -30
    ax.text(px_top, label_y, 'Top Calibration\n($y_{top}$)', color='red',
            ha='center', fontweight='bold', fontsize=13)
    ax.text(px_bottom, label_y, 'Bottom Calibration\n($y_{bottom}$)', color='blue',
            ha='center', fontweight='bold', fontsize=13)
    ax.text(px_meniscus, label_y - 40, 'Detected Meniscus\n($y_{meniscus}$)', color='darkcyan',
            ha='center', fontweight='bold', fontsize=13)
    
    ax.text((px_bottom + px_meniscus) / 2, arr_y + 50, 'Measured Height ($h$)',
            color='forestgreen', ha='center', fontweight='bold', fontsize=15)
    
    ax.axis('off')
    # Tight y-limits: from just above labels to just below arrow text
    ax.set_ylim(arr_y + 100, label_y - 80)
    
    plt.tight_layout(pad=0.5)
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
