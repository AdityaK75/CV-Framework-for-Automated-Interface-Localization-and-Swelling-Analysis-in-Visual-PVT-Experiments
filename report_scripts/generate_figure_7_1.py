import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

IMAGE_PATH = 'results/Report_Air_t10_img_5.png'
OUTPUT_PATH = 'results/representative_figure_7_1.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found"); return

    img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.imshow(img)
    ax.axis('off')

    plt.tight_layout(pad=0.2)
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
