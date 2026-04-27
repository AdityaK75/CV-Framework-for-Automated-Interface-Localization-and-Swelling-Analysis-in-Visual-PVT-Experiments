import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_PATH = 'results/swelling_gallery_figure.png'

# 5 images sorted from least volume to most
# (path, volume_ml, height_mm)
IMAGES = [
    ('results/Report_Air_t0_img_2.png',           2.05,  '13.27 mm'),
    ('results/Report_Air_t10_img_1.png',           4.70,  '29.77 mm'),
    ('results/Report_Air_t10_img_5.png',           7.69,  '48.31 mm'),
    ('results/Report_Air_t10_img_4_corrected.png', 10.95, '68.56 mm'),
    ('results/Report_Air_t10_img_3.png',           12.53, '78.40 mm'),
]

BASE_VOL = IMAGES[0][1]  # 2.05 ml

def generate_figure():
    for path, _, _ in IMAGES:
        if not os.path.exists(path):
            print(f"Error: {path} not found"); return

    imgs = []
    for path, vol, ht in IMAGES:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sf = vol / BASE_VOL
        imgs.append((img, vol, ht, sf))

    # Normalize all images to the same height
    target_h = max(im.shape[0] for im, _, _, _ in imgs)
    resized = []
    for im, vol, ht, sf in imgs:
        h, w = im.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append((cv2.resize(im, (new_w, target_h)), vol, ht, sf))

    n = len(resized)
    fig, axes = plt.subplots(1, n, figsize=(22, 9))

    for i, (im, vol, ht, sf) in enumerate(resized):
        ax = axes[i]
        ax.imshow(im)
        ax.axis('off')

        if i == 0:
            label = f"BASE IMAGE\nV = {vol:.2f} ml  |  h = {ht}\nSF = {sf:.2f}"
        else:
            label = f"V = {vol:.2f} ml  |  h = {ht}\nSF = {sf:.2f}"
        ax.set_title(label, fontsize=11, fontweight='bold', pad=8)

    # --- Clean arrow using matplotlib annotation ---
    fig.text(0.5, 0.025, 'Increasing Swelling  →',
             ha='center', fontsize=16, fontweight='bold', color='forestgreen')
    # Draw a proper arrow using annotate on fig coordinates
    arrow = fig.add_axes([0, 0, 1, 1], frame_on=False)
    arrow.set_xlim(0, 1); arrow.set_ylim(0, 1)
    arrow.annotate('', xy=(0.92, 0.06), xytext=(0.08, 0.06),
                   arrowprops=dict(arrowstyle='->', color='forestgreen', lw=3))
    arrow.axis('off')

    plt.subplots_adjust(bottom=0.10, wspace=0.06, top=0.88)
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.15)
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
