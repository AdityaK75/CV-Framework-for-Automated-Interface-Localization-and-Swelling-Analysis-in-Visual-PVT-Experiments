import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

OUTPUT_PATH = 'results/manual_correction_figure_7_7.png'

AUTO_PATH  = 'results/Report_Air_t10_img_5.png'
CORR_PATH  = 'results/Report_Air_t10_img_5_corrected.png'

AUTO_H, AUTO_V = '48.31 mm', '7.69 ml'
CORR_H, CORR_V = '48.21 mm', '7.67 ml'

def generate_figure():
    for p in [AUTO_PATH, CORR_PATH]:
        if not os.path.exists(p):
            print(f"Error: {p} not found"); return

    img_auto = cv2.cvtColor(cv2.imread(AUTO_PATH), cv2.COLOR_BGR2RGB)
    img_corr = cv2.cvtColor(cv2.imread(CORR_PATH), cv2.COLOR_BGR2RGB)

    # Normalize heights
    target_h = max(img_auto.shape[0], img_corr.shape[0])
    def resize_to(im, th):
        h, w = im.shape[:2]
        s = th / h
        return cv2.resize(im, (int(w * s), th))

    img_auto = resize_to(img_auto, target_h)
    img_corr = resize_to(img_corr, target_h)

    fig, axes = plt.subplots(1, 2, figsize=(10, 14), gridspec_kw={'wspace': 0.15})

    # Panel (a): Automatic result
    axes[0].imshow(img_auto)
    axes[0].set_title(f'(a)  Automatic Detection\nH = {AUTO_H}   V = {AUTO_V}',
                      fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Panel (b): Corrected result
    axes[1].imshow(img_corr)
    axes[1].set_title(f'(b)  After Manual Correction\nH = {CORR_H}   V = {CORR_V}',
                      fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')

    # Delta annotation between the two panels
    delta_h = f'ΔH = {abs(48.31 - 48.21):.2f} mm'
    delta_v = f'ΔV = {abs(7.69 - 7.67):.2f} ml'
    fig.text(0.5, 0.02, f'{delta_h}     {delta_v}',
             ha='center', fontsize=14, fontweight='bold', color='#e74c3c',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3f3', edgecolor='#e74c3c', alpha=0.9))

    # Arrow between panels using an overlay axes
    arrow_ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
    arrow_ax.set_xlim(0, 1); arrow_ax.set_ylim(0, 1)
    arrow_ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
                      arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
    arrow_ax.axis('off')
    fig.text(0.5, 0.52, 'Manual\nCorrection', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#2ecc71',
             transform=fig.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Success! Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_figure()
