import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

# --- Configuration ---
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png'
OUTPUT_PATH = 'results/subpixel_refinement.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}.")
        return

    # --- Pipeline to get gradient profile at the meniscus ---
    raw_img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Sobel vertical gradient
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobel_y)

    # Search in the center 50% of the image to find the meniscus row
    h, w = abs_sobel.shape
    roi_top, roi_bottom = int(h * 0.3), int(h * 0.7)
    column_sum = np.sum(abs_sobel[roi_top:roi_bottom, :], axis=1)
    peak_offset = np.argmax(column_sum)
    raw_peak_y = peak_offset + roi_top

    # Extract local gradient profile (±10 rows around the raw peak)
    window = 10
    local_rows = np.arange(raw_peak_y - window, raw_peak_y + window + 1)
    local_profile = column_sum[peak_offset - window : peak_offset + window + 1]

    # --- Sub-pixel polynomial fit ---
    # Fit a 2nd degree polynomial to the local profile
    # x is local index 0..2*window, y is gradient magnitude
    x_local = np.arange(len(local_profile), dtype=float)
    coeffs = np.polyfit(x_local, local_profile, 2)  # a, b, c
    a, b, c = coeffs

    # Vertex = -b / 2a (sub-pixel refined position in local coords)
    x_vertex_local = -b / (2 * a)
    subpixel_row = x_vertex_local + (raw_peak_y - window)

    # Dense x for smooth parabola plot
    x_dense = np.linspace(0, len(local_profile) - 1, 300)
    y_parabola = np.polyval(coeffs, x_dense)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.6]})

    # Panel (a): Annotated image showing the raw and refined positions
    vis = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).copy()
    raw_y = int(round(raw_peak_y))
    sub_y = int(round(subpixel_row))
    x1, x2 = int(w * 0.1), int(w * 0.9)

    # Draw raw pixel line (yellow dashed look: draw dotted via segments)
    for xi in range(x1, x2, 12):
        cv2.line(vis, (xi, raw_y), (min(xi + 6, x2), raw_y), (255, 200, 0), 2)
    # Draw refined line (cyan solid)
    cv2.line(vis, (x1, sub_y), (x2, sub_y), (0, 230, 230), 3)
    cv2.circle(vis, (w // 2, sub_y), 7, (0, 230, 230), -1)

    axes[0].imshow(vis)
    axes[0].set_title('(a) Raw vs. Refined Interface\non Image', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Legend
    yellow_patch = mpatches.Patch(color='#FFC800', label=f'Raw pixel row = {raw_y}')
    cyan_patch = mpatches.Patch(color='#00E6E6', label=f'Refined row = {subpixel_row:.2f}')
    axes[0].legend(handles=[yellow_patch, cyan_patch], loc='lower center',
                   fontsize=9, framealpha=0.85)

    # Panel (b): Local gradient profile + parabolic fit
    axes[1].bar(local_rows, local_profile, color='#5599dd', alpha=0.5,
                width=0.8, label='Column gradient sum')
    axes[1].plot(x_dense + (raw_peak_y - window), y_parabola,
                 color='#FF6B35', linewidth=2.5, label='Parabolic fit $g(x)=ax^2+bx+c$')
    axes[1].axvline(raw_peak_y, color='#FFC800', linewidth=2, linestyle='--',
                    label=f'Raw peak  $x_{{raw}}={raw_peak_y}$')
    axes[1].axvline(subpixel_row, color='#00E6E6', linewidth=2.5, linestyle='-',
                    label=f'Vertex  $x_v={subpixel_row:.2f}$')

    axes[1].set_xlabel('Image Row (pixels)', fontsize=12)
    axes[1].set_ylabel('Summed |Gradient|', fontsize=12)
    axes[1].set_title('(b) Sub-pixel Refinement\nLocal Gradient Profile + Polynomial Fit',
                      fontsize=13, fontweight='bold', pad=10)
    axes[1].legend(fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(local_rows[0] - 1, local_rows[-1] + 1)

    # Equation annotation
    eq_text = f'$g(x)={a:.1f}x^2{b:+.1f}x{c:+.1f}$\n$x_v = -b/2a = {subpixel_row:.3f}$'
    axes[1].text(0.97, 0.97, eq_text, transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff', alpha=0.9))

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Figure saved to: {OUTPUT_PATH}")
    print(f"  Raw pixel row : {raw_y}")
    print(f"  Sub-pixel row : {subpixel_row:.4f}")
    print(f"  Δ correction  : {abs(subpixel_row - raw_y):.4f} px")

if __name__ == "__main__":
    generate_figure()
