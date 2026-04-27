import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

# --- Configuration ---
IMAGE_PATH = '/Users/adityakanagalekar/Desktop/Screenshot 2026-04-27 at 5.20.58 AM.png'
OUTPUT_PATH = 'results/diagnostic_subpixel_figure_7_4.png'

def generate_figure():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}."); return

    # --- Pipeline ---
    raw_img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobel_y)

    h, w = abs_sobel.shape
    roi_top, roi_bottom = int(h * 0.3), int(h * 0.7)
    column_sum = np.sum(abs_sobel[roi_top:roi_bottom, :], axis=1)
    peak_offset = np.argmax(column_sum)
    raw_peak_y = peak_offset + roi_top

    # Local profile ±10 rows
    window = 10
    local_rows = np.arange(raw_peak_y - window, raw_peak_y + window + 1)
    local_profile = column_sum[peak_offset - window : peak_offset + window + 1]

    # Polynomial fit
    x_local = np.arange(len(local_profile), dtype=float)
    coeffs = np.polyfit(x_local, local_profile, 2)
    a, b, c = coeffs
    x_vertex_local = -b / (2 * a)
    subpixel_row = x_vertex_local + (raw_peak_y - window)

    x_dense = np.linspace(0, len(local_profile) - 1, 300)
    y_parabola = np.polyval(coeffs, x_dense)

    raw_y = int(round(raw_peak_y))
    delta = abs(subpixel_row - raw_y)

    # --- 3-panel figure ---
    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 0.8], wspace=0.3)

    # ---- Panel (a): Image with raw & refined lines ----
    ax_img = fig.add_subplot(gs[0])
    vis = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).copy()
    x1, x2 = int(w * 0.1), int(w * 0.9)

    # Raw dashed yellow
    for xi in range(x1, x2, 12):
        cv2.line(vis, (xi, raw_y), (min(xi + 6, x2), raw_y), (255, 200, 0), 2)
    # Refined solid cyan
    sub_y_int = int(round(subpixel_row))
    cv2.line(vis, (x1, sub_y_int), (x2, sub_y_int), (0, 230, 230), 3)
    cv2.circle(vis, (w // 2, sub_y_int), 7, (0, 230, 230), -1)

    ax_img.imshow(vis)
    ax_img.set_title('(a)  Detected Interface', fontsize=12, fontweight='bold', pad=8)
    ax_img.axis('off')

    yellow_p = mpatches.Patch(color='#FFC800', label=f'Raw Y = {raw_y} px')
    cyan_p   = mpatches.Patch(color='#00E6E6', label=f'Sub-pixel Y = {subpixel_row:.2f} px')
    ax_img.legend(handles=[yellow_p, cyan_p], loc='lower center', fontsize=8, framealpha=0.9)

    # ---- Panel (b): Local gradient profile + fit ----
    ax_prof = fig.add_subplot(gs[1])
    ax_prof.bar(local_rows, local_profile, color='#5599dd', alpha=0.45,
                width=0.8, label='|∇I| column sum')
    ax_prof.plot(x_dense + (raw_peak_y - window), y_parabola,
                 color='#FF6B35', linewidth=2.5, label='Parabolic fit  $g(x)=ax^2+bx+c$')
    ax_prof.axvline(raw_peak_y, color='#FFC800', linewidth=2, linestyle='--',
                    label=f'Raw peak  $y_{{raw}}={raw_peak_y}$')
    ax_prof.axvline(subpixel_row, color='#00E6E6', linewidth=2.5, linestyle='-',
                    label=f'Vertex  $y_v={subpixel_row:.2f}$')

    ax_prof.set_xlabel('Image Row (pixels)', fontsize=11)
    ax_prof.set_ylabel('Summed |Gradient|', fontsize=11)
    ax_prof.set_title('(b)  Gradient Profile & Polynomial Fit', fontsize=12, fontweight='bold', pad=8)
    ax_prof.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
    ax_prof.grid(True, alpha=0.25, linestyle='--')
    ax_prof.set_xlim(local_rows[0] - 1, local_rows[-1] + 1)

    # Equation box
    eq = f'$g(x)={a:.1f}x^2{b:+.1f}x{c:+.1f}$\n$y_v = -b/2a = {subpixel_row:.3f}$'
    ax_prof.text(0.03, 0.97, eq, transform=ax_prof.transAxes, ha='left', va='top',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff', alpha=0.9))

    # ---- Panel (c): Summary diagnostics card ----
    ax_card = fig.add_subplot(gs[2])
    ax_card.axis('off')
    ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1)

    # Background card
    card = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax_card.transAxes,
                          facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=1.5,
                          clip_on=False, zorder=0)
    ax_card.add_patch(card)

    lines = [
        ("DIAGNOSTIC SUMMARY", 0.88, 11, 'bold', '#212529'),
        (f"Raw detected Y:", 0.72, 10, 'bold', '#495057'),
        (f"{raw_y} px", 0.63, 14, 'bold', '#FFC800'),
        (f"Sub-pixel Y:", 0.50, 10, 'bold', '#495057'),
        (f"{subpixel_row:.3f} px", 0.41, 14, 'bold', '#00BFBF'),
        (f"Δ correction:", 0.28, 10, 'bold', '#495057'),
        (f"{delta:.4f} px", 0.19, 14, 'bold', '#FF6B35'),
    ]
    for text, y, fs, fw, col in lines:
        ax_card.text(0.5, y, text, transform=ax_card.transAxes,
                     ha='center', va='center', fontsize=fs, fontweight=fw, color=col)

    ax_card.set_title('(c)  Results', fontsize=12, fontweight='bold', pad=8)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Success! Figure saved to: {OUTPUT_PATH}")
    print(f"  Raw Y       : {raw_y}")
    print(f"  Sub-pixel Y : {subpixel_row:.4f}")
    print(f"  Δ           : {delta:.4f} px")

if __name__ == "__main__":
    generate_figure()
