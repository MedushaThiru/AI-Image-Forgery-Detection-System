import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UPLOAD_FOLDER = "static/uploads"


def _box_filter(arr, size=3):
    """
    Pure numpy 2D box (mean) filter using sliding_window_view.
    No scipy needed.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return windows.mean(axis=(-1, -2))


def _sobel_edges(gray):
    """Pure numpy Sobel edge detection."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    sx = np.zeros_like(gray)
    sy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            sx += padded[i:i+h, j:j+w] * kx[i, j]
            sy += padded[i:i+h, j:j+w] * ky[i, j]
    return sx, sy


def detect_noise(image_path):
    """
    Analyses noise patterns in an image.
    Returns: noise_path, hist_path, edge_path, noise_score, uniformity_score
    """
    timestamp = os.path.splitext(os.path.basename(image_path))[0]

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float32)

    # -------------------------------------------------------
    # STEP 1: Extract noise residual
    # -------------------------------------------------------
    gray = np.mean(img_arr, axis=2)
    smooth = _box_filter(gray, size=3)
    noise_residual = gray - smooth

    # -------------------------------------------------------
    # STEP 2: Noise Score
    # -------------------------------------------------------
    noise_std = float(np.std(noise_residual))
    noise_score = float(np.clip(noise_std * 5.0, 0, 100))

    # -------------------------------------------------------
    # STEP 3: Uniformity Score — quadrant comparison method
    #
    # Split image into 4 quadrants and compare noise std.
    # Real photos: all quadrants have similar noise → low score
    # Edited images: one quadrant differs sharply → high score
    #
    # We use max_quadrant / min_quadrant ratio as the signal.
    # Real photo ratio ≈ 1.0–1.5  → score 0–20
    # Edited image ratio ≈ 2.0+   → score 40–100
    # -------------------------------------------------------
    h, w = noise_residual.shape
    mid_h, mid_w = h // 2, w // 2

    q1 = float(np.std(noise_residual[:mid_h, :mid_w]))
    q2 = float(np.std(noise_residual[:mid_h, mid_w:]))
    q3 = float(np.std(noise_residual[mid_h:, :mid_w]))
    q4 = float(np.std(noise_residual[mid_h:, mid_w:]))

    quadrant_stds = [q1, q2, q3, q4]
    min_q = max(min(quadrant_stds), 0.001)  # avoid divide by zero
    max_q = max(quadrant_stds)

    # Ratio of most noisy quadrant to least noisy quadrant
    ratio = max_q / min_q

    # ratio=1.0 → score=0 (perfectly uniform = authentic)
    # ratio=2.0 → score=33
    # ratio=4.0 → score=75
    # ratio=6.0+ → score=100
    uniformity_score = float(np.clip((ratio - 1.0) / 5.0 * 100, 0, 100))

    print(f"  [noise.py] noise_std={noise_std:.3f} noise_score={noise_score:.2f} "
          f"quadrants=[{q1:.2f},{q2:.2f},{q3:.2f},{q4:.2f}] "
          f"ratio={ratio:.2f} uniformity={uniformity_score:.2f}")

    # -------------------------------------------------------
    # STEP 4: Save noise residual visualisation
    # -------------------------------------------------------
    noise_vis = noise_residual - noise_residual.min()
    noise_max = float(noise_vis.max())
    if noise_max > 0:
        noise_vis = (noise_vis / noise_max * 255).astype(np.uint8)
    else:
        noise_vis = noise_vis.astype(np.uint8)

    noise_img = Image.fromarray(noise_vis)
    noise_path = os.path.join(UPLOAD_FOLDER, f"noise_{timestamp}.jpg")
    noise_img.save(noise_path)

    # -------------------------------------------------------
    # STEP 5: Histogram
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(noise_residual.ravel(), bins=100, color="#00aaff", alpha=0.75,
            label="Noise Distribution")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Noise Residual Histogram", fontsize=12)
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(UPLOAD_FOLDER, f"hist_{timestamp}.jpg")
    plt.savefig(hist_path, dpi=80)
    plt.close()

    # -------------------------------------------------------
    # STEP 6: Edge detection
    # -------------------------------------------------------
    sx, sy = _sobel_edges(gray)
    edge_magnitude = np.hypot(sx, sy)
    edge_max = float(edge_magnitude.max())
    edge_vis = (edge_magnitude / (edge_max + 1e-6) * 255).astype(np.uint8)
    edge_img = Image.fromarray(edge_vis)
    edge_path = os.path.join(UPLOAD_FOLDER, f"edge_{timestamp}.jpg")
    edge_img.save(edge_path)

    return noise_path, hist_path, edge_path, noise_score, uniformity_score