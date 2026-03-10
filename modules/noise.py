from PIL import Image, ImageFilter, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import os


def detect_noise(image_path):
    image = Image.open(image_path).convert('RGB')

    os.makedirs("static/uploads", exist_ok=True)

    # Convert to grayscale
    gray = image.convert("L")

    # Gaussian blur
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))

    # Noise extraction
    diff = ImageChops.difference(gray, blurred)
    diff_array = np.array(diff).astype(np.float32)

    # Normalize noise map
    diff_array = (diff_array / (diff_array.max() + 1)) * 255
    diff_array = diff_array.astype(np.uint8)

    noise_image = Image.fromarray(diff_array)
    noise_path = "static/uploads/noise_result.jpg"
    noise_image.save(noise_path)

    # Histogram
    plt.figure()
    plt.hist(diff_array.flatten(), bins=256)
    plt.title("Noise Histogram")
    hist_path = "static/uploads/noise_histogram.png"
    plt.savefig(hist_path)
    plt.close()

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges).astype(np.float32)
    edge_path = "static/uploads/edge_map.jpg"
    edges.save(edge_path)

    # ===============================
    # STRONGER FORENSIC METRICS
    # ===============================

    mean_noise = np.mean(diff_array)
    std_noise = np.std(diff_array)

    # Color channel imbalance
    r, g, b = image.split()
    r_std = np.std(np.array(r))
    g_std = np.std(np.array(g))
    b_std = np.std(np.array(b))

    color_variation = abs(r_std - g_std) + abs(g_std - b_std)

    # Edge density
    edge_density = np.mean(edge_array) / 255

    # Uniformity score (smoothness indicator)
    uniformity_score = 1 / (std_noise + 1)
    uniformity_score = min(uniformity_score * 100, 100)

    # ===============================
    # FINAL IMPROVED NOISE SCORE
    # ===============================

    noise_score = (
        (mean_noise * 0.3) +
        (std_noise * 0.3) +
        (color_variation * 0.2) +
        (edge_density * 255 * 0.2)
    )

    # Normalize to 0–100
    noise_score = min((noise_score / 255) * 100, 100)
    noise_score = round(noise_score, 2)
    uniformity_score = round(uniformity_score, 2)

    return noise_path, hist_path, edge_path, noise_score, uniformity_score



