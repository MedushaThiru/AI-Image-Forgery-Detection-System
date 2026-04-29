import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

ELA_QUALITY = 75
UPLOAD_FOLDER = "static/uploads"


def compute_dct_score(gray):
    h, w = gray.shape
    block_size = 8
    boundary_diffs = []
    for y in range(0, h - block_size*2, block_size):
        for x in range(0, w - block_size*2, block_size):
            b1 = gray[y:y+block_size, x:x+block_size]
            b2 = gray[y:y+block_size, x+block_size:x+block_size*2]
            b3 = gray[y+block_size:y+block_size*2, x:x+block_size]
            boundary_diffs.append(abs(float(np.std(b1)) - float(np.std(b2))))
            boundary_diffs.append(abs(float(np.std(b1)) - float(np.std(b3))))
    if not boundary_diffs:
        return 0.0
    return float(np.clip(np.mean(boundary_diffs) * 3.0, 0, 100))


def draw_boxes(image_path, ela_gray, timestamp, is_tampered):
    original = Image.open(image_path).convert("RGB")
    original_arr = np.array(original, dtype=np.float32)
    h, w = ela_gray.shape
    original_arr = original_arr[:h, :w]
    boxed = Image.fromarray(original_arr.astype(np.uint8))
    draw  = ImageDraw.Draw(boxed)
    img_h, img_w = ela_gray.shape

    if not is_tampered:
        # Green border + authentic label
        for t in range(6):
            draw.rectangle([t, t, img_w-1-t, img_h-1-t], outline=(0, 220, 120))
        msg = "  NO TAMPERING DETECTED  "
        lw = len(msg)*8+10
        lx = (img_w-lw)//2
        draw.rectangle([lx, 8, lx+lw, 28], fill=(0,160,80))
        draw.text((lx+5, 11), msg, fill=(255,255,255))
    else:
        # ── FIND BEST BLOCK PER QUADRANT ──
        # Divide image into 4 quadrants — pick most suspicious block
        # from each. This guarantees boxes spread across the image.
        block_size = max(min(img_w, img_h) // 10, 50)
        step = block_size // 2

        # Padding: avoid bottom 40px (warning label area) and edges
        pad = 10
        safe_h = img_h - 45
        safe_w = img_w - pad

        quadrants = [
            (pad,         pad,         safe_w//2,  safe_h//2),  # top-left
            (safe_w//2,   pad,         safe_w,     safe_h//2),  # top-right
            (pad,         safe_h//2,   safe_w//2,  safe_h),     # bottom-left
            (safe_w//2,   safe_h//2,   safe_w,     safe_h),     # bottom-right
        ]

        quad_best = []
        for (qx1, qy1, qx2, qy2) in quadrants:
            best_score = -1
            best_bx, best_by = qx1, qy1
            for by in range(qy1, qy2-block_size, step):
                for bx in range(qx1, qx2-block_size, step):
                    block = ela_gray[by:by+block_size, bx:bx+block_size]
                    bm = float(np.mean(block))
                    bs = float(np.std(block))
                    score = bm * (1 + bs * 0.5)
                    if score > best_score:
                        best_score = score
                        best_bx, best_by = bx, by
            quad_best.append((best_score, best_bx, best_by))

        # Sort quadrants by suspicion — top 3 most suspicious quadrants
        quad_best.sort(reverse=True)
        top3 = quad_best[:3]

        g_mean = float(np.mean(ela_gray))
        g_std  = float(np.std(ela_gray))

        for rank, (score, bx, by) in enumerate(top3):
            bx2 = min(bx + block_size, img_w-1)
            by2 = min(by + block_size, safe_h)

            if score > g_mean + 0.8 * g_std:
                color = (255, 40, 40)
                label = f"HIGH RISK {rank+1}"
            else:
                color = (255, 150, 0)
                label = f"MOD RISK {rank+1}"

            for t in range(5):
                draw.rectangle([bx-t, by-t, bx2+t, by2+t], outline=color)

            tx = max(bx+3, 2)
            ty = max(by+3, 2)
            bg = (int(color[0]*0.6), int(color[1]*0.6), 0)
            draw.rectangle([tx-1, ty-1, tx+len(label)*7+3, ty+14], fill=bg)
            draw.text((tx+1, ty+1), label, fill=(255,255,255))

        # Warning at bottom
        warn = "  TAMPERING DETECTED  "
        ww = len(warn)*8+10
        wx = (img_w-ww)//2
        draw.rectangle([wx, img_h-30, wx+ww, img_h-8], fill=(180,0,0))
        draw.text((wx+5, img_h-26), warn, fill=(255,255,255))

    boxed_path = os.path.join(UPLOAD_FOLDER, f"boxed_{timestamp}.jpg")
    boxed.save(boxed_path)
    return boxed_path


def detect_ela(image_path):
    timestamp = os.path.splitext(os.path.basename(image_path))[0]

    original = Image.open(image_path).convert("RGB")
    resaved_path = os.path.join(UPLOAD_FOLDER, f"resaved_{timestamp}.jpg")
    original.save(resaved_path, "JPEG", quality=ELA_QUALITY)
    resaved = Image.open(resaved_path).convert("RGB")

    original_arr = np.array(original, dtype=np.float32)
    resaved_arr  = np.array(resaved,  dtype=np.float32)

    h = min(original_arr.shape[0], resaved_arr.shape[0])
    w = min(original_arr.shape[1], resaved_arr.shape[1])
    original_arr = original_arr[:h, :w]
    resaved_arr  = resaved_arr[:h,  :w]

    ela_arr = np.abs(original_arr - resaved_arr)

    ela_amplified = np.clip(ela_arr * 5, 0, 255).astype(np.uint8)
    ela_img  = Image.fromarray(ela_amplified)
    ela_path = os.path.join(UPLOAD_FOLDER, f"ela_{timestamp}.jpg")
    ela_img.save(ela_path)

    ela_gray      = np.mean(ela_arr, axis=2)
    ela_score_raw = float(np.mean(ela_gray))
    ela_score     = float(np.clip(ela_score_raw * 5.0, 0, 100))
    anomaly_ratio = float(np.mean(ela_gray > float(np.mean(ela_gray)) * 2.0))
    dct_score     = compute_dct_score(np.mean(original_arr, axis=2))

    print(f"  [ela.py] raw={ela_score_raw:.3f} ela_score={ela_score:.2f} "
          f"dct_score={dct_score:.2f} anomaly_ratio={anomaly_ratio:.4f}")

    p99      = float(np.percentile(ela_gray, 99))
    ela_norm = np.clip(ela_gray / (p99 + 1e-6), 0, 1)
    heatmap_color = (cm.jet(ela_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img   = Image.fromarray(heatmap_color)
    heatmap_path  = os.path.join(UPLOAD_FOLDER, f"heatmap_{timestamp}.jpg")
    heatmap_img.save(heatmap_path)

    try:
        os.remove(resaved_path)
    except Exception:
        pass

    return ela_path, heatmap_path, ela_score, anomaly_ratio, dct_score, ela_gray, timestamp