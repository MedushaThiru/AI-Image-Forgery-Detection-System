import cv2
import numpy as np
from PIL import Image, ImageChops
import os


def detect_ela(image_path):

    os.makedirs("static/uploads", exist_ok=True)

    # -------------------------------
    # 1. LOAD IMAGE
    # -------------------------------
    original = Image.open(image_path).convert("RGB")

    # -------------------------------
    # 2. CREATE ELA IMAGE
    # -------------------------------
    temp_path = "static/uploads/temp_resaved.jpg"

    original.save(temp_path, "JPEG", quality=90)

    resaved = Image.open(temp_path)

    ela_image = ImageChops.difference(original, resaved)

    ela_array = np.array(ela_image)

    max_val = ela_array.max() if ela_array.max() > 0 else 1

    ela_scaled = (ela_array / max_val * 255).astype(np.uint8)

    ela_path = "static/uploads/ela_result.jpg"
    Image.fromarray(ela_scaled).save(ela_path)

    # -------------------------------
    # 3. CONVERT FOR PROCESSING
    # -------------------------------
    orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    gray_ela = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)

    # -------------------------------
    # 4. THRESHOLD SUSPICIOUS AREAS
    # -------------------------------
    threshold_value = 30

    _, mask = cv2.threshold(gray_ela, threshold_value, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -------------------------------
    # 5. CREATE RED OVERLAY
    # -------------------------------
    red_layer = np.zeros_like(orig_cv)
    red_layer[:] = (0, 0, 255)

    overlay_cv = cv2.addWeighted(
        orig_cv,
        1.0,
        cv2.bitwise_and(red_layer, red_layer, mask=mask),
        0.5,
        0
    )

    overlay_path = "static/uploads/red_overlay.jpg"
    cv2.imwrite(overlay_path, overlay_cv)

    # -------------------------------
    # 6. DETECT TAMPERED REGIONS
    # -------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed_image = overlay_cv.copy()

    suspicious_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 300:

            suspicious_area += area

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                boxed_image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

    boxed_path = "static/uploads/boxed_regions.jpg"
    cv2.imwrite(boxed_path, boxed_image)

    # -------------------------------
    # 7. CALCULATE ELA SCORE
    # -------------------------------

    image_area = gray_ela.shape[0] * gray_ela.shape[1]

    anomaly_ratio = suspicious_area / image_area

    ela_score = round(anomaly_ratio * 100 * 3, 2)

    ela_score = min(ela_score, 100)

    # -------------------------------
    # RETURN RESULTS
    # -------------------------------
    return ela_path, overlay_path, boxed_path, ela_score
