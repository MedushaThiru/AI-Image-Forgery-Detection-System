import time
import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageFile

from modules.ela import detect_ela, draw_boxes
from modules.noise import detect_noise

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SEP = "=" * 60

# Load ML Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "forensic_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    ML_CLF    = model_data["clf"]
    ML_SCALER = model_data["scaler"]
    print("Forensic ML model loaded successfully.")
except Exception as e:
    print("Could not load ML model:", e)
    ML_CLF    = None
    ML_SCALER = None


def classify_image(ela_score, noise_score, uniformity_score, dct_score):
    """
    Pure ML classification using Random Forest.
    Trained on 43 real tested images across 7 categories.
    Threshold = 0.60 (calibrated against all known test images).
    No manual rules — ML decides everything.
    """
    if ML_CLF is None:
        is_tampered = (ela_score > 15.0)
        return 50.0, is_tampered

    # Feature vector
    features        = np.array([[ela_score, noise_score, uniformity_score, dct_score]])
    features_scaled = ML_SCALER.transform(features)

    # Prediction
    prob_fake   = float(ML_CLF.predict_proba(features_scaled)[0][1])
    is_tampered = bool(prob_fake >= 0.60)
    verdict_str = "FAKE" if is_tampered else "REAL"

    print("  [ML Decision] ELA:" + str(round(ela_score,1)) +
          " | Noise:" + str(round(noise_score,1)) +
          " | Unif:" + str(round(uniformity_score,1)) +
          " | DCT:" + str(round(dct_score,1)))
    print("  [ML Result] Probability: " + str(round(prob_fake,3)) + " -> " + verdict_str)

    return (prob_fake * 100), is_tampered


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        try:
            if "image" not in request.files:
                return "No file part"
            file = request.files["image"]
            if file.filename == "":
                return "No file selected"

            print(SEP)
            print("Processing Image: " + file.filename)

            # Save incoming file
            timestamp = str(int(time.time() * 1000))
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_" + timestamp + ".jpg")
            file.save(temp_path)

            # Preprocessing and Standardisation
            img = Image.open(temp_path).convert("RGB")
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            w, h = img.size
            w = max((w // 16) * 16, 16)
            h = max((h // 16) * 16, 16)
            img = img.resize((w, h), Image.Resampling.LANCZOS)

            upload_path = os.path.join(UPLOAD_FOLDER, "img_" + timestamp + ".jpg")
            img.save(upload_path, "JPEG", quality=95)
            os.remove(temp_path)

            # Feature Extraction
            ela_path, heatmap_path, ela_score, ratio, dct_score, ela_gray, ts = detect_ela(upload_path)
            noise_path, hist_path, edge_path, noise_score, uniformity_score    = detect_noise(upload_path)

            # Classification
            final_score, is_tampered = classify_image(
                ela_score, noise_score, uniformity_score, dct_score
            )

            # Visualisation
            boxed_path = draw_boxes(upload_path, ela_gray, ts, is_tampered)

            # Verdict
            if is_tampered:
                verdict      = "FAKE"
                v_color      = "#ff4d4d"
                risk         = "HIGH"
                display_conf = round(final_score, 2)
            else:
                verdict      = "REAL"
                v_color      = "#00cc7a"
                risk         = "LOW"
                display_conf = round(100 - final_score, 2)

            print("Final Verdict: " + verdict + " (" + str(display_conf) + "%)")
            print(SEP)

            return render_template(
                "index.html",
                uploaded_image   = upload_path,
                ela_image        = ela_path,
                heatmap_image    = heatmap_path,
                boxed_image      = boxed_path,
                noise_image      = noise_path,
                hist_image       = hist_path,
                edge_image       = edge_path,
                final_score      = final_score,
                verdict          = verdict,
                verdict_color    = v_color,
                risk_level       = risk,
                confidence       = display_conf,
                noise_score      = round(noise_score, 2),
                ela_score        = round(ela_score, 2),
                uniformity_score = round(uniformity_score, 2),
                anomaly_ratio    = round(ratio * 100, 2),
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return "System Error: " + str(e)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)