import os
from flask import Flask, render_template, request
from PIL import Image

from modules.ela import detect_ela
from modules.noise import detect_noise

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        try:

            # -----------------------------
            # CHECK FILE
            # -----------------------------
            if "image" not in request.files:
                return "No file part"

            file = request.files["image"]

            if file.filename == "":
                return "No file selected"

            print("Uploaded File:", file.filename)

            # -----------------------------
            # SAVE TEMP FILE
            # -----------------------------
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_input")
            file.save(temp_path)

            # -----------------------------
            # STANDARDIZE IMAGE FORMAT
            # -----------------------------
            img = Image.open(temp_path).convert("RGB")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)

            upload_path = os.path.join(UPLOAD_FOLDER, "standardized.jpg")
            img.save(upload_path, "JPEG", quality=92)

            # -----------------------------
            # RUN FORENSIC ANALYSIS
            # -----------------------------
            ela_path, heatmap_path, boxed_path, ela_score = detect_ela(upload_path)
            noise_path, hist_path, edge_path, noise_score, uniformity_score = detect_noise(upload_path)

            # -----------------------------
            # NORMALIZE SCORES (0 - 100)
            # -----------------------------
            ela_score = min(max(ela_score * 2, 0), 100)
            noise_score = min(max(noise_score * 1.5, 0), 100)
            uniformity_score = min(max(uniformity_score, 0), 100)

            # -----------------------------
            # FINAL HYBRID SCORE
            # -----------------------------
            # FINAL SCORE CALCULATION
            final_score = round(
                (ela_score * 0.6) +
                (noise_score * 0.3) +
                ((100 - uniformity_score) * 0.1),
                2
            )

            final_score = min(max(final_score, 0), 100)

            print("ELA:", ela_score)
            print("Noise:", noise_score)
            print("Uniformity:", uniformity_score)
            print("Final Score:", final_score)

            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            # CLASSIFICATION
            # CLASSIFICATION
            if final_score >= 25:
                verdict = "FAKE"
                verdict_color = "#ff4d4d"
                risk_level = "HIGH"
                confidence = final_score
            else:
                verdict = "REAL"
                verdict_color = "#00ff9c"
                risk_level = "LOW"
                confidence = round(100 - final_score, 2)

            # -----------------------------
            # RETURN RESULTS
            # -----------------------------
            return render_template(
                "index.html",
                uploaded_image=upload_path,
                ela_image=ela_path,
                heatmap_image=heatmap_path,
                boxed_image=boxed_path,
                noise_image=noise_path,
                hist_image=hist_path,
                edge_image=edge_path,
                final_score=final_score,
                verdict=verdict,
                verdict_color=verdict_color,
                risk_level=risk_level,
                confidence=confidence,
                noise_score=noise_score,
                ela_score=ela_score,
                uniformity_score=uniformity_score
            )

        except Exception as e:
            print("ERROR:", e)
            return "Error processing image."

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)