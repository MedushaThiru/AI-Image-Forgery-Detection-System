# AI Image Forgery Detection System

## 📌 Overview
This project detects whether an image is **REAL or FAKE** using digital image forensics techniques combined with a Machine Learning model.  
The system analyzes compression artifacts, noise inconsistencies, and structural anomalies to identify possible tampering.

---

## ⚙️ Technologies Used
- Python
- Flask (Backend Web Framework)
- HTML, CSS, JavaScript (Frontend)
- Scikit-learn (Machine Learning)
- Pillow (Image Processing)
- NumPy (Numerical computations)

---

## 🔍 Features
- Upload image through web interface
- Error Level Analysis (ELA)
- Noise Analysis
- Heatmap visualization of tampered regions
- Edge detection
- Highlighting suspicious regions (bounding boxes)
- ML-based classification (REAL / FAKE)
- Confidence score output
- Risk level indication (LOW / HIGH)

---

## 🧠 Machine Learning Model
- Algorithm: Random Forest Classifier
- Preprocessing: StandardScaler
- Features used:
  - ELA Score
  - Noise Score
  - Uniformity Score
  - DCT Score
- Probability-based classification with threshold decision

---

## 🔬 System Workflow
1. User uploads an image
2. Image is preprocessed (resize, normalization)
3. ELA (Error Level Analysis) is performed
4. Noise and uniformity analysis are calculated
5. Feature vector is generated
6. Machine Learning model predicts tampering probability
7. System outputs:
   - REAL / FAKE verdict
   - Confidence score
   - Visual forensic maps

---

## 🖼️ System Output Includes
- Original Image
- ELA Map
- Noise Map
- Heatmap of tampered regions
- Edge Detection Output
- Highlighted suspicious regions

---

## 📊 Performance
The model is trained using multiple real and tampered images across different categories.  
It achieves reliable classification performance based on extracted forensic features.

---

## ▶️ How to Run the Project

### Step 1: Install dependencies

### Step 2: Run the application

### Step 3: Open in browser 
http://127.0.0.1:5000/

---

## 📁 Project Structure
ai_image_forgery_system/
│
├── app.py # Main Flask application
├── train_model.py # ML model training script
├── forensic_model.pkl # Trained model
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── modules/
│ ├── ela.py # ELA detection module
│ └── noise.py # Noise analysis module
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/
│ └── uploads/ # Uploaded & processed images


---

## 🖥️ System Type
This is a **Web-based Image Forensic Detection System** built using Flask.

---

## 🔐 Limitations
- Accuracy may vary depending on image compression level
- Highly optimized or AI-generated images may be harder to detect
- Model performance depends on training dataset diversity

---

## 🚀 Future Improvements
- Deep learning integration (CNN-based detection)
- Larger and more diverse training dataset
- Real-time video forgery detection
- Cloud deployment (AWS / Azure)
- API integration for external applications

---

## 👨‍💻 Author
**Medusha Thirunavukkarasu**  


