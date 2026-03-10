# AI-Image-Forgery-Detection-System
AI-based Image Forgery Detection System using Python, OpenCV and Flask with Error Level Analysis (ELA)

This project presents a lightweight Artificial Intelligence based system designed to detect manipulated digital images. The system combines classical digital forensic techniques such as Error Level Analysis (ELA) and Noise Pattern Analysis to determine whether an uploaded image is authentic or tampered.

The system is implemented as a web-based application using Python and the Flask framework. Users can upload an image through the interface, and the system automatically performs forensic analysis and classifies the image as REAL or FAKE while providing visual evidence of potential tampering.

---

## Project Objective

The main objective of this project is to design and develop a web-based AI Image Forgery Detection System that can identify manipulated digital images using hybrid forensic analysis techniques.

The system integrates multiple image forensic indicators to improve detection reliability and provide interpretable visual outputs.

---

## Key Features

• Upload images through a simple web interface  
• Automatic image preprocessing and normalization  
• Error Level Analysis (ELA) for compression inconsistency detection  
• Noise Pattern Analysis to detect abnormal pixel variations  
• Hybrid scoring mechanism combining multiple forensic indicators  
• REAL or FAKE classification output  
• Confidence score and risk level indication  
• Visual forensic outputs including ELA maps and heatmaps  
• Automatic storage of analysis results with timestamps  

---

## Technologies Used

Programming Language  
- Python 3

Framework  
- Flask

Libraries  
- OpenCV  
- NumPy  
- Pillow (PIL)  
- Matplotlib  

Frontend  
- HTML5  
- CSS3  

Development Environment  
- PyCharm  
- Windows 10  

---

## System Workflow

1. User uploads an image through the web interface.
2. The system performs image preprocessing such as resizing and format normalization.
3. Error Level Analysis (ELA) is applied to detect compression inconsistencies.
4. Noise Pattern Analysis evaluates irregular pixel noise patterns.
5. A hybrid scoring mechanism combines ELA and noise metrics.
6. The final score is evaluated using a threshold-based classification system.
7. The system displays the result as REAL or FAKE along with visual forensic evidence.

---

## Dataset Used for Testing

The system was validated using the following datasets:

- CASIA Image Tampering Dataset
- Manually edited Photoshop images
- Real-world camera images

---

## Project Structure
AI-Image-Forgery-Detection-System
│
├── app.py
├── modules
│ ├── ela.py
│ └── noise.py
├── templates
├── static
├── uploads
└── results



---

## How to Run the Project

1. Clone the repository
  git clonehttps://github.com/MedushaThiru/AI-Image-Forgery-Detection-System
   

2. Install required libraries
pip install -r requirements.txt


3. Run the Flask application
python app.py


4. Open the browser and visit


5. Upload an image to analyze whether it is REAL or FAKE.

---

## Future Improvements

• Integration of deep learning models for improved accuracy  
• Support for additional image formats (PNG, TIFF, RAW)  
• Extension to video forgery detection  
• Cloud deployment for large-scale analysis  
• Automated classification of forgery types  

---

## Author

Medusha Thirunavukkarasu  
BSc (Hons) Computing – Final Year Student  
University of West London

