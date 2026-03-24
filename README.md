# ML-projects
Machine Learning Projects covering Image, Video, Audio, Text, and Numerical data using ML &amp; Deep Learning models with GUI-based implementation.
# Machine Learning Projects Collection

## 📌 Overview
This repository contains multiple Machine Learning and Deep Learning projects based on different data types including Image, Video, Audio, Text, and Numerical data.

Each project demonstrates complete pipeline implementation including data preprocessing, feature extraction, model training, evaluation, and GUI-based testing.

---

## 🚀 Projects Included

### 1. 🖼️ Image Classification (Garbage Classification)
- Classifies waste into categories like cardboard, glass, metal, paper, plastic, and trash
- Models Used:
  - CNN
  - MobileNetV2 (Transfer Learning)
- Best Accuracy: ~82%

---

### 2. 🎥 Video Classification (Fall Detection)
- Detects fall events from video data for healthcare/surveillance
- Process:
  - Video → Frame Extraction → CNN Classification
- Model Used:
  - Convolutional Neural Network (CNN)

---

### 3. 🔊 Audio Classification (Cough Detection)
- Classifies cough vs non-cough audio signals
- Models Used:
  - SVM (MFCC features)
  - CNN (Spectrogram images)
- Accuracy:
  - SVM: ~77%
  - CNN: ~74%

---

### 4. 📝 Text Classification (Language Detection)
- Classifies text into multiple languages
- Models Used:
  - Logistic Regression
  - Linear SVM
- Best Accuracy: 100% (SVM)

---

### 5. 📊 Numerical Data (Heart Disease Prediction)
- Predicts heart disease based on medical features
- Models Used:
  - Logistic Regression
  - Random Forest
- Best Accuracy: ~94% (Random Forest)

---

## 🧠 Common Workflow
- Data Collection
- Data Preprocessing
- Feature Extraction
- Train-Test Split
- Model Training
- Evaluation (Accuracy, Precision, Recall, F1-score)
- Prediction

---

## 💻 Technologies Used
- Python
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- Librosa
- Streamlit (GUI)

---

## 📂 Features
- Multiple ML & DL models implementation
- Model comparison and evaluation
- GUI-based testing system
- Real-world problem solving

---

## ▶️ How to Run
1. Clone the repository
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
