# 📌 Project 4: Cough Audio Classification using ML & Deep Learning

## 🔹 Description
This project focuses on detecting cough sounds from audio signals using machine learning and deep learning techniques. The system classifies audio inputs into two categories: **Cough** and **Non-Cough**, helping in early detection of respiratory diseases such as COVID-19, asthma, and tuberculosis.

The project combines traditional feature extraction (MFCC) with machine learning (SVM) and deep learning (CNN using spectrograms) to improve classification performance and robustness.

---

## 🔹 Models Used
- MFCC + Support Vector Machine (SVM)  
- Spectrogram + Convolutional Neural Network (CNN)  

---

## 🔹 Key Features
- 🎧 Audio-based classification system  
- 🧠 Combination of ML and Deep Learning  
- 🎼 Feature extraction using MFCC (Mel-Frequency Cepstral Coefficients)  
- 📊 Spectrogram-based deep learning classification  
- 📈 Performance evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC  
- 🖥️ Streamlit-based UI for real-time audio analysis  
- 📉 Visualization of waveform and spectrogram  

---

## 🔹 Dataset
- Cough Audio Dataset  
- ~1246 audio samples (balanced dataset)  
- Classes: Cough, Non-Cough  
- Audio format: WAV  
- Data split: 80% training / 20% testing  
- Preprocessing:
  - Noise reduction  
  - Normalization  
  - Feature extraction (MFCC & Spectrogram) :contentReference[oaicite:0]{index=0}  

---

## 🔹 Model Performance
- MFCC + SVM: ~89% accuracy  
- CNN (Spectrogram): ~84% accuracy  

👉 CNN shows better pattern recognition and higher ROC-AUC (~0.93), while SVM performs well with structured features. :contentReference[oaicite:1]{index=1}  

---

## 🔹 Working Flow

### 🔸 MFCC + SVM Pipeline
1. Capture audio input  
2. Preprocessing (pre-emphasis, framing, windowing)  
3. Extract MFCC features  
4. Train SVM classifier  
5. Predict cough / non-cough  

### 🔸 Spectrogram + CNN Pipeline
1. Capture audio input  
2. Noise reduction and normalization  
3. Convert audio to spectrogram  
4. Feature extraction using CNN layers  
5. Classification using dense layers  
6. Output prediction  

---

## 🔹 Tech Stack
- Python  
- Librosa  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Streamlit  

---

## 🔹 Conclusion
This project demonstrates that both machine learning and deep learning approaches are effective for cough detection. MFCC + SVM provides faster and efficient results for smaller datasets, while CNN with spectrograms offers better feature learning and higher classification capability. The system can be used in real-time health monitoring applications for early detection of respiratory conditions.
