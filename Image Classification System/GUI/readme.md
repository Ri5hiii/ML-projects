**📌 Project 1: Garbage Image Classification**

This project focuses on automated garbage classification using deep learning and machine learning techniques to improve waste management systems. The model classifies garbage images into six categories: cardboard, paper, glass, plastic, metal, and trash, enabling efficient and accurate waste segregation.

The system implements and compares three different models:

Convolutional Neural Network (CNN)
Transfer Learning using MobileNetV2
Random Forest (baseline model)

The objective is to analyze performance across models and identify the most effective approach for real-world applications.

**🔹 Key Features**
📸 Image-based classification using Computer Vision
🧠 Deep learning + traditional ML comparison
⚡ Transfer learning with MobileNetV2 for high accuracy
📊 Performance evaluation using Accuracy, Precision, Recall, F1-score
🖥️ Interactive Streamlit-based UI for real-time predictions

**🔹 Dataset**
~2500 images across 6 classes
Source: TrashNet / Kaggle
Preprocessed to 224×224 resolution for model training

**🔹 Model Performance**
MobileNetV2 (Best Model): ~85–88% accuracy
CNN: ~80–83% accuracy
Random Forest: ~68–72% accuracy

MobileNetV2 outperforms others due to transfer learning and efficient feature extraction, making it suitable for real-time applications.

**🔹 Working Flow**
Input garbage image
Image preprocessing (resize + normalization)
Feature extraction (CNN / MobileNetV2)
Classification using dense layers
Output predicted class with confidence

(Flow is derived from methodology diagrams on pages 3–7 of your document)

**🔹 Tech Stack**
Python
TensorFlow / Keras
OpenCV
Scikit-learn
Streamlit

**🔹 Conclusion**
This project demonstrates that deep learning models significantly outperform traditional machine learning approaches for image classification tasks. Among all models, MobileNetV2 provides the best balance of accuracy, speed, and generalization, making it highly suitable for smart waste management systems and real-world deployment.
