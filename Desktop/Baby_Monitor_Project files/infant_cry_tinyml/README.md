Infant Cry Detection System (tinyML)

📌 Overview

This project implements a lightweight machine learning system to detect infant crying sounds and alert hearing-impaired parents. The system processes audio signals, extracts meaningful features, and classifies them as cry or non-cry.

The model is designed with tinyML principles, making it suitable for deployment on embedded systems such as Arduino.

🎯 Objectives

Detect infant cry sounds in real-time
Minimize missed cries (false negatives)
Maintain low false alarm rate
Ensure compatibility with low-power hardware

🧠 Methodology

1. Audio Preprocessing

Sampling rate: 16 kHz
Mono audio conversion
Normalization
Bandpass filtering (focused on cry-relevant frequencies)

2. Feature Extraction

Extracted features include:

RMS Energy
Zero Crossing Rate (ZCR)
Spectral Centroid
Spectral Bandwidth
Dominant Frequency (FFT)
Band Energy Distribution (Low, Mid, High)
Mid-band Energy Ratio

3. Model

Algorithm: Logistic Regression
Feature Scaling: StandardScaler
Class balancing applied
Threshold tuning for optimal sensitivity

📊 Dataset

The dataset was organized into three subsets:

Train: 100 samples
Validation: 30 samples
Test: 30 samples
Balanced classes (cry vs non-cry)
Real-world audio including noise and variations

Data augmentation applied to training set:
Noise addition
Time shifting
Volume scaling

📈 Results

Validation Performance

Accuracy: 93%
Cry Recall: 100%

Test Performance (Final)

Accuracy: 91%
Precision (Cry): 92%
Recall (Cry): 88%
F1-score: 90%

⚖️ Key Insights

Frequency-domain features significantly improved performance
Threshold tuning (0.45) reduced missed cry events
Balanced dataset and augmentation improved generalization
Trade-off achieved between sensitivity and false alarms

🛠️ Project Structure

infant-cry-tinyml/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── feature_extraction.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
│
├── main.py
├── requirements.txt
└── README.md

▶️ How to Run

1. Install dependencies

pip install -r requirements.txt

2. Train the model

python src/train.py

3. Run prediction

python main.py

🚀 Future Work

Deploy model on Arduino (tinyML)
Real-time microphone input processing
Mobile or IoT alert system
Deep learning (CNN) comparison

👨‍💻 Author

Biomedical Engineering Student
Focus: Signal Processing, TinyML, Embedded Systems

⭐ Acknowledgements

Librosa (audio processing)
Scikit-learn (machine learning)
FFmpeg (audio conversion)