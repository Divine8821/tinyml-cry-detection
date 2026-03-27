import numpy as np
import joblib
from feature_extraction import extract_features

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
threshold = joblib.load("model/threshold.pkl")

def predict_file(file):
    features = extract_features(file)
    features = np.array(features).reshape(1, -1)
    
    features = scaler.transform(features)
    
    proba = model.predict_proba(features)[0][1]
    
    if proba > threshold:
        return "CRY DETECTED"
    else:
        return "NOT CRY"