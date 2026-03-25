import numpy as np
import joblib
from src.feature_extraction import extract_features

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_file(file):
    features = extract_features(file)
    features = np.array(features).reshape(1, -1)
    
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        return "CRY DETECTED"
    else:
        return "NOT CRY"