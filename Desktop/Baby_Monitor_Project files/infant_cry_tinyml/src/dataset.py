import os
from src.feature_extraction import extract_features

def load_dataset(cry_path, noncry_path):
    X = []
    y = []

    # Cry = 1
    for file in os.listdir(cry_path):
        if file.endswith(".wav"):
            path = os.path.join(cry_path, file)
            X.append(extract_features(path))
            y.append(1)

    # Non-cry = 0
    for file in os.listdir(noncry_path):
        if file.endswith(".wav"):
            path = os.path.join(noncry_path, file)
            X.append(extract_features(path))
            y.append(0)

    return X, y