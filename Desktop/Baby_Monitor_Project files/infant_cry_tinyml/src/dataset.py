import os
from feature_extraction import extract_features

def load_split(path):
    X = []
    y = []

    # Cry = 1
    cry_path = os.path.join(path, "cry")
    for file in os.listdir(cry_path):
        if file.endswith(".wav"):
            X.append(extract_features(os.path.join(cry_path, file)))
            y.append(1)

    # Non-cry = 0
    noncry_path = os.path.join(path, "noncry")
    for file in os.listdir(noncry_path):
        if file.endswith(".wav"):
            X.append(extract_features(os.path.join(noncry_path, file)))
            y.append(0)

    return X, y