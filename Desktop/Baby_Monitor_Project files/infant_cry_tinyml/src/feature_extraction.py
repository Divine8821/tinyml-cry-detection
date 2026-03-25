import numpy as np
import librosa

def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)
    
    audio = audio / np.max(np.abs(audio))
    
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    
    return [rms, zcr, centroid, bandwidth]