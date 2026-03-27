import numpy as np
import librosa

def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)
    
    audio = audio / np.max(np.abs(audio))
    
    # Time-domain
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Spectral
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    
    # FFT
    fft = np.abs(np.fft.fft(audio))
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    
    dominant_freq = freqs[np.argmax(fft)]
    
    # Band energy
    low = np.sum(fft[(freqs >= 0) & (freqs < 400)])
    mid = np.sum(fft[(freqs >= 400) & (freqs < 2000)])
    high = np.sum(fft[(freqs >= 2000) & (freqs < 4000)])
    mid_ratio = mid / (low + mid + high + 1e-6)

    return [rms, zcr, centroid, bandwidth, dominant_freq, low, mid, high, mid_ratio]