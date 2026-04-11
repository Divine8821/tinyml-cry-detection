# src/detectors.py
import numpy as np
import scipy.signal as sig  
import pywt

class ECGDetectors:
    @staticmethod
    def pan_tompkins(ecg_data, fs):
        """
        Manual implementation of the Pan-Tompkins QRS detection algorithm.
        """
        # 1. BANDPASS FILTER (5-15 Hz)
        nyq = 0.5 * fs
        low = 5 / nyq
        high = 15 / nyq
        b, a = sig.butter(2, [low, high], btype='band')
        filtered_ecg = sig.filtfilt(b, a, ecg_data)

        # 2. DERIVATIVE 
        diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8
        differentiated_ecg = np.convolve(filtered_ecg, diff_kernel, mode='same')

        # 3. SQUARING
        squared_ecg = differentiated_ecg ** 2

        # 4. MOVING WINDOW INTEGRATION
        window_size = int(0.150 * fs)
        integrated_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')

        # 5. ADAPTIVE THRESHOLDING
        peaks, _ = sig.find_peaks(integrated_ecg, distance=int(0.2 * fs)) 
        
        if len(peaks) > 0:
            threshold = np.mean(integrated_ecg[peaks]) * 0.6
            qrs_peaks = peaks[integrated_ecg[peaks] > threshold]
        else:
            qrs_peaks = np.array([])

        return qrs_peaks

    @staticmethod
    def wavelet_transform(ecg_data, fs):
        """
        QRS detection using Stationary/Discrete Wavelet Transform.
        """
        # 1. Decomposition (4 levels, db4 wavelet)
        coeffs = pywt.wavedec(ecg_data, 'db4', level=4)
        
        # 2. Sub-band Reconstruction (Isolating QRS frequencies)
        filtered_coeffs = [np.zeros_like(c) for c in coeffs]
        filtered_coeffs[1] = coeffs[1] 
        filtered_coeffs[2] = coeffs[2]
        
        reconstructed_sig = pywt.waverec(filtered_coeffs, 'db4')
        
        # 3. Squaring and Integration
        # Handle potential length mismatch from reconstruction
        reconstructed_sig = reconstructed_sig[:len(ecg_data)]
        squared = reconstructed_sig**2
        window = int(0.150 * fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # 4. Adaptive Peak Detection
        peaks, _ = sig.find_peaks(integrated, distance=int(0.25 * fs))
        if len(peaks) > 0:
            threshold = np.mean(integrated[peaks]) * 0.50
            qrs_peaks = peaks[integrated[peaks] > threshold]
        else:
            qrs_peaks = np.array([])

        return qrs_peaks

    @staticmethod
    def hilbert_transform(ecg_data, fs):
        """
        QRS detection using the Hilbert Transform analytic signal.
        """
        # 1. Pre-filtering
        nyq = 0.5 * fs
        b, a = sig.butter(2, [5/nyq, 15/nyq], btype='band')
        filtered = sig.filtfilt(b, a, ecg_data)

        # 2. Differentiation
        diff = np.gradient(filtered)

        # 3. Hilbert Transform to get the Analytic Signal
        analytic_signal = sig.hilbert(diff)
        amplitude_envelope = np.abs(analytic_signal)

        # 4. Smoothing (Moving Average)
        window_len = int(0.150 * fs)
        integrated = np.convolve(amplitude_envelope, np.ones(window_len)/window_len, mode='same')

        # 5. Adaptive Peak Detection
        peaks, _ = sig.find_peaks(integrated, distance=int(0.25 * fs))
        if len(peaks) > 0:
            threshold = np.mean(integrated[peaks]) * 0.50
            qrs_peaks = peaks[integrated[peaks] > threshold]
        else:
            qrs_peaks = np.array([])
            
        return qrs_peaks