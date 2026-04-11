import numpy as np

class NoiseGenerator:
    """Utility class to handle noise injection at specific SNR levels."""
    
    @staticmethod
    def calculate_scaling_factor(clean_signal, noise_signal, target_snr):
        """Calculates the multiplier for noise to achieve a target SNR."""
        p_signal = np.mean(clean_signal**2)
        p_noise = np.mean(noise_signal**2)
        
        # Avoid division by zero
        if p_noise == 0: return 0
        
        p_noise_required = p_signal / (10**(target_snr / 10))
        return np.sqrt(p_noise_required / p_noise)

    @staticmethod
    def apply_noise(clean_signal, raw_noise, target_snr):
        """Injects noise into a clean signal, looping noise if it is shorter."""
        if len(raw_noise) < len(clean_signal):
            raw_noise = np.tile(raw_noise, int(np.ceil(len(clean_signal)/len(raw_noise))))
        
        raw_noise = raw_noise[:len(clean_signal)]
        scale = NoiseGenerator.calculate_scaling_factor(clean_signal, raw_noise, target_snr)
        return clean_signal + (scale * raw_noise)

    @staticmethod
    def generate_composite(bw_noise, ma_noise, em_noise, length):
        """Combines all three noise types using Z-score normalization."""
        def normalize(s): return (s - np.mean(s)) / np.std(s)
        
        # Ensure all noises are normalized and the correct length
        combined = normalize(bw_noise[:length]) + normalize(ma_noise[:length]) + normalize(em_noise[:length])
        return combined / 3.0