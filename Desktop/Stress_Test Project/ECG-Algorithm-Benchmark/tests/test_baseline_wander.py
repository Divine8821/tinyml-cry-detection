import os
import pandas as pd
from src import ECGDetectors, Evaluator, NoiseGenerator
import wfdb
import numpy as np

def run_bw_test():
    MIT_DIR = 'data/mitbih'
    NST_DIR = 'data/nstdb'
    records = wfdb.get_record_list('mitdb')
    
    # Load BW noise from NSTDB
    noise_rec = wfdb.rdrecord(os.path.join(NST_DIR, 'bw'), channels=[0])
    raw_noise = noise_rec.p_signal.flatten()
    
    snr_levels = [24, 18, 12, 6, 0, -6]
    results = []

    for snr in snr_levels:
        print(f"Testing Baseline Wander at {snr}dB...")
        # Metrics storage for this SNR
        metrics = {alg: {'tp': 0, 'fp': 0, 'fn': 0} for alg in ['pt', 'wt', 'ht']}

        for rec_id in records:
            path = os.path.join(MIT_DIR, rec_id)
            record = wfdb.rdrecord(path, channels=[0])
            clean_ecg = record.p_signal.flatten()
            
            ann = wfdb.rdann(path, 'atr')
            true_peaks = ann.sample[np.isin(ann.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])]

            noisy_ecg = NoiseGenerator.apply_noise(clean_ecg, raw_noise, snr)
            fs = record.fs
            window = int(0.150 * fs)

            # Run all three detectors
            peaks_dict = {
                'pt': ECGDetectors.pan_tompkins(noisy_ecg, fs),
                'wt': ECGDetectors.wavelet_transform(noisy_ecg, fs),
                'ht': ECGDetectors.hilbert_transform(noisy_ecg, fs)
            }

            for alg, peaks in peaks_dict.items():
                tp, fp, fn = Evaluator.match_peaks(true_peaks, peaks, window)
                metrics[alg]['tp'] += tp
                metrics[alg]['fp'] += fp
                metrics[alg]['fn'] += fn

        for alg in ['pt', 'wt', 'ht']:
            res = Evaluator.get_metrics(metrics[alg]['tp'], metrics[alg]['fp'], metrics[alg]['fn'])
            results.append({"Algorithm": alg.upper(), "SNR": snr, **res})

    pd.DataFrame(results).to_csv('results/bw_results.csv', index=False)
    print("Baseline Wander test complete. Results saved to results/bw_results.csv")

if __name__ == "__main__":
    run_bw_test()