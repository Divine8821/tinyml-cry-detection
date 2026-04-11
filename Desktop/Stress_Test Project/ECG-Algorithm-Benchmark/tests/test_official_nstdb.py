import wfdb
import os
import pandas as pd
import numpy as np
from src import ECGDetectors, Evaluator

def run_official_nstdb_test():
    # Define paths to your local data
    MIT_DIR = 'data/mitbih'
    NST_DIR = 'data/nstdb'
    
    # Official NSTDB records are 118 and 119 with SNR suffixes
    base_records = ['118', '119']
    snr_suffixes = ['e_6', 'e00', 'e06', 'e12', 'e18', 'e24']
    results = []

    for base in base_records:
        # Load Ground Truth from LOCAL mitdb folder
        ann_path = os.path.join(MIT_DIR, base)
        ann = wfdb.rdann(ann_path, 'atr')
        true_peaks = ann.sample[np.isin(ann.symbol, ['N', 'L', 'R', 'V'])]
        
        for suffix in snr_suffixes:
            rec_id = base + suffix
            print(f"Testing Official NSTDB Record: {rec_id}")
            
            # Load Pre-mixed noisy signal from LOCAL nstdb folder
            rec_path = os.path.join(NST_DIR, rec_id)
            
            try:
                record = wfdb.rdrecord(rec_path, channels=[0])
                ecg = record.p_signal.flatten()
                fs = record.fs
                window = int(0.150 * fs)

                detectors = {
                    'PT': ECGDetectors.pan_tompkins(ecg, fs),
                    'WT': ECGDetectors.wavelet_transform(ecg, fs),
                    'HT': ECGDetectors.hilbert_transform(ecg, fs)
                }

                for name, peaks in detectors.items():
                    tp, fp, fn = Evaluator.match_peaks(true_peaks, peaks, window)
                    m = Evaluator.get_metrics(tp, fp, fn)
                    results.append({
                        "Record": rec_id, 
                        "Algorithm": name, 
                        "SNR": suffix.replace('e', ''), 
                        **m
                    })
            except FileNotFoundError:
                print(f"Skipping {rec_id}: File not found in {NST_DIR}")

    # Ensure results directory exists
    if not os.path.exists('results'): os.makedirs('results')
    
    pd.DataFrame(results).to_csv('results/nstdb_official_results.csv', index=False)
    print("Official NSTDB test complete. Results saved to results/nstdb_official_results.csv")