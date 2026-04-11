import numpy as np

class Evaluator:
    """Class to calculate clinical performance metrics for QRS detection."""
    
    @staticmethod
    def match_peaks(true_peaks, detected_peaks, window_samples):
        """Determines True Positives using a temporal tolerance window."""
        tp = 0
        matched_indices = set()
        
        for gt in true_peaks:
            # Find detected peaks within the tolerance window
            matches = np.where(np.abs(detected_peaks - gt) <= window_samples)[0]
            if len(matches) > 0:
                # Count only the first match to avoid over-counting
                tp += 1
                matched_indices.add(matches[0])
                
        fp = len(detected_peaks) - len(matched_indices)
        fn = len(true_peaks) - tp
        return tp, fp, fn

    @staticmethod
    def get_metrics(tp, fp, fn):
        """Calculates Sensitivity, Positive Predictivity, F1-Score, and Error Rate."""
        se = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        pp = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        
        # F1-Score (normalized 0 to 1)
        f1 = (2 * se * pp) / (se + pp) / 100 if (se + pp) > 0 else 0
        
        # Error Rate (Relative to Ground Truth)
        error_rate = ((fp + fn) / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        
        return {
            "Sensitivity": round(se, 2),
            "Precision": round(pp, 2),
            "F1_Score": round(f1, 4),
            "Error_Rate": round(error_rate, 2)
        }