"""
ECG Algorithm Benchmark Package
A comprehensive tool for stress-testing Pan-Tompkins, Wavelet, and Hilbert QRS detectors.
"""

from .detectors import ECGDetectors
from .evaluator import Evaluator
from .noise_utils import NoiseGenerator

__version__ = "1.0.0"
__author__ = "Divine Matengambiri"

# Define the public API
__all__ = [
    "ECGDetectors",
    "Evaluator",
    "NoiseGenerator",
]