"""Dataset-free example; output is not a seizure-prediction result."""
import numpy as np
from eeg_project.features import filter_signal, extract_features
from eeg_project.evaluation import binary_metrics

rate = 256
times = np.arange(rate * 10) / rate
signal = (20e-6 * np.sin(2 * np.pi * 10 * times))[None, :]
filtered = filter_signal(signal, rate, {"low_hz": 0.5, "high_hz": 40, "notch_hz": 50})
features = extract_features(filtered, rate)
assert features.shape == (7,) and np.isfinite(features).all()
print("Synthetic one-channel feature shape:", features.shape)
print("Toy metrics (not model results):", binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.3, 0.9], 0.5))
