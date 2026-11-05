"""Relative EEG amplitude from the existing window feature representation."""
import numpy as np


def window_amplitude_uv(features, names):
    """Return median channel RMS in microvolts for each filtered EEG window."""
    means = [i for i, name in enumerate(names) if name.endswith(":mean")]
    variances = [i for i, name in enumerate(names) if name.endswith(":variance")]
    if not means or [names[i].rsplit(":", 1)[0] for i in means] != [
        names[i].rsplit(":", 1)[0] for i in variances
    ]:
        raise ValueError("Mean and variance columns must describe the same ordered channels")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Feature matrix does not match the feature names")
    mean, variance = matrix[:, means], matrix[:, variances]
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Invalid EEG amplitude features")
    return np.median(np.sqrt(variance + mean**2), axis=1) * 1e6


def relative_amplitude(amplitude_uv, reference_uv):
    if not np.isfinite(reference_uv) or reference_uv <= 0:
        raise ValueError("Reference amplitude must be finite and positive")
    return np.asarray(amplitude_uv) / reference_uv
