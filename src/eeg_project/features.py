"""Shared EEG preprocessing and features for training and offline inference."""
import mne
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, welch

from eeg_project.dataset import read_edf_channel_labels

BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30))


def window_label(start, end, events, preictal, postictal):
    """Label half-open intervals, skipping mixed and seizure-adjacent windows."""
    positive = False
    for event in events:
        if start < event.end + postictal and end > event.start:
            return None
        boundary = event.start - preictal
        if start < event.start and end > boundary:
            if start < boundary:
                return None
            positive = True
    return int(positive)


def extract_features(window, rate):
    frequencies, psd = welch(window, fs=rate, nperseg=256, axis=-1)
    values = [window.mean(axis=-1), window.std(axis=-1), window.var(axis=-1)]
    for low, high in BANDS:
        bins = (frequencies >= low) & (frequencies < high)
        if not bins.any():
            raise ValueError("Welch resolution does not cover all configured bands")
        values.append(psd[:, bins].mean(axis=-1))
    return np.stack(values, axis=1).ravel()


def preprocess(path, filters):
    labels = read_edf_channel_labels(path)
    with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as raw:
        rate = float(raw.info["sfreq"])
        signal = raw.get_data()
    keep = {}
    for index, name in enumerate(labels):
        if name in keep:
            if not np.array_equal(signal[index], signal[keep[name]]):
                raise ValueError(f"Ambiguous duplicate channel {name} in {path.name}")
        else:
            keep[name] = index
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite samples in {path.name}")
    bandpass = butter(4, [filters["low_hz"], filters["high_hz"]], fs=rate, btype="bandpass", output="sos")
    notch = tf2sos(*iirnotch(filters["notch_hz"], Q=30, fs=rate))
    signal = sosfilt(notch, sosfilt(bandpass, signal[list(keep.values())], axis=-1), axis=-1)

    return tuple(keep), rate, signal


def process(path, events, filters, settings):
    channels, rate, signal = preprocess(path, filters)
    duration = signal.shape[-1] / rate
    if any(not 0 <= event.start < event.end <= duration for event in events):
        raise ValueError(f"Invalid annotation in {path.name}")

    width, stride = (round(settings[key] * rate) for key in ("window_seconds", "stride_seconds"))
    features, targets, starts = [], [], []
    for offset in range(0, signal.shape[-1] - width + 1, stride):
        start = offset / rate
        label = window_label(start, (offset + width) / rate, events,
                             settings["preictal_seconds"], settings["postictal_seconds"])
        if label is not None:
            features.append(extract_features(signal[:, offset:offset + width], rate))
            targets.append(label)
            starts.append(start)
    return channels, features, targets, starts
