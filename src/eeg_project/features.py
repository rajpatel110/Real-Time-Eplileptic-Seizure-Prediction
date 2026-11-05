"""Shared EEG preprocessing and features for training and offline inference."""
from pathlib import Path

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
    window = np.asarray(window, dtype=float)
    if window.ndim != 2 or not window.shape[0] or window.shape[1] < 256:
        raise ValueError("EEG window must have channels and at least 256 samples")
    if not np.isfinite(window).all() or not np.isfinite(rate) or rate <= 60:
        raise ValueError("EEG must be finite and sampling rate must exceed 60 Hz")
    frequencies, psd = welch(window, fs=rate, nperseg=256, axis=-1)
    values = [window.mean(axis=-1), window.std(axis=-1), window.var(axis=-1)]
    for low, high in BANDS:
        bins = (frequencies >= low) & (frequencies < high)
        if not bins.any():
            raise ValueError("Welch resolution does not cover all configured bands")
        values.append(psd[:, bins].mean(axis=-1))
    return np.stack(values, axis=1).ravel()


def preprocess(path, filters):
    path = Path(path)
    labels = read_edf_channel_labels(path)
    with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as raw:
        rate = float(raw.info["sfreq"])
        if len(labels) != len(raw.ch_names) or any(not name for name in labels):
            raise ValueError("EDF channel labels do not match decoded signal channels")
        if raw.n_times * len(labels) > 50_000_000:
            raise ValueError("Recording exceeds the 50 million decoded sample limit; split it first")
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
    signal = filter_signal(signal[list(keep.values())], rate, filters)

    return tuple(keep), rate, signal


def process(path, events, filters, settings):
    channels, rate, signal = preprocess(path, filters)
    duration = signal.shape[-1] / rate
    if any(not 0 <= event.start < event.end <= duration for event in events):
        raise ValueError(f"Invalid annotation in {path.name}")

    for key in ("window_seconds", "stride_seconds", "preictal_seconds", "postictal_seconds"):
        if key not in settings or not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    width, stride = (round(settings[key] * rate) for key in ("window_seconds", "stride_seconds"))
    if width < 256 or stride < 1:
        raise ValueError("Window must contain at least 256 samples and stride at least one sample")
    if signal.shape[-1] < width:
        raise ValueError("Recording is shorter than one complete window")
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


def filter_signal(signal, rate, filters):
    """Causal zero-initial-state filtering of channels x samples, in volts.

    Call once per recording. Calling independently on chunks resets filter state.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or 0 in signal.shape or not np.isfinite(signal).all():
        raise ValueError("Signal must be a nonempty finite channels-by-samples matrix")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    try:
        low, high, notch_hz = (float(filters[k]) for k in ("low_hz", "high_hz", "notch_hz"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Filters require numeric low_hz, high_hz and notch_hz") from exc
    if not 0 < low < high < rate / 2 or not 0 < notch_hz < rate / 2:
        raise ValueError("Filter frequencies must be positive and below Nyquist, with low_hz < high_hz")
    bandpass = butter(4, [low, high], fs=rate, btype="bandpass", output="sos")
    notch = tf2sos(*iirnotch(notch_hz, Q=30, fs=rate))
    return sosfilt(notch, sosfilt(bandpass, signal, axis=-1), axis=-1)
