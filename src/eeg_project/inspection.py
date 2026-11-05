"""Inspect raw EEG without filtering, referencing, resampling or labeling."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import warnings

import mne
import numpy as np

from eeg_project.dataset import RecordingAnnotation, read_edf_channel_labels, sha256_file


def inspect_recording(path: Path, annotation: RecordingAnnotation, figure_path: Path) -> dict:
    if path.name != annotation.filename:
        raise ValueError("Recording and annotation filename mismatch")
    original_labels = read_edf_channel_labels(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / sfreq
        if len(original_labels) != len(raw.ch_names):
            raise ValueError("EDF and MNE channel counts differ")
        for event in annotation.seizures:
            if not 0 <= event.start < event.end <= duration:
                raise ValueError(f"Seizure annotation outside recording: {event}")
        duplicates = {label: [i for i, name in enumerate(original_labels) if name == label]
                      for label, count in Counter(original_labels).items() if count > 1}
        duplicate_differences = {label: 0.0 for label in duplicates}
        minima = np.full(len(raw.ch_names), np.inf)
        maxima = np.full(len(raw.ch_names), -np.inf)
        # Scan the complete file in bounded one-minute chunks, not all into RAM.
        chunk_samples = max(1, int(sfreq * 60))
        for start in range(0, raw.n_times, chunk_samples):
            block = raw.get_data(start=start, stop=min(start + chunk_samples, raw.n_times))
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite samples near {start / sfreq:g}s")
            minima = np.minimum(minima, block.min(axis=1))
            maxima = np.maximum(maxima, block.max(axis=1))
            for label, indices in duplicates.items():
                difference = np.max(np.abs(block[indices[1:]] - block[indices[0]]))
                duplicate_differences[label] = max(duplicate_differences[label], float(difference))
        preview_start = max(0.0, annotation.seizures[0].start - 10) if annotation.seizures else 0.0
        preview_end = min(preview_start + 20, duration)
        plot_preview(raw, annotation, preview_start, preview_end, figure_path)
        return {
            "filename": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "sampling_rate_hz": sfreq,
            "duration_seconds": duration,
            "samples_per_channel": int(raw.n_times),
            "channel_count": len(raw.ch_names),
            "original_channel_labels": original_labels,
            "mne_channel_names": raw.ch_names,
            "duplicate_labels_zero_based_indices": duplicates,
            "duplicate_label_max_difference_volts": duplicate_differences,
            "channel_min_microvolts": (minima * 1e6).tolist(),
            "channel_max_microvolts": (maxima * 1e6).tolist(),
            "flat_channels": [name for name, lo, hi in zip(raw.ch_names, minima, maxima) if lo == hi],
            "all_samples_finite": True,
            "preloaded": bool(raw.preload),
            "seizures": [asdict(event) for event in annotation.seizures],
            "reader_warnings": [str(item.message) for item in caught],
            "preview_interval_seconds": [preview_start, preview_end],
            "preview_file": figure_path.name,
        }
    finally:
        raw.close()


def plot_preview(raw, annotation, start, end, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(4, len(raw.ch_names))
    sfreq = float(raw.info["sfreq"])
    first, last = int(start * sfreq), int(end * sfreq)
    signals = raw.get_data(picks=list(range(count)), start=first, stop=last) * 1e6
    times = np.arange(first, last) / sfreq
    fig, axes = plt.subplots(count, 1, sharex=True, figsize=(12, 7), squeeze=False)
    for index, axis in enumerate(axes[:, 0]):
        axis.plot(times, signals[index], color="#155e75", linewidth=0.65)
        axis.set_ylabel(f"{raw.ch_names[index]}\nµV", fontsize=9)
        axis.grid(alpha=0.15)
        for event in annotation.seizures:
            if event.start < end and event.end > start:
                axis.axvspan(max(start, event.start), min(end, event.end), color="#ef4444", alpha=0.12)
                if start <= event.start < end:
                    axis.axvline(event.start, color="#b91c1c", linewidth=1)
        axis.set_xlim(start, end)
    axes[-1, 0].set_xlabel("Seconds from recording start")
    fig.suptitle(f"{annotation.filename} · raw EEG, first {count} channels\nRed shading = annotated seizure; no preprocessing applied", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
