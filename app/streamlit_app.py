"""Local research demo for recorded CHB01-style EEG."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from eeg_project.features import extract_features, preprocess
from eeg_project.intensity import relative_amplitude, window_amplitude_uv

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/chb01_logistic.joblib"
REFERENCE = ROOT / "reports/chb01_intensity.json"
SAMPLE = ROOT / "data/raw/chb01/chb01_18.edf"


@st.cache_data(max_entries=2, show_spinner="Processing recorded EEG…")
def analyze(edf, model_version, reference_version):
    artifact = joblib.load(MODEL)
    calibration = json.loads(REFERENCE.read_text(encoding="utf-8"))
    config = artifact["configuration"]
    if calibration["configuration"] != config:
        raise ValueError("The amplitude reference and model use different preprocessing")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recording.edf"
        path.write_bytes(edf)
        channels, rate, signal = preprocess(path, config["filters"])
    names = artifact["feature_names"]
    if tuple(name.rsplit(":", 1)[0] for name in names[::7]) != channels or rate != 256:
        raise ValueError("Use an EDF with the CHB01 model's channel order and 256 Hz sampling.")
    window = config["windowing"]
    width = round(window["window_seconds"] * rate)
    stride = round(window["stride_seconds"] * rate)
    offsets = np.arange(0, signal.shape[-1] - width + 1, stride)
    if not offsets.size:
        raise ValueError("The recording needs at least one complete 10-second window.")
    features = np.stack([extract_features(signal[:, i:i + width], rate) for i in offsets])
    scores = artifact["pipeline"].predict_proba(features)[:, 1]
    ratio = relative_amplitude(window_amplitude_uv(features, names), calibration["reference_uv"])
    return channels, rate, signal.astype(np.float32), offsets, width, scores, ratio


def main():
    st.set_page_config(page_title="EEG Recording Explorer", layout="wide")
    st.title("EEG Recording Explorer")
    st.caption("Offline research demo · CHB01 Logistic Regression baseline")
    st.info("The held-out experiment had 10.9% precision at its selected threshold. "
            "Scores are exploratory and the interface does not issue alerts.")
    if not MODEL.is_file() or not REFERENCE.is_file():
        st.error("Run scripts/train_baseline.py and scripts/evaluate_intensity.py first.")
        return
    source = st.radio("Recording source", ["CHB01 held-out sample", "Upload EDF"], horizontal=True)
    if source == "Upload EDF":
        upload = st.file_uploader("CHB01-compatible EDF", type=["edf"])
        if upload is None:
            return
        if upload.size > 50 * 1024**2:
            st.error("Choose an EDF of 50 MiB or less.")
            return
        content = upload.getvalue()
    else:
        if not SAMPLE.is_file():
            st.error("Run scripts/train_baseline.py to download the held-out sample.")
            return
        content = SAMPLE.read_bytes()
    try:
        channels, rate, signal, offsets, width, scores, ratio = analyze(
            content, MODEL.stat().st_mtime_ns, REFERENCE.stat().st_mtime_ns)
    except (ValueError, OSError, RuntimeError, UnicodeError, IndexError) as error:
        st.error(f"Could not analyze this recording. {error}")
        return
    ends = (offsets + width) / rate
    st.subheader("Recording timeline")
    st.line_chart(pd.DataFrame({"Window end (s)": ends, "Model score": scores}), x="Window end (s)", y="Model score")
    st.line_chart(pd.DataFrame({"Window end (s)": ends, "Relative RMS amplitude": ratio}), x="Window end (s)", y="Relative RMS amplitude")
    st.caption("Amplitude is relative to training background. It measures signal magnitude.")
    st.subheader("Recorded EEG playback")
    index = st.select_slider("Window end (seconds)", options=list(range(len(offsets))),
                             format_func=lambda i: f"{ends[i]:g}")
    channel = st.selectbox("Bipolar channel", channels)
    start = offsets[index]
    selection = signal[channels.index(channel), start:start + width] * 1e6
    left, right = st.columns(2)
    left.metric("Model score", f"{scores[index]:.3f}")
    right.metric("Relative RMS amplitude", f"{ratio[index]:.2f}×")
    st.line_chart(pd.DataFrame({"Time (s)": (start + np.arange(width)) / rate,
                                "Filtered EEG (µV)": selection}), x="Time (s)", y="Filtered EEG (µV)")


main()
