# EEG Seizure Prediction Research Prototype

Feature-based pre-ictal versus background classification on CHB-MIT EEG, with
an offline Streamlit explorer. This is a final-year research project, not a
clinically validated medical device or a reliable seizure-warning system.

## Current agreed scope

Forward work uses the supplied Python stack and StandardScaler plus balanced
Logistic Regression only. The active configuration now specifies 0.5-50 Hz and
60 Hz notch, following the user notes. Existing demo/model results below used
the previous 0.5-40 Hz / 50 Hz profile. They have not been retrained or relabeled
as results of the new settings. The user approved retaining 22 distinct bipolar channels (154 features);
average referencing is not applied.
Historical nonlinear comparison configs are retained for traceability; the
current experiment runner rejects them.

## What works

- SHA256-verified CHB-MIT downloads and strict seizure-summary parsing.
- Shared causal filtering, duplicate-channel checks and seven features/channel.
- Recording-separated Logistic Regression experiments and relative RMS amplitude.
- Offline EDF upload, waveform inspection and manual window playback.
- Dataset-free tests, reusable evaluation APIs and a TOML development runner.

The original evaluated baseline obtained **10.9% precision, 100% recall and
0.32865 average precision** on 1,110 windows from two CHB01 recordings containing
one seizure. Its threshold produced 975 false-positive windows. Those recordings
were subsequently reused for development; they are no longer an untouched test
set. Expanded development experiments have not met the 50% precision/50% recall
selection target. Fresh recordings 26 and 32 remain unscored by the new runner.
See [results and boundaries](docs/HANDOFF.md) and [feature experiment](docs/FEATURE_INVESTIGATION.md).

## Install and run checks without data

Clone this repository and work in the folder containing `pyproject.toml`.
Python 3.11-3.14 is declared; Windows/Python 3.14 is locally verified.

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# Or Git Bash on Windows: source .venv/Scripts/activate
# Or Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m unittest discover -s tests -v
python -m ruff check .
python examples/synthetic_pipeline.py
```

If activation is unavailable, use `.venv/Scripts/python.exe` directly on Windows.
`requirements.txt` pins the original Windows/Python 3.14 runtime environment:
install with `python -m pip install -r requirements.txt`. It is a version snapshot,
not a universal hash-locked environment. `pyproject.toml` supports fresh dependency
resolution; versions in experiment reports identify what actually ran.

## Dataset and original demo

The public [CHB-MIT dataset](https://physionet.org/content/chbmit/1.0.0/) is
obtained separately under its own terms; it is not included in this repository.
Use only the official download source. No credentials or paid service is needed.

```bash
python scripts/download_sample.py
python scripts/inspect_sample.py
python scripts/build_features.py
python scripts/train_baseline.py
python scripts/evaluate_intensity.py
python scripts/verify_app.py
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The initial download is about 81 MiB. Training downloads additional recordings
and needs substantially more disk/RAM. Feature extraction loads each recording
into memory. Data, models and reports stay local and are ignored by Git.
The app needs the local model and intensity report generated above. Only load
joblib models you generated and trust: loading one can execute Python code.
The EDF UI accepts CHB01-compatible channels at 256 Hz and limits uploads to
50 MiB; preprocessing also caps decoded samples. This is a local-use interface.

## Reproducible development experiments

```bash
python scripts/train_expanded.py
python -m eeg_project.experiment --config configs/development.toml --output reports/development_run.json
```

The first command builds the archive and runs the historical expanded experiment.
The new runner uses an existing archive, fits only training rows, selects a
threshold using development data and never predicts on the test split. Outputs
include configuration, input hashes, versions and Git state; existing reports
are not overwritten. See [API and experiment guide](docs/RESEARCH_API.md).

## Methods and limitations

EEG retains its original bipolar reference. Identical duplicate T8-P8 channels
are collapsed, leaving 22 channels and 154 features on the current subset.
Filtering uses a causal fourth-order 0.5-40 Hz Butterworth band-pass and 50 Hz
notch (Q=30), with zero initial state per recording. Ten-second windows advance
by five seconds. Features are mean, standard deviation, variance, and mean
Welch PSD in delta/theta/alpha/beta bands. Mean PSD is not integrated bandpower.

A positive window lies entirely within 600 seconds before an annotated seizure.
Ictal, 600-second postictal and mixed-boundary windows are excluded. Boundary
context across recordings and filter startup transients need further study.
All current real-data experiments use CHB01: no patient-independent claim is
supported. Overlapping windows are not independent events. Average precision is
reported explicitly, rather than mislabeled as trapezoidal PR-AUC. There are no
validated event-level alarm metrics or clinical severity labels. RMS amplitude
is only a relative signal-magnitude measure. Playback is offline, not live EEG.

## Contribute and roadmap

See [CONTRIBUTING](CONTRIBUTING.md), [SECURITY](SECURITY.md), [changelog](CHANGELOG.md),
[audit](docs/REPOSITORY_AUDIT.md) and [five proposed contributor issues](docs/CONTRIBUTOR_ISSUES.md).
Priorities: cross-recording labels, event-level evaluation, multiple subjects,
bounded streaming preprocessing and portable locked environments. CI and
Dependabot configurations are included; hosted execution requires publication.
Original project code is [MIT licensed](LICENSE), with owner approval. CHB-MIT
data and third-party material are excluded and retain their own terms.

The implementation is informed by the
[team baseline](https://github.com/rishita-7/epileptic-seizure-prediction-intensity-classification),
with differences documented in [decisions](docs/DECISIONS.md). Third-party code
and data retain their respective terms.
