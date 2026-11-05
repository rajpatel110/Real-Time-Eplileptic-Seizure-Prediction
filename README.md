# Real-Time Epileptic Seizure Prediction

Final-year team project using CHB-MIT EEG recordings and a feature-based machine learning pipeline. We build and verify one milestone at a time.

**Current milestone: Step 1, environment and project setup.** No EEG processing, prediction, intensity scoring or user interface has been implemented yet. No measured model performance is claimed.

## Open the correct folder

The clone currently sits inside a second folder with the same name. Open the folder that contains this README and `.git` in VS Code.

## Use the prepared environment (Git Bash on Windows)

From the repository root:

```bash
source .venv/Scripts/activate
python scripts/check_environment.py
python -m pip check
```

If activation is unavailable, use `.venv/Scripts/python.exe scripts/check_environment.py` directly. In VS Code choose **Python: Select Interpreter**, then `.venv/Scripts/python.exe`.

## Recreate the environment

Use an installed Python 3.11-3.14 interpreter. The verified local setup uses Python 3.14.6. Do not rely on the Windows Store `python` alias if it is not installed.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m pip check
```

`pyproject.toml` declares the direct dependencies. `requirements.txt` records the verified installed versions for reproduction on Windows/Python 3.14; other platforms or Python versions may need to resolve dependencies with `python -m pip install -e .` instead. Editable installation lets scripts import `eeg_project` without modifying `sys.path`.

## Folder map

| Location | Purpose |
| --- | --- |
| `src/eeg_project/` | Reusable EEG processing and ML code |
| `configs/baseline.toml` | Explicit experiment settings and source differences |
| `scripts/` | Commands for checking setup and future data/training tasks |
| `data/raw/` | Original EDF recordings and seizure annotations |
| `data/interim/` | Intermediate processing outputs |
| `data/processed/` | Features, labels and split metadata |
| `models/` | Fitted pipelines and metadata |
| `reports/figures/` | Evaluation figures |
| `notebooks/` | Exploration, separate from reusable processing code |
| `tests/` | Tests added alongside meaningful pipeline behavior |
| `app/` | Future Streamlit application |

Data, fitted models, generated reports and `.venv` are excluded from Git. Empty placeholders keep the folder structure visible.

## Build milestones

1. Environment, package layout and configuration.
2. Download a small CHB-MIT subset; inspect EDF metadata and seizure annotations.
3. Implement preprocessing, segmentation, labeling and feature extraction.
4. Train, evaluate and save Logistic Regression; compare additional models later.
5. Define and evaluate an experimental EEG signal-intensity measure.
6. Build Streamlit upload, visualization, predictions and recorded-file playback.
7. Verify the complete workflow and prepare documentation and viva material.

## Scientific boundaries

The initial classifier distinguishes pre-ictal from inter-ictal windows. A 600-second pre-ictal definition does not establish a ten-minute warning capability. Recorded-file playback will be labeled simulated streaming. Signal intensity will not be described as clinical seizure severity.

The primary evaluation will separate recordings across training, validation and testing. Patient-independent claims require separate patients. Historical random-window results may be reproduced as a labeled comparison only. See `docs/DECISIONS.md` for source conflicts and decisions still pending data inspection.

## Sources

- Team baseline: https://github.com/rishita-7/epileptic-seizure-prediction-intensity-classification
- Dataset: https://physionet.org/content/chbmit/1.0.0/
- MNE: https://mne.tools/stable/
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/

This repository is a new implementation informed by the team baseline, presentations and discussion. Contributions and experimental results should be documented as work is completed.
