# Project handoff and viva guide

## Run the existing local demo

Use Git Bash and open the inner repository folder.

```bash
cd /c/PROJECTS/Real-Time-Eplileptic-Seizure-Prediction/Real-Time-Eplileptic-Seizure-Prediction
./.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

## Reproduce from a clean checkout

Create a virtual environment with Python 3.14 on Windows, install the pinned
dependencies, then execute these commands from the repository root.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/download_sample.py
python scripts/inspect_sample.py
python scripts/build_features.py
python scripts/train_baseline.py
python scripts/evaluate_intensity.py
python -m unittest discover -s tests -v
python scripts/verify_app.py
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Training downloads the additional recordings. Data, feature archives, models
and generated reports are excluded from Git. Network access is needed to
recreate those artifacts. The baseline and demo were published in commit 0342d41. See CHANGELOG.md for
subsequent improvements awaiting review.

## Explain the project in one minute

We built a modular EEG analysis prototype using CHB-MIT recordings. We verify
downloads, parse seizure annotations, preserve bipolar channels, and remove an
identical duplicate channel. Causal filtering prepares the recordings for
10-second windows advanced every five seconds. Seven statistical and spectral
features from each of 22 channels give 154 features per window. A scaler and
balanced Logistic Regression learn pre-ictal versus eligible background
windows. We separate recordings across training, validation and testing.
Streamlit shows model scores alongside EEG traces and an experimental relative
amplitude measure.

## Defend the actual results

These are historical baseline results. Recordings 18 and 20 later became
development data; they are no longer an untouched holdout.

| Test measurement | Value |
| --- | --- |
| Test windows | 1,110 |
| Positive test windows | 119 |
| Average precision | 0.32865 |
| Precision at selected threshold | 0.10878 |
| Recall at selected threshold | 1.0 |
| F1 at selected threshold | 0.19621 |
| TN / FP / FN / TP | 16 / 975 / 0 / 119 |

The validation-F1 threshold was approximately 5.57e-9. It classified almost
every test window as positive. This baseline requires further development
before a useful alert mechanism. Window counts reflect overlapping samples,
so they cannot be interpreted as independent seizure events or alarm counts.
The test set contains one seizure from CHB01.

## Likely viva questions

**Why 154 features when the earlier slides said 161?**
The source has 23 channels, but T8-P8 occurs twice with identical samples.
Our implementation keeps 22 distinct channel labels with seven features each.

**What are the seven features?**
Mean, standard deviation, variance, and mean Welch PSD within delta, theta,
alpha and beta bands. The last four summarize spectral density within bands.
They are not integrated band-energy values.

**What changed from the original team approach?**
We use the repository's 0.5-40 Hz and 50 Hz frequencies, a causal fourth-order
Butterworth band-pass and Q=30 notch, and preserve the original bipolar
reference. These filter choices and duplicate removal differ from an exact
reproduction of the team's earlier implementation.

**How are labels assigned?**
A positive window fits completely inside the 600 seconds preceding a seizure.
Windows overlapping a seizure, its following 600 seconds, or a mixed pre-ictal
boundary are excluded. The 600-second postictal exclusion is an experimental
choice. Recording-local annotations limit boundary context across files.

**Does the model predict ten minutes ahead?**
The experiment labels a ten-minute pre-ictal interval. A specific reliable
warning horizon still requires event-level evaluation.

**Why separate recordings?**
Overlapping windows share samples. Grouping entire recordings prevents shared
windows from entering different splits. All splits still belong to CHB01.

**Why is precision low despite full recall?**
The selected threshold is extremely permissive. It found all positives while
producing 975 false-positive windows. Validation used only one seizure
recording, and its selected threshold transferred poorly.

**What does intensity mean here?**
Median channel RMS divided by the training-background median RMS (23.008 µV).
It describes filtered EEG amplitude. Clinical severity labels were unavailable.
The held-out background and pre-ictal distributions overlap substantially.

**Is playback live?**
It is manual exploration of recorded EDF windows. Full-file inference runs
before the interface displays the timeline.

## Verification boundary

The pipeline was executed on the downloaded recordings and the saved model
reproduced scores after reload. Shared preprocessing reproduced stored
held-out features exactly. Application automation covers the sample, channel
and window navigation, and injected EDF upload content. Native browser file
selection and a clean-machine installation require separate manual checks.
