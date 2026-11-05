# Step 3: feature dataset

Run `python scripts/build_features.py` after downloading the verified sample.
The local output is `data/processed/chb01_features.npz`. It contains X, binary y,
feature names, recording and subject IDs, window times, and configuration.

The pipeline keeps the original bipolar reference and drops only identical
repeated channel labels. It uses causal fourth-order Butterworth band-pass
filtering at 0.5–40 Hz and a 50 Hz notch with Q=30. Each 10-second window,
advanced by 5 seconds, produces mean, standard deviation, variance, and
four mean Welch PSD band values per channel.

Positive windows must fit entirely within the 600 seconds before onset.
Windows overlapping seizures, the following 600 seconds, or a pre-ictal
boundary are excluded. The postictal duration is an experimental choice.
Labels use recording-local annotations. Negative windows are eligible
background under this policy.

The sample produced 1,306 rows, including 118 positives, and 154 features
from 22 channels. All positives come from one recording. Add recordings
with additional seizure events before evaluating with separate splits.
