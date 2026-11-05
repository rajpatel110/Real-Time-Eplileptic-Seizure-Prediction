# Training feature quality diagnostic

Run `python scripts/inspect_training_features.py` after building the approved
user-notes archive. It inspects only configured training rows and refuses to
overwrite its report. No model is fitted, no features/rows are removed, and no
validation or test predictions are computed. It uses existing NumPy utilities.

## Integrity

All 3204 training rows and 154 feature columns are finite. No feature is constant
across the complete training subset. No channel-window has exactly zero variance.
For every channel, standard deviation squared matches variance within the stated
numerical tolerance; magnitude and mean-PSD features are nonnegative. These are
basic integrity checks, not proof of artifact-free EEG. Near-flat segments,
clipping, movement and electrode artifacts require additional signal inspection.

## Differences between recordings

Median channel RMS, summarized by the median across windows (microvolts):

| Recording | Background | Pre-ictal |
| --- | --- | --- |
| 01 | 22.44 | No positive windows |
| 03 | 41.52 | 20.28 |
| 04 | 24.61 | 32.64 |
| 08 | 20.03 | No positive windows |
| 16 | 49.56 | 45.56 |

Amplitude falls before the seizure in 03 but rises in 04. This is descriptive
filtered-signal amplitude, not seizure severity. Across the three seizure
recordings, only 8/154 individual features have median class differences with
the same nonzero sign in all three. Their minimum standardized differences are
small (largest about 0.0201 standard deviations). This does not prove the other
features are useless: multivariate relationships and distribution shape are not
captured by median differences, and only three events are available.

Recording 16 has feature-median shifts reaching about 2.80 standard deviations
relative to the other training recordings. The report records the most shifted
features and the timestamps of the highest-amplitude windows for later review.

For screening, bounds are Q1 - 3*IQR and Q3 + 3*IQR computed on the OTHER training
recordings for each left-out recording. Any-feature window flag rates range
from 27.4% to 82.1%, but with 154 correlated features this is not an artifact rate.
Median within-window flagged feature fractions range from 0% to 14.9%. Flat-IQR
features are excluded from this flag calculation. No automatic rejection is
justified by these flags alone.

## Decision

Retain the approved features, preprocessing and Logistic Regression pipeline.
The results support investigating recording variability, not claiming a proven
cause of low precision or a model improvement. The next bounded experiment can
compare Logistic Regression regularization using recording-grouped folds within
training data. Keep scaling fitted inside each fold and keep external validation
and holdout data separate. Do not select the eight direction-consistent features
on all training rows and then claim unbiased cross-validation on those rows.

Verification: report counts and schema were checked; all finite/nonnegative and
std/variance consistency assertions passed; Ruff passed. No new dependency added.
