# Logistic Regression regularization comparison

The candidate grid was fixed before execution: C = 0.001, 0.01, 0.1, 1.0.
Smaller C means stronger regularization. All other settings match the approved
pipeline: 154 features, original bipolar channels, 0.5-50 Hz band-pass, 60 Hz
notch, StandardScaler, balanced class weights, seed 42 and max_iter=3000.

Each candidate was fitted in five leave-one-training-recording-out folds.
Scalers were fitted inside each fold. Selection maximized pooled out-of-fold
F1 at the fixed 0.5 threshold, with smaller C breaking ties. No validation or
holdout recordings entered this selection.

| C | Training-fold precision | Recall | F1 | Average precision |
| --- | --- | --- | --- | --- |
| 0.001 | 19.52% | 64.79% | 30.01% | 0.16281 |
| 0.01 | 18.65% | 60.56% | 28.51% | 0.15468 |
| 0.1 | 17.39% | 50.99% | 25.93% | 0.15080 |
| 1.0 | 17.05% | 46.48% | 24.94% | 0.15618 |

The selected C=0.001 was then fitted on all training rows and checked once on
the existing development recordings, using configs/user_notes_regularized.toml.
Those recordings have been reused in earlier experiments: this is exploratory
validation, not a new untouched test result.

| Development metric at threshold 0.5 | Previous C=0.01 | Selected C=0.001 |
| --- | --- | --- |
| Precision | 11.48% | 13.00% |
| Recall | 51.45% | 56.85% |
| F1 | 18.77% | 21.16% |
| Average precision | 0.11568 | 0.13255 |

Selected-candidate confusion matrix: [[537, 917], [104, 137]]. There are still
917 false-positive windows. No threshold meets both 50% precision and 50% recall.
Average precision remains below the development positive fraction (0.14218).
This is a modest measured development improvement, not evidence of a useful
alarm system. Selection scores are optimistic, overlapping windows are
correlated, and all data still comes from one patient.

No model was exported or installed in the app. Final holdout predictions were
not computed. The original configuration remains available; the selected
candidate has a separate configuration. No technologies or features were added.

## Reproduce

```bash
python scripts/compare_regularization.py
python -m eeg_project.experiment --config configs/user_notes_regularized.toml --output reports/user_notes_regularized.json
```

Both commands refuse existing report paths. Reports record configurations,
input hashes and dependency versions. Validation checks confirmed disjoint
recording folds and exact reproduction of the previous C=0.01 fold metrics.
Ruff passed. The next priority is more independent training seizure events and
recording diversity, rather than extending the parameter search on the same
small validation set. Dataset expansion should retain the existing pipeline
and preserve the final holdout.
