# Boundary correction and nonlinear model comparison

EDF header date/time offsets for all development recordings and recording 21
agree with the summary timeline. Recording 21 was downloaded from the official
source and SHA256-verified before this check. Dates are used only as relative
recording chronology, not as patient-identifying calendar dates.

## Reproduce

With the expanded archive, official summary, development EDFs and verified
recording 21 available:

```bash
python scripts/audit_boundaries.py
python scripts/correct_boundaries.py
python -m eeg_project.experiment --config configs/boundary_logistic.toml --output reports/boundary_logistic.json
python -m eeg_project.experiment --config configs/boundary_forest.toml --output reports/boundary_forest.json
python -m eeg_project.experiment --config configs/boundary_boosting.toml --output reports/boundary_boosting.json
```

The correction refuses to overwrite its output. It checks parent/archive hashes,
recording dates, old labels and unique row identity. The resulting
chb01_boundary_v1.npz preserves the original archive and embeds correction
provenance, including EDF hashes. Eight development windows are excluded and
four become positive. Total rows change from 5995 to 5987; positives from 710
to 714. Held-out features, labels and row metadata are checked for exact equality.
No signals from recording 21 are added to training or validation.

## Fixed comparison

All three configurations were defined before obtaining their development scores.
They use seed 42, the same split and corrected archive, StandardScaler fitted
on training rows, and balanced class weighting. Logistic Regression uses C=0.01;
Random Forest uses 300 trees, depth 8 and minimum leaf size 10; histogram gradient
boosting uses 150 iterations, 15 leaves and L2=1 with early stopping disabled.
No parameter search was performed. Tree inputs are scaled because tiny raw PSD
magnitudes can interact with numerical split tolerances.

Development contains 1695 windows with 241 positives (14.22%). Metrics below use
the fixed threshold 0.5; AP is average precision, not trapezoidal PR-AUC.

| Model | Precision | Recall | F1 | AP |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.11442 | 0.51037 | 0.18693 | 0.11580 |
| Random Forest | 0.10491 | 0.25726 | 0.14904 | 0.11827 |
| Histogram gradient boosting | 0.01657 | 0.01245 | 0.01422 | 0.09482 |

No threshold reached both 50% precision and 50% recall for any candidate.
No model was selected, exported or installed in the demo. Final holdout
predictions were not computed. Historical results were not rewritten.

This remains exploratory within-patient development on repeatedly reused
recordings. The corrected archive only changes the audited development rows;
it is not a general replacement for subject-wide feature building. Clock
reconstruction elsewhere still has its documented assumptions. The correction
addresses a real label issue but does not explain or solve the poor prediction.
The next useful experiment is a training-recording-grouped validation study to
measure variability across seizure events before collecting more patient data
or expanding the parameter search.
