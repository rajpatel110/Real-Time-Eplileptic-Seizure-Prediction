# Approved configuration run

The user approved 22 distinct bipolar channels (154 features). This resolves the
channel/reference question: retain the existing bipolar reference, without
average referencing, while using the supplied 0.5-50 Hz band-pass and 60 Hz
notch. StandardScaler and balanced Logistic Regression are the only active model
pipeline. No dependency or other model was added.

Commands:

```bash
python scripts/build_user_notes.py
python -m eeg_project.experiment --config configs/user_notes_logistic.toml --output reports/user_notes_logistic.json
```

The builder requires the previous date-verified corrected archive and official
EDF files. It verifies source hashes, re-extracts features, carries forward the
boundary corrections and aligns rows by recording/start/end timestamps. It
asserts that all labels, identifiers, timestamps and feature names match the
previous corrected archive. The resulting archive has 5987 rows and 154 features.
The original archives are preserved. An existing completed output is refused;
a matching intermediate feature archive can be reused after an interrupted run.

The fitted development model uses C=0.01, balanced weights, max_iter=3000 and
seed 42, matching the preceding Logistic Regression comparison. On 1695
validation windows (241 positive), at threshold 0.5:

- Precision: 0.1148148148
- Recall: 0.5145228216
- F1: 0.1877365632
- Average precision: 0.1156827085
- Confusion matrix [[TN, FP], [FN, TP]]: [[498, 956], [117, 124]]

No threshold met both 50% precision and 50% recall. These are exploratory
within-patient development results, not final-test performance. The changed
filters did not resolve the poor precision. No new model was exported or
installed in the existing demo. Holdout features were rebuilt under the same
fixed preprocessing, but holdout predictions were not computed.

Validation: 21 existing tests and Ruff passed; rebuilding asserted exact label
and row membership preservation. Next work should remain within the agreed
Logistic Regression pipeline, examining recording-level variability and feature
quality rather than introducing other models.
