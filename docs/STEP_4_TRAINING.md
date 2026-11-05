# Step 4: Logistic Regression baseline

From the repository root with the virtual environment active, run
`python scripts/train_baseline.py`. It downloads five additional CHB01 recordings
and checks them against the official PhysioNet SHA-256 manifest. It saves the
expanded feature archive, the fitted scaler and classifier with feature names
and threshold, and the evaluation JSON under the ignored local output folders.

| Split | Recording numbers |
| --- | --- |
| Training | 01, 03, 04, 08 |
| Validation | 15 |
| Test | 18, 20 |

The scaler and balanced Logistic Regression fit on training windows only.
The threshold maximizes positive-class F1 on validation windows. The report
includes held-out precision, recall, F1, average precision, positive prevalence
and the confusion matrix. Scores are per window. Overlapping windows within
each recording are correlated. All recordings come from one subject and the
test set contains one seizure, so the results describe this within-subject
experiment. Event-level alarm performance and generalization to new subjects
require separate evaluation.
