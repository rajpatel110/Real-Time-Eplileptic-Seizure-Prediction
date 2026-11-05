# Training-recording diagnostic

Run `python scripts/diagnose_training_recordings.py` after building the approved
user-notes archive. Existing reports are refused to preserve previous results.
The script uses only the five configured training recordings. Each fold fits
StandardScaler and balanced Logistic Regression on four recordings and predicts
the fifth. Settings are fixed at C=0.01, max_iter=3000, seed=42 and threshold=0.5.
There is no model/threshold search and no validation or final-test prediction.

| Left-out recording | Precision | Recall | Average precision | False-positive background windows |
| --- | --- | --- | --- | --- |
| 01 (no positives) | N/A | N/A | N/A | 0 / 719 |
| 03 | 8.31% | 28.81% | 0.11728 | 375 / 478 |
| 04 | 31.72% | 83.05% | 0.41010 | 211 / 466 |
| 08 (no positives) | 0% | N/A | N/A | 121 / 719 |
| 16 | 26.43% | 69.75% | 0.24090 | 231 / 467 |

The JSON metric utility uses zero for undefined thresholded metrics; this table
marks undefined quantities N/A. No-positive folds have null average precision.
False-positive counts refer to overlapping windows, not independent alarms.

Across 3204 out-of-fold windows, precision is 18.65%, recall 60.56%, F1 28.51%
and average precision 0.15468. The pooled confusion matrix is
[[1911, 938], [140, 215]]. These pooled scores come from five different fitted
models; they are not a new final-test score or directly comparable evidence of
improvement over the separate development set. Only three folds contain a
seizure event, and all belong to CHB01.

The diagnostic demonstrates recording-to-recording variability. It does not
prove which feature, artifact or physiological difference caused it. No model
was exported or installed in the app. The approved stack and preprocessing are
unchanged. A useful next step is to inspect feature distributions and quality
in training recordings before making another Logistic Regression adjustment.
