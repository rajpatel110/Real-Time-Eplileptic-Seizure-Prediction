# Feature-scaling investigation

Run `python scripts/train_expanded.py` to create the expanded feature archive,
then `python scripts/investigate_features.py` to reproduce this experiment.

The experiment uses the existing expanded training and development recordings.
It does not compute predictions on the fresh holdout (CHB01 recordings 26 and
32). All scales are fitted on training windows only.

The alternative transforms each feature as sign(x) * log(1 + abs(x)/s), where
s is its median nonzero absolute training value (1 for an all-zero column).
Both alternatives then use StandardScaler and balanced Logistic Regression.

| Features | C | Development average precision |
| --- | --- | --- |
| Raw | 0.01 | 0.11132 |
| Raw | 0.1 | 0.11138 |
| Raw | 1 | 0.11693 |
| Signed log | 0.01 | 0.11499 |
| Signed log | 0.1 | 0.11494 |
| Signed log | 1 | 0.11305 |

Development positive prevalence is 237/1697 = 0.13966. None of the six
configurations has a threshold reaching both precision >= 0.5 and recall >=
0.5. No candidate was selected or installed in the demo.

Log compression alone did not resolve poor development ranking. These results
do not establish the cause of failure. The development set has been reused for
model selection; its scores are exploratory, not independent test results.
Per-recording scores are included in reports/feature_investigation.json.

Next: audit recording chronology and boundary labels, then compare a small,
predeclared set of nonlinear baselines on the same development split. Keep the
fresh holdout reserved until a candidate and evaluation procedure are fixed.
