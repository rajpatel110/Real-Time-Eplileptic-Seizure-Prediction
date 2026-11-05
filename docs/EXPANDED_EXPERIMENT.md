# Expanded CHB01 experiment

Run `python scripts/train_expanded.py` to reproduce this experiment.

Ten verified recordings produce 5,995 eligible windows and 154 features.
Training uses recordings 01, 03, 04, 08 and 16 (3,210 windows, 355 positives).
Validation uses 15, 18 and 20 (1,697 windows, 237 positives).
The fresh holdout uses 26 and 32 (1,088 windows, 118 positives).
These are recording-separated groups rather than a chronological evaluation.
Previously examined test recordings 18 and 20 are now explicitly development data.

Four balanced Logistic Regression models use C values 0.001, 0.01, 0.1 and 1.
Selection requires validation precision >= 0.5 and recall >= 0.5. None qualifies.
Validation average precision ranges from 0.1113 to 0.1282, compared with
positive prevalence 237/1697 (approximately 0.1397). Holdout performance
therefore remains unevaluated and the app retains its original baseline.

This expansion alone did not resolve poor validation performance. Further
work should investigate the feature representation and event-to-event
variation, with the fresh holdout kept separate from development. The
labeling policy still uses recording-local seizure annotations, and
cross-recording exclusion context remains a limitation.
