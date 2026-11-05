# Precision investigation

Run `python scripts/investigate_precision.py` after preparing the evaluation
features. Eight Logistic Regression configurations vary C across 0.001, 0.01,
0.1 and 1.0, with balanced or default class weights. All fitting uses the
original training recordings. Selection requires validation precision and
recall of at least 0.5, prioritizing recall and then average precision.

These targets are experimental screening criteria. None of the eight
configurations met them on chb01_15. Validation average precision ranged
from 0.1174 to 0.1279, below the validation positive fraction of 118/587
(approximately 0.201). Accordingly, no replacement was selected and this
experiment did not run another test-set evaluation.

This indicates poor transfer to the validation recording across the tested
regularization settings. The data currently contains only two training
seizures and one validation seizure. The next experiment should add seizure
events and use event/recording-level validation while reserving newly acquired
recordings for a final evaluation. Prior test recordings have already been
examined and must be identified as such in follow-up research.
