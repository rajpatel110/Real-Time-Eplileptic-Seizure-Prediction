# Reusable research APIs

`eeg_project.features.filter_signal(signal, rate, filters)` accepts a finite
channels-by-samples array in volts and returns causal filtered samples. State
starts at zero each call; call once per recording, not independently per chunk.
`extract_features(window, rate)` returns seven values per channel in original
channel order: mean, std, variance, and mean Welch PSD in delta/theta/alpha/beta.
It requires at least 256 samples and rate > 60 Hz. It does not integrate PSD.

`eeg_project.evaluation.binary_metrics(y, scores, threshold)` returns precision,
recall, F1, [[TN, FP], [FN, TP]], positive fraction and average precision.
Scores must be finite probabilities and labels binary. Average precision is
null for a single-class set; it is not trapezoidal PR-AUC. Undefined thresholded
precision/recall/F1 use zero. These are window metrics, not event metrics.
`select_threshold` uses development scores only, maximizing recall among
thresholds meeting explicit precision/recall targets; returns None if none do.
`split_masks` rejects overlapping, missing or unknown recording assignments;
patient_independent=True additionally rejects shared subjects.

Run `python examples/synthetic_pipeline.py` without any dataset. Its output is
synthetic and is not model performance evidence.

## Configured experiment

After creating the expanded archive with `python scripts/train_expanded.py`, run:

```bash
python -m eeg_project.experiment --config configs/development.toml --output reports/development_run.json
```

The legacy archive-building command also executes its historical training
experiment. The new runner itself never downloads data or scores test windows.
Archive paths resolve relative to the TOML file. Output paths resolve relative
to the working directory; existing outputs are refused. Use a new filename for
each run. The runner validates archive dimensions, timestamps, binary labels,
finite features and explicit recording splits, then fits transformations only
on training windows. Convergence warnings fail the run.

The report includes the complete experiment and preprocessing configuration,
seed, feature names, archive/config SHA256, package versions, Python/platform,
Git revision and dirty-state indicator. A dirty flag is not a source snapshot:
commit reviewed code before experiments intended for external reproduction.
Archive hashes identify the input but do not establish that it was labeled
correctly. Numerical equality across different BLAS/platforms is not promised.

A fixed 0.5-threshold development report is always provided. A selected-threshold
report appears only if the target is met. There is deliberately no final-test
flag: freeze the candidate and evaluation protocol before adding final testing.
