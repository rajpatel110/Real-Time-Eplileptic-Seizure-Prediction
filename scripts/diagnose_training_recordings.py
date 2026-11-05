"""Leave-one-recording-out diagnostic restricted to configured training rows."""
from importlib import metadata
import json
from pathlib import Path
import platform
import tomllib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_project.dataset import sha256_file
from eeg_project.evaluation import binary_metrics, split_masks

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'reports/training_recording_diagnostic.json'
    if output.exists():
        raise ValueError('Diagnostic report already exists; preserve prior results')
    config_path = ROOT/'configs/user_notes_logistic.toml'
    config = tomllib.loads(config_path.read_text())
    archive = (config_path.parent/config['archive']).resolve()
    with np.load(archive, allow_pickle=False) as data:
        masks = split_masks(data['recording'], data['subject'], config['splits'])
        train = masks['train']
        X, y, recording = data['X'][train], data['y'][train], data['recording'][train]
        preprocessing = json.loads(data['configuration'].item())
    if config['model']['transform'] != 'raw':
        raise ValueError('This diagnostic expects the approved StandardScaler-only pipeline')
    scores = np.full(len(y), np.nan)
    folds = []
    for name in sorted(set(recording)):
        held = recording == name
        fit = ~held
        if len(np.unique(y[fit])) != 2:
            raise ValueError(f'Fold training data lacks both classes: {name}')
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=config['model']['C'], max_iter=config['model']['max_iter'],
            class_weight='balanced', random_state=config['seed']))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(X[fit], y[fit])
        prediction = model.predict_proba(X[held])[:, 1]
        scores[held] = prediction
        metrics = binary_metrics(y[held], prediction, .5)
        tn, fp = metrics['confusion_matrix'][0]
        folds.append({'held_recording': name, 'fit_recordings': sorted(set(recording[fit])),
                      'fit_windows': int(fit.sum()), 'positive_windows': int(y[held].sum()),
                      'metrics_at_0_5': metrics, 'false_positive_window_fraction': fp/(tn+fp) if tn+fp else None,
                      'score_quantiles_10_50_90': np.quantile(prediction,[.1,.5,.9]).tolist()})
    if not np.isfinite(scores).all():
        raise ValueError('Every training row must receive one out-of-fold score')
    report = {'scope': 'Training recordings only; validation/test predictions not computed',
              'configuration': config, 'preprocessing': preprocessing,
              'archive_sha256': sha256_file(archive), 'config_sha256': sha256_file(config_path),
              'script_sha256': sha256_file(Path(__file__)), 'python': platform.python_version(),
              'versions': {name: metadata.version(name) for name in ('numpy','scikit-learn','scipy')},
              'folds': folds, 'pooled_out_of_fold_at_0_5': binary_metrics(y,scores,.5),
              'limitations': 'Five folds from one patient; overlapping windows are not independent events. Pooled scores come from different fitted models. No threshold tuning.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps({'folds':folds,'pooled':report['pooled_out_of_fold_at_0_5']},indent=2))


if __name__ == '__main__':
    main()
