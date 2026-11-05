"""Fixed regularization comparison on training-recording folds only."""
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
CANDIDATES = (0.001, 0.01, 0.1, 1.0)


def main():
    output = ROOT/'reports/training_regularization.json'
    if output.exists():
        raise ValueError('Report already exists; preserve previous results')
    path = ROOT/'configs/user_notes_logistic.toml'
    config = tomllib.loads(path.read_text())
    archive = (path.parent/config['archive']).resolve()
    with np.load(archive, allow_pickle=False) as data:
        mask = split_masks(data['recording'], data['subject'], config['splits'])['train']
        X, y, recording = data['X'][mask], data['y'][mask], data['recording'][mask]
        preprocessing = json.loads(data['configuration'].item())
    if config['model']['transform'] != 'raw':
        raise ValueError('Expected approved raw features with StandardScaler')
    candidates = []
    for strength in CANDIDATES:
        scores = np.full(len(y), np.nan)
        seen = np.zeros(len(y), dtype=int)
        folds = []
        for name in sorted(set(recording)):
            held = recording == name
            fit = ~held
            if len(np.unique(y[fit])) != 2:
                raise ValueError('Each fold training set must contain both classes')
            model = make_pipeline(StandardScaler(), LogisticRegression(C=strength,
                max_iter=config['model']['max_iter'], class_weight='balanced', random_state=config['seed']))
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(X[fit], y[fit])
            scores[held] = model.predict_proba(X[held])[:,1]
            seen[held] += 1
            folds.append({'held_recording': name, 'fit_recordings': sorted(set(recording[fit])),
                          'metrics': binary_metrics(y[held],scores[held],.5)})
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError('Each training window must receive exactly one finite out-of-fold score')
        candidates.append({'C':strength,'folds':folds,'pooled_metrics':binary_metrics(y,scores,.5)})
    selected = max(candidates, key=lambda row:(row['pooled_metrics']['f1'], -row['C']))
    report = {'scope':'Training-only hyperparameter selection; no validation or test predictions',
              'selection_rule':'Highest pooled out-of-fold F1 at fixed threshold 0.5; tie: smaller C',
              'configuration':config,'preprocessing':preprocessing,'candidates':candidates,'selected_C':selected['C'],
              'archive_sha256':sha256_file(archive),'config_sha256':sha256_file(path),
              'script_sha256':sha256_file(Path(__file__)),'python':platform.python_version(),
              'versions':{name:metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
              'limitations':'Selection scores are not unbiased final performance. Five folds, one patient, three seizure recordings; overlapping windows are correlated.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps({'selected_C':selected['C'],'candidates':[{'C':r['C'],**r['pooled_metrics']} for r in candidates]},indent=2))


if __name__ == '__main__':
    main()
