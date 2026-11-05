"""Train and evaluate a recording-separated CHB01 baseline."""
import json

import joblib
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import ROOT, main as build_features
from eeg_project.dataset import BASE_URL, SAMPLE_FILES, download_sample

SPLITS = {"train": (1, 3, 4, 8), "validation": (15,), "test": (18, 20)}


def run():
    response = requests.get(BASE_URL + "SHA256SUMS.txt", timeout=60)
    response.raise_for_status()
    official = {name: digest for digest, name in map(str.split, response.text.splitlines())}
    files = dict(SAMPLE_FILES)
    for numbers in SPLITS.values():
        for number in numbers:
            name = f"chb01/chb01_{number:02d}.edf"
            files[name] = official[name]
    download_sample(ROOT / "data/raw", files)
    archive = "chb01_evaluation_features.npz"
    build_features(files, archive)
    with np.load(ROOT / "data/processed" / archive, allow_pickle=False) as data:
        X, y, recording = data["X"], data["y"], data["recording"]
        feature_names = data["feature_names"].tolist()
        configuration = json.loads(data["configuration"].item())
    masks = {split: np.isin(recording, [f"chb01_{n:02d}.edf" for n in numbers])
             for split, numbers in SPLITS.items()}
    if not np.all(sum(masks.values()) == 1):
        raise ValueError("Each window must belong to exactly one split")
    for split, mask in masks.items():
        if len(np.unique(y[mask])) != 2:
            raise ValueError(f"{split} must contain both classes")
    train, validation, test = (masks[key] for key in SPLITS)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42))
    model.fit(X[train], y[train])
    precision, recall, thresholds = precision_recall_curve(
        y[validation], model.predict_proba(X[validation])[:, 1])
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    threshold = float(thresholds[np.argmax(f1[:-1])])
    scores = model.predict_proba(X[test])[:, 1]
    predicted = scores >= threshold
    report = {
        "scope": "Held-out recordings from CHB01 only",
        "splits": {key: {"recordings": sorted(set(recording[mask])),
                         "windows": int(mask.sum()), "positive": int(y[mask].sum())}
                   for key, mask in masks.items()},
        "threshold": threshold,
        "test_average_precision": average_precision_score(y[test], scores),
        "test_positive_fraction": float(y[test].mean()),
        "test_precision": precision_score(y[test], predicted, zero_division=0),
        "test_recall": recall_score(y[test], predicted),
        "test_f1": f1_score(y[test], predicted),
        "test_confusion_matrix_tn_fp_fn_tp": confusion_matrix(y[test], predicted, labels=[0, 1]).ravel().tolist(),
    }
    target = ROOT / "models/chb01_logistic.joblib"
    joblib.dump({"pipeline": model, "threshold": threshold, "feature_names": feature_names,
                 "configuration": configuration}, target)
    restored = joblib.load(target)
    np.testing.assert_allclose(restored["pipeline"].predict_proba(X[test])[:, 1], scores)
    (ROOT / "reports/chb01_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
