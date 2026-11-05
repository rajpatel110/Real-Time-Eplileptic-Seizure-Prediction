"""Expanded CHB01 experiment with a fresh final recording holdout."""
import json
import joblib
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from build_features import ROOT, main as build_features
from eeg_project.dataset import BASE_URL, download_sample

SPLITS = {"train": (1, 3, 4, 8, 16), "validation": (15, 18, 20), "test": (26, 32)}


def main():
    response = requests.get(BASE_URL + "SHA256SUMS.txt", timeout=60)
    response.raise_for_status()
    manifest = {name: digest for digest, name in map(str.split, response.text.splitlines())}
    selected = ["chb01/chb01-summary.txt"] + [f"chb01/chb01_{i:02d}.edf" for ids in SPLITS.values() for i in ids]
    files = {name: manifest[name] for name in selected}
    download_sample(ROOT / "data/raw", files)
    build_features(files, "chb01_expanded.npz")
    with np.load(ROOT / "data/processed/chb01_expanded.npz", allow_pickle=False) as data:
        X, y, recording = data["X"], data["y"], data["recording"]
        names = data["feature_names"].tolist()
        configuration = json.loads(data["configuration"].item())
    masks = {key: np.isin(recording, [f"chb01_{i:02d}.edf" for i in ids]) for key, ids in SPLITS.items()}
    train, validation, test = (masks[key] for key in SPLITS)
    assert np.all(sum(masks.values()) == 1)
    assert all(len(np.unique(y[mask])) == 2 for mask in masks.values())
    candidates, models = [], []
    for strength in (.001, .01, .1, 1):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=strength, class_weight="balanced", max_iter=3000, random_state=42))
        model.fit(X[train], y[train])
        scores = model.predict_proba(X[validation])[:, 1]
        precision, recall, thresholds = precision_recall_curve(y[validation], scores)
        valid = np.flatnonzero((precision[:-1] >= .5) & (recall[:-1] >= .5))
        index = int(valid[np.argmax(recall[valid])]) if len(valid) else None
        candidates.append({"C": strength, "validation_ap": float(average_precision_score(y[validation], scores)),
                           "threshold": None if index is None else float(thresholds[index]),
                           "validation_recall": None if index is None else float(recall[index]),
                           "validation_precision": None if index is None else float(precision[index])})
        models.append(model)
    report = {"splits": {key: {"recordings": sorted(set(recording[mask])), "windows": int(mask.sum()),
                              "positives": int(y[mask].sum())} for key, mask in masks.items()},
              "provenance": "Prior test recordings 18 and 20 now used for development; 26 and 32 are fresh holdout",
              "candidates": candidates, "selected": None}
    eligible = [i for i, row in enumerate(candidates) if row["threshold"] is not None]
    if eligible:
        index = max(eligible, key=lambda i: (candidates[i]["validation_recall"], candidates[i]["validation_ap"]))
        row, model = candidates[index], models[index]
        scores = model.predict_proba(X[test])[:, 1]
        prediction = scores >= row["threshold"]
        report["selected"] = row
        report["test"] = {"precision": float(precision_score(y[test], prediction, zero_division=0)),
                          "recall": float(recall_score(y[test], prediction)),
                          "average_precision": float(average_precision_score(y[test], scores)),
                          "tn_fp_fn_tp": confusion_matrix(y[test], prediction, labels=[0,1]).ravel().tolist()}
        joblib.dump({"pipeline": model, "threshold": row["threshold"], "feature_names": names,
                     "configuration": configuration}, ROOT / "models/chb01_expanded.joblib")
    else:
        report["conclusion"] = "No candidate meets validation precision >= 0.5 and recall >= 0.5. Holdout scores remain unevaluated."
    (ROOT / "reports/chb01_expanded.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
