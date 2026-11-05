"""Development-only test of signed-log features; fresh holdout stays unused."""
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eeg_project.transforms import SignedLogFeatures
from train_expanded import ROOT, SPLITS


def main():
    with np.load(ROOT / "data/processed/chb01_expanded.npz", allow_pickle=False) as data:
        recording = data["recording"]
        train = np.isin(recording, [f"chb01_{i:02d}.edf" for i in SPLITS["train"]])
        validation = np.isin(recording, [f"chb01_{i:02d}.edf" for i in SPLITS["validation"]])
        X_train, y_train = data["X"][train], data["y"][train]
        X_validation, y_validation = data["X"][validation], data["y"][validation]
        validation_recording = recording[validation]
        names = data["feature_names"].tolist()
        configuration = json.loads(data["configuration"].item())
    rows, models = [], []
    for transform in ("raw", "signed_log"):
        for strength in (.01, .1, 1):
            steps = [StandardScaler(), LogisticRegression(C=strength, class_weight="balanced", max_iter=3000, random_state=42)]
            if transform == "signed_log":
                steps.insert(0, SignedLogFeatures())
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_validation)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_validation, scores)
            eligible = np.flatnonzero((precision[:-1] >= .5) & (recall[:-1] >= .5))
            chosen = int(eligible[np.argmax(recall[eligible])]) if eligible.size else None
            per_recording = {}
            for recording_name in sorted(set(validation_recording)):
                mask = validation_recording == recording_name
                per_recording[recording_name] = {
                    "positives": int(y_validation[mask].sum()),
                    "mean_score": float(scores[mask].mean()),
                    "average_precision": float(average_precision_score(y_validation[mask], scores[mask])) if y_validation[mask].any() else None}
            rows.append({"features": transform, "C": strength,
                         "validation_ap": float(average_precision_score(y_validation, scores)),
                         "threshold": None if chosen is None else float(thresholds[chosen]),
                         "precision": None if chosen is None else float(precision[chosen]),
                         "recall": None if chosen is None else float(recall[chosen]),
                         "per_recording": per_recording})
            models.append(model)
    eligible = [i for i, row in enumerate(rows) if row["threshold"] is not None]
    selected = max(eligible, key=lambda i: (rows[i]["recall"], rows[i]["validation_ap"])) if eligible else None
    report = {"candidates": rows, "selected": None if selected is None else rows[selected],
              "scope": "Development only; holdout predictions not computed"}
    if selected is not None:
        target = ROOT / "models/chb01_log_candidate.joblib"
        model = models[selected]
        joblib.dump({"pipeline": model, "threshold": rows[selected]["threshold"], "feature_names": names,
                     "configuration": configuration}, target)
        np.testing.assert_allclose(joblib.load(target)["pipeline"].predict_proba(X_validation), model.predict_proba(X_validation))
    (ROOT / "reports/feature_investigation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
