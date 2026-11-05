"""Bounded validation-selected comparison; preserve the original baseline."""
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_baseline import ROOT, SPLITS


def main():
    with np.load(ROOT / "data/processed/chb01_evaluation_features.npz", allow_pickle=False) as data:
        X, y, recording = data["X"], data["y"], data["recording"]
        names = data["feature_names"].tolist()
        configuration = json.loads(data["configuration"].item())
    masks = {key: np.isin(recording, [f"chb01_{i:02d}.edf" for i in ids]) for key, ids in SPLITS.items()}
    train, validation, test = (masks[key] for key in SPLITS)
    results = []
    models = []
    for weight in ("balanced", None):
        for strength in (0.001, 0.01, 0.1, 1.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=strength, class_weight=weight, max_iter=3000, random_state=42))
            model.fit(X[train], y[train])
            scores = model.predict_proba(X[validation])[:, 1]
            precision, recall, thresholds = precision_recall_curve(y[validation], scores)
            eligible = np.flatnonzero((precision[:-1] >= .5) & (recall[:-1] >= .5))
            chosen = int(eligible[np.argmax(recall[eligible])]) if eligible.size else None
            results.append({"C": strength, "class_weight": weight,
                            "validation_average_precision": float(average_precision_score(y[validation], scores)),
                            "meets_validation_target": chosen is not None,
                            "threshold": None if chosen is None else float(thresholds[chosen]),
                            "validation_precision": None if chosen is None else float(precision[chosen]),
                            "validation_recall": None if chosen is None else float(recall[chosen])})
            models.append(model)
    eligible = [i for i, row in enumerate(results) if row["meets_validation_target"]]
    report = {"selection_rule": "Validation precision >= 0.5 and recall >= 0.5; maximize recall then average precision",
              "candidates": results,
              "test_status": "Existing previously evaluated CHB01 recordings; exploratory follow-up"}
    if eligible:
        selected = max(eligible, key=lambda i: (results[i]["validation_recall"], results[i]["validation_average_precision"]))
        row, model = results[selected], models[selected]
        scores = model.predict_proba(X[test])[:, 1]
        prediction = scores >= row["threshold"]
        report["selected"] = row
        report["test"] = {"precision": float(precision_score(y[test], prediction, zero_division=0)),
                          "recall": float(recall_score(y[test], prediction)),
                          "average_precision": float(average_precision_score(y[test], scores)),
                          "tn_fp_fn_tp": confusion_matrix(y[test], prediction, labels=[0, 1]).ravel().tolist()}
        joblib.dump({"pipeline": model, "threshold": row["threshold"], "feature_names": names,
                     "configuration": configuration}, ROOT / "models/chb01_candidate.joblib")
    else:
        report["selected"] = None
        report["conclusion"] = "No candidate met the validation precision/recall target"
    (ROOT / "reports/precision_investigation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
