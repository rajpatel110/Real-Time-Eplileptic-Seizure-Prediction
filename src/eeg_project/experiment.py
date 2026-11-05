"""Run a development-only experiment from an existing verified feature archive."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import tomllib

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from eeg_project.dataset import sha256_file
from eeg_project.evaluation import binary_metrics, select_threshold, split_masks
from eeg_project.transforms import SignedLogFeatures


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected experiment schema_version = 1")
    archive = (config_path.parent / config["archive"]).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new report path to preserve previous runs")
    archive_digest = sha256_file(archive)
    with np.load(archive, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        recording, subject = data["recording"], data["subject"]
        names = data["feature_names"].tolist()
        preprocessing = json.loads(data["configuration"].item())
        starts, ends = data["start_seconds"], data["end_seconds"]
    if y.ndim != 1 or X.ndim != 2 or not X.shape[0] or X.shape != (len(y), len(names)) or len(set(names)) != len(names):
        raise ValueError("Feature matrix dimensions or feature names are invalid")
    if y.ndim != 1 or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite features and binary labels")
    if starts.shape != y.shape or ends.shape != y.shape or not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("Window timestamps must be finite valid intervals aligned with labels")
    masks = split_masks(recording, subject, config["splits"], config.get("patient_independent", False))
    if recording.shape != y.shape:
        raise ValueError("Recording identifiers must align with feature rows")
    for name in ("train", "validation"):
        if len(np.unique(y[masks[name]])) != 2:
            raise ValueError(f"{name} requires both classes")
    model_config = config["model"]
    if model_config["transform"] not in ("raw", "signed_log"):
        raise ValueError("transform must be raw or signed_log")
    algorithm = model_config.get("algorithm", "logistic_regression")
    if algorithm != "logistic_regression":
        raise ValueError("Project scope permits Logistic Regression only")
    estimator = LogisticRegression(C=model_config["C"], class_weight="balanced",
                                   max_iter=model_config["max_iter"], random_state=config["seed"])
    steps = [StandardScaler(), estimator]
    if model_config["transform"] == "signed_log":
        steps.insert(0, SignedLogFeatures())
    model = make_pipeline(*steps)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X[masks["train"]], y[masks["train"]])
    scores = model.predict_proba(X[masks["validation"]])[:, 1]
    target = config["selection"]
    threshold = select_threshold(y[masks["validation"]], scores, target["min_precision"], target["min_recall"])
    def git(*args):
        result = subprocess.run(["git", "-C", str(config_path.parent), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except OSError:
        revision, status = None, None
    report = {
        "schema_version": 1, "scope": "Development only; test scores not computed",
        "configuration": config, "preprocessing": preprocessing, "feature_names": names,
        "archive_sha256": archive_digest, "config_sha256": sha256_file(config_path),
        "git_revision": revision, "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "epileptic-eeg-project")},
        "splits": {key: {"windows": int(mask.sum()), "subjects": sorted(set(subject[mask]))} for key, mask in masks.items()},
        "selected_threshold": threshold,
        "validation_by_recording_at_0_5": {
            name: binary_metrics(y[masks["validation"]][recording[masks["validation"]] == name],
                                 scores[recording[masks["validation"]] == name], 0.5)
            for name in sorted(set(recording[masks["validation"]]))},
        "validation_at_fixed_0_5": binary_metrics(y[masks["validation"]], scores, 0.5),
        "validation_at_selected_threshold": None if threshold is None else binary_metrics(y[masks["validation"]], scores, threshold),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.config, args.output)
    except (ValueError, KeyError, OSError, ConvergenceWarning) as exc:
        parser.exit(2, f"Experiment failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
