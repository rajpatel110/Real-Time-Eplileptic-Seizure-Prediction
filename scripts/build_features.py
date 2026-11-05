"""Build a local feature dataset from the verified CHB01 sample."""
import json
from pathlib import Path
import tomllib

from eeg_project.features import process
import numpy as np


from eeg_project.dataset import SAMPLE_FILES, parse_summary, sha256_file

ROOT = Path(__file__).resolve().parents[1]
def main(files=SAMPLE_FILES, output="chb01_features.npz"):
    config = tomllib.loads((ROOT / "configs/baseline.toml").read_text())
    raw_dir = ROOT / config["data"]["raw_directory"]
    for relative, checksum in files.items():
        if sha256_file(raw_dir / relative) != checksum:
            raise ValueError(f"Checksum mismatch: {relative}")
    summary = parse_summary((raw_dir / "chb01/chb01-summary.txt").read_text())
    prep = config["preprocessing"]
    profile = prep["profile"]
    filters = prep["profiles"][profile]
    settings = config["windowing"]
    features, targets, starts, recordings = [], [], [], []
    channels = None
    for relative in files:
        if not relative.endswith(".edf"):
            continue
        path = raw_dir / relative
        names, rows, labels, times = process(path, summary[path.name].seizures, filters, settings)
        if channels is not None and names != channels:
            raise ValueError("Recording channel order differs")
        channels = names
        features.extend(rows)
        targets.extend(labels)
        starts.extend(times)
        recordings.extend([path.name] * len(rows))
        print(f"{path.name}: {len(rows)} windows, {sum(labels)} pre-ictal")
    matrix = np.asarray(features)
    feature_names = [f"{channel}:{feature}" for channel in channels for feature in config["features"]["per_channel"]]
    if matrix.shape != (len(targets), len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid feature matrix")
    destination = ROOT / config["data"]["processed_directory"] / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, X=matrix, y=targets, start_seconds=starts,
                            end_seconds=np.asarray(starts) + settings["window_seconds"],
                            recording=recordings, subject=["chb01"] * len(targets),
                            feature_names=feature_names,
                            configuration=json.dumps({"profile": profile, "filters": filters,
                                                      "windowing": settings, "reference": "original_bipolar"}))
    temporary.replace(destination)
    print(f"Saved {matrix.shape} to {destination}")


if __name__ == "__main__":
    main()
