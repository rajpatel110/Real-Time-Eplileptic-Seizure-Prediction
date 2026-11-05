"""Calibrate and summarize experimental EEG amplitude on the Step 4 splits."""
import json

import numpy as np

from train_baseline import ROOT, SPLITS
from eeg_project.intensity import relative_amplitude, window_amplitude_uv


def main():
    with np.load(ROOT / "data/processed/chb01_evaluation_features.npz", allow_pickle=False) as data:
        amplitude = window_amplitude_uv(data["X"], data["feature_names"].tolist())
        recordings, labels = data["recording"], data["y"]
        configuration = json.loads(data["configuration"].item())
    masks = {key: np.isin(recordings, [f"chb01_{i:02d}.edf" for i in numbers])
             for key, numbers in SPLITS.items()}
    background = masks["train"] & (labels == 0)
    if not background.any():
        raise ValueError("Calibration requires training background windows")
    reference = float(np.median(amplitude[background]))
    ratio = relative_amplitude(amplitude, reference)
    report = {"measure": "Median channel RMS relative to training background",
              "reference_uv": reference, "reference_windows": int(background.sum()),
              "reference_recordings": sorted(set(recordings[background])),
              "configuration": configuration, "splits": {}}
    for split, mask in masks.items():
        report["splits"][split] = {}
        for label, name in ((0, "background"), (1, "preictal")):
            selected = ratio[mask & (labels == label)]
            if selected.size:
                report["splits"][split][name] = {
                    "windows": int(selected.size),
                    "ratio_p10_p50_p90": np.quantile(selected, [0.1, 0.5, 0.9]).tolist(),
                }
    output = json.dumps(report, indent=2) + "\n"
    (ROOT / "reports/chb01_intensity.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
