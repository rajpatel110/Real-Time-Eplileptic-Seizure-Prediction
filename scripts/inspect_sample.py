"""Verify and inspect the sample; save machine-readable metadata and EEG plots."""
import json
from pathlib import Path
from eeg_project.dataset import SAMPLE_FILES, parse_summary, sha256_file
from eeg_project.inspection import inspect_recording


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw"
    for relative, checksum in SAMPLE_FILES.items():
        path = raw_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {relative}. Run python scripts/download_sample.py")
        if sha256_file(path) != checksum:
            raise ValueError(f"Checksum mismatch: {relative}")
    annotations = parse_summary((raw_dir / "chb01/chb01-summary.txt").read_text())
    records = []
    for relative in SAMPLE_FILES:
        if not relative.endswith(".edf"):
            continue
        path = raw_dir / relative
        record = inspect_recording(path, annotations[path.name], root / "reports/figures" / f"{path.stem}_raw.png")
        records.append(record)
        print(f"{path.name}: {record['channel_count']} channels, {record['sampling_rate_hz']:g} Hz, "
              f"{record['duration_seconds']:g}s, {len(record['seizures'])} seizure(s)", flush=True)
        print(f"  Duplicate labels: {record['duplicate_labels_zero_based_indices']}", flush=True)
    report = {
        "dataset": "CHB-MIT 1.0.0",
        "source": "https://physionet.org/content/chbmit/1.0.0/",
        "purpose": "Step 2 raw-data inspection; not a training/evaluation dataset",
        "recordings_in_subject_summary": len(annotations),
        "records": records,
    }
    destination = root / "reports/sample_inspection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Inspection saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
