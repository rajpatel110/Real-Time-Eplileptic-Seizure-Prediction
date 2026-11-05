"""Audit development archive labels against clock-derived cross-file context."""
import json
import tomllib
from pathlib import Path

import numpy as np

from eeg_project.dataset import SAMPLE_FILES, sha256_file
from eeg_project.features import window_label
from eeg_project.timeline import summary_timeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = ROOT / "data/raw/chb01/chb01-summary.txt"
    if sha256_file(summary) != SAMPLE_FILES["chb01/chb01-summary.txt"]:
        raise ValueError("Summary checksum mismatch")
    timeline = summary_timeline(summary.read_text())
    config = tomllib.loads((ROOT / "configs/development.toml").read_text())
    development = set(config["splits"]["train"] + config["splits"]["validation"])
    archive = ROOT / "data/processed/chb01_expanded.npz"
    events = [event for entry in timeline.values() for event in entry.events]
    changes, counts = [], {}
    with np.load(archive, allow_pickle=False) as data:
        settings = json.loads(data["configuration"].item())["windowing"]
        for name in sorted(development):
            entry = timeline[name]
            mask = data["recording"] == name
            counts[name] = {"existing_windows": int(mask.sum()), "changed": 0}
            for start, end, old in zip(data["start_seconds"][mask], data["end_seconds"][mask], data["y"][mask]):
                new = window_label(entry.start + start, entry.start + end, events,
                                   settings["preictal_seconds"], settings["postictal_seconds"])
                if new != old:
                    counts[name]["changed"] += 1
                    changes.append({"recording": name, "start_seconds": float(start), "end_seconds": float(end),
                                    "old_label": int(old), "proposed_label": new})
    report = {"scope": "Development labels only; no features or model predictions changed",
              "assumption": "Summary order is chronological, with less than 24 hours between recording starts. Dates must be independently verified before adopting labels.",
              "archive_sha256": sha256_file(archive), "summary_sha256": sha256_file(summary),
              "windowing": settings, "recordings": counts, "changes": changes}
    destination = ROOT / "reports/boundary_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"recordings": counts, "changes": changes}, indent=2))


if __name__ == "__main__":
    main()
