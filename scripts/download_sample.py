"""Download the two baseline recordings and official summary (about 81 MiB)."""
from pathlib import Path
from eeg_project.dataset import download_sample

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    download_sample(root / "data/raw")
