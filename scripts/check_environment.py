"""Run from any directory to check the Step 1 environment without EEG data."""

from importlib import import_module, metadata
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "eeg_project": "epileptic-eeg-project",
    "mne": "mne",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "streamlit": "streamlit",
}


def main() -> int:
    errors = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Project: {ROOT}")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        errors.append("Use Python 3.11 through 3.14.")
    if sys.prefix == sys.base_prefix:
        errors.append("Activate .venv or use .venv/Scripts/python.exe directly.")
    for module, distribution in PACKAGES.items():
        try:
            import_module(module)
            print(f"  OK {distribution}: {metadata.version(distribution)}")
        except Exception as exc:
            errors.append(f"{distribution}: {exc}")
    try:
        with (ROOT / "configs/baseline.toml").open("rb") as handle:
            config = tomllib.load(handle)
        profile = config["preprocessing"]["profile"]
        filters = config["preprocessing"]["profiles"][profile]
        print(f"Configuration: {profile} ({filters['low_hz']}-{filters['high_hz']} Hz)")
        for directory in ("data/raw", "data/interim", "data/processed", "models", "reports/figures", "tests", "app"):
            if not (ROOT / directory).is_dir():
                errors.append(f"Missing directory: {directory}")
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"Configuration: {exc}")
    if errors:
        print("\nSETUP CHECK FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nSETUP CHECK PASSED. No dataset loaded and no model trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
