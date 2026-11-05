"""Verified CHB-MIT sample download and strict seizure-summary parsing."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import requests

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
# From the official dataset SHA256SUMS.txt, verified during Step 2.
SAMPLE_FILES = {
    "chb01/chb01-summary.txt": "77e86183845192d147c88a9bb4263c2b4a32e936c6236029770f86ca2ea023db",
    "chb01/chb01_01.edf": "92ec026f633dca94ee74c2e2b67cf9d58aa50da700b650a1696cf497dac073e3",
    "chb01/chb01_15.edf": "00157966600e41bec0754d189d4fb92b020b214cb7e559df1421c6dc29ee8eca",
}


@dataclass(frozen=True)
class Seizure:
    start: float
    end: float


@dataclass(frozen=True)
class RecordingAnnotation:
    filename: str
    seizures: tuple[Seizure, ...]


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_sample(raw_dir: Path, files=None) -> list[Path]:
    """Download only three approved sample files; never replace corrupt originals."""
    paths = []
    for relative, checksum in (SAMPLE_FILES if files is None else files).items():
        target = raw_dir / relative
        if target.exists():
            if sha256_file(target) != checksum:
                raise ValueError(f"Checksum mismatch in existing file: {target}. Move it aside before retrying.")
            print(f"Verified existing {relative}", flush=True)
            paths.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {relative}", flush=True)
        # A failed/incomplete transfer stays .part; it is never treated as EDF.
        with requests.get(BASE_URL + relative, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            size = 0
            last_report = 0
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                    if size - last_report >= 16 * 1024 * 1024:
                        print(f"  {size / 1024**2:.0f} MiB received", flush=True)
                        last_report = size
        if digest.hexdigest() != checksum:
            raise ValueError(f"Downloaded checksum mismatch: {relative}; retained {temporary.name} for inspection")
        # rename fails if a target appeared concurrently on Windows.
        temporary.rename(target)
        print(f"Verified {relative} ({size:,} bytes)", flush=True)
        paths.append(target)
    return paths


def parse_summary(text: str) -> dict[str, RecordingAnnotation]:
    """Parse zero/one/multiple numbered events, failing on incomplete annotations.

    Event times are seconds from recording start. Wall-clock strings such as
    24:44:29 are intentionally not interpreted as recording offsets.
    """
    result = {}
    sections = re.split(r"(?m)^File Name:\s*", text)[1:]
    if not sections:
        raise ValueError("No recording sections in seizure summary")
    for section in sections:
        lines = section.splitlines()
        filename = lines[0].strip()
        if not re.fullmatch(r"chb\d+_\d+[A-Za-z_]*\.edf", filename):
            raise ValueError(f"Invalid recording name: {filename}")
        if filename in result:
            raise ValueError(f"Duplicate recording: {filename}")
        counts = re.findall(r"(?m)^Number of Seizures in File:\s*(\d+)\s*$", section)
        if len(counts) != 1:
            raise ValueError(f"Missing or ambiguous seizure count: {filename}")
        events = []
        pending = None
        for line in lines[1:]:
            if not line.strip().startswith("Seizure "):
                continue
            match = re.fullmatch(r"Seizure(?: (\d+))? (Start|End) Time:\s*(\d+(?:\.\d+)?) seconds\s*", line.strip())
            if match is None:
                raise ValueError(f"Malformed event in {filename}: {line}")
            number, kind, value = match.groups()
            value = float(value)
            if kind == "Start":
                if pending is not None:
                    raise ValueError(f"Unpaired seizure start: {filename}")
                pending = (number, value)
            else:
                if pending is None or pending[0] != number or value <= pending[1]:
                    raise ValueError(f"Invalid event interval: {filename}")
                event = Seizure(pending[1], value)
                if events and event.start < events[-1].end:
                    raise ValueError(f"Overlapping or unordered seizures: {filename}")
                events.append(event)
                pending = None
        if pending is not None or len(events) != int(counts[0]):
            raise ValueError(f"Event count mismatch: {filename}")
        result[filename] = RecordingAnnotation(filename, tuple(events))
    return result


def read_edf_channel_labels(path: Path) -> list[str]:
    """Read original EDF labels so MNE duplicate-name suffixes stay traceable."""
    with path.open("rb") as handle:
        header = handle.read(256)
        if len(header) != 256 or header[:8].strip() != b"0":
            raise ValueError("Not a supported EDF header")
        count = int(header[252:256])
        if not 0 < count <= 4096:
            raise ValueError("Invalid EDF channel count")
        labels = handle.read(16 * count)
        if len(labels) != 16 * count:
            raise ValueError("Truncated EDF channel labels")
    return [labels[i * 16:(i + 1) * 16].decode("ascii").strip() for i in range(count)]
