# Step 2: real EEG data inspection

## Reproduce (Git Bash, repository root)

```bash
source .venv/Scripts/activate
python scripts/download_sample.py
python scripts/inspect_sample.py
python -m unittest discover -s tests -v
```

The downloader retrieves approximately 81 MiB total from PhysioNet over HTTPS. It checks SHA-256 against the official CHB-MIT 1.0.0 manifest, reuses verified files, and refuses to overwrite a corrupt existing file. Interrupted or checksum-failed transfers never receive an `.edf` final filename. Retry a failed download by rerunning the command; an incomplete `.part` is restarted.

## Observed results

| Recording | Channels | Sampling rate | Duration | Samples per channel | Annotated seizures |
| --- | ---: | ---: | ---: | ---: | --- |
| chb01_01.edf | 23 | 256 Hz | 3600 s | 921600 | 0 |
| chb01_15.edf | 23 | 256 Hz | 3600 s | 921600 | 1, from 1732 to 1772 s |

Each EDF is 42,399,744 bytes. The official summary is 5,355 bytes and contains 42 recording entries. All three downloaded files match the official checksums pinned in `src/eeg_project/dataset.py`.

Both full recordings were scanned in one-minute chunks. All samples are finite, and no channel is constant over the entire recording. This is an integrity check, not an artifact-free or clinical-quality claim. MNE opens them with `preload=False` to avoid retaining the whole signal in memory.

## Channel finding that affects Step 3

The original header contains `T8-P8` twice, at zero-based indices 14 and 22. Their maximum absolute difference is zero across both entire recordings. MNE assigns unique names to duplicate labels; the JSON report retains original and loaded names. These are bipolar channel labels, so reference handling needs an explicit policy rather than automatically treating them as monopolar electrode signals.

Keep all 23 source channels for a clearly identified legacy reproduction if needed. Removing one repeated trace gives 22 channels and changes a seven-feature-per-channel representation from 161 to 154. Do not silently change the feature contract; decide and document it in Step 3.

## Outputs (local and ignored by Git)

- `data/raw/chb01/`: original EDF files and official seizure summary.
- `reports/sample_inspection.json`: checksums, timing, channel names, event intervals and integrity findings.
- `reports/figures/chb01_01_raw.png`: first 20 seconds, first four channels.
- `reports/figures/chb01_15_raw.png`: 1722-1742 seconds, first four channels, seizure onset marked at 1732 seconds.

The plots show raw microvolt traces and annotated seizure timing. No filtering, new reference, resampling, window labels or predictions have been applied.

## Evaluation boundary

This sample contains only one subject and one seizure event. It verifies data handling and the team's example timing. It cannot support independent positive examples in training, validation and testing. Expand the sample with additional seizure recordings before model evaluation; use distinct subjects for patient-independent claims. A seizure-free recording is not automatically evidence that every window satisfies a chosen inter-ictal exclusion policy.

## Sources

- Dataset: https://physionet.org/content/chbmit/1.0.0/
- Official checksum manifest: https://physionet.org/files/chbmit/1.0.0/SHA256SUMS.txt
- Official annotation summary: https://physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt
- Dataset citation: Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R
- MNE EDF reader: https://mne.tools/stable/generated/mne.io.read_raw_edf.html
