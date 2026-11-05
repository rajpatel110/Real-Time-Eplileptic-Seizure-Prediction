"""Verify EDF dates and build a separately versioned development correction."""
from datetime import datetime
import json
from pathlib import Path
import tomllib
import numpy as np
from eeg_project.corrections import correct_rows
from eeg_project.dataset import sha256_file
from eeg_project.timeline import summary_timeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/processed/chb01_expanded.npz'
    target = ROOT/'data/processed/chb01_boundary_v1.npz'
    if target.exists():
        raise ValueError('Corrected archive already exists; preserve it and choose a new version')
    audit = json.loads((ROOT/'reports/boundary_audit.json').read_text())
    summary = ROOT/'data/raw/chb01/chb01-summary.txt'
    if sha256_file(source) != audit['archive_sha256'] or sha256_file(summary) != audit['summary_sha256']:
        raise ValueError('Inputs differ from audited hashes; rerun the audit')
    timeline = summary_timeline(summary.read_text())
    config = tomllib.loads((ROOT/'configs/development.toml').read_text())
    development = set(config['splits']['train'] + config['splits']['validation'])
    # These headers cover all development recordings and the neighboring event in 21.
    dates, hashes = {}, {}
    for name in sorted(development | {'chb01_21.edf'}):
        path = ROOT/'data/raw/chb01'/name
        with path.open('rb') as handle:
            header = handle.read(256)
        dates[name] = datetime.strptime(header[168:184].decode('ascii'), '%d.%m.%y%H.%M.%S')
        hashes[name] = sha256_file(path)
    anchor = min(dates, key=lambda name: timeline[name].start)
    for name, date in dates.items():
        if (date-dates[anchor]).total_seconds() != timeline[name].start-timeline[anchor].start:
            raise ValueError(f'EDF date disagrees with inferred timeline: {name}')
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    corrected = correct_rows(data, audit['changes'], development)
    for key in ('X', 'y', 'recording', 'subject', 'start_seconds', 'end_seconds'):
        old_test = np.isin(data['recording'], config['splits']['test'])
        new_test = np.isin(corrected['recording'], config['splits']['test'])
        np.testing.assert_array_equal(data[key][old_test], corrected[key][new_test])
    provenance = {'parent_archive_sha256': sha256_file(source), 'audit_sha256': sha256_file(ROOT/'reports/boundary_audit.json'),
                  'verified_edf_sha256': hashes, 'date_offsets_match': True,
                  'scope': 'Development corrections only; holdout rows unchanged', 'changes': audit['changes']}
    corrected['correction_provenance'] = np.array(json.dumps(provenance))
    with target.open('xb') as handle:
        np.savez_compressed(handle, **corrected)
    print(json.dumps({'target': target.name, 'before': len(data['y']), 'after': len(corrected['y']),
                      'positive_before': int(data['y'].sum()), 'positive_after': int(corrected['y'].sum()),
                      'date_offsets_match': True, 'holdout_rows_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
