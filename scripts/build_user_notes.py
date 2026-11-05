"""Rebuild the approved filter profile while preserving verified split/label rows."""
import json
from pathlib import Path
import tomllib
import numpy as np
import requests
from build_features import main as build_features
from eeg_project.corrections import correct_rows
from eeg_project.dataset import BASE_URL, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/'data/processed/chb01_user_notes_v1.npz'
    intermediate = ROOT/'data/processed/chb01_user_notes_uncorrected.npz'
    if target.exists():
        raise ValueError('Output already exists; preserve prior artifacts and choose a new version')
    with np.load(ROOT/'data/processed/chb01_boundary_v1.npz', allow_pickle=False) as old:
        template = {key: old[key] for key in old.files}
    provenance = json.loads(template['correction_provenance'].item())
    if not provenance['date_offsets_match']:
        raise ValueError('Date-verified boundary corrections required')
    for name, digest in provenance['verified_edf_sha256'].items():
        if sha256_file(ROOT/'data/raw/chb01'/name) != digest:
            raise ValueError(f'Date-verified EDF changed: {name}')
    settings = tomllib.loads((ROOT/'configs/baseline.toml').read_text())
    prep = settings['preprocessing']
    if prep['profile'] != 'chat_and_sample_slides' or prep['reference_policy'] != 'original_bipolar':
        raise ValueError('Expected approved user-notes profile and bipolar reference')
    response = requests.get(BASE_URL+'SHA256SUMS.txt', timeout=60)
    response.raise_for_status()
    manifest = {name: digest for digest, name in map(str.split, response.text.splitlines())}
    names = ['chb01/chb01-summary.txt'] + ['chb01/'+name for name in sorted(set(template['recording']))]
    files = {name: manifest[name] for name in names}
    if not intermediate.exists():
        build_features(files, intermediate.name)
    with np.load(intermediate, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    config = tomllib.loads((ROOT/'configs/development.toml').read_text())
    development = set(config['splits']['train'] + config['splits']['validation'])
    stored = json.loads(data['configuration'].item())
    if stored['filters'] != prep['profiles'][prep['profile']] or stored['windowing'] != settings['windowing']:
        raise ValueError('Intermediate archive configuration differs; use a new output version')
    result = correct_rows(data, provenance['changes'], development)
    # Historical archive used split order; feature rebuilding uses filename order.
    keys = list(zip(result['recording'], result['start_seconds'], result['end_seconds']))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate row identities')
    lookup = {key: index for index, key in enumerate(keys)}
    order = [lookup[key] for key in zip(template['recording'], template['start_seconds'], template['end_seconds'])]
    if len(order) != len(keys):
        raise ValueError('Row membership changed')
    for key in ('X', 'y', 'recording', 'subject', 'start_seconds', 'end_seconds'):
        result[key] = result[key][order]
    for key in ('y', 'recording', 'subject', 'start_seconds', 'end_seconds', 'feature_names'):
        np.testing.assert_array_equal(result[key], template[key])
    if result['X'].shape[1] != 154:
        raise ValueError('Expected 22 distinct bipolar channels and 154 features')
    provenance = dict(provenance, feature_rebuild_source_sha256=files,
                      previous_corrected_archive_sha256=sha256_file(ROOT/'data/processed/chb01_boundary_v1.npz'),
                      scope='Approved filter rebuild; same corrected labels and split membership; no test scores')
    result['correction_provenance'] = np.array(json.dumps(provenance))
    with target.open('xb') as handle:
        np.savez_compressed(handle, **result)
    print(f'Saved {result["X"].shape} to {target.name}; labels and row identities preserved')


if __name__ == '__main__':
    main()
