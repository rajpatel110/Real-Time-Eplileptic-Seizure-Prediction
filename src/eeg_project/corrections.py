"""Apply reviewed row-level corrections without altering the original archive."""
import numpy as np


def correct_rows(data, changes, allowed_recordings):
    labels = data['y'].copy()
    keep = np.ones(len(labels), dtype=bool)
    seen = set()
    for change in changes:
        name, start, end = (change[key] for key in ('recording', 'start_seconds', 'end_seconds'))
        if name not in allowed_recordings:
            raise ValueError('Correction outside development split')
        key = (name, start, end)
        if key in seen:
            raise ValueError('Duplicate correction')
        seen.add(key)
        match = np.flatnonzero((data['recording'] == name) & (data['start_seconds'] == start) & (data['end_seconds'] == end))
        if len(match) != 1 or labels[match[0]] != change['old_label']:
            raise ValueError('Correction does not match exactly one original row')
        index = match[0]
        proposed = change['proposed_label']
        if proposed is None:
            keep[index] = False
        elif proposed in (0, 1):
            labels[index] = proposed
        else:
            raise ValueError('Invalid proposed label')
    result = dict(data)
    for key in ('X', 'y', 'recording', 'subject', 'start_seconds', 'end_seconds'):
        result[key] = (labels if key == 'y' else data[key])[keep]
    return result
