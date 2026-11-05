# Recording-boundary label audit

Run `python scripts/audit_boundaries.py` after building chb01_expanded.npz.
The script reads configs/development.toml for train/validation membership and
checks those windows only. It does not change the archive, retrain models or
compute holdout predictions. reports/boundary_audit.json records input hashes,
window settings and individual proposed label changes.

The summary-clock audit found 12 affected development windows:

- chb01_04: six background windows starting at 0, 5, 10, 15, 20 and 25 seconds
  overlap the preceding recording's 600-second postictal exclusion.
- chb01_20: two background windows starting at 2625 and 2630 seconds cross the
  preictal boundary for the seizure in recording 21. Four windows starting at
  2635, 2640, 2645 and 2650 seconds are wholly inside that preictal interval.

Thus the proposed correction excludes eight rows and changes four to positive.
These are overlapping windows, not twelve independent events. This small issue
is not evidence that the overall precision problem is solved.

The timeline uses summary order and minimum elapsed time consistent with the
clocks. It supports midnight and 24:xx:xx end times and preserves recording gaps.
It assumes chronological ordering and less than one day between starts. Summary
clocks alone cannot establish recording dates or rule out missing whole days.
Verify affected EDF dates, including recording 21, before adopting the proposed
labels. Durations and events are checked; conflicting overlaps are rejected.
Missing time is not filled with invented EEG samples or labeled windows.

Tests cover midnight rollover, gaps, cross-file exclusions, malformed clocks,
overlapping recordings and out-of-range seizure annotations. The existing
pipeline and historical model remain unchanged. Next: confirm affected dates,
produce a separately versioned corrected archive, and compare models on the
same fixed development split without consuming the fresh holdout.

Follow-up: affected EDF dates have now been checked and a separate corrected
archive evaluated. See [boundary comparison](BOUNDARY_COMPARISON.md).
