# Proposed contributor issues

These are suggestions, not published GitHub issues.

1. **Cross-recording annotation context.** Build a subject timeline that handles
   midnight and gaps. Apply pre/postictal exclusions across adjacent files.
   Acceptance: synthetic midnight, gap and boundary tests; compare changed
   window counts on CHB01; document assumptions without changing old results.
2. **Event-level alarm evaluation.** Implement an explicit warning horizon,
   occurrence period and refractory policy. Acceptance: toy-event tests for
   repeated alarms, missed events and gaps; report false alarms/hour and event
   sensitivity alongside window metrics; never tune on the final holdout.
3. **Patient-independent split configuration.** Generalize feature building to
   multiple subjects and reconcile montages explicitly. Acceptance: disjoint
   subject checks, documented channels and dataset manifest, synthetic tests;
   real patient-independent claims only after an actual measured run.
4. **Bounded streaming preprocessing.** Preserve SOS state across EDF chunks.
   Acceptance: chunked output matches full-recording output within declared
   tolerance, duplicate-channel checks remain valid, measured peak memory falls.
5. **Portable locked environments.** Validate dependency resolution on Windows
   and Linux and the supported Python range. Acceptance: clean-install logs,
   reviewed lock strategy, successful tests and audit; do not erase historical
   environment provenance or claim cross-platform bitwise model equality.
