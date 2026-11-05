# Repository audit - 2026-11-05

Scope: Python package, all scripts, Streamlit app, tests, TOML/dependencies,
documentation, Git status/remote/history and ignored data/model/report paths.
No applicable AGENTS.md or existing LICENSE was found. Baseline milestone
0342d41 was tested, pushed and remotely verified before this improvement work.

## Findings and implemented response

| Finding | Response |
| --- | --- |
| Shared preprocessing exists, but array filtering is embedded in EDF loading | Extracted filter_signal and tested causality and feature behavior |
| Input validation misses invalid rates/window settings and decoded size | Added actionable checks and pre-allocation sample ceiling |
| Custom downloader paths could escape raw directory | Reject unsafe names/checksums and resolved paths outside root |
| Hardcoded training scripts lack reusable experiment orchestration | Added TOML development runner without replacing historical scripts |
| Reports lack complete reproducibility context | New reports capture config, hashes, versions, Git revision/dirty state |
| Metrics/threshold selection duplicated | Added independently tested evaluation API and documented undefined cases |
| Recording splits protect overlapping samples, but patient claims unsupported | Explicit exhaustive/disjoint recording validation and optional patient guard |
| Tests initially focus on downloads/annotations | Added synthetic filtering, labels, amplitude, metrics, splits and runner tests |
| README assumes a prepared local environment | Added fresh setup, dataset-free example and actual execution sequence |
| No contribution/security/templates/automation | Added guides, templates, read-only SHA-pinned workflows and Dependabot |
| Model loading uses pickle-backed joblib | Fixed trusted local model only; documented executable-artifact trust boundary |
| No license found | Owner approved MIT; added LICENSE for original code |
| Dependency audit flagged local pip | Updated local pip from 26.1.2 to 26.3; rerun audit recorded separately |

## Remaining research and engineering gaps

The baseline has poor precision and does not support practical alarms. More
feature scaling did not fix it. All real-data work uses one patient. The old
holdout became development data; preserve the fresh holdout until the candidate
and protocol are fixed. Cross-file annotation context, chronological splitting,
filter transients, acquisition-specific notch choice, event-level alarms and
confidence intervals still need investigation. More overlapping windows do not
substitute for more independent seizure events.

Legacy scripts still contain fixed experiment settings; the new configurable
runner complements them and preserves historical reproducibility. Raw source
checksums are verified during feature building; the runner records archive
hashes, not an independently signed dataset manifest. Dependencies have a pinned
runtime snapshot but no portable hash lock. Dirty-state reports cannot recreate
uncommitted source. Full-recording processing still uses substantial memory.
Path containment is not a defense against a malicious concurrent local writer.
Hostile EDF isolation and multi-user deployment remain out of scope.

CI is prepared locally, not proven by a hosted run. Native browser file choice,
real EEG hardware, independent installation on another OS and community adoption
have not been demonstrated. Tests are evidence for the behavior they cover, not
medical validation or proof of security.

## Program readiness

This is an early research prototype with improving reusable engineering, not an
established research benchmark or widely adopted open-source dependency. There
is no verified evidence here of external adoption or ecosystem impact. Before a
strong application: publish reviewed improvements, obtain CI
and independent reproduction evidence, and show real contributor/user value.
The official Codex for Open Source page invites core maintainers or widely used
public projects and allows applicants to explain other ecosystem importance.
Acceptance cannot be inferred from adding templates or CI.
Source: https://developers.openai.com/community/codex-for-oss (checked 2026-11-05).
