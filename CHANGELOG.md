# Changelog

## Unreleased

- Added date-verified development boundary correction and fixed nonlinear model comparison.
  No candidate met the selection target; demo and fresh holdout predictions are unchanged.

- Added the MIT license with owner approval; dataset/third-party terms remain separate.

- Added a configuration-driven development-only runner with archive/config
  hashes, environment versions, Git revision and dirty-state reporting.
- Added reusable evaluation, threshold-selection and split-validation APIs.
- Extracted array-based causal filtering and added input/resource validation.
- Constrained dataset download paths and SHA256 input formats.
- Added synthetic regression tests and a dataset-free usage example.
- Added contribution/security guides, issue/PR templates, CI and Dependabot
  configurations. Hosted workflows have not run until these changes are pushed.
- Existing measured baseline results and the demo model remain unchanged.

## 2026-11-05 - Completed local baseline (0342d41)

- Verified CHB-MIT downloads, annotations, features and recording-based splits.
- Added Logistic Regression experiments and documented poor precision.
- Added offline Streamlit playback and relative EEG amplitude exploration.
