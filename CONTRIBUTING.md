# Contributing

Start with the README setup and install `python -m pip install -e ".[dev]"`.
Run `python -m unittest discover -s tests -v`, `python -m ruff check .`,
`python examples/synthetic_pipeline.py` and `python -m pip check`.
These checks require no EEG download, account, or credentials.

Discuss large method changes in an issue first. Create a focused branch and PR;
explain the problem, final behavior, validation and known limitations. Keep the
source pipeline and existing demo working. Add synthetic regression tests for
labeling, preprocessing and evaluation changes. CI uses Python 3.11 and 3.14;
local verification currently uses Windows/Python 3.14.

Never submit patient data, EDFs, credentials, joblib/pickle models or generated
reports. Use synthetic reproductions. Do not load models submitted by strangers.
Record configurations, random seeds, data hashes and package versions for any
reported experiment. Keep training transformations inside sklearn pipelines.
Specify whether evaluation is within-patient or patient-independent. Group
whole recordings, and keep shared overlapping windows out of different splits.
Do not reuse test results for model selection or claim clinical validation.

Reviewers should check leakage, provenance, error handling, test independence,
and backwards compatibility. Model improvements need measured evidence, not
only code changes. See docs/CONTRIBUTOR_ISSUES.md for five scoped tasks.

Original project code is MIT licensed. Contributions should be compatible
with that license. CHB-MIT data and third-party material retain their own terms.
