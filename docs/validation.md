# Validate TRACE results

## Check installation with synthetic examples

Run both examples in the [first-run guide](user_guide/getting_started.md) and
inspect their [expected outputs](user_guide/demo_verification.md). The examples
exercise training and prediction without real participant data or development
dependencies. Optional consistency checks require the additional inputs in the
[real-data guide](user_guide/real_data.md).

## Evaluate models on your cohort

Keep training, validation and test participants separate. Use training rows for
estimation, validation rows for calibration and rule selection, and untouched
test rows for final evaluation. Retain the generated split, input, model and
resource identities with your results.

Read prediction scores together with their validity masks. Report usable label
coverage and skipped or unsupported endpoints alongside predictive metrics.
Horizon evaluation for risk prediction reports AUROC and log-loss on
known-outcome complete cases.

For a research evaluation, retain the study's cohort, outcomes, preprocessing,
method settings, comparisons and uncertainty analysis with the results.

## Dependencies

Install the pinned dependency versions using
[constraints/reproducible.txt](../constraints/reproducible.txt).

## Development test suite

For code changes, activate your environment in the TRACE directory and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -ra
```

The suite covers array alignment, prediction masks, training/validation/test
isolation, numerical reference cases, consistency rule selection and application,
and command-line workflows. Numerical tests use synthetic inputs and fixed
reference fixtures under `tests/fixtures/`.
