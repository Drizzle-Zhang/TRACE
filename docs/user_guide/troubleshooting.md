# Common problems and troubleshooting steps

Unless stated otherwise, run commands at the repository root with your Python environment activated. Follow [getting started](getting_started.md) for first-time installation and the [real data guide](real_data.md) to prepare your own files. The quoted TRACE error messages below come from the current code; paths, names, and numbers vary with the inputs.

Read the last few lines of the traceback to find the specific error and file path, then use this page. Preserve original inputs and existing artifacts while investigating. Use a new output directory for steps you rerun, check the results, and then decide how to replace old outputs.

## 1. Python, pip, trace, or trace_agent cannot be found

First identify the environments used by the current interpreter and pip:

```bash
python --version
python -c 'import sys; print(sys.executable)'
python -m pip --version
python -m trace_agent --help
```

The project requires Python 3.10 or later. If your system has only `python3`, use it to create the virtual environment described in getting started, then activate it and use its `python`.

For `No module named trace_agent`, install from the repository root in the same environment:

```bash
python -m pip install -e .
python -m trace_agent --help
```

`trace` is the command entry point created by installation. If `python -m trace_agent --help` works but `trace` reports `command not found`, check that the virtual environment is activated. You can also run the same subcommands directly through `python -m trace_agent`.

**Enter installation commands at the terminal's shell prompt.** If you see Python's `>>>`, enter `exit()` first. Pasting `python -m pip install -e .` after `>>>` causes a Python syntax error. Notebooks use different command conventions; use a terminal consistently when getting started.

## 2. Environment variables are unset, or errors persist after creating .env

Environment variables are optional. The [direct-path YAML workflow](real_data.md#configure-paths-in-yaml) needs no exports or `.env` file. This section applies when your chosen configuration contains environment-variable placeholders.

Common error:

```text
unresolved environment variable(s): TRACE_INPUT_DIR, ...
```

Placeholders such as `${TRACE_INPUT_DIR}` read the environment of the process that launches TRACE. The current configuration loader **does not load `.env` automatically**. Creating `.env`, or assigning a shell variable without exporting it, does not make it readable by a child process.

Set the required variables in the same terminal where you run the command, for example:

```bash
export TRACE_INPUT_DIR=/absolute/path/to/score_bundle
export TRACE_INPUT_PREFIX=test
export TRACE_OUTPUT_DIR=/absolute/path/to/a_new_output
python -c "import os; print('TRACE_INPUT_DIR exported:', 'TRACE_INPUT_DIR' in os.environ)"
```

Set the remaining variables required by your pipeline. Prevalent `infer.yaml` also requires `TRACE_FUSION_MODEL`, `TRACE_CALIBRATORS`, `TRACE_Q_MODEL`, and `TRACE_KNOWLEDGE_MATRIX`; the calibration artifact must match the model. Misspelled variable names also cause unresolved-variable errors.

The variables must exist in the process running the command. Check them again after changing terminals, Notebook kernels, or Python environments. You can also put paths directly in your own profile YAML and pass it with `--profile`.

## 3. LightGBM import fails, or the Risk prediction task lacks XGBoost

Check the actual imports rather than relying only on an installed-package list:

```bash
python -c 'import lightgbm; print(lightgbm.__version__)'
```

If the actual macOS error reports a missing `libomp.dylib` / OpenMP dynamic library, LightGBM's official Python package documentation explains that macOS needs OpenMP and provides the Homebrew installation command. See the [official LightGBM installation instructions](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst#install-from-pypi).

```bash
brew install libomp
python -c 'import lightgbm; print(lightgbm.__version__)'
```

This Homebrew command installs a system dependency; it does not replace `python -m pip install -e .`. If the error persists, keep the complete dynamic-library error and check the current Python and package installation environments. An import failure alone is not a reason to repeatedly reinstall all dependencies.

Native Risk prediction training may report:

```text
Cox module training requires trace-agent[longitudinal]
Cox fusion training requires trace-agent[longitudinal]; ...
```

The base installation does not include XGBoost. Install the corresponding extra for Risk prediction training or loading models that contain XGBoost:

```bash
python -m pip install -e '.[longitudinal]'
python -c 'import xgboost; print(xgboost.__version__)'
```

The quotes above also prevent zsh from treating square brackets as shell patterns. The basic prevalent workflow does not need this extra. `.[moe]` supports MoE Q-models that need PyTorch.

## 4. FileNotFoundError, or dry-run succeeds but execution fails

First check the actual path named in the traceback. Relative configuration paths resolve against **the directory of the YAML file that declares them**. Profiles and inherited configurations retain their own declaration directories. Paths are not necessarily relative to your terminal's current directory. Absolute paths in YAML can reduce ambiguity.

After editing your paths (or exporting variables for a placeholder-based configuration), inspect the resolved configuration. For the [copied YAML templates](real_data.md#configure-paths-in-yaml):

```bash
python -m trace_agent run \
  --config data/configs/prevalent-predict.yaml \
  --dry-run
```

`--dry-run` parses YAML, merges profiles, expands environment variables and paths, and prints the configuration. After checking the resolved paths, run the command without `--dry-run` to execute the configured pipeline.

Without a configuration, `python -m trace_agent infer --dry-run` prints only command-resolution information. An actual run without `--config` reports `infer requires --config. Use --dry-run to inspect command resolution.`

## 5. A joblib file is rejected, or a model manifest is missing or mismatched

Common error fragments:

```text
is not a TRACE model artifact; retrain or convert it to the versioned TRACE artifact format
training artifact manifest is missing or invalid:
training artifact manifest hash mismatch
expected component integration, got ...
expected task prevalent, got ...
```

Check in this order:

1. Check that the configuration points to the correct fusion, calibration, and planner files, and that the task is correct.
2. Check that the model uses the [supported TRACE format](../data_contracts.md#trace-model-artifacts) produced by the training commands or supplied in the matching trained-model package. A directly saved sklearn/LightGBM/XGBoost `joblib` file cannot replace it.
3. Keep fusion, calibration, and planner models produced by TRACE training with their matching adjacent `filename.joblib.manifest.json`. Module banks need `module_bank_manifest.json` and its referenced models; training-mode scoring also needs the OOF cache.
4. Check whether models from different training runs have been mixed. `upstream ... identity differs from the fitted model` means that the recorded upstream file differs from the configured file.

Restore missing files from the complete matching artifact package. If it is unavailable, rerun the relevant training step from the original inputs. Do not edit recorded hashes or remove manifests to bypass identity checks.

## 6. Participant ID, module, endpoint, or feature order does not match

Common error fragments and the first checks to make:

| Error | First check |
| --- | --- |
| `split-table participant ID mismatch at row_index=...` | The split table row's `participant_id` against `<prefix>_participant_ids.npy[row_index]` |
| `feature participant ID mismatch: ...` | Whether the feature table contains every catalog participant exactly once, without extra IDs; row order is aligned automatically |
| `split table must cover every score-bundle row exactly once` | Whether every participant has one split assignment; `row_index` is optional and derived from participant IDs when absent |
| `split table has empty required splits` | Whether train, validation/val, and test all contain rows |
| `fusion artifact module order does not match the input bundle` | The input module array against the fusion model's ordered module axis |
| `fusion artifact endpoint order does not match the input bundle` | The input endpoint array against the fusion model's ordered endpoint axis |
| `module bank selected feature mismatch` / `module scoring feature order mismatch` | The catalog's selected feature list against the saved training list and order |

Do not sort only name files while retaining the original matrices, or sort arrays and their participant IDs independently. Return to data preparation and regenerate the complete score bundle with one canonical participant ID axis and one module/endpoint order. Native feature and outcome tables align by participant ID automatically; feature tables must contain the exact participant set.

Inspect score bundle axes and shapes without modifying them. For example, first set the correct `TRACE_INPUT_DIR` and `TRACE_INPUT_PREFIX`:

```bash
python - <<'PY'
import os
from pathlib import Path
import numpy as np

root = Path(os.environ['TRACE_INPUT_DIR'])
prefix = os.environ['TRACE_INPUT_PREFIX']
for suffix in ('participant_ids', 'modules', 'diseases', 'pred_mask', 'avail'):
    values = np.load(root / f'{prefix}_{suffix}.npy', allow_pickle=False, mmap_mode='r')
    print(suffix, values.shape, values.dtype)
    if suffix in ('modules', 'diseases'):
        print(values[:10].tolist())
PY
```

If a name array reports `is a legacy pickled object array; set allow_legacy_pickle_names=true only for a trusted bundle`, prefer regenerating names as Unicode string arrays. Enable `data.allow_legacy_pickle_names: true` only for a legacy bundle whose provenance you trust; this cannot fix axis misalignment. Full validation workflows require name arrays that do not depend on pickle.

## 7. Example M1/D1 names do not match UKB burden or knowledge names

Common errors:

```text
burden profile is missing modules: ['M1']
knowledge matrix name mismatch; missing modules=..., missing endpoints=...
knowledge matrix shape must be ..., got ...
```

`M1` / `D1` in `configs/module_catalog.example.yaml` are placeholders. The bundled `resources/burden/ukb_relative.yaml` uses UKB module names. Copying both files unchanged does not produce a matching dataset.

Keep catalog names that match your data, and use a local profile to point `artifacts.burden_profile` to a burden file containing those modules. The knowledge matrix must use the same modules and endpoints. CSV knowledge matrices have name axes. For NPY/NPZ numeric matrices, you must ensure axis order yourself; equal shapes can still be misaligned.

## 8. NaN, missing scores, or mask errors

| Error | Meaning and response |
| --- | --- |
| `masked-out scores must be NaN` | A cell with `pred_mask=False` contains a number; use NaN for absent predictions when generating data |
| `valid module scores must be finite` | A cell with `pred_mask=True` contains NaN/inf; check that model outputs and mask generation agree |
| `pred_mask cannot mark an unavailable module as valid` | An unavailable module is marked as a valid prediction; check the provenance of avail and pred_mask |
| `selected features contain infinity: ...` | Raw features contain inf; check reading, conversion, and calculations |
| `initial_used_modules contains unavailable modules` | The initial selected-module mask includes unavailable modules; check it against the current cohort |

Raw features may contain NaN. If all selected features for a participant's pair are missing, that pair produces no valid score. Untrained module/endpoint pairs can also leave predictions missing. **Missing does not mean negative; do not replace every NaN with 0.** Distinguish missing inputs, untrained pairs, invalid predictions, and unknown outcomes, then regenerate according to the corresponding contract.

Read risk results together with `final_valid_mask.npy` or the validity mask in `horizon_risks.npz`. Reading the score array alone hides cells without valid predictions.

## 9. train-modules finishes but some pairs are skipped

The module bank manifest's `pairs` records `status: skipped` and `metadata.reason`. Typical reasons are:

```text
insufficient positive/event or class count
positive/event rate below configured threshold
internal survival split is not feasible
internal survival split needs event and censor in both parts
```

Defaults are `min_positive: 50`, `min_positive_rate: 0.001`, and `insufficient_data_policy: skip`. These checks use eligible train rows and apply to every outer-fold fitting set and the final training set. Exactly 50 positives in the full cohort do not mean each train fitting set has enough.

Internal prevalent calibration also needs enough samples in both classes; internal model selection for Risk prediction needs a feasible event/censoring split. First check split distributions, label masks, missing selected features, and the manifest's `eligible_train_count` to confirm correct data preparation. If you change thresholds, change the training configuration explicitly and retrain. Do not count skipped pairs as successful fits.

A successful return means the program completed the configured workflow, not that every module × endpoint model was trained. Check required coverage against the manifest's trained/skipped records.

## 10. An OOF cache is stale or training provenance does not match

Common error fragments:

```text
training requires OOF scores with score_mode=training, not inference scores
OOF ... mismatch or missing; rerun score-modules in training mode
OOF train row mismatch
OOF participant mismatch
module OOF train target mismatch; rerun train-modules and score-modules
prepared target identity/split contract mismatch
prepared target hash mismatch:
artifact hash mismatch:
```

Choose rebuilding steps based on what changed:

1. Only native score bundle provenance fields are outdated or missing, while original inputs, splits, the module bank, and the OOF cache are unchanged: set `module_data.score_mode: training` in the configuration and rerun `score-modules`.
2. Features, participants, splits, or training labels changed: rerun `prepare-data`, `train-modules`, and training-mode `score-modules` from the original aligned inputs, then retrain affected downstream models.
3. A prepared-target manifest is missing: rerun `prepare-data` from original inputs. If the directory has a native score manifest but no target manifest, the first error may be FileNotFoundError. Do not invent new hashes around manually edited labels.
4. Files are corrupted or mixed across runs: restore a complete matching set or rebuild affected steps.

Training mode reuses actual train OOF and has final module models score only validation/test rows. Do not rename final-model predictions on train rows as OOF, or change `score_mode` strings, hashes, or masks to bypass checks. See [what OOF means](concepts.md#oof-score-training-participants-without-seeing-their-labels).

## 11. Evaluation fails, or the metrics are unclear

Evaluation requires prediction identity axes in the same order as the complete score bundle, then selects `test` rows through the split table. Common identity errors include `evaluation predictions must align with the complete score bundle` and prediction participant ID / endpoint order mismatches. Do not put only a test-subset matrix at a prediction path that requires the complete cohort.

Risk prediction evaluation reads `horizon_risks.npz` or the configured `evaluation.horizon_predictions_path` and needs event times and censoring information. If files or probabilities are missing, first check whether inference generated absolute risks. Cox log-risk in `final_scores.npy` cannot replace horizon probabilities.

At a given horizon, participants censored earlier have unknown outcomes and are excluded. Cells with labels but no valid predictions also do not enter their corresponding metrics. The reported `participants` count is the test population, which may differ from each endpoint's actual evaluation sample size. Risk prediction reports use `known_outcomes_at_horizon (complete-case, not IPCW)`; these are not censoring-adjusted metrics or a survival C-index.

Complete tuning, fusion calibration, and consistency rule selection on train/validation, then evaluate test after freezing the method. Do not select rules using test metrics and present the same results as an independent test.

## Information to keep when further investigation is needed

Keep the executed subcommand, config/profile paths, full traceback, current Python path, relevant manifests, and output directories. Remove credentials and participant-level data before sharing with collaborators. Error messages, array shapes, configuration field names, and file identity records are often enough to investigate.

See the [data contracts](../data_contracts.md) for detailed input requirements and [core concepts](concepts.md) for terminology.
