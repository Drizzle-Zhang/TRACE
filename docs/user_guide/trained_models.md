# Trained models

Use the supplied fitted models to predict for new participants from numeric feature tables. Each task has a shared module bank and fusion model; each priority has its own planner and frozen burden settings. Basic prediction needs no outcome labels, split table or training step.

Choose `performance` to prioritize predictive performance, `cost` to prioritize lower monetary and time burden, or `balanced` for a trade-off. Both tasks offer all three priorities.

The prevalent package covers 25 modules and 617 endpoints. The risk prediction package covers 28 modules and 934 endpoints, including incident and death endpoints. GWAS, Metabolome and Proteome models are excluded. Only module/endpoint pairs listed in the supplied catalog are available, with their selected-feature order fixed.

## Download and predict

Download the source archive and `TRACE-v0.1.0-resources.tar.gz`. For each task you need, download its `TRACE-v0.1.0-<task>-common.tar.gz` and at least one `TRACE-v0.1.0-<task>-<priority>.tar.gz`. A public download location is not currently listed. For supplied archives, use their `SHA256SUMS` to check integrity. See [distribution contents](../distribution.md) for the package layout.

Extract the source, then extract the resource, common and priority archives **inside the resulting TRACE directory**. Their contents already use `data/trained_models/` and `data/inputs/`; no path overrides or environment exports are required.

If the model files are already present, skip downloading and extraction. For separately supplied archives, run these example commands from the `TRACE` directory, replacing `/path/to/` with their location:

```bash
tar -xzf /path/to/TRACE-v0.1.0-resources.tar.gz
tar -xzf /path/to/TRACE-v0.1.0-prevalent-common.tar.gz
tar -xzf /path/to/TRACE-v0.1.0-prevalent-balanced.tar.gz
```

Use the corresponding common and priority archives for Risk prediction or another priority. The extracted files use these locations, where `<task>` is `prevalent` or `survival` and `<priority>` is `performance`, `balanced` or `cost`:

| Location | Contents |
| --- | --- |
| `data/trained_models/<task>/common/` | `module_bank/`, `fusion_model.joblib`, `knowledge.npy` and shared task files |
| `data/trained_models/<task>/<priority>/` | `q_model.joblib`, `burden.yaml` and `MODEL_BUNDLE.json` |
| `data/trained_models/common/medical_evidence/v1/` | Medical evidence for optional consistency scores |
| `data/inputs/<task>/` | `participants.csv`, `features/*.csv` and `catalog.yaml` |

Keep the matching `knowledge.npy` and priority-specific `burden.yaml` with the models: the planner needs both even when consistency is disabled. Risk prediction also uses `baseline_cumhaz.npy` and `time_grid.npy` from its common directory to produce absolute risks. The separate medical-evidence directory is used by the optional consistency workflow below.

Prepare one numeric feature table with a unique, non-empty `participant_id` column and the feature columns you have. Use the model's exact column identifiers and follow [input semantics](#input-semantics) for encodings and units. After installing TRACE, generate all module inputs with:

```bash
trace prepare-inputs --config configs/trained_models/prevalent-balanced.yaml --input-table /path/to/cohort.csv
```

For Risk prediction, use `configs/trained_models/survival-balanced.yaml`. The command creates `participants.csv` and every required module table at the supplied catalog's paths, preserving your input row order. Missing feature columns become empty cells, so you do not need to build tables for unavailable modules. The printed report lists supplied and missing features and unused input columns; check it for misspelled feature names. CSV, TSV and Parquet inputs are accepted; use `--id-column <name>` if your ID column is not `participant_id`. Repeating this command replaces the generated participant and feature tables.

The catalog and selected-feature lists are supplied with the models; retain their module and endpoint order. Outcomes are not opened during preparation or prediction. If you already have separate module tables, you can use them directly: each must contain the complete participant set, and TRACE aligns rows by ID automatically.

All six supplied configurations start with `Baseline_characteristics`. Each participant must have at least one finite value among that module's selected features, such as a known age in `p21003_i0`; otherwise inference stops with `an initial module is unavailable for at least one participant`. Leave unknown values missing. A noninitial module whose selected feature cells are all missing is unavailable for that participant and is not selected by the planner.

Follow the [environment installation guide](getting_started.md#2-install-the-runtime-environment-once), then run the chosen variant from the `TRACE` directory with the environment activated:

| Task | Configuration value | Priority | Command |
| --- | --- | --- | --- |
| Prevalent | `prevalent` | Performance | `trace run --config configs/trained_models/prevalent-performance.yaml` |
| Prevalent | `prevalent` | Balanced | `trace run --config configs/trained_models/prevalent-balanced.yaml` |
| Prevalent | `prevalent` | Cost | `trace run --config configs/trained_models/prevalent-cost.yaml` |
| Risk prediction | `longitudinal` | Performance | `trace run --config configs/trained_models/survival-performance.yaml` |
| Risk prediction | `longitudinal` | Balanced | `trace run --config configs/trained_models/survival-balanced.yaml` |
| Risk prediction | `longitudinal` | Cost | `trace run --config configs/trained_models/survival-cost.yaml` |

These commands execute `score-modules` followed by `infer`. They write intermediate module scores to `data/scores/<task>/<priority>/` and final outputs to `data/predictions/<task>/<priority>/`. Repeating a command replaces the files at those locations; copy results you want to keep before using the same configuration for another cohort.

## Read the predictions

Open the generated CSVs in `data/predictions/<task>/<priority>/` directly in a spreadsheet application or text editor. For the balanced configurations:

| Output | How to read it |
| --- | --- |
| `data/predictions/prevalent/balanced/predictions.csv` | Prevalent disease probabilities for all 617 endpoints |
| `data/predictions/survival/balanced/predictions_<horizon>.csv` | Event probabilities for all 934 endpoints, with one file per available horizon from `1y` through `15y`, such as `predictions_10y.csv` |
| `agent_results.jsonl` | One JSON object per participant, with selected modules, trajectory, burden and validation checks |
| `run_manifest.json` | Run configuration, model provenance, axis order and generated CSV paths |

Every CSV has a `participant_id` column followed by all endpoint names. Participant rows follow `participants.csv`; empty cells mean unavailable predictions. Values are probabilities between 0 and 1: `0.12` in the prevalent CSV means a 12% estimated probability of disease already present, while `0.12` in `predictions_10y.csv` means a 12% estimated event probability within 10 years. These are probabilities, not binary classifications. For the same participant and endpoint, 10-year risk should be at least 5-year risk.

In `agent_results.jsonl`, `endpoints` contains up to 20 prioritized endpoints by default; the CSVs contain every endpoint. Use `selected_modules` and `trajectory` to read the assessment path. `accumulated_burden` is the sum of the selected modules' frozen normalized burden values, including the initial module; it is not a currency amount or elapsed minutes. `validation_checks` records software output checks; evaluating predictive accuracy requires observed outcomes.

The original NumPy outputs remain available for programmatic analysis. `final_scores.npy` stores prevalent probabilities or Risk prediction Cox log-risk; **Cox log-risk scores are not percentages**. `horizon_risks.npz` stores the risk probabilities with participant IDs, endpoint names and a validity mask. See [how to read predictions](concepts.md#read-predictions) for array masks, CSV manifest paths and JSON fields, or the [real data workflow](real_data.md) for evaluation.

## Input semantics

The supplied models expect prepared numeric features in the representation used during training. The input reader parses numbers; it does not convert measurement units, translate category labels or derive indicator columns from raw questionnaire answers.

### Field meanings and encodings

Use the matching `data/trained_models/<task>/common/schema.json`, where `<task>` is `prevalent` or `survival`. This dictionary is included in the task's common model package. Within each module, names, units and types align by position with `feature_codes`:

| Schema entry | What it records |
| --- | --- |
| `feature_codes` | Exact column identifiers used in the feature tables |
| `features` | Human-readable field meanings |
| `feature_units` | Recorded measurement units |
| `feature_types` | Original field types, such as integer or categorical |
| `feature_value_maps` | Category labels and recorded codes or grouping identifiers; numeric codes where provided |
| `feature_boolean_rules` | Source fields and categories for some derived indicator columns |

The supplied catalog and selected-feature lists determine which columns each fitted model reads. Use the exact feature codes in your input table; display names are not replacement column identifiers. Some diagnosis, medication and procedure mappings name a grouping column rather than a numeric input value; its numeric representation must come from the matched preprocessing definition. Medical-evidence feature mappings describe evidence relationships, not model input encodings.

Examples recorded in the supplied schemas:

| Column | Meaning | Unit or encoding |
| --- | --- | --- |
| `p21003_i0` | Age when attending the assessment centre | Years; for example, `58` |
| `p31` | Sex | `0` = Female; `1` = Male |
| `p20116_i0` | Smoking status | `0` = Never; `1` = Previous; `2` = Current |
| `p48_i0` | Waist circumference | Centimetres; for example, `92` |

Confirm each encoding, unit and transformation against the matching training preprocessing records or field dictionary before supplying values. Use the recorded preprocessing definition for derived indicators and unknown values.

### Missing values

Leave unavailable feature cells empty, or use `NaN`. Missing values remain missing. Zero is a valid measurement or category code for many fields, including Female in `p31`; it must not be substituted for unknown values. Finite values such as `-1` are not automatically recognised as missing, so handle dataset-specific missing codes according to the training preprocessing.

Write numeric values without unit strings: use `92` in a centimetre-valued field, not `92 cm`. Category text such as `Male`, space-only cells and infinity are not accepted as numeric features. Keep participant IDs as text; generated predictions follow the participant file's order.

### Filling example

This excerpt uses fictional participants and two real schema columns:

```csv
participant_id,p21003_i0,p31
synthetic_001,58,1
synthetic_002,62,
```

The first row records age 58 years and the schema's code for Male. The second records age 62 years with unknown sex. You can save this table as `cohort.csv` and pass it directly to `prepare-inputs`; all other model features will be missing. For your own cohort, add the prepared numeric feature columns you have.

## Compute consistency scores

The common archive supplies the aligned knowledge matrix, aggregate feature weights, feature-name axis and evidence schema. The shared resource archive supplies medical evidence. Prepare a separate cohort contribution array for each selected priority using your attribution workflow, and save it at `data/inputs/<task>/<priority>_contributions.npy` (for example, `data/inputs/prevalent/balanced_contributions.npy`). Order each array `(participants, endpoints, features)` according to the downloaded `feature_names.npy` and the prediction axes. Use contributions from that priority's selected models and the modules used in each participant's final prediction. Set contributions for unavailable features to NaN, and retain the supplied feature-name axis and aggregate weights.

```bash
trace run --config configs/trained_models/prevalent-balanced-consistency.yaml
trace run --config configs/trained_models/survival-balanced-consistency.yaml
```

For Risk prediction, first provide `data/inputs/survival/validation_splits.csv` with `participant_id,split`, marking the validation participants as `validation`. The file may contain just that subset or the complete split table; TRACE resolves row indices and estimates the module-score scale from validation rows only, saving it beside the selected planner. Later cohorts reuse that scale with a single command:

```bash
trace run --config configs/trained_models/survival-balanced-consistency-predict.yaml
```

Replace `balanced` with your chosen priority. This workflow scores the cohort and computes consistency without fitting the scale again.

Read `consistency_scores.npz` in `data/consistency/<task>/<priority>/`. To filter or adjust classifications, provide a labeled validation/test cohort and the required consistency inputs, select rules on validation data and evaluate them on test data using the [consistency workflow](real_data.md#check-consistency). Apply the resulting frozen, model-bound rules with `apply-consistency` for later cohorts.
