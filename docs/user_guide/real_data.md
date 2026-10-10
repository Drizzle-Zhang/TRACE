# Connect your own data or authorized UKB data

This guide is for readers who have completed the synthetic demo in [getting started](getting_started.md) and now want to use their own research cohort. `synthetic_001`, `History`, `Lab`, and `D1` are fictional examples. Before running commands, enter the repository root in a Python environment with TRACE installed. The Risk prediction task also requires `python -m pip install -e '.[longitudinal]'`.

Start with [paths configured in YAML](#configure-paths-in-yaml). The environment-variable commands below provide an alternative for running stages separately; both use the same input contracts.

First identify what you have, then choose a workflow:

| Materials you have | Entry point | What happens next |
| --- | --- | --- |
| Numeric feature tables, selected feature lists, outcomes, and a complete cohort split | `pipelines/<task>/full_train.yaml` | Train modules, generate a score bundle, and train fusion and the planner |
| Precomputed module score bundles, outcome arrays, and a complete cohort split | `pipelines/<task>/train.yaml` | Start training at the fusion layer |
| New-cohort feature tables and a complete frozen model set | `pipelines/<task>/score.yaml`, then `infer.yaml` | Score and infer without labels |
| A new-cohort score bundle and frozen fusion/planning models | `pipelines/<task>/infer.yaml` | Run inference directly |

`<task>` is `prevalent` for disease present at baseline or `longitudinal` for Risk prediction (events after baseline). The same cohort can support both tasks, but their label definitions, score scales, and models are not interchangeable. See the [concepts guide](concepts.md) for the required background.

Here, starting from features means using **prepared numeric feature tables**. TRACE does not automatically download UKB data, perform ETL on arbitrary raw UKB fields, or select features. Researchers must first define outcomes and baseline dates, encode categorical variables, convert units, select features, and assign the primary split. Feature selection and other preprocessing learned from data must respect train/validation/test separation; do not select features using test labels.

Keep real participant data, participant IDs, individual predictions, and trajectories in the project's approved controlled environment, outside public repositories, examples, and attachments.

## Configure paths in YAML

Run these commands from the repository root with the TRACE environment activated. Copy the [configuration templates](../../configs/examples/) once:

```bash
mkdir -p data/configs
cp configs/examples/*.yaml data/configs/
```

Edit the copied files in `data/configs/`. Keep them in that directory so their inheritance paths remain valid. Paths can be absolute or relative; relative paths resolve from the YAML file that declares them. The templates expect inputs under `data/study/`, but you can change those paths. No `export` commands or `.env` file are needed for this workflow. Configurations, data and results under `data/` are ignored by Git.

| File to edit | What to configure |
| --- | --- |
| `catalog.yaml` | Training participants, numeric feature tables, outcome files, module/endpoint names and selected-feature lists |
| `new_catalog.yaml` | New participants and features, with the same module, endpoint and selected-feature order as training |
| `burden.yaml` | Relative acquisition cost for each module; replace the example `M1`/`M2` names with your module names |
| `<task>-train.yaml` | Catalog, train/validation/test split table, knowledge matrix and model/output paths |
| `<task>-predict.yaml` | New catalog and prediction output paths; it inherits trained model paths from `<task>-train.yaml` |
| `<task>-evaluate.yaml` | Test-evaluation output path; it reuses the training configuration |
| `<task>-consistency.yaml` | Labeled score cohort, matching frozen models, feature contributions and medical evidence resources |
| `<task>-new-consistency.yaml` | New-cohort contributions, score and result paths; it reuses the validation run's evidence, rules and scale |

Here `<task>` is `prevalent` or `longitudinal` (Risk prediction). Adapt the catalog's example names and align the knowledge matrix with its module and endpoint order. Keep the two tasks' model and output directories separate.

### Train

After preparing the training catalog and configuring its paths, check the input files and generate the task's label arrays:

```bash
trace prepare-data --config data/configs/prevalent-train.yaml
```

Then run the full training workflow:

```bash
trace run --config data/configs/prevalent-train.yaml
```

The full workflow repeats preparation, then trains the module bank, fusion model, probability calibration and planner. For Risk prediction, use `data/configs/longitudinal-train.yaml` in both commands; it fits Cox models without a separate prevalent calibrator. With the copied paths, module models are saved under `data/study/<task>/modules/`, and fusion/planner models under `data/study/<task>/models/`. Keep generated manifests with the models.

The three-row tables below illustrate file formats. Training needs enough eligible cases and controls at every fitting boundary; see [splits and sample requirements](#4-split-the-complete-cohort-then-understand-oof). Inspect each pair's `trained` or `skipped` status in `module_bank_manifest.json` before using the model bank. The default training workflow ends after training the planner; it does not run inference or test evaluation.

### Evaluate the held-out test split

After training, run the supplied evaluation template with the complete labeled cohort and original split table still in place:

```bash
trace run --config data/configs/prevalent-evaluate.yaml
```

This reuses the frozen models and prepared score bundle, writes predictions for the complete cohort, and calculates metrics only on test rows. Read `data/study/prevalent/cohort_evaluation/evaluation.json` for test metrics. For Risk prediction, run `data/configs/longitudinal-evaluate.yaml`; its results go to `data/study/longitudinal/cohort_evaluation/`. No training step is repeated. See [training from features](#6-train-from-features-for-the-complete-cohort-and-evaluate-test) for the equivalent separate commands.

### Predict

After training, point `new_catalog.yaml` at the new cohort and run:

```bash
trace run --config data/configs/prevalent-predict.yaml
```

For Risk prediction, use `data/configs/longitudinal-predict.yaml`. The command scores the new features and applies the frozen models without outcomes, split rows or retraining. With the copied paths, results are saved under `data/study/<task>/predictions/`, and the intermediate module score bundle under `data/study/<task>/new_scores/`.

Open `predictions.csv` for prevalent disease probabilities, or `predictions_<horizon>.csv` for Risk prediction event probabilities, with one file per available horizon. Every CSV has a `participant_id` column followed by all endpoint names, retains the input participant order, and leaves unavailable predictions empty. Read `agent_results.jsonl` for selected modules, trajectories and burden. The NumPy outputs remain available for programmatic analysis; Risk prediction `final_scores.npy` contains Cox log-risk, not percentages. See [how to read predictions](concepts.md#read-predictions) for probability meanings, masks and manifest paths, or [new-cohort preparation](#8-new-cohorts-without-labels-score-then-infer) for the full input requirements.

### Check consistency

Once the basic workflow succeeds, prepare the additional inputs described in [consistency setup](#9-add-consistency-inputs-after-the-basic-workflow-succeeds). Provide the complete labeled score cohort and split table, aligned contributions `(N,D,F)`, feature weights `(D,F)` and feature names `(F,)`. `N`, `D` and `F` denote participants, endpoints and features; their order must match. These attribution inputs are not generated by the training command. Evidence resources must match your endpoint and feature names; replace the bundled UKB evidence where necessary.

```bash
trace run --config data/configs/prevalent-consistency.yaml
```

For Risk prediction, use `data/configs/longitudinal-consistency.yaml`. These commands select rules on validation rows and evaluate on test rows. Open `consistency_predictions_baseline.csv` and `consistency_predictions_filtered.csv` to compare binary classifications before and after applying the rules: `1` is positive, `0` is negative, and empty cells are unavailable. Results also include consistency scores, frozen rules, NumPy classifications and `evaluation.json`. For new cohorts, [apply the frozen rules](#9-add-consistency-inputs-after-the-basic-workflow-succeeds); Risk prediction also reuses the frozen scale and horizon.

Append `--dry-run` to any `trace run` command to inspect configuration resolution without executing the pipeline. This does not check input files or model execution.

## 1. Prepare files and fix the participant order

The example below uses a private working directory. Replace `/secure/trace-study` with your absolute path:

```text
/secure/trace-study/
├── raw/
│   ├── participants.csv
│   ├── features/history.csv
│   ├── features/lab.csv
│   ├── outcomes/D1.csv
│   ├── selected_features/History_D1.csv
│   ├── selected_features/Lab_D1.csv
│   └── catalog.yaml
├── splits.csv
└── settings/
    ├── knowledge.csv
    ├── burden.yaml
    └── profile.yaml
```

`participants.csv` determines the row order of predictions. Participant IDs must be nonempty and unique. Save them as text to preserve identifiers such as `001`. Every feature table must contain exactly the same participants, but its rows may be in any order: TRACE aligns them by ID and rejects duplicate, missing or extra IDs. Participant, feature, outcome, and split tables support CSV, TSV/TAB, and Parquet. Parquet needs an available engine, such as PyArrow supplied by `.[io]`.

If your features are in one numeric table, declare the catalog and selected-feature lists, then generate the participant and module tables automatically:

```bash
trace prepare-inputs --config data/configs/prevalent-train.yaml --input-table /secure/trace-study/cohort.csv
```

Use the prediction configuration instead when preparing a new cohort, so its catalog receives the files. The input needs a `participant_id` column (or `--id-column <name>`) and exact feature column names. TRACE writes the tables at the catalog's declared paths, fills missing feature columns with NaN, and prints supplied/missing features and unused columns. Check that report before training or prediction. Repeating preparation replaces those generated tables. Outcome files, selected-feature lists and split assignments remain your inputs; preparation does not derive clinical encodings or units.

The following examples show file formats only; these few rows cannot train a real model:

`participants.csv`:

```csv
participant_id
synthetic_001
synthetic_002
synthetic_003
```

`features/history.csv`:

```csv
participant_id,age,smoking
synthetic_001,58,1
synthetic_002,62,
synthetic_003,,
```

`features/lab.csv` can use a different row order, for example:

```csv
participant_id,hba1c
synthetic_003,60
synthetic_001,45
synthetic_002,
```

All selected features must be convertible to numbers. Use empty cells or `NaN` for missing values; `inf` is not allowed. Rows with some missing features remain, and the model handles their NaNs. If all selected features for a module/disease pair are missing, that cell receives no prediction. Do not fill every missing value with 0, because 0 is often a valid measurement.

Outcome tables align by participant ID. They may have a different order from the participant table or cover only some participants. Participants without outcome records remain in the cohort with masked labels. Duplicate participant IDs or participant IDs outside the cohort raise errors. Split-table row indices are derived from participant IDs when omitted.

## 2. Declare inputs in a catalog

Save the following as `raw/catalog.yaml`. It defines two modules, one disease, and two pairs to train. This example uses the optional `${TRACE_RAW_INPUT_DIR}` placeholder; you can instead [put paths directly in YAML](#configure-paths-in-yaml):

```yaml
version: 1
participants:
  path: ${TRACE_RAW_INPUT_DIR}/participants.csv
  participant_id_column: participant_id
feature_tables:
  - id: history_table
    path: ${TRACE_RAW_INPUT_DIR}/features/history.csv
    participant_id_column: participant_id
  - id: lab_table
    path: ${TRACE_RAW_INPUT_DIR}/features/lab.csv
    participant_id_column: participant_id
modules:
  - name: History
    feature_table: history_table
  - name: Lab
    feature_table: lab_table
endpoints:
  - name: D1
    outcome_path: ${TRACE_RAW_INPUT_DIR}/outcomes/D1.csv
    participant_id_column: participant_id
    target_column: target_y
    time_column: BL2Target_yrs
pairs:
  - module: History
    endpoint: D1
    features_path: ${TRACE_RAW_INPUT_DIR}/selected_features/History_D1.csv
    feature_column: feature_code
  - module: Lab
    endpoint: D1
    features_path: ${TRACE_RAW_INPUT_DIR}/selected_features/Lab_D1.csv
    feature_column: feature_code
```

| Field | What to provide |
| --- | --- |
| `version` | The current version is `1` |
| `participants.path / participant_id_column` | Complete cohort participant file and its ID column |
| `feature_tables[].id / path / participant_id_column` | Internal feature table name, file path, and ID column |
| `modules[].name / feature_table` | Module name and the feature table `id` it references |
| `endpoints[].name / outcome_path` | Disease name and corresponding outcome file |
| `endpoints[].target_column / time_column` | 0/1 outcome column and time relative to baseline; defaults are shown above |
| `pairs[].module / endpoint` | Module and disease to train or score; unlisted pairs are not trained automatically |
| `pairs[].features_path / feature_column` | Selected-feature CSV for the pair and the column containing feature column names |

`History_D1.csv` is a **list of feature names**, not a participant matrix:

```csv
feature_code
age
smoking
```

For the laboratory table example above, `Lab_D1.csv` is:

```csv
feature_code
hba1c
```

The list must be nonempty and contain no duplicate or blank names. Every name must match a numeric column in the corresponding feature table. `001`, `1`, `1.0`, and `NA` can be distinct feature codes. The model stores the list's order, which must remain the same during later scoring. Module and disease order in the catalog also becomes the axis order in score bundles and models.

The catalog describes data only. Put training settings, output paths, and burden overrides in the pipeline/profile; do not add fields such as `output` to the catalog root. Relative paths resolve against the directory of the YAML file declaring them, and environment variables must already be set. See [module_catalog.example.yaml](../../configs/module_catalog.example.yaml) for the complete field template.

## 3. Check outcome times and the three masks

The raw outcome CSV's `BL2Target_yrs` is measured in **years** relative to the baseline defined by the researcher. Disease onset before baseline is negative; onset after baseline is positive. Participants without events have positive follow-up durations.

For example, `outcomes/D1.csv`:

```csv
participant_id,target_y,BL2Target_yrs
synthetic_001,1,-2
synthetic_002,0,8
synthetic_003,1,3
```

| `target_y` | Raw `BL2Target_yrs` | Prevalent training label | Risk prediction training label |
| --- | ---: | --- | --- |
| `1` | `-2` | Existing case, `y=1` | Pre-baseline case, excluded from event training for this disease |
| `1` | `3` | Future case, masked | `event=1, time=3, signed_time=3` |
| `0` | `8` | Control, `y=0` | `event=0, time=8, signed_time=-8` |
| `0` | Missing | Control, `y=0` | Unknown follow-up, masked |
| Missing | Any/missing | Masked | Masked |

Control follow-up times are positive in the raw CSV. For Risk prediction, censoring times become negative only in the generated `signed_time.npy`; keep these conventions distinct. Event time 0, nonpositive follow-up for a control with known follow-up in Risk prediction, infinite values, and nonnumeric values raise errors. A prevalent case with missing onset time is also masked because its pre-baseline status cannot be determined.

The three masks answer different questions:

| Name | Shape | Meaning |
| --- | --- | --- |
| `avail` | Participant × module | Does this participant have usable features for this module? |
| `pred_mask` | Participant × module × disease | Does this cell actually have a valid module prediction? |
| `y_mask` / `label_mask` | Participant × disease | Is the label known and eligible for the task during offline training/evaluation? |

For example, a participant has questionnaire data but no laboratory data: `avail=[true,false]`. If the questionnaire model scores successfully, their `D1` scores can be `[0.2,NaN]`, with `pred_mask=[true,false]`. If their D1 outcome is unknown, the label mask remains `false`, but inference is still possible. If a pair was not trained because of insufficient samples, its `pred_mask` is `false` even when features are available.

Scores with `pred_mask=false` must be NaN; scores with `true` must be finite and cannot be marked valid when `avail=false`. The planner also maintains an observed-module state for evidence currently in use; this is distinct from `avail`. Label masks do not participate in candidate selection or ADD/STOP decisions during inference.

## 4. Split the complete cohort, then understand OOF

Assign every participant to exactly one primary split and save it as `splits.csv`:

```csv
participant_id,split
synthetic_001,train
synthetic_002,validation
synthetic_003,test
```

The real file must cover all participants, and `train`, `validation` (also accepted as `val`), and `test` must all be nonempty. Rows may be unordered; TRACE derives row indices by matching participant IDs. If you supply an optional zero-based `row_index` column, it must agree with those IDs. TRACE does not choose a random primary split for you.

- `train`: fit modules, fusion, and the planner.
- `validation`: assess model performance and fit prevalent fusion probability calibration. Consistency thresholds, when needed, must also be selected here.
- `test`: perform the final evaluation after models are frozen.

OOF means that a participant's training-time prediction comes from a model not fitted on that participant. By default, shared random fivefold splits are generated **within train**: each fit trains on four folds and scores the remaining fold. The final module model fits on train and scores validation/test. These five folds do not change the primary split, and validation/test do not enter module fitting.

To reuse existing fivefold assignments within train, add this to a profile:

```yaml
training:
  modules:
    fold_table:
      path: ${TRACE_FOLD_TABLE}
      row_index_column: row_index
      participant_id_column: participant_id
      fold_column: iv_fold
```

`TRACE_FOLD_TABLE` points to a `participant_id,iv_fold` table covering exactly all train rows; `row_index` is optional. Fold numbers must be integers `1`–`5`, and every fold must be nonempty. Default module settings require at least 50 positives/events and a positive rate of at least `0.001` at each fitting boundary, plus the required class counts and internal calibration/holdout conditions. Having 50 cases across the entire cohort does not mean fivefold training is feasible.

By default, `full_train.yaml` skips pairs with insufficient samples and records the reason. The profile below changes this to `error` to expose pairs that cannot be fitted during your first data integration. Check `trained` / `skipped` statuses in `module_bank_manifest.json`; a completed command does not mean every pair was trained.

## 5. Provide the knowledge matrix and module burden

The knowledge matrix is a finite numeric module × disease array used for knowledge-guided candidate ranking and consistency calculations. It comes from your research knowledge or reviewed project assets; this workflow does not automatically build a knowledge graph from participant data. New modules or diseases need corresponding knowledge associations, with recorded provenance, version, and weight meanings. Start with a named CSV to avoid ordering errors in unnamed arrays.

Save the following as `settings/knowledge.csv`. Its first column contains module names, and later columns contain disease names:

```csv
module,D1
History,1.0
Lab,0.8
```

These weights illustrate the format only. The CSV loader aligns required modules/diseases by name. For `.npy`/`.npz`, explicitly store shape `(M,D)` in the same order as the catalog/score bundle. Shape alone cannot reveal transposition, especially for a square matrix.

Save the following as `settings/burden.yaml`:

```yaml
History: 5
Lab: 70
```

These are example relative acquisition burdens, not prices in Chinese yuan. The bundled [ukb_relative.yaml](../../resources/burden/ukb_relative.yaml) uses existing UKB module names and does not include custom `History`/`Lab` entries, so you must override it. If you have actual costs and durations, you can use:

```yaml
modules:
  History: {monetary: 0, time: 5}
  Lab: {monetary: 70, time: 10}
weights:
  monetary: 1.0
  time: 0.2
```

Combined burden is `monetary × weight + time × weight`. Researchers must choose the units and relative weights. Values and weights must be finite and nonnegative, with at least one positive weight. Every input module must have a burden entry.

Save the following as `settings/profile.yaml` so all commands use your burden file and an acquisition limit of two steps:

```yaml
artifacts:
  burden_profile: burden.yaml
planner:
  max_steps: 2
  initial_modules: []
training:
  modules:
    insufficient_data_policy: error
consistency:
  enabled: false
```

The profile merges into the selected pipeline and overrides matching fields. Setting an environment variable that no YAML field references does not change the configuration. `initial_modules: []` starts with no evidence. If you specify required initial modules, all participants must have them available or an error is raised. The planner can still STOP early, so `max_steps: 2` does not guarantee acquisition of two modules.

## 6. Train from features for the complete cohort and evaluate test

The two blocks below show the optional environment-variable approach for independent tasks. Paths follow this page's directory layout, and training and evaluation outputs are kept separately. For direct paths and a single training command, use the [YAML training workflow](#train).

### Prevalent

```bash
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_RAW_INPUT_DIR="$TRACE_DATA_ROOT/raw"
export TRACE_MODULE_CATALOG="$TRACE_RAW_INPUT_DIR/catalog.yaml"
export TRACE_MODULE_MODEL_DIR="$TRACE_DATA_ROOT/prevalent/modules"
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/prevalent/prepared"
export TRACE_INPUT_PREFIX=cohort
export TRACE_SPLIT_TABLE="$TRACE_DATA_ROOT/splits.csv"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/prevalent/artifacts"

trace run --config pipelines/prevalent/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml" --dry-run
trace prepare-data --config pipelines/prevalent/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
trace run --config pipelines/prevalent/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"

export TRACE_FUSION_MODEL="$TRACE_DATA_ROOT/prevalent/artifacts/fusion_model.joblib"
export TRACE_CALIBRATORS="$TRACE_DATA_ROOT/prevalent/artifacts/calibration_model.joblib"
export TRACE_Q_MODEL="$TRACE_DATA_ROOT/prevalent/artifacts/q_model.joblib"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/prevalent/runs/cohort_eval"

trace run --config pipelines/prevalent/evaluate.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
```

### Risk prediction

```bash
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_RAW_INPUT_DIR="$TRACE_DATA_ROOT/raw"
export TRACE_MODULE_CATALOG="$TRACE_RAW_INPUT_DIR/catalog.yaml"
export TRACE_MODULE_MODEL_DIR="$TRACE_DATA_ROOT/longitudinal/modules"
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/longitudinal/prepared"
export TRACE_INPUT_PREFIX=cohort
export TRACE_SPLIT_TABLE="$TRACE_DATA_ROOT/splits.csv"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/longitudinal/artifacts"

trace run --config pipelines/longitudinal/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml" --dry-run
trace prepare-data --config pipelines/longitudinal/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
trace run --config pipelines/longitudinal/full_train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"

export TRACE_FUSION_MODEL="$TRACE_DATA_ROOT/longitudinal/artifacts/fusion_model.joblib"
export TRACE_Q_MODEL="$TRACE_DATA_ROOT/longitudinal/artifacts/q_model.joblib"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/longitudinal/runs/cohort_eval"

trace run --config pipelines/longitudinal/evaluate.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
```

`--dry-run` prints the resolved YAML paths and settings without loading inputs or models. `prepare-data` checks features, outcomes and splits, and generates labels. `run` executes the full training pipeline, including train OOF scores and validation/test scoring. Prevalent training also fits fusion probability calibration on validation.

The evaluation pipeline runs `infer` for the **complete cohort**, then `evaluate` selects test rows from the split table to calculate metrics. Do not pass test-only prediction arrays alongside complete-cohort labels. Outputs are saved under the current `TRACE_OUTPUT_DIR`. Open `predictions.csv` for Prevalent or each `predictions_<horizon>.csv` for Risk prediction. Then read `agent_results.jsonl` for paths and burden, `evaluation.json` for test metrics, and `run_manifest.json` for configuration and output paths. Diseases with no valid evidence after STOP have empty CSV cells; evaluation uses both prediction and label masks.

Native module and fusion base scores for Risk prediction are **Cox log-risk**, not event probabilities. Native fusion models include baseline survival estimated during training, so an external `TRACE_ABSOLUTE_RISK` is not needed by default. The horizon CSVs and `horizon_risks.npz` hold probabilities for each available horizon. Risk prediction evaluation uses them to calculate AUROC/log-loss among participants with known outcomes; these are not a C-index or IPCW-adjusted metrics. Censoring before a horizon leaves an unknown outcome; censoring exactly at the horizon can count as negative at that horizon.

## 7. Start at fusion when you already have score bundles

If upstream scores were generated correctly, you do not need a catalog or `train-modules`. Every score bundle must contain:

| File (using the prefix `cohort`) | Shape/requirement |
| --- | --- |
| `cohort_participant_ids.npy` | `(N,)`, participant order |
| `cohort_modules.npy` | `(M,)`, module names |
| `cohort_diseases.npy` | `(D,)`, disease names |
| `cohort_oof_probs.npy` (prevalent) | `(N,M,D)`, probabilities; the train portion must be OOF |
| `cohort_scores.npy` (Risk prediction) | `(N,M,D)`, log-risk matching the model's convention; the train portion must be OOF |
| `cohort_pred_mask.npy` | `(N,M,D)`, Boolean prediction validity mask |
| `cohort_avail.npy` | `(N,M)`, Boolean module availability mask |

Training also needs labels in the same directory with the same prefix. Prevalent labels are `cohort_y.npy` and `cohort_y_mask.npy`; Risk prediction labels are `cohort_event.npy`, `cohort_time.npy`, `cohort_signed_time.npy`, and `cohort_label_mask.npy`. All have shape `(N,D)`. Observed events/labels are 0/1. For Risk prediction, `time` is positive, and `signed_time` is positive for events and negative for censoring.

The following independent training commands use the optional environment-variable approach and assume the knowledge matrix and profile from section 5 are ready. The existing bundle uses prefix `cohort`; the split table must cover its complete participant set. Row order is resolved from participant IDs, and optional explicit indices must agree:

```bash
# Prevalent
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/precomputed/prevalent"
export TRACE_INPUT_PREFIX=cohort
export TRACE_SPLIT_TABLE="$TRACE_DATA_ROOT/splits.csv"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/prevalent/artifacts"
trace run --config pipelines/prevalent/train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml" --dry-run
trace run --config pipelines/prevalent/train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"

# Risk prediction
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/precomputed/longitudinal"
export TRACE_INPUT_PREFIX=cohort
export TRACE_SPLIT_TABLE="$TRACE_DATA_ROOT/splits.csv"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/longitudinal/artifacts"
trace run --config pipelines/longitudinal/train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml" --dry-run
trace run --config pipelines/longitudinal/train.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
```

After training, run the relevant task's infer/evaluate commands from section 6, starting at `export TRACE_FUSION_MODEL=...`, while retaining this section's score input directory. Score bundle module names, disease names, and order must match the knowledge matrix, burden, and existing models. Prefer NumPy text arrays for names. Object name arrays from trusted legacy bundles require explicit `data.allow_legacy_pickle_names: true`; the default training templates do not enable this option.

Keep generated score and target manifests with the bundle; training checks them against its OOF and split records. Training requires `score_mode: training`; `inference` scores cannot replace train OOF. For external bundles without these manifests, verify label alignment and OOF origin before training.

For risk prediction, prepare scores in the scale required by the matched models and verify their endpoint alignment before training.

## 8. New cohorts without labels: score, then infer

Prepare the new cohort's participant table, features and catalog, or use `prepare-inputs` with the new-cohort configuration to generate tables from one numeric feature file. Separate feature tables align by participant ID automatically. The new catalog's module/disease order and every pair's selected features and feature order must match the frozen module bank. Participant IDs and feature values can be new. `endpoints` must still declare names, outcome paths, and column names, but outcome files **do not need to exist** in `score_mode: inference`.

For a single command with paths saved in YAML, use the [YAML prediction workflow](#predict). The optional environment-variable commands below assume new inputs are in `new_cohort/raw`, and their catalog also uses `${TRACE_RAW_INPUT_DIR}`. Keep the original training model bank and model files, and change only scoring and inference outputs:

```bash
# Prevalent
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_RAW_INPUT_DIR="$TRACE_DATA_ROOT/new_cohort/raw"
export TRACE_MODULE_CATALOG="$TRACE_RAW_INPUT_DIR/catalog.yaml"
export TRACE_MODULE_MODEL_DIR="$TRACE_DATA_ROOT/prevalent/modules"
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/new_cohort/prevalent/scores"
export TRACE_INPUT_PREFIX=new
export TRACE_FUSION_MODEL="$TRACE_DATA_ROOT/prevalent/artifacts/fusion_model.joblib"
export TRACE_CALIBRATORS="$TRACE_DATA_ROOT/prevalent/artifacts/calibration_model.joblib"
export TRACE_Q_MODEL="$TRACE_DATA_ROOT/prevalent/artifacts/q_model.joblib"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/new_cohort/prevalent/run"

trace score-modules --config pipelines/prevalent/score.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
trace infer --config pipelines/prevalent/infer.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
```

```bash
# Risk prediction
export TRACE_DATA_ROOT=/secure/trace-study
export TRACE_RAW_INPUT_DIR="$TRACE_DATA_ROOT/new_cohort/raw"
export TRACE_MODULE_CATALOG="$TRACE_RAW_INPUT_DIR/catalog.yaml"
export TRACE_MODULE_MODEL_DIR="$TRACE_DATA_ROOT/longitudinal/modules"
export TRACE_INPUT_DIR="$TRACE_DATA_ROOT/new_cohort/longitudinal/scores"
export TRACE_INPUT_PREFIX=new
export TRACE_FUSION_MODEL="$TRACE_DATA_ROOT/longitudinal/artifacts/fusion_model.joblib"
export TRACE_Q_MODEL="$TRACE_DATA_ROOT/longitudinal/artifacts/q_model.joblib"
export TRACE_KNOWLEDGE_MATRIX="$TRACE_DATA_ROOT/settings/knowledge.csv"
export TRACE_OUTPUT_DIR="$TRACE_DATA_ROOT/new_cohort/longitudinal/run"

trace score-modules --config pipelines/longitudinal/score.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
trace infer --config pipelines/longitudinal/infer.yaml \
  --profile "$TRACE_DATA_ROOT/settings/profile.yaml"
```

This workflow needs no split, outcomes, label arrays, or OOF cache, and performs no retraining. Without labels, you cannot run `evaluate`, which requires test labels, or reselect probability calibration or classification thresholds on the new cohort. If a new-cohort score bundle is already available, skip `score-modules`, point `TRACE_INPUT_DIR` and the prefix at that bundle, and run infer directly.

When moving models, retain the entire module bank and `module_bank_manifest.json`. For models trained with TRACE, keep fusion, prevalent calibration and Q-model `*.joblib` files with their adjacent `*.joblib.manifest.json` files. Use the matching task, module and endpoint order, model dependencies and TRACE version. A plain model exported by another training program needs a TRACE-compatible format before loading. See [model files and manifests](concepts.md#model-files-and-manifests).

## 9. Add consistency inputs after the basic workflow succeeds

This page's profile disables consistency checks, so feature contribution arrays are not required. [Prevalent full validation](../../pipelines/full_validation.yaml) and [Risk prediction full validation](../../pipelines/longitudinal/full_validation.yaml) additionally use contributions, feature weights/names and matching evidence. The pipelines select and freeze classification rules on validation rows automatically. Risk prediction validation also fits and freezes its consistency scale for the configured horizon. The current `contributions` shape is `(N,D,F)` (participant × disease × feature), `feature_weights` has shape `(D,F)`, and `feature_names` is a name array of length `F`. All three must use the same disease/feature order.

Supply feature contributions matching the current participant, disease and feature order. The training workflows and installation of `.[explain]` do not generate these arrays automatically.

To deploy rules selected by the [YAML consistency workflow](#check-consistency), set the new participants in `new_catalog.yaml` and new contributions in the supplied `<task>-new-consistency.yaml`, then run:

```bash
trace run --config data/configs/prevalent-new-consistency.yaml
```

For Risk prediction, use `data/configs/longitudinal-new-consistency.yaml`. Set the new-cohort score and result directories in that template if you changed the default layout. These templates inherit the validation configuration's evidence, feature weights/names, rules and frozen risk scale/horizon. If you changed validation output locations, update `consistency.rules_path` and, for Risk prediction, `consistency.longitudinal_scale_path` in the validation configuration to match. You do not need to recreate those settings in the deployment file.

Use contributions aligned with the new participants. Classification CSVs and `consistency_predictions.npz` go to the configured prediction directory. These pipelines score, infer and apply frozen rules without labels, split tables or refitting calibration.

For detailed checks of arrays, manifests, or model formats, see the [data contracts](../data_contracts.md). For environment variable, participant ID, axis order, insufficient sample, or model format errors, see [troubleshooting](troubleshooting.md).
