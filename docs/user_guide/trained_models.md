# Trained models

The packages support inference with the supplied fitted models. Each task has a shared module bank and fusion model; each priority has its own planner and frozen burden settings. Training inputs, participant tables, OOF caches and participant-specific feature contributions are not included.

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

Fill `data/inputs/<task>/participants.csv` and every CSV in `data/inputs/<task>/features/`. The headers contain the exact model input columns. Keep the same participant order in every file. Prepare values according to [input semantics](#input-semantics), including the original encodings, units and missing-value conventions. The inference catalog and selected-feature lists are already supplied. Outcomes are not read for prediction.

Follow the [environment installation guide](getting_started.md#2-install-the-runtime-environment-once), then run the chosen variant from the `TRACE` directory with the environment activated:

| Task | Configuration value | Priority | Command |
| --- | --- | --- | --- |
| Prevalent | `prevalent` | Performance | `trace run --config configs/trained_models/prevalent-performance.yaml` |
| Prevalent | `prevalent` | Balanced | `trace run --config configs/trained_models/prevalent-balanced.yaml` |
| Prevalent | `prevalent` | Cost | `trace run --config configs/trained_models/prevalent-cost.yaml` |
| Risk prediction | `longitudinal` | Performance | `trace run --config configs/trained_models/survival-performance.yaml` |
| Risk prediction | `longitudinal` | Balanced | `trace run --config configs/trained_models/survival-balanced.yaml` |
| Risk prediction | `longitudinal` | Cost | `trace run --config configs/trained_models/survival-cost.yaml` |

Outputs are in `data/predictions/<task>/<priority>/`. Read `final_scores.npy` with `final_valid_mask.npy`, and use `agent_results.jsonl` for each participant's selected modules, assessment trajectory and accumulated burden. For Risk prediction, `final_scores.npy` contains Cox log-risk; `horizon_risks.npz` contains event probabilities at the configured horizons.

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

The supplied catalog and selected-feature lists determine which columns each fitted model reads. Keep the downloaded template headers; display names are not replacement column identifiers. Some diagnosis, medication and procedure mappings name a grouping column rather than a numeric input value; its numeric representation must come from the matched preprocessing definition. Medical-evidence feature mappings describe evidence relationships, not model input encodings.

Examples recorded in the supplied schemas:

| Column | Meaning | Unit or encoding |
| --- | --- | --- |
| `p21003_i0` | Age when attending the assessment centre | Years; for example, `58` |
| `p31` | Sex | `0` = Female; `1` = Male |
| `p20116_i0` | Smoking status | `0` = Never; `1` = Previous; `2` = Current |
| `p48_i0` | Waist circumference | Centimetres; for example, `92` |

The supplied dictionaries are partial. Some template columns lack a matching field definition; some derived-field rules identify a source category without specifying the output encoding or unknown-value handling. An empty unit entry does not establish that a measurement is dimensionless. If an encoding, unit or transformation is unspecified, confirm it against the matching training preprocessing records or field dictionary before supplying values. Do not infer an indicator value from its column suffix alone.

### Missing values

Leave unavailable feature cells empty, or use `NaN`. Missing values remain missing. Zero is a valid measurement or category code for many fields, including Female in `p31`; it must not be substituted for unknown values. Finite values such as `-1` are not automatically recognised as missing, so handle dataset-specific missing codes according to the training preprocessing.

Write numeric values without unit strings: use `92` in a centimetre-valued field, not `92 cm`. Category text such as `Male`, space-only cells and infinity are not accepted as numeric features. Keep participant IDs as text, and preserve the participant file's row order in every feature table.

### Filling example

This excerpt uses fictional participants and two real schema columns:

```csv
eid,p21003_i0,p31
example_001,58,1
example_002,62,
```

The first row records age 58 years and the schema's code for Male. The second records age 62 years with unknown sex. Copy values into the corresponding columns of the full downloaded template, retaining its other columns; this excerpt is not a complete module input file. Use the same IDs and row order in `participants.csv` and all feature tables. See [file preparation](real_data.md#1-prepare-files-and-fix-the-participant-order) for the full table requirements.

## Compute consistency scores

The common archive supplies the aligned knowledge matrix, aggregate feature weights, feature-name axis and evidence schema. The bundled medical evidence has matched rules for 267 of 617 prevalent endpoints and 374 of 934 Risk prediction endpoints; unmatched endpoints use the documented neutral consistency value. The shared resource archive supplies medical evidence. Provide a separate cohort contribution array for each selected priority at `data/inputs/<task>/<priority>_contributions.npy` (for example, `data/inputs/prevalent/balanced_contributions.npy`). Order each array `(participants, endpoints, features)` according to the downloaded `feature_names.npy` and the prediction axes. Contributions must come from that priority's selected models and the modules used in each participant's final prediction; do not reuse one array across priorities unless those final predictions and attribution inputs are identical. TRACE training and these packages do not automatically produce them. Contributions for unavailable features must be NaN. Global importance codes outside the model schema are retained as reference weights; they do not add input modules or authorize substituting unrelated columns.

The bundled scoring rules include template-generated statements pending clinical review. Consistency scores do not establish clinical correctness; see the [resource scope and review status](../../resources/medical_evidence/v1/README.md#statement-provenance-and-scoring-limitations).

```bash
trace run --config configs/trained_models/prevalent-balanced-consistency.yaml
trace run --config configs/trained_models/survival-balanced-consistency.yaml
```

For Risk prediction, first fill `data/inputs/survival/validation_splits.csv` with `row_index,eid,split`. Use zero-based row indices and EIDs matching the score cohort; include the separate validation subset with `split` set to `validation`, and leave other rows out if desired; the command estimates the module-score scale from those rows only and saves it beside the selected planner. Later cohorts must reuse this scale: run `score-modules` and `consistency` separately with the same consistency configuration, without rerunning `calibrate-consistency-scale`.

Read `consistency_scores.npz` in `data/consistency/<task>/<priority>/`. These configurations compute scores; the model packages do not supply frozen classification rules for the six variants. Default thresholds are illustrative. To filter or adjust classifications, provide a labeled validation/test cohort and the required consistency inputs, select rules on validation data and evaluate them on test data using the [consistency workflow](real_data.md#check-consistency). Apply the resulting frozen, model-bound rules with `apply-consistency` for later cohorts.

## Known limitations and compatibility

- Fifteen baseline-cache columns for risk prediction were found to contain another endpoint's values. Corrective retraining of the affected fusion models has not been established.
- Risk prediction model checks produced unavailable fusion outputs for `E4_DM1` and `I9_HYPTENSHR`; validity masks mark these outputs invalid.
- The balanced/cost risk prediction validation results record cost/time settings that differ from the corresponding planner configurations. Those results cannot establish matched model-specific consistency policies.

The reference runtime is Linux CPU. Results can differ across operating systems. For Cox models, tiny relative-risk differences can cross saved fusion-tree thresholds and produce larger log-risk differences. Assessment trajectories also depend on TRACE's masks and planner settings.
