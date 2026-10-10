# Data and artifact contracts

## Canonical axes

All core arrays use named, fixed axes:

- participant: `N`;
- module: `M`;
- endpoint/disease: `D`;
- attribution feature: `F`.

Module scores use `(N, M, D)`, availability uses `(N, M)`, and integrated
risks use `(N, D)`. Names are unique and are validated before any model call.

## Missingness

`pred_mask[n,m,d]` is authoritative for a module-endpoint prediction.
Masked-out score values must be `NaN`; valid values must be finite. A module
cannot be marked valid when it is unavailable.

The planner's observed-module mask is separate from availability: availability
states what can be acquired, while observed states what has already been used.

## Label-isolation boundary

Module selection and online planning never load or receive `y_mask`/label
masks. Candidate generation, candidate ranking and Q-state construction may
use only prediction-time inputs: current integrated risk, `pred_mask`, module
availability, already observed modules, knowledge links and burden.

Labels and their masks are confined to offline training rewards and evaluation.
Changing a test-set label mask therefore cannot change a selected module or an
inference trajectory.

## Task semantics

| Contract | Prevalent | Risk prediction |
| --- | --- | --- |
| module score | probability | Cox/log-risk score |
| integrated raw score | probability | log/relative-risk score |
| optional post-process | calibration | standardization + absolute risk |
| planning summary | 10-value logit summary | 8-value risk summary |
| reward | masked BCE reduction | multi-horizon masked BCE reduction |

## Training split table

Supervised training requires a user-supplied table with exactly one assignment
for every participant. Column names default to:

| Column | Meaning |
| --- | --- |
| `row_index` (optional) | zero-based row in every participant-axis NumPy array; derived from participant ID when omitted |
| `participant_id` | participant identifier |
| `split` | `train`, `validation`/`val`, or `test` |

Set `participant_id_column` in the catalog, split table or fold table configuration
when the identifier column uses another name. Participant IDs must be non-empty,
unique and cover the complete participant axis; input rows may be unordered.
All three splits must be non-empty. Without `row_index`, TRACE resolves indices
from `<prefix>_participant_ids.npy`. Explicit indices must be unique, integral, in range,
complete and match the corresponding participant IDs. Both forms yield the same canonical
split identity.

The split responsibilities are fixed:

- `train`: module, fusion and planner estimator fitting;
- `validation`: fusion metrics, prevalent calibration, and planner metrics;
- `test`: final frozen-model evaluation only.

There is no random main-split fallback. Test targets never enter fitting or
model selection. Preparation encodes the complete cohort; trainers slice the
approved rows before passing targets to fitting functions.

## Raw module inputs and cross-fitting

`configs/module_catalog.example.yaml` explicitly declares participant IDs,
feature tables, modules, endpoints and configured module/endpoint pairs. No
filename scanning or path-derived identities are used. Participant IDs must be
unique; every feature table must contain the exact full participant set. Tables
are aligned by participant ID to the participant file's order before model calls.
Missing features remain NaN without dropping any row. Selected-feature CSVs
must be non-empty, unique, and reference numeric columns; order is preserved
per pair. Endpoint CSVs align by participant ID; duplicate/unknown IDs error, absent
participants remain label-masked.

Feature codes are read as text: `001`, `1`, `1.0` and names such as `NA` remain
distinct identifiers. Empty or whitespace-only feature-list entries are errors.

`trace prepare-inputs --config <pipeline.yaml> --input-table <cohort.csv>`
materializes the catalog's participant and feature tables from one numeric
feature table. Selected-feature lists determine each table's columns; absent
columns become NaN. IDs are validated and numeric values checked before writing.
The command reports supplied/missing features and unused source columns, and
does not read outcomes, choose splits or transform units and categorical codes.

`target_y` is 0/1 (or missing); `BL2Target_yrs` is measured in years relative to
baseline. Prevalent cases have target 1 and negative time; target 0 controls
remain eligible even if time is missing. Future cases are excluded. Incident
training excludes preexisting cases and requires positive follow-up: event 1
uses positive signed time, censor 0 uses negative signed time. Event time zero,
nonpositive known control follow-up, infinity, and malformed values error.

The supplied main split remains unchanged. A shared shuffled fivefold split is
generated **within train only**, without labels, unless an explicit fold table
is supplied:

```yaml
training:
  modules:
    fold_table:
      path: ${TRACE_FOLD_TABLE} # train-only participant_id,iv_fold CSV; row_index is optional
      row_index_column: row_index
      participant_id_column: participant_id
      fold_column: iv_fold
```

Fold IDs must be integers 1–5, all non-empty; the table must cover precisely
the train rows and match their participant IDs. Each pair prechecks every outer complement
and the final fit before any estimator is fitted. `min_positive` (default 50)
and `min_positive_rate` (default 0.001) apply at each fit boundary. Insufficient
pairs are entirely skipped with a reason, or error under
`insufficient_data_policy: error`; other failures are never swallowed.

`prepare-data` with `module_data.input_mode: modules` writes task targets and a
hash-checked target manifest, without requiring existing score arrays.
`train-modules` produces versioned per-pair models, a train OOF NPZ and the bank
manifest (published last). The bank records all effective settings, feature
order, input hashes, train/fold fingerprints, optional fold-table source,
trained/skipped pair statuses and score scale.

`score-modules` requires an explicit `module_data.score_mode`:

- `training`: exact participant, split and feature-source identity; final models
  score validation/test only, then train entries are filled from verified OOF;
- `inference`: new participants and feature values are allowed; M/D and selected
  feature order must match, but labels, splits and OOF are not read.

Both modes publish the standard score arrays and a score-bundle manifest with
file hashes. The array loader verifies a manifest when present; legacy bundles
without one remain supported. Scoring never consults label masks, including
when generating train OOF for held-out rows with ineligible labels.

Native banks validate each model's recorded feature sources, split, folds,
effective settings, feature order and score scale against the bank manifest.
OOF loading checks declared coverage counts; training generation and scoring
also check the exact feature-available train rows for each pair. A missing
expected OOF prediction is an error, while an entirely missing feature row
remains a valid NaN/false cache entry.

## Training target arrays

Prevalent training requires `<prefix>_y.npy` and `<prefix>_y_mask.npy`.
Training for risk prediction requires `<prefix>_event.npy`, `<prefix>_time.npy`,
`<prefix>_signed_time.npy`, and `<prefix>_label_mask.npy`. Every array must be
`(N, D)` and aligned to the score bundle. Observed labels/events must be binary;
observed times must be finite and strictly positive. For every observed entry,
`signed_time` must equal `time` when `event == 1` and `-time` when `event == 0`.
Contradictory censoring codes or durations raise errors before fitting;
unobserved entries may retain their missing-value placeholders.

For native bundles, all downstream training and evaluation loaders verify the
prepared-target manifest's task, ordered participant fingerprint, module and
endpoint axes, split identity, filenames and file hashes. A native score bundle
without the prepared-target manifest is rejected. A native target bundle cannot
silently use a missing score manifest either. Training additionally requires
`score_mode: training` and matching split, training-row and training-participant fingerprints
in the score manifest; inference-mode predictions cannot replace OOF scores.
Evaluation accepts frozen inference-mode scores but still checks target identity.
The score-mode `prepare-data` command accepts those frozen scores too, allowing
the full validation pipeline to prepare/evaluate them without fitting. This
does not relax the OOF requirement of any training command.
Scoring and inference remain label-free and do not perform target-file checks.

Externally prepared bundles with neither manifest use the array interface and
shape, value and row identity validation. Confirm their label identity and OOF origin
against the upstream preparation records. Keep native manifests with their
bundles. Native score manifests missing
OOF fingerprints require rerunning `score-modules` with `score_mode: training`.

## TRACE model artifacts

Module, integration, calibration, and planner models are serialized inside a versioned
`TraceModelArtifact` envelope. The envelope records task, component, exact
module and endpoint order, training settings, validation metrics, framework
identity, and split/input provenance. Loading rejects raw `joblib` objects and
artifacts for the wrong task or component. Runtime construction also checks
the envelope axes against the input score bundle.

Training records the command, package version and effective settings, including
default values. Deterministic calibration records no random seed. New fusion,
calibration and planner artifacts declare `training.manifest_required: true`;
their adjacent `*.joblib.manifest.json` is required and its model hash,
component and task must match. Keep both files together when moving models.
Older envelopes without the marker remain loadable. Module models continue to
use the whole-bank manifest.

The loader rejects incomplete or mismatched model/manifest pairs. Restore a
complete matching pair or rerun training if either file is missing or damaged.

Every file artifact used by inference is recorded with size and SHA-256 in the
run manifest.

Medical-evidence weights may be supplied as an aligned array or compiled at
runtime from `resources/medical_evidence/v1` using an explicit feature-name
array. This keeps evidence-to-feature alignment name-based rather than
position-based.

Native fusion artifacts for risk prediction contain their fitted baseline-survival
arrays, which the default inference template for risk prediction uses without an
external file. `configs/profiles/absolute_risk.example.yaml` explicitly enables
an external NPZ through `TRACE_ABSOLUTE_RISK`; the environment variable alone is
not an override. For a separately supplied risk prediction model, absolute risk also
accepts either a compact NPZ containing `endpoints` and
`baseline_survival_<horizon>` arrays, or an explicit cumulative-hazard grid:

```yaml
artifacts:
  absolute_risk:
    baseline_cumhaz: /path/to/baseline_cumhaz_grid.npy
    time_grid: /path/to/curve_times_years.npy
    endpoints_file: /path/to/diseases.npy
    horizons: [1, 5, 10]
```

## Threshold provenance

Calibration and consistency/refinement thresholds may only be selected on the
validation split. Inference and frozen-rule application never inspect labels.
Test evaluation uses labels only for metrics and does not select thresholds.

Prevalent `calibrate-consistency` fits one probability cutoff and filtering rule
per endpoint from validation rows. Its default strategy grid is 0 to 1 inclusive
in increments of .001. The strategies are KG × feature, evidence, and
KG × feature × evidence; baseline is retained unless validation F1 improves.
`apply-consistency` reads only inference artifacts and the frozen JSON rules,
validates task/horizon/endpoint/model identities, and saves boolean baseline
and filtered arrays with their validity mask. `evaluate` verifies the input
and rule hashes before reporting baseline/filtered test precision, recall, F1
and confusion counts. It never selects a rule from test labels.

Selection for risk prediction defaults to the source posthoc search: hard KG/medical,
joint normalized gates, soft score adjustments, disease gates, finalists and
per-endpoint selectors. Settings `threshold_points`, `joint_grid_points` and
`max_finalists` default to 41, 9 and 120. Its rules use schema
`trace.survival_consistency_policies.v1`; adjusted policies may add positives
outside baseline. Application reuses validation q99/cutoffs/gates in every
participant chunk; evaluation reports source finalist/selector counts on test
only. `longitudinal_strategy_search: product_filters` explicitly selects the
earlier native three-product behavior. Previously saved three-product JSON
rules remain applicable through their original schema.

Evidence compilation supports all three source tables, disease aliases and
feature schema names (`feature_schema_path` or inline `feature_code_to_name`).
Raw `feature_values_path` must have shape `(N,F)` in the participant order and
`feature_names` order. Prevalent evidence threshold diagnostics default on;
those for risk prediction default off. `enable_threshold_checks` overrides this. Diagnostic CSV/JSON
outputs never alter risk/filter scores. Missing raw inputs remain explicitly
`missing_feature`. Evidence sources, schema and rank settings bind frozen-rule
identity; participant-specific raw values and contributions may change between
cohorts.

Consistency for risk prediction additionally requires `consistency.horizon` and a
frozen `longitudinal_consistency_scale.json` (or `longitudinal_scale_path`).
`calibrate-consistency-scale` reads validation module scores and the split
table, without loading labels. Native log-risk is transformed as
`exp(min(log_risk - log(q99(relative_risk)), 0))`, preserving the original
clipped positive-risk normalization. Computation stays in log space to avoid
overflow. Explicit `module_score_kind: relative_risk` supports the original
positive-risk scale; the default is native `log_risk`. Inference never fits
this scale from its current cohort.

Every inference saves `final_valid_mask.npy` and `prediction_endpoints.npy`;
`final_scores.npy` stores NaN for invalid predictions. Consistency outputs
also include an NPZ with `participant_ids`, `endpoints`, `kg`, `feature`, and
`evidence`. An empty plan therefore does not contribute a finite model prior
to probability evaluation. Generic fixed-threshold screening remains in the
inference JSON; frozen per-endpoint classifications are saved separately as
`consistency_predictions.npz`.

`apply-consistency` also writes `consistency_predictions_baseline.csv` and
`consistency_predictions_filtered.csv` alongside that NPZ (a custom NPZ stem
changes the CSV prefixes). Each table contains `participant_id` followed by every endpoint,
preserves the participant axis, and uses `0`/`1` for valid classifications and
empty cells for invalid predictions. These CSVs represent frozen decisions,
not probabilities. The NPZ remains the programmatic output.

Inference also writes probability CSVs automatically: `predictions.csv` for
Prevalent, or `predictions_<horizon>.csv` for each available Risk prediction
horizon. The first column is `participant_id`; all endpoint columns follow the model axis,
and rows preserve participant order. Invalid predictions are empty cells.
The run manifest records `outputs.predictions_csv` or the horizon-to-path
mapping `outputs.horizon_predictions_csv`. Risk prediction CSVs contain
absolute probabilities, while `final_scores.npy` retains Cox log-risk.

## Command behavior

`trace train-modules`, `trace score-modules`, `trace train-integration`,
`trace calibrate`, and `trace train-planner` execute
the training stages described above. `trace
prepare-data` validates targets and the split table when a split table is
configured, while inference-only cohorts remain label-free. `trace evaluate`
requires predictions for the complete cohort plus an accompanying participant
ID array, verifies its order, and computes metrics only on rows assigned to
`test`.

Prevalent evaluation requires probability predictions `(N,D)`, participant IDs
`(N,)`, endpoint names `(D,)`, and a boolean prediction-valid mask `(N,D)` for the
complete cohort. By default these are `final_scores.npy`, `participant_ids.npy`,
`prediction_endpoints.npy`, and `final_valid_mask.npy` under `output.directory`.
External files can be configured with `evaluation.predictions`,
`evaluation.participant_ids`, `evaluation.endpoints_file`, and
`evaluation.valid_mask_path`. When a metadata setting is omitted, evaluation
loads its companion filename from the prediction file's directory; it does not
use a different run's files from `output.directory`. Explicit metadata paths
override those companion filenames.
Both input routes require matching participant ID and endpoint order and the same shapes.
Every prediction-valid cell must contain a finite probability in `[0,1]`.
Supply probabilities rather than logits. Prediction-invalid cells are excluded regardless of their
stored value, and label masks additionally restrict the test metrics.

Inference for risk prediction also writes `horizon_risks.npz`, independent of
`output.top_endpoints`. It contains one `(N,D)` probability array per year-based
horizon (`5y`, `10y`, etc.), `participant_ids` `(N,)`, `endpoints` `(D,)`, and a
boolean prediction `valid_mask` `(N,D)`. This mask is derived only from inference
outputs, never from labels. Evaluation checks participant ID and endpoint order, all array
shapes and the finite `[0,1]` probability contract for valid predictions.

`evaluation.horizon_predictions_path` may override the default output artifact.
For each horizon h, event by h is positive; follow-up through h without an event
by h is negative (including censoring exactly at h); censoring before h is
unknown and excluded. The prediction mask and target mask also apply. Metrics
include known/unknown counts and per-endpoint AUROC/log-loss. These are
complete-case horizon metrics, not IPCW-adjusted metrics or a C-index. Missing
horizon probabilities are an error: raw Cox `final_scores.npy` is never treated
as a probability. No test targets are passed into inference or model selection.
