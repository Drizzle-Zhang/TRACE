# Understand TRACE's core concepts

Complete [getting started](getting_started.md) first, then read this page as needed. When preparing your own data, use it alongside the [real data guide](real_data.md) and [data contracts](../data_contracts.md).

## What happens during a run

TRACE divides its inputs into evidence modules. Each module produces scores; the fusion model combines scores from the modules currently selected; the planner decides whether to add another module. The final output includes risks, the selected path, and accumulated burden.

```text
Input features → Module models → Module scores
                                      ↓
                         Selected modules → Fused risk
                                      ↑          ↓
                           ADD ← KG ranking + Q decisions → STOP
                                                             ↓
                                                Final risk, path, burden
                                                             ↓
                                                    Consistency checks
```

## Three data coordinates: participant_id, module, and endpoint

| Term | Meaning in this project | Small example |
| --- | --- | --- |
| `participant_id` | Participant identifier used to match the same person across files | `synthetic_001`, appearing in the participant, feature and outcome tables |
| `module` | A group of evidence or features, together with models based on them | A questionnaire or a category of tests; templates use `M1` as a placeholder |
| `endpoint` | An outcome to predict, often called `disease` in code and filenames | Templates use `D1` as a placeholder |

Use a stable, unique identifier to match each participant across tables. Identifiers such as `synthetic_001` in these examples are fictional.

One module can score multiple endpoints, and one endpoint can receive scores from multiple modules. A native module model corresponds to a specific `(module, endpoint)` pair and stores that pair's feature names and order.

In the documentation, `N` is the participant count, `M` is the module count, and `D` is the endpoint count. For example, module scores have shape `(N, M, D)`, while labels usually have shape `(N, D)`. Names, order, and shape are all part of the input contract. Equal shapes do not mean files represent the same people or models.

## What available, observed, and pred_mask tell you

| Name | Shape | Question it answers |
| --- | --- | --- |
| `availability`, stored in `<prefix>_avail.npy` | `(N, M)` | Does this participant have usable input for this module? |
| `observed_modules` / `used_modules` | `(N, M)`, or `(M,)` when planning for one person | Which modules has the current planning path selected? |
| `pred_mask`, stored in `<prefix>_pred_mask.npy` | `(N, M, D)` | Does this participant's module have a valid score for this endpoint? |

For example, a participant has input for `M1`, but the current path has not selected it: `availability=True`, `observed=False`. If the module has a valid model score only for `D1`, and none for `D2`, its `pred_mask` is `[True, False]`, and its scores can be `[0.3, NaN]`. After `ADD M1`, the module is selected. This does not create a valid score for `D2`.

The native feature reader determines availability from features. Within a module's configured pairs, the module can be available if at least one selected feature for a pair has a finite value for that participant. Outcome labels do not determine module availability. `pred_mask` separately records whether a model exists and produces a finite score for that endpoint.

Missingness must be explicit. Score cells with `pred_mask=False` must be `NaN`; cells with `pred_mask=True` must be finite. A valid prediction cannot come from a module with `availability=False`. The number `0` is a real value and cannot stand for a missing score. Known labels have their own `y_mask` / `label_mask`; prediction masks cannot establish whether labels are known.

## train, validation, and test: their roles

| Split | Purpose |
| --- | --- |
| `train` | Fit modules, the fusion model, and the planner |
| `validation` (also accepted as `val`) | Select settings, fit fusion probability calibration, and select consistency rules |
| `test` | Evaluate performance after the method is frozen |

A preassigned split table determines these groups. Provide `participant_id,split`; TRACE matches IDs to the participant axis and derives row indices automatically. The table must assign every participant once, and all three splits must be nonempty. An optional `row_index` column is checked against IDs when supplied.

Test labels do not enter fitting or model selection. Module selection during inference also receives no target labels or label masks. `evaluate` can read prediction files for the complete cohort, but its metrics use only the split table's `test` rows.

## OOF: score training participants without seeing their labels

OOF means out-of-fold prediction. Native module training divides `train` into five folds. It fits on four folds and predicts the remaining fold, then combines these predictions into OOF scores for the training population. Separately, it fits the final module model on all eligible train rows to score validation, test, or a new cohort.

Predictions from the final model on the train rows it has already learned from cannot replace OOF scores.

Fusion and planner training require training scores with `score_mode=training` and matching split and participant records. New-cohort scores with `score_mode=inference` cannot replace training OOF.

Prevalent score files are always named `<prefix>_oof_probs.npy`, including those generated for new-cohort inference. **A filename containing `oof` does not prove that the scores are valid training OOF.** Check the score bundle manifest's provenance.

## calibration: bring scores closer to probability meaning

TRACE's calibration steps fit probability mappings, set consistency scales or select classification rules. Probability calibration uses model scores and known outcomes; the table shows the reference rows for each step:

| Step | What it learns | Rows it uses |
| --- | --- | --- |
| Internal prevalent module isotonic calibration | A probability mapping for one module model | Train rows supplied for that module fit, using five internal folds |
| `calibrate` | Endpoint probability mappings for the fusion model | Validation rows |
| `calibrate-consistency-scale` | The consistency scaling reference for Risk prediction module scores | Validation rows |
| `calibrate-consistency` | Endpoint classification thresholds and consistency strategies | Validation rows |

The last two steps do not automatically turn Cox log-risk into absolute probability. Keep the calibrated fusion model, calibration artifact, and planner trained with that calibration together as one set. The program checks upstream model identities.

## KG, burden, and Q: why add this module

**KG (knowledge graph)** is represented by a module × endpoint knowledge matrix. It ranks candidate modules and can contribute to consistency scores. Its values describe configured associations, not participant disease probabilities. Its module and endpoint axes must match the inputs.

**burden** is the cost of acquiring evidence. A configuration can include monetary and time dimensions, combined through weights for accumulated burden and planning cost. The repository's `ukb_relative.yaml` defines relative burden; its values cannot be read directly as local test prices.

The **Q-model** predicts an action value for taking an action in the current state, balancing prediction improvement against burden. Q values are planning scores, not risk probabilities. Candidates are first ranked by KG/burden, then Q values help select an action.

- `ADD`: add an available candidate module that has not yet been selected, then recompute fused risk.
- `STOP`: end the current path. This can happen when no candidates remain, gain does not exceed the threshold, the STOP value wins, or the step limit has been reached.

The output path records selected evidence and the reason for stopping. STOP does not mean an endpoint is negative or that all risks have been ruled out.

## How to read prevalent probabilities and Risk prediction Cox scores

`prevalent` is the configuration value for assessing/classifying outcomes already present; Risk prediction estimates the risk of events after baseline and uses `longitudinal` in native workflows. Their score scales differ:

| Output | Meaning | Can you read it as a percentage? |
| --- | --- | --- |
| Prevalent `predictions.csv` | Final fusion probabilities; calibrated when the calibration artifact is configured | Yes; for example, `0.2` means a 20% model probability |
| Risk prediction module `<prefix>_scores.npy` | Native Cox module log-risk | No |
| Risk prediction `final_scores.npy` | Final fusion Cox log-risk | No |
| Risk prediction `predictions_<horizon>.csv` | Absolute probabilities at the named time, such as `predictions_10y.csv` | Yes, for nonempty probability cells |

Cox log-risk, written as `r`, can be negative or greater than 1. `exp(r)` is a risk multiplier relative to the model baseline, often called HR / relative risk. For two people, the ratio of their multipliers is `exp(r_a-r_b)`. This is still not the probability of an event within a time interval.

Absolute probability at time `t` also requires the model's baseline survival `S0(t)`. The relation is `1 - S0(t) ** exp(r)`. Native fusion models for Risk prediction store baseline survival estimated during training and generate `horizon_risks.npz`. If absolute probabilities are missing, evaluation does not substitute log-risk for them.

For relative-risk module inputs, set `consistency.module_score_kind: relative_risk` when computing Risk prediction consistency. Prepare fusion inputs in the score scale required by the matching model.

## Read predictions

Open the prediction CSVs directly in a spreadsheet application or text editor:

- Prevalent: `predictions.csv` contains the final disease probabilities.
- Risk prediction: `predictions_<horizon>.csv` contains event probabilities for that horizon, with one file for every available horizon, such as `predictions_5y.csv` and `predictions_10y.csv`.

Each CSV starts with `participant_id`, followed by **all endpoint names** as column headers. Rows retain the input participant order; endpoint columns follow the model's endpoint order. A value of `0.2` means a model-estimated 20% probability of prevalent disease, or a 20% event probability by the horizon named in the file. Empty cells mean that the prediction is unavailable, not zero risk or a negative outcome. These tables contain probabilities rather than binary classifications.

`run_manifest.json` records their paths under `outputs.predictions_csv` for Prevalent or the `outputs.horizon_predictions_csv` horizon-to-path mapping for Risk prediction. Risk prediction CSVs contain absolute probabilities; `final_scores.npy` retains Cox log-risk scores.

The arrays remain available for programmatic analysis. Read their values with their identity arrays and validity mask: cell `[i, j]` belongs to `participant_ids.npy[i]` and `prediction_endpoints.npy[j]`.

| File under the output directory | How to read it |
| --- | --- |
| `final_scores.npy` | `(N, D)` final prevalent probabilities or Risk prediction Cox log-risk scores |
| `final_valid_mask.npy` | `(N, D)` booleans; only `True` cells are valid predictions |
| `participant_ids.npy`, `prediction_endpoints.npy` | Row identifiers `(N,)` and column identifiers `(D,)` for the prediction matrices |
| `horizon_risks.npz` | Risk prediction probabilities: one `(N, D)` array per horizon, plus `participant_ids`, `endpoints`, and `valid_mask` |
| `final_used_modules.npy` | `(N, M)` booleans; columns follow `modules` in `run_manifest.json` |
| `accumulated_burden.npy` | `(N,)` burden totals in the configured burden scale |
| `agent_results.jsonl` | One JSON record per participant, with named selected modules, endpoint summaries and the action trajectory |

Invalid `final_scores.npy` cells are `NaN`. Apply the validity mask to horizon arrays as well: a numeric value in a masked cell does not establish a valid prediction. CSVs already leave invalid predictions empty. These prediction masks are separate from masks that identify known outcome labels.

In `agent_results.jsonl`, `endpoints` lists valid endpoints in priority order and can be shortened by `output.top_endpoints`. The CSVs and arrays contain the complete endpoint matrix. An endpoint's `endpoint_index` points into `prediction_endpoints.npy`; its `score` has the same meaning as `final_scores.npy`, while `raw_score` is the fusion result before optional prevalent calibration. Risk prediction probabilities appear in that endpoint's `horizon_risks` mapping.

`selected_modules` gives readable names. Boolean `initial_used_modules`, `final_used_modules`, and `observed_modules` follow the module order in `run_manifest.json`; trajectory `module_index` and `candidate_module_indices` use zero-based positions in that order. JSON action values are lowercase `add` and `stop`. Each trajectory's `risk_profile` is the profile used to choose that action. An `add` appears in the next profile's observed modules; the final `stop` profile contains the final risks. Its `metadata.stop_reason` explains why planning ended.

The demonstration produces probabilities and planning paths. It does not select endpoint classification thresholds. `retained_by_consistency` is `True` for every valid endpoint when consistency is disabled; it does not indicate a positive diagnosis. With consistency enabled, it records the configured screening strategy.

After `apply-consistency`, open `consistency_predictions_baseline.csv` for the risk-only classifications or `consistency_predictions_filtered.csv` for classifications after the frozen consistency rules. Both contain `participant_id` and all endpoint columns: `1` means classified positive, `0` means classified negative, and an empty cell means unavailable. These are classifications, not probabilities or observed diagnoses. The same results remain in `consistency_predictions.npz` as `baseline` and `filtered`, with identity arrays and `valid_mask`. A custom NPZ output name changes the CSV filename prefixes accordingly. See the [consistency workflow](real_data.md#check-consistency) to generate them.

## consistency: check agreement between predictions and evidence

Consistency modules use KG, feature contributions, and evidence rules to assess agreement between current predictions and the evidence used. Run these checks after prediction, then use validation data to select any classification adjustments.

Two outputs are easy to confuse:

- Consistency scores from inference, and raw-feature threshold checks in `threshold_checks.csv` / `threshold_summary.json`. Missing raw features produce `missing_feature`; these threshold checks do not change risks or classifications.
- Consistency classification rules selected on validation and then frozen. Applying them writes baseline and adjusted classification CSVs plus `consistency_predictions.npz`. Some Risk prediction strategies can also add positives outside the risk-only baseline.

Risk prediction evaluation determines outcomes at each horizon. Events by the horizon are positive; participants observed through the horizon without an event are negative; earlier censoring leaves the outcome unknown at that horizon, so those participants are excluded. Evaluation reports AUROC and log-loss on these known-outcome complete cases.

## Model files and manifests

TRACE model files store the fitted model, task, module order and endpoint order. Use a TRACE version and model dependencies compatible with the saved files. A `.joblib` extension alone does not make a model compatible with TRACE.

A manifest records file hashes, training splits, inputs and settings. Models trained with TRACE need their adjacent `*.joblib.manifest.json` files for fusion, calibration and planning. A module bank needs `module_bank_manifest.json` and its referenced models; keep its OOF cache for further training. Copy the matching model files and manifests together, including all files supplied in a downloaded model package.

If a file hash, task or axis check fails, restore the matching files or regenerate the affected models using the [troubleshooting guide](troubleshooting.md). Keep manifests as generated so these checks remain effective.
