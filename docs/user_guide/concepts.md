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

## Three data coordinates: eid, module, and endpoint

| Term | Meaning in this project | Small example |
| --- | --- | --- |
| `eid` | Participant identifier used to match the same person across files | `p001`, appearing in the participant, feature and outcome tables |
| `module` | A group of evidence or features, together with models based on them | A questionnaire or a category of tests; templates use `M1` as a placeholder |
| `endpoint` | An outcome to predict, often called `disease` in code and filenames | Templates use `D1` as a placeholder |

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

A preassigned split table determines these groups. Its default columns are `row_index,eid,split`. `row_index` starts at 0 and points to a row in the same score bundle; its `eid` must match that row exactly. The table must cover every row once, and all three splits must be nonempty.

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
| Prevalent `final_scores.npy` | Final fusion scores; calibrated probabilities when the calibration artifact is configured | Yes, for valid probability cells; for example, `0.2` means a 20% model probability |
| Risk prediction module `<prefix>_scores.npy` | Native Cox module log-risk | No |
| Risk prediction `final_scores.npy` | Final fusion Cox log-risk | No |
| Risk prediction `horizon_risks.npz` | Absolute probabilities at specified times, such as `10y`, with identity axes and a validity mask | Yes, for valid probability cells |

Cox log-risk, written as `r`, can be negative or greater than 1. `exp(r)` is a risk multiplier relative to the model baseline, often called HR / relative risk. For two people, the ratio of their multipliers is `exp(r_a-r_b)`. This is still not the probability of an event within a time interval.

Absolute probability at time `t` also requires the model's baseline survival `S0(t)`. The relation is `1 - S0(t) ** exp(r)`. Native fusion models for Risk prediction store baseline survival estimated during training and generate `horizon_risks.npz`. If absolute probabilities are missing, evaluation does not substitute log-risk for them.

Old arrays may contain HR. `consistency.module_score_kind: relative_risk` controls only Risk prediction consistency scaling; it does not convert a fusion model's input contract. Check the actual score scale before using old models.

## consistency: check agreement between predictions and evidence

Consistency modules use KG, feature contributions, and evidence rules to assess agreement between current predictions and the evidence used. This is supporting analysis after the model runs. High or low scores cannot replace data quality checks or prove that a prediction is correct.

Two outputs are easy to confuse:

- Consistency scores from inference, and raw-feature threshold checks in `threshold_checks.csv` / `threshold_summary.json`. Missing raw features produce `missing_feature`; these threshold checks do not change risks or classifications.
- Consistency classification rules selected on validation and then frozen. Applying them writes `consistency_predictions.npz` and may change final classifications. Some Risk prediction strategies can also add positives outside the risk-only baseline.

Risk prediction evaluation determines outcomes at each horizon. Events before the horizon are positive; participants observed through the horizon without an event are negative; earlier censoring leaves the outcome unknown at that horizon, so those participants are excluded. These are known-outcome complete-case metrics, not IPCW-adjusted metrics or a survival C-index.

## Model files and manifests

TRACE model files store the fitted model, task, module order and endpoint order. Use a TRACE version and model dependencies compatible with the saved files. A `.joblib` extension alone does not make a model compatible with TRACE.

A manifest records file hashes, training splits, inputs and settings. Models trained with TRACE need their adjacent `*.joblib.manifest.json` files for fusion, calibration and planning. A module bank needs `module_bank_manifest.json` and its referenced models; keep its OOF cache for further training. Copy the matching model files and manifests together, including all files supplied in a downloaded model package.

If a file hash, task or axis check fails, restore the matching files or regenerate the affected models using the [troubleshooting guide](troubleshooting.md). Keep manifests as generated so these checks remain effective.
