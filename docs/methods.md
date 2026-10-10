# TRACE methods

TRACE (**TRajectory planning Agent for Cost-Effective disease evaluation**)
provides research workflows for prevalent disease classification and
risk prediction for future events. It combines predictions from groups of
features, called modules, and learns which available module to add next. A
module may represent a questionnaire, a measurement group, or another explicitly
configured feature source. The planner produces an ADD/STOP trajectory, the
modules used, their accumulated burden, and the final risk estimates.

The two task families share alignment, missingness, candidate ranking, and
planning code. Their estimators, score meanings, and training rewards differ.

| Stage | Prevalent | Risk prediction |
| --- | --- | --- |
| Task | Disease presence at baseline | Events after baseline, including configured death endpoints |
| Module estimator | LightGBM classifier with internal isotonic calibration | XGBoost Cox model |
| Module output | Probability | Cox log-risk |
| Native fusion estimator | One LightGBM classifier per endpoint | One XGBoost Cox model per endpoint |
| Risk processing | Validation-fitted isotonic calibration | Train-derived standardization and baseline survival |
| Planner reward | Reduction in masked binary cross-entropy | Reduction in masked binary cross-entropy across named time horizons |

The repository supplies algorithms and synthetic examples. Researchers supply
their cohort definition, feature preparation, outcome definitions, selected
feature lists, and approved split assignments. TRACE does not reproduce feature
selection or generate a study cohort from raw UK Biobank records. Software
checks and model-validation requirements are described in [verification](validation.md).

## Inputs and prediction validity

Every array has an explicit participant, module, and endpoint order. Module
scores have shape `(N, M, D)`, module availability has shape `(N, M)`, and final
risks have shape `(N, D)`. Participant identifiers and feature codes are textual
identities; leading zeros and distinct codes are preserved. Catalogs explicitly
declare feature tables and module/endpoint pairs. Selected feature order is
preserved, and feature selection is not repeated by the trainer.

Missing feature values remain `NaN`. A row with some observed features may be
predicted by the tree estimator; a row with no observed selected features has
no module prediction. Participant rows are retained. The prediction mask
`pred_mask[n, m, d]` determines whether a score is usable: valid scores must be
finite, and invalid scores must be `NaN`. Module availability describes what can
be acquired; the observed-module mask describes what the planner has already
used. An available module need not predict every endpoint.

Outcome masks have a separate purpose. They describe which labels can be used
in offline fitting or evaluation. They are not supplied to candidate ranking,
Q-state construction, or online planning. A participant with available features
may receive a prediction even when their outcome is unknown or ineligible for
training. A final endpoint is valid only when at least one observed module has
a valid prediction for it; an empty plan does not turn a finite estimator prior
into an evaluable prediction.

Raw outcome inputs use binary `target_y` and baseline-relative
`BL2Target_yrs`, measured in years. Prevalent positives have `target_y=1` and
negative time; future cases are excluded. Controls with `target_y=0` remain
eligible without a follow-up time. Fitting for risk prediction excludes preexisting
cases and requires a positive duration: signed Cox time is positive for an
event and negative for censoring. Missing outcomes are masked. Zero event times,
invalid labels, infinity, and nonpositive known control follow-up are rejected
rather than silently recoded. Prepared labels, event indicators, durations, and
signed times are checked for agreement. See the detailed
[data contracts](data_contracts.md) for file formats and alignment requirements.

## Splits and module cross-fitting

Supervised workflows require a complete, user-supplied split table with one
assignment per participant: `train`, `validation`, or `test`. Row indices and
identifiers must agree with every participant-axis array. There is no random
fallback for the main split. Training artifacts record split and input
fingerprints so that downstream supervised stages can reject stale models.

Module training creates a shared five-fold partition within `train` only. The
default shuffled partition does not use labels; an explicit, aligned train-only
fold table may instead preserve a study's folds. For each configured
module/endpoint pair, five models fit the outer training complements and predict
their held-out fold. These predictions form the train out-of-fold (OOF) cache.
A final module model fits the eligible full training set for deployment.

Eligibility is checked for every outer complement and the final fit before any
model is fitted. Default admission requires at least 50 positives or events and
a positive fraction of at least 0.001, together with the class requirements of
the internal fitting procedure. An insufficient pair is skipped with its
reason, or raises an error under the configured error policy. A skipped pair is
not a trained model. OOF coverage includes feature-available held-out rows even
when their labels are ineligible; those labels never enter fitting.

Prevalent module fitting uses `CalibratedClassifierCV` with five internal folds,
isotonic calibration, and `ensemble=True`. Each internal base classifier fits
one part of the supplied outer complement and calibrates on its internal held
part. The outer held-out fold, main validation split, and test split participate
in neither internal fitting nor internal calibration. The final deployment
model uses the same internal procedure within the full eligible train set.

Module fitting for risk prediction selects boosting rounds using an event-stratified
holdout inside the supplied training rows, with a default 80/20 division and
Cox negative log-likelihood for early stopping. It then refits a new estimator
on all those eligible rows for the selected number of rounds. Thus, each outer
OOF model selects rounds within its complement, without consulting its outer
held-out labels. The final model performs selection within the main train set
and refits that full eligible set. Predictions use `output_margin=True`: they
are log-risk scores, not probabilities. Exponentiating a margin gives the
corresponding relative-risk score.

`score-modules` has two explicit modes. Training mode verifies the participant,
feature-source, split, and OOF identities; it uses verified OOF for train rows
and final models for validation/test rows. Inference mode predicts a new cohort
from frozen models and aligned features without reading outcomes, split tables,
or OOF. Native downstream training requires training-mode scores. The OOF
contract applies to module predictions; fusion and Q are fitted subsequently
and are not themselves claimed to have an additional outer cross-fit.

## Fusion, calibration, and absolute risk

Native fusion learns a separate estimator for each endpoint. Each training
participant contributes a fully available module state and additional random
nonempty subsets, exposing the estimator to incomplete observations. For `M`
modules, an endpoint's fusion features contain `2M` values: active module scores
followed by binary active-mask indicators. Inactive scores are zero-filled.
Prevalent probabilities first become clipped logits; inputs for risk prediction
remain log-risk. Label masks restrict the endpoint targets used to fit fusion.

When an endpoint has too few classes or events for its configured fusion
estimator, a constant estimator supplies its output through the same array
interface. Prediction validity still depends on observed module evidence.

Prevalent fusion calibration fits endpoint-specific isotonic mappings on the
validation split only, with clipped extrapolation. By default each class must
have at least 10 usable validation examples; otherwise the mapping is identity.
Calibration metrics describe fitting and selection on the validation rows;
final performance is evaluated on test rows.

Fusion for risk prediction uses a Cox objective. Its default number of rounds is fixed;
optional early stopping selects rounds on eligible main-validation rows while
gradient fitting uses train rows only. The fitted fusion model stores
train-derived means and scales for standardized planning signals and Breslow
baseline survival at the configured horizons. Absolute risk at horizon `t` is
computed from the frozen baseline survival and raw fusion log-risk `r` as
`1 - S0(t) ** exp(r)`, with numerical safeguards. Standardized planning signals
are separate from this probability calculation. Native inference uses the
baseline survival embedded in the fusion artifact; an explicitly configured
external baseline may be supplied for a compatible model. No baseline curve is
fitted from test outcomes or the inference cohort.

## Learning and executing ADD/STOP decisions

The planner first ranks available, unobserved modules using endpoint uncertainty,
urgency, knowledge-graph links, uncovered endpoint coverage, and burden. Ranking
uses prediction-time inputs only. An endpoint that is currently unobserved can
still justify acquiring a module that would cover it.

Offline Q training samples observed-module states and pairs them with candidate
ADD actions. A prevalent action's target is its masked binary cross-entropy
reduction. A target for risk prediction combines such reductions across configured
absolute-risk horizons and known outcomes. Both subtract
`lambda_cost * normalized_module_burden`, where normalization uses the maximum
configured module burden. A LightGBM regressor learns these immediate action
values from train examples. This is a supervised estimate of gain minus burden;
the native trainer does not fit a discounted multistep return. Validation action
errors and frozen validation trajectories are diagnostics.

State features include task-specific risk summaries, the observed-module mask,
the number of observed modules, and accumulated burden. Action features include
module identity, burden, and a STOP indicator. Online execution recomputes the
risk profile after each ADD and ends when the configured Q/STOP criterion, lack
of candidates, or step limit is reached. Default STOP has value zero; optional
model STOP uses the learned action estimator. The planning experiment selects
among modules declared available in its inputs; actual collection of new
measurements is outside this software workflow.

## Optional consistency and frozen thresholds

Consistency can combine three aligned signals: knowledge-graph coverage,
feature consistency from signed model contributions and medical feature
weights, and medical-evidence consistency from rule-level support and conflict.
The evidence compiler preserves distinct matched rules and their reliability
rather than cancelling supporting and conflicting rules before scoring. An
already aggregated weight array cannot reconstruct omitted source rules.

For feature/evidence consistency, prepare contribution arrays from the selected
models and align them to the participant, endpoint and feature axes. Some
diagnostics also use raw feature values. See the
[consistency workflow](user_guide/real_data.md#check-consistency) and
[evidence resource guide](../resources/medical_evidence/v1/README.md).

Probability cutoffs and consistency policies are selected on validation rows
only, then frozen. Prevalent selection compares KG × feature, evidence, and
KG × feature × evidence strategies per endpoint, retaining baseline unless
validation F1 improves. Selection for risk prediction uses its named horizon and the
implemented hard gates, joint gates, soft adjustments, and endpoint policy
selectors; `product_filters` selects a three-product search. Some adjusted
policies for risk prediction may add positives outside the
baseline classification.

Consistency for risk prediction also freezes a validation-derived q99 normalization
of module relative risk. Scale fitting reads validation scores and splits,
without reading labels. The native log-risk normalization is calculated in log
space and clipped at one; inference reuses the frozen scale rather than fitting
it to each new cohort. Frozen rule application reads inference outputs and rules
only. It checks task, horizon, endpoint order, and model/knowledge identities;
evaluation additionally checks the rule and input hashes.

## Final evaluation

Final evaluation uses only rows assigned to `test`, intersecting prediction
validity with observed target validity. For horizon `h` in risk prediction, an
event by `h` is positive; follow-up through `h` without an event by `h` is
negative, including censoring exactly at `h`. Censoring before `h` is unknown
and excluded. The resulting AUROC and log-loss are known-outcome complete-case
metrics. Evaluation of risk prediction
requires horizon probabilities and never treats raw Cox margins as probabilities.
Frozen consistency evaluation reports baseline and adjusted test precision,
recall, F1, and confusion counts without selecting policies on test labels.

Synthetic checks verify software behavior. Assess predictive performance on
an independent cohort with the study's preprocessing, outcomes and configuration.
