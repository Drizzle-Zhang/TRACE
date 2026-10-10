# TRACE architecture

TRACE (**TRajectory planning Agent for Cost-Effective disease evaluation**)
uses one shared execution model for adaptive medical
decision-making, with task-specific strategies for multi-disease diagnosis and
risk prediction.

The implementation keeps evidence acquisition and trajectory planning shared
across the two task families.

## Shared layers

1. Data contracts and alignment.
2. Module risk-skill registry.
3. Missingness-aware risk integration.
4. KG-guided candidate ranking.
5. Action-value ADD/STOP planning.
6. Attribution and consistency validation.
7. Artifact manifests, checkpoints, and structured outputs.

## Task-specific behavior

### Prevalent

- native module training: LightGBM + internal fivefold isotonic calibration;
- module output: disease probability;
- integration: disease-wise LightGBM;
- calibration: isotonic;
- reward: masked binary-cross-entropy reduction.

### Risk prediction

- native module training: XGBoost Cox, internal early stopping then full refit;
- native module output: Cox log-risk;
- integration: disease-wise Cox model;
- post-processing: standardized risk and horizon-specific absolute risk;
- reward: masked multi-horizon binary-cross-entropy reduction.

## Dependency direction

```text
contracts <- data <- risk_skills <- integration <- planning
                                      ^              |
                                      |              v
knowledge --------------------------> consistency <- agents
runtime/evaluation <---------------------------------+
```

No core layer imports CLI or pipeline code. Shell scripts may invoke the CLI,
but cannot contain model or scoring logic.

## Native module training boundary

The catalog/feature/outcome readers live in `data`; estimator wrappers live in
`risk_skills/module_model.py`. `training/modules` owns validated settings,
train-only fold construction, task-specific fitting, outer cross-fitting,
bank/cache publication and prepare/train/score orchestration. Outcome reading
is confined to preparation; training validates prepared targets and passes
only train slices to fitting. Scoring consumes only features and frozen models,
plus strict train OOF identity in training mode. Shared-matrix module
adapters and precomputed-score pipelines are supported.

```text
catalog -> strict features + main split + aligned targets
        -> train-only outer crossfit -> final module bank + train OOF
        -> validation/test final-model scores + train OOF
        -> fusion -> prevalent calibration / incident absolute risk -> planner
```
