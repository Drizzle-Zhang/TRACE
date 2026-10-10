# TRACE

**TRACE** (**TRajectory planning Agent for Cost-Effective disease evaluation**) is a computational framework for adaptive multi-disease assessment guided by a knowledge graph. It supports prevalent disease classification and **risk prediction** for future events.

Clinical information is grouped into modules, such as questionnaires, physical measurements and laboratory tests. Module models generate disease scores, a fusion model combines selected modules, and a planner chooses the next available module or STOP. Risk estimates update as modules are added, balancing estimated predictive benefit against configured monetary and time costs. Outputs include disease scores, selected modules, assessment trajectories and accumulated burden.

## Quick start

Clone or download [this repository](https://github.com/Drizzle-Zhang/TRACE). Use Python 3.13 for the verified setup, and run these commands from the `TRACE` directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/reproducible.txt -e '.[longitudinal]'
```

Try training and prediction with a small synthetic cohort:

```bash
python examples/prepare_demo.py --task prevalent
trace run --config data/demo/prevalent/train.yaml
trace run --config data/demo/prevalent/predict_new.yaml
```

See the [first-run guide](docs/user_guide/getting_started.md) for the risk prediction example and expected outputs.

## Use trained models

The supplied configurations cover prevalent disease assessment and risk prediction. Each task offers three priorities: `performance` emphasizes predictive performance, `cost` emphasizes lower acquisition burden, and `balanced` provides a trade-off.

Follow the [trained-model guide](docs/user_guide/trained_models.md) to set up the packages. Prepare your data using the [input semantics](docs/user_guide/trained_models.md#input-semantics), keeping field encodings and units consistent with model training. Then run a configuration, for example:

```bash
trace run --config configs/trained_models/prevalent-balanced.yaml
```

For risk prediction, use `configs/trained_models/survival-balanced.yaml`. Replace `balanced` with `performance` or `cost` to choose another priority. Results include disease scores and each participant's assessment trajectory and burden.

Model files are distributed separately; a public download link is not yet available. Review the [known model limitations](docs/user_guide/trained_models.md#known-limitations-and-compatibility) before using them.

## Train and predict with your own data

Define your clinical modules, disease outcomes and acquisition burden, then configure the supplied templates using the [data preparation guide](docs/user_guide/real_data.md#configure-paths-in-yaml). After preparing the training and new-cohort configurations, run:

```bash
trace run --config data/configs/prevalent-train.yaml
trace run --config data/configs/prevalent-predict.yaml
```

Training learns the module models, fusion model and planner. Prediction applies the trained models to a new cohort without requiring outcome labels. The guide also provides training and prediction configurations for risk prediction.

## Check consistency

Optional consistency checks assess how well predictions agree with the knowledge graph, the features driving each prediction and medical evidence. The resulting scores can guide filtering or adjustment of classifications, using rules selected on validation data and reused for new cohorts.

```bash
trace run --config data/configs/prevalent-consistency.yaml
```

See the [consistency guide](docs/user_guide/real_data.md#9-add-consistency-inputs-after-the-basic-workflow-succeeds) for preparation and use. The bundled evidence is template-derived and awaits medical review; see its [scoring limitations](resources/medical_evidence/v1/README.md#statement-provenance-and-scoring-limitations).

## User guides

- [First run](docs/user_guide/getting_started.md): install TRACE and run synthetic examples.
- [Trained models](docs/user_guide/trained_models.md): choose a variant and prepare prediction inputs.
- [Your own data](docs/user_guide/real_data.md): configure training, prediction and consistency checks.
- [Core concepts](docs/user_guide/concepts.md): understand the stages and interpret results.
- [Troubleshooting](docs/user_guide/troubleshooting.md): resolve common setup and input errors.

## License

[MIT License](LICENSE).
