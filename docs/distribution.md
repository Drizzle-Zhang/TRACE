# Code and model distribution

The source repository is [Drizzle-Zhang/TRACE](https://github.com/Drizzle-Zhang/TRACE).
The source supports synthetic training and prediction without
model downloads. Trained-model configurations additionally require the separate
inference assets described below.

## GitHub contents

| Files | Purpose |
| --- | --- |
| `src/trace_agent/` | Runtime, training, inference, planning and evaluation code |
| `configs/`, `pipelines/` | Portable examples, task workflows and six trained-model variants |
| `resources/` | Burden/evidence tables, schemas and checksums |
| `tests/`, `examples/` | Normal tests, numerical reference fixtures and synthetic inputs |
| `tools/` | Local model packaging utilities |
| `docs/` | Methods, architecture, input contracts and user instructions |
| `pyproject.toml`, `requirements-dev.txt`, `constraints/`, `MANIFEST.in` | Installation, reproducible versions and source packaging |
| `README.md`, `LICENSE`, `CITATION.md`, `CONTRIBUTING.md`, `.env.example`, `.gitignore`, `.gitattributes` | Project information and portable repository settings |

The GitHub tree contains no trained model binaries or populated participant
inputs. Numerical reference fixtures are synthetic software checks.

## Cloud model packages

The planned distribution has one shared resource archive, one common archive
per task, and one archive per planner priority. Download the common archives
once, then choose the planners you need. The archives are not yet published;
sizes below describe uncompressed assets.

The Risk prediction packages retain `survival` in archive and directory names; see [task naming](user_guide/getting_started.md#task-naming) for the public names and configuration values.

| Task | Archive | Extraction contents | Approximate uncompressed size |
| --- | --- | --- | ---: |
| Both tasks | `TRACE-v0.1.0-resources.tar.gz` | Matching evidence under `data/trained_models/common/medical_evidence/v1/` | 2.48 MB |
| Prevalent | `TRACE-v0.1.0-prevalent-common.tar.gz` | `data/trained_models/prevalent/common/` and empty `data/inputs/prevalent/` templates | 8.31 GiB |
| Risk prediction | `TRACE-v0.1.0-survival-common.tar.gz` | `data/trained_models/survival/common/` and empty `data/inputs/survival/` templates | 1.591 GiB |
| Prevalent | `TRACE-v0.1.0-prevalent-performance.tar.gz` | `data/trained_models/prevalent/performance/` | 6.70 MB |
| Prevalent | `TRACE-v0.1.0-prevalent-balanced.tar.gz` | `data/trained_models/prevalent/balanced/` | 6.70 MB |
| Prevalent | `TRACE-v0.1.0-prevalent-cost.tar.gz` | `data/trained_models/prevalent/cost/` | 6.70 MB |
| Risk prediction | `TRACE-v0.1.0-survival-performance.tar.gz` | `data/trained_models/survival/performance/` | 54.5 KB |
| Risk prediction | `TRACE-v0.1.0-survival-balanced.tar.gz` | `data/trained_models/survival/balanced/` | 54.9 KB |
| Risk prediction | `TRACE-v0.1.0-survival-cost.tar.gz` | `data/trained_models/survival/cost/` | 54.7 KB |

Each task common package contains its inference-only module bank, selected
feature lists, fusion model, knowledge/consistency arrays and matching schemas.
A priority package contains `q_model.joblib`, `burden.yaml` and
`MODEL_BUNDLE.json`. The input templates contain headers and no participant rows.

Use the archive checksums and package index supplied with a download to verify
its files and source version. Saved consistency rules require their matching
models and evidence resources. Setup and additional consistency
inputs are described in the
[trained-model guide](user_guide/trained_models.md).

## Private research files

Keep populated participant tables, identifiers, outcomes, features, split rows,
OOF/training caches, individual attributions and predictions, checkpoints,
fitting material, credentials and machine-specific profiles in the authorized
private environment. They belong
in neither the GitHub source tree nor the public model download packages.
