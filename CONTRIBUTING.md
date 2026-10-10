# Contributing to TRACE

Use English for documentation, comments, docstrings, messages, configuration annotations, and distributed metadata. Preserve disease and model identifiers, scientific source fingerprints, and numerical reference fixtures.

Run changes from a source checkout. The development requirements include the validated scientific versions and Parquet support so the complete suite can run. Optional PyTorch MoE, explanations, and LLM integrations have separate extras; install them only for those workflows:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -ra
python -m ruff check src tests tools examples
python -m black --check src tests tools examples
python -m mypy src
```

Keep regression checks for row identity, label eligibility, OOF isolation, model and score provenance, masks, candidate ordering, and ADD/STOP behavior. Describe intentional algorithm changes in `docs/methods.md` and their validation methods and results in `docs/validation.md`.

Do not commit real participant data or identifiers, model binaries, credentials, private infrastructure paths, generated results, or internal development plans and run inventories. Use synthetic examples and keep patient inputs in their authorized environment. The `.env.example` is a checklist; TRACE does not automatically load `.env`.

Preserve the runtime resource checksums. Resource presentation changes must retain original and distributed hashes separately; never rewrite a recorded original hash to make an edited resource appear unchanged.
