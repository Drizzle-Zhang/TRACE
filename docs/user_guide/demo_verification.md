# Check the synthetic examples

Use these checks after running the examples to confirm training, frozen prediction, test-only evaluation, and prediction for an unlabeled cohort. Install with the [reproducibility constraints](../../constraints/reproducible.txt); see [verification instructions](../validation.md) for the test suite and model-validation requirements.

## Run the examples

Use the installation steps in the [first-run guide](getting_started.md). From the repository root, with dependencies installed and new output directories:

```bash
python examples/prepare_demo.py --task prevalent
python -m trace_agent run --config data/demo/prevalent/train.yaml
python -m trace_agent run --config data/demo/prevalent/predict_new.yaml

python examples/prepare_demo.py --task longitudinal
python -m trace_agent run --config data/demo/longitudinal/train.yaml
python -m trace_agent run --config data/demo/longitudinal/predict_new.yaml
```

Each command should finish with exit code 0. The generator refuses an existing output directory. Use `--output data/demo/prevalent-2` or another new directory if needed, and use its generated configurations in later commands.

## Confirm the results

For each task, read files under `data/demo/<task>/`:

| Check | Expected result |
| --- | --- |
| `bank/module_bank_manifest.json` | Two module/endpoint pairs have `status=trained`; no skips or placeholders |
| Prevalent `output/predictions.csv` | Header `participant_id,D1`; 180 participant rows in input order with probabilities in 0–1 |
| Risk prediction `output/predictions_5y.csv`, `output/predictions_10y.csv` | Header `participant_id,D1`; 180 participant rows per file; 10-year probability at least 5-year probability |
| The corresponding CSVs in `new_output/` | Six participant rows in the new cohort's input order, with valid probabilities |
| `output/final_scores.npy` | Shape `(180, 1)`; finite scores |
| `output/final_valid_mask.npy` | All 180 predictions valid |
| `output/participant_ids.npy`, `output/prediction_endpoints.npy` | 180 participant IDs in input order and one endpoint, `D1`; these identify the rows and columns of the scores |
| `output/final_used_modules.npy` | Shape `(180, 2)`; columns follow `modules` in `output/run_manifest.json` |
| `output/agent_results.jsonl` | 180 records; each ends its trajectory with `"action": "stop"` and has all `validation_checks` set to `true` |
| `output/evaluation.json` | `split=test`, with 30 participants |
| `new_cohort/` | Prediction succeeds without outcome, target, or split files |
| Risk prediction `output/horizon_risks.npz` | `5y` and `10y` arrays with shape `(180, 1)`, matching `participant_ids` and `endpoints`, and a boolean `valid_mask`; valid probabilities in 0–1 and 10-year risk at least 5-year risk |

Open the generated CSVs directly to read probabilities; no conversion is needed. Empty cells mean unavailable predictions. See [how to read predictions](concepts.md#read-predictions) for output meanings and masks. Risk prediction CSVs contain event probabilities; the retained `final_scores.npy` contains Cox log-risk.

For an optional automated check after running both tasks, copy this block. It checks CSV headers, participant order, probabilities and agreement with the NumPy outputs, plus planning paths and test-only evaluation. It assumes the default demo directories; change `root` if you generated them elsewhere.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("data/demo")
for task in ("prevalent", "longitudinal"):
    for folder, count in (("output", 180), ("new_output", 6)):
        out = root / task / folder
        source = root / task / ("new_cohort" if folder == "new_output" else "")
        ids = np.load(out / "participant_ids.npy", allow_pickle=False)
        endpoints = np.load(out / "prediction_endpoints.npy", allow_pickle=False)
        scores = np.load(out / "final_scores.npy", allow_pickle=False)
        valid = np.load(out / "final_valid_mask.npy", allow_pickle=False)
        expected_ids = pd.read_csv(source / "participants.csv", dtype=str)["participant_id"].to_numpy()
        assert np.array_equal(ids, expected_ids)
        assert endpoints.tolist() == ["D1"]
        assert scores.shape == valid.shape == (count, 1)
        assert valid.dtype.kind == "b" and valid.all()
        assert np.isfinite(scores[valid]).all()

        manifest = json.loads((out / "run_manifest.json").read_text())
        used = np.load(out / "final_used_modules.npy", allow_pickle=False)
        assert used.shape == (count, len(manifest["modules"])) == (count, 2)
        rows = [json.loads(line) for line in (out / "agent_results.jsonl").read_text().splitlines()]
        assert len(rows) == count
        for i, row in enumerate(rows):
            assert str(row["participant_id"]) == str(ids[i])
            assert row["endpoint_order"] == endpoints.tolist()
            assert row["final_used_modules"] == used[i].tolist()
            assert row["selected_modules"] == [m for m, selected in zip(manifest["modules"], used[i]) if selected]
            assert row["trajectory"][-1]["action"] == "stop"
            assert all(row["validation_checks"].values())

        if task == "prevalent":
            assert ((scores[valid] >= 0) & (scores[valid] <= 1)).all()
            assert Path(manifest["outputs"]["predictions_csv"]).name == "predictions.csv"
            tables = {"predictions.csv": scores}
        else:
            with np.load(out / "horizon_risks.npz", allow_pickle=False) as risks:
                assert np.array_equal(risks["participant_ids"], ids)
                assert np.array_equal(risks["endpoints"], endpoints)
                assert np.array_equal(risks["valid_mask"], valid)
                for name in ("5y", "10y"):
                    assert risks[name].shape == scores.shape
                    assert np.isfinite(risks[name][valid]).all()
                    assert ((risks[name][valid] >= 0) & (risks[name][valid] <= 1)).all()
                assert (risks["10y"][valid] >= risks["5y"][valid]).all()
                tables = {f"predictions_{name}.csv": risks[name] for name in ("5y", "10y")}
            paths = manifest["outputs"]["horizon_predictions_csv"]
            assert set(paths) == {"5y", "10y"}
            assert all(Path(path).name == f"predictions_{name}.csv" for name, path in paths.items())
        for filename, values in tables.items():
            table = pd.read_csv(out / filename, dtype={"participant_id": str})
            assert table.columns.tolist() == ["participant_id", *endpoints.tolist()]
            assert table["participant_id"].tolist() == ids.tolist()
            np.testing.assert_allclose(
                table.iloc[:, 1:].to_numpy(), np.where(valid, values, np.nan),
                rtol=1e-12, atol=1e-15, equal_nan=True,
            )
        print(task, folder, "OK:", scores.shape)

    report = json.loads((root / task / "output/evaluation.json").read_text())
    assert report["split"] == "test" and report["participants"] == 30
    assert not (root / task / "new_cohort/outcome.csv").exists()
PY
```

Expected output is one `OK` line for each task and output folder. The checks use dimensions, identities, masks and workflow behavior; they do not require a particular metric value or selected path.

## What this verifies

The two fictional modules and endpoint exercise the estimation and planning path. The new cohort checks that frozen scoring and prediction do not require outcome labels.

The examples use three boosting rounds and disable consistency. Their outputs do not include frozen binary classifications, `consistency_predictions.npz`, or evidence threshold diagnostics. For consistency checks, prepare the aligned contribution and evidence inputs described in the [consistency workflow](real_data.md#check-consistency).

The examples use synthetic inputs generated by `examples/prepare_demo.py`. Generated files are excluded from Git under `data/`.
