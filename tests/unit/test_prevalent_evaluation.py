"""The same named probability contract applies to every prevalent evaluation input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trace_agent.cli.operations import evaluate


def _config(root: Path, *, explicit: bool) -> dict[str, Any]:
    eids = np.arange(10, 16)
    labels = np.array([[0, 1], [1, 0]] * 3, dtype=np.float32)
    scores = np.full((6, 1, 2), 0.5, dtype=np.float32)
    for name, values in {
        "eids": eids,
        "modules": np.array(["M1"]),
        "diseases": np.array(["D1", "D2"]),
        "oof_probs": scores,
        "pred_mask": np.ones_like(scores, dtype=bool),
        "avail": np.ones((6, 1), dtype=bool),
        "y": labels,
        "y_mask": np.ones_like(labels, dtype=bool),
    }.items():
        np.save(root / f"cohort_{name}.npy", values)
    (root / "splits.csv").write_text(
        "row_index,eid,split\n"
        "0,10,train\n1,11,train\n2,12,validation\n3,13,validation\n"
        "4,14,test\n5,15,test\n",
        encoding="utf-8",
    )
    output = root / "output"
    output.mkdir()
    for name, values in {
        "final_scores": 0.1 + 0.8 * labels,
        "participant_ids": eids,
        "prediction_endpoints": np.array(["D1", "D2"]),
        "final_valid_mask": np.ones_like(labels, dtype=bool),
    }.items():
        np.save(output / f"{name}.npy", values)
    return {
        "task": "prevalent",
        "data": {
            "input_dir": str(root),
            "prefix": "cohort",
            "split_table": {"path": str(root / "splits.csv")},
        },
        "evaluation": (
            {
                "predictions": str(output / "final_scores.npy"),
                "participant_ids": str(output / "participant_ids.npy"),
            }
            if explicit
            else {}
        ),
        "output": {"directory": str(output)},
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_prevalent_evaluation_rejects_reordered_endpoint_columns(tmp_path: Path, explicit: bool) -> None:
    config = _config(tmp_path, explicit=explicit)
    np.save(tmp_path / "output/prediction_endpoints.npy", np.array(["D2", "D1"]))

    with pytest.raises(ValueError, match="endpoint.*order"):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("name", ["prediction_endpoints", "final_valid_mask"])
def test_prevalent_evaluation_requires_prediction_sidecars(tmp_path: Path, explicit: bool, name: str) -> None:
    config = _config(tmp_path, explicit=explicit)
    (tmp_path / f"output/{name}.npy").unlink()

    with pytest.raises(FileNotFoundError, match=name):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
def test_prevalent_evaluation_requires_a_boolean_prediction_mask(tmp_path: Path, explicit: bool) -> None:
    config = _config(tmp_path, explicit=explicit)
    np.save(tmp_path / "output/final_valid_mask.npy", np.ones((6, 2), dtype=np.int8))

    with pytest.raises(ValueError, match="valid.mask.*boolean"):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "name,values,error",
    [
        ("participant_ids", np.arange(10, 16)[None, :], "EIDs.*shape|EIDs.*order"),
        ("prediction_endpoints", np.array([["D1", "D2"]]), "endpoint.*shape|endpoint.*order"),
        ("final_valid_mask", np.ones((2, 6), dtype=bool), "valid.mask.*shape"),
    ],
)
def test_prevalent_evaluation_rejects_misaligned_prediction_metadata(
    tmp_path: Path, explicit: bool, name: str, values: np.ndarray, error: str
) -> None:
    config = _config(tmp_path, explicit=explicit)
    np.save(tmp_path / f"output/{name}.npy", values)

    with pytest.raises(ValueError, match=error):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("value", [-10.0, 10.0, np.nan, np.inf])
def test_prevalent_evaluation_rejects_invalid_valid_probabilities(
    tmp_path: Path, explicit: bool, value: float
) -> None:
    config = _config(tmp_path, explicit=explicit)
    path = tmp_path / "output/final_scores.npy"
    predictions = np.load(path)
    predictions[4, 0] = value
    np.save(path, predictions)

    with pytest.raises(ValueError, match="probabilities.*finite.*\\[0, 1\\]"):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
def test_prevalent_evaluation_checks_valid_probabilities_across_the_complete_cohort(
    tmp_path: Path, explicit: bool
) -> None:
    config = _config(tmp_path, explicit=explicit)
    path = tmp_path / "output/final_scores.npy"
    predictions = np.load(path)
    predictions[0, 0] = 10.0
    np.save(path, predictions)

    with pytest.raises(ValueError, match="probabilities.*finite.*\\[0, 1\\]"):
        evaluate(config)


@pytest.mark.parametrize("explicit", [False, True])
def test_prevalent_evaluation_excludes_masked_finite_predictions(tmp_path: Path, explicit: bool) -> None:
    config = _config(tmp_path, explicit=explicit)
    np.save(tmp_path / "output/final_valid_mask.npy", np.zeros((6, 2), dtype=bool))

    report = json.loads(evaluate(config).read_text(encoding="utf-8"))

    assert report["participants"] == 2
    assert report["n_valid"] == 0
    assert report["macro_auroc"] is None
    assert report["masked_log_loss"] is None


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("value", [-10.0, np.nan, np.inf])
def test_prevalent_evaluation_ignores_values_in_prediction_invalid_cells(
    tmp_path: Path, explicit: bool, value: float
) -> None:
    config = _config(tmp_path, explicit=explicit)
    output = tmp_path / "output"
    predictions = np.load(output / "final_scores.npy")
    predictions[4, 0] = value
    np.save(output / "final_scores.npy", predictions)
    valid = np.ones((6, 2), dtype=bool)
    valid[4, 0] = False
    np.save(output / "final_valid_mask.npy", valid)

    report = json.loads(evaluate(config).read_text(encoding="utf-8"))

    assert report["n_valid"] == 3
    assert [row["n"] for row in report["endpoints"]] == [1, 2]


def test_prevalent_evaluation_accepts_explicit_metadata_paths(tmp_path: Path) -> None:
    config = _config(tmp_path, explicit=True)
    output = tmp_path / "output"
    (output / "prediction_endpoints.npy").unlink()
    (output / "final_valid_mask.npy").unlink()
    np.save(tmp_path / "external_endpoints.npy", np.array(["D1", "D2"]))
    np.save(tmp_path / "external_valid.npy", np.zeros((6, 2), dtype=bool))
    config["evaluation"].update(
        endpoints_file=str(tmp_path / "external_endpoints.npy"),
        valid_mask_path=str(tmp_path / "external_valid.npy"),
    )

    report = json.loads(evaluate(config).read_text(encoding="utf-8"))

    assert report["n_valid"] == 0


def test_explicit_prevalent_predictions_use_their_own_directory_sidecars(tmp_path: Path) -> None:
    config = _config(tmp_path, explicit=True)
    output = tmp_path / "output"
    external = tmp_path / "external"
    external.mkdir()
    np.save(external / "probabilities.npy", np.load(output / "final_scores.npy"))
    np.save(external / "participant_ids.npy", np.arange(10, 16))
    np.save(external / "prediction_endpoints.npy", np.array(["D1", "D2"]))
    np.save(external / "final_valid_mask.npy", np.zeros((6, 2), dtype=bool))
    # A separate native run's metadata must not be paired with external predictions.
    np.save(output / "participant_ids.npy", np.arange(10, 16)[::-1])
    np.save(output / "prediction_endpoints.npy", np.array(["D2", "D1"]))
    config["evaluation"] = {"predictions": str(external / "probabilities.npy")}

    report = json.loads(evaluate(config).read_text(encoding="utf-8"))

    assert report["n_valid"] == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_prevalent_evaluation_accepts_probability_boundaries(tmp_path: Path, explicit: bool) -> None:
    config = _config(tmp_path, explicit=explicit)
    np.save(tmp_path / "output/final_scores.npy", np.load(tmp_path / "cohort_y.npy"))

    report = json.loads(evaluate(config).read_text(encoding="utf-8"))

    assert report["n_valid"] == 4
    assert report["macro_auroc"] == 1.0
    assert np.isfinite(report["masked_log_loss"])
