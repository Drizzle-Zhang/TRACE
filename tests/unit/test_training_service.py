from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trace_agent.runtime.model_artifact import load_trace_artifact
from trace_agent.training.service import (
    calibrate,
    dataset_from_config,
    train_integration,
    train_planner,
)


def _write_training_bundle(root: Path, *, task: str) -> dict[str, object]:
    root.mkdir()
    prefix = "cohort"
    eids = np.arange(100, 109)
    labels = np.array(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
        dtype=np.float32,
    )
    scores = np.empty((9, 2, 1), dtype=np.float32)
    scores[:, 0, 0] = 0.2 + 0.6 * labels[:, 0]
    scores[:, 1, 0] = 0.3 + 0.4 * labels[:, 0]
    np.save(root / f"{prefix}_eids.npy", eids)
    np.save(root / f"{prefix}_modules.npy", np.array(["M1", "M2"]))
    np.save(root / f"{prefix}_diseases.npy", np.array(["D1"]))
    np.save(root / f"{prefix}_pred_mask.npy", np.ones_like(scores, dtype=bool))
    np.save(root / f"{prefix}_avail.npy", np.ones((9, 2), dtype=bool))
    if task == "prevalent":
        np.save(root / f"{prefix}_oof_probs.npy", scores)
        np.save(root / f"{prefix}_y.npy", labels)
        np.save(root / f"{prefix}_y_mask.npy", np.ones_like(labels, dtype=bool))
    else:
        times = np.array(
            [[12], [3], [11], [7], [15], [4], [13], [9], [14]],
            dtype=np.float32,
        )
        np.save(root / f"{prefix}_scores.npy", scores)
        np.save(root / f"{prefix}_event.npy", labels)
        np.save(root / f"{prefix}_time.npy", times)
        np.save(root / f"{prefix}_signed_time.npy", np.where(labels > 0.5, times, -times))
        np.save(root / f"{prefix}_label_mask.npy", np.ones_like(labels, dtype=bool))

    split_path = root / "splits.csv"
    split_path.write_text(
        "row,eid,partition\n"
        + "\n".join(
            f"{row},{eid},{'train' if row < 5 else 'validation' if row < 7 else 'test'}"
            for row, eid in enumerate(eids)
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(root / "kg.npy", np.ones((2, 1), dtype=np.float32))
    (root / "burden.yaml").write_text("M1: 1\nM2: 2\n", encoding="utf-8")
    output = root / "output"
    return {
        "version": 1,
        "task": task,
        "data": {
            "input_dir": str(root),
            "prefix": prefix,
            "split_table": {
                "path": str(split_path),
                "row_index_column": "row",
                "eid_column": "eid",
                "split_column": "partition",
            },
        },
        "training": {
            "integration": {
                "seed": 3,
                "subsets_per_participant": 1,
                "n_estimators": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
                **(
                    {"min_class_count": 10, "num_leaves": 3}
                    if task == "prevalent"
                    else {"min_events": 10, "max_depth": 2, "horizons": [5.0, 10.0]}
                ),
            },
            "calibration": {"min_class_count": 10},
            "planner": {
                "seed": 3,
                "states_per_participant": 1,
                "candidate_actions": 2,
                "n_estimators": 4,
                "learning_rate": 0.1,
                "num_leaves": 3,
                "min_child_samples": 1,
                "n_jobs": 1,
            },
            "longitudinal_reward": {
                "horizons": [5.0, 10.0],
                "horizon_weights": [0.5, 0.5],
            },
        },
        "artifacts": {
            "fusion_model": str(output / "fusion_model.joblib"),
            "calibrators": str(output / "calibration_model.joblib"),
            "q_model": str(output / "q_model.joblib"),
            "knowledge_matrix": str(root / "kg.npy"),
            "burden_profile": str(root / "burden.yaml"),
        },
        "output": {"directory": str(output)},
        "_meta": {"sha256": "f" * 64},
    }


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_train_integration_writes_native_artifact_and_manifest(
    tmp_path: Path,
    task: str,
) -> None:
    config = _write_training_bundle(tmp_path / task, task=task)

    summary = train_integration(config)

    artifact = load_trace_artifact(
        summary.artifact_path,
        component="integration",
        task=task,
    )
    assert artifact.task.value == task
    assert artifact.provenance["split_table_canonical_sha256"]
    assert summary.manifest_path.is_file()
    assert summary.train_rows == 5
    assert summary.validation_rows == 2
    assert summary.test_rows_used == 0


def test_dataset_from_config_rejects_row_eid_mismatch(tmp_path: Path) -> None:
    config = _write_training_bundle(tmp_path / "mismatch", task="prevalent")
    split_path = Path(config["data"]["split_table"]["path"])
    text = split_path.read_text(encoding="utf-8").replace("0,100,train", "0,999,train")
    split_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="row_index=0.*999.*100"):
        dataset_from_config(config)


def test_prevalent_calibration_and_planner_write_native_artifacts(
    tmp_path: Path,
) -> None:
    config = _write_training_bundle(tmp_path / "prevalent", task="prevalent")
    train_integration(config)

    calibration = calibrate(config)
    planner = train_planner(config)

    calibration_artifact = load_trace_artifact(
        calibration.artifact_path,
        component="calibration",
        task="prevalent",
    )
    planner_artifact = load_trace_artifact(
        planner.artifact_path,
        component="planner",
        task="prevalent",
    )
    assert calibration.train_rows == 0
    assert calibration.validation_rows == 2
    assert calibration.test_rows_used == 0
    assert planner.train_rows == 5
    assert planner.validation_rows == 2
    assert planner.test_rows_used == 0
    assert calibration_artifact.modules == planner_artifact.modules == ("M1", "M2")


def test_longitudinal_planner_writes_native_artifact(tmp_path: Path) -> None:
    config = _write_training_bundle(tmp_path / "longitudinal", task="longitudinal")
    train_integration(config)

    summary = train_planner(config)

    artifact = load_trace_artifact(
        summary.artifact_path,
        component="planner",
        task="longitudinal",
    )
    assert artifact.provenance["splits"]["test"]["count"] == 2
    assert summary.test_rows_used == 0
