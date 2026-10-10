from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from trace_agent.cli.main import main
from trace_agent.runtime.model_artifact import load_trace_artifact


def _write_pipeline(root: Path, *, task: str) -> Path:
    data = root / "data"
    data.mkdir()
    prefix = "cohort"
    eids = np.arange(300, 309)
    labels = np.array(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
        dtype=np.float32,
    )
    scores = np.empty((9, 2, 1), dtype=np.float32)
    scores[:, 0, 0] = 0.15 + 0.7 * labels[:, 0]
    scores[:, 1, 0] = 0.25 + 0.5 * labels[:, 0]
    np.save(data / f"{prefix}_eids.npy", eids)
    np.save(data / f"{prefix}_modules.npy", np.array(["M1", "M2"]))
    np.save(data / f"{prefix}_diseases.npy", np.array(["D1"]))
    np.save(data / f"{prefix}_pred_mask.npy", np.ones_like(scores, dtype=bool))
    np.save(data / f"{prefix}_avail.npy", np.ones((9, 2), dtype=bool))
    if task == "prevalent":
        np.save(data / f"{prefix}_oof_probs.npy", scores)
        np.save(data / f"{prefix}_y.npy", labels)
        np.save(data / f"{prefix}_y_mask.npy", np.ones_like(labels, dtype=bool))
    else:
        times = np.array(
            [[12], [3], [11], [7], [15], [4], [13], [9], [14]],
            dtype=np.float32,
        )
        np.save(data / f"{prefix}_scores.npy", scores)
        np.save(data / f"{prefix}_event.npy", labels)
        np.save(data / f"{prefix}_time.npy", times)
        np.save(data / f"{prefix}_signed_time.npy", np.where(labels > 0.5, times, -times))
        np.save(data / f"{prefix}_label_mask.npy", np.ones_like(labels, dtype=bool))
    (data / "splits.csv").write_text(
        "row_index,eid,split\n"
        + "\n".join(
            f"{row},{eid},{'train' if row < 5 else 'validation' if row < 7 else 'test'}"
            for row, eid in enumerate(eids)
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(data / "kg.npy", np.ones((2, 1), dtype=np.float32))
    (data / "burden.yaml").write_text("M1: 1\nM2: 2\n", encoding="utf-8")
    integration = {
        "seed": 2,
        "subsets_per_participant": 1,
        "n_estimators": 3,
        "learning_rate": 0.1,
        "n_jobs": 1,
        **(
            {"min_class_count": 10, "num_leaves": 3}
            if task == "prevalent"
            else {"min_events": 10, "max_depth": 2, "horizons": [5.0, 10.0]}
        ),
    }
    training: dict[str, object] = {
        "integration": integration,
        "planner": {
            "seed": 2,
            "states_per_participant": 1,
            "candidate_actions": 2,
            "n_estimators": 3,
            "learning_rate": 0.1,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    }
    if task == "prevalent":
        training["calibration"] = {"min_class_count": 10}
    else:
        training["longitudinal_reward"] = {
            "horizons": [5.0, 10.0],
            "horizon_weights": [0.5, 0.5],
        }
    steps = ["prepare-data", "train-integration"]
    if task == "prevalent":
        steps.append("calibrate")
    steps.append("train-planner")
    config = {
        "version": 1,
        "task": task,
        "data": {
            "input_dir": "./data",
            "prefix": prefix,
            "split_table": {"path": "./data/splits.csv"},
        },
        "training": training,
        "artifacts": {
            "fusion_model": "./output/fusion_model.joblib",
            "calibrators": "./output/calibration_model.joblib",
            "q_model": "./output/q_model.joblib",
            "knowledge_matrix": "./data/kg.npy",
            "burden_profile": "./data/burden.yaml",
        },
        "output": {"directory": "./output"},
        "pipeline": {"steps": steps},
    }
    path = root / "train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_cli_prevalent_training_pipeline_uses_shared_split_table(
    tmp_path: Path,
) -> None:
    config_path = _write_pipeline(tmp_path, task="prevalent")

    assert main(["run", "--config", str(config_path)]) == 0

    for name, component in (
        ("fusion_model.joblib", "integration"),
        ("calibration_model.joblib", "calibration"),
        ("q_model.joblib", "planner"),
    ):
        artifact = load_trace_artifact(
            tmp_path / "output" / name,
            component=component,
            task="prevalent",
        )
        assert artifact.provenance["split_table_canonical_sha256"]
        assert artifact.training["test_rows_used"] == 0
        assert artifact.model.__class__.__module__.startswith("trace_agent.")


def test_cli_longitudinal_training_pipeline_uses_shared_split_table(
    tmp_path: Path,
) -> None:
    pytest.importorskip("xgboost")
    config_path = _write_pipeline(tmp_path, task="longitudinal")

    assert main(["run", "--config", str(config_path)]) == 0

    fusion = load_trace_artifact(
        tmp_path / "output" / "fusion_model.joblib",
        component="integration",
        task="longitudinal",
    )
    planner = load_trace_artifact(
        tmp_path / "output" / "q_model.joblib",
        component="planner",
        task="longitudinal",
    )
    assert fusion.provenance["split_table_canonical_sha256"]
    assert (
        planner.provenance["split_table_canonical_sha256"]
        == fusion.provenance["split_table_canonical_sha256"]
    )
    assert planner.training["test_rows_used"] == 0
    assert fusion.model.__class__.__module__.startswith("trace_agent.")
    assert planner.model.__class__.__module__.startswith("trace_agent.")
