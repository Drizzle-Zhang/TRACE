from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def module_config(root, *, task):
    return {
        "task": task,
        "module_data": {"catalog_path": str(root / "catalog.yaml"), "input_mode": "modules"},
        "data": {
            "input_dir": str(root / "prepared"),
            "prefix": "cohort",
            "split_table": {"path": str(root / "splits.csv")},
        },
        "training": {
            "modules": {
                "seed": 7,
                "min_positive": 5,
                "n_estimators": 3,
                "min_child_samples": 1,
                "early_stopping_rounds": 2,
            }
        },
        "artifacts": {"module_model_dir": str(root / "bank")},
        "output": {"directory": str(root / "output")},
    }


def write_module_inputs(root: Path, *, task: str = "prevalent") -> Path:
    """Create aligned synthetic data, never read private cohort files."""
    root.mkdir(parents=True, exist_ok=True)
    n = 180
    eids = np.arange(1000, 1000 + n)
    labels = (np.arange(n) % 2).astype(float)
    rng = np.random.default_rng(7)
    pd.DataFrame({"eid": eids}).to_csv(root / "participants.csv", index=False)
    pd.DataFrame(
        {
            "eid": eids,
            "a": rng.normal(size=n) + labels,
            "b": rng.normal(size=n),
            "c": rng.normal(size=n) - labels,
        }
    ).to_csv(root / "features.csv", index=False)
    time = np.where(labels == 1, -2.0, 12.0)
    if task != "prevalent":
        time = np.where(labels == 1, 2.0 + np.arange(n) % 7, 12.0)
    pd.DataFrame({"eid": eids, "target_y": labels, "BL2Target_yrs": time}).to_csv(
        root / "outcome.csv", index=False
    )
    pd.DataFrame({"feature_code": ["b", "a"]}).to_csv(root / "M1_D1.csv", index=False)
    pd.DataFrame({"feature_code": ["c"]}).to_csv(root / "M2_D1.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(n),
            "eid": eids,
            "split": ["train"] * 120 + ["validation"] * 30 + ["test"] * 30,
        }
    ).to_csv(root / "splits.csv", index=False)
    catalog = {
        "version": 1,
        "participants": {"path": "participants.csv", "eid_column": "eid"},
        "feature_tables": [{"id": "baseline", "path": "features.csv", "eid_column": "eid"}],
        "modules": [
            {"name": "M1", "feature_table": "baseline"},
            {"name": "M2", "feature_table": "baseline"},
        ],
        "endpoints": [{"name": "D1", "outcome_path": "outcome.csv"}],
        "pairs": [
            {"module": "M1", "endpoint": "D1", "features_path": "M1_D1.csv"},
            {"module": "M2", "endpoint": "D1", "features_path": "M2_D1.csv"},
        ],
    }
    path = root / "catalog.yaml"
    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def write_full_module_pipeline(root: Path, *, task: str) -> Path:
    write_module_inputs(root, task=task)
    config = module_config(root, task=task)
    np.save(root / "kg.npy", np.ones((2, 1), dtype=np.float32))
    (root / "burden.yaml").write_text("M1: 1\nM2: 2\n", encoding="utf-8")
    config["module_data"]["score_mode"] = "training"
    config["training"]["integration"] = {
        "seed": 7,
        "subsets_per_participant": 1,
        "n_estimators": 3,
        "learning_rate": 0.1,
        "n_jobs": 1,
        **(
            {"min_class_count": 2, "num_leaves": 3}
            if task == "prevalent"
            else {"min_events": 2, "max_depth": 2, "horizons": [5.0, 10.0]}
        ),
    }
    config["training"]["planner"] = {
        "seed": 7,
        "states_per_participant": 1,
        "candidate_actions": 2,
        "n_estimators": 3,
        "learning_rate": 0.1,
        "num_leaves": 3,
        "min_child_samples": 1,
        "n_jobs": 1,
    }
    config["artifacts"].update(
        {
            "fusion_model": str(root / "output/fusion_model.joblib"),
            "q_model": str(root / "output/q_model.joblib"),
            "knowledge_matrix": str(root / "kg.npy"),
            "burden_profile": str(root / "burden.yaml"),
        }
    )
    steps = ["prepare-data", "train-modules", "score-modules", "train-integration"]
    if task == "prevalent":
        config["training"]["calibration"] = {"min_class_count": 2}
        config["artifacts"]["calibrators"] = str(root / "output/calibration_model.joblib")
        steps.append("calibrate")
    else:
        config["training"]["longitudinal_reward"] = {
            "horizons": [5.0, 10.0],
            "horizon_weights": [0.5, 0.5],
        }
    steps += ["train-planner", "infer"]
    config["pipeline"] = {"steps": steps}
    config["planner"] = {"initial_modules": [], "max_steps": 2, "candidate_actions": 2}
    path = root / "full_train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
