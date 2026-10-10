"""Write synthetic inputs and YAML workflows for the beginner guide.

Run from a source checkout after installing trace-agent. This script reads no
participant data and fits no models; use the printed CLI commands afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_yaml(path: Path, values: dict) -> None:
    path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_catalog(root: Path, *, features_path: str, participants_path: str) -> None:
    write_yaml(
        root / "catalog.yaml",
        {
            "version": 1,
            "participants": {"path": participants_path, "eid_column": "eid"},
            "feature_tables": [{"id": "demo", "path": features_path, "eid_column": "eid"}],
            "modules": [
                {"name": "M1", "feature_table": "demo"},
                {"name": "M2", "feature_table": "demo"},
            ],
            "endpoints": [{"name": "D1", "outcome_path": "outcome.csv"}],
            "pairs": [
                {"module": "M1", "endpoint": "D1", "features_path": "M1_D1.csv"},
                {"module": "M2", "endpoint": "D1", "features_path": "M2_D1.csv"},
            ],
        },
    )


def prepare_demo(root: Path, *, task: str) -> None:
    root.mkdir(parents=True)
    n = 180
    eids = np.array([f"demo_{i:04d}" for i in range(n)])
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
    if task == "longitudinal":
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
    write_catalog(root, features_path="features.csv", participants_path="participants.csv")
    np.save(root / "kg.npy", np.ones((2, 1), dtype=np.float32))
    (root / "burden.yaml").write_text("M1: 1\nM2: 2\n", encoding="utf-8")

    artifacts = {
        "module_model_dir": str(root / "bank"),
        "fusion_model": str(root / "output/fusion_model.joblib"),
        "q_model": str(root / "output/q_model.joblib"),
        "knowledge_matrix": str(root / "kg.npy"),
        "burden_profile": str(root / "burden.yaml"),
    }
    training = {
        "modules": {
            "seed": 7,
            "min_positive": 5,
            "n_estimators": 3,
            "n_jobs": 1,
            "min_child_samples": 1,
            "early_stopping_rounds": 2,
        },
        "integration": {
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
        },
        "planner": {
            "seed": 7,
            "states_per_participant": 1,
            "candidate_actions": 2,
            "n_estimators": 3,
            "learning_rate": 0.1,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    }
    steps = ["prepare-data", "train-modules", "score-modules", "train-integration"]
    if task == "prevalent":
        artifacts["calibrators"] = str(root / "output/calibration_model.joblib")
        training["calibration"] = {"min_class_count": 2}
        steps.append("calibrate")
    else:
        training["longitudinal_reward"] = {
            "horizons": [5.0, 10.0],
            "horizon_weights": [0.5, 0.5],
        }
    planner = {"initial_modules": ["M1"], "max_steps": 2, "candidate_actions": 2}
    write_yaml(
        root / "train.yaml",
        {
            "task": task,
            "module_data": {
                "catalog_path": str(root / "catalog.yaml"),
                "input_mode": "modules",
                "score_mode": "training",
            },
            "data": {
                "input_dir": str(root / "prepared"),
                "prefix": "cohort",
                "split_table": {"path": str(root / "splits.csv")},
            },
            "training": training,
            "artifacts": artifacts,
            "planner": planner,
            "consistency": {"enabled": False},
            "output": {"directory": str(root / "output")},
            "pipeline": {"steps": steps + ["train-planner", "infer", "evaluate"]},
        },
    )

    new = root / "new_cohort"
    new.mkdir()
    new_eids = np.array([f"new_{i:04d}" for i in range(6)])
    pd.DataFrame({"eid": new_eids}).to_csv(new / "participants.csv", index=False)
    pd.DataFrame(
        {"eid": new_eids, "a": rng.normal(size=6), "b": rng.normal(size=6), "c": rng.normal(size=6)}
    ).to_csv(new / "features.csv", index=False)
    # The new catalog declares the required outcome path, but no outcome file
    # or split is created: score_mode=inference does not read either one.
    write_catalog(new, features_path="features.csv", participants_path="participants.csv")
    for name in ("M1_D1.csv", "M2_D1.csv"):
        (new / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
    write_yaml(
        root / "predict_new.yaml",
        {
            "task": task,
            "module_data": {"catalog_path": str(new / "catalog.yaml"), "score_mode": "inference"},
            "data": {"input_dir": str(root / "new_scores"), "prefix": "new"},
            "artifacts": artifacts,
            "planner": planner,
            "consistency": {"enabled": False},
            "output": {"directory": str(root / "new_output")},
            "pipeline": {"steps": ["score-modules", "infer"]},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TRACE beginner inputs and configurations without reading UKB data."
    )
    parser.add_argument("--task", choices=("prevalent", "longitudinal"), default="prevalent")
    parser.add_argument("--output", type=Path, help="New output directory; defaults to data/demo/<task>.")
    args = parser.parse_args()
    root = (args.output or Path("data/demo") / args.task).expanduser().resolve()
    if root.exists():
        parser.error(f"Output directory already exists: {root}; choose a new directory with --output.")
    prepare_demo(root, task=args.task)
    print(f"Synthetic demo generated: {root}")
    print("180 synthetic participants (train=120, validation=30, test=30), 2 modules, 1 synthetic endpoint.")
    print(f'Train, predict and evaluate: python -m trace_agent run --config "{root / "train.yaml"}"')
    print(f'Predict a new unlabeled cohort: python -m trace_agent run --config "{root / "predict_new.yaml"}"')


if __name__ == "__main__":
    main()
