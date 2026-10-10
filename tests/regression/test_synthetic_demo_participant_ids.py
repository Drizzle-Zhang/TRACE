"""Synthetic beginner inputs use the public participant identifier contract."""

import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_synthetic_demo_uses_participant_id_columns(tmp_path, task):
    script = Path(__file__).resolve().parents[2] / "examples/prepare_demo.py"
    prepare_demo = runpy.run_path(str(script))["prepare_demo"]
    root = tmp_path / task
    prepare_demo(root, task=task)

    for cohort, names in (
        (root, ("participants.csv", "features.csv", "outcome.csv", "splits.csv")),
        (root / "new_cohort", ("participants.csv", "features.csv")),
    ):
        expected_ids = pd.read_csv(cohort / "participants.csv", dtype=str)["participant_id"].tolist()
        assert len(expected_ids) == len(set(expected_ids))
        for name in names:
            table = pd.read_csv(cohort / name, dtype=str)
            assert "eid" not in table.columns
            assert table["participant_id"].tolist() == expected_ids
        catalog = yaml.safe_load((cohort / "catalog.yaml").read_text(encoding="utf-8"))
        assert catalog["participants"]["participant_id_column"] == "participant_id"
        assert catalog["feature_tables"][0]["participant_id_column"] == "participant_id"
