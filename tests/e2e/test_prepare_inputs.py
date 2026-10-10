from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trace_agent.cli.main import main
from trace_agent.data.module_features import load_catalog, load_module_features


def _config(tmp_path: Path) -> Path:
    (tmp_path / "m1_features.csv").write_text("feature_code\nage\nsex\n", encoding="utf-8")
    (tmp_path / "m2_features.csv").write_text("feature_code\nlab\n", encoding="utf-8")
    catalog = {
        "version": 1,
        "participants": {"path": "inputs/participants.csv", "eid_column": "person"},
        "feature_tables": [
            {"id": "t1", "path": "inputs/features/m1.csv", "eid_column": "id"},
            {"id": "t2", "path": "inputs/features/m2.tsv"},
        ],
        "modules": [{"name": "M1", "feature_table": "t1"}, {"name": "M2", "feature_table": "t2"}],
        "endpoints": [{"name": "D1", "outcome_path": "absent.csv"}],
        "pairs": [
            {"module": "M1", "endpoint": "D1", "features_path": "m1_features.csv"},
            {"module": "M2", "endpoint": "D1", "features_path": "m2_features.csv"},
        ],
    }
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
    config = tmp_path / "predict.yaml"
    config.write_text("module_data:\n  catalog_path: catalog.yaml\n", encoding="utf-8")
    return config


def test_prepare_inputs_builds_every_module_from_one_table(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    source = tmp_path / "cohort.tsv"
    source.write_text("subject\tage\tsex\tunused\n001\t58\t0\t8\nNA\t62\t\t9\n", encoding="utf-8")

    assert (
        main(
            [
                "prepare-inputs",
                "--config",
                str(config),
                "--input-table",
                str(source),
                "--eid-column",
                "subject",
            ]
        )
        == 0
    )

    features = load_module_features(load_catalog(tmp_path / "catalog.yaml"))
    assert features.participant_ids.tolist() == ["001", "NA"]
    np.testing.assert_allclose(features.matrices[("M1", "D1")], [[58, 0], [62, np.nan]], equal_nan=True)
    assert np.isnan(features.matrices[("M2", "D1")]).all()
    assert features.availability.tolist() == [[True, False], [True, False]]
    assert pd.read_csv(tmp_path / "inputs/features/m1.csv").columns.tolist() == ["id", "age", "sex"]
    report = capsys.readouterr().out
    assert '"participants": 2' in report
    assert '"missing_features": [' in report
    assert '"lab"' in report
    assert '"unused_columns": [' in report
    assert not (tmp_path / "absent.csv").exists()


@pytest.mark.parametrize(
    "rows", ["001,58\n001,62\n", ",58\n002,62\n", "001,inf\n002,62\n", "001,58 years\n002,62\n"]
)
def test_prepare_inputs_validates_before_replacing_any_table(tmp_path: Path, rows: str) -> None:
    config = _config(tmp_path)
    source = tmp_path / "cohort.csv"
    source.write_text("eid,age\n" + rows, encoding="utf-8")
    target = tmp_path / "inputs/participants.csv"
    target.parent.mkdir()
    target.write_text("person\noriginal\n", encoding="utf-8")

    with pytest.raises((ValueError, TypeError)):
        main(["prepare-inputs", "--config", str(config), "--input-table", str(source)])

    assert target.read_text(encoding="utf-8") == "person\noriginal\n"
    assert not (tmp_path / "inputs/features/m1.csv").exists()


def test_prepare_inputs_dry_run_does_not_open_or_write_data(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    assert (
        main(
            [
                "prepare-inputs",
                "--config",
                str(config),
                "--input-table",
                str(tmp_path / "absent.csv"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert "prepare-inputs" in capsys.readouterr().out
    assert not (tmp_path / "inputs").exists()


def test_prepare_inputs_respects_configured_source_and_id_column(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "cohort.csv"
    source.write_text("subject,age\n001,58\n", encoding="utf-8")
    config.write_text(
        "module_data:\n  catalog_path: catalog.yaml\n  input_table_path: cohort.csv\n"
        "  input_eid_column: subject\n",
        encoding="utf-8",
    )
    assert main(["prepare-inputs", "--config", str(config)]) == 0
    assert load_module_features(load_catalog(tmp_path / "catalog.yaml")).participant_ids.tolist() == ["001"]


@pytest.mark.parametrize("protected_name", ["m1_features.csv", "absent.csv"])
def test_prepare_inputs_does_not_replace_feature_lists_or_outcomes(
    tmp_path: Path, protected_name: str
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "cohort.csv"
    source.write_text("eid,age\n001,58\n", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["feature_tables"][0]["path"] = protected_name
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    protected = tmp_path / protected_name
    if not protected.exists():
        protected.write_text("eid,target_y\n001,1\n", encoding="utf-8")
    before = protected.read_bytes()
    with pytest.raises(ValueError, match="paths.*distinct|overlap"):
        main(["prepare-inputs", "--config", str(config), "--input-table", str(source)])
    assert protected.read_bytes() == before
    assert not (tmp_path / "inputs/participants.csv").exists()


@pytest.mark.parametrize("source_column", ["participant_id", "subject_code"])
def test_prepare_inputs_accepts_neutral_participant_ids(tmp_path: Path, source_column: str) -> None:
    config = _config(tmp_path)
    path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog["participants"] = {"path": "inputs/participants.csv", "participant_id_column": "participant_id"}
    for table in catalog["feature_tables"]:
        table.pop("eid_column", None)
    path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    source = tmp_path / "cohort.csv"
    source.write_text(f"{source_column},age,sex\n001,58,0\nNA,62,\n", encoding="utf-8")
    arguments = ["prepare-inputs", "--config", str(config), "--input-table", str(source)]
    if source_column != "participant_id":
        arguments.extend(["--id-column", source_column])

    assert main(arguments) == 0
    features = load_module_features(load_catalog(path))
    assert features.participant_ids.tolist() == ["001", "NA"]
    assert pd.read_csv(tmp_path / "inputs/participants.csv").columns.tolist() == ["participant_id"]
    assert pd.read_csv(tmp_path / "inputs/features/m1.csv").columns.tolist() == [
        "participant_id",
        "age",
        "sex",
    ]
    np.testing.assert_allclose(features.matrices[("M1", "D1")], [[58, 0], [62, np.nan]], equal_nan=True)
    assert np.isnan(features.matrices[("M2", "D1")]).all()


def test_prepare_inputs_help_uses_neutral_identifier_option(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["prepare-inputs", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--id-column" in output
    assert "participant_id" in output
    assert "--eid-column" not in output
