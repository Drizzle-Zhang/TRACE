from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.module_fixtures import write_module_inputs
from trace_agent.data import split_table
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.data.module_outcomes import load_module_outcomes
from trace_agent.training.modules.folds import build_outer_folds
from trace_agent.training.modules.service import module_splits


@pytest.mark.parametrize("column", ["participant_id", "eid"])
@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t"), (".tab", "\t")])
def test_default_identifier_reader_preserves_text_and_numeric_missing_values(
    tmp_path, column, suffix, separator
):
    path = tmp_path / f"table{suffix}"
    path.write_text(
        separator.join([column, "value"])
        + "\n"
        + "\n".join(separator.join(row) for row in [("001", "1"), ("1", "NA"), ("NA", "3")])
        + "\n",
        encoding="utf-8",
    )

    frame = split_table.read_eid_table(path)

    assert frame["participant_id"].tolist() == ["001", "1", "NA"]
    assert frame["value"].iloc[0] == 1
    assert np.isnan(frame["value"].iloc[1])


def test_neutral_split_default_preserves_frozen_canonical_hash(tmp_path):
    path = tmp_path / "splits.csv"
    ids = np.array(["001", "1", "NA"])
    pd.DataFrame({"participant_id": ids, "split": ["train", "val", "test"]}).to_csv(path, index=False)

    splits = split_table.load_split_table(path, participant_ids=ids)

    canonical = b"row_index,eid,split\n0,001,train\n1,1,validation\n2,NA,test\n"
    assert splits.eids("train").tolist() == ["001"]
    assert splits.eids("validation").tolist() == ["1"]
    assert splits.eids("test").tolist() == ["NA"]
    assert splits.canonical_sha256 == hashlib.sha256(canonical).hexdigest()
    legacy_path = tmp_path / "legacy.csv"
    pd.DataFrame({"eid": ids, "split": ["train", "val", "test"]}).to_csv(legacy_path, index=False)
    assert (
        split_table.load_split_table(legacy_path, participant_ids=ids).canonical_sha256
        == splits.canonical_sha256
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, "participant_id"),
        ({"participant_id_column": "subject"}, "subject"),
        ({"eid_column": "legacy_subject"}, "legacy_subject"),
        ({"participant_id_column": "subject", "eid_column": "subject"}, "subject"),
    ],
)
def test_identifier_column_configuration_accepts_neutral_and_legacy_keys(config, expected):
    assert split_table.id_column_from_config(config) == expected


def test_identifier_column_configuration_rejects_conflicting_keys():
    with pytest.raises(ValueError, match="participant_id_column.*eid_column"):
        split_table.id_column_from_config({"participant_id_column": "subject", "eid_column": "eid"})


@pytest.mark.parametrize("column", ["participant_id", "eid"])
def test_catalog_defaults_align_neutral_and_legacy_participant_tables(tmp_path, column):
    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["participants"].pop("eid_column")
    values["feature_tables"][0].pop("eid_column")
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    ids = np.array(["001", "1", "NA", *[str(value) for value in range(1003, 1180)]])
    for name in ("participants", "features", "outcome"):
        source = tmp_path / f"{name}.csv"
        frame = pd.read_csv(source)
        frame["eid"] = ids
        frame = frame.rename(columns={"eid": column})
        if name != "participants":
            frame = frame.iloc[::-1]
        frame.to_csv(source, index=False)

    catalog = load_catalog(path)
    features = load_module_features(catalog)
    outcomes = load_module_outcomes(catalog, features.participant_ids, task="prevalent")

    assert catalog.eid_column == "participant_id"
    assert catalog.feature_tables[0].eid_column == "participant_id"
    assert catalog.endpoints[0].eid_column == "participant_id"
    np.testing.assert_array_equal(features.participant_ids, ids)
    np.testing.assert_array_equal(outcomes.labels[:, 0], np.arange(180) % 2)


def test_catalog_accepts_custom_neutral_identifier_configuration(tmp_path):
    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    records = [values["participants"], values["feature_tables"][0], values["endpoints"][0]]
    for record in records:
        record.pop("eid_column", None)
        record["participant_id_column"] = "subject"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    for name in ("participants", "features", "outcome"):
        source = tmp_path / f"{name}.csv"
        pd.read_csv(source).rename(columns={"eid": "subject"}).to_csv(source, index=False)

    catalog = load_catalog(path)
    features = load_module_features(catalog)

    assert (
        catalog.eid_column
        == catalog.feature_tables[0].eid_column
        == catalog.endpoints[0].eid_column
        == "subject"
    )
    assert features.participant_ids.tolist() == [str(value) for value in range(1000, 1180)]
    assert load_module_outcomes(catalog, features.participant_ids, task="prevalent").labels.shape == (180, 1)


@pytest.mark.parametrize("section", ["participants", "feature_tables", "endpoints"])
def test_catalog_rejects_conflicting_identifier_column_keys(tmp_path, section):
    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = values[section] if section == "participants" else values[section][0]
    record["participant_id_column"] = "subject"
    record["eid_column"] = "eid"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="participant_id_column.*eid_column"):
        load_catalog(path)


@pytest.mark.parametrize("column", ["participant_id", "eid"])
def test_fold_defaults_align_neutral_and_legacy_participant_ids(tmp_path, column):
    ids = np.array(["001", "1", "NA", "a", "b"])
    path = tmp_path / "folds.csv"
    pd.DataFrame({column: ids[::-1], "fold": [5, 4, 3, 2, 1]}).to_csv(path, index=False)

    assert build_outer_folds(ids, np.arange(5), path=path).tolist() == [1, 2, 3, 4, 5]


def test_module_split_configuration_accepts_neutral_identifier_key(tmp_path):
    path = tmp_path / "splits.csv"
    pd.DataFrame({"subject": ["001", "1", "NA"], "split": ["train", "val", "test"]}).to_csv(path, index=False)
    config = {"data": {"split_table": {"path": str(path), "participant_id_column": "subject"}}}

    splits = module_splits(config, np.array(["001", "1", "NA"]))

    assert splits.eids("test").tolist() == ["NA"]
