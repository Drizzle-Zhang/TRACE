"""Text participant IDs must survive all tabular participant-identity boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.module_fixtures import write_module_inputs
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.data.module_outcomes import load_module_outcomes
from trace_agent.data.split_table import SplitColumns, load_split_table
from trace_agent.training.modules.folds import build_outer_folds


@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t"), (".tab", "\t")])
def test_split_table_preserves_custom_string_eids(tmp_path, suffix, separator):
    ids = np.array(["001", "002", "NA"])
    path = tmp_path / f"split{suffix}"
    pd.DataFrame({"row": [0, 1, 2], "participant": ids, "role": ["train", "val", "test"]}).to_csv(
        path, sep=separator, index=False
    )
    result = load_split_table(path, participant_ids=ids, columns=SplitColumns("row", "participant", "role"))
    assert result.eids("train").tolist() == ["001"]
    assert result.eids("test").tolist() == ["NA"]


@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t")])
def test_external_folds_preserve_custom_string_eids(tmp_path, suffix, separator):
    ids = np.array(["001", "002", "003", "004", "005"])
    path = tmp_path / f"folds{suffix}"
    pd.DataFrame({"row": range(5), "participant": ids, "iv_fold": range(1, 6)}).to_csv(
        path, sep=separator, index=False
    )
    result = build_outer_folds(
        ids, np.arange(5), path=path, row_column="row", eid_column="participant", fold_column="iv_fold"
    )
    assert result.tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t")])
def test_raw_tables_preserve_ids_and_numeric_missing_features(tmp_path, suffix, separator):
    catalog = load_catalog(write_module_inputs(tmp_path))
    ids = np.array(["001", "1", "NA", *[f"0{value}" for value in range(1003, 1180)]])
    tables = []
    for original in ("participants", "features", "outcome"):
        frame = pd.read_csv(tmp_path / f"{original}.csv")
        frame["eid"] = ids
        frame = frame.rename(columns={"eid": "participant"})
        if original == "features":
            frame.loc[0, ["a", "b"]] = np.nan
            frame = frame.iloc[::-1]
        elif original == "outcome":
            frame = frame.sample(frac=1, random_state=17)
        target = tmp_path / f"{original}_strings{suffix}"
        frame.to_csv(target, sep=separator, index=False)
        tables.append(target)
    catalog = replace(
        catalog,
        participants_path=tables[0],
        eid_column="participant",
        feature_tables=(replace(catalog.feature_tables[0], path=tables[1], eid_column="participant"),),
        endpoints=(replace(catalog.endpoints[0], outcome_path=tables[2], eid_column="participant"),),
    )
    features = load_module_features(catalog)
    np.testing.assert_array_equal(features.participant_ids, ids)
    assert np.isnan(features.matrices[("M1", "D1")][0]).all()
    targets = load_module_outcomes(catalog, ids, task="prevalent")
    assert targets.labels[:, 0].tolist() == (np.arange(180) % 2).tolist()


def test_padded_participants_cannot_match_unpadded_features(tmp_path):
    catalog = load_catalog(write_module_inputs(tmp_path))
    frame = pd.read_csv(catalog.participants_path)
    frame["eid"] = frame["eid"].map(lambda value: f"0{value}")
    frame.to_csv(catalog.participants_path, index=False)
    with pytest.raises(ValueError, match="feature participant ID mismatch"):
        load_module_features(catalog)


def test_outcomes_do_not_merge_padded_and_unpadded_participants(tmp_path):
    catalog = load_catalog(write_module_inputs(tmp_path))
    frame = pd.read_csv(catalog.endpoints[0].outcome_path)
    frame["eid"] = frame["eid"].map(lambda value: f"0{value}")
    frame.to_csv(catalog.endpoints[0].outcome_path, index=False)
    with pytest.raises(ValueError, match="unknown outcome participant ID"):
        load_module_outcomes(catalog, np.arange(1000, 1180), task="prevalent")
