from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.module_fixtures import write_module_inputs


def test_features_preserve_selected_order_and_missing_rows(tmp_path):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "features.csv")
    frame.loc[0, ["a", "b"]] = np.nan
    frame.to_csv(tmp_path / "features.csv", index=False)
    features = load_module_features(load_catalog(path))
    assert features.feature_names[("M1", "D1")] == ("b", "a")
    matrix = features.matrices[("M1", "D1")]
    assert matrix.shape == (180, 2)
    assert np.isnan(matrix[0]).all()
    assert not features.availability[0, 0]
    assert features.availability[0, 1]
    assert features.participant_ids.size == 180
    np.testing.assert_allclose(matrix[1], frame.loc[1, ["b", "a"]].to_numpy(dtype=float))


def test_feature_tables_align_independently_to_participant_order(tmp_path):
    from trace_agent.data.module_features import load_catalog, load_module_features

    catalog = load_catalog(write_module_inputs(tmp_path))
    participants = pd.read_csv(tmp_path / "participants.csv").sample(frac=1, random_state=19)
    participants.to_csv(tmp_path / "participants.csv", index=False)
    frame = pd.read_csv(tmp_path / "features.csv")
    frame.loc[0, ["a", "b"]] = np.nan
    frame.loc[1, "c"] = np.nan
    frame.loc[:, ["c", "eid", "b", "a"]].sample(frac=1, random_state=7).to_csv(
        tmp_path / "features.csv", index=False
    )
    second = frame.loc[:, ["c", "eid"]].rename(columns={"eid": "participant"})
    second["c"] += 10
    second.sample(frac=1, random_state=11).to_csv(tmp_path / "second.csv", index=False)
    catalog = replace(
        catalog,
        feature_tables=(
            catalog.feature_tables[0],
            replace(
                catalog.feature_tables[0], id="second", path=tmp_path / "second.csv", eid_column="participant"
            ),
        ),
        modules=(catalog.modules[0], replace(catalog.modules[1], feature_table="second")),
    )
    (tmp_path / "outcome.csv").unlink()

    features = load_module_features(catalog)

    expected = frame.set_index("eid").loc[participants["eid"]]
    np.testing.assert_array_equal(features.participant_ids, participants["eid"].astype(str))
    assert features.feature_names[("M1", "D1")] == ("b", "a")
    assert features.feature_names[("M2", "D1")] == ("c",)
    np.testing.assert_allclose(features.matrices[("M1", "D1")], expected[["b", "a"]], equal_nan=True)
    np.testing.assert_allclose(features.matrices[("M2", "D1")], expected[["c"]] + 10, equal_nan=True)
    np.testing.assert_array_equal(
        features.availability,
        np.column_stack((expected[["b", "a"]].notna().any(axis=1), expected["c"].notna())),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("drop", "feature participant ID mismatch.*missing 1.*extra 0"),
        ("duplicate", "duplicate feature participant ID"),
        ("unknown", "feature participant ID mismatch.*missing 1.*extra 1"),
        ("extra", "feature participant ID mismatch.*missing 0.*extra 1"),
        ("missing", "missing participant ID"),
    ],
)
def test_features_reject_invalid_eids(tmp_path, change, message):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "features.csv")
    if change == "drop":
        frame = frame.iloc[1:]
    elif change == "duplicate":
        frame.loc[0, "eid"] = frame.loc[1, "eid"]
    elif change == "extra":
        frame = pd.concat((frame, frame.iloc[[0]].assign(eid=999999)), ignore_index=True)
    elif change == "missing":
        frame.loc[0, "eid"] = np.nan
    else:
        frame.loc[0, "eid"] = 999999
    frame.to_csv(tmp_path / "features.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_module_features(load_catalog(path))


@pytest.mark.parametrize("value", [np.nan, 1001])
def test_participants_reject_missing_or_duplicate_eids(tmp_path, value):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "participants.csv").astype(float)
    frame.loc[0, "eid"] = value
    frame.to_csv(tmp_path / "participants.csv", index=False)
    with pytest.raises(ValueError, match="missing participant ID|unique"):
        load_module_features(load_catalog(path))


@pytest.mark.parametrize(
    ("names", "message"),
    [(["b", "b"], "unique"), ([], "empty"), (["b", None], "missing names"), (["unknown"], "missing")],
)
def test_feature_lists_reject_invalid_names(tmp_path, names, message):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    pd.DataFrame({"feature_code": names}).to_csv(tmp_path / "M1_D1.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_module_features(load_catalog(path))


@pytest.mark.parametrize("value", [np.inf, -np.inf, "not_numeric"])
def test_selected_features_reject_invalid_values(tmp_path, value):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "features.csv").astype(object)
    frame.loc[0, "b"] = value
    frame.to_csv(tmp_path / "features.csv", index=False)
    with pytest.raises(ValueError):
        load_module_features(load_catalog(path))


def test_configured_feature_file_missing_is_not_an_omitted_pair(tmp_path):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    (tmp_path / "M1_D1.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_module_features(load_catalog(path))


def test_omitted_pairs_keep_named_axes_and_module_unavailability(tmp_path):
    from trace_agent.data.module_features import load_catalog, load_module_features

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["pairs"] = [values["pairs"][0]]
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    (tmp_path / "outcome.csv").unlink()
    features = load_module_features(load_catalog(path))
    assert features.availability.shape == (180, 2)
    assert not features.availability[:, 1].any()
    assert set(features.matrices) == {("M1", "D1")}
