"""CSV feature lists must select columns by their literal text identities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trace_agent.data.module_features import load_catalog, load_module_features


def _write_catalog(tmp_path: Path, columns: dict[str, list[int]], *, feature_column="feature_code"):
    pd.DataFrame({"eid": [1001, 1002]}).to_csv(tmp_path / "participants.csv", index=False)
    pd.DataFrame({"eid": [1001, 1002], **columns}).to_csv(tmp_path / "features.csv", index=False)
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "participants": {"path": "participants.csv"},
                "feature_tables": [{"id": "baseline", "path": "features.csv"}],
                "modules": [{"name": "M1", "feature_table": "baseline"}],
                "endpoints": [{"name": "D1", "outcome_path": "unused_outcomes.csv"}],
                "pairs": [
                    {
                        "module": "M1",
                        "endpoint": "D1",
                        "features_path": "selected.csv",
                        "feature_column": feature_column,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_catalog(path)


@pytest.mark.parametrize("feature_column", ["feature_code", "selected_code"])
def test_padded_feature_code_selects_its_exact_column(tmp_path, feature_column):
    catalog = _write_catalog(tmp_path, {"001": [11, 12], "1": [-11, -12]}, feature_column=feature_column)
    pd.DataFrame({feature_column: ["001"]}).to_csv(tmp_path / "selected.csv", index=False)

    features = load_module_features(catalog)

    assert features.feature_names[("M1", "D1")] == ("001",)
    np.testing.assert_array_equal(features.matrices[("M1", "D1")], [[11], [12]])


def test_numeric_feature_codes_keep_declared_order_and_spelling(tmp_path):
    catalog = _write_catalog(
        tmp_path,
        {
            "001": [11, 12],
            "1": [-11, -12],
            "2.00": [21, 22],
            "1.0": [31, 32],
            "1e2": [41, 42],
        },
    )
    pd.DataFrame({"feature_code": ["2.00", "001", "1.0", "1e2"]}).to_csv(
        tmp_path / "selected.csv", index=False
    )

    features = load_module_features(catalog)

    assert features.feature_names[("M1", "D1")] == ("2.00", "001", "1.0", "1e2")
    np.testing.assert_array_equal(features.matrices[("M1", "D1")], [[21, 11, 31, 41], [22, 12, 32, 42]])


@pytest.mark.parametrize("feature_code", ["NA", "null", "N/A", "NaN", "None", "NULL", "nan", "<NA>"])
def test_na_like_feature_codes_select_literal_columns(tmp_path, feature_code):
    catalog = _write_catalog(tmp_path, {feature_code: [11, 12], "other": [-11, -12]})
    pd.DataFrame({"feature_code": [feature_code]}).to_csv(tmp_path / "selected.csv", index=False)

    features = load_module_features(catalog)

    assert features.feature_names[("M1", "D1")] == (feature_code,)
    np.testing.assert_array_equal(features.matrices[("M1", "D1")], [[11], [12]])


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("feature_code\n", "feature list must not be empty"),
        ('feature_code\n""\n', "missing names"),
        ("feature_code,position\n001,0\n,1\n", "missing names"),
        ("feature_code\n001\n\n", "missing names"),
        ('feature_code\n"   "\n', "missing names|non-empty"),
        ("other\n001\n", "feature list missing column"),
    ],
)
def test_empty_or_missing_feature_names_are_rejected(tmp_path, contents, message):
    catalog = _write_catalog(tmp_path, {"001": [11, 12], "1": [-11, -12]})
    (tmp_path / "selected.csv").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_module_features(catalog)


@pytest.mark.parametrize("names", [["001", "001"], ["NA", "NA"], ["001", " 001 "]])
def test_duplicate_feature_names_are_rejected(tmp_path, names):
    catalog = _write_catalog(tmp_path, {"001": [11, 12], "1": [-11, -12], "NA": [21, 22]})
    pd.DataFrame({"feature_code": names}).to_csv(tmp_path / "selected.csv", index=False)

    with pytest.raises(ValueError, match="unique"):
        load_module_features(catalog)
