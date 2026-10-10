"""Feature aliases accepted by the original matrix_input schema loader."""

import json

import numpy as np
import pytest

from trace_agent.runtime.artifacts import load_evidence_configuration


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"feature_codes": {"p21001": "Body mass index (BMI)"}}, {"p21001": "Body mass index (BMI)"}),
        (
            {"feature_codes": [{"feature_code": "p21001", "feature_name": "Body mass index (BMI)"}]},
            {"p21001": "Body mass index (BMI)"},
        ),
        (
            {"final_feature_names": [{"code": "p4079", "label": "Diastolic blood pressure"}]},
            {"p4079": "Diastolic blood pressure"},
        ),
        (
            {"features": [{"field": "p4080", "display_name": "Systolic blood pressure"}]},
            {"p4080": "Systolic blood pressure"},
        ),
        (
            {"features": {"p21001": "Body mass index (BMI)", "p31": "Sex"}},
            {"p21001": "Body mass index (BMI)", "p31": "Sex"},
        ),
        (
            {"modules": [{"name": "anthropometry", "feature_codes": ["p21001"], "features": ["BMI"]}]},
            {"p21001": "BMI"},
        ),
        (
            {
                "modules": [
                    {"module": "anthropometry", "final_feature_names": [{"id": "p21001", "name": "BMI"}]}
                ]
            },
            {"p21001": "BMI"},
        ),
        (
            {"modules": [{"id": "blood_pressure", "features": [{"code": "p4079", "label": "DBP"}]}]},
            {"p4079": "DBP"},
        ),
        (
            {
                "feature_codes": {"p21001": "Global BMI"},
                "final_feature_names": {"p21001": "Later global BMI"},
                "modules": [
                    {"name": "M1", "feature_codes": ["p21001", "p31"], "features": ["Module BMI", "Sex"]},
                    {"name": "M2", "features": [{"code": "p31", "name": "Later sex"}]},
                ],
            },
            {"p21001": "Global BMI", "p31": "Sex"},
        ),
        (
            {
                "feature_codes": ["p21001"],
                "modules": [{"name": "M1", "feature_codes": ["p21001"], "features": ["Module BMI"]}],
            },
            {"p21001": "p21001"},
        ),
        (
            {
                "feature_code_to_name": {"p21001": "Explicit global BMI"},
                "feature_names": ["p31"],
                "features": {"p21001": "Legacy global BMI"},
                "module_feature_code_to_name": {"M1": {"p21001": "Explicit module BMI", "p4079": "DBP"}},
                "modules": [{"code": "M2", "features": [{"code": "p4079", "name": "Later DBP"}]}],
            },
            {"p21001": "Explicit global BMI", "p31": "p31", "p4079": "DBP"},
        ),
    ],
)
def test_legacy_feature_schema_names_and_precedence(tmp_path, schema, expected):
    names_path = tmp_path / "names.npy"
    np.save(names_path, np.array(["p21001", "p31", "p4079", "p4080"]))
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    store, features, names = load_evidence_configuration(
        {
            "feature_names": str(names_path),
            "feature_schema_path": str(schema_path),
            "evidence_bundle": str(tmp_path),
        }
    )
    assert store.root == tmp_path
    assert features == ("p21001", "p31", "p4079", "p4080")
    assert names == expected


def test_inline_feature_names_override_legacy_schema(tmp_path):
    names_path = tmp_path / "names.npy"
    np.save(names_path, ["p21001"])
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"features": {"p21001": "Schema BMI"}}))
    _, _, names = load_evidence_configuration(
        {
            "feature_names": str(names_path),
            "feature_schema_path": str(schema_path),
            "evidence_bundle": str(tmp_path),
            "feature_code_to_name": {"p21001": "Inline BMI"},
        }
    )
    assert names == {"p21001": "Inline BMI"}
