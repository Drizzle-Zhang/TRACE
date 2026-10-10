from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.module_fixtures import write_module_inputs


def test_catalog_resolves_paths_and_keeps_order(tmp_path: Path) -> None:
    from trace_agent.data.module_features import load_catalog

    catalog = load_catalog(write_module_inputs(tmp_path))
    assert catalog.participants_path == tmp_path / "participants.csv"
    assert tuple(item.name for item in catalog.modules) == ("M1", "M2")
    assert catalog.pairs[0].features_path == tmp_path / "M1_D1.csv"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("pairs", "module", "UNKNOWN", "unknown pair reference"),
        ("pairs", "endpoint", "UNKNOWN", "unknown pair reference"),
        ("modules", "feature_table", "UNKNOWN", "unknown feature table reference"),
        ("modules", "name", "", "non-empty"),
        ("pairs", "features_path", None, "non-empty"),
        ("pairs", "unrecognized", "x", "unknown fields"),
    ],
)
def test_catalog_rejects_invalid_records(tmp_path, section, field, value, message):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values[section][0][field] = value
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_catalog(path)


@pytest.mark.parametrize("section", ["feature_tables", "modules", "endpoints", "pairs"])
def test_catalog_rejects_duplicate_records(tmp_path, section):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values[section].append(values[section][0].copy())
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="unique|duplicate"):
        load_catalog(path)


def test_catalog_missing_required_field_has_location(tmp_path):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    del values["endpoints"][0]["outcome_path"]
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match=r"endpoints\[0\].*missing"):
        load_catalog(path)


def test_catalog_does_not_open_outcomes_and_allows_omitted_pairs(tmp_path):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    (tmp_path / "outcome.csv").unlink()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["pairs"] = []
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    catalog = load_catalog(path)
    assert len(catalog.modules) == 2
    assert catalog.pairs == ()


def test_catalog_environment_paths_are_resolved_from_catalog(tmp_path, monkeypatch):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["participants"]["path"] = "${TRACE_TEST_RAW_DIR}/participants.csv"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    monkeypatch.setenv("TRACE_TEST_RAW_DIR", str(tmp_path))
    assert load_catalog(path).participants_path == tmp_path / "participants.csv"


@pytest.mark.parametrize("version", [None, 2])
def test_catalog_rejects_unsupported_version(tmp_path, version):
    from trace_agent.data.module_features import load_catalog

    path = write_module_inputs(tmp_path)
    values = yaml.safe_load(path.read_text())
    values["version"] = version
    path.write_text(yaml.safe_dump(values))
    with pytest.raises((ValueError, TypeError)):
        load_catalog(path)
