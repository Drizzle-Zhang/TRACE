from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.consistency import evidence
from trace_agent.knowledge.evidence_store import MedicalEvidenceStore

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/fixtures/migration_evidence_reference.json").read_text())


def test_evidence_reference_records_frozen_source_identity() -> None:
    manifest = json.loads((ROOT / "docs/provenance/source_fingerprints.json").read_text())
    source = REFERENCE["source"]
    original = next(entry for entry in manifest["files"] if entry["path"] == source["path"])
    assert source["sha256"] == original["sha256"]
    assert source["matches_frozen_manifest"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_compiled_rule_evidence_matches_frozen_outputs(tmp_path: Path, case: dict) -> None:
    for filename, rows in (
        ("risk_factor_evidence.csv", case["rules"]),
        ("disease_evidence_feature_map.csv", case["mappings"]),
    ):
        with (tmp_path / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    compiled = MedicalEvidenceStore(tmp_path).compile(endpoints=("D1",), features=tuple(case["features"]))
    assert hasattr(compiled, "rules"), "Rule identities must survive endpoint/feature aggregation"
    np.testing.assert_allclose(compiled.rules.weights, case["expected_rule_weights"])
    np.testing.assert_allclose(compiled.rules.reliability, case["expected_rule_reliability"])

    result = evidence.compute_rule_evidence_consistency(
        contributions=np.asarray(case["contributions"], dtype=np.float32), rules=compiled.rules
    )

    np.testing.assert_allclose(result.scores, case["expected_scores"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.support_scores, case["expected_support"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.no_conflict_scores, case["expected_no_conflict"], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.matched_counts, case["expected_matches"])


def test_direct_weight_arrays_keep_api_and_normalize_all_finite_contributions() -> None:
    case = REFERENCE["cases"][0]
    result = evidence.compute_evidence_consistency(
        contributions=np.asarray(case["contributions"], dtype=np.float32),
        evidence_weights=np.array([[12.0, 0.0]]),
        reliability=np.array([[1.0, 0.0]]),
    )

    np.testing.assert_allclose(result.scores, case["expected_scores"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.support_scores, case["expected_support"], rtol=1e-6, atol=1e-7)
