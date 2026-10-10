"""Synthetic evidence scoring and threshold diagnostic reference cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.consistency.evidence import compute_rule_evidence_consistency
from trace_agent.knowledge.evidence_store import MedicalEvidenceStore


def _csv(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["statement_id"]
    with (root / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bundle(root: Path, risk: list[dict[str, object]]) -> None:
    _csv(root, "risk_factor_evidence.csv", risk)
    _csv(
        root,
        "disease_evidence_feature_map.csv",
        [{"factor": "smoking", "feature_code": "F", "weight": 1, "match_type": "manual"}],
    )


def _rule(**changes: object) -> dict[str, object]:
    return {
        "statement_id": "risk",
        "disease": "D",
        "factor": "smoking",
        "evidence_bucket": "risk_factor",
        "certainty_rank": 4,
        "strength_rank": 3,
        **changes,
    }


def _score(root: Path, **kwargs: object) -> float:
    compiled = MedicalEvidenceStore(root).compile(endpoints=("D",), features=("F",), **kwargs)
    return float(
        compute_rule_evidence_consistency(contributions=np.ones((1, 1, 1)), rules=compiled.rules).scores[0, 0]
    )


def test_consensus_risk_rules_are_scored_with_text_rank_and_direction(tmp_path: Path) -> None:
    _bundle(tmp_path, [])
    _csv(
        tmp_path,
        "consensus_evidence.csv",
        [
            _rule(
                statement_id="",
                consensus_id="consensus",
                certainty_rank="",
                strength_rank="",
                max_certainty="high",
                final_strength="strong",
                consensus_stance="against",
            )
        ],
    )
    compiled = MedicalEvidenceStore(tmp_path).compile(endpoints=("D",), features=("F",))
    assert compiled.rules.weights.tolist() == [-12.0]
    assert _score(tmp_path) == pytest.approx(0.18242553)


def test_risk_rule_consensus_stance_preserves_protective_sign(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule(consensus_stance="against")])
    assert _score(tmp_path) == pytest.approx(0.18242553)


def test_outcome_longname_and_compressed_aliases_resolve_to_endpoint(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule(disease="Compressed Disease")])
    _csv(tmp_path, "outcome_filter.csv", [{"NAME": "D", "LONGNAME": "Long Disease"}])
    _csv(
        tmp_path,
        "disease_map_full.csv",
        [{"original_longname": "Long Disease", "compressed_disease_name": "Compressed Disease"}],
    )
    assert _score(tmp_path) == pytest.approx(0.8175745)


def test_feature_name_synonym_fallback_matches_reference_discount(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule()])
    _csv(tmp_path, "disease_evidence_feature_map.csv", [])
    assert _score(tmp_path, feature_code_to_name={"F": "Smoking status"}) == pytest.approx(0.754915)


def test_consensus_duplicates_across_sources_count_only_once(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule()])
    _csv(
        tmp_path,
        "consensus_evidence.csv",
        [_rule(), _rule(statement_id="different", consensus_stance="against")],
    )
    compiled = MedicalEvidenceStore(tmp_path).compile(endpoints=("D",), features=("F",))
    assert compiled.rules.weights.tolist() == [12.0, -12.0]
    result = compute_rule_evidence_consistency(contributions=np.ones((1, 1, 1)), rules=compiled.rules)
    assert result.scores[0, 0] == pytest.approx(0.5)
    assert result.support_scores[0, 0] == pytest.approx(0.8175745)
    assert result.no_conflict_scores[0, 0] == pytest.approx(0.18242553)


def test_source_rank_filter_uses_consensus_text_ranks(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule(certainty_rank=1)])
    _csv(
        tmp_path,
        "consensus_evidence.csv",
        [
            _rule(
                statement_id="",
                consensus_id="consensus",
                certainty_rank="",
                strength_rank="",
                max_certainty="high",
                final_strength="strong",
            )
        ],
    )
    compiled = MedicalEvidenceStore(tmp_path).compile(
        endpoints=("D",), features=("F",), min_certainty_rank=3, min_strength_rank=2
    )
    assert compiled.rules.weights.tolist() == [12.0]


def test_threshold_diagnostics_preserve_reference_statuses_and_csv_fields(tmp_path: Path) -> None:
    _bundle(tmp_path, [])
    _csv(
        tmp_path,
        "disease_evidence_feature_map.csv",
        [
            {"factor": "smoking", "feature_code": "F"},
            {"factor": "missing", "feature_code": "MISSING"},
        ],
    )
    _csv(
        tmp_path,
        "decision_screening_evidence.csv",
        [
            _rule(statement_id="none", evidence_bucket="decision_screening", threshold=""),
            _rule(
                statement_id="parsed",
                evidence_bucket="decision_screening",
                threshold=">= 2",
                source="guideline",
                snippet="demo",
            ),
            _rule(statement_id="unparsed", evidence_bucket="decision_screening", threshold="as appropriate"),
            _rule(
                statement_id="missing",
                factor="missing",
                evidence_bucket="decision_screening",
                threshold="> 0",
            ),
        ],
    )
    rows, summary = MedicalEvidenceStore(tmp_path).check_thresholds(
        endpoints=("D",),
        features=("F",),
        participant_ids=np.array(["one", "two"]),
        feature_values=np.array([[3.0], [1.0]]),
    )
    assert summary == {
        "n_rows": 8,
        "status_counts": {
            "not_applicable": 2,
            "pass": 1,
            "unparsed_threshold": 2,
            "missing_feature": 2,
            "fail": 1,
        },
        "n_samples": 2,
    }
    assert rows[1] == {
        "sample_id": "one",
        "sample_index": 0,
        "disease": "D",
        "feature_code": "F",
        "feature_name": "F",
        "factor": "smoking",
        "factor_token": "smoking",
        "threshold": ">= 2",
        "threshold_status": "pass",
        "observed_value": 3.0,
        "source": "guideline",
        "snippet": "demo",
    }


def test_threshold_blood_pressure_pair_uses_the_mapped_feature(tmp_path: Path) -> None:
    _bundle(tmp_path, [])
    _csv(
        tmp_path,
        "disease_evidence_feature_map.csv",
        [
            {"factor": "blood_pressure", "feature_code": "p4080"},
            {"factor": "blood_pressure", "feature_code": "p4079"},
        ],
    )
    _csv(
        tmp_path,
        "decision_screening_evidence.csv",
        [
            _rule(
                factor="blood_pressure",
                evidence_bucket="decision_screening",
                threshold="150/90 mmHg or higher",
            )
        ],
    )
    rows, summary = MedicalEvidenceStore(tmp_path).check_thresholds(
        endpoints=("D",),
        features=("p4080", "p4079"),
        participant_ids=np.array(["one", "two"]),
        feature_values=np.array([[140.0, 95.0], [160.0, 85.0]]),
    )
    assert [row["threshold_status"] for row in rows] == ["fail", "pass", "pass", "fail"]
    assert summary["status_counts"] == {"fail": 2, "pass": 2}


def test_evidence_provenance_includes_consensus_and_external_alias_metadata(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _bundle(bundle, [_rule(disease="Long Disease")])
    _csv(bundle, "consensus_evidence.csv", [])
    _csv(metadata, "outcome_test.csv", [{"NAME": "D", "LONGNAME": "Long Disease"}])
    store = MedicalEvidenceStore(bundle, disease_metadata_dir=metadata)
    assert _score(bundle) == 0.5
    assert (metadata / "outcome_test.csv") in store.source_paths
    assert (bundle / "consensus_evidence.csv") in store.source_paths
    result = compute_rule_evidence_consistency(
        contributions=np.ones((1, 1, 1)), rules=store.compile(endpoints=("D",), features=("F",)).rules
    )
    assert result.scores[0, 0] == pytest.approx(0.8175745)


def test_bundled_disease_metadata_is_discovered(tmp_path: Path) -> None:
    _bundle(tmp_path, [_rule(disease="Long Disease")])
    metadata = tmp_path / "disease_metadata"
    metadata.mkdir()
    _csv(metadata, "outcome_filter.csv", [{"NAME": "D", "LONGNAME": "Long Disease"}])
    assert _score(tmp_path) == pytest.approx(0.8175745)


_THRESHOLD_REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/consistency_threshold_reference.json").read_text()
)


@pytest.mark.parametrize("case", _THRESHOLD_REFERENCE["cases"], ids=lambda case: case["name"])
def test_threshold_entrypoint_matches_reference(tmp_path: Path, case: dict) -> None:
    _bundle(tmp_path, [])
    _csv(tmp_path, "decision_screening_evidence.csv", case["rules"])
    _csv(tmp_path, "disease_evidence_feature_map.csv", case["mappings"])
    values = case["feature_values"]
    rows, summary = MedicalEvidenceStore(tmp_path).check_thresholds(
        endpoints=("D",),
        features=tuple(case["features"]),
        participant_ids=case["participant_ids"],
        feature_values=None if values is None else np.asarray(values),
        feature_code_to_name=case["feature_code_to_name"],
    )
    assert rows == case["expected_rows"]
    assert summary == case["expected_summary"]
