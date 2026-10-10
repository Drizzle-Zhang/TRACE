from __future__ import annotations

from pathlib import Path

from trace_agent.knowledge.evidence_store import MedicalEvidenceStore


def test_medical_evidence_store_compiles_signed_endpoint_feature_weights(
    tmp_path: Path,
) -> None:
    (tmp_path / "risk_factor_evidence.csv").write_text(
        (
            "model_labels,disease,factor,direction,certainty_rank,"
            "strength_rank,source_priority,needs_manual_review\n"
            "D1,D1,smoking,positive,4,3,1,0\n"
            "D2,D2,exercise,protective,3,2,1,0\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "disease_evidence_feature_map.csv").write_text(
        (
            "factor,feature_code,match_type,weight,notes\n"
            "smoking,F1,manual,0.8,\n"
            "exercise,F2,manual,0.9,\n"
        ),
        encoding="utf-8",
    )

    compiled = MedicalEvidenceStore(tmp_path).compile(
        endpoints=("D1", "D2"),
        features=("F1", "F2"),
    )

    assert compiled.weights[0, 0] > 0
    assert compiled.weights[1, 1] < 0
    assert compiled.matched_rule_counts.tolist() == [1, 1]
    assert compiled.reliability.min() >= 0
    assert compiled.reliability.max() <= 1
