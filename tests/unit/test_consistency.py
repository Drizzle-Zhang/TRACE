from __future__ import annotations

import numpy as np
import pytest

from trace_agent.consistency.evidence import compute_evidence_consistency
from trace_agent.consistency.feature import compute_feature_consistency
from trace_agent.consistency.kg import compute_kg_consistency
from trace_agent.consistency.strategy import (
    apply_consistency_filter,
    learn_consistency_thresholds,
)
from trace_agent.consistency.validator import ConsistencyValidator
from trace_agent.contracts.consistency import ConsistencyThresholds
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType


def test_kg_consistency_is_a_masked_knowledge_weighted_mean() -> None:
    module_scores = np.array([[[0.9, 0.2], [0.3, 0.8]]], dtype=np.float32)
    pred_mask = np.ones_like(module_scores, dtype=bool)
    kg = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    scores = compute_kg_consistency(module_scores, pred_mask, kg)

    np.testing.assert_allclose(scores, [[0.6, 0.8]])


def test_feature_and_evidence_scores_are_neutral_without_matches() -> None:
    contributions = np.array([[[2.0, -1.0], [4.0, 1.0]]], dtype=np.float32)
    feature_weights = np.array([[4.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    match_mask = feature_weights != 0

    feature = compute_feature_consistency(
        contributions=contributions,
        medical_weights=feature_weights,
        match_mask=match_mask,
    )

    assert feature.scores[0, 0] == pytest.approx(1.0 / (1.0 + np.exp(-0.75)))
    assert feature.scores[0, 1] == pytest.approx(0.5)
    assert feature.matched_counts.tolist() == [[2, 0]]

    evidence = compute_evidence_consistency(
        contributions=contributions,
        evidence_weights=np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        match_mask=match_mask,
        reliability=np.array([[1.0, 0.5], [1.0, 1.0]], dtype=np.float32),
    )
    assert evidence.scores.shape == (1, 2)
    assert evidence.support_scores[0, 0] > 0.5
    assert evidence.no_conflict_scores[0, 0] < 0.5
    assert evidence.scores[0, 1] == pytest.approx(0.5)


def test_consistency_thresholds_are_learned_on_validation_and_then_frozen() -> None:
    predictions = np.array([[0.9], [0.8], [0.7], [0.6]], dtype=np.float32)
    labels = np.array([[1], [1], [0], [0]], dtype=np.float32)
    mask = np.ones_like(labels, dtype=bool)
    kg = np.array([[0.9], [0.8], [0.3], [0.2]], dtype=np.float32)
    feature = np.array([[0.8], [0.7], [0.4], [0.3]], dtype=np.float32)
    evidence = np.array([[0.9], [0.7], [0.3], [0.4]], dtype=np.float32)

    with pytest.raises(ValueError, match="validation"):
        learn_consistency_thresholds(
            predictions=predictions,
            labels=labels,
            label_mask=mask,
            kg_scores=kg,
            feature_scores=feature,
            evidence_scores=evidence,
            split_role=SplitRole.TEST,
            grid=np.array([0.25, 0.5, 0.75]),
        )

    thresholds = learn_consistency_thresholds(
        predictions=predictions,
        labels=labels,
        label_mask=mask,
        kg_scores=kg,
        feature_scores=feature,
        evidence_scores=evidence,
        split_role=SplitRole.VALIDATION,
        grid=np.array([0.25, 0.5, 0.75]),
    )
    result = apply_consistency_filter(
        predictions=predictions,
        kg_scores=kg,
        feature_scores=feature,
        evidence_scores=evidence,
        thresholds=thresholds,
    )

    assert thresholds.source_split is SplitRole.VALIDATION
    assert result.strategy_masks["joint"].reshape(-1).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert result.combined_scores.shape == predictions.shape


def test_validator_composes_the_three_evidence_chains() -> None:
    batch = ModuleScoreBatch(
        participant_ids=np.array([11]),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=np.array([[[0.8], [0.6]]], dtype=np.float32),
        pred_mask=np.ones((1, 2, 1), dtype=bool),
        availability=np.ones((1, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    profile = RiskProfile(
        participant_ids=np.array([11]),
        endpoints=("D1",),
        raw_scores=np.array([[0.9]], dtype=np.float32),
        calibrated_scores=np.array([[0.9]], dtype=np.float32),
        valid_mask=np.ones((1, 1), dtype=bool),
        observed_modules=np.ones((1, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    validator = ConsistencyValidator(
        kg_matrix=np.ones((2, 1), dtype=np.float32),
        medical_weights=np.array([[1.0, 1.0]], dtype=np.float32),
        evidence_weights=np.array([[1.0, 0.5]], dtype=np.float32),
        thresholds=ConsistencyThresholds(
            prediction=0.5,
            kg=0.5,
            feature=0.5,
            evidence=0.5,
            combined=0.5,
        ),
    )

    result = validator.validate(
        batch=batch,
        profile=profile,
        contributions=np.array([[[2.0, 1.0]]], dtype=np.float32),
    )

    assert result.kg_scores[0, 0] == pytest.approx(0.7)
    assert result.strategy_masks["joint"][0, 0]
