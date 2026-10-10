from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.training.common import sample_observed_masks
from trace_agent.training.prevalent.integration import (
    PrevalentFusionModel,
    PrevalentIntegrationConfig,
    train_prevalent_integration,
)


def _view(role: SplitRole, *, offset: int, n: int) -> TrainingView:
    labels = np.asarray(
        [[index % 2, (index // 2) % 2] for index in range(n)],
        dtype=np.float32,
    )
    scores = np.empty((n, 2, 2), dtype=np.float32)
    scores[:, 0, 0] = 0.15 + 0.7 * labels[:, 0]
    scores[:, 1, 0] = 0.25 + 0.5 * labels[:, 0]
    scores[:, 0, 1] = 0.2 + 0.6 * labels[:, 1]
    scores[:, 1, 1] = 0.3 + 0.4 * labels[:, 1]
    batch = ModuleScoreBatch(
        participant_ids=np.arange(offset, offset + n),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((n, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    return TrainingView(
        role=role,
        row_indices=np.arange(offset, offset + n),
        scores=batch,
        targets=TrainingTargets(
            labels=labels,
            label_mask=np.ones_like(labels, dtype=bool),
        ),
    )


def test_sample_observed_masks_is_deterministic_and_includes_full_masks() -> None:
    availability = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)

    first_rows, first_masks = sample_observed_masks(
        availability,
        subsets_per_participant=2,
        seed=9,
    )
    second_rows, second_masks = sample_observed_masks(
        availability,
        subsets_per_participant=2,
        seed=9,
    )

    np.testing.assert_array_equal(first_rows, second_rows)
    np.testing.assert_array_equal(first_masks, second_masks)
    np.testing.assert_array_equal(first_rows, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(first_masks[[0, 3]], availability)
    assert np.all(np.sum(first_masks, axis=1) > 0)


def test_prevalent_fusion_trains_and_fuses_partial_masks() -> None:
    train = _view(SplitRole.TRAIN, offset=0, n=12)
    validation = _view(SplitRole.VALIDATION, offset=12, n=8)

    result = train_prevalent_integration(
        train,
        validation,
        config=PrevalentIntegrationConfig(
            seed=3,
            subsets_per_participant=2,
            min_class_count=1,
            n_estimators=8,
            num_leaves=7,
            n_jobs=1,
        ),
    )

    assert isinstance(result.model, PrevalentFusionModel)
    probability = result.model.fuse(
        validation.scores.scores[0],
        np.array([True, False]),
        pred_mask_md=validation.scores.pred_mask[0],
    )
    assert probability.shape == (2,)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert result.train_row_indices.tolist() == train.row_indices.tolist()
    assert result.validation_row_indices.tolist() == validation.row_indices.tolist()
    assert result.metrics["participants"] == 8


def test_prevalent_integration_rejects_test_view() -> None:
    with pytest.raises(ValueError, match="train view"):
        train_prevalent_integration(
            _view(SplitRole.TEST, offset=0, n=4),
            _view(SplitRole.VALIDATION, offset=4, n=4),
            config=PrevalentIntegrationConfig(
                n_estimators=4,
                min_class_count=1,
            ),
        )


def test_prevalent_integration_uses_constant_model_without_two_classes() -> None:
    train = _view(SplitRole.TRAIN, offset=0, n=8)
    train.targets.labels[:, 1] = 1.0
    validation = _view(SplitRole.VALIDATION, offset=8, n=4)

    result = train_prevalent_integration(
        train,
        validation,
        config=PrevalentIntegrationConfig(
            n_estimators=4,
            min_class_count=1,
        ),
    )

    predictions = result.model.predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=validation.scores.availability,
    )
    np.testing.assert_allclose(predictions[:, 1], 1.0)
