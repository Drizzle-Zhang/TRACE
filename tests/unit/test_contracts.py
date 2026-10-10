from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.schema import AgentSchema, EndpointSpec, ModuleSpec
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType


def test_task_aliases_and_split_threshold_policy() -> None:
    assert TaskType.parse("diagnosis") is TaskType.PREVALENT
    assert TaskType.parse("survival") is TaskType.LONGITUDINAL
    assert SplitRole.VALIDATION.allows_threshold_selection
    assert not SplitRole.TEST.allows_threshold_selection


def test_schema_rejects_duplicate_names_and_exposes_stable_indices() -> None:
    schema = AgentSchema(
        modules=(ModuleSpec("Baseline", ("age", "sex")), ModuleSpec("Lab", ("crp",))),
        endpoints=(EndpointSpec("D1"), EndpointSpec("D2")),
    )

    assert schema.module_index("Lab") == 1
    assert schema.endpoint_index("D1") == 0

    with pytest.raises(ValueError, match="duplicate module"):
        AgentSchema(
            modules=(ModuleSpec("M1"), ModuleSpec("M1")),
            endpoints=(EndpointSpec("D1"),),
        )


def test_module_score_batch_preserves_missingness_instead_of_low_risk() -> None:
    scores = np.array([[[0.8], [np.nan]]], dtype=np.float32)
    batch = ModuleScoreBatch(
        participant_ids=np.array([101]),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=scores,
        pred_mask=np.array([[[1], [0]]], dtype=bool),
        availability=np.array([[1, 1]], dtype=bool),
        task=TaskType.PREVALENT,
    )

    assert np.isnan(batch.scores[0, 1, 0])
    assert not batch.pred_mask[0, 1, 0]

    with pytest.raises(ValueError, match="masked-out scores must be NaN"):
        ModuleScoreBatch(
            participant_ids=np.array([101]),
            modules=("M1", "M2"),
            endpoints=("D1",),
            scores=np.array([[[0.8], [0.0]]], dtype=np.float32),
            pred_mask=np.array([[[1], [0]]], dtype=bool),
            availability=np.array([[1, 1]], dtype=bool),
            task=TaskType.PREVALENT,
        )


def test_risk_profile_validates_endpoint_order_and_shapes() -> None:
    profile = RiskProfile(
        participant_ids=np.array([101, 102]),
        endpoints=("D1", "D2"),
        raw_scores=np.ones((2, 2), dtype=np.float32),
        valid_mask=np.ones((2, 2), dtype=bool),
        observed_modules=np.array([[1, 0], [1, 1]], dtype=bool),
        task=TaskType.LONGITUDINAL,
        calibrated_scores=None,
        horizon_risks={"10y": np.full((2, 2), 0.2, dtype=np.float32)},
    )

    assert profile.n_participants == 2
    assert profile.horizon_risks["10y"].shape == (2, 2)

    with pytest.raises(ValueError, match="raw_scores"):
        RiskProfile(
            participant_ids=np.array([101, 102]),
            endpoints=("D1", "D2"),
            raw_scores=np.ones((2, 1), dtype=np.float32),
            valid_mask=np.ones((2, 2), dtype=bool),
            observed_modules=np.ones((2, 2), dtype=bool),
            task=TaskType.PREVALENT,
        )
