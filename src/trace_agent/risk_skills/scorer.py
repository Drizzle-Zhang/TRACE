from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.risk_skills.registry import RiskSkillRegistry


def score_feature_modules(
    *,
    participant_ids: np.ndarray,
    modules: Sequence[str],
    endpoints: Sequence[str],
    features_by_module: Mapping[str, np.ndarray],
    availability: np.ndarray,
    registry: RiskSkillRegistry,
    task: TaskType | str,
) -> ModuleScoreBatch:
    ids = np.asarray(participant_ids).reshape(-1)
    module_names = tuple(str(module) for module in modules)
    endpoint_names = tuple(str(endpoint) for endpoint in endpoints)
    available = np.asarray(availability, dtype=bool)
    expected_availability = (ids.size, len(module_names))
    if available.shape != expected_availability:
        raise ValueError(f"availability shape must be {expected_availability}, got {available.shape}")

    scores = np.full(
        (ids.size, len(module_names), len(endpoint_names)),
        np.nan,
        dtype=np.float32,
    )
    pred_mask = np.zeros(scores.shape, dtype=bool)
    resolved_task = TaskType.parse(task)

    for module_index, module in enumerate(module_names):
        values = features_by_module.get(module)
        if values is None:
            continue
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[0] != ids.size:
            raise ValueError(f"feature matrix for module {module!r} must have {ids.size} rows")
        for endpoint_index, endpoint in enumerate(endpoint_names):
            skill = registry.get(task=resolved_task, module=module, endpoint=endpoint)
            if skill is None:
                continue
            predicted = np.asarray(skill.predict(matrix), dtype=np.float32).reshape(-1)
            if predicted.shape != (ids.size,):
                raise ValueError(f"risk skill for {module}/{endpoint} returned shape {predicted.shape}")
            valid = available[:, module_index] & np.isfinite(predicted)
            scores[valid, module_index, endpoint_index] = predicted[valid]
            pred_mask[valid, module_index, endpoint_index] = True

    return ModuleScoreBatch(
        participant_ids=ids,
        modules=module_names,
        endpoints=endpoint_names,
        scores=scores,
        pred_mask=pred_mask,
        availability=available,
        task=resolved_task,
    )
