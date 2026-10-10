from __future__ import annotations

from typing import Any

import numpy as np

from trace_agent.contracts.results import AgentResult
from trace_agent.contracts.scores import RiskProfile


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _risk_profile_to_dict(profile: RiskProfile) -> dict[str, Any]:
    valid = np.asarray(profile.valid_mask[0], dtype=bool)

    def scores(values: np.ndarray) -> list[float | None]:
        row = np.asarray(values)[0]
        return [
            float(value) if valid[index] and np.isfinite(value) else None for index, value in enumerate(row)
        ]

    return {
        "endpoint_order": list(profile.endpoints),
        "raw_scores": scores(profile.raw_scores),
        "decision_scores": scores(profile.decision_scores),
        "valid_mask": valid.tolist(),
        "observed_modules": profile.observed_modules[0].tolist(),
        "horizon_risks": {str(name): scores(values) for name, values in profile.horizon_risks.items()},
    }


def agent_result_to_dict(result: AgentResult, *, top_endpoints: int = 0) -> dict[str, Any]:
    endpoint_rows = list(result.prioritized_endpoints)
    if top_endpoints > 0:
        endpoint_rows = endpoint_rows[: int(top_endpoints)]
    profile = result.planning.final_profile
    trajectory = [
        {
            "step": step.step_index,
            "action": step.action.kind.value,
            "module_index": step.action.module_index,
            "module_name": step.action.module_name,
            "candidate_module_indices": list(step.candidate_module_indices),
            "q_values": step.q_values.tolist(),
            "accumulated_burden": step.accumulated_burden,
            "risk_profile": _risk_profile_to_dict(step.risk_profile),
            "metadata": _jsonable(step.metadata),
        }
        for step in result.planning.trajectory
    ]
    consistency = None
    if result.consistency is not None:
        consistency = {
            "kg_scores": result.consistency.kg_scores[0].tolist(),
            "feature_scores": result.consistency.feature_scores[0].tolist(),
            "evidence_scores": result.consistency.evidence_scores[0].tolist(),
            "combined_scores": result.consistency.combined_scores[0].tolist(),
            "strategy_masks": {
                name: values[0].tolist() for name, values in result.consistency.strategy_masks.items()
            },
        }
    row_metadata = {key: value for key, value in result.metadata.items() if key != "artifact_provenance"}
    return {
        "participant_id": _scalar(result.participant_id),
        "task": result.task.value,
        "selected_modules": list(result.selected_modules),
        "initial_used_modules": result.planning.initial_used_modules.tolist(),
        "final_used_modules": result.planning.final_used_modules.tolist(),
        "accumulated_burden": result.planning.accumulated_burden,
        "endpoints": [
            {
                "endpoint": assessment.endpoint,
                "endpoint_index": assessment.endpoint_index,
                "score": assessment.score,
                "raw_score": assessment.raw_score,
                "retained_by_consistency": (assessment.retained_by_consistency),
                "horizon_risks": dict(assessment.horizon_risks),
            }
            for assessment in endpoint_rows
        ],
        "endpoint_order": list(profile.endpoints),
        "trajectory": trajectory,
        "rationale": list(result.rationale),
        "validation_checks": dict(result.validation_checks),
        "consistency": consistency,
        "metadata": _jsonable(row_metadata),
    }
