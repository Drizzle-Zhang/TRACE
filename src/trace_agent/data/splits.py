from __future__ import annotations

from trace_agent.contracts.task import SplitRole


def require_threshold_selection_role(role: SplitRole | str) -> None:
    resolved = role if isinstance(role, SplitRole) else SplitRole(str(role))
    if not resolved.allows_threshold_selection:
        raise ValueError("thresholds and refinement strategies may only be selected on the validation split")
