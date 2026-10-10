from __future__ import annotations

TRACE_NAME = "TRACE"
TRACE_FULL_NAME = "TRajectory planning Agent for Cost-Effective disease evaluation"


def framework_metadata() -> dict[str, str]:
    """Return a fresh machine-readable TRACE identity mapping."""

    return {"name": TRACE_NAME, "full_name": TRACE_FULL_NAME}
