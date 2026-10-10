from __future__ import annotations

import ast
import inspect
from pathlib import Path

from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlanningEngine


def test_module_selection_cannot_receive_labels() -> None:
    for function in (KnowledgeCandidateRanker.rank, PlanningEngine.plan):
        parameters = inspect.signature(function).parameters
        assert "y" not in parameters
        assert "y_mask" not in parameters
        assert "label_mask" not in parameters


def test_module_selection_sources_do_not_reference_label_masks() -> None:
    project_root = Path(__file__).resolve().parents[2]
    selection_sources = (
        project_root / "src/trace_agent/planning/candidate_ranker.py",
        project_root / "src/trace_agent/planning/engine.py",
    )
    forbidden = {"y_mask", "y_mask_d", "label_mask", "label_mask_d", "mask_state_by_y_mask"}

    for source in selection_sources:
        names = {node.id for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.Name)}
        assert names.isdisjoint(forbidden), f"{source} references a label mask"
