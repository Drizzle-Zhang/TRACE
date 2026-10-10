from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the declared development backport.
    import tomli as tomllib

import yaml

from trace_agent import TRACEAgent
from trace_agent.identity import TRACE_FULL_NAME, TRACE_NAME, framework_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FULL_NAME = "TRajectory planning Agent for Cost-Effective disease evaluation"


def test_trace_public_identity_is_exact() -> None:
    assert TRACE_NAME == "TRACE"
    assert TRACE_FULL_NAME == EXPECTED_FULL_NAME
    assert TRACEAgent.__name__ == "TRACEAgent"
    assert framework_metadata() == {
        "name": "TRACE",
        "full_name": EXPECTED_FULL_NAME,
    }


def test_packaging_exposes_trace_distribution_and_command() -> None:
    payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["name"] == "trace-agent"
    assert payload["project"]["scripts"] == {"trace": "trace_agent.cli.main:main"}


def test_maintained_surfaces_do_not_contain_obsolete_identity() -> None:
    obsolete = (
        "Health" + "Agent",
        "health" + "agent",
        "HEALTH" + "AGENT_",
    )
    roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "pipelines",
        PROJECT_ROOT / "docs" / "architecture",
    )
    files = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "docs" / "data_contracts.md",
        PROJECT_ROOT / "docs" / "methods.md",
        PROJECT_ROOT / "docs" / "validation.md",
        PROJECT_ROOT / "docs" / "provenance" / "README.md",
    ]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())

    offenders: list[str] = []
    for path in sorted(set(files)):
        if path.suffix not in {"", ".md", ".py", ".toml", ".yaml", ".yml"}:
            continue
        content = path.read_text(encoding="utf-8")
        if any(value in content for value in obsolete):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
    assert not (PROJECT_ROOT / "src" / ("health" + "agent")).exists()


def test_canonical_native_training_pipelines_are_present() -> None:
    expected_steps = {
        "prevalent": [
            "prepare-data",
            "train-integration",
            "calibrate",
            "train-planner",
        ],
        "longitudinal": [
            "prepare-data",
            "train-integration",
            "train-planner",
        ],
    }
    for task, steps in expected_steps.items():
        path = PROJECT_ROOT / "pipelines" / task / "train.yaml"
        assert path.is_file()
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["task"] == task
        assert payload["data"]["split_table"]["path"] == "${TRACE_SPLIT_TABLE}"
        assert payload["pipeline"]["steps"] == steps


def test_trace_project_has_no_legacy_python_package() -> None:
    assert not (PROJECT_ROOT / "src" / ("uk" + "bpp")).exists()
