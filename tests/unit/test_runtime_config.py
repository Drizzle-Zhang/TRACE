from __future__ import annotations

from pathlib import Path

import pytest

from trace_agent.runtime.config import load_config


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_full_module_training_examples_are_portable(monkeypatch, tmp_path, task):
    for name in (
        "TRACE_MODULE_CATALOG",
        "TRACE_INPUT_DIR",
        "TRACE_SPLIT_TABLE",
        "TRACE_OUTPUT_DIR",
        "TRACE_MODULE_MODEL_DIR",
        "TRACE_KNOWLEDGE_MATRIX",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("TRACE_INPUT_PREFIX", "cohort")
    config = load_config(Path("pipelines") / task / "full_train.yaml")
    assert config["pipeline"]["steps"][:3] == ["prepare-data", "train-modules", "score-modules"]
    assert config["module_data"]["score_mode"] == "training"
    assert Path(config["artifacts"]["module_model_dir"]).is_absolute()


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_frozen_score_examples_require_no_label_or_split_config(monkeypatch, tmp_path, task):
    for name in ("TRACE_MODULE_CATALOG", "TRACE_INPUT_DIR", "TRACE_MODULE_MODEL_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("TRACE_INPUT_PREFIX", "cohort")
    config = load_config(Path("pipelines") / task / "score.yaml")
    assert config["pipeline"]["steps"] == ["score-modules"]
    assert config["module_data"]["score_mode"] == "inference"
    assert "split_table" not in config["data"]


def test_config_expands_environment_merges_profile_and_resolves_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "input"
    monkeypatch.setenv("TRACE_TEST_INPUT", str(data_dir))
    base = tmp_path / "pipeline.yaml"
    profile = tmp_path / "profile.yaml"
    base.write_text(
        """
version: 1
task: prevalent
data:
  input_dir: ${TRACE_TEST_INPUT}
  prefix: validation
planner:
  candidate_actions: 16
output:
  directory: ./results
""".strip()
        + "\n",
        encoding="utf-8",
    )
    profile.write_text(
        """
planner:
  candidate_actions: 4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(base, profile_path=profile)

    assert config["data"]["input_dir"] == str(data_dir.resolve())
    assert config["planner"]["candidate_actions"] == 4
    assert config["output"]["directory"] == str((tmp_path / "results").resolve())
    assert config["_meta"]["config_path"] == str(base.resolve())


def test_config_rejects_unresolved_environment_variables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "version: 1\ndata:\n  input_dir: ${NOT_DEFINED_FOR_TEST}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NOT_DEFINED_FOR_TEST"):
        load_config(path)


def test_config_supports_relative_inheritance(
    tmp_path: Path,
) -> None:
    defaults = tmp_path / "defaults.yaml"
    pipeline = tmp_path / "pipelines" / "infer.yaml"
    pipeline.parent.mkdir()
    defaults.write_text(
        "version: 1\nplanner:\n  max_steps: 5\n  candidate_actions: 16\n",
        encoding="utf-8",
    )
    pipeline.write_text(
        "extends: ../defaults.yaml\nplanner:\n  candidate_actions: 3\n",
        encoding="utf-8",
    )

    config = load_config(pipeline)

    assert config["planner"] == {
        "max_steps": 5,
        "candidate_actions": 3,
    }


def test_config_resolves_nested_split_table_path(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        "version: 1\n" "data:\n" "  split_table:\n" "    path: ./data/splits.csv\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["data"]["split_table"]["path"] == str((tmp_path / "data" / "splits.csv").resolve())


def test_evaluation_participant_ids_resolve_from_config_directory(tmp_path, monkeypatch) -> None:
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    other_working_directory = tmp_path / "unrelated"
    other_working_directory.mkdir()
    path = configuration / "evaluate.yaml"
    path.write_text("evaluation:\n  participant_ids: ./outputs/participant_ids.npy\n", encoding="utf-8")
    monkeypatch.chdir(other_working_directory)

    config = load_config(path)

    assert config["evaluation"]["participant_ids"] == str(configuration / "outputs/participant_ids.npy")


@pytest.mark.parametrize("configured", ["~/trace-input", "${TRACE_TEST_HOME_PATH}"])
def test_config_expands_home_before_deciding_if_path_is_relative(tmp_path, monkeypatch, configured) -> None:
    path = tmp_path / "infer.yaml"
    path.write_text(f"data:\n  input_dir: {configured}\n", encoding="utf-8")
    monkeypatch.setenv("TRACE_TEST_HOME_PATH", "~/trace-input")

    config = load_config(path)

    assert config["data"]["input_dir"] == str(Path.home() / "trace-input")
