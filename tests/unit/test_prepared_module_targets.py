from __future__ import annotations

import numpy as np
import pytest

from tests.module_fixtures import write_module_inputs
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.data.module_outcomes import load_module_outcomes


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_targets_can_be_prepared_without_module_scores(tmp_path, task):
    from trace_agent.data.module_outcomes import write_target_arrays
    from trace_agent.data.training_dataset import load_training_targets

    features = load_module_features(load_catalog(write_module_inputs(tmp_path, task=task)))
    targets = load_module_outcomes(features.catalog, features.participant_ids, task=task)
    root = tmp_path / "prepared"
    paths = write_target_arrays(root, prefix="cohort", targets=targets, task=task)
    loaded = load_training_targets(root, prefix="cohort", task=task, expected=(180, 1))
    assert loaded.label_mask.all()
    np.testing.assert_allclose(loaded.labels, targets.labels)
    expected = {"y", "y_mask"} if task == "prevalent" else {"event", "time", "signed_time", "label_mask"}
    assert set(paths) == expected
    assert not (root / "cohort_oof_probs.npy").exists()
    assert not list(root.glob("*.tmp"))
    if task == "longitudinal":
        np.testing.assert_allclose(loaded.signed_time, targets.signed_time)


def test_prepared_targets_check_expected_shape(tmp_path):
    from trace_agent.data.module_outcomes import write_target_arrays
    from trace_agent.data.training_dataset import load_training_targets

    features = load_module_features(load_catalog(write_module_inputs(tmp_path)))
    targets = load_module_outcomes(features.catalog, features.participant_ids, task="prevalent")
    write_target_arrays(tmp_path / "prepared", prefix="cohort", targets=targets, task="prevalent")
    with pytest.raises(ValueError, match="shape"):
        load_training_targets(tmp_path / "prepared", prefix="cohort", task="prevalent", expected=(179, 1))


@pytest.mark.parametrize("prefix", ["", "../escape", "nested/cohort"])
def test_target_prefix_cannot_escape_output_directory(tmp_path, prefix):
    from trace_agent.data.module_outcomes import write_target_arrays

    features = load_module_features(load_catalog(write_module_inputs(tmp_path)))
    targets = load_module_outcomes(features.catalog, features.participant_ids, task="prevalent")
    with pytest.raises(ValueError, match="prefix"):
        write_target_arrays(tmp_path / "prepared", prefix=prefix, targets=targets, task="prevalent")
