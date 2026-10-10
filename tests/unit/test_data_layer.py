from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trace_agent.contracts.task import SplitRole
from trace_agent.data.alignment import align_named_axis, build_name_permutation
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.splits import require_threshold_selection_role


def test_named_axis_alignment_uses_names_not_positions() -> None:
    source = ("M2", "M1")
    target = ("M1", "M2")
    values = np.array([[20, 10]], dtype=np.int64)

    assert build_name_permutation(source, target).tolist() == [1, 0]
    assert align_named_axis(values, source, target, axis=1).tolist() == [[10, 20]]

    with pytest.raises(ValueError, match="missing target names"):
        build_name_permutation(("M1",), ("M1", "M2"))


def test_burden_profile_combines_money_and_time_without_mutating_components() -> None:
    profile = BurdenProfile(
        modules=("M1", "M2"),
        monetary=np.array([2.0, 6.0]),
        time=np.array([8.0, 4.0]),
        monetary_weight=0.25,
        time_weight=0.75,
    )

    np.testing.assert_allclose(profile.composite, [6.5, 4.5])
    assert profile.accumulated(np.array([1, 0], dtype=bool)) == pytest.approx(6.5)


def test_only_validation_split_can_select_thresholds() -> None:
    require_threshold_selection_role(SplitRole.VALIDATION)
    with pytest.raises(ValueError, match="validation"):
        require_threshold_selection_role(SplitRole.TEST)


def test_load_module_score_bundle_reads_canonical_array_contract(tmp_path: Path) -> None:
    np.save(tmp_path / "sample_eids.npy", np.array([11, 12]))
    np.save(tmp_path / "sample_modules.npy", np.array(["M1", "M2"]))
    np.save(tmp_path / "sample_diseases.npy", np.array(["D1"]))
    np.save(tmp_path / "sample_oof_probs.npy", np.array([[[0.2], [np.nan]], [[0.3], [0.4]]]))
    np.save(tmp_path / "sample_pred_mask.npy", np.array([[[1], [0]], [[1], [1]]], dtype=np.uint8))
    np.save(tmp_path / "sample_avail.npy", np.array([[1, 1], [1, 1]], dtype=np.uint8))

    batch = load_module_score_bundle(tmp_path, prefix="sample", task="prevalent")

    assert batch.modules == ("M1", "M2")
    assert batch.endpoints == ("D1",)
    assert batch.scores.shape == (2, 2, 1)


def test_legacy_pickled_name_arrays_require_explicit_opt_in(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "legacy_eids.npy", np.array([11]))
    np.save(
        tmp_path / "legacy_modules.npy",
        np.array(["M1"], dtype=object),
    )
    np.save(
        tmp_path / "legacy_diseases.npy",
        np.array(["D1"], dtype=object),
    )
    np.save(
        tmp_path / "legacy_oof_probs.npy",
        np.array([[[0.2]]], dtype=np.float32),
    )
    np.save(
        tmp_path / "legacy_pred_mask.npy",
        np.ones((1, 1, 1), dtype=bool),
    )
    np.save(
        tmp_path / "legacy_avail.npy",
        np.ones((1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="allow_legacy_pickle_names"):
        load_module_score_bundle(tmp_path, prefix="legacy", task="prevalent")

    batch = load_module_score_bundle(
        tmp_path,
        prefix="legacy",
        task="prevalent",
        allow_legacy_pickle_names=True,
    )
    assert batch.modules == ("M1",)
