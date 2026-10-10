from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.contracts.task import SplitRole
from trace_agent.data.alignment import align_named_axis, build_name_permutation
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.splits import require_threshold_selection_role
from trace_agent.runtime.provenance import file_provenance


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


@pytest.mark.parametrize("id_name", ["participant_ids", "eids"])
@pytest.mark.parametrize("with_manifest", [False, True])
def test_load_module_score_bundle_preserves_text_ids_from_both_array_contracts(
    tmp_path: Path, id_name: str, with_manifest: bool
) -> None:
    ids = np.array(["001", "NA"])
    np.save(tmp_path / f"sample_{id_name}.npy", ids)
    np.save(tmp_path / "sample_modules.npy", np.array(["M1", "M2"]))
    np.save(tmp_path / "sample_diseases.npy", np.array(["D1"]))
    np.save(tmp_path / "sample_oof_probs.npy", np.array([[[0.2], [np.nan]], [[0.3], [0.4]]]))
    np.save(tmp_path / "sample_pred_mask.npy", np.array([[[1], [0]], [[1], [1]]], dtype=np.uint8))
    np.save(tmp_path / "sample_avail.npy", np.array([[1, 1], [1, 1]], dtype=np.uint8))
    if with_manifest:
        files = {}
        for name in (id_name, "modules", "diseases", "oof_probs", "pred_mask", "avail"):
            path = tmp_path / f"sample_{name}.npy"
            files[name] = {"file": path.name, "sha256": file_provenance(path)["sha256"]}
        (tmp_path / "sample_score_bundle_manifest.json").write_text(
            json.dumps({"schema_version": 1, "task": "prevalent", "files": files})
        )
        other_id_name = "eids" if id_name == "participant_ids" else "participant_ids"
        np.save(tmp_path / f"sample_{other_id_name}.npy", np.array(["stale-1", "stale-2"]))

    batch = load_module_score_bundle(tmp_path, prefix="sample", task="prevalent")

    assert batch.modules == ("M1", "M2")
    assert batch.endpoints == ("D1",)
    assert batch.scores.shape == (2, 2, 1)
    assert batch.metadata["participant_ids_file"] == f"sample_{id_name}.npy"
    np.testing.assert_array_equal(batch.participant_ids, ids)
    np.testing.assert_array_equal(batch.pred_mask, [[[True], [False]], [[True], [True]]])


def test_score_bundle_without_manifest_prefers_canonical_participant_ids(tmp_path: Path) -> None:
    for name, values in {
        "participant_ids": np.array(["001"]),
        "eids": np.array(["stale"]),
        "modules": np.array(["M1"]),
        "diseases": np.array(["D1"]),
        "oof_probs": np.array([[[0.2]]]),
        "pred_mask": np.ones((1, 1, 1), bool),
        "avail": np.ones((1, 1), bool),
    }.items():
        np.save(tmp_path / f"sample_{name}.npy", values)

    batch = load_module_score_bundle(tmp_path, prefix="sample", task="prevalent")

    assert batch.participant_ids.tolist() == ["001"]


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
