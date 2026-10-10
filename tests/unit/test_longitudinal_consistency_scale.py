import csv
import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.consistency.validator import ConsistencyValidator
from trace_agent.contracts.consistency import ConsistencyThresholds
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile


def test_q99_reference_matches_frozen_source_identity():
    manifest = json.loads((Path(__file__).parents[2] / "tests/fixtures/reference_sources.json").read_text())
    source = next(
        item for item in manifest["files"] if item["path"] == "survival_consistency_full/kg_scores.py"
    )
    assert source["sha256"] == "7eb0b8d2d6094391f5c59262f4ec6bad54771ad543dd14a3b84769bb40f0e689"


def test_log_risk_scale_reproduces_frozen_positive_risk_q99():
    from trace_agent.consistency.longitudinal import fit_longitudinal_scale

    # Frozen survival_consistency_full/kg_scores.py: HR / quantile(HR, .99), clipped.
    ratios = np.array([1.0, 2.0, 4.0, 8.0, 1e30], dtype=np.float32).reshape(5, 1, 1)
    scale = fit_longitudinal_scale(
        np.log(ratios), np.ones_like(ratios, dtype=bool), np.arange(4), score_kind="log_risk"
    )
    np.testing.assert_allclose(
        scale.transform(np.log(ratios))[:4], np.clip(ratios[:4] / 7.88, 0, 1), rtol=1e-6
    )
    assert scale.log_reference == pytest.approx(np.log(7.88), abs=1e-7)
    assert scale.transform(np.log(ratios))[-1, 0, 0] == 1.0


def test_scale_is_stable_for_large_native_log_risks():
    from trace_agent.consistency.longitudinal import fit_longitudinal_scale

    scores = np.array([1000.0, 1001.0, np.nan]).reshape(3, 1, 1)
    scale = fit_longitudinal_scale(scores, np.isfinite(scores), np.arange(3), score_kind="log_risk")
    result = scale.transform(scores)
    assert np.isfinite(result[:2]).all()
    assert np.isnan(result[2, 0, 0])
    assert scale.log_reference == pytest.approx(np.logaddexp(1000 + np.log(0.01), 1001 + np.log(0.99)))


def test_q99_sampling_does_not_retain_full_validation_chunks():
    import tracemalloc

    from trace_agent.consistency.longitudinal import fit_longitudinal_scale

    scores = np.ones((20000, 8, 8), dtype=np.float32)
    mask = np.ones_like(scores, dtype=bool)
    rows = np.arange(scores.shape[0])
    tracemalloc.start()
    try:
        fit_longitudinal_scale(scores, mask, rows, chunk_size=512, max_values=1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Bounded sampling must not retain the complete 5.1 MB validation score data.
    assert peak < 3_000_000


def test_longitudinal_filter_uses_named_probability_horizon():
    batch = ModuleScoreBatch(
        participant_ids=np.array(["1", "2"]),
        modules=("M",),
        endpoints=("D",),
        scores=np.array([[[2.0]], [[-1.0]]]),
        pred_mask=np.ones((2, 1, 1), bool),
        availability=np.ones((2, 1), bool),
        task="longitudinal",
    )
    profile = RiskProfile(
        participant_ids=batch.participant_ids,
        endpoints=batch.endpoints,
        raw_scores=np.array([[2.0], [-1.0]]),
        valid_mask=np.ones((2, 1), bool),
        observed_modules=batch.availability,
        task="longitudinal",
        horizon_risks={"5y": np.array([[0.1], [0.8]])},
    )
    validator = ConsistencyValidator(
        kg_matrix=np.ones((1, 1)),
        thresholds=ConsistencyThresholds(0.5, 0.5, 0.5, 0.5, 0.5),
        module_score_transform=lambda values: np.exp(np.minimum(values - 2.0, 0)),
        prediction_horizon="5y",
    )
    result = validator.validate(batch=batch, profile=profile)
    assert result.strategy_masks["baseline"].tolist() == [[False], [True]]
    with pytest.raises(ValueError, match="longitudinal.*scale"):
        ConsistencyValidator(
            kg_matrix=np.ones((1, 1)), thresholds=ConsistencyThresholds(0.5, 0.5, 0.5, 0.5, 0.5)
        ).validate(batch=batch, profile=profile)


def test_scale_cli_uses_only_validation_without_loading_labels(tmp_path):
    from tests.module_fixtures import write_full_module_pipeline
    from trace_agent.cli.consistency_scale import calibrate_consistency_scale
    from trace_agent.cli.main import main
    from trace_agent.runtime.config import load_config

    path = write_full_module_pipeline(tmp_path, task="longitudinal")
    assert main(["run", "--config", str(path)]) == 0
    config = load_config(path)
    config["consistency"] = {"enabled": True, "horizon": "5y"}
    for name in ("event", "time", "signed_time", "label_mask"):
        (tmp_path / "prepared" / f"cohort_{name}.npy").unlink()
    artifact = calibrate_consistency_scale(config)
    record = json.loads(artifact.read_text())
    assert record["validation_rows"] == 30
    assert record["test_rows_used"] == 0
    assert record["module_score_kind"] == "log_risk"
    assert main(["calibrate-consistency-scale", "--config", str(path)]) == 0


def _scale_only_config(tmp_path, rows):
    scores = np.log(np.array([1000.0, 1.0, 1000.0, 2.0, 1000.0, 4.0, 1000.0])).reshape(7, 1, 1)
    root = tmp_path / "scores"
    root.mkdir()
    for suffix, values in {
        "scores": scores.astype(np.float32),
        "pred_mask": np.ones(scores.shape, dtype=bool),
        "avail": np.ones((7, 1), dtype=bool),
        "eids": np.array([str(100 + index) for index in range(7)]),
        "modules": np.array(["M"]),
        "diseases": np.array(["D"]),
    }.items():
        np.save(root / f"cohort_{suffix}.npy", values, allow_pickle=False)
    split_path = tmp_path / "validation.csv"
    with split_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "eid", "split"])
        writer.writerows(rows)
    return {
        "task": "longitudinal",
        "data": {"input_dir": str(root), "prefix": "cohort", "split_table": {"path": str(split_path)}},
        "consistency": {"longitudinal_scale_path": str(tmp_path / "scale.json")},
        "output": {"directory": str(tmp_path / "output")},
    }


def test_scale_cli_accepts_only_designated_validation_rows(tmp_path):
    from trace_agent.cli.consistency_scale import calibrate_consistency_scale

    config = _scale_only_config(
        tmp_path, [(5, "105", "validation"), (1, "101", "val"), (3, "103", "validation")]
    )
    path = calibrate_consistency_scale(config)
    first = json.loads(path.read_text())
    assert first["validation_rows"] == 3
    assert first["test_rows_used"] == 0
    assert first["log_reference"] == pytest.approx(np.log(3.96), abs=1e-7)

    score_path = Path(config["data"]["input_dir"]) / "cohort_scores.npy"
    scores = np.load(score_path, allow_pickle=False)
    scores[[0, 2, 4, 6]] = np.array([-1000.0, 1000.0, -500.0, 500.0]).reshape(4, 1, 1)
    np.save(score_path, scores, allow_pickle=False)
    second = json.loads(calibrate_consistency_scale(config).read_text())
    assert second == first


def test_scale_cli_accepts_neutral_participant_id_configuration(tmp_path):
    from trace_agent.cli.consistency_scale import calibrate_consistency_scale

    config = _scale_only_config(tmp_path, [(1, "101", "validation"), (3, "103", "validation")])
    split = config["data"]["split_table"]
    split_path = Path(split["path"])
    split_path.write_text(split_path.read_text().replace("row_index,eid,split", "row_index,subject,split"))
    split["participant_id_column"] = "subject"

    record = json.loads(calibrate_consistency_scale(config).read_text())

    assert record["validation_rows"] == 2
    assert record["log_reference"] == pytest.approx(np.log(1.99), abs=1e-7)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "empty required splits.*validation"),
        ([(0, "100", "test")], "empty required splits.*validation"),
        ([(0, "100", "inference")], "unknown split"),
        ([(1.5, "101", "validation")], "row_index must contain integers"),
        ([(1, "101", "validation"), (1, "101", "validation")], "duplicate row_index"),
        ([(1, "101", "validation"), (3, "101", "validation")], "duplicate participant ID"),
        ([(7, "107", "validation")], "outside"),
        ([(1, "999", "validation")], "participant ID mismatch"),
    ],
)
def test_scale_cli_rejects_invalid_validation_subsets(tmp_path, rows, message):
    from trace_agent.cli.consistency_scale import calibrate_consistency_scale

    config = _scale_only_config(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        calibrate_consistency_scale(config)
    assert not Path(config["consistency"]["longitudinal_scale_path"]).exists()


def test_default_split_loader_still_requires_all_supervised_roles(tmp_path):
    from trace_agent.data.split_table import load_split_table

    config = _scale_only_config(tmp_path, [(index, str(100 + index), "validation") for index in range(7)])
    with pytest.raises(ValueError, match="empty required splits.*train.*test"):
        load_split_table(config["data"]["split_table"]["path"], participant_ids=np.arange(100, 107))
