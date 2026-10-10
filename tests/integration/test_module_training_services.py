import json

import numpy as np
import pytest

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.training.modules.score_bundle import load_module_bank, load_oof_cache


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_native_module_training_publishes_train_only_bank(tmp_path, task):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    from trace_agent.training.modules.service import prepare_module_data, train_modules

    write_module_inputs(tmp_path, task=task)
    config = module_config(tmp_path, task=task)
    prepare_module_data(config)
    (tmp_path / "outcome.csv").unlink()
    path = train_modules(config)
    manifest, models = load_module_bank(tmp_path / "bank", task=task)
    cache = load_oof_cache(tmp_path / "bank", manifest)
    assert path.name == "module_bank_manifest.json"
    assert set(models) == {("M1", "D1"), ("M2", "D1")}
    assert cache["scores"].shape == (120, 2, 1)
    np.testing.assert_array_equal(cache["train_rows"], np.arange(120))
    assert manifest["training"]["test_rows_used"] == 0
    assert manifest["training"]["validation_rows_used"] == 0
    assert manifest["settings"]["insufficient_data_policy"] == "skip"
    assert manifest["settings"]["inner_validation_fraction"] == 0.2
    assert manifest["settings"]["min_positive_rate"] == 0.001
    for record in manifest["pairs"]:
        assert set(record["metadata"]["final_fit"]["fit_rows"]) == set(range(120))


@pytest.mark.parametrize("change", ["target", "features", "split"])
def test_stale_prepared_inputs_error(tmp_path, change):
    from trace_agent.training.modules.service import prepare_module_data, train_modules

    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    prepare_module_data(config)
    if change == "target":
        np.save(tmp_path / "prepared/cohort_y.npy", np.zeros((180, 1)))
    elif change == "features":
        path = tmp_path / "features.csv"
        path.write_text(path.read_text() + "\n")
    else:
        import pandas as pd

        path = tmp_path / "splits.csv"
        frame = pd.read_csv(path)
        frame.loc[[0, 120], "split"] = ["validation", "train"]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        train_modules(config)
    assert not (tmp_path / "bank/module_bank_manifest.json").exists()


def test_all_pairs_skipped_are_explicit(tmp_path):
    from trace_agent.training.modules.service import prepare_module_data, train_modules

    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    config["training"]["modules"]["min_positive"] = 1000
    prepare_module_data(config)
    path = train_modules(config)
    manifest, models = load_module_bank(path.parent, task="prevalent")
    assert models == {}
    assert all(record["status"] == "skipped" for record in manifest["pairs"])
    cache = load_oof_cache(path.parent, manifest)
    assert not cache["pred_mask"].any()
    assert np.isnan(cache["scores"]).all()


def test_failed_publish_cannot_be_loaded_as_a_new_valid_bank(tmp_path, monkeypatch):
    from trace_agent.training.modules import service

    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    service.prepare_module_data(config)
    service.train_modules(config)
    old_manifest = json.loads((tmp_path / "bank/module_bank_manifest.json").read_text())
    config["training"]["modules"]["n_estimators"] = 4
    real_dump = service.dump_trace_artifact
    calls = 0

    def failing_dump(artifact, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted publish")
        return real_dump(artifact, path)

    monkeypatch.setattr(service, "dump_trace_artifact", failing_dump)
    with pytest.raises(RuntimeError, match="interrupted publish"):
        service.train_modules(config)
    assert json.loads((tmp_path / "bank/module_bank_manifest.json").read_text()) == old_manifest
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_module_bank(tmp_path / "bank", task="prevalent")


@pytest.mark.parametrize("id_column", ["eid", "participant_id", "subject"])
def test_external_fold_source_is_recorded(tmp_path, id_column):
    import pandas as pd

    from trace_agent.training.modules.service import prepare_module_data, train_modules

    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "row_index": np.arange(120),
            id_column: np.arange(1000, 1120),
            "iv_fold": np.tile(np.arange(1, 6), 24),
        }
    ).to_csv(path, index=False)
    config["training"]["modules"]["fold_table"] = {"path": str(path), "fold_column": "iv_fold"}
    if id_column == "subject":
        config["training"]["modules"]["fold_table"]["participant_id_column"] = id_column
    prepare_module_data(config)
    manifest = json.loads(train_modules(config).read_text())
    assert manifest["fold_source"]["sha256"]
    cache = load_oof_cache(tmp_path / "bank", manifest)
    np.testing.assert_array_equal(cache["folds"], np.tile(np.arange(1, 6), 24))
