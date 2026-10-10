import numpy as np
import pandas as pd
import pytest

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.training.modules.score_bundle import load_module_bank, load_oof_cache
from trace_agent.training.modules.service import prepare_module_data, train_modules


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_module_fitting_is_independent_of_validation_test_labels(tmp_path, task):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    write_module_inputs(tmp_path, task=task)
    config = module_config(tmp_path, task=task)
    prepare_module_data(config)
    train_modules(config)
    first_manifest, first_models = load_module_bank(tmp_path / "bank", task=task)
    first_cache = load_oof_cache(tmp_path / "bank", first_manifest)
    features = load_module_features(load_catalog(tmp_path / "catalog.yaml"))
    first_predictions = {key: model.predict(features.matrices[key]) for key, model in first_models.items()}
    outcomes = pd.read_csv(tmp_path / "outcome.csv")
    outcomes.loc[120:, "target_y"] = 1 - outcomes.loc[120:, "target_y"]
    outcomes.loc[120:, "BL2Target_yrs"] = (
        np.where(outcomes.loc[120:, "target_y"] == 1, -4.0, 15.0) if task == "prevalent" else 8.0
    )
    # Mask additional validation/test labels; they must not affect training or OOF.
    outcomes.loc[[120, 150], "target_y"] = np.nan
    outcomes.to_csv(tmp_path / "outcome.csv", index=False)
    prepare_module_data(config)
    train_modules(config)
    second_manifest, second_models = load_module_bank(tmp_path / "bank", task=task)
    second_cache = load_oof_cache(tmp_path / "bank", second_manifest)
    np.testing.assert_allclose(first_cache["scores"], second_cache["scores"], equal_nan=True)
    np.testing.assert_array_equal(first_cache["pred_mask"], second_cache["pred_mask"])
    np.testing.assert_array_equal(first_cache["folds"], second_cache["folds"])
    assert set(first_models) == set(second_models)
    for key, model in second_models.items():
        np.testing.assert_allclose(first_predictions[key], model.predict(features.matrices[key]))
