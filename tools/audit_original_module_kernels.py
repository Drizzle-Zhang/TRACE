#!/usr/bin/env python3
"""Rebuild independent module-kernel evidence without original top-level execution.

Run with PYTHONPATH=src and explicitly supplied read-only source paths. This tool
reads the two verified sources, fits synthetic data, and prints JSON; it never
opens a cohort, imports an original script, or writes files.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

LIGHTGBM_SHA256 = "62fac5a04483c04f107040ebaeae84622021d9119c2780a3cd51c4938196daf2"
COX_SHA256 = "85c887b818b68a8107725ac215af1d0b7a822aa93b2dde221a21e6cd518664a8"
FUNCTIONS = {"model_training", "pred_train_iter"}


def checked_source(path: Path, expected_sha256: str) -> bytes:
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return source


def extract_functions(path: Path, expected_sha256: str, namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(checked_source(path, expected_sha256).decode("utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in definitions} != FUNCTIONS or len(definitions) != len(FUNCTIONS):
        raise ValueError("verified source must contain exactly the two original training functions")
    isolated = dict(namespace)
    exec(  # noqa: S102 -- SHA-256 gated source; compile only the two function definitions.
        compile(ast.Module(body=definitions, type_ignores=[]), "<verified-original-functions>", "exec"),
        isolated,
    )
    return isolated


def approved_cox_function(source: bytes, namespace: dict[str, Any]) -> Any:
    """Apply the documented split/refit changes to the original constructor AST.

    This is explicitly an adapted-source reference, never a claim that the native
    Cox predictions equal the unchanged original algorithm.
    """
    tree = ast.parse(source.decode("utf-8"))
    function = copy.deepcopy(
        next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "model_training"
        )
    )
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "train_test_split"
        ):
            node.keywords.append(
                ast.keyword(arg="stratify", value=ast.parse("(y_train > 0).astype(int)", mode="eval").body)
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            for keyword in node.keywords:
                if keyword.arg == "verbose":
                    keyword.value = ast.Constant(False)
    constructor = copy.deepcopy(
        next(
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "my_xgb" for target in node.targets)
        )
    )
    if not isinstance(constructor.value, ast.Call):
        raise ValueError("original Cox constructor has changed")  # noqa: TRY004 -- source AST schema mismatch
    constructor.value.keywords = [
        keyword for keyword in constructor.value.keywords if keyword.arg != "early_stopping_rounds"
    ]
    for keyword in constructor.value.keywords:
        if keyword.arg == "n_estimators":
            keyword.value = ast.Name(id="selected_rounds", ctx=ast.Load())
    prediction_position = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "test_pred" for target in node.targets)
    )
    function.body[prediction_position:prediction_position] = [
        *ast.parse("selected_rounds = my_xgb.best_iteration + 1").body,
        constructor,
        *ast.parse("my_xgb.fit(X_train, y_train, verbose=False)").body,
    ]
    isolated = dict(namespace)
    exec(  # noqa: S102 -- adapt only the verified original function AST, never top-level code.
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<adapted-original-cox-function>",
            "exec",
        ),
        isolated,
    )
    return isolated["model_training"]


def audit(lightgbm_source: Path, cox_source: Path) -> dict[str, Any]:
    # Reject changed sources before importing learner runtimes or compiling code.
    checked_source(lightgbm_source, LIGHTGBM_SHA256)
    cox_bytes = checked_source(cox_source, COX_SHA256)
    import lightgbm
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBRegressor

    from trace_agent.risk_skills.module_model import ModuleModel
    from trace_agent.training.modules.crossfit import crossfit_pair
    from trace_agent.training.modules.longitudinal import fit_longitudinal_module
    from trace_agent.training.modules.prevalent import fit_prevalent_module
    from trace_agent.training.modules.settings import ModuleTrainingSettings

    lgb_functions = extract_functions(
        lightgbm_source,
        LIGHTGBM_SHA256,
        {"pd": pd, "LGBMClassifier": LGBMClassifier, "CalibratedClassifierCV": CalibratedClassifierCV},
    )
    cox_namespace = {"pd": pd, "XGBRegressor": XGBRegressor}
    cox_functions = extract_functions(cox_source, COX_SHA256, cox_namespace)
    adapted_cox = approved_cox_function(cox_bytes, cox_namespace)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rng = np.random.default_rng(29017)
        matrix = rng.normal(size=(500, 4))
        matrix[::11, 2] = np.nan
        labels = (matrix[:, 1] + rng.normal(size=500) > 0.2).astype(float)
        names = ("z", "k", "c", "v")
        frame = pd.DataFrame(matrix, columns=names)
        frame["eid"] = np.arange(500) + 10000
        frame["diagnose_label"] = labels
        train, test, step2 = frame.iloc[:420], frame.iloc[420:460], frame.iloc[460:]
        test_old, step2_old, _ = lgb_functions["model_training"](train, test, step2, list(names), 126)
        fitted = fit_prevalent_module(
            matrix[:420], labels[:420], np.arange(420), ModuleTrainingSettings(seed=126)
        )
        final_new = ModuleModel("prevalent", "m", "d", names, fitted.estimator).predict(matrix[420:])
        final_old = np.r_[test_old, step2_old]
        np.testing.assert_allclose(final_new, final_old, rtol=0, atol=2e-7)
        folds = np.arange(420) % 5 + 1
        old_parts = [
            lgb_functions["pred_train_iter"](
                train,
                list(names),
                train.eid[folds != fold].tolist(),
                train.eid[folds == fold].tolist(),
                127,
                "m",
            )
            for fold in range(1, 6)
        ]
        oof_old = pd.concat(old_parts).set_index("eid").loc[train.eid, "m_pred"].to_numpy()
        oof_new = crossfit_pair(
            task="prevalent",
            module="m",
            endpoint="d",
            feature_names=names,
            matrix=matrix[:420],
            labels=labels[:420],
            label_mask=np.ones(420, dtype=bool),
            row_indices=np.arange(420),
            folds=folds,
            settings=ModuleTrainingSettings(seed=127),
        )
        np.testing.assert_allclose(oof_new.oof_scores, oof_old, rtol=0, atol=2e-7)

        rng = np.random.default_rng(710391)
        matrix_cox = rng.normal(size=(260, 3))
        matrix_cox[::13, 1] = np.nan
        events = rng.uniform(size=260) > 0.45
        signed = np.where(events, 1 + rng.uniform(size=260) * 8, -1 - rng.uniform(size=260) * 11)
        cox_names = ("b", "k", "a")
        frame_cox = pd.DataFrame(matrix_cox, columns=cox_names)
        frame_cox["eid"] = np.arange(260)
        frame_cox["model_yrs"] = signed
        train_cox, probe = frame_cox.iloc[:220], frame_cox.iloc[220:]
        _, _, original_model = cox_functions["model_training"](train_cox, probe, probe, list(cox_names), 316)
        hr_adapted, _, adapted_model = adapted_cox(train_cox, probe, probe, list(cox_names), 316)
        native = fit_longitudinal_module(
            matrix_cox[:220],
            signed[:220],
            np.arange(220),
            ModuleTrainingSettings.from_mapping({"seed": 316}, task="longitudinal"),
        )
        margin_adapted = adapted_model.predict(matrix_cox[220:], output_margin=True)
        margin_native = native.estimator.predict(matrix_cox[220:], output_margin=True)
        hr_native = native.estimator.predict(matrix_cox[220:])
        np.testing.assert_allclose(margin_native, margin_adapted, rtol=0, atol=2e-7)
        np.testing.assert_allclose(hr_native, hr_adapted, rtol=2e-7, atol=0)
        assert adapted_model.n_estimators == native.metadata["selected_rounds"]
    return {
        "source_sha256": {"lightgbm": LIGHTGBM_SHA256, "cox": COX_SHA256},
        "versions": {item.__name__: item.__version__ for item in (np, pd, sklearn, lightgbm, xgboost)},
        "prevalent": {
            "synthetic_rows": 500,
            "features": list(names),
            "train_rows": 420,
            "final_seed": 126,
            "oof_seed": 127,
            "final_probability_max_abs_error": float(np.max(np.abs(final_old - final_new))),
            "oof_probability_max_abs_error": float(np.max(np.abs(oof_old - oof_new.oof_scores))),
        },
        "cox_adapted_source": {
            "scope": "documented stratified selection/full refit, not unchanged original parity",
            "synthetic_rows": 260,
            "features": list(cox_names),
            "train_rows": 220,
            "seed": 316,
            "original_rounds": int(original_model.best_iteration + 1),
            "adapted_rounds": int(adapted_model.n_estimators),
            "native_rounds": int(native.metadata["selected_rounds"]),
            "margin_max_abs_error": float(np.max(np.abs(margin_adapted - margin_native))),
            "hr_max_abs_error": float(np.max(np.abs(np.asarray(hr_adapted) - hr_native))),
            "unchanged_original_margin_max_abs_difference": float(
                np.max(np.abs(original_model.predict(matrix_cox[220:], output_margin=True) - margin_native))
            ),
        },
        "warnings_captured": len(captured),
        "original_top_level_executed": False,
        "cohort_files_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-source", required=True, type=Path)
    parser.add_argument("--cox-source", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        report = audit(arguments.lightgbm_source, arguments.cox_source)
    except (ValueError, AssertionError) as error:
        parser.exit(1, f"audit failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
