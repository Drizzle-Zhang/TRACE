from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_folds_only_cover_train_and_are_deterministic():
    from trace_agent.training.modules.folds import build_outer_folds

    ids = np.arange(20).astype(str)
    rows = np.arange(15)
    first = build_outer_folds(ids, rows, seed=7)
    second = build_outer_folds(ids, rows, seed=7)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == {1, 2, 3, 4, 5}
    assert first.shape == rows.shape
    assert np.array_equal(np.bincount(first)[1:], np.full(5, 3))


def test_external_iv_fold_keeps_train_row_order(tmp_path):
    from trace_agent.training.modules.folds import build_outer_folds

    path = tmp_path / "folds.csv"
    rows = np.array([1, 3, 5, 7, 9])
    pd.DataFrame({"row_index": rows[::-1], "eid": rows[::-1], "iv_fold": [5, 4, 3, 2, 1]}).to_csv(
        path, index=False
    )
    assignment = build_outer_folds(np.arange(12), rows, path=path, fold_column="iv_fold")
    assert assignment.tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("change", ["eid", "drop", "extra", "duplicate", "fractional", "empty_fold"])
def test_external_folds_reject_bad_rows_or_folds(tmp_path, change):
    from trace_agent.training.modules.folds import build_outer_folds

    path = tmp_path / "folds.csv"
    frame = pd.DataFrame({"row_index": range(5), "eid": range(5), "iv_fold": range(1, 6)}).astype(float)
    if change == "eid":
        frame.loc[0, "eid"] = 999
    elif change == "drop":
        frame = frame.iloc[1:]
    elif change == "extra":
        frame.loc[5] = [5, 5, 1]
    elif change == "duplicate":
        frame.loc[0, "row_index"] = 1
    elif change == "fractional":
        frame.loc[0, "iv_fold"] = 1.5
    else:
        frame.loc[0, "iv_fold"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        build_outer_folds(np.arange(7), np.arange(5), path=path, fold_column="iv_fold")


@pytest.mark.parametrize("rows", [np.arange(4), np.array([0, 1, 2, 3, 3]), np.arange(5) + 4])
def test_invalid_train_rows_error(rows):
    from trace_agent.training.modules.folds import build_outer_folds

    with pytest.raises(ValueError):
        build_outer_folds(np.arange(7), rows)
