from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trace_agent.contracts.task import SplitRole
from trace_agent.data.split_table import SplitColumns, load_split_table


def _complete_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "eid": [101, 102, 103],
            "split": ["train", "validation", "test"],
        }
    )


def test_load_split_table_validates_eids_and_sorts_rows(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "row_index": [2, 0, 3, 1],
            "eid": [103, 101, 104, 102],
            "split": ["test", "train", "validation", "val"],
        }
    ).to_csv(path, index=False)

    splits = load_split_table(
        path,
        participant_ids=np.array([101, 102, 103, 104]),
    )

    assert splits.indices(SplitRole.TRAIN).tolist() == [0]
    assert splits.indices(SplitRole.VALIDATION).tolist() == [1, 3]
    assert splits.indices(SplitRole.TEST).tolist() == [2]
    assert splits.eids(SplitRole.VALIDATION).tolist() == ["102", "104"]
    assert len(splits.file_sha256) == 64
    assert len(splits.canonical_sha256) == 64


def test_load_split_table_supports_explicit_column_names(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    frame = _complete_frame().rename(columns={"row_index": "row", "eid": "participant", "split": "set"})
    frame.to_csv(path, index=False)

    splits = load_split_table(
        path,
        participant_ids=np.array([101, 102, 103]),
        columns=SplitColumns(row_index="row", eid="participant", split="set"),
    )

    assert splits.indices("test").tolist() == [2]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"row_index": [0], "eid": [101]}),
            "missing required columns",
        ),
        (
            pd.DataFrame(
                {
                    "row_index": [0.5, 1, 2],
                    "eid": [101, 102, 103],
                    "split": ["train", "validation", "test"],
                }
            ),
            "row_index must contain integers",
        ),
        (
            pd.DataFrame(
                {
                    "row_index": [0, 0, 2],
                    "eid": [101, 102, 103],
                    "split": ["train", "validation", "test"],
                }
            ),
            "duplicate row_index",
        ),
        (
            pd.DataFrame(
                {
                    "row_index": [0, 1, 2],
                    "eid": [101, 102, 103],
                    "split": ["train", "holdout", "test"],
                }
            ),
            "unknown split",
        ),
        (
            pd.DataFrame(
                {
                    "row_index": [0, 1, 2],
                    "eid": [101, 101, 103],
                    "split": ["train", "validation", "test"],
                }
            ),
            "duplicate eid",
        ),
        (
            pd.DataFrame(
                {
                    "row_index": [0, 1, 3],
                    "eid": [101, 102, 103],
                    "split": ["train", "validation", "test"],
                }
            ),
            "outside",
        ),
    ],
)
def test_load_split_table_rejects_invalid_records(
    tmp_path: Path,
    frame: pd.DataFrame,
    message: str,
) -> None:
    path = tmp_path / "splits.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_split_table(path, participant_ids=np.array([101, 102, 103]))


def test_load_split_table_requires_complete_coverage(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "eid": [101, 102, 103],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="cover every score-bundle row"):
        load_split_table(path, participant_ids=np.array([101, 102, 103, 104]))


def test_load_split_table_rejects_row_eid_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    frame = _complete_frame()
    frame.loc[0, "eid"] = 999
    frame.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match=r"row_index=0.*table eid=999.*bundle eid=101",
    ):
        load_split_table(path, participant_ids=np.array([101, 102, 103]))


def test_load_split_table_reads_tsv(tmp_path: Path) -> None:
    path = tmp_path / "splits.tsv"
    _complete_frame().to_csv(path, sep="\t", index=False)

    splits = load_split_table(path, participant_ids=np.array([101, 102, 103]))

    assert splits.indices("test").tolist() == [2]


def test_load_split_table_reads_parquet_when_engine_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "splits.parquet"
    _complete_frame().to_parquet(path, index=False)

    splits = load_split_table(path, participant_ids=np.array([101, 102, 103]))

    assert splits.indices("validation").tolist() == [1]


def test_load_split_table_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported split-table format"):
        load_split_table(path, participant_ids=np.array([101, 102, 103]))
