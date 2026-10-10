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
    "suffix, separator", [(".csv", ","), (".tsv", "\t"), (".tab", "\t"), (".parquet", None)]
)
def test_eid_only_split_table_derives_rows_and_preserves_canonical_identity(
    tmp_path: Path, suffix: str, separator: str | None
) -> None:
    ids = np.array(["002", "NA", "001", "004"])
    frame = pd.DataFrame(
        {
            "participant": ["004", "001", "002", "NA"],
            "role": ["test", "val", "train", "validation"],
        }
    )
    path = tmp_path / f"splits{suffix}"
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, sep=separator, index=False)
    columns = SplitColumns(row_index="row", eid="participant", split="role")

    derived = load_split_table(path, participant_ids=ids, columns=columns)
    explicit_path = tmp_path / "explicit.csv"
    frame.insert(0, "row", [3, 2, 0, 1])
    frame.to_csv(explicit_path, index=False)
    explicit = load_split_table(explicit_path, participant_ids=ids, columns=columns)

    assert derived.indices("train").tolist() == [0]
    assert derived.indices("validation").tolist() == [1, 2]
    assert derived.eids("validation").tolist() == ["NA", "001"]
    assert derived.indices("test").tolist() == [3]
    assert derived.canonical_sha256 == explicit.canonical_sha256
    assert derived.file_sha256 != explicit.file_sha256


def test_eid_only_split_table_supports_validation_subsets(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    pd.DataFrame({"eid": [104, 102], "split": ["validation", "val"]}).to_csv(path, index=False)

    splits = load_split_table(
        path,
        participant_ids=np.array([101, 102, 103, 104]),
        require_complete=False,
        required_roles=(SplitRole.VALIDATION,),
    )

    assert splits.indices("validation").tolist() == [1, 3]
    assert splits.eids("validation").tolist() == ["102", "104"]
    assert splits.count("train") == splits.count("test") == 0


@pytest.mark.parametrize(
    ("frame", "ids", "message"),
    [
        (
            pd.DataFrame({"eid": [101, 101, 103], "split": ["train", "val", "test"]}),
            [101, 102, 103],
            "duplicate participant ID",
        ),
        (
            pd.DataFrame({"eid": [101, None, 103], "split": ["train", "val", "test"]}),
            [101, 102, 103],
            "missing value",
        ),
        (
            pd.DataFrame({"eid": [101, 102, 999], "split": ["train", "val", "test"]}),
            [101, 102, 103],
            "outside.*participant axis",
        ),
        (
            pd.DataFrame({"eid": [101, 102, 103], "split": ["train", "holdout", "test"]}),
            [101, 102, 103],
            "unknown split",
        ),
        (
            pd.DataFrame({"eid": [101, 102, 103], "split": ["train", "val", "test"]}),
            [101, 102, 103, 104],
            "cover every score-bundle row",
        ),
        (
            pd.DataFrame({"eid": [101, 102, 103], "split": ["validation"] * 3}),
            [101, 102, 103],
            "empty required splits.*train.*test",
        ),
        (
            pd.DataFrame({"eid": [101, 102], "split": ["train", "validation"]}),
            [101, 102, 102],
            "participant axis contains duplicate participant ID",
        ),
    ],
)
def test_eid_only_split_table_rejects_invalid_identity_or_roles(
    tmp_path: Path, frame: pd.DataFrame, ids: list[int], message: str
) -> None:
    path = tmp_path / "splits.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_split_table(path, participant_ids=np.asarray(ids))


def test_present_row_index_cannot_be_replaced_by_eid_derivation(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    frame = _complete_frame()
    frame.loc[0, "row_index"] = np.nan
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing value.*row_index"):
        load_split_table(path, participant_ids=np.array([101, 102, 103]))


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
            "duplicate participant ID",
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
        match=r"row_index=0.*table participant ID=999.*bundle participant ID=101",
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
