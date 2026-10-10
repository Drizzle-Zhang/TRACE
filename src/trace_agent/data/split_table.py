from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from trace_agent.contracts.task import SplitRole

_SUPERVISED_ROLES = (
    SplitRole.TRAIN,
    SplitRole.VALIDATION,
    SplitRole.TEST,
)
_SPLIT_ALIASES = {
    "train": SplitRole.TRAIN,
    "validation": SplitRole.VALIDATION,
    "val": SplitRole.VALIDATION,
    "test": SplitRole.TEST,
}


@dataclass(frozen=True)
class SplitColumns:
    row_index: str = "row_index"
    eid: str = "participant_id"
    split: str = "split"

    def __post_init__(self) -> None:
        values = (str(self.row_index), str(self.eid), str(self.split))
        if any(not value.strip() for value in values):
            raise ValueError("split-table column names must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("split-table column names must be distinct")


@dataclass(frozen=True)
class DatasetSplits:
    table_path: Path
    row_indices: Mapping[SplitRole, np.ndarray]
    eid_values: Mapping[SplitRole, np.ndarray]
    file_sha256: str
    canonical_sha256: str

    def indices(self, role: SplitRole | str) -> np.ndarray:
        resolved = _supervised_role(role)
        return np.asarray(self.row_indices[resolved], dtype=np.int64).copy()

    def eids(self, role: SplitRole | str) -> np.ndarray:
        resolved = _supervised_role(role)
        return np.asarray(self.eid_values[resolved], dtype=np.str_).copy()

    def count(self, role: SplitRole | str) -> int:
        return int(self.indices(role).size)


def _supervised_role(role: SplitRole | str) -> SplitRole:
    resolved = SplitRole.parse(role)
    if resolved not in _SUPERVISED_ROLES:
        raise ValueError("inference is not a supervised dataset split")
    return resolved


def id_column_from_config(values: Mapping[str, object]) -> str:
    """Resolve the participant ID column, accepting the legacy configuration key."""
    if (
        "participant_id_column" in values
        and "eid_column" in values
        and values["participant_id_column"] != values["eid_column"]
    ):
        raise ValueError("participant_id_column and eid_column must name the same column")
    return str(values.get("participant_id_column", values.get("eid_column", "participant_id")))


def read_eid_table(
    path: str | Path, *, eid_column: str = "participant_id", label: str = "table"
) -> pd.DataFrame:
    """Preserve participant ID text, with a legacy column alias for the default name."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        identifier_columns = (eid_column, "eid") if eid_column == "participant_id" else (eid_column,)
        frame = pd.read_csv(
            path,
            sep="," if suffix == ".csv" else "\t",
            converters={name: lambda value: value if value.strip() else None for name in identifier_columns},
        )
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported {label} format {suffix!r}; expected .csv, .tsv, .tab, or .parquet")
    if eid_column == "participant_id" and eid_column not in frame and "eid" in frame:
        frame = frame.rename(columns={"eid": eid_column})
    return frame


def _normalize_eid(value: object, *, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{label} contains a missing participant ID")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} contains a boolean participant ID")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"{label} contains a non-integral numeric participant ID {value!r}")
        return str(int(number))
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} contains an empty participant ID")
    return normalized


def normalize_eids(values: np.ndarray, *, label: str = "participant ids") -> np.ndarray:
    flattened = np.asarray(values).reshape(-1)
    return np.asarray(
        [_normalize_eid(value, label=label) for value in flattened.tolist()],
        dtype=np.str_,
    )


def canonical_eids_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            normalize_eids(left, label="left participant ids"),
            normalize_eids(right, label="right participant ids"),
        )
    )


def resolve_eid_rows(values: np.ndarray, *, participant_ids: np.ndarray, label: str) -> np.ndarray:
    """Resolve unique table participant IDs to positions on a unique participant axis."""
    axis = pd.Index(normalize_eids(participant_ids, label="participant axis"))
    if axis.has_duplicates:
        duplicates = axis[axis.duplicated(keep=False)].tolist()
        raise ValueError(f"participant axis contains duplicate participant ID values: {duplicates[:10]}")
    eids = pd.Index(normalize_eids(values, label=label))
    if eids.has_duplicates:
        duplicates = eids[eids.duplicated(keep=False)].tolist()
        raise ValueError(f"{label} contains duplicate participant ID values: {duplicates[:10]}")
    rows = axis.get_indexer(eids)
    if np.any(rows < 0):
        outside = eids[rows < 0].tolist()
        raise ValueError(f"{label} contains participant IDs outside the participant axis: {outside[:10]}")
    return rows.astype(np.int64)


def _canonicalize(frame: pd.DataFrame, *, columns: SplitColumns, participant_ids: np.ndarray) -> pd.DataFrame:
    has_rows = columns.row_index in frame.columns
    requested = (columns.eid, columns.split)
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise ValueError(f"split table is missing required columns: {missing}")

    selected = frame.loc[:, [columns.row_index, *requested] if has_rows else list(requested)].copy()
    selected.columns = ["row_index", "eid", "split"] if has_rows else ["eid", "split"]
    if selected.isna().any(axis=None):
        locations = np.argwhere(selected.isna().to_numpy())
        first_row, first_column = locations[0].tolist()
        raise ValueError(
            "split table contains a missing value at "
            f"record {first_row}, column {selected.columns[first_column]!r}"
        )

    if not has_rows:
        selected.insert(
            0,
            "row_index",
            resolve_eid_rows(
                selected["eid"].to_numpy(), participant_ids=participant_ids, label="split table"
            ),
        )

    try:
        numeric_rows = pd.to_numeric(selected["row_index"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_index must contain integers") from exc
    if not np.isfinite(numeric_rows).all() or not np.equal(numeric_rows, np.floor(numeric_rows)).all():
        raise ValueError("row_index must contain integers")
    selected["row_index"] = numeric_rows.astype(np.int64)
    if selected["row_index"].duplicated().any():
        duplicates = selected.loc[selected["row_index"].duplicated(keep=False), "row_index"].tolist()
        raise ValueError(f"split table contains duplicate row_index values: {duplicates[:10]}")

    normalized_splits = selected["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized_splits).difference(_SPLIT_ALIASES))
    if unknown:
        raise ValueError(f"split table contains unknown split values: {unknown}")
    selected["split"] = [_SPLIT_ALIASES[value].value for value in normalized_splits.tolist()]

    selected["eid"] = normalize_eids(
        selected["eid"].to_numpy(),
        label="split table",
    )
    if selected["eid"].duplicated().any():
        duplicates = selected.loc[selected["eid"].duplicated(keep=False), "eid"].tolist()
        raise ValueError(f"split table contains duplicate participant ID values: {duplicates[:10]}")
    return selected.sort_values("row_index", kind="stable").reset_index(drop=True)


def _validate_rows(
    frame: pd.DataFrame,
    *,
    n_rows: int,
    require_complete: bool,
    required_roles: tuple[SplitRole, ...],
) -> None:
    rows = frame["row_index"].to_numpy(dtype=np.int64)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n_rows):
        bad = rows[(rows < 0) | (rows >= n_rows)]
        raise ValueError(f"split-table row_index values are outside [0, {n_rows}): {bad[:10].tolist()}")
    if require_complete and not np.array_equal(rows, np.arange(n_rows, dtype=np.int64)):
        expected = set(range(n_rows))
        actual = set(rows.tolist())
        missing = sorted(expected.difference(actual))
        raise ValueError(
            "split table must cover every score-bundle row exactly once; "
            f"missing row_index values: {missing[:10]}"
        )
    roles = set(frame["split"].tolist())
    missing_roles = [role.value for role in required_roles if role.value not in roles]
    if missing_roles:
        raise ValueError(f"split table has empty required splits: {missing_roles}")


def _validate_eid_alignment(frame: pd.DataFrame, *, participant_ids: np.ndarray) -> None:
    bundle_eids = normalize_eids(participant_ids, label="score bundle")
    rows = frame["row_index"].to_numpy(dtype=np.int64)
    table_eids = frame["eid"].to_numpy(dtype=np.str_)
    mismatches = np.flatnonzero(table_eids != bundle_eids[rows])
    if mismatches.size:
        position = int(mismatches[0])
        row_index = int(rows[position])
        raise ValueError(
            "split-table participant ID mismatch at "
            f"row_index={row_index}: table participant ID={table_eids[position]}, "
            f"bundle participant ID={bundle_eids[row_index]}"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_splits(frame: pd.DataFrame, *, table_path: Path) -> DatasetSplits:
    row_indices: dict[SplitRole, np.ndarray] = {}
    eid_values: dict[SplitRole, np.ndarray] = {}
    for role in _SUPERVISED_ROLES:
        selected = frame["split"] == role.value
        row_indices[role] = frame.loc[selected, "row_index"].to_numpy(dtype=np.int64)
        eid_values[role] = frame.loc[selected, "eid"].to_numpy(dtype=np.str_)
    canonical = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return DatasetSplits(
        table_path=table_path,
        row_indices=MappingProxyType(row_indices),
        eid_values=MappingProxyType(eid_values),
        file_sha256=_sha256_bytes(table_path.read_bytes()),
        canonical_sha256=_sha256_bytes(canonical),
    )


def load_split_table(
    path: str | Path,
    *,
    participant_ids: np.ndarray,
    columns: SplitColumns | None = None,
    require_complete: bool = True,
    required_roles: Sequence[SplitRole | str] = _SUPERVISED_ROLES,
) -> DatasetSplits:
    """Load split assignments, deriving absent row indices from participant IDs.

    A present row-index column is always validated against the participant axis.
    """
    resolved = Path(path).expanduser().resolve()
    participant_array = np.asarray(participant_ids).reshape(-1)
    columns = columns or SplitColumns()
    canonical = _canonicalize(
        read_eid_table(resolved, eid_column=columns.eid, label="split-table"),
        columns=columns,
        participant_ids=participant_array,
    )
    _validate_rows(
        canonical,
        n_rows=int(participant_array.size),
        require_complete=bool(require_complete),
        required_roles=tuple(_supervised_role(role) for role in required_roles),
    )
    _validate_eid_alignment(canonical, participant_ids=participant_array)
    return _build_splits(canonical, table_path=resolved)
