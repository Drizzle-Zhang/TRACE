"""Explicit raw inputs for native per-module model training and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trace_agent.data.split_table import id_column_from_config, normalize_eids, read_eid_table
from trace_agent.runtime.config import load_config
from trace_agent.runtime.provenance import file_provenance


@dataclass(frozen=True)
class FeatureTableSpec:
    id: str
    path: Path
    eid_column: str = "participant_id"


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    feature_table: str


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    outcome_path: Path
    eid_column: str = "participant_id"
    target_column: str = "target_y"
    time_column: str = "BL2Target_yrs"


@dataclass(frozen=True)
class PairSpec:
    module: str
    endpoint: str
    features_path: Path
    feature_column: str = "feature_code"


@dataclass(frozen=True)
class ModuleCatalog:
    path: Path
    participants_path: Path
    eid_column: str
    feature_tables: tuple[FeatureTableSpec, ...]
    modules: tuple[ModuleSpec, ...]
    endpoints: tuple[EndpointSpec, ...]
    pairs: tuple[PairSpec, ...]


def _records(values: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    items = values.get(key)
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise ValueError(f"catalog.{key} must be a list of mappings")
    return items


def _validate_catalog_shape(values: Mapping[str, Any]) -> None:
    accepted = {"version", "_meta", "participants", "feature_tables", "modules", "endpoints", "pairs"}
    if set(values).difference(accepted):
        raise ValueError("unknown catalog root fields")
    participant = values.get("participants")
    if not isinstance(participant, Mapping) or not participant.get("path"):
        raise ValueError("catalog.participants.path is required")
    specifications = {
        "feature_tables": ({"id", "path"}, {"id", "path", "participant_id_column", "eid_column"}),
        "modules": ({"name", "feature_table"}, {"name", "feature_table"}),
        "endpoints": (
            {"name", "outcome_path"},
            {"name", "outcome_path", "participant_id_column", "eid_column", "target_column", "time_column"},
        ),
        "pairs": (
            {"module", "endpoint", "features_path"},
            {"module", "endpoint", "features_path", "feature_column"},
        ),
    }
    for key, (required, allowed) in specifications.items():
        for index, item in enumerate(_records(values, key)):
            if not required.issubset(item) or set(item).difference(allowed):
                raise ValueError(f"catalog.{key}[{index}] has missing or unknown fields")
            if any(not isinstance(value, str) or not value.strip() for value in item.values()):
                raise ValueError(f"catalog.{key}[{index}] values must be non-empty strings")
    if set(participant).difference({"path", "participant_id_column", "eid_column"}):
        raise ValueError("unknown participants fields")
    if any(not isinstance(value, str) or not value.strip() for value in participant.values()):
        raise ValueError("participants values must be non-empty strings")


def _require_unique(names: tuple[str, ...], label: str) -> None:
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{label} names must be non-empty and unique")


def load_catalog(path: str | Path) -> ModuleCatalog:
    """Resolve declared paths without opening features or outcome files."""
    resolved = Path(path).expanduser().resolve()
    values = load_config(resolved)
    _validate_catalog_shape(values)
    participant = values["participants"]
    tables = tuple(
        FeatureTableSpec(str(item["id"]), Path(item["path"]), id_column_from_config(item))
        for item in _records(values, "feature_tables")
    )
    modules = tuple(
        ModuleSpec(str(item["name"]), str(item["feature_table"])) for item in _records(values, "modules")
    )
    endpoints = tuple(
        EndpointSpec(
            str(item["name"]),
            Path(item["outcome_path"]),
            id_column_from_config(item),
            str(item.get("target_column", "target_y")),
            str(item.get("time_column", "BL2Target_yrs")),
        )
        for item in _records(values, "endpoints")
    )
    pairs = tuple(
        PairSpec(
            str(item["module"]),
            str(item["endpoint"]),
            Path(item["features_path"]),
            str(item.get("feature_column", "feature_code")),
        )
        for item in _records(values, "pairs")
    )
    for names, label in (
        (tuple(item.id for item in tables), "feature table"),
        (tuple(item.name for item in modules), "module"),
        (tuple(item.name for item in endpoints), "endpoint"),
    ):
        _require_unique(names, label)
        if not names:
            raise ValueError(f"catalog requires at least one {label}")
    table_names = {item.id for item in tables}
    module_names = {item.name for item in modules}
    endpoint_names = {item.name for item in endpoints}
    if any(item.feature_table not in table_names for item in modules):
        raise ValueError("unknown feature table reference")
    keys = tuple((item.module, item.endpoint) for item in pairs)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate module/endpoint pair")
    if any(module not in module_names or endpoint not in endpoint_names for module, endpoint in keys):
        raise ValueError("unknown pair reference")
    return ModuleCatalog(
        resolved,
        Path(participant["path"]),
        id_column_from_config(participant),
        tables,
        modules,
        endpoints,
        pairs,
    )


@dataclass(frozen=True)
class ModuleFeatures:
    catalog: ModuleCatalog
    participant_ids: np.ndarray
    matrices: Mapping[tuple[str, str], np.ndarray]
    feature_names: Mapping[tuple[str, str], tuple[str, ...]]
    availability: np.ndarray
    provenance: Mapping[str, Any]


def load_module_features(catalog: ModuleCatalog) -> ModuleFeatures:
    """Align unique feature participant IDs; availability uses features only."""
    participants = read_eid_table(catalog.participants_path, eid_column=catalog.eid_column)
    if catalog.eid_column not in participants:
        raise ValueError(f"participants missing participant ID column {catalog.eid_column!r}")
    ids = normalize_eids(participants[catalog.eid_column].to_numpy(), label="participants")
    if ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("participants must contain unique non-empty participant IDs")
    frames: dict[str, pd.DataFrame] = {}
    sources = {
        "catalog": file_provenance(catalog.path),
        "participants": file_provenance(catalog.participants_path),
    }
    for table in catalog.feature_tables:
        frame = read_eid_table(table.path, eid_column=table.eid_column)
        if table.eid_column not in frame:
            raise ValueError(f"feature table {table.id!r} missing participant ID column {table.eid_column!r}")
        table_ids = normalize_eids(frame[table.eid_column].to_numpy(), label=table.id)
        table_index = pd.Index(table_ids)
        if not table_index.is_unique:
            raise ValueError(f"duplicate feature participant ID: {table.id}")
        row_order = table_index.get_indexer(ids)
        missing_count = int(np.count_nonzero(row_order < 0))
        extra_count = int(table_ids.size - ids.size + missing_count)
        if missing_count or extra_count:
            raise ValueError(
                f"feature participant ID mismatch: {table.id}; missing {missing_count}, extra {extra_count}; "
                "each feature table must contain every participant exactly once"
            )
        frames[table.id] = frame if np.array_equal(ids, table_ids) else frame.iloc[row_order]
        sources[f"table:{table.id}"] = file_provenance(table.path)
    lookup = {item.name: item.feature_table for item in catalog.modules}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    names_by_pair: dict[tuple[str, str], tuple[str, ...]] = {}
    available = np.zeros((ids.size, len(catalog.modules)), dtype=bool)
    module_index = {item.name: index for index, item in enumerate(catalog.modules)}
    for pair in catalog.pairs:
        # Feature codes are column identities, including numeric text and NA-like names.
        selected = pd.read_csv(
            pair.features_path,
            converters={pair.feature_column: lambda value: value if value.strip() else None},
            skip_blank_lines=False,
        )
        if pair.feature_column not in selected:
            raise ValueError(f"feature list missing column {pair.feature_column!r}")
        raw_names = selected[pair.feature_column]
        if raw_names.isna().any():
            raise ValueError("feature list contains missing names")
        names = tuple(str(value).strip() for value in raw_names.tolist())
        _require_unique(names, "feature")
        if not names:
            raise ValueError("feature list must not be empty")
        frame = frames[lookup[pair.module]]
        missing = sorted(set(names).difference(frame.columns))
        if missing:
            raise ValueError(f"selected feature columns missing for {pair.module}/{pair.endpoint}: {missing}")
        matrix = frame.loc[:, list(names)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if np.isinf(matrix).any():
            raise ValueError(f"selected features contain infinity: {pair.module}/{pair.endpoint}")
        key = (pair.module, pair.endpoint)
        matrices[key] = matrix
        names_by_pair[key] = names
        available[:, module_index[pair.module]] |= np.isfinite(matrix).any(axis=1)
        sources[f"features:{pair.module}:{pair.endpoint}"] = file_provenance(pair.features_path)
    return ModuleFeatures(catalog, ids, matrices, names_by_pair, available, sources)
