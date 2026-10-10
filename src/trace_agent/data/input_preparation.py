"""Materialize catalog input tables from one participant feature table."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trace_agent.data.module_features import load_catalog
from trace_agent.data.split_table import normalize_eids, read_eid_table


def prepare_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("module_data", {})
    if not isinstance(settings, Mapping) or not settings.get("catalog_path"):
        raise ValueError("prepare-inputs requires module_data.catalog_path")
    if not settings.get("input_table_path"):
        raise ValueError("prepare-inputs requires --input-table")
    source = Path(settings["input_table_path"]).expanduser().resolve()
    id_column = str(settings.get("input_id_column", settings.get("input_eid_column", "participant_id")))
    frame = read_eid_table(source, eid_column=id_column)
    if id_column not in frame:
        raise ValueError(f"input table missing participant ID column {id_column!r}")
    ids = normalize_eids(frame[id_column].to_numpy(), label="input table")
    if not ids.size or np.unique(ids).size != ids.size:
        raise ValueError("input table must contain unique non-empty participant IDs")
    catalog = load_catalog(settings["catalog_path"])
    table_features: dict[str, list[str]] = {table.id: [] for table in catalog.feature_tables}
    module_tables = {module.name: module.feature_table for module in catalog.modules}
    for pair in catalog.pairs:
        selected = pd.read_csv(
            pair.features_path,
            converters={pair.feature_column: lambda value: value if value.strip() else None},
            skip_blank_lines=False,
        )
        if pair.feature_column not in selected or selected[pair.feature_column].isna().any():
            raise ValueError(f"feature list missing column or names: {pair.features_path}")
        names = [str(value).strip() for value in selected[pair.feature_column]]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"feature names must be non-empty and unique: {pair.features_path}")
        required = table_features[module_tables[pair.module]]
        required.extend(name for name in names if name not in required)

    outputs = [(catalog.participants_path, pd.DataFrame({catalog.eid_column: ids}))]
    tables: dict[str, Any] = {}
    used_columns = {id_column}
    for table in catalog.feature_tables:
        names = table_features[table.id]
        if table.eid_column in names or id_column in names:
            raise ValueError("a selected feature name conflicts with a participant ID column")
        present = [name for name in names if name in frame]
        values = frame.reindex(columns=names).apply(pd.to_numeric, errors="raise")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"selected features contain infinity: {table.id}")
        values.insert(0, table.eid_column, ids)
        outputs.append((table.path, values))
        used_columns.update(present)
        tables[table.id] = {
            "path": str(table.path),
            "provided_features": present,
            "missing_features": [name for name in names if name not in frame],
        }

    # Validate all values and destinations before replacing any existing input.
    paths = [path.resolve() for path, _ in outputs]
    protected_paths = {source, catalog.path}
    protected_paths.update(pair.features_path.resolve() for pair in catalog.pairs)
    protected_paths.update(endpoint.outcome_path.resolve() for endpoint in catalog.endpoints)
    if len(set(paths)) != len(paths) or protected_paths.intersection(paths):
        raise ValueError("source, catalog, feature-list, outcome and generated table paths must be distinct")
    if any(path.suffix.lower() not in {".csv", ".tsv", ".tab", ".parquet"} for path in paths):
        raise ValueError("generated tables require .csv, .tsv, .tab, or .parquet paths")
    for path, values in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            values.to_parquet(path, index=False)
        else:
            values.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
    return {
        "participants": int(ids.size),
        "participant_table": str(catalog.participants_path),
        "feature_tables": tables,
        "unused_columns": [str(name) for name in frame if name not in used_columns],
    }
