from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trace_agent.consistency.longitudinal import fit_longitudinal_scale
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.data.split_table import SplitColumns, id_column_from_config, load_split_table
from trace_agent.runtime.provenance import array_sha256


def calibrate_consistency_scale(config: Mapping[str, Any]) -> Path:
    if TaskType.parse(config.get("task", "prevalent")) is not TaskType.LONGITUDINAL:
        raise ValueError("consistency scale calibration requires the longitudinal task")
    data = config["data"]
    consistency = config.get("consistency", {})
    batch = load_module_score_bundle(
        data["input_dir"],
        prefix=data["prefix"],
        task=TaskType.LONGITUDINAL,
        allow_legacy_pickle_names=bool(data.get("allow_legacy_pickle_names", False)),
    )
    split = data["split_table"]
    splits = load_split_table(
        split["path"],
        participant_ids=batch.participant_ids,
        columns=SplitColumns(
            row_index=split.get("row_index_column", "row_index"),
            eid=id_column_from_config(split),
            split=split.get("split_column", "split"),
        ),
        require_complete=False,
        required_roles=(SplitRole.VALIDATION,),
    )
    rows = splits.indices(SplitRole.VALIDATION)
    scale = fit_longitudinal_scale(
        batch.scores,
        batch.pred_mask,
        rows,
        score_kind=str(consistency.get("module_score_kind", "log_risk")),
    )
    record = {
        "schema": "trace.longitudinal_consistency_scale.v1",
        "task": "longitudinal",
        "modules": list(batch.modules),
        "endpoints": list(batch.endpoints),
        "module_score_kind": scale.score_kind,
        "log_reference": scale.log_reference,
        "source_split": "validation",
        "quantile": 0.99,
        "validation_rows": int(rows.size),
        "test_rows_used": 0,
        "split_table_canonical_sha256": splits.canonical_sha256,
        "validation_eids_sha256": array_sha256(splits.eids(SplitRole.VALIDATION)),
    }
    path = Path(
        consistency.get("longitudinal_scale_path")
        or (Path(config["output"]["directory"]) / "longitudinal_consistency_scale.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
