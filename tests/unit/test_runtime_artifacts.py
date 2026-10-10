from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trace_agent.runtime.artifacts import (
    load_absolute_risk_mapper,
    load_burden_profile,
    load_knowledge_matrix,
)


def test_artifact_loaders_align_names_and_legacy_level_prefixes(
    tmp_path: Path,
) -> None:
    burden_path = tmp_path / "burden.yaml"
    burden_path.write_text("Basic: 1\nBlood: 2\n", encoding="utf-8")
    profile = load_burden_profile(
        burden_path,
        modules=("Level_1__Basic", "Level_2__Blood"),
    )
    np.testing.assert_allclose(profile.composite, [1.0, 2.0])

    kg_path = tmp_path / "kg.csv"
    pd.DataFrame(
        {"D2": [0.2, 0.4], "D1": [0.1, 0.3]},
        index=["Blood", "Basic"],
    ).to_csv(kg_path)
    matrix = load_knowledge_matrix(
        kg_path,
        modules=("Level_1__Basic", "Level_2__Blood"),
        endpoints=("D1", "D2"),
    )
    np.testing.assert_allclose(matrix, [[0.3, 0.4], [0.1, 0.2]])


def test_absolute_risk_loader_accepts_legacy_cumulative_hazard_grid(
    tmp_path: Path,
) -> None:
    np.save(
        tmp_path / "baseline.npy",
        np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]]),
    )
    np.save(tmp_path / "times.npy", np.array([1.0, 5.0, 10.0]))
    np.save(tmp_path / "endpoints.npy", np.array(["D1", "D2"]))

    mapper = load_absolute_risk_mapper(
        {
            "baseline_cumhaz": str(tmp_path / "baseline.npy"),
            "time_grid": str(tmp_path / "times.npy"),
            "endpoints_file": str(tmp_path / "endpoints.npy"),
            "horizons": [5.0, 10.0],
        },
        endpoints=("D1", "D2"),
    )
    assert mapper is not None
    risks = mapper.transform(np.zeros((1, 2), dtype=np.float32), ("D1", "D2"))
    np.testing.assert_allclose(
        risks["5y"],
        1.0 - np.exp(-np.array([[0.2, 0.4]])),
        atol=1e-7,
    )
