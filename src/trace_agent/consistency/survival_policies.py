"""Native survival policies from the original validation-only post-hoc workflow.

The numerical policy functions preserve the source algorithms without importing
its runtime, package layout, shell utilities, or cohort files. The native service
below freezes all choices on validation and applies them without target access.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from trace_agent.contracts.task import SplitRole

EPS = 1e-12


@dataclass
class Counts:
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    obs: np.ndarray
    pos: np.ndarray
    neg: np.ndarray


@dataclass
class StrategySpec:
    name: str
    family: str
    kind: str
    uses: str
    kg_cutoff: float | None = None
    med_cutoff: float | None = None
    combined_cutoff: float | None = None
    adjusted_cutoffs: np.ndarray | None = None
    enabled: np.ndarray | None = None
    transform: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    selected_on: str = "validation"
    notes: str = ""


def safe_div(num: float, den: float, default: float = float("nan")) -> float:
    return float(num / den) if den else float(default)


def f1_from_pr(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def f1_arrays(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision = np.divide(tp, tp + fp, out=np.full(tp.shape, np.nan, dtype=np.float64), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full(tp.shape, np.nan, dtype=np.float64), where=(tp + fn) > 0)
    f1 = np.full(tp.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(precision) & np.isfinite(recall) & ((precision + recall) > 0)
    f1[ok] = 2.0 * precision[ok] * recall[ok] / (precision[ok] + recall[ok])
    return f1, np.nan_to_num(f1, nan=0.0)


def make_baseline_pred(risk: np.ndarray, known_mask: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    risk_arr = np.asarray(risk, dtype=np.float32)
    known = np.asarray(known_mask, dtype=bool)
    cutoff_arr = np.asarray(cutoffs, dtype=np.float32).reshape(1, -1)
    return (risk_arr >= cutoff_arr) & known & np.isfinite(risk_arr)


def counts_from_pred(pred: np.ndarray, y: np.ndarray, known_mask: np.ndarray) -> Counts:
    known = np.asarray(known_mask, dtype=bool)
    pred_bool = np.asarray(pred, dtype=bool) & known
    y_bool = (np.asarray(y, dtype=np.uint8) > 0) & known
    tp = np.sum(pred_bool & y_bool, axis=0).astype(np.int64)
    fp = np.sum(pred_bool & ~y_bool & known, axis=0).astype(np.int64)
    fn = np.sum(~pred_bool & y_bool, axis=0).astype(np.int64)
    tn = np.sum(~pred_bool & ~y_bool & known, axis=0).astype(np.int64)
    obs = np.sum(known, axis=0).astype(np.int64)
    pos = np.sum(y_bool, axis=0).astype(np.int64)
    return Counts(tp=tp, fp=fp, fn=fn, tn=tn, obs=obs, pos=pos, neg=obs - pos)


def normalize_kg(kg_score: np.ndarray, kg_q99: float) -> np.ndarray:
    return np.clip(np.asarray(kg_score, dtype=np.float32) / max(float(kg_q99), EPS), 0.0, 1.0)


def normalize_med(med_score: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(med_score, dtype=np.float32), 0.0, 1.0)


def combined_score(kg_score: np.ndarray, med_score: np.ndarray, spec: StrategySpec) -> np.ndarray:
    kg_norm = normalize_kg(kg_score, float(spec.params["kg_q99"]))
    med_norm = normalize_med(med_score)
    mode = str(spec.params["combined_mode"])
    if mode == "product":
        return kg_norm * med_norm
    if mode == "weighted_sum":
        w = float(spec.params["w"])
        # Pure KG/medical policies must not depend on the unused input's mask.
        if w == 1.0:
            return kg_norm
        if w == 0.0:
            return med_norm
        return w * kg_norm + (1.0 - w) * med_norm
    if mode == "min_gate":
        return np.minimum(kg_norm, med_norm)
    raise ValueError(f"unknown combined mode: {mode}")


def adjusted_score(
    risk: np.ndarray, kg_score: np.ndarray, med_score: np.ndarray, spec: StrategySpec
) -> np.ndarray:
    # Use float64 clipping here: float32 cannot represent 1 - 1e-8 distinctly
    # from 1.0, which would create infinite logits for saturated risks.
    eps = 1e-6
    p_clip = np.clip(np.asarray(risk, dtype=np.float64), eps, 1.0 - eps)
    if spec.transform == "logit_soft_reweight":
        out: np.ndarray = np.log(p_clip / (1.0 - p_clip)).astype(np.float32, copy=False)
        a = float(spec.params.get("a", 0.0))
        b = float(spec.params.get("b", 0.0))
        if a != 0.0:
            out = out + a * (normalize_kg(kg_score, float(spec.params["kg_q99"])) - 0.5)
        if b != 0.0:
            out = out + b * (normalize_med(med_score) - 0.5)
        return out
    if spec.transform == "kg_p_times_power":
        gamma = float(spec.params["gamma"])
        return p_clip * np.power(
            np.clip(normalize_kg(kg_score, float(spec.params["kg_q99"])), eps, None), gamma
        )
    raise ValueError(f"unknown transform: {spec.transform}")


def build_pred(
    *,
    kind: str,
    risk: np.ndarray,
    known_mask: np.ndarray,
    base_cutoffs: np.ndarray,
    kg_score: np.ndarray,
    med_score: np.ndarray,
    kg_cutoff: float | None = None,
    med_cutoff: float | None = None,
    combined_cutoff: float | None = None,
    combined_mode: str = "product",
    kg_q99: float = 1.0,
    adjusted_cutoffs: np.ndarray | None = None,
    transform: str | None = None,
    a: float = 0.0,
    b: float = 0.0,
    gamma: float = 1.0,
    enabled: np.ndarray | None = None,
) -> np.ndarray:
    base = make_baseline_pred(risk, known_mask, base_cutoffs)
    kg = np.asarray(kg_score, dtype=np.float32)
    med = np.asarray(med_score, dtype=np.float32)
    if kind == "baseline":
        pred = base
    elif kind == "hard_kg":
        pred = base & np.isfinite(kg) & (kg >= float(cast(float, kg_cutoff)))
    elif kind == "hard_med":
        pred = base & np.isfinite(med) & (med >= float(cast(float, med_cutoff)))
    elif kind == "hard_kg_med_and":
        pred = (
            base
            & np.isfinite(kg)
            & np.isfinite(med)
            & (kg >= float(cast(float, kg_cutoff)))
            & (med >= float(cast(float, med_cutoff)))
        )
    elif kind == "hard_combined":
        tmp_spec = StrategySpec(
            name="tmp",
            family="tmp",
            kind="hard_combined",
            uses="kg_med",
            params={"combined_mode": combined_mode, "kg_q99": kg_q99, "w": 0.5},
        )
        if combined_mode == "weighted_sum":
            tmp_spec.params["w"] = float(a)
        score = combined_score(kg, med, tmp_spec)
        pred = base & np.isfinite(score) & (score >= float(cast(float, combined_cutoff)))
    elif kind == "adjusted":
        if adjusted_cutoffs is None:
            raise ValueError("adjusted_cutoffs is required for adjusted predictions")
        tmp_spec = StrategySpec(
            name="tmp",
            family="tmp",
            kind="adjusted",
            uses="kg_med",
            transform=transform,
            params={"kg_q99": kg_q99, "a": a, "b": b, "gamma": gamma},
        )
        score = adjusted_score(risk, kg, med, tmp_spec)
        pred = np.isfinite(score) & (score >= np.asarray(adjusted_cutoffs, dtype=np.float32)[None, :])
    else:
        raise ValueError(f"unknown strategy kind: {kind}")
    pred = pred & np.asarray(known_mask, dtype=bool)
    if enabled is not None:
        enabled_bool = np.asarray(enabled, dtype=bool)
        pred = np.where(enabled_bool[None, :], pred, base)
    return pred


def pred_for_spec(
    spec: StrategySpec,
    risk: np.ndarray,
    known_mask: np.ndarray,
    base_cutoffs: np.ndarray,
    kg_score: np.ndarray,
    med_score: np.ndarray,
) -> np.ndarray:
    if spec.kind == "baseline":
        return build_pred(
            kind="baseline",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
        )
    if spec.kind == "hard_kg":
        return build_pred(
            kind="hard_kg",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
            kg_cutoff=spec.kg_cutoff,
            enabled=spec.enabled,
        )
    if spec.kind == "hard_med":
        return build_pred(
            kind="hard_med",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
            med_cutoff=spec.med_cutoff,
            enabled=spec.enabled,
        )
    if spec.kind == "hard_kg_med_and":
        return build_pred(
            kind="hard_kg_med_and",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
            kg_cutoff=spec.kg_cutoff,
            med_cutoff=spec.med_cutoff,
            enabled=spec.enabled,
        )
    if spec.kind == "hard_combined":
        return build_pred(
            kind="hard_combined",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
            combined_cutoff=spec.combined_cutoff,
            combined_mode=str(spec.params["combined_mode"]),
            kg_q99=float(spec.params["kg_q99"]),
            a=float(spec.params.get("w", 0.5)),
            enabled=spec.enabled,
        )
    if spec.kind == "adjusted":
        return build_pred(
            kind="adjusted",
            risk=risk,
            known_mask=known_mask,
            base_cutoffs=base_cutoffs,
            kg_score=kg_score,
            med_score=med_score,
            adjusted_cutoffs=spec.adjusted_cutoffs,
            transform=spec.transform,
            kg_q99=float(spec.params["kg_q99"]),
            a=float(spec.params.get("a", 0.0)),
            b=float(spec.params.get("b", 0.0)),
            gamma=float(spec.params.get("gamma", 1.0)),
            enabled=spec.enabled,
        )
    raise ValueError(f"unknown strategy kind: {spec.kind}")


def select_f1_cutoffs(score: np.ndarray, y: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    score_arr = np.asarray(score, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.uint8)
    known = np.asarray(known_mask, dtype=bool)
    d_count = score_arr.shape[1]
    cutoffs = np.full(d_count, np.inf, dtype=np.float32)
    for d in range(d_count):
        obs = known[:, d] & np.isfinite(score_arr[:, d])
        if not np.any(obs):
            continue
        s = score_arr[obs, d].astype(np.float64, copy=False)
        yy = (y_arr[obs, d] > 0).astype(np.int8, copy=False)
        pos = int(np.sum(yy))
        if pos <= 0:
            continue
        order = np.argsort(-s, kind="mergesort")
        s_sorted = s[order]
        y_sorted = yy[order]
        tp_cum = np.cumsum(y_sorted)
        pred_count = np.arange(1, y_sorted.size + 1, dtype=np.float64)
        f1 = 2.0 * tp_cum / (pred_count + float(pos))
        last = np.r_[np.flatnonzero(s_sorted[:-1] != s_sorted[1:]), y_sorted.size - 1]
        best_pos = int(last[int(np.argmax(f1[last]))])
        cutoffs[d] = float(s_sorted[best_pos])
    return cutoffs


def make_threshold_grid(values: np.ndarray, max_points: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float32)
    quantiles = np.unique(
        np.r_[
            np.linspace(0.0, 0.90, 41),
            np.linspace(0.91, 0.99, 17),
            [0.995, 0.999, 1.0],
        ]
    )
    if quantiles.size > int(max_points):
        keep = np.unique(np.linspace(0, quantiles.size - 1, int(max_points)).round().astype(int))
        quantiles = quantiles[keep]
    grid = np.quantile(finite.astype(np.float64), quantiles).astype(np.float32)
    grid = np.unique(np.r_[finite.min(), grid, finite.max()])
    return np.sort(grid.astype(np.float32))


def summary_row(
    spec: StrategySpec, counts: Counts, baseline_counts: Counts | None, split: str
) -> dict[str, Any]:
    tp_sum = int(counts.tp.sum())
    fp_sum = int(counts.fp.sum())
    fn_sum = int(counts.fn.sum())
    tn_sum = int(counts.tn.sum())
    precision = safe_div(tp_sum, tp_sum + fp_sum)
    recall = safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = f1_from_pr(precision, recall)
    f1_nan, f1_zero = f1_arrays(counts.tp, counts.fp, counts.fn)
    row: dict[str, Any] = {
        "split": split,
        "strategy": spec.name,
        "family": spec.family,
        "kind": spec.kind,
        "uses": spec.uses,
        "micro_f1": micro_f1,
        "macro_f1_zero_filled": float(np.mean(f1_zero)),
        "macro_f1_nanmean": float(np.nanmean(f1_nan)) if np.any(np.isfinite(f1_nan)) else float("nan"),
        "precision": precision,
        "recall": recall,
        "tp": tp_sum,
        "fp": fp_sum,
        "fn": fn_sum,
        "tn": tn_sum,
        "pred_positive": tp_sum + fp_sum,
        "fp_reduction_rate": 0.0,
        "tp_retention_rate": 1.0,
        "improved_disease_count": 0,
        "worsened_disease_count": 0,
        "unchanged_disease_count": int(counts.tp.size),
        "enabled_disease_count": (
            int(np.sum(spec.enabled)) if spec.enabled is not None else int(counts.tp.size)
        ),
        "kg_cutoff": spec.kg_cutoff,
        "med_cutoff": spec.med_cutoff,
        "combined_cutoff": spec.combined_cutoff,
        "transform": spec.transform,
        "selected_on": spec.selected_on,
        "notes": spec.notes,
    }
    for key, value in spec.params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"param_{key}"] = value
    if baseline_counts is not None:
        base_precision = safe_div(
            int(baseline_counts.tp.sum()), int((baseline_counts.tp + baseline_counts.fp).sum())
        )
        base_recall = safe_div(
            int(baseline_counts.tp.sum()), int((baseline_counts.tp + baseline_counts.fn).sum())
        )
        base_micro = f1_from_pr(base_precision, base_recall)
        _, base_f1_zero = f1_arrays(baseline_counts.tp, baseline_counts.fp, baseline_counts.fn)
        row.update(
            {
                "micro_f1_delta": micro_f1 - base_micro,
                "macro_f1_zero_filled_delta": row["macro_f1_zero_filled"] - float(np.mean(base_f1_zero)),
                "precision_delta": precision - base_precision,
                "recall_delta": recall - base_recall,
                "fp_reduction_rate": safe_div(
                    int(baseline_counts.fp.sum()) - fp_sum, int(baseline_counts.fp.sum()), 0.0
                ),
                "tp_retention_rate": safe_div(tp_sum, int(baseline_counts.tp.sum()), 1.0),
                "improved_disease_count": int(np.sum(f1_zero > base_f1_zero + 1e-12)),
                "worsened_disease_count": int(np.sum(f1_zero < base_f1_zero - 1e-12)),
                "unchanged_disease_count": int(np.sum(np.abs(f1_zero - base_f1_zero) <= 1e-12)),
            }
        )
    else:
        row.update(
            {
                "micro_f1_delta": 0.0,
                "macro_f1_zero_filled_delta": 0.0,
                "precision_delta": 0.0,
                "recall_delta": 0.0,
            }
        )
    return row


def per_disease_frame(
    spec: StrategySpec,
    disease_names: list[str],
    counts: Counts,
    baseline_counts: Counts,
    split: str,
) -> pd.DataFrame:
    strategy_precision = np.divide(
        counts.tp,
        counts.tp + counts.fp,
        out=np.full(counts.tp.shape, np.nan, dtype=np.float64),
        where=(counts.tp + counts.fp) > 0,
    )
    strategy_recall = np.divide(
        counts.tp,
        counts.tp + counts.fn,
        out=np.full(counts.tp.shape, np.nan, dtype=np.float64),
        where=(counts.tp + counts.fn) > 0,
    )
    baseline_precision = np.divide(
        baseline_counts.tp,
        baseline_counts.tp + baseline_counts.fp,
        out=np.full(counts.tp.shape, np.nan, dtype=np.float64),
        where=(baseline_counts.tp + baseline_counts.fp) > 0,
    )
    baseline_recall = np.divide(
        baseline_counts.tp,
        baseline_counts.tp + baseline_counts.fn,
        out=np.full(counts.tp.shape, np.nan, dtype=np.float64),
        where=(baseline_counts.tp + baseline_counts.fn) > 0,
    )
    _, strategy_f1 = f1_arrays(counts.tp, counts.fp, counts.fn)
    _, baseline_f1 = f1_arrays(baseline_counts.tp, baseline_counts.fp, baseline_counts.fn)
    enabled = (
        spec.enabled.astype(np.int8)
        if spec.enabled is not None
        else np.ones(len(disease_names), dtype=np.int8)
    )
    return pd.DataFrame(
        {
            "split": split,
            "strategy": spec.name,
            "family": spec.family,
            "disease": disease_names,
            "disease_index": np.arange(len(disease_names), dtype=np.int64),
            "baseline_f1": baseline_f1,
            "strategy_f1": strategy_f1,
            "f1_delta": strategy_f1 - baseline_f1,
            "baseline_tp": baseline_counts.tp.astype(np.int64),
            "strategy_tp": counts.tp.astype(np.int64),
            "baseline_fp": baseline_counts.fp.astype(np.int64),
            "strategy_fp": counts.fp.astype(np.int64),
            "baseline_fn": baseline_counts.fn.astype(np.int64),
            "strategy_fn": counts.fn.astype(np.int64),
            "baseline_precision": baseline_precision,
            "strategy_precision": strategy_precision,
            "baseline_recall": baseline_recall,
            "strategy_recall": strategy_recall,
            "fp_reduction_rate": np.divide(
                baseline_counts.fp - counts.fp,
                baseline_counts.fp,
                out=np.zeros(counts.tp.shape, dtype=np.float64),
                where=baseline_counts.fp > 0,
            ),
            "tp_retention_rate": np.divide(
                counts.tp,
                baseline_counts.tp,
                out=np.ones(counts.tp.shape, dtype=np.float64),
                where=baseline_counts.tp > 0,
            ),
            "enabled": enabled,
        }
    )


def build_initial_specs(
    arrays: dict[str, np.ndarray],
    base_pred: np.ndarray,
    kg_q99: float,
    threshold_points: int,
    joint_grid_points: int,
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    kg_values = arrays["kg"][base_pred & np.isfinite(arrays["kg"])]
    med_values = arrays["med"][base_pred & np.isfinite(arrays["med"])]
    for t in make_threshold_grid(kg_values, threshold_points):
        specs.append(
            StrategySpec(
                name=f"kg_only_hard_filter_kg_ge_{float(t):.8g}",
                family="kg_only_hard_filter",
                kind="hard_kg",
                uses="kg_only",
                kg_cutoff=float(t),
            )
        )
    for t in make_threshold_grid(med_values, threshold_points):
        specs.append(
            StrategySpec(
                name=f"med_only_hard_filter_med_ge_{float(t):.8g}",
                family="med_only_hard_filter",
                kind="hard_med",
                uses="med_only",
                med_cutoff=float(t),
            )
        )
    kg_joint_grid = make_threshold_grid(kg_values, joint_grid_points)
    med_joint_grid = make_threshold_grid(med_values, joint_grid_points)
    for kg_t in kg_joint_grid:
        for med_t in med_joint_grid:
            specs.append(
                StrategySpec(
                    name=f"joint_hard_filter_kg_ge_{float(kg_t):.8g}_med_ge_{float(med_t):.8g}",
                    family="joint_hard_filter",
                    kind="hard_kg_med_and",
                    uses="kg_med",
                    kg_cutoff=float(kg_t),
                    med_cutoff=float(med_t),
                )
            )
    combined_templates: list[tuple[str, dict[str, Any]]] = [
        ("product", {"combined_mode": "product", "kg_q99": kg_q99}),
        ("min_gate", {"combined_mode": "min_gate", "kg_q99": kg_q99}),
    ]
    for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        combined_templates.append(
            ("weighted_sum", {"combined_mode": "weighted_sum", "kg_q99": kg_q99, "w": w})
        )
    for label, params in combined_templates:
        tmp = StrategySpec(name="tmp", family="tmp", kind="hard_combined", uses="kg_med", params=params)
        score = combined_score(arrays["kg"], arrays["med"], tmp)
        use_label = (
            "kg_only"
            if label == "weighted_sum" and params.get("w") == 1.0
            else "med_only" if label == "weighted_sum" and params.get("w") == 0.0 else "kg_med"
        )
        for t in make_threshold_grid(score[base_pred & np.isfinite(score)], threshold_points):
            name = f"{use_label}_{label}_hard_combined_ge_{float(t):.8g}"
            if label == "weighted_sum":
                name = f"{use_label}_weighted_sum_w{params['w']:g}_hard_combined_ge_{float(t):.8g}"
            specs.append(
                StrategySpec(
                    name=name,
                    family=f"{use_label}_{label}_hard_combined",
                    kind="hard_combined",
                    uses=use_label,
                    combined_cutoff=float(t),
                    params=params.copy(),
                )
            )
    soft_values = [1.0, 2.0, 4.0, 8.0]
    soft_pairs = (
        [(a, 0.0) for a in soft_values]
        + [(0.0, b) for b in soft_values]
        + [(a, b) for a in soft_values for b in soft_values]
    )
    for a, b in soft_pairs:
        use_label = "kg_med" if a > 0 and b > 0 else "kg_only" if a > 0 else "med_only"
        specs.append(
            StrategySpec(
                name=f"{use_label}_logit_soft_reweight_a{a:g}_b{b:g}",
                family=f"{use_label}_logit_soft_reweight",
                kind="adjusted",
                uses=use_label,
                transform="logit_soft_reweight",
                params={"kg_q99": kg_q99, "a": a, "b": b},
            )
        )
    return specs


def populate_adjusted_cutoffs(spec: StrategySpec, arrays: dict[str, np.ndarray]) -> None:
    if spec.kind != "adjusted" or spec.adjusted_cutoffs is not None:
        return
    score = adjusted_score(arrays["risk"], arrays["kg"], arrays["med"], spec)
    spec.adjusted_cutoffs = select_f1_cutoffs(score, arrays["y"], arrays["known"])


def clone_with_gate(spec: StrategySpec, gate_name: str, enabled: np.ndarray) -> StrategySpec:
    return StrategySpec(
        name=f"{spec.name}_{gate_name}",
        family=f"{spec.family}_disease_gate",
        kind=spec.kind,
        uses=spec.uses,
        kg_cutoff=spec.kg_cutoff,
        med_cutoff=spec.med_cutoff,
        combined_cutoff=spec.combined_cutoff,
        adjusted_cutoffs=None if spec.adjusted_cutoffs is None else spec.adjusted_cutoffs.copy(),
        enabled=enabled.astype(bool, copy=True),
        transform=spec.transform,
        params={**spec.params, "gate": gate_name},
        selected_on=spec.selected_on,
        notes=spec.notes,
    )


def gate_specs(spec: StrategySpec, counts: Counts, baseline_counts: Counts) -> list[StrategySpec]:
    _, base_f1 = f1_arrays(baseline_counts.tp, baseline_counts.fp, baseline_counts.fn)
    _, strategy_f1 = f1_arrays(counts.tp, counts.fp, counts.fn)
    fp_red = np.divide(
        baseline_counts.fp - counts.fp,
        baseline_counts.fp,
        out=np.zeros(counts.tp.shape, dtype=np.float64),
        where=baseline_counts.fp > 0,
    )
    tp_ret = np.divide(
        counts.tp,
        baseline_counts.tp,
        out=np.ones(counts.tp.shape, dtype=np.float64),
        where=baseline_counts.tp > 0,
    )
    low_tp = baseline_counts.tp < 3
    gates = {
        "gate_val_f1_positive": strategy_f1 > base_f1 + 1e-12,
        "gate_val_f1_noninferior_fp_reduced": (strategy_f1 + 1e-12 >= base_f1) & (fp_red > 0),
        "gate_val_margin001_tp90_fp_reduced": (strategy_f1 + 0.001 >= base_f1)
        & (fp_red > 0)
        & ((tp_ret >= 0.90) | low_tp),
        "gate_val_margin002_tp85_fp_reduced": (strategy_f1 + 0.002 >= base_f1)
        & (fp_red > 0)
        & ((tp_ret >= 0.85) | low_tp),
    }
    return [
        clone_with_gate(spec, gate_name, enabled) for gate_name, enabled in gates.items() if np.any(enabled)
    ]


def validation_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    cand = summary[summary["strategy"] != "baseline_risk_only"].copy()
    if cand.empty:
        return cand
    cols = ["micro_f1", "macro_f1_zero_filled", "fp_reduction_rate", "tp_retention_rate"]
    arr = cand[cols].to_numpy(dtype=np.float64)
    worsened = cand["worsened_disease_count"].to_numpy(dtype=np.float64)
    dominated = np.zeros(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            if i == j:
                continue
            better_equal = np.all(arr[j] >= arr[i] - 1e-12) and worsened[j] <= worsened[i] + 1e-12
            strictly = np.any(arr[j] > arr[i] + 1e-12) or worsened[j] < worsened[i] - 1e-12
            if better_equal and strictly:
                dominated[i] = True
                break
    return cand.loc[~dominated].sort_values(
        ["micro_f1", "macro_f1_zero_filled", "fp_reduction_rate"], ascending=False
    )


def choose_best_by_family(summary: pd.DataFrame, family_mask: pd.Series, baseline_macro: float) -> str | None:
    cand = summary[family_mask & (summary["strategy"] != "baseline_risk_only")].copy()
    if cand.empty:
        return None
    eligible = cand[cand["macro_f1_zero_filled"] >= baseline_macro - 0.001].copy()
    if eligible.empty:
        eligible = cand
    row = eligible.sort_values(
        ["micro_f1", "macro_f1_zero_filled", "fp_reduction_rate", "tp_retention_rate"],
        ascending=False,
    ).iloc[0]
    return str(row["strategy"])


def choose_finalists(
    summary: pd.DataFrame, specs: dict[str, StrategySpec], max_finalists: int
) -> list[StrategySpec]:
    baseline_macro = float(
        summary.loc[summary["strategy"] == "baseline_risk_only", "macro_f1_zero_filled"].iloc[0]
    )
    names = ["baseline_risk_only"]
    family_choices = [
        choose_best_by_family(summary, summary["uses"] == "kg_only", baseline_macro),
        choose_best_by_family(summary, summary["uses"] == "med_only", baseline_macro),
        choose_best_by_family(summary, summary["uses"] == "kg_med", baseline_macro),
    ]
    for name in family_choices:
        if name and name not in names:
            names.append(name)
    pareto = validation_pareto(summary).head(max_finalists)
    for pareto_name in pareto["strategy"].astype(str):
        if pareto_name not in names:
            names.append(str(pareto_name))
    return [specs[name] for name in names if name in specs][:max_finalists]


def spec_to_json(spec: StrategySpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "kind": spec.kind,
        "uses": spec.uses,
        "kg_cutoff": spec.kg_cutoff,
        "med_cutoff": spec.med_cutoff,
        "combined_cutoff": spec.combined_cutoff,
        "transform": spec.transform,
        "params": spec.params,
        "selected_on": spec.selected_on,
        "notes": spec.notes,
    }
    if spec.adjusted_cutoffs is not None:
        out["adjusted_cutoffs"] = np.asarray(spec.adjusted_cutoffs, dtype=float).tolist()
    if spec.enabled is not None:
        out["enabled"] = np.asarray(spec.enabled, dtype=bool).astype(int).tolist()
    return out


def _json_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _arrays(
    predictions: np.ndarray, kg_scores: np.ndarray, evidence_scores: np.ndarray
) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (predictions, kg_scores, evidence_scores))
    if arrays[0].ndim != 2 or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("survival policy arrays must share participant-by-endpoint shape")
    return arrays


def _decode_spec(record: Mapping[str, Any]) -> StrategySpec:
    values = dict(record)
    if "adjusted_cutoffs" in values:
        values["adjusted_cutoffs"] = np.asarray(
            [np.inf if value is None else value for value in values["adjusted_cutoffs"]], dtype=np.float32
        )
    if "enabled" in values:
        values["enabled"] = np.asarray(values["enabled"], dtype=bool)
    return StrategySpec(**values)


def select_survival_policies(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    kg_scores: np.ndarray,
    evidence_scores: np.ndarray,
    endpoints: tuple[str, ...],
    split_role: SplitRole | str,
    horizon: str,
    threshold_points: int = 41,
    joint_grid_points: int = 9,
    max_finalists: int = 120,
) -> dict[str, Any]:
    """Preserve the complete source candidate search, gates and 4-way selector.

    The medical input is source ``med_score`` (disease evidence), independently
    of the separate signed MCS feature diagnostic. Test inputs are never accepted.
    """
    if SplitRole.parse(split_role) is not SplitRole.VALIDATION:
        raise ValueError("survival policies may only be selected on validation")
    p, kg, med = _arrays(predictions, kg_scores, evidence_scores)
    truth = np.asarray(labels)
    mask = np.asarray(label_mask, dtype=bool)
    if truth.shape != p.shape or mask.shape != p.shape or len(endpoints) != p.shape[1]:
        raise ValueError("survival labels, mask and endpoints must align with predictions")
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("survival policies require unique endpoints")
    if min(threshold_points, joint_grid_points, max_finalists) < 1:
        raise ValueError("survival search sizes must be positive")
    try:
        horizon_value = (
            float(horizon[:-1]) if isinstance(horizon, str) and horizon.endswith("y") else float("nan")
        )
    except ValueError:
        horizon_value = float("nan")
    if not np.isfinite(horizon_value) or horizon_value <= 0:
        raise ValueError("survival policies require a positive year-based horizon")
    known = mask & np.isfinite(truth) & np.isin(truth, (0, 1)) & np.isfinite(p)
    if np.any(known & ((p < 0) | (p > 1))):
        raise ValueError("survival policies require probabilities in [0, 1]")
    arrays = {
        "risk": p,
        "kg": kg,
        "med": med,
        "y": np.where(known, truth, 0).astype(np.uint8),
        "known": known,
    }
    base_cutoffs = select_f1_cutoffs(p, arrays["y"], known)
    base_pred = make_baseline_pred(p, known, base_cutoffs)
    base_counts = counts_from_pred(base_pred, arrays["y"], known)
    baseline = StrategySpec(
        "baseline_risk_only",
        "baseline",
        "baseline",
        "p_base_only",
        notes=f"Validation-selected {horizon_value:g}-year baseline risk cutoffs.",
    )
    baseline_row = summary_row(baseline, base_counts, None, "validation")
    kg_values = kg[base_pred & np.isfinite(kg)]
    kg_q99 = float(np.quantile(kg_values.astype(np.float64), 0.99)) if kg_values.size else 1.0
    initial = build_initial_specs(arrays, base_pred, kg_q99, threshold_points, joint_grid_points)
    specs = {baseline.name: baseline}
    counts = {baseline.name: base_counts}
    rows = [baseline_row]

    def evaluate(spec: StrategySpec) -> None:
        populate_adjusted_cutoffs(spec, arrays)
        predicted = pred_for_spec(spec, p, known, base_cutoffs, kg, med)
        measured = counts_from_pred(predicted, arrays["y"], known)
        specs[spec.name] = spec
        counts[spec.name] = measured
        rows.append(summary_row(spec, measured, base_counts, "validation"))

    for spec in initial:
        evaluate(spec)
    for spec in initial:
        if spec.uses in {"kg_only", "med_only", "kg_med"}:
            for gated in gate_specs(spec, counts[spec.name], base_counts):
                evaluate(gated)
    summary = pd.DataFrame(rows).sort_values(
        ["micro_f1", "macro_f1_zero_filled", "fp_reduction_rate"], ascending=False
    )
    finalists = choose_finalists(summary, specs, max_finalists)
    candidates = finalists[:4]
    have_uses = {spec.uses for spec in candidates}
    for desired in ("kg_only", "med_only", "kg_med"):
        if desired not in have_uses:
            name = choose_best_by_family(
                summary, summary["uses"] == desired, float(baseline_row["macro_f1_zero_filled"])
            )
            if name and name in specs and name not in {spec.name for spec in candidates}:
                candidates.append(specs[name])
    # Source priority is indexed by uses; its p_base_only baseline ranks 99.
    # Keep that exact tie behavior rather than silently changing the selector.
    priority = {"baseline_risk_only": 0, "kg_only": 1, "kg_med": 2, "med_only": 3}
    scores = {
        spec.name: f1_arrays(counts[spec.name].tp, counts[spec.name].fp, counts[spec.name].fn)[1]
        for spec in candidates
    }
    baseline_f1 = f1_arrays(base_counts.tp, base_counts.fp, base_counts.fn)[1]
    rules = []
    for index, endpoint in enumerate(endpoints):
        selected = candidates[0]
        selected_f1 = scores[selected.name][index]
        selected_priority = priority.get(selected.uses, 99)
        for spec in candidates[1:]:
            score, rank = scores[spec.name][index], priority.get(spec.uses, 99)
            if score > selected_f1 + 1e-12 or (
                abs(score - selected_f1) <= 1e-12 and rank < selected_priority
            ):
                selected, selected_f1, selected_priority = spec, score, rank
        rules.append(
            {
                "endpoint": endpoint,
                "endpoint_index": index,
                "cutoff": float(base_cutoffs[index]) if np.isfinite(base_cutoffs[index]) else None,
                "cutoff_valid": bool(np.isfinite(base_cutoffs[index])),
                "selected_strategy": selected.name,
                "selected_score_family": selected.uses,
                "use_consistency_cutoff": selected.name != baseline.name,
                "validation_selected_f1": float(selected_f1),
                "baseline_validation_f1": float(baseline_f1[index]),
            }
        )
    artifact = _json_values(
        {
            "schema": "trace.survival_consistency_policies.v1",
            "schema_version": 1,
            "task": "longitudinal",
            "horizon": horizon,
            "selection_source": "validation_only",
            "endpoints": list(endpoints),
            "settings": {
                "threshold_points": int(threshold_points),
                "joint_grid_points": int(joint_grid_points),
                "max_finalists": int(max_finalists),
            },
            "kg_q99": kg_q99,
            "base_cutoffs": base_cutoffs,
            "rules": rules,
            "all_strategy_specs": {name: spec_to_json(spec) for name, spec in specs.items()},
            "finalists": [spec.name for spec in finalists],
            "selector_candidates": [spec.name for spec in candidates],
            "validation_summary": summary.to_dict(orient="records"),
            "validation_pareto": validation_pareto(summary).to_dict(orient="records"),
            "validation_per_disease": pd.concat(
                [
                    per_disease_frame(spec, list(endpoints), counts[spec.name], base_counts, "validation")
                    for spec in finalists
                ],
                ignore_index=True,
            ).to_dict(orient="records"),
        }
    )
    validate_survival_policies(artifact)
    return artifact


def validate_survival_policies(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema") != "trace.survival_consistency_policies.v1"
        or artifact.get("task") != "longitudinal"
        or artifact.get("selection_source") != "validation_only"
    ):
        raise ValueError("survival policies require the longitudinal validation-only schema")
    endpoints, rows = artifact.get("endpoints"), artifact.get("rules")
    if not isinstance(endpoints, list) or not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("survival policies require unique endpoints")
    if (
        not isinstance(rows, list)
        or len(rows) != len(endpoints)
        or len(artifact.get("base_cutoffs", [])) != len(endpoints)
    ):
        raise ValueError("survival policy endpoint count mismatch")
    specs = artifact.get("all_strategy_specs")
    if not isinstance(specs, Mapping) or "baseline_risk_only" not in specs:
        raise ValueError("survival policies require frozen strategy definitions")
    for index, (endpoint, row) in enumerate(zip(endpoints, rows)):
        if (
            row.get("endpoint") != endpoint
            or row.get("endpoint_index") != index
            or row.get("selected_strategy") not in specs
        ):
            raise ValueError("survival selector endpoint/strategy identity mismatch")
        cutoff = artifact["base_cutoffs"][index]
        if cutoff is not None and (
            not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1
        ):
            raise ValueError("survival baseline cutoffs must be probabilities or null")
    for name, record in specs.items():
        spec = _decode_spec(record)
        if (
            spec.name != name
            or spec.kind
            not in {"baseline", "hard_kg", "hard_med", "hard_kg_med_and", "hard_combined", "adjusted"}
            or spec.selected_on != "validation"
        ):
            raise ValueError("invalid frozen survival strategy")
        if spec.enabled is not None and spec.enabled.shape != (len(endpoints),):
            raise ValueError("survival disease gate must align with endpoints")
        if spec.kind == "adjusted" and (
            spec.adjusted_cutoffs is None or spec.adjusted_cutoffs.shape != (len(endpoints),)
        ):
            raise ValueError("survival adjusted cutoff must align with endpoints")
    for name in artifact.get("finalists", []) + artifact.get("selector_candidates", []):
        if name not in specs:
            raise ValueError("survival finalist/candidate is missing a frozen strategy")


def apply_survival_policies(
    artifact: Mapping[str, Any],
    *,
    predictions: np.ndarray,
    valid_mask: np.ndarray,
    kg_scores: np.ndarray,
    evidence_scores: np.ndarray,
    include_strategies: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply every finalist and the frozen selector without reading labels."""
    validate_survival_policies(artifact)
    p, kg, med = _arrays(predictions, kg_scores, evidence_scores)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != p.shape or p.shape[1] != len(artifact["endpoints"]):
        raise ValueError("survival policy prediction mask/endpoint mismatch")
    if np.any(valid & (~np.isfinite(p) | (p < 0) | (p > 1))):
        raise ValueError("survival policy predictions must be finite probabilities")
    cutoffs = np.asarray(
        [np.inf if value is None else value for value in artifact["base_cutoffs"]], dtype=np.float32
    )
    names = ["baseline_risk_only"] + [row["selected_strategy"] for row in artifact["rules"]]
    if include_strategies:
        names += artifact["finalists"] + artifact["selector_candidates"]
    names = list(dict.fromkeys(names))
    strategies = {
        name: pred_for_spec(_decode_spec(artifact["all_strategy_specs"][name]), p, valid, cutoffs, kg, med)
        for name in names
    }
    selected = np.stack(
        [strategies[row["selected_strategy"]][:, index] for index, row in enumerate(artifact["rules"])],
        axis=1,
    )
    return strategies["baseline_risk_only"], selected, strategies


def evaluate_survival_policies(
    artifact: Mapping[str, Any],
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    kg_scores: np.ndarray,
    evidence_scores: np.ndarray,
    split_role: SplitRole | str = SplitRole.TEST,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Report the source finalist and selector metrics on a named evaluation split."""
    role = SplitRole.parse(split_role)
    if role is not SplitRole.TEST:
        raise ValueError("frozen survival policy reporting requires the test split")
    p, kg, med = _arrays(predictions, kg_scores, evidence_scores)
    truth, known = np.asarray(labels), np.asarray(label_mask, dtype=bool)
    if truth.shape != known.shape or truth.shape != p.shape:
        raise ValueError("survival policy evaluation labels and mask must align")
    known = known & np.isfinite(truth) & np.isin(truth, (0, 1))
    validate_survival_policies(artifact)
    if chunk_size < 1:
        raise ValueError("survival evaluation chunk_size must be positive")
    if np.any(known & (~np.isfinite(p) | (p < 0) | (p > 1))):
        raise ValueError("survival policy predictions must be finite probabilities")
    cutoffs = np.asarray(
        [np.inf if value is None else value for value in artifact["base_cutoffs"]], dtype=np.float32
    )
    endpoints = artifact["endpoints"]
    names = list(
        dict.fromkeys(
            ["baseline_risk_only"]
            + artifact["finalists"]
            + [row["selected_strategy"] for row in artifact["rules"]]
        )
    )
    measured = {}
    for name in names:
        spec = _decode_spec(artifact["all_strategy_specs"][name])
        counts = Counts(*(np.zeros(len(endpoints), dtype=np.int64) for _ in range(7)))
        for start in range(0, p.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            predicted = pred_for_spec(spec, p[rows], known[rows], cutoffs, kg[rows], med[rows])
            current = counts_from_pred(predicted, np.where(known[rows], truth[rows], 0), known[rows])
            for field_name in ("tp", "fp", "fn", "tn", "obs", "pos", "neg"):
                getattr(counts, field_name)[:] += getattr(current, field_name)
        measured[name] = counts
    base_counts = measured["baseline_risk_only"]
    summary, per_disease = [], []
    for name in artifact["finalists"]:
        spec = _decode_spec(artifact["all_strategy_specs"][name])
        counts = measured[name]
        summary.append(
            summary_row(spec, counts, None if name == "baseline_risk_only" else base_counts, "test")
        )
        per_disease.extend(
            per_disease_frame(spec, endpoints, counts, base_counts, "test").to_dict(orient="records")
        )
    selector = StrategySpec(
        "per_disease_4way_selector",
        "selector",
        "selector",
        "mixed",
        enabled=np.asarray([row["use_consistency_cutoff"] for row in artifact["rules"]]),
        notes="Per-disease validation F1 selector over baseline, best KG, best medical, and best joint strategies.",
    )
    counts = Counts(*(np.zeros(len(endpoints), dtype=np.int64) for _ in range(7)))
    for index, row in enumerate(artifact["rules"]):
        selected_counts = measured[row["selected_strategy"]]
        for field_name in ("tp", "fp", "fn", "tn", "obs", "pos", "neg"):
            getattr(counts, field_name)[index] = getattr(selected_counts, field_name)[index]
    return _json_values(
        {
            "split": "test",
            "summary": summary,
            "per_disease": per_disease,
            "selector_summary": summary_row(selector, counts, base_counts, "test"),
            "selector_per_disease": per_disease_frame(
                selector, endpoints, counts, base_counts, "test"
            ).to_dict(orient="records"),
        }
    )


def write_survival_policies(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_survival_policies(artifact)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_values(artifact), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_survival_policies(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_survival_policies(artifact)
    return artifact
