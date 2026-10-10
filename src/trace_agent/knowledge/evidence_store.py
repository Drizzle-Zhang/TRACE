from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np


def _normalize(value: object) -> str:
    text = str(value or "").strip().lower().replace("β", "beta")
    return "_".join(token for token in re.split(r"[^a-z0-9]+", text) if token)


def _float(value: object, default: float) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _direction_sign(row: dict[str, str]) -> float:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("direction", "consensus_stance", "action", "final_strength", "strength")
    ).lower()
    negative = (
        "against",
        "avoid",
        "negative",
        "protective",
        "do not",
        "not recommend",
    )
    return -1.0 if any(token in text for token in negative) else 1.0


def _match_reliability(row: dict[str, str], *, factor: str, match_type: str, notes: str) -> float:
    """Apply the original rule/mapping discounts independently of certainty rank."""
    reliability = 1.0
    if _truthy(row.get("certainty_imputed")):
        reliability *= 0.85
    if _truthy(row.get("needs_manual_review")):
        reliability *= 0.5
    kind = match_type.lower()
    if kind in {"token", "synonym"}:
        reliability *= 0.75
    if kind in {"llm_curated", "manual_seed"} and "proxy" in notes.lower():
        reliability *= 0.7
    if factor in {
        "family_history",
        "laboratory_evaluation",
        "risk_assessment",
        "severity_assessment",
        "structured_symptom_assessment",
        "exposure_history",
        "comorbidity_burden",
    }:
        reliability *= 0.45
    return max(0.05, min(1.0, reliability))


@dataclass(frozen=True)
class CompiledEvidenceRules:
    """One entry per rule-feature match, retaining conflicting rules separately.

    Feature index -1 denotes a mapped feature outside the contribution axis. Its
    rule still determines the endpoint's weight scale, as in the source scorer.
    """

    endpoints: tuple[str, ...]
    features: tuple[str, ...]
    endpoint_indices: np.ndarray
    feature_indices: np.ndarray
    weights: np.ndarray
    reliability: np.ndarray


@dataclass(frozen=True)
class CompiledEvidenceWeights:
    endpoints: tuple[str, ...]
    features: tuple[str, ...]
    weights: np.ndarray
    reliability: np.ndarray
    matched_rule_counts: np.ndarray
    rules: CompiledEvidenceRules


@dataclass(frozen=True)
class _SourceMatch:
    row: dict[str, str]
    endpoint_index: int
    factor_token: str
    feature_code: str
    feature_name: str
    match_type: str
    weight: float
    reliability: float


class ThresholdSummary(TypedDict):
    n_rows: int
    n_samples: int
    status_counts: dict[str, int]


@dataclass(frozen=True)
class _ThresholdSpec:
    op: str
    value: float
    upper: float | None = None

    def accepts(self, value: float) -> bool:
        if self.op == "between":
            return self.upper is not None and self.value <= value <= self.upper
        return {
            ">=": value >= self.value,
            ">": value > self.value,
            "<=": value <= self.value,
            "<": value < self.value,
            "=": value == self.value,
        }.get(self.op, False)


def _threshold_spec(text: str, *, feature_code: str, feature_name: str) -> _ThresholdSpec | None:
    value = text.strip().lower()
    if not value:
        return None
    if "/" in value and "mmhg" in value:
        pairs = [
            (float(a), float(b))
            for a, b in re.findall(r"(\d{2,3}(?:\.\d+)?)\s*/\s*(\d{2,3}(?:\.\d+)?)", value)
        ]
        feature = f"{feature_code} {feature_name}".lower()
        values = None
        if "diastolic" in feature or "p4079" in feature or re.search(r"\bdbp\b", feature):
            values = [diastolic for _, diastolic in pairs]
        elif "systolic" in feature or "p4080" in feature or re.search(r"\bsbp\b", feature):
            values = [systolic for systolic, _ in pairs]
        if values:
            if any(
                token in value
                for token in ("or higher", "or greater", "above", "over", ">=", "high blood pressure")
            ):
                return _ThresholdSpec(">=", min(values))
            if any(token in value for token in ("or lower", "or less", "below", "under", "<=")):
                return _ThresholdSpec("<=", max(values))
            return _ThresholdSpec(">=", min(values))
    found = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", value)
    if found:
        bounds = [float(found.group(1)), float(found.group(2))]
        return _ThresholdSpec("between", min(bounds), max(bounds))
    for pattern, op in (
        (r"(?:at least|no less than|aged?|age)\s*(\d+(?:\.\d+)?)", ">="),
        (r"(?:greater than|more than|above|over)\s*(\d+(?:\.\d+)?)", ">"),
        (r"(?:less than|below|under)\s*(\d+(?:\.\d+)?)", "<"),
        (r"(?:up to|no more than)\s*(\d+(?:\.\d+)?)", "<="),
    ):
        found = re.search(pattern, value)
        if found:
            return _ThresholdSpec(op, float(found.group(1)))
    found = re.search(r"(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", value)
    if found:
        return _ThresholdSpec(found.group(1), float(found.group(2)))
    found = re.search(r"(\d+(?:\.\d+)?)\s*(?:or greater|and above|and over)", value)
    return _ThresholdSpec(">=", float(found.group(1))) if found else None


class MedicalEvidenceStore:
    """Compile the versioned CSV evidence bundle into model-aligned arrays."""

    def __init__(self, root: str | Path, *, disease_metadata_dir: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.disease_metadata_dir = (
            None if disease_metadata_dir is None else Path(disease_metadata_dir).expanduser().resolve()
        )

    @staticmethod
    def _rows(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                {str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)
            ]

    def _metadata_root(self) -> Path | None:
        if self.disease_metadata_dir is not None:
            if not self.disease_metadata_dir.is_dir():
                raise NotADirectoryError(self.disease_metadata_dir)
            return self.disease_metadata_dir
        for candidate in (
            self.root / "disease_metadata",
            self.root,
            self.root.parent,
            self.root.parent.parent,
        ):
            if (candidate / "outcome_filter.csv").is_file() or (candidate / "disease_map_full.csv").is_file():
                return candidate
        return None

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """All existing tables used by rule compilation, for provenance hashing."""
        paths = [
            self.root / name
            for name in (
                "risk_factor_evidence.csv",
                "decision_screening_evidence.csv",
                "consensus_evidence.csv",
                "disease_evidence_feature_map.csv",
                "compressed_disease_resolution.csv",
            )
        ]
        metadata = self._metadata_root()
        if metadata is not None:
            paths.extend(
                metadata / name for name in ("outcome_filter.csv", "outcome_test.csv", "disease_map_full.csv")
            )
        return tuple(dict.fromkeys(path for path in paths if path.is_file()))

    def _endpoint_lookup(self, endpoints: tuple[str, ...]) -> dict[str, int]:
        # Schema identities are inserted first; metadata collisions cannot replace them.
        lookup: dict[str, int] = {}
        for index, name in enumerate(endpoints):
            lookup.setdefault(_normalize(name), index)
        metadata = self._metadata_root()
        if metadata is None:
            return lookup
        endpoint_indices = {name: index for index, name in enumerate(endpoints)}
        code_to_long: dict[str, str] = {}
        long_to_code: dict[str, str] = {}
        for filename in ("outcome_filter.csv", "outcome_test.csv"):
            path = metadata / filename
            for row in self._rows(path) if path.is_file() else []:
                code, name = row.get("NAME", "").strip(), row.get("LONGNAME", "").strip()
                if code not in endpoint_indices or not name:
                    continue
                code_to_long.setdefault(code, name)
                long_to_code.setdefault(_normalize(name), code)
                lookup.setdefault(_normalize(name), endpoint_indices[code])
        simplified: dict[str, str] = {}
        path = metadata / "disease_map_full.csv"
        for row in self._rows(path) if path.is_file() else []:
            # Unicode escapes retain legacy external header aliases; shipped metadata uses English.
            original = row.get("\u539f\u59cb LONGNAME") or row.get("original_longname") or ""
            alias = (
                row.get("\u7b80\u5e76\u540e\u75be\u75c5\u540d") or row.get("compressed_disease_name") or ""
            )
            if original.strip() and alias.strip():
                simplified[_normalize(original)] = alias.strip()
        for code, name in code_to_long.items():
            simplified_alias = simplified.get(_normalize(name))
            if simplified_alias:
                lookup.setdefault(_normalize(simplified_alias), endpoint_indices[code])
        path = self.root / "compressed_disease_resolution.csv"
        for row in self._rows(path) if path.is_file() else []:
            alias = row.get("compressed_disease_name", "").strip()
            targets = {
                long_to_code[_normalize(name)]
                for key in ("original_longnames", "matched_original_longnames")
                for name in row.get(key, "").split("|")
                if name.strip() and _normalize(name) in long_to_code
            }
            if alias and len(targets) == 1:
                lookup.setdefault(_normalize(alias), endpoint_indices[next(iter(targets))])
        return lookup

    @staticmethod
    def _auto_matches(factor: str, names: Mapping[str, str]) -> list[tuple[str, str, float, str]]:
        normalized_to_codes: dict[str, list[str]] = {}
        for code, name in names.items():
            for key in {_normalize(code), _normalize(name)}:
                if key:
                    normalized_to_codes.setdefault(key, []).append(code)
        aliases = {
            "bmi": {"body_mass_index", "body_mass_index_bmi", "bmi"},
            "body_mass_index": {"body_mass_index", "bmi"},
            "obesity": {"bmi", "body_mass_index", "waist_circumference"},
            "overweight_or_obesity": {"bmi", "body_mass_index", "waist_circumference"},
            "hba1c": {"hba1c", "glycated_haemoglobin", "glycated_hemoglobin"},
            "fasting_plasma_glucose": {"fasting_glucose", "glucose", "fasting_plasma_glucose"},
            "clinic_blood_pressure": {
                "blood_pressure",
                "systolic_blood_pressure",
                "diastolic_blood_pressure",
            },
            "blood_pressure": {"blood_pressure", "systolic_blood_pressure", "diastolic_blood_pressure"},
            "hypertension": {"blood_pressure", "systolic_blood_pressure", "diastolic_blood_pressure"},
            "smoking": {"smoking", "smoking_status", "tobacco"},
            "smoking_or_vaping_exposure": {"smoking", "smoking_status", "tobacco"},
            "age": {"age", "age_at_recruitment"},
            "estimated_glomerular_filtration_rate": {"egfr", "estimated_glomerular_filtration_rate"},
            "urine_albumin": {"albuminuria", "urine_albumin", "albumin_creatinine_ratio"},
            "nt_probnp": {"nt_probnp", "nt_pro_bnp"},
        }
        matches = [(code, "exact", 1.0, "") for code in normalized_to_codes.get(factor, [])]
        for alias in sorted(aliases.get(factor, ())):
            matches.extend((code, "synonym", 0.9, "") for code in normalized_to_codes.get(alias, []))
        if not matches:
            tokens = {token for token in factor.split("_") if len(token) >= 3}
            if tokens:
                matches = [
                    (code, "token", 0.6, "")
                    for code, name in names.items()
                    if tokens.issubset(set(_normalize(name).split("_")))
                ][:5]
        unique: dict[str, tuple[str, str, float, str]] = {}
        for match in matches:
            unique.setdefault(match[0], match)
        return list(unique.values())

    def _matches(
        self,
        *,
        endpoints: tuple[str, ...],
        features: tuple[str, ...],
        feature_code_to_name: Mapping[str, str] | None = None,
        min_certainty_rank: float = 0.0,
        min_strength_rank: float = 0.0,
    ) -> list[_SourceMatch]:
        endpoint_lookup = self._endpoint_lookup(endpoints)
        names = {str(code): str(name) for code, name in (feature_code_to_name or {}).items()}
        for feature in features:
            names.setdefault(feature, feature)
        mappings: dict[str, list[tuple[str, str, float, str]]] = {}
        mapping_path = self.root / "disease_evidence_feature_map.csv"
        for row in self._rows(mapping_path) if mapping_path.is_file() else []:
            factor, code = _normalize(row.get("factor", "")), row.get("feature_code", "").strip()
            # An explicit factor mapping suppresses automatic matches, as in the source.
            if factor:
                mappings.setdefault(factor, []).append(
                    (
                        code,
                        row.get("match_type") or "manual",
                        _float(row.get("weight"), 1.0),
                        row.get("notes", ""),
                    )
                )
        result: list[_SourceMatch] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for filename in (
            "risk_factor_evidence.csv",
            "decision_screening_evidence.csv",
            "consensus_evidence.csv",
        ):
            path = self.root / filename
            for row in self._rows(path) if path.is_file() else []:
                aliases = [
                    _normalize(alias)
                    for key in ("model_labels", "disease", "disease_name", "name")
                    for alias in row.get(key, "").split("|")
                    if alias.strip()
                ]
                endpoint = next(
                    (endpoint_lookup[alias] for alias in aliases if alias in endpoint_lookup), None
                )
                if endpoint is None:
                    continue
                certainty, strength = (
                    _float(row.get("certainty_rank"), 0.0),
                    _float(row.get("strength_rank"), 0.0),
                )
                if filename == "consensus_evidence.csv":
                    certainty = max(
                        certainty,
                        {"high": 4.0, "moderate": 3.0, "low": 2.0, "very low": 1.0}.get(
                            row.get("max_certainty", "").strip().lower(), 0.0
                        ),
                    )
                    strength_text = row.get("final_strength", "").lower()
                    strength = max(
                        strength,
                        (
                            3.0
                            if "strong" in strength_text
                            else 1.0 if "conditional" in strength_text or "weak" in strength_text else 0.0
                        ),
                    )
                if certainty < min_certainty_rank or strength < min_strength_rank:
                    continue
                bucket = row.get("evidence_bucket", "").strip()
                if not bucket and filename == "risk_factor_evidence.csv":
                    # Retain support for the existing compact risk-only bundle schema.
                    bucket = "risk_factor"
                row = {**row, "evidence_bucket": bucket}
                rule_weight = (
                    _direction_sign(row)
                    * max(1.0, certainty)
                    * max(1.0, strength)
                    / max(1.0, _float(row.get("source_priority"), 1.0))
                )
                if _truthy(row.get("needs_manual_review")):
                    rule_weight *= 0.5
                statement = (
                    row.get("statement_id") or row.get("consensus_id") or row.get("candidate_id") or ""
                )
                for token in (
                    value.strip() for value in re.split(r"[|;,]", row.get("factor", "")) if value.strip()
                ):
                    factor = _normalize(token)
                    matches = mappings[factor] if factor in mappings else self._auto_matches(factor, names)
                    for code, kind, weight, notes in matches:
                        key = (statement, token, code, kind, bucket)
                        if not code or key in seen:
                            continue
                        seen.add(key)
                        result.append(
                            _SourceMatch(
                                row,
                                endpoint,
                                token,
                                code,
                                names.get(code, code),
                                kind,
                                rule_weight * weight,
                                _match_reliability(row, factor=factor, match_type=kind, notes=notes),
                            )
                        )
        return result

    def compile(
        self,
        *,
        endpoints: tuple[str, ...],
        features: tuple[str, ...],
        feature_code_to_name: Mapping[str, str] | None = None,
        min_certainty_rank: float = 0.0,
        min_strength_rank: float = 0.0,
    ) -> CompiledEvidenceWeights:
        endpoint_names = tuple(str(value) for value in endpoints)
        feature_names = tuple(str(value) for value in features)
        feature_lookup = {str(value): index for index, value in enumerate(feature_names)}
        shape = (len(endpoint_names), len(feature_names))
        weights = np.zeros(shape, dtype=np.float32)
        reliability_sum = np.zeros(shape, dtype=np.float32)
        reliability_count = np.zeros(shape, dtype=np.int32)
        matched_rule_counts = np.zeros(len(endpoint_names), dtype=np.int32)
        rule_endpoints: list[int] = []
        rule_features: list[int] = []
        rule_weights: list[float] = []
        rule_reliabilities: list[float] = []
        matched_statements: set[tuple[int, str, str]] = set()
        for match in self._matches(
            endpoints=endpoint_names,
            features=feature_names,
            feature_code_to_name=feature_code_to_name,
            min_certainty_rank=min_certainty_rank,
            min_strength_rank=min_strength_rank,
        ):
            if match.row["evidence_bucket"] != "risk_factor":
                continue
            endpoint_index = match.endpoint_index
            feature_index = feature_lookup.get(match.feature_code, feature_lookup.get(match.feature_name, -1))
            rule_endpoints.append(endpoint_index)
            rule_features.append(feature_index)
            rule_weights.append(match.weight)
            rule_reliabilities.append(match.reliability)
            if feature_index < 0:
                continue
            weights[endpoint_index, feature_index] += match.weight
            reliability_sum[endpoint_index, feature_index] += match.reliability
            reliability_count[endpoint_index, feature_index] += 1
            statement = (
                match.row.get("statement_id")
                or match.row.get("consensus_id")
                or match.row.get("candidate_id")
                or ""
            )
            matched_statements.add((endpoint_index, statement, match.row.get("factor", "")))
        for endpoint_index, _, _ in matched_statements:
            matched_rule_counts[endpoint_index] += 1
        reliability_matrix = np.divide(
            reliability_sum,
            reliability_count,
            out=np.zeros(shape, dtype=np.float32),
            where=reliability_count > 0,
        )
        return CompiledEvidenceWeights(
            endpoints=endpoint_names,
            features=feature_names,
            weights=weights,
            reliability=reliability_matrix,
            matched_rule_counts=matched_rule_counts,
            rules=CompiledEvidenceRules(
                endpoints=endpoint_names,
                features=feature_names,
                endpoint_indices=np.asarray(rule_endpoints, dtype=np.int64),
                feature_indices=np.asarray(rule_features, dtype=np.int64),
                weights=np.asarray(rule_weights, dtype=np.float64),
                reliability=np.asarray(rule_reliabilities, dtype=np.float64),
            ),
        )

    def check_thresholds(
        self,
        *,
        endpoints: tuple[str, ...],
        features: tuple[str, ...],
        participant_ids: Sequence[object] | np.ndarray,
        feature_values: np.ndarray | None,
        feature_code_to_name: Mapping[str, str] | None = None,
        min_certainty_rank: float = 0.0,
        min_strength_rank: float = 0.0,
    ) -> tuple[list[dict[str, Any]], ThresholdSummary]:
        """Batch diagnostic checks on raw values; never alter risk/filter scores.

        Return the original threshold CSV fields and summary keys. Missing raw
        values remain explicitly ``missing_feature`` rather than neutral passes.
        """
        ids = np.asarray(participant_ids)
        if ids.ndim != 1:
            raise ValueError("threshold participant_ids must be one-dimensional")
        values = None if feature_values is None else np.asarray(feature_values)
        if values is not None and values.shape != (ids.size, len(features)):
            raise ValueError("threshold feature_values must align with participant and feature axes")
        feature_lookup = {str(name): index for index, name in enumerate(features)}
        prepared = []
        for match in self._matches(
            endpoints=endpoints,
            features=features,
            feature_code_to_name=feature_code_to_name,
            min_certainty_rank=min_certainty_rank,
            min_strength_rank=min_strength_rank,
        ):
            if match.row["evidence_bucket"] != "decision_screening":
                continue
            threshold = match.row.get("threshold", "").strip()
            spec = _threshold_spec(
                threshold, feature_code=match.feature_code, feature_name=match.feature_name
            )
            feature_index = feature_lookup.get(match.feature_code, feature_lookup.get(match.feature_name, -1))
            prepared.append((match, threshold, spec, feature_index))
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for index, participant_id in enumerate(ids):
            for match, threshold, spec, feature_index in prepared:
                value = None
                if not threshold:
                    status = "not_applicable"
                elif spec is None:
                    status = "unparsed_threshold"
                else:
                    if values is not None and feature_index >= 0:
                        candidate = _float(values[index, feature_index], float("nan"))
                        if np.isfinite(candidate):
                            value = candidate
                    status = "missing_feature" if value is None else "pass" if spec.accepts(value) else "fail"
                counts[status] = counts.get(status, 0) + 1
                rows.append(
                    {
                        "sample_id": str(participant_id),
                        "sample_index": index,
                        "disease": endpoints[match.endpoint_index],
                        "feature_code": match.feature_code,
                        "feature_name": match.feature_name,
                        "factor": match.row.get("factor", "").strip(),
                        "factor_token": match.factor_token,
                        "threshold": threshold,
                        "threshold_status": status,
                        "observed_value": value,
                        "source": (match.row.get("source") or match.row.get("source_family") or "").strip(),
                        "snippet": (
                            match.row.get("snippet") or match.row.get("recommendation_text") or ""
                        ).strip(),
                    }
                )
        return rows, {"n_rows": len(rows), "status_counts": counts, "n_samples": int(ids.size)}
