from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


def _require_unique(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{label} names must be non-empty")
    seen: set[str] = set()
    duplicate: list[str] = []
    for value in normalized:
        if value in seen and value not in duplicate:
            duplicate.append(value)
        seen.add(value)
    if duplicate:
        raise ValueError(f"duplicate {label} names: {duplicate}")
    return normalized


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    features: tuple[str, ...] = ()
    feature_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("module name must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "features", tuple(str(x) for x in self.features))
        object.__setattr__(self, "feature_codes", tuple(str(x) for x in self.feature_codes))
        if self.feature_codes and self.features and len(self.feature_codes) != len(self.features):
            raise ValueError(f"feature code/name length mismatch for module {name}")


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("endpoint name must be non-empty")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class AgentSchema:
    modules: tuple[ModuleSpec, ...]
    endpoints: tuple[EndpointSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modules = tuple(self.modules)
        endpoints = tuple(self.endpoints)
        _require_unique((module.name for module in modules), label="module")
        _require_unique((endpoint.name for endpoint in endpoints), label="endpoint")
        if not modules:
            raise ValueError("schema must define at least one module")
        if not endpoints:
            raise ValueError("schema must define at least one endpoint")
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "endpoints", endpoints)

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    @property
    def endpoint_names(self) -> tuple[str, ...]:
        return tuple(endpoint.name for endpoint in self.endpoints)

    def module_index(self, name: str) -> int:
        try:
            return self.module_names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"unknown module: {name}") from exc

    def endpoint_index(self, name: str) -> int:
        try:
            return self.endpoint_names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"unknown endpoint: {name}") from exc
