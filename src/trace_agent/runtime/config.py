from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PATH_KEYS = {
    "absolute_risk",
    "baseline_cumhaz",
    "burden_profile",
    "calibrators",
    "command_cwd",
    "contributions",
    "directory",
    "evidence_bundle",
    "evidence_reliability",
    "evidence_weights",
    "endpoints_file",
    "feature_weights",
    "feature_names",
    "fusion_model",
    "initial_mask_path",
    "input_dir",
    "knowledge_matrix",
    "label_mask",
    "labels",
    "output_dir",
    "participant_ids",
    "path",
    "predictions",
    "q_model",
    "results",
    "time_grid",
}


@dataclass(frozen=True)
class _DeclaredPath:
    value: str
    directory: Path


def _mark_path_origins(value: Any, *, key: str, directory: Path) -> Any:
    if isinstance(value, dict):
        return {str(k): _mark_path_origins(v, key=str(k), directory=directory) for k, v in value.items()}
    if isinstance(value, list):
        return [_mark_path_origins(item, key=key, directory=directory) for item in value]
    if isinstance(value, str) and value and _looks_like_path(key):
        return _DeclaredPath(value, directory)
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_extends(path: Path, *, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"configuration inheritance cycle: {chain}")
    payload = _read_yaml(resolved)
    extends = payload.pop("extends", None)
    payload = _mark_path_origins(payload, key="", directory=resolved.parent)
    if extends is None:
        return payload
    parents = [extends] if isinstance(extends, str) else extends
    if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
        raise ValueError(f"extends must be a path or path list: {resolved}")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = _deep_merge(
            merged,
            _read_with_extends(parent_path, stack=(*stack, resolved)),
        )
    return _deep_merge(merged, payload)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, _DeclaredPath):
        return _DeclaredPath(_expand_environment(value.value), value.directory)
    if isinstance(value, str):
        missing = sorted(
            {match.group(1) for match in _ENV_PATTERN.finditer(value) if match.group(1) not in os.environ}
        )
        if missing:
            raise ValueError("unresolved environment variable(s): " + ", ".join(missing))
        return _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def _looks_like_path(key: str) -> bool:
    normalized = str(key).lower()
    return normalized in _PATH_KEYS or normalized.endswith(("_path", "_dir", "_file"))


def _resolve_paths(value: Any, *, key: str, root: Path) -> Any:
    if isinstance(value, _DeclaredPath):
        return _resolve_paths(value.value, key=key, root=value.directory)
    if isinstance(value, dict):
        return {
            str(child_key): _resolve_paths(
                child_value,
                key=str(child_key),
                root=root,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_paths(item, key=key, root=root) for item in value]
    if isinstance(value, str) and value and _looks_like_path(key) and "://" not in value:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else root / path).resolve())
    return value


def _fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(
    config_path: str | Path,
    *,
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load, merge and resolve one TRACE pipeline configuration."""

    path = Path(config_path).expanduser().resolve()
    config = _read_with_extends(path)
    profile = None
    if profile_path is not None:
        profile = Path(profile_path).expanduser().resolve()
        config = _deep_merge(config, _read_with_extends(profile))
    config = _expand_environment(config)
    config = _resolve_paths(config, key="", root=path.parent)
    version = config.get("version", 1)
    if int(version) != 1:
        raise ValueError(f"unsupported configuration version {version!r}; expected 1")
    config["version"] = 1
    config["_meta"] = {
        "config_path": str(path),
        "profile_path": None if profile is None else str(profile),
    }
    config["_meta"]["sha256"] = _fingerprint({key: value for key, value in config.items() if key != "_meta"})
    return config
