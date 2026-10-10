from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_SUFFIXES = (
    "_sublevel_RS_model_calibration.joblib",
    "_sublevel_RS_model_calibrate.joblib",
    "_sublevel_RS_model.joblib",
)


@dataclass
class JoblibModelBank:
    root: Path | str
    suffixes: Sequence[str] = DEFAULT_MODEL_SUFFIXES
    _cache: dict[tuple[str, str], Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.suffixes = tuple(str(suffix) for suffix in self.suffixes)

    def resolve_path(self, module: str, endpoint: str) -> Path:
        module_dir = Path(self.root) / str(module)
        candidates = [module_dir / f"{endpoint}{suffix}" for suffix in self.suffixes]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"no model found for module={module!r}, endpoint={endpoint!r}; "
            f"checked {[str(path) for path in candidates]}"
        )

    def load(self, module: str, endpoint: str) -> Any:
        key = (str(module), str(endpoint))
        if key not in self._cache:
            import joblib

            self._cache[key] = joblib.load(self.resolve_path(*key))
        return self._cache[key]
