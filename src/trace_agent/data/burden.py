from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurdenProfile:
    modules: tuple[str, ...]
    monetary: np.ndarray
    time: np.ndarray
    monetary_weight: float = 1.0
    time_weight: float = 0.0

    def __post_init__(self) -> None:
        modules = tuple(str(module) for module in self.modules)
        if len(set(modules)) != len(modules):
            raise ValueError("burden modules must be unique")
        monetary = np.asarray(self.monetary, dtype=np.float32).reshape(-1)
        time = np.asarray(self.time, dtype=np.float32).reshape(-1)
        expected = (len(modules),)
        if monetary.shape != expected or time.shape != expected:
            raise ValueError(
                f"burden arrays must both have shape {expected}; "
                f"got monetary={monetary.shape}, time={time.shape}"
            )
        if np.any(~np.isfinite(monetary)) or np.any(monetary < 0):
            raise ValueError("monetary burden must be finite and non-negative")
        if np.any(~np.isfinite(time)) or np.any(time < 0):
            raise ValueError("time burden must be finite and non-negative")
        weights = np.asarray([self.monetary_weight, self.time_weight], dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("burden weights must be finite and non-negative")
        if float(np.sum(weights)) <= 0:
            raise ValueError("at least one burden weight must be positive")
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "monetary", monetary)
        object.__setattr__(self, "time", time)

    @property
    def composite(self) -> np.ndarray:
        return (float(self.monetary_weight) * self.monetary + float(self.time_weight) * self.time).astype(
            np.float32, copy=False
        )

    def accumulated(self, used_modules: np.ndarray) -> float:
        used = np.asarray(used_modules, dtype=bool).reshape(-1)
        if used.shape != self.composite.shape:
            raise ValueError(f"used_modules shape must be {self.composite.shape}, got {used.shape}")
        return float(np.sum(self.composite[used]))
