from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    PREVALENT = "prevalent"
    LONGITUDINAL = "longitudinal"

    @classmethod
    def parse(cls, value: str | TaskType) -> TaskType:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "diagnosis": cls.PREVALENT,
            "incident": cls.LONGITUDINAL,
            "survival": cls.LONGITUDINAL,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


class SplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"

    @classmethod
    def parse(cls, value: str | SplitRole) -> SplitRole:
        return value if isinstance(value, cls) else cls(str(value).strip().lower())

    @property
    def allows_threshold_selection(self) -> bool:
        return self is SplitRole.VALIDATION
