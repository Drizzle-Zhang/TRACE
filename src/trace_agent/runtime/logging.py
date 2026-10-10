from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    resolved = str(level).upper()
    if resolved not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"unsupported log level: {level!r}")
    logging.basicConfig(
        level=getattr(logging, resolved),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
