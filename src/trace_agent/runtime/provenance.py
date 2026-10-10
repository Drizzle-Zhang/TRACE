from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def array_sha256(values: np.ndarray) -> str:
    """Fingerprint values together with their exact dtype, shape and order."""
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    if array.dtype.kind in {"O", "U"}:
        digest.update(
            json.dumps(
                [str(value) for value in array.reshape(-1).tolist()],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def row_array_sha256(values: np.ndarray, rows: np.ndarray) -> str:
    """Hash selected rows like array_sha256, without copying the whole selection."""
    array = np.asarray(values)
    indices = np.asarray(rows, dtype=np.int64).reshape(-1)
    if array.ndim == 0 or np.any(indices < 0) or np.any(indices >= array.shape[0]):
        raise ValueError("fingerprint rows must index the first array axis")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps([indices.size, *array.shape[1:]], separators=(",", ":")).encode())
    row_bytes = max(1, int(np.prod(array.shape[1:])) * array.dtype.itemsize)
    chunk_size = max(1, (8 * 1024 * 1024) // row_bytes)
    strings = array.dtype.kind in {"O", "U"}
    first = True
    if strings:
        digest.update(b"[")
    for start in range(0, indices.size, chunk_size):
        chunk = array[indices[start : start + chunk_size]]
        if strings:
            for value in chunk.reshape(-1):
                if not first:
                    digest.update(b",")
                digest.update(json.dumps(str(value), ensure_ascii=False).encode("utf-8"))
                first = False
        else:
            digest.update(np.ascontiguousarray(chunk).tobytes())
    if strings:
        digest.update(b"]")
    return digest.hexdigest()


def file_provenance(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }
