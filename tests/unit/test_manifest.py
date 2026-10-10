from __future__ import annotations

import hashlib
from pathlib import Path

from trace_agent.runtime.manifests import build_source_manifest


def test_build_source_manifest_hashes_files_and_excludes_runtime_secrets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.py").write_text("value = 1\n", encoding="utf-8")
    (source / "thirdparty_api").write_text("secret", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "keep.pyc").write_bytes(b"cache")

    manifest = build_source_manifest(source)

    assert manifest["source_root"] == str(source.resolve())
    assert manifest["file_count"] == 1
    assert manifest["files"] == [
        {
            "path": "keep.py",
            "size": 10,
            "sha256": hashlib.sha256(b"value = 1\n").hexdigest(),
        }
    ]
    assert manifest["excluded_security_paths"] == ["thirdparty_api"]
