"""Package explicitly listed local model files without changing their bytes.

Each archive includes a SHA-256 manifest. Completed archives are reused after
checking their recorded checksum and model count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any, BinaryIO


class _HashedReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.digest.update(chunk)
        return chunk


def checksum(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_group(group: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Write one archive and its completion record, or reuse a verified archive."""
    started = time.monotonic()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / (group["name"] + ".tar.gz")
    record_path = archive.with_suffix(archive.suffix + ".json")
    if archive.is_file() and record_path.is_file():
        old = json.loads(record_path.read_text())
        assert old["archive_sha256"] == checksum(archive)
        assert old["model_files"] == sum(item["kind"] == "model" for item in group["files"])
        return old
    assert not archive.exists(), "unowned existing archive: " + str(archive)
    temporary = archive.with_name(archive.name + ".part")
    records = []
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for item in group["files"]:
            path = Path(item["source"])
            before = path.stat()
            assert path.is_file() and (before.st_size > 0 or item["kind"] == "reference_code"), str(path)
            if "expected_size" in item:
                assert before.st_size == item["expected_size"]
            info = tar.gettarinfo(str(path), arcname=item["member"])
            if not info.isfile():
                info = tarfile.TarInfo(item["member"])
                info.size = before.st_size
                info.mode = 0o600
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                stream = _HashedReader(handle)
                tar.addfile(info, stream)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            records.append(
                {
                    "member": item["member"],
                    "source": str(path),
                    "kind": item["kind"],
                    "size_bytes": before.st_size,
                    "sha256": stream.digest.hexdigest(),
                    "source_mtime_ns": before.st_mtime_ns,
                }
            )
        manifest = json.dumps(
            {
                "schema": "trace.legacy_model_package.v1",
                "name": group["name"],
                "original_models_modified": False,
                "files": records,
            },
            indent=2,
        ).encode()
        info = tarfile.TarInfo("PACKAGE_MANIFEST.json")
        info.size = len(manifest)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(manifest))
    os.replace(temporary, archive)
    result = {
        "name": group["name"],
        "archive": str(archive),
        "archive_sha256": checksum(archive),
        "archive_bytes": archive.stat().st_size,
        "original_bytes": sum(record["size_bytes"] for record in records),
        "file_count": len(records),
        "model_files": sum(record["kind"] == "model" for record in records),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    temp_record = record_path.with_name(record_path.name + ".part")
    temp_record.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp_record, record_path)
    return result


def package_specification(specification: dict[str, Any]) -> list[dict[str, Any]]:
    """Package the groups in an explicit local specification using two workers."""
    root = Path(specification["output"])
    root.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(package_group, group, root) for group in specification["groups"]]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["name"])
    (root / "packages.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path, help="JSON file listing local sources and archive groups")
    arguments = parser.parse_args()
    results = package_specification(json.loads(arguments.specification.read_text()))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
