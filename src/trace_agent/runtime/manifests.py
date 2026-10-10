from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "analysis_outputs",
    "artifacts",
    "artifacts_q_grid",
    "outputs",
}
_EXCLUDED_NAMES = {
    ".DS_Store",
    ".env",
    "thirdparty_api",
}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _is_excluded(path: Path, source_root: Path) -> tuple[bool, bool]:
    relative = path.relative_to(source_root)
    if any(part in _EXCLUDED_DIRS for part in relative.parts[:-1]):
        return True, False
    if path.name in _EXCLUDED_NAMES or path.name.startswith(".env."):
        return True, path.name == "thirdparty_api" or path.name.startswith(".env")
    if path.suffix in _EXCLUDED_SUFFIXES:
        return True, False
    return False, False


def build_source_manifest(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"source root does not exist or is not a directory: {root}")

    files: list[dict[str, Any]] = []
    excluded_security: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        excluded, security = _is_excluded(path, root)
        relative = path.relative_to(root).as_posix()
        if excluded:
            if security:
                excluded_security.append(relative)
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")

    return {
        "source_root": str(root),
        "file_count": len(files),
        "content_manifest_sha256": digest.hexdigest(),
        "files": files,
        "excluded_security_paths": sorted(excluded_security),
    }


def write_source_manifest(source_root: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest(source_root)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a sanitized source-project manifest.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output = write_source_manifest(args.source_root, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
