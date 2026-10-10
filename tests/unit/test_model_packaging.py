import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


def _packer():
    path = Path(__file__).resolve().parents[2] / "tools/package_model_files.py"
    spec = importlib.util.spec_from_file_location("legacy_packer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, files):
    value = {"output": str(tmp_path / "packages"), "groups": [{"name": "example", "files": files}]}
    _packer().package_specification(value)
    return tmp_path / "packages/example.tar.gz"


def test_package_preserves_empty_python_initializers_and_model_hashes(tmp_path):
    initializer = tmp_path / "__init__.py"
    initializer.write_bytes(b"")
    model = tmp_path / "old.joblib"
    model.write_bytes(b"unchanged original model bytes")
    files = [
        {
            "source": str(initializer),
            "member": "reference_code/ukbpp/__init__.py",
            "kind": "reference_code",
        },
        {"source": str(model), "member": "module_models/old.joblib", "kind": "model"},
    ]
    archive = _run(tmp_path, files)
    with tarfile.open(archive) as tar:
        assert tar.extractfile("reference_code/ukbpp/__init__.py").read() == b""
        assert tar.extractfile("module_models/old.joblib").read() == model.read_bytes()
        manifest = json.load(tar.extractfile("PACKAGE_MANIFEST.json"))
        assert manifest["schema"] == "trace.legacy_model_package.v1"
        assert manifest["name"] == "example"
        assert manifest["original_models_modified"] is False
        records = {r["kind"]: r for r in manifest["files"]}
        assert records["reference_code"]["sha256"] == hashlib.sha256(b"").hexdigest()
        assert records["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
        assert records["model"]["size_bytes"] == model.stat().st_size

    original_archive = archive.read_bytes()
    summary = json.loads(archive.with_suffix(".gz.json").read_text())
    assert summary["archive_sha256"] == hashlib.sha256(original_archive).hexdigest()
    assert summary["file_count"] == 2
    assert summary["model_files"] == 1
    assert _run(tmp_path, files).read_bytes() == original_archive
    assert json.loads((tmp_path / "packages/packages.json").read_text()) == [summary]
    with pytest.raises(AssertionError):
        _run(tmp_path, [])


def test_empty_model_never_becomes_a_completed_package(tmp_path):
    model = tmp_path / "empty.joblib"
    model.write_bytes(b"")
    with pytest.raises(AssertionError):
        _run(tmp_path, [{"source": str(model), "member": "empty.joblib", "kind": "model"}])
    assert not (tmp_path / "packages/example.tar.gz").exists()
    assert not (tmp_path / "packages/example.tar.gz.json").exists()
